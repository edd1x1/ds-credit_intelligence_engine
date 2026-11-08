"""Out-of-time, calibrated early warning for active loans.
Only internal point-in-time balances, instalments and transaction aggregates.
Forecasts NEW DPD30+ in 30/60/90 days, excluding already-DPD30+ loans.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss,log_loss
from .ml_core import PROJECT_ROOT

PATH=PROJECT_ROOT/'data/processed/early_warning_snapshots.csv'
ASOF=pd.Timestamp('2026-10-31')
FEATURES=['CurrentDPD','LoanAgeMonths','TermMonthsRemaining','DueUnpaidKGS',
          'PaymentsDue6M','MeanPaymentDue6M','AgreementSumN','Rate',
          'LoanTermMonths','MonthlyPaymentKGS','IncomeMean6M_KGS',
          'CashflowMargin6M','AppActivityDays','InternalMonthlyDebtKGS','InternalDTI']
FINANCIAL=['IncomeMean6M_KGS','CashflowMargin6M','AppActivityDays',
           'InternalMonthlyDebtKGS','InternalDTI']

@lru_cache(maxsize=1)
def read_snapshots():
    return pd.read_csv(PATH,parse_dates=['SnapshotDate'])

def pipeline():
    return Pipeline([('fill',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('scale',StandardScaler()),('lr',LogisticRegression(max_iter=400,random_state=42))])

def metrics(y,p):
    y=np.asarray(y,int);p=np.clip(np.asarray(p,float),1e-7,1-1e-7)
    return {'n':int(len(y)),'events':int(y.sum()),'event_rate':float(y.mean()),
        'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),
        'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p))}

def fit_calibrated(train,cal,target,features):
    m=pipeline();m.fit(train[features],train[target].astype(int))
    calibrated=CalibratedClassifierCV(FrozenEstimator(m),method='sigmoid')
    calibrated.fit(cal[features],cal[target].astype(int))
    return calibrated

def evaluate_ews(h=30):
    df=read_snapshots();target=f'TargetNewDPD30_{h}D'
    base=df[df.CurrentDPD.lt(30)&df[target].isin([0,1])].copy()
    # The label for the latest training row must have been observable before
    # the first row in calibration; then test is a later OOT calendar cohort.
    tr=base[(base.SnapshotDate>='2023-01-01')&(base.SnapshotDate<'2025-01-01')]
    cal=base[(base.SnapshotDate>='2025-04-01')&(base.SnapshotDate<'2025-07-01')]
    test=base[(base.SnapshotDate>='2025-07-01')&(base.SnapshotDate<'2026-01-01')]
    # 90d labels on Dec 2024 mature by Mar 2025, prior to Apr calibration.
    # Calibration Apr-Jun 2025 labels mature by Sep 2025; OOT evaluation
    # is retrospective, so calibrated model uses future labels relative to
    # test's July 2025 start. This is INVALID for an online July backtest.
    # Use a later test cohort to ensure all calibration labels are mature:
    test=base[(base.SnapshotDate>='2025-10-01')&(base.SnapshotDate<'2026-04-01')]
    model=fit_calibrated(tr,cal,target,FEATURES)
    pred=model.predict_proba(test[FEATURES])[:,1]
    res=metrics(test[target],pred)
    score=pd.DataFrame({'p':pred,'y':test[target].astype(int).to_numpy()})
    score['decile']=pd.qcut(score.p.rank(method='first'),10,labels=False)
    reliability=score.groupby('decile',as_index=False).agg(
        n=('y','size'),predicted=('p','mean'),observed=('y','mean'))
    # Compare with an intentionally minimal historical-feature baseline.
    small=[x for x in FEATURES if x not in FINANCIAL]
    abl=fit_calibrated(tr,cal,target,small)
    ablation=metrics(test[target],abl.predict_proba(test[small])[:,1])
    # Fit deployment model on more recent FULLY MATURE training snapshots;
    # calibration occurs later. Never use current 2026-10 snapshot labels.
    prodtr=base[(base.SnapshotDate<'2026-01-01')]
    prodcal=base[(base.SnapshotDate>='2026-04-01')&(base.SnapshotDate<'2026-07-01')]
    deployed=fit_calibrated(prodtr,prodcal,target,FEATURES)
    return deployed,res,reliability,ablation

@lru_cache(maxsize=1)
def all_ews():
    current=read_snapshots().query('SnapshotDate == @ASOF').copy()
    current['EligibleNewEvent']=current.CurrentDPD.lt(30)
    evaluation=[]; reliability=[]; ablations=[]
    for h in (30,60,90):
        model,m,r,a=evaluate_ews(h)
        current[f'EW_PD30_{h}D']=np.where(current.EligibleNewEvent,
               model.predict_proba(current[FEATURES])[:,1],np.nan)
        evaluation.append({'horizon_days':h,**m})
        ablations.append({'horizon_days':h,'full_pr_auc':m['pr_auc'],
                'no_financial_pr_auc':a['pr_auc'],'full_brier':m['brier'],
                'no_financial_brier':a['brier']})
        r['horizon_days']=h; reliability.append(r)
    return current,pd.DataFrame(evaluation),pd.concat(reliability),pd.DataFrame(ablations)

def psi(expected,actual,bins=10):
    expected=np.asarray(expected,float); actual=np.asarray(actual,float)
    expected=expected[np.isfinite(expected)];actual=actual[np.isfinite(actual)]
    if min(len(expected),len(actual))<10:return float('nan')
    edges=np.unique(np.quantile(expected,np.linspace(0,1,bins+1)))
    if len(edges)<3:return 0.
    edges[0]=-np.inf;edges[-1]=np.inf
    p=np.histogram(expected,bins=edges)[0]/len(expected)
    q=np.histogram(actual,bins=edges)[0]/len(actual)
    p=np.clip(p,1e-5,None);q=np.clip(q,1e-5,None)
    return float(np.sum((q-p)*np.log(q/p)))

def monitoring_reports():
    s=read_snapshots();s=s[s.CurrentDPD.notna()].copy()
    s['Bucket']=pd.cut(s.CurrentDPD,[-1,0,29,59,89,np.inf],labels=['0','1-29','30-59','60-89','90+'])
    s=s.sort_values(['CreditID','SnapshotDate'])
    s['NextBucket']=s.groupby('CreditID').Bucket.shift(-1)
    s['NextDate']=s.groupby('CreditID').SnapshotDate.shift(-1)
    pairs=s[(s.NextDate-s.SnapshotDate).dt.days.between(28,31)].copy()
    rolls=pairs.groupby(['Bucket','NextBucket'],observed=True).size().rename('N').reset_index()
    rolls['RollRate']=rolls.N/rolls.groupby('Bucket',observed=True).N.transform('sum')
    loans=pd.read_csv(PROJECT_ROOT/'data/raw/loans.csv',usecols=['CreditID','CreditOpenDate','ProductFamily'])
    mart=pd.read_csv(PROJECT_ROOT/'data/processed/model_dataset.csv',usecols=['CreditID','TargetDPD30_12M','IsMatureTarget'])
    loans['Vintage']=pd.to_datetime(loans.CreditOpenDate).dt.to_period('Q').astype(str)
    vint=loans.merge(mart,on='CreditID',how='inner')
    vint=vint[vint.IsMatureTarget.eq(1)&vint.TargetDPD30_12M.isin([0,1])]
    vint=vint.groupby(['Vintage','ProductFamily'],as_index=False).agg(
        N=('CreditID','size'),BadRate12M=('TargetDPD30_12M','mean'))
    cur=s[s.SnapshotDate.eq(ASOF)]
    train=s[s.SnapshotDate.between('2023-01-31','2024-12-31')]
    drift=pd.DataFrame([{'Feature':feature,'PSI':psi(train[feature],cur[feature])} for feature in FEATURES])
    return rolls,vint,drift

if __name__=='__main__':
    x,e,r,a=all_ews()
    print('CURRENT',len(x));print(e.to_string(index=False));print(a.to_string(index=False))
