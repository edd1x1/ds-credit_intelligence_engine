"""Multi-offer internal-data-only Decision Engine and EWS portfolio reporting.
No paid bureau queries. Only preliminary candidates, never credit approvals.
PR means interest income; expected profit applies contact cost ONCE per contact.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from . import decision_engine as old
from .early_warning import all_ews,monitoring_reports
from .ml_core import read_model,time_splits,evaluate,TARGET,PROJECT_ROOT

@dataclass(frozen=True)
class DecisionSettings(old.DecisionSettings):
    loyalty_discount_pp: float=0.0

ASOF=old.AS_OF
RECOMMENDABLE_CODES=old.RECOMMENDABLE_CODES
annuity_payment=old.annuity_payment
principal_from_payment=old.principal_from_payment
expected_amortization=old.expected_amortization
pd_to_horizon=old.pd_to_horizon
DECISION_FEATURES=old.DECISION_FEATURES
PRODUCT_LGD=old.PRODUCT_LGD
REPORTS=PROJECT_ROOT/'reports'

@lru_cache(maxsize=1)
def risk_model_and_validation():
    df=read_model();s=time_splits(df)
    model=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('scale',StandardScaler()),('lr',LogisticRegression(max_iter=700,C=.8,random_state=42))])
    model.fit(s['train'][DECISION_FEATURES],s['train'][TARGET].astype(int))
    cal=CalibratedClassifierCV(FrozenEstimator(model),method='sigmoid')
    cal.fit(s['validation'][DECISION_FEATURES],s['validation'][TARGET].astype(int))
    p=cal.predict_proba(s['test'][DECISION_FEATURES])[:,1]
    metrics=evaluate(s['test'][TARGET].astype(int),p)
    metrics={k:float(v) for k,v in metrics.items() if k!='cm'}
    dfcal=pd.DataFrame({'score':p,'target':s['test'][TARGET].astype(int).to_numpy()})
    dfcal['decile']=pd.qcut(dfcal.score.rank(method='first'),10,labels=False)
    reliability=dfcal.groupby('decile',as_index=False).agg(
        N=('target','size'),PredictedPD=('score','mean'),ActualBadRate=('target','mean'))
    return cal,metrics,reliability

def internal_dti(existing_payment,new_payment,mean_income):
    """Internal bank-only DTI; unknown external debt is NOT assumed absent."""
    income=np.asarray(mean_income,dtype=float)
    return (np.asarray(existing_payment,float)+np.asarray(new_payment,float))/np.maximum(income,1)

@lru_cache(maxsize=1)
def all_proposals():
    base=old.base_candidates().copy()
    loans=pd.read_csv(old.LOANS_PATH,parse_dates=['CreditOpenDate','CreditEndDate'])
    status=pd.read_csv(PROJECT_ROOT/'data/processed/status_reconciliation.csv')
    loans=loans.merge(status[['CreditID','CorrectedStatusAtCutoff']],on='CreditID',how='left')
    loans=loans[loans.CorrectedStatusAtCutoff.eq('Active')]
    dept=loans.groupby('CustomerID').MonthlyPaymentKGS.sum()
    base['InternalExistingDebtKGS']=base.CustomerID.map(dept).fillna(0).astype(float)
    # Four proposals, then score/rank separately; no inferred purchase intent
    # for auto, mortgages or secured cash.
    offers=[('CASH_ONLINE_200','Онлайн-кредит 200',24.,24,2.5,200000.),
            ('CASH_ONLINE_200','Онлайн-кредит 200',22.,12,1.5,200000.),
            ('CASH_STANDARD_500','Наличные Стандарт 500',21.,36,4.0,500000.),
            ('INSTALL_3M','Рассрочка 3 месяца',0.,3,1.0,500000.)]
    chunks=[]
    for code,label,rate,term,mult,limit in offers:
        a=base.copy()
        a['ProductCode']=code;a['RecommendedProduct']=label
        a['ProductFamily']='Installment' if code=='INSTALL_3M' else 'Cash'
        a['OfferRatePct']=rate
        a['OfferTermMonths']=term
        a['RequestedAmountKGS']=np.round(np.minimum(limit,a.IncomeMean6M_KGS*mult)/1000)*1000
        a['ProductMinKGS']=10000 if code=='CASH_ONLINE_200' else 15000
        a['ProductMaxKGS']=limit
        chunks.append(a)
    return pd.concat(chunks,ignore_index=True)

@lru_cache(maxsize=16)
def decision_table(settings:DecisionSettings):
    c=all_proposals().copy()
    c['BaseOfferRatePct']=c.OfferRatePct.copy()
    reward=(c.NoCreditHistory.eq(0)&c.MaxHistoricalDPD.eq(0)&
            c.IncomeCV6M.le(.15)&c.ProductFamily.eq('Cash'))
    c['OfferRatePct']=np.where(reward,
        np.maximum(18.,c.OfferRatePct-settings.loyalty_discount_pp),c.OfferRatePct)
    c['LoyaltyDiscountPctPts']=c.BaseOfferRatePct-c.OfferRatePct
    remaining=(c.IncomeMean6M_KGS*settings.max_dti-c.InternalExistingDebtKGS).clip(lower=0)
    affordable=principal_from_payment(remaining,c.OfferRatePct,c.OfferTermMonths)
    c['OfferAmountKGS']=np.floor(np.minimum(c.RequestedAmountKGS,affordable)/1000)*1000
    c['MonthlyPaymentKGS']=annuity_payment(c.OfferAmountKGS,c.OfferRatePct,c.OfferTermMonths)
    c['ProposalDTI']=internal_dti(c.InternalExistingDebtKGS,c.MonthlyPaymentKGS,c.IncomeMean6M_KGS)
    # Training-scoring DTI of a proposed new credit; another-banks' payments unknown.
    x=old.risk_features_for_proposals(c,c.ProductCode,c.OfferAmountKGS,c.OfferRatePct,c.OfferTermMonths)
    x['DebtToIncomeRatio']=c.ProposalDTI
    model,metrics,reliability=risk_model_and_validation()
    c['PD12']=model.predict_proba(x[DECISION_FEATURES])[:,1]
    allowed=(c.AgeAtSnapshot.between(21,65)&c.IncomeMean6M_KGS.ge(settings.min_income)&
             c.IncomeCV6M.le(settings.max_income_cv)&
             c.MaxHistoricalDPD.le(settings.max_hist_dpd)&
             c.HistoricalDPD30Count.le(1 if settings.max_hist_dpd>=30 else 0)&
             c.ActiveLoansAtSnapshot.eq(0)&c.OfferAmountKGS.ge(c.ProductMinKGS)&
             c.OfferAmountKGS.le(c.ProductMaxKGS)&
             c.ProposalDTI.le(settings.max_dti+1e-8)&
             c.PD12.le(settings.max_pd)&c.ProductCode.isin(RECOMMENDABLE_CODES))
    if not settings.include_no_history:allowed &= c.NoCreditHistory.eq(0)
    c=c.loc[allowed].copy()
    if c.empty:return c
    c['Segment']=np.where(c.NoCreditHistory.eq(1),'Новая кредитная история',
                     np.where(c.MaxHistoricalDPD.eq(0),'Чистая кредитная история','Незначительная просрочка'))
    interest,avg_balance=expected_amortization(c.OfferAmountKGS.values,c.OfferRatePct.values,
                                               c.OfferTermMonths.values,365)
    c['ProjectedPR12M_KGS']=interest
    horizon=np.minimum(365.,c.OfferTermMonths.to_numpy(float)*30.4375)
    lgd=np.minimum(.95,c.ProductFamily.map(PRODUCT_LGD).fillna(.55).to_numpy()*settings.lgd/.55)
    c['ExpectedLoss12M_KGS']=pd_to_horizon(c.PD12,horizon)*lgd*c.OfferAmountKGS
    c['FundingCost12M_KGS']=avg_balance*(settings.funding_rate/100)*(horizon/365)
    c['MerchantIncomeKGS']=np.where(c.ProductFamily.eq('Installment'),
                                     c.OfferAmountKGS*settings.merchant_discount_pct/100,0.)
    # Operational servicing assumption is an explicit scenario; cost occurs only
    # when the offer converts. Contact cost occurs for every contacted candidate.
    c['ServicingCostIfAcceptedKGS']=500.0
    c['ContributionIfAcceptedKGS']=(c.ProjectedPR12M_KGS+c.MerchantIncomeKGS-
                     c.ExpectedLoss12M_KGS-c.FundingCost12M_KGS-c.ServicingCostIfAcceptedKGS)
    c['ScenarioAcceptanceProbability']=float(settings.conversion)
    c['ExpectedValuePerContactKGS']=c.ScenarioAcceptanceProbability*c.ContributionIfAcceptedKGS-settings.contact_cost
    c['NetContribution12M_KGS']=c.ContributionIfAcceptedKGS # compatibility for v5 UI
    c=c[c.ExpectedValuePerContactKGS.gt(0)].copy()
    if c.empty:return c
    c['RecommendationReason']=np.select([
           c.NoCreditHistory.eq(1),c.ProductFamily.eq('Installment'),
           c.MaxHistoricalDPD.eq(0)],
           ['Нагрузка проходит, но нет подтвержденной кредитной истории',
            'Финансовая доступность; потребность в покупке не подтверждена',
            'Кредитная дисциплина без просрочек'],
           default='Платёжная дисциплина и внутренний ПДН в границах отбора')
    c['DecisionStatus']='PRE_OFFER_REVIEW'
    c['ExternalDebtStatus']='UNKNOWN_NOT_QUERIED'
    c['SelfBanStatus']='NOT_CHECKED_PREOFFER'
    c['MarketingConsentStatus']='UNKNOWN_REVIEW_REQUIRED'
    c['ModelVersion']='risk-calibration-v6'
    c['DecisionDate']=ASOF.date().isoformat()
    c=c.sort_values(['ExpectedValuePerContactKGS','PD12'],ascending=[False,True])
    # Best offer per customer, selected from all eligible alternatives.
    c=c.drop_duplicates('CustomerID',keep='first')
    if settings.diversify_products:
        # Optional product caps affect selection, NOT risk, pricing or eligibility.
        caps={'CASH_STANDARD_500':.50,'CASH_ONLINE_200':.35,'INSTALL_3M':.15}
        selected=[]
        for code,cap in caps.items():
            selected.append(c[c.ProductCode.eq(code)].head(int(np.ceil(settings.top_n*cap))))
        c=pd.concat(selected).sort_values('ExpectedValuePerContactKGS',ascending=False)
    c=c.head(settings.top_n).reset_index(drop=True)
    return c

@lru_cache(maxsize=1)
def current_portfolio_v6():
    base=old.active_portfolio().copy()
    ews,validation,reliability,ablation=all_ews()
    assert ews.CreditID.is_unique
    active=base.merge(ews[['CreditID','CurrentDPD','InternalDTI','EW_PD30_30D','EW_PD30_60D','EW_PD30_90D']],
          on='CreditID',how='left',suffixes=('','_EWS'),validate='one_to_one')
    # Older original dashboard's overdue calculation is retained for balances,
    # but point-in-time event DPD comes from EWS mart.
    active['CurrentDPD']=active['CurrentDPD_EWS'].fillna(active.CurrentDPD)
    active.drop(columns=['CurrentDPD_EWS'],inplace=True)
    return active

def scenario_dashboard(settings:DecisionSettings):
    active=current_portfolio_v6();offers=decision_table(settings)
    eligible=active[active.CurrentDPD.lt(30)]
    interest90=[]
    rows=[]
    for days in (30,60,90):
        p=eligible[f'EW_PD30_{days}D'].fillna(0).to_numpy(float)
        lgd=np.minimum(.95,eligible.ProductFamily.map(PRODUCT_LGD).fillna(.55).to_numpy(float)*settings.lgd/.55)
        n=float(np.sum(p))
        loss=float(np.sum(p*eligible.EstimatedOutstandingKGS.to_numpy(float)*lgd))
        baseline=float(np.sum(active.EstimatedOutstandingKGS*active.Rate*days/36500))
        new_interest,_=expected_amortization(offers.OfferAmountKGS.values,offers.OfferRatePct.values,
                                            offers.OfferTermMonths.values,days) if len(offers) else (np.array([]),np.array([]))
        incr=float(np.sum(new_interest))*settings.conversion
        rows.append(dict(HorizonDays=days,ExpectedNewDPD30=n,ExpectedLossKGS=loss,
                         ExistingPRKGS=baseline,IncrementalPRKGS=incr,
                         ScenarioPRKGS=baseline+incr,GrowthPRPct=(100*incr/baseline if baseline else 0)))
    by=active.groupby('ProductFamily').agg(ActiveLoans=('CreditID','size'),
          EstimatedBalanceKGS=('EstimatedOutstandingKGS','sum'),
          CurrentDPD30=('CurrentDPD',lambda x:int((x>=30).sum())),
          MeanPD12=('PD12MonitoringScenario','mean')).reset_index()
    val=all_ews()[1]
    k={'as_of':str(ASOF.date()),'active_loans':len(active),
       'estimated_balance':float(active.EstimatedOutstandingKGS.sum()),
       'current_dpd30':int(active.CurrentDPD.ge(30).sum()),
       'currently_overdue':int(active.CurrentDPD.gt(0).sum()),
       'candidates':len(offers),'candidate_total':len(old.base_candidates()),
       'volume_offers':float(offers.OfferAmountKGS.sum()) if len(offers) else 0.,
       'expected_accepted':len(offers)*settings.conversion,
       'test_auc':float(val.loc[val.horizon_days.eq(30),'roc_auc'].iloc[0]),
       'test_pr_auc':float(val.loc[val.horizon_days.eq(30),'pr_auc'].iloc[0])}
    return k,pd.DataFrame(rows),by,active

def export_decisions(settings:DecisionSettings):
    c=decision_table(settings)
    REPORTS.mkdir(parents=True,exist_ok=True)
    path=REPORTS/'recommended_customers_v6.csv'
    columns=['CustomerID','RecommendedProduct','ProductCode','OfferAmountKGS',
       'OfferRatePct','LoyaltyDiscountPctPts','OfferTermMonths','PD12','InternalExistingDebtKGS',
       'MonthlyPaymentKGS','ProposalDTI','ScenarioAcceptanceProbability',
       'ContributionIfAcceptedKGS','ExpectedValuePerContactKGS','ProjectedPR12M_KGS',
       'ExpectedLoss12M_KGS','RecommendationReason','DecisionStatus','ModelVersion',
       'ExternalDebtStatus','SelfBanStatus','MarketingConsentStatus']
    c.reindex(columns=columns).to_csv(path,index=False,encoding='utf-8-sig')
    return str(path)

def save_reports():
    a,e,r,abl=all_ews();roll,vint,psi=monitoring_reports()
    REPORTS.mkdir(exist_ok=True)
    for name,frame in [('ews_oot_validation',e),('ews_calibration',r),
        ('ews_ablation',abl),('ews_roll_rates',roll),('ews_vintage',vint),
        ('ews_psi',psi)]:frame.to_csv(REPORTS/f'{name}.csv',index=False,encoding='utf-8-sig')
    risk,rm,rel=risk_model_and_validation()
    rel.to_csv(REPORTS/'credit_risk_calibration.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([rm]).to_csv(REPORTS/'credit_risk_oot_validation.csv',index=False)
    k,h,f,act=scenario_dashboard(DecisionSettings())
    h.to_csv(REPORTS/'risk_forecast_v6.csv',index=False)
    export_decisions(DecisionSettings())
    act[['CustomerID','CreditID','ProductFamily','CurrentDPD',
         'EstimatedOutstandingKGS','EW_PD30_30D','EW_PD30_60D','EW_PD30_90D']].to_csv(
        REPORTS/'portfolio_risk_watchlist_v6.csv',index=False,encoding='utf-8-sig')
    return k,h,act,e

if __name__=='__main__':
    k,h,a,e=save_reports()
    print('Dashboard',k)
    print('EWS',e.to_string(index=False))
    print('Leads',len(decision_table(DecisionSettings())))

def explain_customer_offers(customer_id:str,settings:DecisionSettings):
    """All offer alternatives for a given internal CustomerID (not approvals)."""
    allp=all_proposals()
    c=allp[allp.CustomerID.astype(str).eq(str(customer_id).strip())].copy()
    if c.empty:return pd.DataFrame()
    reward=c.NoCreditHistory.eq(0)&c.MaxHistoricalDPD.eq(0)&c.IncomeCV6M.le(.15)&c.ProductFamily.eq('Cash')
    c['OfferRatePct']=np.where(reward,np.maximum(18.,c.OfferRatePct-settings.loyalty_discount_pp),c.OfferRatePct)
    remaining=(c.IncomeMean6M_KGS*settings.max_dti-c.InternalExistingDebtKGS).clip(lower=0)
    c['OfferAmountKGS']=np.floor(np.minimum(c.RequestedAmountKGS,
         principal_from_payment(remaining,c.OfferRatePct,c.OfferTermMonths))/1000)*1000
    c['MonthlyPaymentKGS']=annuity_payment(c.OfferAmountKGS,c.OfferRatePct,c.OfferTermMonths)
    c['InternalDTI']=internal_dti(c.InternalExistingDebtKGS,c.MonthlyPaymentKGS,c.IncomeMean6M_KGS)
    feat=old.risk_features_for_proposals(c,c.ProductCode,c.OfferAmountKGS,c.OfferRatePct,c.OfferTermMonths)
    feat['DebtToIncomeRatio']=c.InternalDTI
    model,_,_=risk_model_and_validation()
    c['PD12']=model.predict_proba(feat[DECISION_FEATURES])[:,1]
    income,average=expected_amortization(c.OfferAmountKGS.to_numpy(),
                                    c.OfferRatePct.to_numpy(),c.OfferTermMonths.to_numpy(),365)
    horizon=np.minimum(365,c.OfferTermMonths.to_numpy(float)*30.4375)
    lgd=np.minimum(.95,c.ProductFamily.map(PRODUCT_LGD).fillna(.55).to_numpy()*settings.lgd/.55)
    loss=pd_to_horizon(c.PD12,horizon)*lgd*c.OfferAmountKGS
    funding=average*(settings.funding_rate/100)*(horizon/365)
    merchant=np.where(c.ProductFamily.eq('Installment'),
                       c.OfferAmountKGS*settings.merchant_discount_pct/100,0.)
    c['ExpectedValuePerContactKGS']=settings.conversion*(income+merchant-loss-funding-500)-settings.contact_cost
    c['OfferRulesPassed']=(c.AgeAtSnapshot.between(21,65)&c.IncomeMean6M_KGS.ge(settings.min_income)&
        c.IncomeCV6M.le(settings.max_income_cv)&c.MaxHistoricalDPD.le(settings.max_hist_dpd)&
        c.PD12.le(settings.max_pd)&c.InternalDTI.le(settings.max_dti)&
        c.OfferAmountKGS.ge(c.ProductMinKGS)&c.ExpectedValuePerContactKGS.gt(0))
    if not settings.include_no_history:c['OfferRulesPassed'] &= c.NoCreditHistory.eq(0)
    c['ContactPermission']='UNKNOWN'
    return c[['CustomerID','RecommendedProduct','OfferAmountKGS','OfferTermMonths',
        'OfferRatePct','PD12','InternalExistingDebtKGS','InternalDTI',
        'ExpectedValuePerContactKGS','OfferRulesPassed','ContactPermission']].reset_index(drop=True)
