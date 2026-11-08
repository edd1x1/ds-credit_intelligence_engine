"""Credit Intelligence v5 — exploratory risk & campaign scenario engine.

All monetary forecasts are EXPLICIT scenario estimates, not validated forecasts
or credit approval decisions. Dataset provenance is retained in raw columns.
No region or city enters model training, campaign eligibility or pricing.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .ml_core import PROJECT_ROOT, read_model, TARGET, time_splits, evaluate

ROOT = PROJECT_ROOT
AS_OF = pd.Timestamp('2026-10-31')
# Shared features at loan origination and when creating a comparable new offer.
# City/region, CustomerID, future repayment and known labels deliberately omitted.
DECISION_FEATURES = [
    'PreviousLoansCount', 'MaxHistoricalDPD', 'NoCreditHistory',
    'IncomeMean6M_KGS', 'IncomeCV6M', 'CashflowMargin6M',
    'AppActiveDays6M', 'AgreementSumN', 'Rate', 'LoanTermMonths',
    'MonthlyPaymentKGS', 'DebtToIncomeRatio',
]
CANDIDATE_PATH = ROOT / 'data/processed/marketing_candidates.csv'
LOANS_PATH = ROOT / 'data/raw/loans.csv'
SCHEDULE_PATH = ROOT / 'data/raw/repayment_schedule.csv'
CATALOG_PATH = ROOT / 'data/reference/product_catalog.csv'
REPORTS = ROOT / 'reports'
# Product fit requires observed purchase/pledge intent. For now, only these
# product families can be recommended in a non-binding outbound shortlist.
RECOMMENDABLE_CODES = {'CASH_ONLINE_200', 'CASH_STANDARD_500', 'INSTALL_3M'}
PRODUCT_LGD = {'Cash': 0.65, 'Installment': 0.55, 'Auto': 0.40,
               'Mortgage': 0.25, 'SecuredCash': 0.35}  # illustrative assumptions


def annuity_payment(principal, annual_rate, months):
    """Vectorized contractual monthly annuity; 0% product handled exactly."""
    p = np.asarray(principal, dtype=float)
    r = np.asarray(annual_rate, dtype=float)/1200
    n = np.asarray(months, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        out = np.where(np.abs(r) < 1e-10, p/n, p * r/(1-(1+r)**(-n)))
    return out


def principal_from_payment(payment, annual_rate, months):
    payment=np.asarray(payment,dtype=float)
    r=np.asarray(annual_rate,dtype=float)/1200
    n=np.asarray(months,dtype=float)
    with np.errstate(divide='ignore',invalid='ignore'):
        factor=np.where(np.abs(r)<1e-10,n,(1-(1+r)**(-n))/r)
    return payment*factor


def expected_amortization(principal, rate, months, horizon_days):
    """Expected annuity interest and average exposure until horizon.

    Monthly steps approximate (horizon_days/30.4375); remaining days prorated.
    No delinquency, early repayment, fees or day-count conventions modeled.
    """
    p=np.asarray(principal,dtype=float)
    rate=np.asarray(rate,dtype=float)
    months=np.asarray(months,dtype=float)
    horizon=float(horizon_days)/30.4375
    nfull=int(np.floor(horizon)); partial=horizon-nfull
    pay=annuity_payment(p,rate,months)
    balance=p.copy()
    interest=np.zeros_like(p)
    exposure_time=np.zeros_like(p)
    for i in range(nfull+1):
        fraction=partial if i==nfull else 1.0
        if fraction<=0:continue
        active=(months>i)&(balance>0.01)
        chunk=np.where(active,balance,0.)
        exposure_time += chunk*fraction
        interest+=chunk*(rate/1200)*fraction
        full_principal=np.maximum(0,pay-chunk*rate/1200)
        balance=np.where(active,np.maximum(0.,balance-full_principal*fraction),balance)
    avg_exposure=exposure_time/max(1e-9,horizon)
    return interest,avg_exposure


def pd_to_horizon(pd12, days):
    """Constant-hazard SCENARIO: not a trained 30/60/90-day model."""
    p=np.clip(np.asarray(pd12,dtype=float),0,0.999)
    return 1-np.power(1-p,np.asarray(days,dtype=float)/365.)


@lru_cache(maxsize=1)
def model_and_validation():
    raw=read_model()
    known=raw[(raw.IsMatureTarget==1)&(raw[TARGET].isin([0,1]))&
              (raw.LabelReadyDate <= AS_OF)].copy()
    X=known[DECISION_FEATURES]
    y=known[TARGET].astype(int)
    pipeline=Pipeline([
        ('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('scaler',StandardScaler()),
        ('model',LogisticRegression(max_iter=1000,C=0.8,random_state=42)),
    ])
    # An independently trained historical validation baseline, not a claim
    # that this is an accurate live risk model.
    subsets=time_splits(raw)
    history=pd.concat([subsets['train'],subsets['validation']])
    holdout=subsets['test']
    test_model=Pipeline([
        ('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('scaler',StandardScaler()),
        ('model',LogisticRegression(max_iter=1000,C=0.8,random_state=42))
    ])
    test_model.fit(history[DECISION_FEATURES],history[TARGET].astype(int))
    p=test_model.predict_proba(holdout[DECISION_FEATURES])[:,1]
    metrics=evaluate(holdout[TARGET].astype(int).values,p)
    pipeline.fit(X,y)
    return pipeline,{k:float(v) for k,v in metrics.items() if k not in ('cm',)},len(known)


@lru_cache(maxsize=1)
def load_sources():
    c=pd.read_csv(CANDIDATE_PATH,parse_dates=['SnapshotDate'])
    l=pd.read_csv(LOANS_PATH,parse_dates=['CreditOpenDate','CreditEndDate'])
    r=pd.read_csv(SCHEDULE_PATH,parse_dates=['DueDate','SettledDate'])
    catalog=pd.read_csv(CATALOG_PATH)
    return c,l,r,catalog


@lru_cache(maxsize=1)
def base_candidates():
    c,l,r,catalog=load_sources()
    assert c.CustomerID.is_unique
    assert c.ActiveLoansAtSnapshot.eq(0).all()
    c=c.copy()
    observed=l[l.CreditOpenDate<=AS_OF]
    latest=(observed.sort_values('CreditOpenDate')
            .drop_duplicates('CustomerID',keep='last')
            [['CustomerID','ProductFamily']]
            .rename(columns={'ProductFamily':'LastProductFamily'}))
    c=c.merge(latest,on='CustomerID',how='left',validate='one_to_one')
    return c


def risk_features_for_proposals(base,code,amount,rate,term):
    x=base.copy()
    x['AgreementSumN']=np.asarray(amount,dtype=float)
    x['Rate']=np.asarray(rate,dtype=float)
    x['LoanTermMonths']=np.asarray(term,dtype=float)
    x['MonthlyPaymentKGS']=annuity_payment(x['AgreementSumN'],x['Rate'],x['LoanTermMonths'])
    x['DebtToIncomeRatio']=x['MonthlyPaymentKGS']/x['IncomeMean6M_KGS'].clip(lower=1)
    return x


@lru_cache(maxsize=1)
def proposal_pool():
    """One preliminary cash/installment offer per inactive client.

    Not a propensity model, approval, consent or product-intent classifier.
    """
    c=base_candidates().copy()
    _,_,_,catalog=load_sources()
    bycode=catalog.set_index('ProductCode')
    segment=np.where(c['LastProductFamily'].eq('Installment') & (c['PreviousLoansCount']>0),
                     'INSTALL_3M',
             np.where(c['IncomeMean6M_KGS']>=75000,'CASH_STANDARD_500','CASH_ONLINE_200'))
    c['ProductCode']=segment
    c['RecommendedProduct']=c['ProductCode'].map(bycode['ScenarioProductName'])
    c['ProductFamily']=c['ProductCode'].map(bycode['ProductFamily'])
    c['OfferRatePct']=np.select(
        [c.ProductCode.eq('INSTALL_3M'),c.ProductCode.eq('CASH_STANDARD_500')],
        [0.0,20.0],default=24.0)
    c['OfferTermMonths']=np.select(
        [c.ProductCode.eq('INSTALL_3M'),c.ProductCode.eq('CASH_STANDARD_500')],
        [3,36],default=24).astype(int)
    c['ProductMinKGS']=c['ProductCode'].map(bycode['ScenarioMinKGS']).astype(float)
    c['ProductMaxKGS']=c['ProductCode'].map(bycode['ScenarioMaxKGS']).astype(float)
    factor=np.where(c.ProductCode.eq('CASH_STANDARD_500'),3.5,
                    np.where(c.ProductCode.eq('INSTALL_3M'),1.25,2.0))
    c['RequestedAmountKGS']=np.minimum(c.ProductMaxKGS,
        np.maximum(c.ProductMinKGS,c.IncomeMean6M_KGS*factor))
    c['RequestedAmountKGS']=(c['RequestedAmountKGS']/5000).round().clip(lower=1)*5000
    c['RequestedAmountKGS']=np.minimum(c.RequestedAmountKGS,c.ProductMaxKGS)
    return c


@dataclass(frozen=True)
class DecisionSettings:
    max_pd: float=0.18
    max_dti: float=0.40
    min_income: float=30000
    max_hist_dpd: int=29
    include_no_history: bool=False
    max_income_cv: float=0.40
    top_n: int=300
    funding_rate: float=10.0
    lgd: float=0.55
    contact_cost: float=20.0
    merchant_discount_pct: float=3.0
    conversion: float=0.08
    diversify_products: bool=True


@lru_cache(maxsize=24)
def decision_table(settings:DecisionSettings):
    c=proposal_pool().copy()
    max_payment=np.maximum(0.,c.IncomeMean6M_KGS*settings.max_dti)
    affordable=principal_from_payment(max_payment,c.OfferRatePct,c.OfferTermMonths)
    amount=np.minimum(c.RequestedAmountKGS,affordable)
    amount=np.floor(amount/1000)*1000
    amount=np.minimum(amount,c.ProductMaxKGS)
    c['OfferAmountKGS']=np.maximum(0,amount)
    model,metrics,n=model_and_validation()
    feat=risk_features_for_proposals(c,c.ProductCode,c.OfferAmountKGS,
                                     c.OfferRatePct,c.OfferTermMonths)
    c['MonthlyPaymentKGS']=feat['MonthlyPaymentKGS']
    c['ProposalDTI']=feat['DebtToIncomeRatio']
    c['PD12']=model.predict_proba(feat[DECISION_FEATURES])[:,1]
    mask=(c.AgeAtSnapshot.between(21,65) & c.IncomeMean6M_KGS.ge(settings.min_income)&
          c.IncomeCV6M.le(settings.max_income_cv)&
          c.MaxHistoricalDPD.le(settings.max_hist_dpd)&
          c.HistoricalDPD30Count.le(1 if settings.max_hist_dpd>=30 else 0)&
          c.ActiveLoansAtSnapshot.eq(0)&
          c.OfferAmountKGS.ge(c.ProductMinKGS)&
          c.ProposalDTI.le(settings.max_dti+1e-8)&
          c.PD12.le(settings.max_pd)&
          c.ProductCode.isin(RECOMMENDABLE_CODES))
    if not settings.include_no_history:
        mask &= c.NoCreditHistory.eq(0)
    c=c.loc[mask].copy()
    if len(c)==0:
        return c
    c['Segment']=np.where(c.NoCreditHistory.eq(1),'Нет кредитной истории',
                   np.where(c.MaxHistoricalDPD.eq(0),'Без просрочек',
                            'Просрочки до порога'))
    interest,avg_balance=expected_amortization(c.OfferAmountKGS.values,
                            c.OfferRatePct.values,c.OfferTermMonths.values,365)
    c['ProjectedPR12M_KGS']=interest
    observed_horizon=np.minimum(365.,c.OfferTermMonths.to_numpy(float)*30.4375)
    # For short installments, loss is relevant over the 3-month life, not 12m.
    scenario_lgd=np.minimum(0.95,c.ProductFamily.map(PRODUCT_LGD).fillna(0.55).to_numpy()*settings.lgd/0.55)
    c['ExpectedLoss12M_KGS']=pd_to_horizon(c.PD12,observed_horizon)*scenario_lgd*c.OfferAmountKGS
    c['FundingCost12M_KGS']=avg_balance*(settings.funding_rate/100)*(observed_horizon/365.)
    c['MerchantIncomeKGS']=np.where(c.ProductFamily.eq('Installment'),
                   c.OfferAmountKGS*settings.merchant_discount_pct/100,0.)
    c['NetContribution12M_KGS']=c.ProjectedPR12M_KGS+c.MerchantIncomeKGS - \
        c.ExpectedLoss12M_KGS-c.FundingCost12M_KGS-settings.contact_cost
    c['RecommendationReason']=np.select(
        [c.NoCreditHistory.eq(1),c.LastProductFamily.eq('Installment'),
         c.MaxHistoricalDPD.eq(0)],
        ['Доход и нагрузка подходят; историю нужно проверить',
         'Повторная продуктовая категория и достаточный доход',
         'Нет прошлых просрочек и допустимая нагрузка'],
        default='Нет DPD30+ в истории, параметры в пределах правил')
    c=c[c.NetContribution12M_KGS>0].copy()
    c=c.sort_values(['NetContribution12M_KGS','PD12'],ascending=[False,True])
    if settings.diversify_products:
        # Deliberate campaign-mix cap; prevents largest-ticket product from
        # filling the entire list. Never forces unprofitable recommendations.
        caps={'CASH_STANDARD_500':.50,'CASH_ONLINE_200':.35,'INSTALL_3M':.15}
        picked=[]
        for code,weight in caps.items():
            k=int(np.ceil(settings.top_n*weight))
            picked.append(c[c.ProductCode.eq(code)].head(k))
        c=pd.concat(picked).sort_values(['NetContribution12M_KGS','PD12'],ascending=[False,True]).head(int(settings.top_n))
    else:
        c=c.head(int(settings.top_n))
    c['CustomerID']=c.CustomerID.astype(str)
    return c.reset_index(drop=True)


@lru_cache(maxsize=1)
def active_portfolio():
    c,l,r,cat=load_sources()
    active=l[l.CreditStatusAtCutoff.eq('Active')].copy()
    if len(active)==0:return active
    active['CreditOpenDate']=pd.to_datetime(active.CreditOpenDate)
    # Amortized principal proxy, not an ABS principal balance.
    elapsed=((AS_OF-active.CreditOpenDate).dt.days/30.4375).clip(lower=0)
    initial=active.AgreementSumN.to_numpy(float)
    pay=active.MonthlyPaymentKGS.to_numpy(float)
    rr=active.Rate.to_numpy(float)/1200.
    ee=np.minimum(elapsed.to_numpy(float),active.LoanTermMonths.to_numpy(float))
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        growth=np.power(1+rr,ee)
        principal=np.where(rr>1e-9,initial*growth-pay*(growth-1)/np.where(rr>1e-9,rr,1.),
                           initial-pay*ee)
    # Active, overdue past-maturity loans require a balance snapshot from the ABS.
    # Floor is a declared approximation rather than pretending the principal is zero.
    past_maturity=active.CreditEndDate.lt(AS_OF)
    principal=np.where(past_maturity & (principal<=0), initial*.07,principal)
    active['EstimatedOutstandingKGS']=np.clip(principal,0,initial)
    unpaid=r[r.RepaymentStateAtCutoff.eq('OPEN') & r.DueDate.le(AS_OF)]
    overdue=unpaid.groupby('CreditID').agg(
        CurrentDPD=('DaysPastDueAtCutoff','max'),
        DueUnpaidKGS=('ScheduledPaymentKGS','sum'))
    active=active.merge(overdue,on='CreditID',how='left',validate='one_to_one')
    active[['CurrentDPD','DueUnpaidKGS']]=active[['CurrentDPD','DueUnpaidKGS']].fillna(0)
    indexed=read_model().set_index('CreditID')
    X=indexed.reindex(active.CreditID)[DECISION_FEATURES].copy()
    if X.isna().all(axis=1).any():
        raise ValueError('Активные кредиты отсутствуют в исходной матрице признаков')
    model,_,_=model_and_validation()
    active['PD12AtIssueProxy']=model.predict_proba(X)[:,1]
    # Current overdue increases monitoring risk, but is not fed back into
    # application-scoring model, which was trained at issue time.
    active['PD12MonitoringScenario']=np.clip(active.PD12AtIssueProxy.to_numpy()*
        np.select([active.CurrentDPD.ge(30),active.CurrentDPD.gt(0)],
                  [1.8,1.20],default=1.),0,0.95)
    return active


def scenario_dashboard(settings:DecisionSettings):
    active=active_portfolio()
    offers=decision_table(settings)
    eligible=active[active.CurrentDPD<30]
    dp=eligible.PD12MonitoringScenario.to_numpy(float)
    principal=eligible.EstimatedOutstandingKGS.to_numpy(float)
    rows=[]
    for days in [30,60,90]:
        p=pd_to_horizon(dp,days)
        events=float(np.sum(p))
        product_lgd=np.minimum(.95,eligible.ProductFamily.map(PRODUCT_LGD).fillna(.55).to_numpy(float)*settings.lgd/.55)
        loss=float(np.sum(p*principal*product_lgd))
        # PR = Процентный доход / portfolio interest revenue in this project.
        baseline_pr=float((active.EstimatedOutstandingKGS*active.Rate*days/36500.).sum())
        if len(offers):
            new_interest,_=expected_amortization(offers.OfferAmountKGS.values,
                                  offers.OfferRatePct.values,offers.OfferTermMonths.values,days)
            incr_pr=float(np.sum(new_interest))*settings.conversion
            if days==30:
                # A campaign is assumed to disburse at horizon start, not across days.
                pass
        else:incr_pr=0.
        rows.append(dict(HorizonDays=days,ExpectedNewDPD30=events,
                         ExpectedLossKGS=loss,ExistingPRKGS=baseline_pr,
                         IncrementalPRKGS=incr_pr,
                         ScenarioPRKGS=baseline_pr+incr_pr,
                         GrowthPRPct=(incr_pr/baseline_pr*100 if baseline_pr>0 else 0)))
    horizon=pd.DataFrame(rows)
    by_family=active.groupby('ProductFamily').agg(
        ActiveLoans=('CreditID','size'),
        EstimatedBalanceKGS=('EstimatedOutstandingKGS','sum'),
        CurrentDPD30=('CurrentDPD',lambda x:int((x>=30).sum())),
        MeanPD12=('PD12MonitoringScenario','mean')).reset_index()
    kpi={
        'as_of':AS_OF.date().isoformat(),
        'active_loans':int(len(active)),
        'estimated_balance':float(active.EstimatedOutstandingKGS.sum()),
        'current_dpd30':int((active.CurrentDPD>=30).sum()),
        'currently_overdue':int((active.CurrentDPD>0).sum()),
        'candidates':int(len(offers)),
        'candidate_total':len(base_candidates()),
        'volume_offers':float(offers.OfferAmountKGS.sum()) if len(offers) else 0,
        'expected_accepted':len(offers)*settings.conversion,
        'test_auc':model_and_validation()[1]['roc_auc'],
        'test_pr_auc':model_and_validation()[1]['pr_auc'],
    }
    return kpi,horizon,by_family,active


def export_decisions(settings:DecisionSettings):
    rows=decision_table(settings)
    REPORTS.mkdir(exist_ok=True,parents=True)
    out=REPORTS/'recommended_customers.csv'
    public_columns=['CustomerID','RecommendedProduct','ProductCode','City','Region',
                    'OfferAmountKGS','OfferRatePct','OfferTermMonths','MonthlyPaymentKGS',
                    'PD12','ProposalDTI','Segment','ProjectedPR12M_KGS',
                    'ExpectedLoss12M_KGS','NetContribution12M_KGS','RecommendationReason']
    rows.reindex(columns=public_columns).to_csv(out,index=False,encoding='utf-8-sig')
    return str(out)
