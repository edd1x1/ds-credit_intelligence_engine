"""Neutralize the fully synthetic reference files and add point-in-time cashflows.
Run once from the provided underlying synthetic CSV files (included in the package).
No actual persons or institution-specific client data are used.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/source_synthetic'
RAW=ROOT/'data/raw'; PROCESSED=ROOT/'data/processed'; REFERENCE=ROOT/'data/reference'; REPORTS=ROOT/'reports'
for p in (RAW,PROCESSED,REFERENCE,REPORTS):p.mkdir(parents=True,exist_ok=True)
seed=20261103
rng=np.random.default_rng(seed)
ASOF=pd.Timestamp('2026-10-31')
product_names={
'CASH_ONLINE_200':'Онлайн-кредит 200',
'CASH_STANDARD_500':'Наличные Стандарт 500',
'CASH_GUARANT_1M':'Наличные с поручителем 1 млн',
'SECURED_STANDARD':'Наличные под залог Стандарт',
'SECURED_PLUS':'Наличные под залог Плюс',
'AUTO_USED':'Автокредит: с пробегом',
'AUTO_DEALER':'Автокредит: новый',
'MORTGAGE_STANDARD':'Ипотека Стандарт',
'MORTGAGE_BANK_PROPERTY':'Ипотека Готовое жильё',
'INSTALL_3M':'Рассрочка 3 месяца',
'INSTALL_6_12M':'Рассрочка 6–12 месяцев',
'INSTALL_24M':'Рассрочка 24 месяца'
}
code_names={'CASH_GUARANT_1M':'CASH_GUARANTOR_1M','AUTO_DEALER':'AUTO_NEW_18M','MORTGAGE_BANK_PROPERTY':'MORTGAGE_PROPERTY_15M'}
cs=pd.read_csv(BASE/'customers.csv',encoding='utf-8-sig')
ls=pd.read_csv(BASE/'loans.csv',encoding='utf-8-sig')
ps=pd.read_csv(BASE/'repayments.csv',encoding='utf-8-sig')
md=pd.read_csv(BASE/'model_dataset.csv',encoding='utf-8-sig')
mc=pd.read_csv(BASE/'marketing_candidates.csv',encoding='utf-8-sig')
prd=pd.read_csv(BASE/'product_catalog.csv',encoding='utf-8-sig')
geo=pd.read_csv(BASE/'geography_scenarios.csv',encoding='utf-8-sig')

cs=cs.rename(columns={'SyntheticMonthlyIncomeKGS':'SyntheticIncome2026KGS','IncomeVolatility':'SyntheticRiskVolatility'})
cs=cs.drop(columns=['GeographyUse'],errors='ignore')
ls['ProductName']=ls['ProductCode'].map(product_names)
ls['ProductCode']=ls['ProductCode'].replace(code_names)
ls=ls.rename(columns={'OtherMonthlyDebtKGS':'OtherDebtPaymentAtIssueKGS','CreditStatus':'CreditStatusAtCutoff'})
ps=ps.rename(columns={'DueAmountKGS':'ScheduledPaymentKGS','PaymentDate':'SettledDate','PaidAmountKGS':'SettledAmountKGS',
                      'DPDAtCutoff':'DaysPastDueAtCutoff','IsPaymentCompletedAtCutoff':'PaidByCutoff'})
ps['RepaymentStateAtCutoff']=np.where(ps['PaidByCutoff'].eq(1),'PAID','OPEN')
ps['DaysPastDueAtCutoff']=ps['DaysPastDueAtCutoff'].clip(lower=0)
# If there is any unpaid scheduled installment, a matured loan is not classified as closed.
unpaid=ps.loc[ps['PaidByCutoff'].eq(0),'CreditID'].unique()
ls.loc[ls['CreditID'].isin(unpaid),'CreditStatusAtCutoff']='Active'

md['ProductCode']=md['ProductCode'].replace(code_names)
mc=mc.rename(columns={'Age':'AgeAtSnapshot','PastDPD30Count':'HistoricalDPD30Count'})
prd['ScenarioSubproductName']=prd['ProductCode'].map(product_names)
prd['ProductCode']=prd['ProductCode'].replace(code_names)
prd=prd.drop(columns=['PublicListedSubproduct','SourceURL'],errors='ignore')
prd=prd.rename(columns={'ScenarioSubproductName':'ScenarioProductName'})
prd['PublicLimitVerified']='PUBLIC_FAMILY_LIMIT'
prd['ScenarioSubproductIsSynthetic']=1
geo=geo.drop(columns=['Source','Note'],errors='ignore')
geo=geo.rename(columns={'ExtraSeverePaymentLogOddsScenario':'SyntheticExtraPaymentRiskLogOdds'})
geo['ScenarioInterpretation']='HYPOTHETICAL_STRESS_NOT_OBSERVED_DEFAULT_RATES'

# 88 months 2019-07 .. 2026-10, vectorized and repeatable, with no future activity
# used for an earlier origination record. Earnings/spend and town shocks are invented.
months=pd.date_range('2019-07-01','2026-10-01',freq='MS')
ends=months+pd.offsets.MonthEnd(0)
N=len(cs);M=len(months)
base=cs['SyntheticIncome2026KGS'].to_numpy(dtype=float)[:,None]
latent_vol=cs['SyntheticRiskVolatility'].to_numpy(dtype=float)[:,None]
month_num=np.arange(M)[None,:]
years_from_cutoff=(M-1-month_num)/12
trend=np.exp(-.032*years_from_cutoff)
season=1+.035*np.sin(2*np.pi*month_num/12 + np.arange(N)[:,None]*.17)
noise=rng.normal(size=(N,M))*np.maximum(.035,latent_vol*.48)
shock=((cs['City'].isin(['Кара-Балта','Кара-Суу']).to_numpy())[:,None] &
       (np.asarray([(2022<=m.year<=2024 and m.month in (2,3,4,9,10)) for m in months])[None,:]))
common=np.array([.98 if m.year in (2022,2024) else 1.0 for m in months],dtype=float)[None,:]
inflows=np.maximum(5000,np.round(base*trend*season*np.exp(noise-.5*(latent_vol*.48)**2)*(1-.10*shock)*common/100)*100).astype(int)
spend_ratio=np.clip(.60+.04*rng.normal(size=(N,1))+.095*rng.normal(size=(N,M)),.27,1.15)
outflows=np.round(inflows*spend_ratio/100)*100
balances=np.maximum(0,np.round((inflows-outflows)*rng.uniform(.25,2.4,size=(N,M))/100)*100)
app_days=np.minimum(30,np.maximum(0,rng.poisson(lam=np.clip(15-(cs['AgeAt2026'].to_numpy()[:,None]-27)*.14,3,18),size=(N,M))))
salary=(cs['Occupation'].isin(['Наемный сотрудник','Госслужащий']).to_numpy(dtype=int))
activity_file=RAW/'customer_monthly_activity.csv'
# Write in bounded month chunks without a giant in-memory row-per-event list.
with activity_file.open('w',encoding='utf-8-sig',newline='') as f:
    f.write('CustomerID,MonthEnd,InflowsKGS,OutflowsKGS,MonthEndBalanceKGS,SalaryFlag,AppActivityDays,IncomeShockFlag,DataOrigin\n')
    for m in range(M):
        data=pd.DataFrame({
           'CustomerID':cs['CustomerID'].to_numpy(),
           'MonthEnd':ends[m].strftime('%Y-%m-%d'),
           'InflowsKGS':inflows[:,m],
           'OutflowsKGS':outflows[:,m].astype(int),
           'MonthEndBalanceKGS':balances[:,m].astype(int),
           'SalaryFlag':salary,'AppActivityDays':app_days[:,m],
           'IncomeShockFlag':shock[:,m].astype(int),'DataOrigin':'SYNTHETIC'})
        data.to_csv(f,index=False,header=False)

# All 6-month features use COMPLETE prior month-end snapshots, strictly less than issuance.
customer_index=md['CustomerID'].str.extract(r'(\d+)$')[0].astype(int).to_numpy()-1
issue_dates=pd.to_datetime(md['SnapshotDate']).to_numpy(dtype='datetime64[ns]')
end_dates=ends.to_numpy(dtype='datetime64[ns]')
last=np.searchsorted(end_dates,issue_dates,side='left')
if np.any(last<6):raise ValueError('Missing 6 months of activity history')
ix=last[:,None]-np.arange(6,0,-1)[None,:]
six_in=inflows[customer_index[:,None],ix]
six_out=outflows[customer_index[:,None],ix]
six_app=app_days[customer_index[:,None],ix]
mean6=six_in.mean(axis=1)
mean_out=six_out.mean(axis=1)
cv=six_in.std(axis=1)/np.maximum(mean6,1)
margin=(mean6-mean_out)/np.maximum(mean6,1)
trend3=six_in[:,3:].mean(axis=1)/np.maximum(six_in[:,:3].mean(axis=1),1)-1
md['IncomeMean6M_KGS']=mean6.round(2)
md['IncomeCV6M']=cv.round(4)
md['CashflowMargin6M']=margin.round(4)
md['IncomeTrend3v6M']=trend3.round(4)
md['AppActiveDays6M']=six_app.sum(axis=1)
md['DebtToIncomeRatioAtIssueNew']=(md['MonthlyPaymentKGS'].to_numpy()+ls.set_index('CreditID').reindex(md['CreditID'])['OtherDebtPaymentAtIssueKGS'].to_numpy())/np.maximum(mean6,1)
# Keep original scenario underwriting DTI to distinguish it from the retroactively computed observed DTI.
md=md.rename(columns={'DebtToIncomeRatio':'SyntheticUnderwritingDTI'})
md['DebtToIncomeRatio']=md['DebtToIncomeRatioAtIssueNew'].round(4)
md=md.drop(columns=['DebtToIncomeRatioAtIssueNew','SyntheticMonthlyIncomeKGS','IncomeVolatility'])
# Target availability is explicit. Target blank if a year cannot be fully observed.
issue_dt=pd.to_datetime(md['SnapshotDate'])
md['LabelReadyDate']=(issue_dt+pd.DateOffset(months=12)+pd.Timedelta(days=30)).dt.strftime('%Y-%m-%d')
md['TargetDPD30_12M']=md['TargetDPD30_12M'].where(md['IsMatureTarget'].eq(1),np.nan)

# This subset has no current credit and UNKNOWN future repayment/conversion labels.
mi=mc['CustomerID'].str.extract(r'(\d+)$')[0].astype(int).to_numpy()-1
mc['IncomeMean6M_KGS']=inflows[mi,-6:].mean(axis=1).round(2)
mc['IncomeCV6M']=(inflows[mi,-6:].std(axis=1)/np.maximum(1,inflows[mi,-6:].mean(axis=1))).round(4)
mc['CashflowMargin6M']=((inflows[mi,-6:].mean(axis=1)-outflows[mi,-6:].mean(axis=1))/np.maximum(1,inflows[mi,-6:].mean(axis=1))).round(4)
mc['AppActiveDays6M']=app_days[mi,-6:].sum(axis=1)
mc=mc.drop(columns=['SyntheticMonthlyIncomeKGS','IncomeVolatility'],errors='ignore')

for f in (cs,ls,ps,md,mc,prd,geo):
    if 'DataOrigin' in f.columns:f['DataOrigin']='SYNTHETIC'
cs.drop(columns=['SyntheticRiskVolatility'],inplace=True)
cs.to_csv(RAW/'customers.csv',index=False,encoding='utf-8-sig')
ls.to_csv(RAW/'loans.csv',index=False,encoding='utf-8-sig')
ps.to_csv(RAW/'repayment_schedule.csv',index=False,encoding='utf-8-sig')
md.to_csv(PROCESSED/'model_dataset.csv',index=False,encoding='utf-8-sig')
mc.to_csv(PROCESSED/'marketing_candidates.csv',index=False,encoding='utf-8-sig')
prd.to_csv(REFERENCE/'product_catalog.csv',index=False,encoding='utf-8-sig')
geo.to_csv(REFERENCE/'geography_scenarios.csv',index=False,encoding='utf-8-sig')
md_labeled=md.loc[md['IsMatureTarget'].eq(1)].copy()
by_city=md_labeled.groupby('City').agg(LoanCount=('CreditID','size'),BadRate12M=('TargetDPD30_12M','mean')).sort_values('BadRate12M',ascending=False)
by_prod=md_labeled.groupby(['ProductFamily','ProductCode']).agg(LoanCount=('CreditID','size'),BadRate12M=('TargetDPD30_12M','mean'),MeanLoanKGS=('AgreementSumN','mean')).reset_index()
by_city.to_csv(REPORTS/'eda_city_rates.csv',encoding='utf-8-sig')
by_prod.to_csv(REPORTS/'eda_product_rates.csv',index=False,encoding='utf-8-sig')
meta={'dataset_version':'3.0','random_seed':seed,'snapshot_date':ASOF.strftime('%Y-%m-%d'),
      'records':{'customers':len(cs),'loans':len(ls),'installments':len(ps),'monthly_activity':N*M,'model_observations':len(md),'mature_targets':len(md_labeled),'marketing_candidates':len(mc)},
      'target_rate_dpd30_12m':float(md_labeled['TargetDPD30_12M'].mean()),
      'time_split_counts':md_labeled['TimeSplit'].value_counts().to_dict(),
      'synthetic_risk_scenario':['Кара-Суу','Кара-Балта'],
      'sources_scope':'Real city names and public maximum loan limits; all borrowers and risk distributions invented',
      'caveats':['Synthetic only. No real account identifiers or bank client observations.',
                 'No actual city-level default statistics underpin city stress assumption.',
                 'Monthly cashflows are synthetic correlated signals; original generated labels were simulated using borrower capacity and latent factors.',
                 'Payment schedule assumes complete installment settlements without partial payments or restructures.',
                 'Product subtypes and nominal rates are illustrative.',
                 'Geography is not a risk-model feature by default; use it for validation and scenario monitoring.',
                 'For prospective marketing, conversion and future repayment target are unknown.',
                 'Last 12-month issue cohorts without follow-up are deliberately unlabeled.']}
(ROOT/'dataset_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(meta,ensure_ascii=False,indent=2))
print('\nCITY RATES\n',by_city.head(8).to_string())
