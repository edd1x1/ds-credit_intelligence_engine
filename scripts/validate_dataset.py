"""Structural, time-window and brand-neutral dataset checks. Run after generation."""
from pathlib import Path
import re
import json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
d=ROOT/'data'
cs=pd.read_csv(d/'raw/customers.csv')
ls=pd.read_csv(d/'raw/loans.csv',parse_dates=['CreditOpenDate','CreditEndDate'])
rs=pd.read_csv(d/'raw/repayment_schedule.csv',parse_dates=['DueDate','SettledDate'])
md=pd.read_csv(d/'processed/model_dataset.csv',parse_dates=['SnapshotDate','LabelReadyDate'])
mc=pd.read_csv(d/'processed/marketing_candidates.csv')
activity=pd.read_csv(d/'raw/customer_monthly_activity.csv',usecols=['CustomerID','MonthEnd','InflowsKGS','OutflowsKGS'],parse_dates=['MonthEnd'])
product=pd.read_csv(d/'reference/product_catalog.csv')
assert len(cs)==cs.CustomerID.nunique()
assert len(ls)==ls.CreditID.nunique()
assert len(md)==md.CreditID.nunique()
assert set(ls.CustomerID).issubset(set(cs.CustomerID))
assert set(rs.CreditID).issubset(set(ls.CreditID))
assert set(md.CreditID)==set(ls.CreditID)
assert len(activity)==len(cs)*activity.MonthEnd.nunique()
assert all(activity.InflowsKGS>0)
assert all(activity.OutflowsKGS>=0)
assert all(rs.DaysPastDueAtCutoff>=0)
assert all(rs.DueDate<=pd.Timestamp('2026-10-31'))
assert all(ls.AgreementSumN>0)
assert md.SnapshotDate.equals(ls.set_index('CreditID').reindex(md.CreditID)['CreditOpenDate'].reset_index(drop=True))
assert set(mc.CustomerID).issubset(set(cs.CustomerID))
assert mc.TargetDPD30_12M.isna().all() and mc.PropensityTarget.isna().all()
assert (md.loc[md.IsMatureTarget==0,'TargetDPD30_12M'].isna()).all()
assert (md.loc[md.IsMatureTarget==1,'TargetDPD30_12M'].notna()).all()
assert all(md.loc[md.IsMatureTarget==1,'LabelReadyDate']<=pd.Timestamp('2026-10-31'))
assert all(md.loc[md.IsMatureTarget==0,'LabelReadyDate']>pd.Timestamp('2026-10-31'))
assert md.NoCreditHistory.isin([0,1]).all()
assert (md.loc[md.PreviousLoansCount==0,'NoCreditHistory']==1).all()
assert (md.loc[md.PreviousLoansCount>0,'NoCreditHistory']==0).all()
limits=product.set_index('ProductCode').ScenarioMaxKGS
joined=ls.join(limits,on='ProductCode')
assert all(joined.AgreementSumN<=joined.ScenarioMaxKGS)
assert all((md.DebtToIncomeRatio>=0) & md.IncomeMean6M_KGS.gt(0))
# Direct recomputation of label from past-due installment events.
t=rs[['CreditID','DueDate','DaysPastDueAtCutoff']].merge(ls[['CreditID','CreditOpenDate']],on='CreditID')
first12=t[(t.DueDate>t.CreditOpenDate) & ((t.DueDate-t.CreditOpenDate).dt.days<=365)]
actual=first12.groupby('CreditID').DaysPastDueAtCutoff.max().ge(30)
check=md[md.IsMatureTarget==1][['CreditID','TargetDPD30_12M']].join(actual.rename('DerivedTarget'),on='CreditID')
assert (check.TargetDPD30_12M.astype(int)==check.DerivedTarget.fillna(False).astype(int)).all(), 'Target mismatch'
# All published loan product names must match the neutral scenario catalog.
allowed_names=set(product.ScenarioProductName)
assert ls.ProductName.isin(allowed_names).all()
report={'passed':True,'customers':len(cs),'loans':len(ls),'repayments':len(rs),
        'months':int(activity.MonthEnd.nunique()),'monthly_activity':len(activity),
        'mature_target_rows':int(md.IsMatureTarget.sum()),'marketing_candidates':len(mc),
        'city_stress_rate':md.loc[(md.City.isin(['Кара-Суу','Кара-Балта'])) & md.IsMatureTarget.eq(1),'TargetDPD30_12M'].mean(),
        'other_city_rate':md.loc[(~md.City.isin(['Кара-Суу','Кара-Балта'])) & md.IsMatureTarget.eq(1),'TargetDPD30_12M'].mean()}
(ROOT/'reports/validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
