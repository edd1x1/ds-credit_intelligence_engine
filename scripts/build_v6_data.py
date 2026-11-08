"""Upgrade v5 source exports to normalized events and point-in-time monitoring mart.
Reproducible scenario records. No banking identifiers or external data.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ASOF=pd.Timestamp('2026-10-31')
R=ROOT/'data/raw'; P=ROOT/'data/processed'

def build():
    l=pd.read_csv(R/'loans.csv',parse_dates=['CreditOpenDate','CreditEndDate'])
    r=pd.read_csv(R/'repayment_schedule.csv',parse_dates=['DueDate','SettledDate'])
    rng=np.random.default_rng(20261108)
    # Original installments are contractual obligations. Repayment transactions
    # are a separate event table. Splitting paid installments preserves final
    # settlement and target while testing partial-payment accounting.
    obligations=r[['CreditID','CustomerID','InstallmentNo','DueDate','ScheduledPaymentKGS']].copy()
    obligations.rename(columns={'ScheduledPaymentKGS':'DueAmountKGS'},inplace=True)
    obligations.to_csv(R/'repayment_obligations.csv',index=False)
    paid=r[r.SettledDate.notna()].copy()
    pieces=rng.random(len(paid))<.22
    frac=rng.uniform(.2,.65,len(paid))
    first=paid.loc[pieces,['CreditID','InstallmentNo','DueDate','SettledDate','ScheduledPaymentKGS']].copy()
    first['PaymentDate'] = first[['DueDate','SettledDate']].min(axis=1)
    first['PaidAmountKGS'] = np.round(first.ScheduledPaymentKGS*frac[pieces],2)
    last=paid[['CreditID','InstallmentNo','SettledDate','ScheduledPaymentKGS']].copy()
    last['PaymentDate']=last.SettledDate
    last['PaidAmountKGS']=last.ScheduledPaymentKGS.to_numpy()-np.where(pieces,np.round(paid.ScheduledPaymentKGS.to_numpy()*frac,2),0)
    openinst=r[r.SettledDate.isna() & (r.DueDate<=ASOF)].copy()
    partial=openinst.loc[rng.random(len(openinst))<.20,['CreditID','InstallmentNo','DueDate','ScheduledPaymentKGS']].copy()
    partial['PaymentDate']=partial.DueDate+pd.Timedelta(days=1)
    partial['PaidAmountKGS']=np.round(partial.ScheduledPaymentKGS*rng.uniform(.1,.4,len(partial)),2)
    events=pd.concat([first[['CreditID','InstallmentNo','PaymentDate','PaidAmountKGS']],
                      last[['CreditID','InstallmentNo','PaymentDate','PaidAmountKGS']],
                      partial[['CreditID','InstallmentNo','PaymentDate','PaidAmountKGS']]],ignore_index=True)
    events=events[events.PaidAmountKGS>0].sort_values(['CreditID','InstallmentNo','PaymentDate']).reset_index(drop=True)
    events.insert(0,'PaymentID',np.arange(1,len(events)+1))
    events.to_csv(R/'repayment_transactions.csv',index=False)
    # Detect closure using full obligation settlement as of cutoff, not maturity date.
    paid_tot=events[events.PaymentDate<=ASOF].groupby(['CreditID','InstallmentNo']).PaidAmountKGS.sum()
    o=obligations.join(paid_tot,on=['CreditID','InstallmentNo'])
    o['PaidAmountKGS']=o.PaidAmountKGS.fillna(0)
    o['RemainingKGS']=(o.DueAmountKGS-o.PaidAmountKGS).clip(lower=0)
    unpaid_past=o[(o.DueDate<=ASOF)&(o.RemainingKGS>.01)].groupby('CreditID').RemainingKGS.sum()
    l['RemainingPastDueKGS']=l.CreditID.map(unpaid_past).fillna(0)
    l['CorrectedStatusAtCutoff']=np.where((l.CreditEndDate<=ASOF)&(l.RemainingPastDueKGS<=.01),'Closed','Active')
    l[['CreditID','CustomerID','CreditStatusAtCutoff','CorrectedStatusAtCutoff','RemainingPastDueKGS']].to_csv(P/'status_reconciliation.csv',index=False)
    # Monthly internal activity: only observation known up to that month.
    m=pd.read_csv(R/'customer_monthly_activity.csv',usecols=['CustomerID','MonthEnd','InflowsKGS','OutflowsKGS','MonthEndBalanceKGS','AppActivityDays'],parse_dates=['MonthEnd'])
    m.sort_values(['CustomerID','MonthEnd'],inplace=True)
    m['IncomeMean6M_KGS']=m.groupby('CustomerID')['InflowsKGS'].transform(lambda s:s.rolling(6,min_periods=3).mean())
    income_6=m.groupby('CustomerID')['InflowsKGS'].transform(lambda x:x.rolling(6,min_periods=3).sum())
    expense_6=m.groupby('CustomerID')['OutflowsKGS'].transform(lambda x:x.rolling(6,min_periods=3).sum())
    m['CashflowMargin6M']=(income_6-expense_6)/income_6.clip(lower=1)
    m=m[['CustomerID','MonthEnd','IncomeMean6M_KGS','CashflowMargin6M','AppActivityDays']]
    # Time-varying observations; labels are computed from FUTURE settlement events.
    snapshots=[]
    for t in pd.date_range('2023-01-31','2026-10-31',freq='ME'):
        avail=l[l.CreditOpenDate<=t].copy()
        due=r[r.DueDate<=t].copy()
        outstanding=due[due.SettledDate.isna() | (due.SettledDate>t)]
        if len(outstanding):
            earliest=outstanding.groupby('CreditID').DueDate.min()
            due_amt=outstanding.groupby('CreditID').ScheduledPaymentKGS.sum()
        else:
            earliest=pd.Series(dtype='datetime64[ns]'); due_amt=pd.Series(dtype=float)
        avail['OldestUnpaidDueDate']=avail.CreditID.map(earliest)
        avail['CurrentDPD']=(t-avail.OldestUnpaidDueDate).dt.days.clip(lower=0).fillna(0).astype(int)
        avail['DueUnpaidKGS']=avail.CreditID.map(due_amt).fillna(0)
        avail=avail[(avail.CreditEndDate>t)|(avail.DueUnpaidKGS>.01)].copy()
        if avail.empty:continue
        recent=due[due.DueDate>t-pd.Timedelta(days=180)]
        recent_amt=recent.groupby('CreditID').agg(
            PaymentsDue6M=('InstallmentNo','count'),
            MeanPaymentDue6M=('ScheduledPaymentKGS','mean'))
        avail=avail.join(recent_amt,on='CreditID')
        avail['PaymentsDue6M']=avail.PaymentsDue6M.fillna(0)
        avail['MeanPaymentDue6M']=avail.MeanPaymentDue6M.fillna(avail.MonthlyPaymentKGS)
        avail['LoanAgeMonths']=(t-avail.CreditOpenDate).dt.days.clip(lower=0)/30.4375
        avail['TermMonthsRemaining']=(avail.LoanTermMonths-avail.LoanAgeMonths).clip(lower=0)
        avail['SnapshotDate']=t
        avail=avail.merge(m[m.MonthEnd.eq(t)],left_on=['CustomerID','SnapshotDate'],right_on=['CustomerID','MonthEnd'],how='left',validate='many_to_one')
        # Sum contractual payments on same bank's active loans, point in time.
        debt=avail.groupby('CustomerID').MonthlyPaymentKGS.transform('sum')
        avail['InternalMonthlyDebtKGS']=debt
        avail['InternalDTI']=debt/avail.IncomeMean6M_KGS.clip(lower=1)
        # New 30+ onset: crossing due+30 inside future horizon, and unpaid
        # through that date. Existing DPD30+ loans are excluded from new events.
        onset=r[(r.DueDate>=t-pd.Timedelta(days=29)) &
                ((r.SettledDate.isna())|(r.SettledDate >= r.DueDate+pd.Timedelta(days=30)))][['CreditID','DueDate']].copy()
        onset['CrossDate']=onset.DueDate+pd.Timedelta(days=30)
        onset=onset[onset.CrossDate>t]
        for h in (30,60,90):
            observed=t+pd.Timedelta(days=h)<=ASOF
            count=onset.loc[onset.CrossDate<=t+pd.Timedelta(days=h),'CreditID'].unique() if observed else []
            avail[f'TargetNewDPD30_{h}D']=np.where(observed,avail.CreditID.isin(count).astype(float),np.nan)
        cols=['CustomerID','CreditID','SnapshotDate','ProductFamily','ProductCode',
              'AgreementSumN','Rate','LoanTermMonths','MonthlyPaymentKGS',
              'CurrentDPD','DueUnpaidKGS','PaymentsDue6M','MeanPaymentDue6M',
              'LoanAgeMonths','TermMonthsRemaining','IncomeMean6M_KGS',
              'CashflowMargin6M','AppActivityDays','InternalMonthlyDebtKGS','InternalDTI',
              'TargetNewDPD30_30D','TargetNewDPD30_60D','TargetNewDPD30_90D']
        snapshots.append(avail[cols])
    mart=pd.concat(snapshots,ignore_index=True)
    mart=mart[~mart.duplicated(['CreditID','SnapshotDate'])].copy()
    mart.to_csv(P/'early_warning_snapshots.csv',index=False)
    print('Obligations',len(o),'payments',len(events),'snapshots',len(mart),'status changes',int((l.CreditStatusAtCutoff!=l.CorrectedStatusAtCutoff).sum()))
    print('Early warning labels', {h:(int(mart[f'TargetNewDPD30_{h}D'].notna().sum()),int(mart[f'TargetNewDPD30_{h}D'].sum())) for h in (30,60,90)})
    return mart

if __name__=='__main__':build()
