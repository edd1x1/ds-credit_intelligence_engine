import numpy as np
import pandas as pd
from src.decision_engine_v6 import (
 DecisionSettings,decision_table,all_proposals,internal_dti,
 scenario_dashboard,explain_customer_offers,risk_model_and_validation,
)
from src.early_warning import read_snapshots,all_ews,monitoring_reports
from src.propensity import audit
from src.ml_core import PROJECT_ROOT

def test_payment_events_reconcile_contractual_schedule():
    raw=PROJECT_ROOT/'data/raw'
    due=pd.read_csv(raw/'repayment_obligations.csv')
    paid=pd.read_csv(raw/'repayment_transactions.csv')
    assert len(due)==497691
    assert due[['CreditID','InstallmentNo']].duplicated().sum()==0
    assert (paid.PaidAmountKGS>0).all()
    sums=paid.groupby(['CreditID','InstallmentNo']).PaidAmountKGS.sum().rename('paid')
    joined=due.join(sums,on=['CreditID','InstallmentNo'])
    assert (joined.paid.fillna(0)<=joined.DueAmountKGS+0.02).all()
    assert (joined.paid.fillna(0)>=0).all()
    assert paid.PaymentID.is_unique

def test_corrected_credit_status_not_based_only_on_maturity():
    df=pd.read_csv(PROJECT_ROOT/'data/processed/status_reconciliation.csv')
    assert df.CreditID.is_unique
    assert df.CorrectedStatusAtCutoff.isin(['Closed','Active']).all()
    assert (df.RemainingPastDueKGS>=0).all()

def test_snapshot_labels_have_future_horizon():
    s=read_snapshots()
    assert s[['CreditID','SnapshotDate']].duplicated().sum()==0
    assert (s.CurrentDPD>=0).all()
    for h in (30,60,90):
        target=f'TargetNewDPD30_{h}D'
        assert s[target].dropna().isin([0,1]).all()
        immature=s.SnapshotDate+pd.Timedelta(days=h)>pd.Timestamp('2026-10-31')
        assert s.loc[immature,target].isna().all()

def test_internal_dti_includes_existing_and_new_loans():
    value=internal_dti(20000,10000,100000)
    assert abs(value-0.30)<1e-9
    assert internal_dti(0,10000,100000)==.10

def test_early_warning_is_separate_from_issuance_risk():
    current,validation,reliability,ablation=all_ews()
    assert len(current)==4324
    assert validation.horizon_days.tolist()==[30,60,90]
    assert (validation.event_rate>0).all()
    assert (validation.pr_auc>=validation.event_rate).all()
    for h in (30,60,90):
        col=f'EW_PD30_{h}D'
        assert current.loc[current.CurrentDPD>=30,col].isna().all()
        assert current.loc[current.CurrentDPD<30,col].between(0,1).all()
    assert len(ablation)==3

def test_pd12_calibration_on_temporal_holdout():
    _,m,rel=risk_model_and_validation()
    assert m['n']>100
    assert 0<m['brier']<1 and 0<m['pr_auc']<1
    assert len(rel)==10

def test_multi_offer_decision_and_export():
    base=all_proposals()
    assert base.groupby('CustomerID').size().eq(4).all()
    d=decision_table(DecisionSettings())
    assert 0<len(d)<=300
    assert d.CustomerID.is_unique
    assert d.ActiveLoansAtSnapshot.eq(0).all()
    assert d.PD12.le(.18).all()
    assert d.ProposalDTI.le(.4).all()
    assert d.ExpectedValuePerContactKGS.gt(0).all()
    assert d.DecisionStatus.eq('PRE_OFFER_REVIEW').all()
    assert d.ExternalDebtStatus.eq('UNKNOWN_NOT_QUERIED').all()
    assert d.SelfBanStatus.eq('NOT_CHECKED_PREOFFER').all()
    assert d.MarketingConsentStatus.eq('UNKNOWN_REVIEW_REQUIRED').all()
    assert 'City' not in ['CurrentDPD','LoanAgeMonths','IncomeMean6M_KGS']

def test_contact_economics_counted_once():
    a=decision_table(DecisionSettings(conversion=.08,contact_cost=20))
    assert np.allclose(a.ExpectedValuePerContactKGS,
        a.ScenarioAcceptanceProbability*a.ContributionIfAcceptedKGS-20)

def test_risk_dashboard_reconciles():
    k,h,by,active=scenario_dashboard(DecisionSettings())
    assert len(h)==3
    assert h.HorizonDays.tolist()==[30,60,90]
    assert h.ExpectedNewDPD30.is_monotonic_increasing
    assert by.ActiveLoans.sum()==k['active_loans']==len(active)
    assert k['current_dpd30']<=k['currently_overdue']

def test_optional_propensity_no_fabricated_model():
    s=audit()
    assert s['status']=='NOT_TRAINED'
    assert s['events']==0
