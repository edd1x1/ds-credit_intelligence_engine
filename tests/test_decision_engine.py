"""Reproducibility and scenario-rule checks for the interactive decision engine."""
import numpy as np
from src.decision_engine import (
    DecisionSettings, RECOMMENDABLE_CODES, annuity_payment,
    principal_from_payment, pd_to_horizon, decision_table,
    active_portfolio, scenario_dashboard, model_and_validation,
    export_decisions,
)


def test_monthly_payment_and_inverse():
    p=np.array([100000.,200000.,500000.])
    r=np.array([0.,18.,24.])
    n=np.array([3,18,36])
    assert np.allclose(principal_from_payment(annuity_payment(p,r,n),r,n),p)


def test_horizon_probability_increasing():
    p=np.array([.01,.15,.45])
    a=pd_to_horizon(p,30)
    b=pd_to_horizon(p,90)
    assert (a>0).all() and (a<b).all() and (b<p).all()


def test_decision_rules_default():
    s=DecisionSettings()
    d=decision_table(s)
    assert 0<len(d)<=s.top_n
    assert d.CustomerID.is_unique
    assert d.ActiveLoansAtSnapshot.eq(0).all()
    assert d.ProductCode.isin(RECOMMENDABLE_CODES).all()
    assert d.NoCreditHistory.eq(0).all()
    assert d.PD12.le(s.max_pd+1e-10).all()
    assert d.ProposalDTI.le(s.max_dti+1e-10).all()
    assert d.MaxHistoricalDPD.le(s.max_hist_dpd).all()
    assert d.NetContribution12M_KGS.gt(0).all()
    assert d.OfferAmountKGS.le(d.ProductMaxKGS).all()
    assert d.OfferAmountKGS.ge(d.ProductMinKGS).all()
    assert set(d.ProductCode.unique())=={'CASH_STANDARD_500','CASH_ONLINE_200','INSTALL_3M'}


def test_higher_restriction_no_new_passes():
    d=decision_table(DecisionSettings(max_pd=0.05,top_n=50))
    assert len(d)==0 or d.PD12.le(0.05+1e-10).all()


def test_dashboard_outstanding_and_pr_horizons():
    k,h,by_family,active=scenario_dashboard(DecisionSettings())
    assert k['active_loans']==len(active)
    assert k['active_loans']>0
    assert k['currently_overdue']>=k['current_dpd30']
    assert h.HorizonDays.tolist()==[30,60,90]
    assert h.ExpectedNewDPD30.is_monotonic_increasing
    assert h.ExistingPRKGS.is_monotonic_increasing
    assert h.IncrementalPRKGS.is_monotonic_increasing
    assert by_family.ActiveLoans.sum()==k['active_loans']
    assert active.EstimatedOutstandingKGS.ge(0).all()


def test_model_has_time_holdout():
    model,metrics,known=model_and_validation()
    assert metrics['n']>100
    assert 0<=metrics['roc_auc']<=1
    assert 0<=metrics['pr_auc']<=1
    assert known>1000


def test_export_csv():
    from pathlib import Path
    path=Path(export_decisions(DecisionSettings()))
    assert path.exists()
    assert path.stat().st_size>500
    line=path.read_text(encoding='utf-8-sig').splitlines()[0]
    assert 'CustomerID' in line and 'RecommendedProduct' in line
