"""Model governance artifacts for repeatable historical validation."""
from pathlib import Path
import json
from datetime import datetime,timezone
from src.decision_engine_v6 import risk_model_and_validation
from src.early_warning import all_ews,FEATURES
from src.propensity import audit
from src.ml_core import PROJECT_ROOT

registry={
 'project_version':'6.0',
 'as_of':'2026-10-31',
 'created_at_utc':datetime.now(timezone.utc).isoformat(),
 'data_origin':'SYNTHETIC_SCENARIO',
 'sources':['internal_loans','internal_repayments','internal_monthly_activity'],
 'external_bureau_calls':False,
 'automatic_credit_approval':False,
 'consent_status':'UNVERIFIED_REQUIRE_INTERNAL_CRM_CONFIRMATION',
 'dtiratio_scope':'SAME_BANK_ONLY',
 'model_use':'PRELIMINARY_MARKETING_ANALYTICS',
 'risk_model':{'target':'Any DPD30+ within 12 months after loan issue',
    'validation':'TRAIN historical / CALIBRATION 2023 / OOT 2025',
    'metrics':risk_model_and_validation()[1]},
 'early_warning':{'target':'New onset of DPD30+ in 30/60/90 days on active loans',
    'features':FEATURES,'validation':'TRAIN 2023-24 / CALIBRATION 2025Apr-Jun / OOT 2025Oct-2026Mar',
    'metrics':all_ews()[1].to_dict('records')},
 'propensity':audit(),
 'risk_controls':['PIT features','temporal holdout','calibration',
        'PSI','vintage','roll rates','ablation','human review'],
 'limitations':['Synthetic labels and behavioral assumptions',
   'External debts unknown; DTI internal only',
   'No marketing consent input',
   'No propensity model without actual campaign outcomes',
   'Scenario LGD/EAD and PR not accounting forecasts',
   'No self-ban check at pre-offer stage; regulated lending workflow remains separate']}
path=PROJECT_ROOT/'reports/model_registry_v6.json'
path.write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
print('Wrote',path)
