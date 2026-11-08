"""Offline refresh from provided internal CSVs, no network or bureau calls."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_v6_data import build
from src.decision_engine_v6 import save_reports
from src.early_warning import read_snapshots,all_ews
from src.decision_engine_v6 import risk_model_and_validation,decision_table,all_proposals

if __name__=='__main__':
    build()
    read_snapshots.cache_clear();all_ews.cache_clear()
    risk_model_and_validation.cache_clear();decision_table.cache_clear();all_proposals.cache_clear()
    save_reports()
    from scripts.build_model_card import registry
    print('Data + models + reports refreshed')
