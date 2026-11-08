"""Pure functions for the educational credit risk dashboard.

No regional geography / customer ID in risk training by default.
All fitted preprocessing is trained on earlier, fully-observed loan vintages.
Synthetic data: NOT for real credit decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    log_loss, precision_recall_fscore_support, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / 'data/processed/model_dataset.csv'
TARGET = 'TargetDPD30_12M'

# Point-in-time features; no label/time/ID/city/region, nor other engineered target aliases.
SAFE_NUMERIC_FEATURES: list[str] = [
    'PreviousLoansCount', 'ActivePreviousLoansCount', 'PreviousPaymentsObserved',
    'MaxHistoricalDPD', 'DPD30CountHistory', 'DPD30CountLast12M',
    'LatePaymentRatio', 'MonthsSinceLastDPD30', 'TotalPriorLoanAmountKGS',
    'NoCreditHistory', 'IncomeMean6M_KGS', 'IncomeCV6M',
    'CashflowMargin6M', 'IncomeTrend3v6M', 'AppActiveDays6M',
    'DebtToIncomeRatio', 'AgreementSumN', 'Rate', 'LoanTermMonths',
    'MonthlyPaymentKGS',
]
DEFAULT_FEATURES = [
    'MaxHistoricalDPD', 'PreviousLoansCount', 'DPD30CountLast12M',
    'LatePaymentRatio', 'IncomeCV6M', 'DebtToIncomeRatio',
]

FEATURE_RU = {
    'PreviousLoansCount': 'Прошлых кредитов',
    'ActivePreviousLoansCount': 'Действующих прошлых кредитов',
    'PreviousPaymentsObserved': 'Наблюдавшихся платежей',
    'MaxHistoricalDPD': 'Максимальный исторический DPD',
    'DPD30CountHistory': 'Исторических событий DPD 30+',
    'DPD30CountLast12M': 'DPD 30+ за последние 12 мес.',
    'LatePaymentRatio': 'Доля просроченных платежей',
    'MonthsSinceLastDPD30': 'Месяцев после последнего DPD 30+',
    'TotalPriorLoanAmountKGS': 'Сумма прошлых кредитов, сом',
    'NoCreditHistory': 'Нет кредитной истории',
    'IncomeMean6M_KGS': 'Средний доход 6 мес., сом',
    'IncomeCV6M': 'Вариативность дохода за 6 мес.',
    'CashflowMargin6M': 'Денежный поток / доход',
    'IncomeTrend3v6M': 'Тренд дохода 3 vs 6 мес.',
    'AppActiveDays6M': 'Дней активности за 6 мес.',
    'DebtToIncomeRatio': 'ПДН (DTI)',
    'AgreementSumN': 'Сумма кредита, сом',
    'Rate': 'Ставка, %',
    'LoanTermMonths': 'Срок кредита, мес.',
    'MonthlyPaymentKGS': 'Платёж по кредиту, сом',
}

@lru_cache(maxsize=1)
def read_model() -> pd.DataFrame:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'Файл не найден: {MODEL_PATH}')
    return pd.read_csv(MODEL_PATH, parse_dates=['SnapshotDate', 'LabelReadyDate'])


def time_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Leakage-conscious vintages for a didactic historical backtest.

    Train may only use labels observable by 2023-01-01 (validation start).
    Validation may only use labels available by 2025-01-01 (test start).
    Test labels are mature in the published dataset (as of the 2026 cutoff).
    2022 & 2024 issues are intentionally left unused in this conservative setup.
    """
    labeled = df[df[TARGET].isin((0, 1)) & df['IsMatureTarget'].eq(1)].copy()
    train = labeled[
        (labeled['SnapshotDate'] < '2023-01-01') &
        (labeled['LabelReadyDate'] <= '2023-01-01')
    ].copy()
    valid = labeled[
        (labeled['SnapshotDate'] >= '2023-01-01') &
        (labeled['SnapshotDate'] < '2024-01-01') &
        (labeled['LabelReadyDate'] <= '2025-01-01')
    ].copy()
    test = labeled[
        (labeled['SnapshotDate'] >= '2025-01-01') &
        (labeled['SnapshotDate'] < '2026-01-01')
    ].copy()
    for name, split in [('train', train), ('validation', valid), ('test', test)]:
        if len(split) == 0 or split[TARGET].nunique() < 2:
            raise ValueError(f'Недостаточно данных двух классов в {name}')
    return {'train': train, 'validation': valid, 'test': test}


@dataclass
class Features:
    x_train: np.ndarray
    x_valid: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    imputer: SimpleImputer
    scaler: StandardScaler
    names: list[str]


def prepare_features(splits: dict[str, pd.DataFrame], names: Iterable[str]) -> Features:
    names = list(names)
    if not names or len(set(names)) != len(names):
        raise ValueError('Выберите неповторяющиеся признаки')
    if any(name not in SAFE_NUMERIC_FEATURES for name in names):
        raise ValueError('Выбран неразрешённый или неизвестный признак')
    src = {k: splits[k][names].copy() for k in ('train','validation','test')}
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = StandardScaler()
    tr = scaler.fit_transform(imp.fit_transform(src['train']))
    va = scaler.transform(imp.transform(src['validation']))
    te = scaler.transform(imp.transform(src['test']))
    if not all(np.isfinite(a).all() for a in (tr,va,te)):
        raise ValueError('Найдены бесконечные значения признаков')
    return Features(
        tr, va, te,
        *[splits[k][TARGET].to_numpy(dtype=float) for k in ('train','validation','test')],
        imp, scaler, names,
    )


def with_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def objective_and_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray, l2: float = 0.0):
    """Mean logistic log loss + L2 penalty, intercept not penalized."""
    if l2 < 0:
        raise ValueError('L2 must be >= 0')
    z = X @ theta
    # logaddexp prevents overflow from exp(z) for extreme manual weights.
    m = len(y)
    base = np.mean(np.logaddexp(0, z) - y * z)
    regularization = 0.5 * l2 * np.dot(theta[1:], theta[1:]) / m
    errors = expit(z) - y
    grad = X.T @ errors / m
    grad[1:] += l2 * theta[1:] / m
    return float(base + regularization), grad


def gd_step(X, y, theta, learning_rate: float, l2: float = 0.0):
    loss, grad = objective_and_gradient(X,y,theta,l2)
    new_theta = theta - learning_rate * grad
    return new_theta, loss, grad


def train_gd(f: Features, lr: float, epochs: int, l2: float = 0.0,
             initial_theta: np.ndarray | None = None, batch_size: int = 0,
             seed: int = 42, checkpoint_every: int = 10):
    """Yield progress to show actual theta & loss evolving during training.

    Default is full batch gradient descent. Sampling uses a fixed subset of train.
    Validation and test are never used in gradient updates.
    """
    rng = np.random.default_rng(seed)
    x_full = with_intercept(f.x_train)
    if 0 < batch_size < len(x_full):
        idx = rng.choice(len(x_full), size=batch_size, replace=False)
        X, y = x_full[idx], f.y_train[idx]
    else:
        X,y = x_full, f.y_train
    Xv = with_intercept(f.x_valid)
    theta = np.zeros(X.shape[1], dtype=float) if initial_theta is None else np.asarray(initial_theta,dtype=float).copy()
    if len(theta) != X.shape[1]:
        raise ValueError('Длина theta не совпадает с числом признаков')
    history = []
    for epoch in range(epochs + 1):
        if epoch % max(1,checkpoint_every) == 0 or epoch == epochs:
            tr_loss, _ = objective_and_gradient(X,y,theta,l2)
            val_loss, _ = objective_and_gradient(Xv,f.y_valid,theta,l2)
            history.append({
                'epoch':epoch,
                'train_loss':tr_loss,
                'val_loss':val_loss,
                **{f'theta_{i}':float(v) for i,v in enumerate(theta)},
            })
            yield epoch, theta.copy(), pd.DataFrame(history)
        if epoch < epochs:
            theta, _, _ = gd_step(X,y,theta,lr,l2)


def probabilities(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return expit(with_intercept(X) @ theta)


def evaluate(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    p = np.clip(np.asarray(prob, dtype=float),1e-9,1-1e-9)
    pred = (p >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    return {
        'n': len(y), 'event_rate':float(np.mean(y)),
        'roc_auc':float(roc_auc_score(y,p)),
        'pr_auc':float(average_precision_score(y,p)),
        'log_loss':float(log_loss(y,p)),
        'brier':float(brier_score_loss(y,p)),
        'precision':float(precision), 'recall':float(recall), 'f1':float(f1),
        'cm':confusion_matrix(y,pred,labels=[0,1]),
    }
