"""Optional propensity model trained ONLY after internal campaign outcomes exist.
Never infer uptake probability from repayment records alone.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score
from .ml_core import PROJECT_ROOT

PATH=PROJECT_ROOT/'data/raw/internal_marketing_events.csv'
FIELDS=['CustomerID','OfferDate','ProductCode','Contacted','AcceptedWithin30D',
        'MarketingConsentAtOffer','IncomeMean6M_KGS','PreviousLoansCount',
        'MaxHistoricalDPD','AppActiveDays6M']
NUMS=['IncomeMean6M_KGS','PreviousLoansCount','MaxHistoricalDPD','AppActiveDays6M']

def audit():
    if not PATH.is_file():
        return {'status':'MISSING','events':0,'reason':'Нет истории внутренних кампаний; применяется сценарная конверсия.'}
    df=pd.read_csv(PATH)
    if df.empty:
        return {'status':'NOT_TRAINED','events':0,
                'reason':'История маркетинговых кампаний отсутствует. Вероятность отклика не обучена; конверсия задаётся сценарием.'}
    missing=set(FIELDS)-set(df.columns)
    if missing:return {'status':'INVALID_SCHEMA','events':len(df),'reason':f'Отсутствуют поля {sorted(missing)}'}
    ok=df.Contacted.eq(1)&df.AcceptedWithin30D.isin([0,1])
    eligible=df.loc[ok].copy()
    if len(eligible)<400 or eligible.AcceptedWithin30D.sum()<30:
        return {'status':'INSUFFICIENT','events':len(eligible),
            'reason':'Для обучения требуется >=400 завершённых контактов и >=30 принятий.'}
    return {'status':'READY','events':len(eligible),'reason':'Есть достаточно записей для временной валидации.'}

def train_propensity():
    """Train example: date-split history, only contact-time fields.
    Not an uplift/causal impact estimator. Caller must validate consent.
    """
    state=audit()
    if state['status']!='READY':return None,state
    df=pd.read_csv(PATH,parse_dates=['OfferDate'])
    df=df[df.Contacted.eq(1)&df.AcceptedWithin30D.isin([0,1])].sort_values('OfferDate')
    cutoff=df.OfferDate.quantile(.8)
    tr=df[df.OfferDate<cutoff];te=df[df.OfferDate>=cutoff]
    if te.AcceptedWithin30D.nunique()<2 or tr.AcceptedWithin30D.nunique()<2:
        return None,{'status':'INVALID_TARGET','reason':'Нет обоих классов в временном тесте.'}
    pre=ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMS),
        ('cat',OneHotEncoder(handle_unknown='ignore'),['ProductCode'])])
    model=Pipeline([('pre',pre),('lr',LogisticRegression(max_iter=500,random_state=42))])
    x=NUMS+['ProductCode']
    model.fit(tr[x],tr.AcceptedWithin30D.astype(int))
    p=model.predict_proba(te[x])[:,1]
    result={'status':'VALIDATED','train':len(tr),'test':len(te),
       'roc_auc':float(roc_auc_score(te.AcceptedWithin30D,p)),
       'pr_auc':float(average_precision_score(te.AcceptedWithin30D,p)),
       'base_rate':float(te.AcceptedWithin30D.mean())}
    return model,result
