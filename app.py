from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from catboost import CatBoostClassifier

from src.ml_core import (
    PROJECT_ROOT, MODEL_PATH, TARGET, SAFE_NUMERIC_FEATURES, DEFAULT_FEATURES,
    FEATURE_RU, read_model, time_splits, prepare_features,
    objective_and_gradient, gd_step, train_gd, probabilities, evaluate, with_intercept,
)

from src.decision_engine import (
    DecisionSettings, decision_table, scenario_dashboard, export_decisions,
    model_and_validation, pd_to_horizon, active_portfolio,
)
from src.decision_engine_v6 import (
    DecisionSettings, decision_table, scenario_dashboard, export_decisions,
    current_portfolio_v6, risk_model_and_validation, explain_customer_offers,
    all_proposals,
)
from src.early_warning import all_ews, monitoring_reports


DATASETS = {
    'Обучающая выборка — модель': 'data/processed/model_dataset.csv',
    'Аудитория без активных кредитов': 'data/processed/marketing_candidates.csv',
    'Клиенты': 'data/raw/customers.csv',
    'Кредитные договоры': 'data/raw/loans.csv',
    'Исходные графики платежей': 'data/raw/repayment_schedule.csv',
    'Плановые обязательства': 'data/raw/repayment_obligations.csv',
    'Фактические платежные транзакции': 'data/raw/repayment_transactions.csv',
    'Помесячные снимки Early Warning': 'data/processed/early_warning_snapshots.csv',
    'Сверка статусов кредитов': 'data/processed/status_reconciliation.csv',
    'Месячные финансы': 'data/raw/customer_monthly_activity.csv',
    'Каталог кредитных продуктов': 'data/reference/product_catalog.csv',
    'География стресс-сценария': 'data/reference/geography_scenarios.csv',
}


def empty_plot(title='Выберите параметры и запустите расчёт'):
    fig=go.Figure()
    fig.update_layout(title=title, template='plotly_white',height=360,
                      margin=dict(l=45,r=20,t=60,b=45))
    return fig


def preview_dataset(which: str, n_rows: int):
    path=PROJECT_ROOT / DATASETS[which]
    if not path.exists():
        return f'Не найден файл `{path}`. Для всех исходных таблиц используйте **полный** архив.', pd.DataFrame()
    n=int(n_rows)
    df=pd.read_csv(path,nrows=n)
    desc=(f'**{which}** — {len(df):,} первых строк, {len(df.columns)} столбцов; '
          f'размер файла {path.stat().st_size/1024/1024:.1f} МБ. '
          'Это срез, не вся таблица.')
    return desc,df


@lru_cache(maxsize=1)
def get_splits():
    return time_splits(read_model())


def eda_dashboard(family: str, grouping: str, numeric: str, min_group: int):
    data=read_model()
    d=data[data[TARGET].notna()].copy()
    if family!='Все продукты':
        d=d[d['ProductFamily']==family]
    if len(d)==0:
        return 'Нет данных по выбранному фильтру',empty_plot(),empty_plot(),empty_plot(),empty_plot()
    total=len(d); events=int(d[TARGET].sum()); event_rate=d[TARGET].mean()
    split=d.groupby(d['SnapshotDate'].dt.year).agg(
        n=(TARGET,'size'), rate=(TARGET,'mean')).reset_index(names='year')
    group=d.groupby(grouping,dropna=False).agg(n=(TARGET,'size'),bad_rate=(TARGET,'mean')).reset_index()
    group=group[group['n']>=int(min_group)].sort_values('bad_rate',ascending=False)
    fig1=go.Figure()
    fig1.add_trace(go.Bar(x=group[grouping].fillna('Нет данных').astype(str),y=group['bad_rate']*100,
                          customdata=group['n'],hovertemplate='%{x}<br>Bad rate: %{y:.1f}%<br>n=%{customdata}<extra></extra>'))
    fig1.update_layout(title='Доля DPD30+ по сегментам',yaxis_title='Доля, %',template='plotly_white',height=400,
                       xaxis_tickangle=-25,margin=dict(l=45,r=15,t=60,b=110))
    fig2=px.histogram(d,x=numeric,nbins=45,
                       title=f'Распределение: {numeric}',template='plotly_white')
    fig2.update_layout(height=360,margin=dict(l=45,r=20,t=60,b=45))
    fig3=go.Figure()
    fig3.add_trace(go.Scatter(x=split.year,y=split.rate*100,mode='lines+markers',name='Bad rate'))
    fig3.update_layout(title='Доля проблемных кредитов по году выдачи',template='plotly_white',
                       xaxis_title='Год',yaxis_title='DPD30+ (%)',height=360,
                       margin=dict(l=45,r=15,t=60,b=45))
    md=(f'**Кредитов:** {total:,} · **DPD30+:** {events:,} · '
        f'**Доля:** {event_rate:.1%} · **Период:** '
        f'{d.SnapshotDate.min().date()} — {d.SnapshotDate.max().date()}\n\n'
        'Географические различия здесь используются для исследовательской сегментации, '
        'а не как самостоятельный фактор отказа или ставки.')
    corr_features=list(dict.fromkeys([numeric]+DEFAULT_FEATURES+[TARGET]))
    correlations=d[corr_features].corr()
    fig4=px.imshow(correlations,text_auto='.2f',zmin=-1,zmax=1,color_continuous_scale='RdBu',
                   title='Корреляция признаков и Target (Pearson)',template='plotly_white')
    fig4.update_layout(height=560,margin=dict(l=45,r=20,t=60,b=60))
    return md,fig1,fig2,fig3,fig4


def loss_chart(history):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=history.epoch,y=history.train_loss,mode='lines+markers',name='Train Log Loss'))
    fig.add_trace(go.Scatter(x=history.epoch,y=history.val_loss,mode='lines+markers',name='Validation Log Loss'))
    fig.update_layout(title='Сходимость градиентного спуска',template='plotly_white',
                      xaxis_title='Эпоха',yaxis_title='Log Loss + L2',height=390,
                      margin=dict(l=45,r=20,t=60,b=45))
    return fig


def weights_chart(history,features):
    fig=go.Figure()
    for i,feature in enumerate(['θ₀ (intercept)']+features):
        fig.add_trace(go.Scatter(x=history.epoch,y=history[f'theta_{i}'],mode='lines',name=feature))
    fig.update_layout(title='Как изменяются веса θ',template='plotly_white',
                      xaxis_title='Эпоха',yaxis_title='θ на стандартизованных признаках',height=390,
                      margin=dict(l=45,r=20,t=60,b=45))
    return fig


def roc_figure(y,prob,which='Test'):
    fpr,tpr,_=roc_curve(y,prob)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',name=f'{which} ROC'))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(dash='dash'),name='Random'))
    fig.update_layout(title=f'ROC-кривая — {which}',template='plotly_white',
                      xaxis_title='False Positive Rate',yaxis_title='True Positive Rate',height=380,
                      margin=dict(l=45,r=20,t=60,b=45))
    return fig


def confusion_figure(cm):
    fig=px.imshow(cm,text_auto=True,labels=dict(x='Прогноз',y='Факт',color='Кредитов'),
                  x=['Good (0)','Bad (1)'],y=['Good (0)','Bad (1)'],
                  title='Confusion Matrix — Test',color_continuous_scale='Blues')
    fig.update_layout(height=380,template='plotly_white',margin=dict(l=45,r=20,t=60,b=45))
    return fig


def score_table(rows):
    return pd.DataFrame(rows,columns=['Выборка','N','Bad rate','ROC-AUC','PR-AUC','Log Loss','Brier'])


def score_row(label,y,prob):
    m=evaluate(y,prob)
    return [label,m['n'],round(m['event_rate'],3),round(m['roc_auc'],3),
            round(m['pr_auc'],3),round(m['log_loss'],3),round(m['brier'],3)]


def train_manual(features: list[str],lr: float,epochs: int,l2: float,sample: str,
                 threshold: float,progress=gr.Progress()):
    if len(features)<2 or len(features)>8:
        raise gr.Error('Выберите от 2 до 8 числовых признаков')
    f=prepare_features(get_splits(),features)
    bs=0 if sample=='Все данные' else int(sample)
    every=max(1,int(epochs)//24)
    progress(0,desc='Градиентный спуск')
    for step,theta,history in train_gd(f,float(lr),int(epochs),float(l2),
                                      batch_size=bs,checkpoint_every=every):
        progress(step/max(1,int(epochs)),desc=f'Эпоха {step}/{epochs}')
        tp=probabilities(f.x_train,theta)
        vp=probabilities(f.x_valid,theta)
        rows=score_table([score_row('Train',f.y_train,tp),score_row('Validation',f.y_valid,vp)])
        state={ 'theta':theta, 'names':features, 'x_test':f.x_test,'y_test':f.y_test,
                'test_prob':None }
        if step==int(epochs):
            testp=probabilities(f.x_test,theta)
            rows=score_table([score_row('Train',f.y_train,tp),score_row('Validation',f.y_valid,vp),
                               score_row('Test (финальный)',f.y_test,testp)])
            state['test_prob']=testp
            ev=evaluate(f.y_test,testp,float(threshold))
            status=(f'**Обучение завершено:** эпох {step}, признаков {len(features)}, '
                    f'learning rate={lr}, L2={l2}.  '
                    f'**Test:** ROC-AUC={ev["roc_auc"]:.3f}, PR-AUC={ev["pr_auc"]:.3f}, '
                    f'Precision={ev["precision"]:.2%}, Recall={ev["recall"]:.2%} '
                    f'при пороге {threshold:.2f}.')
            yield (status,loss_chart(history),weights_chart(history,features),rows,
                   roc_figure(f.y_test,testp),confusion_figure(ev['cm']),state)
        else:
            status=(f'**Эпоха {step} / {epochs}.**  Log Loss train: '
                    f'{history.train_loss.iloc[-1]:.4f}; validation: {history.val_loss.iloc[-1]:.4f}. '
                    'Test пока не показан.')
            yield (status,loss_chart(history),weights_chart(history,features),rows,
                   gr.skip(),gr.skip(),state)


def change_threshold(state,threshold: float):
    if not state or state.get('test_prob') is None:
        return 'Сначала завершите обучение, затем изменяйте порог.',empty_plot()
    ev=evaluate(state['y_test'],state['test_prob'],float(threshold))
    msg=(f'**Test, порог={threshold:.2f}:** Precision={ev["precision"]:.2%}, '
         f'Recall={ev["recall"]:.2%}, F1={ev["f1"]:.3f}. '
         f'ROC-AUC={ev["roc_auc"]:.3f} не меняется при изменении порога.')
    return msg,confusion_figure(ev['cm'])


@lru_cache(maxsize=100)
def two_features(a: str, b: str):
    if a==b:
        raise ValueError('Нужно два разных признака')
    return prepare_features(get_splits(),[a,b])


def theta_contour(f,theta):
    x=f.x_valid
    low=np.percentile(x,2,axis=0) - 0.4
    high=np.percentile(x,98,axis=0) + 0.4
    # Safeguard for extremely constant standardized predictors
    lo=np.minimum(low,high-1)
    x0=np.linspace(lo[0],high[0],65)
    x1=np.linspace(lo[1],high[1],65)
    xx,yy=np.meshgrid(x0,x1)
    grid=np.column_stack([np.ones(xx.size),xx.ravel(),yy.ravel()])
    pp=expit(grid@theta).reshape(xx.shape)
    fig=go.Figure()
    fig.add_trace(go.Contour(x=x0,y=x1,z=pp,colorscale='RdBu_r',zmin=0,zmax=1,
                             colorbar=dict(title='P(DPD30+)'),opacity=.72,
                             contours=dict(start=0,end=1,size=.1)))
    rng=np.random.default_rng(25)
    picks=rng.choice(len(x),min(650,len(x)),replace=False)
    for y_value,label in ((0,'Good'),(1,'Bad')):
        group=picks[f.y_valid[picks]==y_value]
        fig.add_trace(go.Scatter(x=x[group,0],y=x[group,1],mode='markers',name=label,
                                 marker=dict(size=5,opacity=.45,symbol='circle' if y_value==0 else 'x')))
    fig.update_layout(title='Карта предсказанной вероятности и фактические наблюдения',
                      xaxis_title=f'{f.names[0]} (z-score)',yaxis_title=f'{f.names[1]} (z-score)',
                      template='plotly_white',height=470,margin=dict(l=45,r=20,t=65,b=45))
    return fig


def theta_inspect(a,b,t0,t1,t2,lr,l2):
    if a==b:
        return 'Выберите два различных признака.',empty_plot()
    f=two_features(a,b)
    theta=np.array([t0,t1,t2],dtype=float)
    xtr=with_intercept(f.x_train)
    xv=with_intercept(f.x_valid)
    tr_loss,grad=objective_and_gradient(xtr,f.y_train,theta,float(l2))
    val_loss,_=objective_and_gradient(xv,f.y_valid,theta,float(l2))
    next_theta=theta-float(lr)*grad
    md=(f'**Train Log Loss:** {tr_loss:.4f} · **Validation:** {val_loss:.4f}\n\n'
        f'Градиент: **∂L/∂θ** = [{grad[0]:+.4f}, {grad[1]:+.4f}, {grad[2]:+.4f}]\n\n'
        f'По формуле `θ_new = θ − α × ∇L`, при α={lr} следующий шаг даст: '
        f'[{next_theta[0]:+.3f}, {next_theta[1]:+.3f}, {next_theta[2]:+.3f}]. '
        'Коэффициенты указаны для **стандартизованных признаков**, не в исходных сомах/днях.')
    return md,theta_contour(f,theta)


def theta_advance(a,b,t0,t1,t2,lr,l2,nsteps):
    if a==b:
        raise gr.Error('Выберите разные признаки')
    f=two_features(a,b)
    X=with_intercept(f.x_train)
    theta=np.array([t0,t1,t2],dtype=float)
    for _ in range(int(nsteps)):
        theta,_,_=gd_step(X,f.y_train,theta,float(lr),float(l2))
        theta=np.clip(theta,-15,15) # slider range guard for runaway experiments
    return tuple(float(v) for v in theta)


def theta_reset():
    return 0.0,0.0,0.0


def compare_model(name,features,trees,depth,boost_lr):
    if len(features)<2:
        raise gr.Error('Выберите минимум 2 признака')
    f=prepare_features(get_splits(),features)
    if name=='Logistic Regression (sklearn)':
        mdl=LogisticRegression(max_iter=1500,random_state=42)
    elif name=='Random Forest':
        mdl=RandomForestClassifier(n_estimators=int(trees),max_depth=int(depth),
                                   min_samples_leaf=15,random_state=42,n_jobs=-1)
    elif name=='CatBoost':
        mdl=CatBoostClassifier(iterations=int(trees),depth=min(int(depth),10),
                               learning_rate=float(boost_lr),verbose=False,
                               random_seed=42,thread_count=4,loss_function='Logloss')
    else:
        raise gr.Error('Неизвестный алгоритм')
    mdl.fit(f.x_train,f.y_train.astype(int))
    p=mdl.predict_proba(f.x_valid)[:,1]
    m=evaluate(f.y_valid,p)
    message=(f'**{name} — Validation (2023):** '
             f'ROC-AUC={m["roc_auc"]:.3f}, PR-AUC={m["pr_auc"]:.3f}, '
             f'Brier={m["brier"]:.3f}, n={m["n"]:,}. '
             'Параметры сравнивайте по validation. Test — один финальный контроль.')
    if hasattr(mdl,'feature_importances_'):
        importances=np.asarray(mdl.feature_importances_)
    else:
        importances=np.abs(mdl.coef_[0])
    imp=pd.DataFrame({'Feature':features,'Importance':importances}).sort_values('Importance',ascending=False)
    fig=px.bar(imp.head(15).sort_values('Importance'),x='Importance',y='Feature',orientation='h',
               title='Важность признаков / абсолютные коэффициенты',template='plotly_white')
    fig.update_layout(height=420,margin=dict(l=45,r=20,t=60,b=45))
    state={'model':mdl,'features':f, 'algorithm':name}
    return (message,score_table([score_row('Validation',f.y_valid,p)]),fig,
            roc_figure(f.y_valid,p,'Validation'),state)


def final_compare_test(state):
    if not state:
        return 'Сначала обучите модель.',empty_plot()
    f=state['features']; model=state['model']
    p=model.predict_proba(f.x_test)[:,1]
    m=evaluate(f.y_test,p)
    return (f'**FINAL TEST ({state["algorithm"]}, 2025):** ROC-AUC={m["roc_auc"]:.3f}, '
            f'PR-AUC={m["pr_auc"]:.3f}, Brier={m["brier"]:.3f}, '
            f'событий={m["event_rate"]:.1%}, кредитов={m["n"]:,}. '
            'На тест нельзя повторно настраивать гиперпараметры.'),roc_figure(f.y_test,p,'Final Test')



def money(value):
    v=float(value)
    return f'{v/1e9:,.2f} млрд сом' if abs(v)>=1e9 else (f'{v/1e6:,.2f} млн сом' if abs(v)>=1e6 else f'{v:,.0f} сом')


def risk_horizon_plot(horizon):
    fig=go.Figure()
    fig.add_trace(go.Bar(x=horizon.HorizonDays.astype(str)+' дней',
                         y=horizon.ExpectedNewDPD30,name='Новые DPD30+ — ожидаемое число'))
    fig.update_layout(template='plotly_white',title='Сценарное число новых событий DPD30+ за 30 / 60 / 90 дней',
                      xaxis_title='Горизонт',yaxis_title='Ожидаемые события (кредиты)',height=390,
                      margin=dict(l=45,r=20,t=65,b=55))
    return fig


def portfolio_pr_plot(horizon):
    fig=go.Figure()
    x=horizon.HorizonDays.astype(str)+' дней'
    fig.add_trace(go.Bar(x=x,y=horizon.ExistingPRKGS/1e6,name='Портфель без кампании'))
    fig.add_trace(go.Bar(x=x,y=horizon.IncrementalPRKGS/1e6,name='Дополнительный доход кампании'))
    fig.update_layout(template='plotly_white',barmode='stack',
                      title='PR — процентный доход портфеля, сценарий',
                      xaxis_title='Горизонт',yaxis_title='Млн сом',height=390,
                      margin=dict(l=45,r=20,t=65,b=55))
    return fig


def portfolio_risk_plot(family):
    x=family.sort_values('EstimatedBalanceKGS',ascending=True)
    fig=go.Figure()
    fig.add_trace(go.Bar(x=x.EstimatedBalanceKGS/1e6,
                         y=x.ProductFamily,orientation='h',name='Оценка остатка'))
    fig.update_layout(template='plotly_white',title='Приближённая экспозиция по продуктам',
                      xaxis_title='Млн сом',height=390,
                      margin=dict(l=65,r=20,t=65,b=55))
    return fig


def render_risk_dashboard(settings):
    settings=settings or DecisionSettings()
    k,h,f,active=scenario_dashboard(settings)
    row90=h.loc[h.HorizonDays.eq(90)].iloc[0]
    overview=(
        f'### Портфель на {k["as_of"]} · сценарий 30/60/90 дней\n'
        f'**Действующих кредитов:** {k["active_loans"]:,} · '
        f'**Оценка остатка:** {money(k["estimated_balance"])} · '
        f'**Уже DPD30+:** {k["current_dpd30"]:,} · '
        f'**С текущей просрочкой 1+:** {k["currently_overdue"]:,}\n\n'
        f'**Прогноз новых событий DPD30+ за 90 дней:** {row90.ExpectedNewDPD30:,.0f} · '
        f'**Стресс-прокси потерь за 90 дней:** {money(row90.ExpectedLossKGS)} · '
        f'**PR без кампании (90 дней):** {money(row90.ExistingPRKGS)} · '
        f'**Прирост PR от кампании:** {row90.GrowthPRPct:+.2f}% '
        f'({money(row90.IncrementalPRKGS)})\n\n'
        f'В сценарии использовано **{k["candidates"]}** адресных предложений; '
        f'доля оформлений — **{settings.conversion:.0%}** '
        f'(ожидается **{k["expected_accepted"]:,.1f}** выдач). '
        f'ROC-AUC OOT для EWS30: **{k["test_auc"]:.3f}**. '
    )
    table=h.rename(columns={
        'HorizonDays':'Горизонт, дней','ExpectedNewDPD30':'Новых DPD30+ (ожид.)',
        'ExpectedLossKGS':'Стресс-прокси потерь, сом','ExistingPRKGS':'PR текущий, сом',
        'IncrementalPRKGS':'Прирост PR, сом','ScenarioPRKGS':'PR с кампанией, сом',
        'GrowthPRPct':'Рост PR, %'})
    for col in table.select_dtypes(include=['float','int']).columns:
        if col!='Горизонт, дней':table[col]=table[col].round(2)
    byfamily=f.rename(columns={'ProductFamily':'Продукт','ActiveLoans':'Активные кредиты',
        'EstimatedBalanceKGS':'Остаток (оценка), сом','CurrentDPD30':'DPD30+ сейчас',
        'MeanPD12':'Средняя PD12 (сценарий)'})
    byfamily['Средняя PD12 (сценарий)']=(100*byfamily['Средняя PD12 (сценарий)']).round(1)
    byfamily['Остаток (оценка), сом']=byfamily['Остаток (оценка), сом'].round(0)
    top_risk=active.sort_values(['CurrentDPD','EW_PD30_30D'],ascending=False).head(30)[
        ['CustomerID','CreditID','ProductName','CurrentDPD',
         'EstimatedOutstandingKGS','EW_PD30_30D','EW_PD30_60D','EW_PD30_90D']].copy()
    top_risk['EstimatedOutstandingKGS']=top_risk.EstimatedOutstandingKGS.round(0)
    for col in ['EW_PD30_30D','EW_PD30_60D','EW_PD30_90D']:
        top_risk[col]=(top_risk[col]*100).round(1)
    top_risk=top_risk.rename(columns={'ProductName':'Продукт','CurrentDPD':'Текущий DPD',
        'EstimatedOutstandingKGS':'Остаток (сценарий), сом',
        'EW_PD30_30D':'P(DPD30+) 30д, %','EW_PD30_60D':'P(DPD30+) 60д, %',
        'EW_PD30_90D':'P(DPD30+) 90д, %'})
    caveat=(
        '**Как читать:** Early Warning — отдельная ML-модель на исторических снимках; '
        'на горизонтах 30/60/90 дней оценивает возникновение DPD30+ у ещё не просроченных 30+ кредитов. '
        'Оценки потерь = вероятность события × приблизительный остаток × сценарный LGD: '
        'это НЕ IFRS 9 ECL. PR — процентный доход, не чистая прибыль. '
        'Остатки — амортизационная оценка, а не бухгалтерский баланс. '
        'Аналитическая дата среза: 31.10.2026. Прогноз является демонстрацией на учебных данных.'
    )
    return (overview,risk_horizon_plot(h),portfolio_pr_plot(h),
            portfolio_risk_plot(f),table,byfamily,top_risk,caveat)


def decision_ui(max_pd,max_dti,min_income,max_hist,include_new,top_n,conv,lgd,
                funding,discount,contact,balanced,loyalty_discount):
    settings=DecisionSettings(max_pd=float(max_pd),max_dti=float(max_dti),
        min_income=float(min_income),max_hist_dpd=int(max_hist),
        include_no_history=bool(include_new),top_n=int(top_n),
        conversion=float(conv),lgd=float(lgd),funding_rate=float(funding),
        merchant_discount_pct=float(discount),contact_cost=float(contact),
        diversify_products=bool(balanced),loyalty_discount_pp=float(loyalty_discount))
    r=decision_table(settings)
    filename=export_decisions(settings)
    if len(r)==0:
        return settings,('Нет клиентов, прошедших заданные правила. Ослабьте один из фильтров.'),
        pd.DataFrame(),empty_plot('Нет рекомендаций'),filename
    expected_accepted=len(r)*settings.conversion
    expected_pr=r.ProjectedPR12M_KGS.sum()*settings.conversion
    expected_net=r.ExpectedValuePerContactKGS.sum()
    overview=(f'**Отобрано:** {len(r):,} из 12 392 · '
              f'**Объём потенциальных предложений:** {money(r.OfferAmountKGS.sum())} · '
              f'**Ожидаемые выдачи:** {expected_accepted:,.1f} '
              f'(конверсия {settings.conversion:.0%})\n\n'
              f'**Ожидаемый PR по принятым оферам за первый год / срок кредита:** '
              f'{money(expected_pr)} · '
              f'**Ожидаемый эффект кампании с учётом конверсии и затрат на контакт:** {money(expected_net)}. '
              f'Это сценарные расчёты, а не прогноз конверсии.')
    table=r[['CustomerID','RecommendedProduct','OfferAmountKGS','OfferRatePct',
        'OfferTermMonths','PD12','ProposalDTI','IncomeMean6M_KGS','Segment',
        'ProjectedPR12M_KGS','ExpectedValuePerContactKGS','RecommendationReason']].copy()
    table['PD12']=(table['PD12']*100).round(1)
    table['ProposalDTI']=(table['ProposalDTI']*100).round(1)
    for col in ['OfferAmountKGS','IncomeMean6M_KGS','ProjectedPR12M_KGS','ExpectedValuePerContactKGS']:
        table[col]=table[col].round(0)
    table.columns=['CustomerID','Рекомендованный продукт','Сумма, сом','Ставка, %',
        'Срок, мес.','PD12, %','ПДН, %','Доход 6М, сом','Сегмент',
        'PR при выдаче, сом','EV одного контакта, сом','Причина отбора']
    mix=r.groupby('RecommendedProduct').agg(
        n=('CustomerID','size'),sum_amount=('OfferAmountKGS','sum')).reset_index()
    fig=px.bar(mix,x='RecommendedProduct',y='n',text='n',template='plotly_white',
               title='Структура рекомендаций по продуктам')
    fig.update_layout(height=375,xaxis_title='',yaxis_title='Клиенты',
                      margin=dict(l=35,r=20,t=60,b=90),xaxis_tickangle=-15)
    return settings,overview,table,fig,filename



def governance_ui():
    from src.decision_engine_v6 import risk_model_and_validation
    ews,valid,calib,ablation=all_ews()
    rolls,vintage,drift=monitoring_reports()
    _,risk_metrics,rv=risk_model_and_validation()
    ri=(f'**Risk Scoring / OOT:** ROC-AUC={risk_metrics["roc_auc"]:.3f}, '
        f'PR-AUC={risk_metrics["pr_auc"]:.3f}, Brier={risk_metrics["brier"]:.3f}; '
        f'**EWS / OOT:** 30/60/90 дней; показаны результаты независимой проверки и ablation. '
        'PSI оценивает изменение распределений признаков, а не точность модели.')
    cal=calib[calib.horizon_days.eq(30)].copy()
    reliability=go.Figure()
    reliability.add_trace(go.Scatter(x=cal.predicted,y=cal.observed,mode='lines+markers',name='Наблюдаемая частота'))
    reliability.add_trace(go.Scatter(x=[0,max(.01,cal.predicted.max())],
                   y=[0,max(.01,cal.predicted.max())],name='Идеальная калибровка',line=dict(dash='dash')))
    reliability.update_layout(title='Калибровка Early Warning 30 дней (OOT)',
        template='plotly_white',xaxis_title='Средний прогноз',yaxis_title='Фактическая частота')
    r=rolls.pivot(index='Bucket',columns='NextBucket',values='RollRate').fillna(0)
    rollfig=px.imshow(r,text_auto='.0%',aspect='auto',title='Миграции DPD между месяцами',color_continuous_scale='Blues')
    latest=vintage[vintage.N>=20]
    vinfig=px.line(latest,x='Vintage',y='BadRate12M',color='ProductFamily',
                  title='Доля DPD30+ по кварталам выдач (vintage)',markers=True)
    return ri,valid.round(4),ablation.round(4),reliability,drift.round(3),rollfig,vinfig,rv.round(4)


def explain_ui(customer_id,settings):
    x=explain_customer_offers(customer_id,settings or DecisionSettings())
    if x.empty:return 'CustomerID не найден среди рассматриваемых кандидатов.',pd.DataFrame()
    x=x.copy()
    for name in ['PD12','InternalDTI']:
        x[name]=(x[name]*100).round(1)
    return ('Четыре варианта офера сравниваются по риску, внутренней ПДН и '
            'ожидаемому эффекту на контакт. Это предофер, не согласование выдачи.'),x.round(2)

CUSTOM_CSS="""
.gradio-container {max-width: 1420px !important;}
footer {display:none !important;}
"""

with gr.Blocks(title='Credit Intelligence Lab v6',analytics_enabled=False) as demo:
    gr.Markdown('''# Credit Intelligence Lab — интерактивный Data Science проект
**Данные · EDA · Обучение ML · Risk Dashboard · Decision Engine.** Управляй θ, learning rate, параметрами риск-сценария и портфелем предложений. Расчёты выполняются локально.
''')
    decision_settings=gr.State(DecisionSettings())
    with gr.Tabs():
        with gr.Tab('1 · Данные'):
            gr.Markdown('### Исходные CSV и обучающая выборка\nФайлы открываются локально; большие таблицы отображаются порциями.')
            with gr.Row():
                file_sel=gr.Dropdown(choices=list(DATASETS),value=list(DATASETS)[0],label='Таблица',scale=3)
                limit=gr.Slider(50,1000,value=100,step=50,label='Количество строк',scale=1)
                load_button=gr.Button('Показать данные',variant='primary',scale=1)
            file_info=gr.Markdown()
            preview=gr.Dataframe(label='Первые строки CSV',interactive=False,wrap=True)
            load_button.click(preview_dataset,[file_sel,limit],[file_info,preview])
            file_sel.change(preview_dataset,[file_sel,limit],[file_info,preview])
            demo.load(lambda:preview_dataset(list(DATASETS)[0],100),outputs=[file_info,preview])

        with gr.Tab('2 · EDA'):
            gr.Markdown('### Исследование будущих просрочек\nДля EDA используются только кредиты с известным Target. Город и область — для исследования, не для назначения ставки.')
            with gr.Row():
                family=gr.Dropdown(['Все продукты']+sorted(read_model().ProductFamily.dropna().unique().tolist()),
                                   value='Все продукты',label='Семейство кредита')
                group=gr.Dropdown(['ProductFamily','ProductCode','City','Region','Occupation','NoCreditHistory'],
                                  value='ProductFamily',label='Группировка')
                num=gr.Dropdown(SAFE_NUMERIC_FEATURES,value='DebtToIncomeRatio',label='Числовой признак')
                min_n=gr.Slider(20,300,value=60,step=20,label='Минимум кредитов в группе')
            eda_btn=gr.Button('Построить EDA',variant='primary')
            eda_stats=gr.Markdown()
            eda_bar=gr.Plot()
            with gr.Row():
                eda_hist=gr.Plot()
                eda_time=gr.Plot()
            eda_corr=gr.Plot()
            eda_btn.click(eda_dashboard,[family,group,num,min_n],[eda_stats,eda_bar,eda_hist,eda_time,eda_corr])
            demo.load(lambda:eda_dashboard('Все продукты','ProductFamily','DebtToIncomeRatio',60),
                      outputs=[eda_stats,eda_bar,eda_hist,eda_time,eda_corr])

        with gr.Tab('3 · Градиентный спуск'):
            gr.Markdown('''### Логистическая регрессия с нуля (NumPy)
`z = θ₀ + θ₁×x₁ + …`, `PD = sigmoid(z)`, `θ := θ − α·∇LogLoss`.
Все признаки стандартизируются по **train**. Можно наблюдать train/validation loss и движение каждого веса после эпох обучения.
''')
            features_train=gr.Dropdown(SAFE_NUMERIC_FEATURES,value=DEFAULT_FEATURES,
                                       multiselect=True,label='Признаки для модели (2–8)')
            with gr.Row():
                lr=gr.Dropdown([0.001,0.005,0.01,0.03,0.05,0.1,0.2,0.5,1.0],value=0.1,
                               label='Learning rate α')
                epochs=gr.Slider(20,800,step=20,value=200,label='Число эпох')
                l2=gr.Slider(0,1000,step=25,value=0,label='L2 штраф λ')
                train_size=gr.Dropdown(['2000','5000','Все данные'],value='5000',label='Train observations')
            with gr.Row():
                start=gr.Button('Обучить и показать динамику',variant='primary')
                threshold=gr.Slider(0.05,0.90,step=0.05,value=0.50,label='Порог классификации (только для Confusion Matrix)')
            training_status=gr.Markdown('Выберите признаки и нажмите «Обучить».')
            with gr.Row():
                loss_plot=gr.Plot(value=empty_plot('Train/Validation Log Loss'))
                weight_plot=gr.Plot(value=empty_plot('Траектории θ'))
            training_metrics=gr.Dataframe(label='Метрики по временным выборкам',interactive=False)
            with gr.Row():
                roc_plot=gr.Plot(value=empty_plot('Test ROC'))
                cm_plot=gr.Plot(value=empty_plot('Test Confusion Matrix'))
            train_state=gr.State(None)
            threshold_result=gr.Markdown()
            start.click(train_manual,[features_train,lr,epochs,l2,train_size,threshold],
                        [training_status,loss_plot,weight_plot,training_metrics,roc_plot,cm_plot,train_state])
            threshold.change(change_threshold,[train_state,threshold],[threshold_result,cm_plot])
            gr.Markdown('**Проверка:** если validation loss растёт, а train loss падает, это может быть переобучение или сдвиг распределения. Learning rate влияет на шаг обновления, L2 — на размер весов.')

        with gr.Tab('4 · θ-лаборатория'):
            gr.Markdown('''### Меняй θ вручную и делай шаги Gradient Descent
θ₀ — свободный коэффициент; θ₁ и θ₂ — веса **двух стандартизованных признаков**.
Установи любые значения, сравни Log Loss с validation и попробуй **1 или 10 шагов**: это покажет направление градиента.
''')
            with gr.Row():
                f1=gr.Dropdown(SAFE_NUMERIC_FEATURES,value='MaxHistoricalDPD',label='x₁')
                f2=gr.Dropdown(SAFE_NUMERIC_FEATURES,value='DebtToIncomeRatio',label='x₂')
            with gr.Row():
                t0=gr.Slider(-15,15,step=.05,value=0,label='θ₀ — Intercept')
                t1=gr.Slider(-15,15,step=.05,value=0,label='θ₁ — Вес x₁')
                t2=gr.Slider(-15,15,step=.05,value=0,label='θ₂ — Вес x₂')
            with gr.Row():
                lab_lr=gr.Dropdown([0.001,0.01,0.05,0.1,0.3,0.5,1.0,2.0],value=0.3,label='Learning rate α')
                lab_l2=gr.Slider(0,1000,step=50,value=0,label='L2 штраф λ')
                show_btn=gr.Button('Рассчитать θ и Log Loss')
                one_step=gr.Button('1 шаг GD',variant='primary')
                ten_steps=gr.Button('10 шагов GD')
                reset_btn=gr.Button('Сброс θ')
            lab_stats=gr.Markdown()
            lab_plot=gr.Plot(value=empty_plot('θ-Lab карта вероятностей'))
            inp=[f1,f2,t0,t1,t2,lab_lr,lab_l2]
            show_btn.click(theta_inspect,inp,[lab_stats,lab_plot])
            for control in [f1,f2,t0,t1,t2,lab_lr,lab_l2]:
                control.change(theta_inspect,inp,[lab_stats,lab_plot])
            one_step.click(lambda *x:theta_advance(*x,nsteps=1),inp,[t0,t1,t2])
            ten_steps.click(lambda *x:theta_advance(*x,nsteps=10),inp,[t0,t1,t2])
            reset_btn.click(theta_reset,outputs=[t0,t1,t2])
            demo.load(lambda:theta_inspect('MaxHistoricalDPD','DebtToIncomeRatio',0,0,0,0.3,0),
                      outputs=[lab_stats,lab_plot])

        with gr.Tab('5 · Сравнение ML'):
            gr.Markdown('''### Logistic Regression / Random Forest / CatBoost
Параметры деревьев и learning rate CatBoost управляются отдельно от θ логистической регрессии.
**Validation** используется для сравнения гиперпараметров. Нажимай финальный Test только после выбора конфигурации.
''')
            comp_feats=gr.Dropdown(SAFE_NUMERIC_FEATURES,value=DEFAULT_FEATURES,
                                   multiselect=True,label='Признаки')
            with gr.Row():
                model_name=gr.Dropdown(['Logistic Regression (sklearn)','Random Forest','CatBoost'],
                                       value='Logistic Regression (sklearn)',label='Алгоритм')
                trees=gr.Slider(50,300,step=25,value=150,label='Деревья / итерации')
                depth=gr.Slider(2,10,step=1,value=5,label='Глубина')
                boost_lr=gr.Dropdown([0.01,0.03,0.05,0.1,0.2,0.3],value=0.05,
                                     label='Learning rate CatBoost')
            comp_btn=gr.Button('Обучить выбранную модель',variant='primary')
            comp_info=gr.Markdown()
            comp_table=gr.Dataframe(interactive=False)
            with gr.Row():
                comp_importance=gr.Plot(value=empty_plot('Важность признаков'))
                comp_roc=gr.Plot(value=empty_plot('Validation ROC'))
            compare_state=gr.State(None)
            comp_btn.click(compare_model,[model_name,comp_feats,trees,depth,boost_lr],
                           [comp_info,comp_table,comp_importance,comp_roc,compare_state])
            test_btn=gr.Button('Один финальный запуск на Test')
            final_info=gr.Markdown()
            final_roc=gr.Plot(value=empty_plot('Final Test'))
            test_btn.click(final_compare_test,[compare_state],[final_info,final_roc])

        with gr.Tab('6 · Risk Dashboard'):
            gr.Markdown('''### Риски портфеля и ожидаемый рост PR
**PR = процентный доход.** Отдельные модели Early Warning на 30/60/90 дней; качество — на OOT выборке. Экспозиции и потери — приближённый сценарий от даты снимка.
Параметры кампании изменяются во вкладке **Decision Engine** и затем применяются здесь.
''')
            risk_btn=gr.Button('Обновить дашборд по выбранному сценарию',variant='primary')
            risk_overview=gr.Markdown()
            with gr.Row():
                risk_plot=gr.Plot(value=empty_plot('Риски на 30 / 60 / 90 дней'))
                pr_plot=gr.Plot(value=empty_plot('Рост PR'))
            family_plot=gr.Plot(value=empty_plot('Портфель по продуктам'))
            gr.Markdown('#### Показатели по горизонтам')
            horizon_table=gr.Dataframe(interactive=False)
            gr.Markdown('#### Риск по семействам кредитов')
            family_table=gr.Dataframe(interactive=False)
            gr.Markdown('#### Кредиты для дополнительной проверки (текущие DPD и риск)')
            highrisk_table=gr.Dataframe(interactive=False)
            risk_caveat=gr.Markdown()
            risk_outputs=[risk_overview,risk_plot,pr_plot,family_plot,
                          horizon_table,family_table,highrisk_table,risk_caveat]
            risk_btn.click(render_risk_dashboard,[decision_settings],risk_outputs)
            demo.load(lambda:render_risk_dashboard(DecisionSettings()),outputs=risk_outputs)

        with gr.Tab('7 · Decision Engine'):
            gr.Markdown('''### Кому предложить кредит и на каких условиях
Система формирует **предварительные рекомендации**, не одобрения. Для заявок без
активного кредита использованы история погашений и финансовое поведение. Без
подтверждённой цели и обеспечения ипотека, авто и залоговые кредиты не предлагаются.
**PD12** — оценка риска достижения DPD30+ в 12 месяцев после предполагаемой выдачи.
''')
            with gr.Row():
                maxpd=gr.Slider(0.05,0.50,value=0.18,step=0.01,label='Максимальная PD12')
                maxdti=gr.Slider(0.20,0.60,value=0.40,step=0.05,label='Максимальная ПДН')
                minincome=gr.Slider(15000,120000,value=30000,step=5000,label='Минимальный доход, сом/мес')
                maxhist=gr.Dropdown([0,7,14,29,59],value=29,label='Макс. прошлый DPD, дней')
            with gr.Row():
                incl_new=gr.Checkbox(value=False,label='Допускать клиентов без кредитной истории')
                limitrec=gr.Slider(50,1500,value=300,step=50,label='Максимум предложений')
                balanced=gr.Checkbox(value=True,label='Разнообразная продуктовая подборка')
            gr.Markdown('#### Экономика сценария')
            with gr.Row():
                conv=gr.Slider(0.01,0.30,value=0.08,step=0.01,label='Допущение: конверсия в выдачу')
                lgd=gr.Slider(0.1,0.9,value=0.55,step=0.05,label='Допущение: LGD')
                funding=gr.Slider(0,25,value=10,step=0.5,label='Стоимость фондирования, % годовых')
                disc=gr.Slider(0,10,value=3,step=0.5,label='Мерчант-дисконт рассрочки, %')
                contact=gr.Slider(0,200,value=20,step=10,label='Стоимость контакта, сом')
                loyalty=gr.Slider(0,3,value=0,step=0.5,label='Скидка за дисциплину, п.п. (сценарий)')
            dec_btn=gr.Button('Рассчитать рекомендации и экономику',variant='primary')
            dec_overview=gr.Markdown()
            dec_plot=gr.Plot(value=empty_plot('Состав аудитории'))
            dec_table=gr.Dataframe(label='CustomerID и рекомендуемый продукт',interactive=False,
                                   wrap=True)
            rec_file=gr.File(label='Экспорт списка кандидатов в CSV',interactive=False)
            args=[maxpd,maxdti,minincome,maxhist,incl_new,limitrec,conv,lgd,
                  funding,disc,contact,balanced,loyalty]
            outs=[decision_settings,dec_overview,dec_table,dec_plot,rec_file]
            dec_btn.click(decision_ui,args,outs).then(render_risk_dashboard,
                                    inputs=[decision_settings],outputs=risk_outputs)
            demo.load(lambda:decision_ui(.18,.40,30000,29,False,300,.08,.55,10,3,20,True,0),
                      outputs=outs)
            gr.Markdown('''**Правила:** нет активного кредита; ограничения по прошлому DPD,
            доходу и PD; прогнозная ПДН не выше порога. Приоритет по условному
            доходу после ожидаемых потерь. *Включить без кредитной истории* — отдельный
            сценарий, требующий собственной валидации. Для рассрочки учитывается
            предполагаемый мерчант-дисконт; без него экономика может быть отрицательной.
            **Не учтены:** потребность в продукте, подтверждённое согласие на маркетинг и внешние долги.
            ПДН — только внутренние кредиты. Скидка не увеличивает конверсию автоматически: без доказанного эффекта
            снижение ставки уменьшает доход. Выгрузка — для ручной проверки, не для одобрения.''')

        with gr.Tab('8 · Model Monitoring'):
            gr.Markdown('''### Качество, калибровка, стабильность и миграции портфеля
OOT — отложенные по времени даты. Для Early Warning используется собственный
Target новых событий DPD30+ за 30/60/90 дней. Текущий DPD30+ не считается новым событием.''')
            mon_btn=gr.Button('Пересчитать отчёты',variant='primary')
            mon_info=gr.Markdown()
            ew_metrics=gr.Dataframe(label='Early Warning OOT',interactive=False)
            ew_ablation=gr.Dataframe(label='Ablation: влияние финансовых данных',interactive=False)
            with gr.Row():
                ew_cal=gr.Plot(label='Калибровка 30 дней')
                ew_roll=gr.Plot(label='Roll rates')
            ew_psi=gr.Dataframe(label='PSI — мониторинг сдвига данных',interactive=False)
            ew_vintage=gr.Plot(label='Vintage Analysis')
            risk_cal=gr.Dataframe(label='Калибровка PD12, OOT',interactive=False)
            mon_outputs=[mon_info,ew_metrics,ew_ablation,ew_cal,ew_psi,ew_roll,ew_vintage,risk_cal]
            mon_btn.click(governance_ui,outputs=mon_outputs)
            demo.load(governance_ui,outputs=mon_outputs)

        with gr.Tab('9 · Анализ отдельного офера'):
            gr.Markdown('''### Почему именно этот продукт?
Введите CustomerID из списка кандидатов. Система покажет четыре альтернативных
предложения и ограничения по ним. Внешние обязательства не проверяются,
поэтому «ПДН» здесь обозначает исключительно внутреннюю расчётную нагрузку.''')
            client_filter=gr.Textbox(label='CustomerID',value='KG_SYN2_0000029')
            inspect_btn=gr.Button('Сравнить четыре офера',variant='primary')
            customer_info=gr.Markdown()
            customer_offers=gr.Dataframe(interactive=False)
            inspect_btn.click(explain_ui,[client_filter,decision_settings],[customer_info,customer_offers])
    gr.Markdown('''---
**О данных и ограничениях:** учебный демонстрационный набор, а не реальные записи клиентов. Данные о реальном риске городов отсутствуют. Модели и сценарные расчёты не используются для кредитных решений или отправки рекламы. Новые клиенты без кредитной истории требуют отдельной валидации.
''')


if __name__=='__main__':
    demo.queue(default_concurrency_limit=2).launch(server_name='127.0.0.1',server_port=7860,
                                                    inbrowser=True,show_error=True,
                                                    css=CUSTOM_CSS,theme=gr.themes.Soft())
