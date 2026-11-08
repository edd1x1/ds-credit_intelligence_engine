from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def md(source):return {'cell_type':'markdown','metadata':{},'source':source.splitlines(keepends=True)}
def code(source):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':source.splitlines(keepends=True)}
def write(name,cells):
 nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
 (ROOT/'notebooks'/name).write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')

write('01_eda_start.ipynb',[
md('# 01. Exploratory Data Analysis\n\n**Учебный проект.** Все клиенты и риски вымышлены. Главная задача: исследовать качество, структуру риска и изменения во времени, не смешивая будущие события с признаками.\n\nЗапускай Notebook из корня проекта.'),
code('''from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

model = pd.read_csv(ROOT / "data/processed/model_dataset.csv", parse_dates=["SnapshotDate", "LabelReadyDate"])
if (ROOT / "data/raw/loans.csv").exists():
    loans = pd.read_csv(ROOT / "data/raw/loans.csv", parse_dates=["CreditOpenDate"])
    clients = pd.read_csv(ROOT / "data/raw/customers.csv")
    repayments = pd.read_csv(ROOT / "data/raw/repayment_schedule.csv", parse_dates=["DueDate", "SettledDate"])
else:
    print('Облегчённый набор: EDA по model_dataset. Для анализа событий платежей нужен полный архив.')
    loans = model[['CreditID', 'CustomerID', 'AgreementSumN']].copy()
    clients = model[['CustomerID']].drop_duplicates().copy()
    repayments = pd.DataFrame()
products = pd.read_csv(ROOT / "data/reference/product_catalog.csv")
print('Rows:', {"model": len(model), "loans": len(loans), "clients": len(clients), "payments": len(repayments)})
display(model.head())'''),
md('## 1. Data Quality\n\nПроверь уникальность ключей, типы, пропуски, датированность `SnapshotDate`, диапазоны сумм и связь таблиц. **Вопрос:** что означают `NaN` в Target и `LatePaymentRatio`?'),
code('''display(model.isna().mean().sort_values(ascending=False).head(15))
print('Duplicate CreditID:', model.CreditID.duplicated().sum())
print('Unlabeled loans:', model.TargetDPD30_12M.isna().sum())
print('Loan amount min/max:', loans.AgreementSumN.min(), loans.AgreementSumN.max())
# TODO: Проверь, что Rate и сроки соответствуют справочнику product_catalog.'''),
md('## 2. Таргет и временные когорты\n\nПроверяем не только общий Bad Rate, но и *vintage* по кварталам выдачи. Пропущенные Target не превращаем в Good.'),
code('''labeled = model[model.IsMatureTarget.eq(1)].copy()
labeled['OriginationQuarter'] = labeled.SnapshotDate.dt.to_period('Q').astype(str)
vintage = labeled.groupby('OriginationQuarter').agg(
    loans=('CreditID', 'size'), bad_rate=('TargetDPD30_12M', 'mean')
)
display(vintage.tail(12))
vintage.bad_rate.plot(figsize=(12, 4), marker='o', title='Доля DPD30+ по кварталам выдачи (синтетика)')
plt.ylabel('Доля кредитов с Target=1')
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()
# TODO: Какие кварталы отличаются и почему? Учитывай число кредитов.'''),
md('## 3. Продукты\n\n**Гипотеза:** вероятность наблюдать DPD30+ в течение 12 месяцев может зависеть не только от риска клиента, но и от числа платежей и особенностей подпродукта.'),
code('''product_risk = labeled.groupby(['ProductFamily', 'ProductCode']).agg(
    n=('CreditID', 'size'), rate=('TargetDPD30_12M', 'mean'),
    avg_amount=('AgreementSumN', 'mean'), avg_term=('LoanTermMonths', 'mean')
).sort_values('rate', ascending=False)
display(product_risk)
# TODO: Построй график n и rate. Не делай выводы по группам с маленькой численностью.'''),
md('## 4. География — только анализ гипотетического сценария\n\nПовышенные значения для двух населённых пунктов введены в генератор. Они не отражают реальную платёжную дисциплину населения. Анализируй не только сырой Bad Rate, но и распределение продуктов, DTI, дохода. Город не является причинным объяснением просрочки.'),
code('''geo = labeled.groupby('City').agg(
    n=('CreditID', 'size'), bad_rate=('TargetDPD30_12M', 'mean'),
    mean_dti=('DebtToIncomeRatio', 'mean'), median_income=('IncomeMean6M_KGS', 'median')
).query('n >= 100').sort_values('bad_rate', ascending=False)
display(geo.head(12))
# TODO: Сравни две группы городов внутри Cash и Installment при одинаковых интервалах DTI.'''),
md('## 5. Биннинг и разделяющая способность признаков\n\nОтдели `NoCreditHistory=1` от `MaxHistoricalDPD=0`: это принципиально разные ситуации.'),
code('''labeled['dti_bin'] = pd.cut(labeled.DebtToIncomeRatio,
                            bins=[0, .2, .35, .5, .7, 1.0, np.inf], include_lowest=True)
dti_risk = labeled.groupby('dti_bin', observed=True).agg(
    n=('CreditID', 'size'), rate=('TargetDPD30_12M', 'mean')
)
display(dti_risk)
# TODO: Повтори анализ для IncomeCV6M, DPD30CountLast12M и CashflowMargin6M.'''),
md('## 6. Data Leakage audit\n\nЗапрещено использовать `TargetDPD30_12M`, `LabelReadyDate`, `IsMatureTarget`, информацию об оплате после `SnapshotDate`, `CreditStatusAtCutoff` или итоги за будущий срок как X.\n\n**Задание:** составь таблицу «признак / момент возникновения / доступен ли на T / риск утечки». Сохрани её рядом с EDA отчётом.'),
md('## Итоговые вопросы\n\n1. Какие 5 признаков выглядят наиболее информативными для риска?\n2. Что меняется после разделения по продуктам?\n3. Почему нельзя автоматически считать всех клиентов без истории надёжными?\n4. Какие гипотезы о региональном стресс-сценарии можно проверить, не приписывая городам причинный риск?\n5. Какие улучшения нужны до обучения модели?')
])
write('02_modeling_start.ipynb',[
md('# 02. Risk model — учебный каркас\n\nЦель: прогноз `TargetDPD30_12M` на момент выдачи кредита. Не используем `City` и `Region` в базовой risk-модели; изучаем их только для мониторинга ошибок.\n\n**Задача для самостоятельной работы:** сначала Logistic Regression как baseline, затем сравнить с деревьями.'),
code('''from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path.cwd()
if not (ROOT / "data").exists(): ROOT = ROOT.parent
all_rows = pd.read_csv(ROOT / 'data/processed/model_dataset.csv')
df = all_rows[all_rows.IsMatureTarget.eq(1)].copy()
NUM = ['AgeAtIssue','IncomeMean6M_KGS','IncomeCV6M','CashflowMargin6M',
       'IncomeTrend3v6M','AppActiveDays6M','AgreementSumN','Rate','LoanTermMonths',
       'MonthlyPaymentKGS','DebtToIncomeRatio','PreviousLoansCount','ActivePreviousLoansCount',
       'PreviousPaymentsObserved','MaxHistoricalDPD','DPD30CountHistory',
       'DPD30CountLast12M','LatePaymentRatio','MonthsSinceLastDPD30',
       'TotalPriorLoanAmountKGS','NoCreditHistory']
CAT = ['ProductFamily','ProductCode','Occupation']
features = NUM + CAT
train = df[df.TimeSplit.eq('train')]
valid = df[df.TimeSplit.eq('validation')]
test  = df[df.TimeSplit.eq('test')]
print('Train / Valid / Test:', len(train), len(valid), len(test))'''),
md('## Baseline Logistic Regression\n\nКритично: imputer, scaler и encoder обучаются только на `train`, а не на всём датасете.'),
code('''preprocessor = ColumnTransformer([
    ('num', Pipeline([('fill', SimpleImputer(strategy='median')),
                      ('scale', StandardScaler())]), NUM),
    ('cat', Pipeline([('fill', SimpleImputer(strategy='most_frequent')),
                      ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CAT)
])
model = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=800, class_weight=None))
])
model.fit(train[features], train.TargetDPD30_12M)
valid_pd = model.predict_proba(valid[features])[:, 1]
print('Validation ROC-AUC:', roc_auc_score(valid.TargetDPD30_12M, valid_pd))
print('Validation PR-AUC:', average_precision_score(valid.TargetDPD30_12M, valid_pd))
print('Validation Brier:', brier_score_loss(valid.TargetDPD30_12M, valid_pd))
print('Validation prevalence:', valid.TargetDPD30_12M.mean())
# TODO: Почему accuracy здесь — недостаточная метрика?'''),
md('## Самостоятельное задание\n\n- Добавь `RandomForestClassifier` и CatBoost, оцени на validation.\n- Проверь calibration plot: можно ли считать предсказанную вероятность реальным PD?\n- Проанализируй out-of-time тест только **после** выбора модели, не используя его для подбора гиперпараметров.\n- Запусти ablation: только кредитные признаки; затем добавь финансовые 6M; сравни прирост.\n- Добавь анализ ошибок по продуктам и городам, но **не делай** выводов о реальном риске жителей городов.\n- Учти, что `TimeSplit` — учебная схема, а для строгого production backtest дополнительно потребуется label-availability cutoff и embargo.')
])
print('notebooks generated')
