#!/usr/bin/env python3
"""Reproducible fully synthetic Kyrgyz retail-loan panel for ML coursework.
No real borrower data. Regional stress is a deliberately invented sensitivity scenario.
Run: python generate_v2.py [--clients 16000] [--seed 20261103]
"""
import argparse, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp('2026-10-31')
MONTH = 30.4375
ROOT = Path(__file__).resolve().parents[1] / 'data' / 'source_synthetic'
SOURCES = {
  'cash':'public-product-reference',
  'secured':'public-product-reference',
  'auto':'public-product-reference',
  'mortgage':'public-product-reference',
  'installment':'public-product-reference',
  'geo_osh':'https://www.stat.gov.kg/ru/oshskaya-oblast/',
  'geo_chuy':'https://www.stat.gov.kg/ru/chujskaya-oblast/',
  'renaming':'public-admin-name-reference'
}
# Public family maxima are reference anchors; product splits and rates are assumptions.
# All simulated nominal-rate ranges, subproduct splits and distribution weights are NOT public terms.
PRODUCTS = [
  # code, family, display, public_limit_KGS, synthetic_min_KGS, synthetic_max_KGS, term_choices_months, nominal_range, weight, public_link, publicly_listed
  ('CASH_ONLINE_200','Cash','Онлайн-кредит 200',200000,10000,200000,[6,9,12,18,24,36],(22,29),.27,'cash',False),
  ('CASH_STANDARD_500','Cash','Наличные Стандарт 500',1000000,20001,500000,[6,12,18,24,36],(18,25),.21,'cash',False),
  ('CASH_GUARANT_1M','Cash','Наличные с поручителем 1 млн',1000000,300000,1000000,[12,18,24,36],(18,25),.055,'cash',False),
  ('SECURED_STANDARD','SecuredCash','Наличные под залог Стандарт',10000000,300000,5000000,[12,24,36,48,60],(16,23),.046,'secured',False),
  ('SECURED_PLUS','SecuredCash','Наличные под залог Плюс',10000000,2500000,10000000,[36,48,60],(16,22),.012,'secured',False),
  ('AUTO_USED','Auto','Автокредит: с пробегом',10000000,300000,10000000,[24,36,48,60],(19,26),.067,'auto',True),
  ('AUTO_DEALER','Auto','Автокредит: новый',18000000,500000,18000000,[24,36,48,60],(17,23),.025,'auto',True),
  ('MORTGAGE_STANDARD','Mortgage','Ипотека Стандарт',20000000,800000,20000000,[60,84,120,144],(14,19),.033,'mortgage',True),
  ('MORTGAGE_BANK_PROPERTY','Mortgage','Ипотека Готовое жильё',15000000,600000,15000000,[60,84,120],(9.5,15),.009,'mortgage',True),
  ('INSTALL_3M','Installment','Рассрочка 3 месяца',500000,3000,180000,[3],(0,0),.105,'installment',False),
  ('INSTALL_6_12M','Installment','Рассрочка 6–12 месяцев',500000,5000,350000,[6,9,12],(0,0),.116,'installment',False),
  ('INSTALL_24M','Installment','Рассрочка 24 месяца',500000,15000,500000,[18,24],(0,0),.047,'installment',False)
]
CITIES = [ # oblast, locality, location_type, synthetic_weight, income_factor
 ('г. Бишкек','Бишкек','город',22,1.31),
 ('г. Ош','Ош','город',10,1.09),
 ('Чуйская область','Кара-Балта','город',3.3,1.00),
 ('Чуйская область','Токмок','город',3.0,1.02),
 ('Чуйская область','Кант','город',2.0,1.06),
 ('Чуйская область','Сокулук','село',2.2,1.03),
 ('Чуйская область','Беловодское','село',1.2,0.93),
 ('Ошская область','Кара-Суу','город',4.2,0.91),
 ('Ошская область','Узген','город',2.7,0.87),
 ('Ошская область','Ноокат','город',2.5,0.88),
 ('Ошская область','Араван','село',2.0,0.85),
 ('Джалал-Абадская область','Манас','город',5.1,0.95),
 ('Джалал-Абадская область','Базар-Коргон','город',2.5,0.87),
 ('Джалал-Абадская область','Токтогул','город',1.9,0.88),
 ('Джалал-Абадская область','Кочкор-Ата','город',1.7,0.92),
 ('Иссык-Кульская область','Каракол','город',5.0,1.00),
 ('Иссык-Кульская область','Балыкчы','город',2.5,0.93),
 ('Иссык-Кульская область','Чолпон-Ата','город',1.6,1.04),
 ('Нарынская область','Нарын','город',3.5,0.90),
 ('Нарынская область','Кочкор','село',1.9,0.89),
 ('Таласская область','Талас','город',3.2,0.96),
 ('Таласская область','Кызыл-Адыр','село',1.8,0.89),
 ('Баткенская область','Баткен','город',3.5,0.91),
 ('Баткенская область','Кызыл-Кия','город',2.7,0.90),
 ('Баткенская область','Раззаков','город',1.8,0.87)
]
STRESS_CITIES={'Кара-Суу','Кара-Балта'}
JOBS = ['Госслужащий','Наемный сотрудник','Самозанятый','Торговля/услуги','Сезонная занятость']
JOB_P = [.22,.38,.18,.14,.08]
JOB_VOL = {'Госслужащий':.13,'Наемный сотрудник':.20,'Самозанятый':.43,'Торговля/услуги':.50,'Сезонная занятость':.72}

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-35,35)))
def payment_months(date, n):
    base = pd.Timestamp(date)
    return [base+pd.DateOffset(months=k) for k in range(1,n+1)]
def annuity(amount,annual_rate,months):
    r=annual_rate/1200
    return amount/months if r==0 else amount*r/(1-(1+r)**(-months))
def max_principal(income, existing_payment, annual_rate, term, target_dti):
    affordable=max(1200, income*target_dti-existing_payment)
    return affordable*term if annual_rate == 0 else affordable*(1-(1+annual_rate/1200)**(-term))/(annual_rate/1200)

def main(n=16000, seed=20261103):
    rng=np.random.default_rng(seed)
    weights=np.array([c[3] for c in CITIES],float); weights /= weights.sum()
    pw=np.array([p[8] for p in PRODUCTS]); pw/=pw.sum()
    city_ix = rng.choice(len(CITIES),size=n,p=weights)
    ages=rng.integers(21,67,size=n)
    occupations=rng.choice(JOBS,size=n,p=JOB_P)
    base_income=np.clip(rng.lognormal(np.log(56000),.58,size=n),18000,370000)
    volatility=np.clip(np.array([JOB_VOL[j] for j in occupations])+rng.normal(0,.075,size=n),.05,.93)
    income=np.array([round(base_income[i]*CITIES[ci][4]/1000)*1000 for i,ci in enumerate(city_ix)]).astype(int)
    customer_risk=np.clip(rng.normal(0,0.65,size=n)+.55*((volatility-.35)/.35),-2,2)
    n_loans=rng.choice([0,1,2,3,4],size=n,p=[.075,.375,.30,.185,.065])
    customer_rows=[]; loan_rows=[]; pay_rows=[]; model_rows=[]; cand_rows=[]
    cid_base=900000000
    for i in range(n):
        cid=f'KG_SYN2_{i+1:07d}'
        region,city,city_type,weight,incfac=CITIES[city_ix[i]]
        monthly_income=int(income[i]); vol=float(volatility[i]); job=str(occupations[i]); crisk=float(customer_risk[i]); age=int(ages[i])
        city_stress = 0.65 if city in STRESS_CITIES else 0.0
        customer_rows.append(dict(CustomerID=cid,Region=region,City=city,LocationType=city_type,AgeAt2026=age,
            Occupation=job,IncomeVolatility=round(vol,3),SyntheticMonthlyIncomeKGS=monthly_income,
            GeographyUse='EDA_STRESS_TEST_ONLY',DataOrigin='SYNTHETIC_NOT_REAL'))
        # Different issuance dates per borrower allow point-in-time feature calculations.
        first = pd.Timestamp('2020-01-01')+pd.Timedelta(days=int(rng.integers(0,1650)))
        start_dates=[]; next_date=first
        for _ in range(int(n_loans[i])):
            if next_date>CUTOFF-pd.Timedelta(days=25): break
            start_dates.append(next_date)
            next_date += pd.Timedelta(days=int(rng.integers(190,660)))
        prior_loans=[]; prior_payments=[]
        for j,issue in enumerate(start_dates):
            # Only contract details already known at origination are usable as predictors.
            previous=[]; past_dpds=[]; past_due=[]; prior_bad_payment_dates=[]
            for prior_payment in prior_payments:
                d=prior_payment['due_date']; delay=prior_payment['delay']
                if d>=issue: continue
                observed_dpd=max(0,min(delay,int((issue-d).days)))
                past_dpds.append(observed_dpd); past_due.append(d)
                if observed_dpd>=30: prior_bad_payment_dates.append(d+pd.Timedelta(days=30))
            past12=[past_dpds[k] for k,d in enumerate(past_due) if d>=issue-pd.Timedelta(days=365)]
            previous_max=max(past_dpds,default=0)
            past_dpd30=sum(x>=30 for x in past_dpds)
            lastbad=max(prior_bad_payment_dates,default=None)
            months_since_bad=((issue-lastbad).days/30.4375 if lastbad is not None else None)
            existing_payment=0.0
            active=0
            for prev in prior_loans:
                prev_start, prev_end, pmt=prev['issue'],prev['maturity'],prev['pmt']
                if prev_start<issue<prev_end:
                    active+=1
                    existing_payment+=pmt
            # No location in underwriting features; optional simulated shock impacts monthly cashflow only.
            no_history=int(len(prior_loans)==0)
            preferred=int(rng.choice(len(PRODUCTS),p=pw)); p=PRODUCTS[preferred]
            code,family,display,pubmax,minamount,maxamount,terms,rate_range,pweight,urlkey,official=p
            term=int(rng.choice(terms))
            rate=round(float(rng.uniform(*rate_range)),2) if rate_range[1]>rate_range[0] else 0.0
            # Log uniform credit amount + ability-to-pay constraint, allowing a tail of higher DTI.
            amount=float(np.exp(rng.uniform(np.log(minamount),np.log(maxamount))))
            if family!='Installment':
                cap=max_principal(monthly_income,existing_payment,rate,term,target_dti=float(rng.uniform(.48,.83)))
                if cap < 0.80*minamount:
                    # If unaffordable, simulate issuing a smaller cash/installment credit instead.
                    p=PRODUCTS[int(rng.choice([0,1,9,10],p=[.36,.22,.27,.15]))]
                    code,family,display,pubmax,minamount,maxamount,terms,rate_range,pweight,urlkey,official=p
                    term=int(rng.choice(terms)); rate=round(float(rng.uniform(*rate_range)),2) if rate_range[1]>rate_range[0] else 0
                    cap=max_principal(monthly_income,existing_payment,rate,term,target_dti=.72)
                    amount=float(np.exp(rng.uniform(np.log(minamount),np.log(maxamount))))
                amount=min(amount, max(minamount,cap))
            amount=int(np.clip(round(amount/1000)*1000,minamount,maxamount))
            pmt=round(annuity(amount,rate,term),2)
            dti=float((existing_payment+pmt)/max(monthly_income,1))
            # Delinquencies are generated from measurable economic factors plus a persistent borrower factor.
            # Geo stress is a *documented invented stress experiment*, not a factual risk estimate for cities.
            job_adjust=.24 if job=='Сезонная занятость' else (.10 if job=='Торговля/услуги' else 0)
            prod_adjust={'Installment':-.28,'Cash':.20,'SecuredCash':-.13,'Auto':-.08,'Mortgage':-.22}[family]
            macro_adjust=0.17 if issue.year in (2022,2024) else 0
            base_logit=-5.0+1.75*min(dti,1.7)+.80*vol+.65*crisk+job_adjust+prod_adjust+macro_adjust
            # Stress mechanism: temporary liquidity/earnings disruption in the selected scenario locations.
            stress_signal=city_stress if issue.year in [2022,2023,2024,2025] else 0
            severe_prob=float(sigmoid(base_logit+stress_signal))
            mild_prob=float(np.clip(.065+.10*vol+.07*min(dti,1.5),.06,.3))
            creditid=cid_base+len(loan_rows)+1
            due_dates=payment_months(issue,term)
            due_dates=[d for d in due_dates if d<=CUTOFF]
            monthly_payments=[]; targetbad=False
            for installment_no,due in enumerate(due_dates,1):
                u=rng.random()
                if u<severe_prob:
                    delay=int(rng.choice([rng.integers(30,60),rng.integers(60,95),rng.integers(95,160)],p=[.65,.29,.06]))
                elif u<severe_prob+mild_prob:
                    delay=int(rng.integers(1,30))
                else:
                    delay=0
                actual_date=due+pd.Timedelta(days=delay)
                observed_date=actual_date if actual_date<=CUTOFF else pd.NaT
                observed_dpd=delay if observed_date is not pd.NaT else min(delay,(CUTOFF-due).days)
                if 0<(due-issue).days<=365 and observed_dpd>=30: targetbad=True
                monthly_payments.append(dict(due_date=due,delay=delay))
                prior_payments.append(dict(due_date=due,delay=delay))
                pay_rows.append(dict(CreditID=creditid,CustomerID=cid,InstallmentNo=installment_no,
                    DueDate=due.date().isoformat(),DueAmountKGS=pmt,
                    PaymentDate=observed_date.date().isoformat() if pd.notna(observed_date) else '',
                    PaidAmountKGS=pmt if pd.notna(observed_date) else 0,
                    DPDAtCutoff=int(observed_dpd),
                    IsPaymentCompletedAtCutoff=int(pd.notna(observed_date)),
                    DataOrigin='SYNTHETIC_NOT_REAL'))
            maturity=issue+pd.DateOffset(months=term)
            status='Closed' if maturity<=CUTOFF else 'Active'
            collateral='Vehicle' if family=='Auto' else ('RealEstate' if family in ['Mortgage','SecuredCash'] else 'None')
            guarantee=int(code=='CASH_GUARANT_1M')
            loan_rows.append(dict(CreditID=creditid,CustomerID=cid,Region=region,City=city,
                ProductFamily=family,ProductCode=code,ProductName=display,
                CreditOpenDate=issue.date().isoformat(),CreditEndDate=maturity.date().isoformat(),
                AgreementSumN=amount,Rate=rate,LoanTermMonths=term,MonthlyPaymentKGS=pmt,
                DebtToIncomeRatio=round(dti,4),OtherMonthlyDebtKGS=round(existing_payment,2),
                GuaranteeFlag=guarantee,CollateralType=collateral,
                CreditStatus=status,DataOrigin='SYNTHETIC_NOT_REAL'))
            # Need a full 12-month observation window including a 30-day delinquency maturation period.
            enough_observation=(issue+pd.DateOffset(months=12)+pd.Timedelta(days=30)<=CUTOFF)
            segment='train' if issue<pd.Timestamp('2024-01-01') else ('validation' if issue<pd.Timestamp('2025-01-01') else 'test')
            model_rows.append(dict(CustomerID=cid,CreditID=creditid,SnapshotDate=issue.date().isoformat(),
                Region=region,City=city,ProductFamily=family,ProductCode=code,AgeAtIssue=max(18,age-(2026-issue.year)),
                Occupation=job,SyntheticMonthlyIncomeKGS=monthly_income,
                IncomeVolatility=round(vol,3),AgreementSumN=amount,Rate=rate,LoanTermMonths=term,
                MonthlyPaymentKGS=pmt,DebtToIncomeRatio=round(dti,4),
                PreviousLoansCount=len(prior_loans),ActivePreviousLoansCount=active,
                PreviousPaymentsObserved=len(past_dpds),MaxHistoricalDPD=previous_max,
                DPD30CountHistory=past_dpd30,DPD30CountLast12M=sum(x>=30 for x in past12),
                LatePaymentRatio=round(np.mean(np.array(past_dpds)>0),4) if past_dpds else None,
                MonthsSinceLastDPD30=round(months_since_bad,1) if months_since_bad is not None else None,
                TotalPriorLoanAmountKGS=sum(x['amount'] for x in prior_loans),
                NoCreditHistory=no_history,TimeSplit=segment,
                TargetDPD30_12M=int(targetbad) if enough_observation else None,
                IsMatureTarget=int(enough_observation),DataOrigin='SYNTHETIC_NOT_REAL'))
            prior_loans.append(dict(issue=issue,maturity=maturity,pmt=pmt,amount=amount))
        # Candidate pool: no active loan on cutoff, no automatically assigned Target.
        n_active=sum(x['maturity']>CUTOFF for x in prior_loans)
        if n_active==0:
            prior_dpd=[min(pay['delay'],(CUTOFF-pay['due_date']).days) for pay in prior_payments if pay['due_date']<=CUTOFF]
            prior_dpd=[max(0,d) for d in prior_dpd]
            cand_rows.append(dict(CustomerID=cid,SnapshotDate=CUTOFF.date().isoformat(),
                Region=region,City=city,Age=age,Occupation=job,
                SyntheticMonthlyIncomeKGS=monthly_income,IncomeVolatility=round(vol,3),
                PreviousLoansCount=len(prior_loans),MaxHistoricalDPD=max(prior_dpd,default=0),
                PastDPD30Count=sum(d>=30 for d in prior_dpd),
                NoCreditHistory=int(len(prior_loans)==0),ActiveLoansAtSnapshot=0,
                PropensityTarget='',TargetDPD30_12M='',
                DataOrigin='SYNTHETIC_NOT_REAL'))
    out=ROOT; out.mkdir(parents=True,exist_ok=True)
    datasets={
      'customers.csv':pd.DataFrame(customer_rows),'loans.csv':pd.DataFrame(loan_rows),
      'repayments.csv':pd.DataFrame(pay_rows),
      'model_dataset.csv':pd.DataFrame(model_rows),
      'marketing_candidates.csv':pd.DataFrame(cand_rows)
    }
    catalog=[]
    for p in PRODUCTS:
        code,family,display,limit,lo,hi,terms,rate_range,wt,urlkey,isofficial=p
        catalog.append(dict(ProductCode=code,ProductFamily=family,ScenarioSubproductName=display,
              PublicListedSubproduct=int(isofficial),PublicMaximumKGS=limit,
              ScenarioMinKGS=lo,ScenarioMaxKGS=hi,ScenarioTermsMonths='/'.join(map(str,terms)),
              ScenarioNominalRateMinPct=rate_range[0],ScenarioNominalRateMaxPct=rate_range[1],
              SourceURL=SOURCES[urlkey],Definition='PUBLIC_FAMILY_LIMIT_AND_SYNTHETIC_PARAMETERS'))
    datasets['product_catalog.csv']=pd.DataFrame(catalog)
    geo=[]
    for region,city,typ,w,mult in CITIES:
        geo.append(dict(Region=region,City=city,LocationType=typ,SyntheticSamplingWeight=w,
                SyntheticIncomeFactor=mult,SyntheticStressFlag=int(city in STRESS_CITIES),
                ExtraSeverePaymentLogOddsScenario=(0.65 if city in STRESS_CITIES else 0),
                Source='https://www.stat.gov.kg/ru/statistika-po-regionam/',
                Note='City spellings checked; weights, income factors and stress values are assumptions'))
    datasets['geography_scenarios.csv']=pd.DataFrame(geo)
    for name,df in datasets.items():
        df.to_csv(out/name,index=False,encoding='utf-8-sig')
    dm=datasets['model_dataset.csv']; clean=dm.loc[dm.IsMatureTarget.eq(1)]
    report=clean.groupby('City').agg(Loans=('CreditID','size'),DPD30_12M_Rate=('TargetDPD30_12M','mean')).sort_values('DPD30_12M_Rate',ascending=False)
    report.to_csv(out/'eda_city_target_rates.csv',encoding='utf-8-sig')
    prod=clean.groupby(['ProductFamily','ProductCode']).agg(Loans=('CreditID','size'),DPD30_12M_Rate=('TargetDPD30_12M','mean'),AvgAmountKGS=('AgreementSumN','mean')).reset_index()
    prod.to_csv(out/'eda_product_target_rates.csv',index=False,encoding='utf-8-sig')
    meta=dict(version='2.0',random_seed=seed,generated_at_reference='2026-11-03',cutoff=str(CUTOFF.date()),
         **{name:len(df) for name,df in datasets.items()},mature_model_rows=len(clean),
         overall_target_rate=float(clean.TargetDPD30_12M.mean()),city_stress_assumptions={'Кара-Суу':'synthetic','Кара-Балта':'synthetic'},
         clearly_synthetic=True,sources=SOURCES,
         caveats=['No real bank customers or historical location default rates','Partial payments are not modeled','Income at issue is simulated; not a verified salary statement','New-to-credit marketing applicants have no validated supervised target','Rates are scenario nominal values; publicly published rates are effective minima','Geography is excluded from model features; analyze for stress and fairness'])
    with open(ROOT/'dataset_metadata.json','w',encoding='utf-8') as f: json.dump(meta,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in meta.items() if k not in ['sources','caveats']},ensure_ascii=False,indent=2))
    print('TOP CITY RISK / COUNTS\n',report.to_string())
    print('PRODUCT\n',prod.to_string(index=False))
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--clients',type=int,default=16000)
    parser.add_argument('--seed',type=int,default=20261103)
    args=parser.parse_args()
    main(args.clients,args.seed)
