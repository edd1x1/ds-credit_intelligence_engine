# Decision Engine v6 — алгоритм и расчёт

**Только предварительный отбор для внутреннего рассмотрения. Не является банковским решением о выдаче.**

## Фильтрация

- Клиент без действующих кредитов в анализируемом банке (по умолчанию).
- Доход >=30 000 сом, возраст 21–65 лет, стабильность поступлений и история DPD в регулируемых UI пределах.
- 4 альтернативных офера, выбранный должен проходить пороги: PD12<=18%, внутренний ДТИ<=40%, допустимый размер кредита. Эти границы — сценарные **внутренние** правила, не тождественные нормативным порогам.
- Внешние кредиты отсутствуют в источниках, не подставляются нулём как «реальные нулевые обязательства»; поле `ExternalDebtStatus=UNKNOWN_NOT_QUERIED`.
- `SelfBanStatus=NOT_CHECKED_PREOFFER`; `MarketingConsentStatus=UNKNOWN_REVIEW_REQUIRED`.

## Финансовая модель

`InternalDTI=(InternalExistingMonthlyPayments+ProposedAnnuityPayment)/IncomeMean6M`.

`ContributionIfAccepted=ProjectedInterest+MerchantIncome-ScenarioEL-FundingCost-ServicingCost`.

`EVperContact=P(Accept)×ContributionIfAccepted-ContactCost`.

**Главное исправление v6:** CostContact вычитается **один раз из каждого контакта**, не повторно на каждого принявшего кредит. `P(Accept)` на данный момент одинаковая, задана слайдером (8% по умолчанию), не предсказана. Поэтому ранжирование по EV не моделирует персональную склонность к кредитованию.

Сценарная льготная ставка применяется только при прошлой дисциплине без просрочек и стабильном доходе. Уменьшает процентный доход, если отдельный эффект на конверсию не подтвержден. Слайдер по умолчанию 0 п.п. — пользователь может выполнить sensitivity analysis.

## Пример структуры экспорта

`CustomerID, RecommendedProduct, ProductCode, OfferAmountKGS, OfferRatePct, LoyaltyDiscountPctPts, PD12, InternalExistingDebtKGS, MonthlyPaymentKGS, ProposalDTI, ScenarioAcceptanceProbability, ContributionIfAcceptedKGS, ExpectedValuePerContactKGS, RecommendationReason, DecisionStatus, ModelVersion, ExternalDebtStatus, SelfBanStatus, MarketingConsentStatus`.

Статус `PRE_OFFER_REVIEW`: не отправлять push без проверки законности коммуникации и согласия. Реальное оформление кредита требует отдельного основного кредитного процесса.
