# AquaLimit

> 위성 InSAR로 관측한 지표 변형으로 지하 모델을 갱신하고, 모델의 불확실성까지 반영해 취수정별 취수계획을 제시하는 지하수 취수 최적화 엔진.

“지반침하 위험을 관리기준 안에 두면서 필요한 물을 공급하려면, 어느 우물에서 언제 얼마나 뽑아야 하는가?”를 계산한다. 조건을 만족하는 계획이 없으면 “추가 용수 확보 없이는 수요를 충족할 수 없다”고 반환한다. 관리기관에는 지금 규칙(예: 취수 금지)이 기준을 지키기에 충분한지, 가장 싼 보완책이 무엇인지, 규칙이 실제로 지켜지는지(위성 이행 점검)를 알려 준다.

## 문서

| 문서 | 역할 |
|---|---|
| [기획서](docs/PROPOSAL.md) | 문제·고객·산출물·범위·차별점 |
| [기술 설계](docs/TECH_APPROACH.md) | 위성 관측 → 지하 상태 추정 → 미래 계산 → 강건 최적화 |
| [데이터](docs/DATA_SOURCES.md) | 필요한 자료와 확보 경로, 확인 상태 |
| [검증 지역 선정](docs/REGION_SELECTION.md) | 지역 후보와 선정 기준 |
| [검증 계획](docs/VALIDATION_PLAN.md) | 4단계 비교 실험과 평가 지표 |
| [사업 모델](docs/BUSINESS_MODEL.md) | 고객, 고객 가치(정책 비교·금지 반경 비교·이행 점검), 수익 구조, 검증할 가설 |
| [국내 적용](docs/KOREA.md) | 국산 위성(농림위성·수자원위성·아리랑 6호)으로 입력 교체, 국내 수요, 시범 지역 |
| [고객 검증 메일](docs/outreach/README.md) | WDWA GSA·DWR CASP 질문 메일 초안 |
| [실행 계획](docs/EXECUTION_PLAN.md) | 역할, 주별 일정, 판단 관문 |
| [위험과 대응](docs/RISKS.md) | 주요 위험과 축소 방안 |
| [9/30 발표](docs/progress/0930/README.md) | 서비스 설명 발표 원고 |
| [10/7 발표](docs/progress/1007/README.md) | 합성 대수층 검증 결과 발표 원고 |
| [10/13 발표](docs/progress/1013/README.md) | 실제 지역 첫 결과(CVHM2 모델, GPS 검증, 계획, AI) |
| [10/28 발표](docs/progress/1028/README.md) | 모델 v3, 실제 우물·규제값 정책 비교, 금지 반경 비교, NASA OPERA 위성 점검 |
| [가상 데이터 데모](demo/README.md) | 불확실성 반영 취수 계획 데모 실행 방법 |
| [실제 검증 지역](real/README.md) | Aqueduct MP 195~215 실제 자료 → MODFLOW(CVHM2) 역산·GPS 검증, 정책 비교, 취수 계획, 위치 입력 AI, 최종 데모 |
| [채점표](real/SCOREBOARD.md) | 모델 구조 후보별 연습 시험·최종 시험 점수 |
| [OPERA 위성 점검](real/results_opera/RESULTS.md) | DWR vs NASA OPERA vs GPS, 금지 전후 위성 이중차분, 현장 확인 후보 |
| [합성 대수층 검증](synthetic/README.md) | 역산 복원·E1~E4 비교 실험(R2 관문), [결과](synthetic/results/RESULTS.md), [MODFLOW 6 참값 결과](synthetic/results_mf6/RESULTS.md) |
