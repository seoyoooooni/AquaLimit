# AquaLimit 검증 지역 선정

> 결정(10/12): **캘리포니아 샌호아킨밸리 서부, California Aqueduct 연접 구간(Kern County Subbasin)** 을 1차 검증 지역으로 한다. 대만 줘수이강 선상지는 2차(보조) 후보, 국내는 적용 시장으로만 설명한다.

## 1. 선정 기준

| 기준 | 이유 |
|---|---|
| 지하수 취수로 인한 침하가 관측·보고된 지역 | 문제가 실재해야 검증 의미가 있음 |
| InSAR 변형 시계열 확보 가능 | 모듈 ① 입력 |
| 관측정 수위 시계열 확보 가능 | 보정·검증 |
| 취수 기록 또는 합리적 추정 가능 | 최적화 대상 |
| 기존 지하수 모델·지질 자료 존재 | 층 구성 시간 단축 |
| 국내 적용 가능성 | 국내 시장 확장 근거 |

## 2. 후보별 충족표

○ 공개·즉시 사용 가능 / △ 있으나 신청·가공 필요 또는 일부만 공개 / × 확인 못 함

| 기준 | 센트럴밸리(샌호아킨) | 대만 줘수이강 선상지(윈린) | 멕시코시티 | 자카르타 | 국내 |
|---|---|---|---|---|---|
| 취수로 인한 침하 | ○ 과거 최대 약 8.5m, 가뭄기 재침하 | ○ 윈린 최대 연 60~70mm | ○ 연 수십 cm급 | ○ 연 수 cm~수십 cm | × 광역 양수 침하 공개 사례 못 찾음(터널 공사 수위저하 등 국지 사례만) |
| InSAR | ○ DWR TRE Altamira(100m, 월별, 2015.6~) + NASA OPERA DISP(30m, 2016.7~) | △ Sentinel-1 원자료 직접 처리 필요 | ○ OPERA 범위(미국 남부~파나마)에 포함 | △ 원자료 직접 처리 | △ 원자료 직접 처리 |
| 관측정 수위 | ○ DWR 정기·연속 측정, USGS NWIS | △ 4개 대수층 관측정 200여 곳(수자원서 자료, 신청 필요) | × 공개 시계열 확인 못 함 | × | ○ 국가지하수정보센터 |
| 취수 기록 | △ 관정별 공개 없음. GSP 연차보고서(구역 합계), CASP 완충구역 관정별 계량 보고(2024~, 공개 여부 미확인) | × 확인 못 함 | × | × | △ 허가량 수준 |
| 기존 모델·지질 | ○ USGS CVHM2(MODFLOW-OWHM, 12층) 모델 파일·입력 공개, 텍스처 모델 공개 | ○ 다층 압밀계 32곳, 수준측량 약 500점, 지질 시추 | △ 논문 수준 | △ 논문 수준 | △ |
| 검증용 독립 자료 | ○ 연속 GPS, 신장계(extensometer) | ○ GNSS 400여 곳, 연 1회 수준측량 | △ | △ | × |
| 국내 적용 설명 | △ | △ 지질·농업 양수 구조가 국내와 비슷한 충적층 | × | × | ○ |
| **필수 자료(InSAR·수위·지질) 충족** | **충족** | 신청 필요 | 미충족 | 미충족 | 침하 사례 부족 |

## 3. 결정

**1차 검증: 샌호아킨밸리 서부, California Aqueduct 이정표(MP) 195~215 주변(Kern County Subbasin)**

- 필수 자료 세 가지를 모두 공개 자료로 바로 받을 수 있는 유일한 후보다.
- 침하가 국가 기간시설(수로)을 실제로 손상시키고 있어, “침하 기준 안에서 취수 배분”이라는 서비스 문제와 정확히 맞는다.
- DWR 수로 침하 프로그램(CASP)과 지역 GSA가 원인 조사(2020~2023 연구 5건)와 관정 계량 의무화(완충구역 2.5마일, 2024~)를 하고 있어 고객 설명이 쉽다.
- CVHM2(USGS)의 층 구조·파라미터를 초기값으로 쓰면 MODFLOW 6 + CSUB 모델 구축 시간이 줄어든다.
- 대상 범위는 약 30km × 30km 이내 소구역으로 잘라 쓴다(계산량 관리).

**취수 기록 처리** — 관정별 공개 자료가 없으므로 [데이터 원칙](DATA_SOURCES.md#3-자료-원칙)대로 한다.
- 실제 InSAR·수위 자료는 모델 보정(역산)과 예측 오차 검증에 쓴다. 이건 실제 관측으로 확인한 성능이다.
- 최적화 대상 우물·취수량은 CVHM2 양수 추정치와 GSP 연차보고서 합계를 바탕으로 만든 합성 취수 시나리오로 둔다.
- 따라서 결과는 “실제 운영 개선”이 아니라 “실제 지역 자료로 보정한 모델에서의 시뮬레이션 결과”로 표현한다.

**2차(보조) 후보: 대만 줘수이강 선상지** — 압밀계·관측정 밀도가 가장 좋다. 자료 신청이 늦으면 제외하고, 받으면 “다른 지역에서도 작동하는가” 확인용으로만 쓴다.

**국내** — 지하수 양수로 인한 광역 침하를 관측한 공개 사례와 자료를 찾지 못했다. 국내는 적용 시장(지하수 의존 농촌·산업단지, 국가지하수정보센터 관측망 활용)으로만 설명한다.

## 4. 다음 할 일

| 작업 | 내용 |
|---|---|
| 소구역 경계 확정 | MP 195~215 주변 30km 이내, CVHM2 격자와 맞춤 |
| InSAR 내려받기 | DWR TRE Altamira 월별 래스터(2015~) 소구역 잘라내기, OPERA DISP 프레임 확인 |
| 수위 자료 | DWR 정기·연속 측정에서 소구역 관측정 목록, 측정 빈도 확인 |
| CVHM2 | 모델 파일 내려받아 소구역 층 구조·파라미터 추출 |
| 대만 자료 | 수자원서(WRA) 자료 신청 가능 여부 문의 |

## 출처

- DWR TRE Altamira InSAR: https://catalog.data.gov/dataset/tre-altamira-insar-subsidence-data-3a443
- NASA OPERA DISP-S1: https://www.earthdata.nasa.gov/news/opera-project-releases-level-3-displacement-products
- USGS CVHM2: https://ca.water.usgs.gov/projects/hydrologic-modeling/CVHM2.html
- DWR 정기 지하수위 측정: https://catalog.data.gov/dataset/periodic-groundwater-level-measurements-bc2a9
- DWR 연속 지하수위 측정: https://catalog.data.gov/dataset/continuous-groundwater-level-measurements-5481b
- USGS California Aqueduct 침하 연구: https://ca.water.usgs.gov/projects/central-valley/california-aqueduct-subsidence.html
- WDWA GSA 관정 계량 보고(CASP 완충구역): https://wdwagsa.specialdistrict.org/well-extraction-volume-reporting-within-proximity-to-critical-infrastructure-impacted-by-subsidence-ed042d0
- WDWA GSA 침하 연구 목록: https://wdwagsa.specialdistrict.org/subsidence-studies
- 대만 줘수이강 선상지(Nguyen 외, 2024, Remote Sensing): https://www.mdpi.com/2072-4292/16/20/3789
- 대만 줘수이강 선상지(Hung 외, 2024, Frontiers in Earth Science): https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2024.1370626/full
- 국내 InSAR 침하 사례(문준식·오형석, 2022): https://accesson.kr/jkta/v.24/2/171/27637
