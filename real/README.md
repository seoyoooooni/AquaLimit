# 실제 검증 지역 (California Aqueduct MP 195~215)

Lost Hills 주변 31km × 31km. 위성(InSAR)·Landsat·기상·관측정·USGS CVHM2로 지하수·침하 물리모델(MODFLOW 6 + CSUB)을 만들고, 보정에 쓰지 않은 **GPS로 예측을 검증**한 뒤, 실제 정책(2024.12 수로 주변 취수 금지)과 비교하는 정책 계산, 사업자 우물 계획, 위치 입력 AI, 최종 데모까지 만든 폴더다.

(10/22 갱신: 지금 쓰는 모델은 v3(검증)·v3ops(운영)다. 아래 '이전 버전' 결과 폴더는 비교용으로만 남겨 둔다.)

## 지금 쓰는 모델과 결과

| 모델 | 보정 자료 | 용도 | 결과 폴더 |
|---|---|---|---|
| **v3 검증 모델** | DWR InSAR + 관측정, 2015.7~2022.12 | 예측 성능 채점(보정에 안 쓴 2023~2026 GPS) | `results_v3/`, `results_validate_v3/` |
| **v3ops 운영 모델** | + 2023~2026.4 GPS·관측정, 금지 후 취수 배수 추정 | 정책 비교·계획·AI·데모 | `results_v3ops/`, `results_validate_v3ops/`, `results_policy_v3ops/`, `results_plan_v3ops/`, `results_ai_v3ops/` |
| v3ops-opera (10/22) | 위성을 NASA OPERA로 바꾸고 2026.2까지 위성도 보정에 씀 | 금지 후 취수 배수 재추정(점검용) | `results_v3ops-opera/`, `results_validate_v3ops-opera/` |

v3의 바뀐 점: 지연 압밀(남·북 구역 따로), 2005~2015 과거 취수 이력(spin-up), 관개 회귀수 함양, 현실적 사전범위, 90% 범위 계산 수정. 취수량은 Landsat NDVI × gridMET 기준증발산(기온·습도·바람·일사)으로 칸별 작물 증발산을 구하고, 비·수로 물 배분율을 뺀 나머지를 지하수로 둔다.

| 결과 | 문서 | 한 줄 요약 |
|---|---|---|
| 검증 점수 | [results_v3/RESULTS.md](results_v3/RESULTS.md), [구조 오차 다시 재기](results_v3/RESULTS_uncertainty.md) | GPS 침하 오차 E2 70.9 → **E3 21.2mm**, E4 90% 범위 GPS 포함 99%(폭 97mm). 단순 추세 연장 7.2mm가 아직 더 정확. 구조 오차를 OPERA 정답으로 다시 재면 폭 59mm·포함 82%(90% 미달 → 기존 유지) |
| 채점표 | [SCOREBOARD.md](SCOREBOARD.md), [RESCORE_PRACTICE.md](RESCORE_PRACTICE.md) | 구조 후보 v4(6종)는 연습 시험에서 모두 v3보다 나빠 채택 안 함. 10/23 연습 시험 정답을 DWR → OPERA로 바꿔(DWR가 2020~22 GPS 침하의 절반만 잡음) 다시 채점해도 v3가 1등, 모델 편향 +2mm |
| 자연 실험·원인 분해 | [results_validate_v3/](results_validate_v3/RESULTS.md), [results_validate_v3ops/](results_validate_v3ops/RESULTS.md) | 습한 해 감속은 맞힘, 금지 후는 못 맞힘. 운영 모델: 금지구역 안 취수 위성 추정의 1.36배 |
| 정책 비교 | [results_policy_v3ops/RESULTS.md](results_policy_v3ops/RESULTS.md) | 4km 전면 금지로는 평년 45%·가뭄 32% 수로 지점만 기준(연 0.5인치) 충족 |
| 금지 반경 비교 (MODFLOW 확인) | [results_policy_v3ops/RESULTS_refine.md](results_policy_v3ops/RESULTS_refine.md) (선형 응답 판: [RESULTS_radius.md](results_policy_v3ops/RESULTS_radius.md)) | 같은 신뢰도(MODFLOW 모든 구성원·지점 기준 안)에서 가뭄에 맞춘 고정 반경(6.9km)보다 평년 20%·습한 해 29%·가뭄 10% 싸다. 선형으로 푼 계획은 MODFLOW로 확인·보정(60~97% → 100%) |
| OPERA 위성 점검 | [results_opera/RESULTS.md](results_opera/RESULTS.md) | GPS와 방향 일치 DWR 3/5 → OPERA 5/5. 위성 이중차분: 금지 후 금지구역 안이 밖보다 +2.9mm/년 덜 좋아짐 |
| 사업자 계획 | [results_plan_v3ops/RESULTS.md](results_plan_v3ops/RESULTS.md), [표본 밖](results_plan_v3ops/RESULTS_oos.md) | 금지구역 밖 실제 우물 6개. 기준 초과 E1 47% → E4 7%(계획에 안 쓴 땅속 상태) |
| 위치 입력 AI | [results_ai_v3ops/RESULTS_loc.md](results_ai_v3ops/RESULTS_loc.md) | R² 0.92, 1,000개 땅속 상태 × 우물 6곳 MODFLOW 11시간 → 1초 미만 |

## 1. 자료

![침하 속도 지도](figures/real-1-velocity-map.png)

| 자료 | 내용 | 파일 |
|---|---|---|
| InSAR 수직 변위 (DWR) | DWR TRE Altamira(Sentinel-1) 측정점 74,944개, 2015.1~2026.4, 약 6일 간격. **2023년 이후는 GPS와 어긋나 쓰지 않음** | `data/insar_points_velocity.csv`, `data/insar_grid500m_monthly.npz` |
| InSAR 수직 변위 (NASA OPERA) | OPERA DISP-S1 30m, 상승 36541·하강 38502·38503 → 500m 칸 수직·동서 성분, 2016.7~2026.2 | `data/opera/reduced/`(줄인 파일), `data/opera_grid500m_monthly.npz` (`fetch_opera.py`, `prepare_opera.py`) |
| 지하수위 | DWR 정기 측정 관측정 30곳, 1,831건 | `data/wells.csv`, `data/gwl_measurements.csv` |
| GPS 수직 위치 | 수로 GPS 5곳(Nevada Geodetic Lab, IGS20) — v3 검증 전용, v3ops 보정에도 사용 | `data/gps_monthly.csv` |
| 작물·기상 | Landsat NDVI, gridMET 기준증발산, USGS SSEBop, 기온 | `data/ndvi_monthly.npz`, `data/eto_gridmet_monthly.npz`, `data/et_ssebop_monthly.npz`, `data/temp_monthly.csv` |
| 수로 물 배분율 | SWP 연도별 배분율 | `data/swp_allocation.csv` |
| 강수 | NOAA Lost Hills·Buttonwillow 관측소 월 강수 | `data/precip_monthly.csv` |
| 우물 신고 | DWR Well Completion Reports(농업용 신규 우물) | `data/wcr_raw.json` |
| 층 구조·지질 | USGS CVHM2 13개 층 → 4개 묶음 | `data/cvhm2_box.npz` (`fetch_cvhm2.py`) |
| 수로·측량점·유전 | SWP 반복구간 선형, 정밀측량점, CalGEM 유전 경계 | `data/*.geojson` |

### 자료에서 발견한 것

- **유전 침하가 섞여 있다.** 수로 2km 이내 격자 39%가 Lost Hills·Belridge 유전 안이다. 유전 격자는 역산·취수에서 뺐다.
- **2023년 이후 DWR InSAR와 GPS가 어긋난다.** P544·Q204에서 DWR는 지반 상승, GPS는 침하. NASA OPERA는 5곳 모두 GPS와 같은 방향(누적 오차 중앙값 DWR 17.0mm, OPERA 8.3mm). → 공개 위성 자료도 GPS로 교차 검증하는 **자료 품질 검사**가 서비스 기능이다.
- **수로 GPS에는 계절 변동이 있다.** 여름에 수 mm 올라가고 겨울에 내려간다(같은 칸 OPERA는 여름에 내려감). 속도 비교는 같은 달 → 같은 달 1년 단위로 해야 한다.
- **수로 옆 관측정·GPS는 최근에 생겼다.** 다층 관측정 4곳은 2023~2025년 설치, GPS Q198·Q204·Q208·Q213은 2023~2025년 시작(DWR CASP 관측망).

## 2. 모델

- MODFLOW 6 + CSUB, 1km 격자 32×31, 4개 층 묶음(CVHM2), 2005.7~2015.6 과거 이력 + 2015.7~2026.4 월 단위
- 압축층: 과거 최저 수위 아래로 내려가면 비탄성(영구) 압축. 대수층 안 점토층은 지연 압밀(남·북 구역별 투수계수), 코코란 점토는 비지연(지연으로 두면 수렴 안 함)
- 역산: 가우스-뉴턴(레벤버그-마쿼트) 최소제곱, 파라미터 21개(운영 모델 23개). 불확실성: 선형화 사후분포 앙상블 + 되돌아보기 검증으로 잰 모델 구조 오차

## 3. 실행

```bash
cd real
pip install -r requirements.txt
get-modflow :flopy                                   # MODFLOW 6 실행파일 설치

caffeinate -i bash run_all.sh 2>&1 | tee log_all.txt # v3·v3ops 전체 12단계(10코어 맥 약 3시간, 멈추면 다시 실행)
streamlit run app_final.py                           # 최종 데모(운영 모델 v3ops)

# 10/22 추가
python3 prepare_opera.py --build                     # OPERA 줄인 파일 → 500m 수직 변위(몇 초)
python3 opera_analysis.py                            # OPERA vs DWR vs GPS, 금지 전후 이중차분 → results_opera/
AQ_VARIANT=v3ops python3 policy_ban_radius.py        # 금지 반경 넓히기 vs AquaLimit(몇 초, MODFLOW 안 돌림)
AQ_INSAR=opera AQ_VARIANT=v3ops python3 run_real.py           # OPERA 운영 모델 → results_v3ops-opera/
AQ_INSAR=opera AQ_VARIANT=v3ops python3 validate_response.py  # 금지 후 원인 분해(OPERA 모델)

bash run_dev.sh v4-xxx                               # 구조 후보를 연습 시험으로 비교 → SCOREBOARD.md
```

결과 폴더마다 `model_version.json`(structure, pumping, et_source, variant, insar, 파라미터 이름)이 있고, 지금 설정과 다르면 스크립트가 멈추고 어떤 환경변수로 실행해야 하는지 알려 준다.

| 파일 | 역할 |
|---|---|
| `real_model.py` | 자료 읽기, MODFLOW 모델, 역산, 불확실성, 평가. 환경변수 `AQ_VARIANT`(v3·v3ops·v4-…), `AQ_INSAR`(dwr·opera) |
| `run_real.py` | E2·E3·E4 역산 + GPS 검증(`--dev` 연습 시험, `--e3-only` 입력별 기여) |
| `validate_response.py` | 자연 실험(습한 해·금지)과 금지 후 원인 분해 |
| `policy_real.py` | 실제 우물 56개로 전면 금지 vs AquaLimit(과거 가뭄 재현, 다음 관개기 날씨별) |
| `policy_ban_radius.py` | 금지 반경 넓히기 vs AquaLimit(저장된 응답으로 계산) |
| `plan_real.py`, `run_plan.py`, `run_weather.py`, `eval_plan_oos.py` | 사업자 우물 계획, 날씨 시나리오, 표본 밖 평가 |
| `surrogate_loc.py` | 위치 입력 AI 대리모델 |
| `app_final.py` | 최종 Streamlit 데모 |
| `fetch_data.py`, `prepare_insar.py`, `fetch_cvhm2.py`, `fetch_et.py`, `fetch_landsat.py`, `fetch_opera.py`, `prepare_opera.py` | 자료 수집·가공 |
| `opera_analysis.py` | OPERA 위성 점검(모델 없음) |
| `scoreboard.py`, `run_dev.sh` | 구조 후보 채점표 |
| `rescore_practice.py` | 연습 시험 다시 채점(정답 DWR / OPERA / GPS P544, 재보정 없음) |
| `rescore_uncertainty.py` | E4 구조 오차를 OPERA 정답으로 다시 재고 GPS 포함률 확인 |
| `refine_plan.py` | 정책 계획을 MODFLOW로 확인·보정 + 금지 반경을 MODFLOW로 탐색(같은 신뢰도 비용 비교) |
| `verify_ban_radius.py` | 금지 반경안(선형)의 MODFLOW 확인(refine_plan.py 이전 단계) |
| `hybrid_forecast.py` | 결합 예측(추세 + 물리모델 변화) 채점 |
| `run_all.sh` | 최종 계산 전체 |

### 이전 버전(비교용)

| 폴더 | 내용 |
|---|---|
| `results_uniform/`, `results_plan/`, `results_ai/`, `app_real.py` | 10/13 첫 번째 모델(취수 균일 가정, 가정 우물). `AQ_PUMPING=uniform` |
| `results/` | 10/13 새 버전(v2, 위성 증발산 취수) |
| `results_simple/` | CVHM2 대신 단순 3개 층 가정 |
| `results_v4*/` | 구조 후보 4종(연습 시험에서 탈락) |

## 4. 한계

- 3년 전체 침하량 예측은 단순 추세 연장(7.2mm)보다 아직 부정확하다(v3 E3 21.2mm). 정책·계획은 응답(취수 변화의 효과)으로 정해지므로 응답은 자연 실험으로 따로 확인한다.
- 관정별 실제 취수(계량) 기록은 비공개다. 위성 증발산·기상·수로 배분율로 추정하고 나머지는 보정한다. 우물 56개 중 신고 양수량이 있는 우물은 5개뿐이다.
- 연 0.5인치 기준은 이웃 관리기관(Wheeler Ridge-Maricopa) 값이다.
- 원본 자료(`raw/`, 약 270MB)와 OPERA 원본(.nc)은 git에서 뺐다. 받기 스크립트로 다시 받을 수 있다.

## 출처

- DWR TRE Altamira InSAR: https://data.cnra.ca.gov/dataset/tre-altamira-insar-subsidence
- NASA OPERA DISP-S1: https://www.jpl.nasa.gov/go/opera
- DWR 정기 지하수위: https://data.cnra.ca.gov/dataset/periodic-groundwater-level-measurements
- GPS(Nevada Geodetic Laboratory): https://geodesy.unr.edu
- 강수(NOAA GHCN-Daily): https://www.ncei.noaa.gov
- USGS CVHM2: Traum, Faunt, Boyce (2024), https://doi.org/10.5066/P14N8VFT ; Faunt 외 (2024), Water 16(8):1189
- SWP 반복구간·정밀측량점: https://data.cnra.ca.gov/dataset/i17-statewaterproject-repayment-reaches
- CalGEM 유전 경계: https://data.cnra.ca.gov/dataset/oil-and-gas-field-administrative-boundaries1
