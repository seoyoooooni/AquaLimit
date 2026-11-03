"""⑥ 전체 발표용 실험 한눈표. 저장된 결과만 읽고 모델을 다시 실행하지 않는다."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read_json(path):
    path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# 각 줄은 서로 다른 질문에 대한 실험이다. 지표의 분모와 기간이 다르므로 한 순위로 합치지 않는다.
EXPERIMENTS = [
    ("가상 검증", "해석해 참값 10회", "위성 관측이 역산과 예측에 도움이 되는가?",
     "침하 예측 오차 E2 11.86 → E3 3.10mm; E4 기준 초과 0/10", "가상 자료", "synthetic/results/RESULTS.md"),
    ("가상 검증", "MODFLOW 참값 10회", "더 복잡한 땅에서도 같은 결론인가?",
     "침하 예측 5.14 → 3.19mm; E4도 기준 초과 4/10", "가상 자료", "synthetic/results_mf6/RESULTS.md"),
    ("가상 검증", "실행 불가 입력", "수요·침하 조건을 함께 못 지키면 알아채는가?",
     "침하 제약과 공급 용량 초과를 각각 실행 불가로 판정", "기능 시험", "synthetic/results_mf6/RESULTS.md"),
    ("실제 자료", "초기 모델", "공개 자료로 실제 지역 모델을 만들 수 있는가?",
     "균일 취수 가정 모델도 GPS 오차 20.5mm까지 검증; 이후 입력·구조 개선", "독립 GPS", "real/results_uniform/RESULTS.md"),
    ("실제 자료", "v3 최종 시험", "위성을 추가하면 처음 보는 GPS를 더 잘 맞히는가?",
     "2023~26 GPS 오차 E2 70.9 → E3 21.2mm; 추세 연장 7.2mm가 더 정확", "독립 GPS", "real/results_v3/RESULTS.md"),
    ("실제 자료", "입력 자료 제거", "Landsat·기상·수로 배분율이 예측에 주는 영향은?",
     "각 입력을 뺀 모델을 같은 GPS로 재채점(아래 세부표)", "독립 GPS", "real/SCOREBOARD.md"),
    ("예측 검증", "오차 범위 되돌아보기", "90% 예측 범위가 실제 GPS를 포함하는가?",
     "파라미터만 반영하면 29%; 과거 오차를 더하면 99%(범위 폭 97mm)", "독립 GPS", "real/results_v3/RESULTS.md"),
    ("모델 선택", "구조·보정 후보 17개", "구조를 바꾸면 2020~22 수로 예측이 개선되는가?",
     "OPERA 연습 시험: v3 14.0mm, 17개 후보 모두 미달", "연습 시험", "real/SCOREBOARD.md"),
    ("모델 선택", "연습 시험 정답 교체", "DWR와 OPERA 중 어떤 위성으로 채점해야 하는가?",
     "P544 GPS와 대조해 OPERA로 재채점; DWR 점수는 기록만 유지", "관측 비교", "real/RESCORE_PRACTICE.md"),
    ("자료 점검", "위성 ↔ GPS 교차검증", "어느 위성 자료를 채점에 쓸 수 있는가?",
     "2023~26 침하 방향 일치 OPERA 5/5, DWR 3/5", "관측 비교", "real/results_opera/RESULTS.md"),
    ("응답 검증", "습한 해·취수 금지 자연 실험", "수로 물·취수가 변할 때 침하 반응을 맞히는가?",
     "v3: 습한 해 감속은 일치, 금지 후 3곳은 불일치(총 1/4)", "독립 GPS", "real/results_validate_v3/RESULTS_annual.md"),
    ("자료 점검", "금지 전후·위약 검사", "위성에서 취수 금지 효과가 보이는가?",
     "금지 후 안−밖 +2.9mm/년; 과거 위약 검사 후 효과 확인 불가", "관측 비교", "real/results_opera/RESULTS_checks.md"),
    ("정책 계산", "현재 4km 금지", "기존 취수 금지만으로 침하 기준을 지키는가?",
     "평년 수로 지점 45%, 가뭄 32%만 연 12.7mm 기준 충족", "모델 시뮬레이션", "real/results_policy_v3ops/RESULTS.md"),
    ("정책 계산", "금지 반경과 비용 비교", "같은 침하 기준을 더 싸게 지킬 수 있는가?",
     "MODFLOW 100% 충족 조건: 고정 6.9km 금지 대비 비용 10~29% 감소", "모델 시뮬레이션", "real/results_policy_v3ops/RESULTS_refine.md"),
    ("계획 계산", "새 땅속 상태 30개", "계획에 안 쓴 상태에서도 기준을 지키는가?",
     "평년 기준 초과 E1 47% → E4 7%; 응답 1.5배면 E4도 47%", "모델 시뮬레이션", "real/results_plan_v3ops/RESULTS_oos.md"),
    ("속도 검증", "위치 입력 AI ↔ MODFLOW", "우물 위치가 바뀌어도 응답을 빠르게 계산하는가?",
     "새 위치·상태 R² 0.92; AI 계획을 MODFLOW 30개로 재확인", "모델 시뮬레이션", "real/results_ai_v3ops/RESULTS_loc.md"),
]


OLDER = {
    "v4": "초기 구조 변경", "v4-aq": "수로 띠 압축", "v4-corc": "코코란 점토",
    "v4-emult": "증발산 배수", "v4-kv9": "수직 투수계수 9구역", "v4-pfrac": "취수 비율",
}
RECENT = {
    "v3-opera": "OPERA로 보정", "v4-oil": "유전 침하", "v4-oil-hw": "유전 + 수위 가중치",
    "v4-oil-aq": "유전 + 수로 띠", "v4-oil-opera": "유전 + OPERA", "v4-pp": "파일럿 포인트",
    "v4-oil-pp": "유전 + 파일럿", "v4-pp-opera": "파일럿 + OPERA",
    "v4-oil-pp-opera": "유전 + 파일럿 + OPERA", "v4-sw2": "수로물 안·밖",
    "v4-oil-pp-sw2": "유전 + 파일럿 + 수로물",
}


@st.cache_data
def model_scores():
    rows = []
    for key in ("v3", *OLDER, *RECENT):
        result = read_json(f"real/results_{key}/results_dev_rescore.json")
        if not result:
            continue
        near = result["opera"]["near"]
        rows.append({
            "구분": "기준" if key == "v3" else "이전 6개" if key in OLDER else "최근 11개",
            "후보": key,
            "바꾼 것": "현재 모델" if key == "v3" else (OLDER | RECENT)[key],
            "수로 2km 오차 (mm)": round(near["model"], 1),
            "편향 (mm)": round(near["bias_model"], 1),
            "GPS P544 오차 (mm)": round(result["p544"]["model"], 1),
        })
    return pd.DataFrame(rows)


@st.cache_data
def input_scores():
    names = {"v3": "전체 입력", "v3_nolandsat": "Landsat 제거", "v3_noweather": "기상 제거", "v3_noalloc": "수로 배분율 제거"}
    rows = []
    for key, name in names.items():
        result = read_json(f"real/results_{key}/results_e3only.json") or read_json(f"real/results_{key}/results.json")
        if result and result.get("E3", {}).get("val", {}).get("gps_rmse") is not None:
            rows.append({"입력": name, "최종 GPS 오차 (mm)": round(result["E3"]["val"]["gps_rmse"], 1)})
    return pd.DataFrame(rows)


def render():
    st.markdown("#### 전체 실험 · 무엇을 시험했고 무엇을 알았나")
    st.caption("가상 자료 → 실제 관측 검증 → 모델 선택 → 정책·계획 시뮬레이션. 수치의 기간·대상이 달라 행 사이 점수 순위는 매기지 않습니다.")
    overview = pd.DataFrame(EXPERIMENTS, columns=["단계", "실험", "확인한 질문", "핵심 결과", "근거", "결과 파일"])
    st.dataframe(overview[["단계", "실험", "핵심 결과", "근거"]], hide_index=True, width="stretch", height=640,
                 column_config={"실험": st.column_config.TextColumn(width="medium"),
                                "핵심 결과": st.column_config.TextColumn(width="large")})
    st.info("발표 요약: 위성은 지하 모델의 GPS 예측을 개선했지만, 3년 누적 침하는 단순 추세보다 아직 부정확합니다. "
            "모델 후보 17개를 연습 시험으로 거르고, 실제 정책·계획 효과는 MODFLOW 시뮬레이션과 표본 밖 시험으로 확인했습니다.")

    with st.expander("모델 후보 17개 · 연습 시험 전체 점수"):
        st.caption("2015~19 보정 → 2020~22 NASA OPERA로 채점. 수로 2km 오차가 작을수록 좋습니다. "
                   "최근 11개 후보의 실험 과정과 그림은 ⑤ 탭에 있습니다.")
        scores = model_scores()
        if scores.empty:
            st.warning("저장된 연습 시험 결과가 없습니다.")
        else:
            st.dataframe(scores.sort_values("수로 2km 오차 (mm)"), hide_index=True, width="stretch")
            st.caption("비교 기준: 위성 추세 연장 10.4mm. 후보 17개는 기준 모델 v3(14.0mm)를 넘지 못했습니다. "
                       "출처: real/SCOREBOARD.md, real/results_*/results_dev_rescore.json")

    with st.expander("입력 자료 제거 · 독립 GPS 점수"):
        st.caption("2022년까지 보정한 모델을 2023~26 수로 GPS로 평가한 별도 시험입니다. "
                   "연습 시험의 OPERA 점수와 직접 비교하지 않습니다.")
        scores = input_scores()
        if scores.empty:
            st.warning("저장된 입력별 시험 결과가 없습니다.")
        else:
            st.dataframe(scores, hide_index=True, width="stretch")
            st.caption("출처: real/SCOREBOARD.md, real/results_v3_*/results_e3only.json")

    with st.expander("실험별 질문과 결과 파일"):
        st.dataframe(overview[["실험", "확인한 질문", "결과 파일"]], hide_index=True, width="stretch")

    st.caption("가상 대수층 성능은 실제 지역 성능이 아니며, 정책·취수계획·AI 결과는 모델 시뮬레이션입니다. "
               "관측으로 확인한 GPS·위성 시험과 구분해서 해석해야 합니다.")
