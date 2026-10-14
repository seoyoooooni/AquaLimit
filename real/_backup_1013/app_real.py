"""AquaLimit 실제 지역 데모 (California Aqueduct MP 195~215).

실행: cd real && streamlit run app_real.py
미리 계산한 결과를 읽어서 바로 계획을 세운다(MODFLOW를 다시 돌리지 않음).
  results/             역산 결과, GPS 검증 그림
  results_plan/R4.npz  MODFLOW 응답 행렬(E4 앙상블 30개)
  results_plan/R3.npz  MODFLOW 응답 행렬(E3) + 수위 여유
  results_ai/R_ai1000.npz  AI 대리모델 응답 행렬(앙상블 1,000개)
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 데모는 계획·AI 결과를 만든 첫 번째 모델(uniform) 기준으로 고정한다. 사용자가 AQ_PUMPING=et를 설정해 두었어도 덮어쓴다.
os.environ["AQ_PUMPING"] = "uniform"
import real_model as r  # noqa: E402
import plan_real as p  # noqa: E402

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="AquaLimit 실제 지역 데모", layout="wide")


@st.cache_data
def load():
    site = r.load_site()
    pts, wells = p.mgmt_points(site), p.operator_wells(site)
    PLAN, AI = r.result_dir("plan", create=False), r.result_dir("ai", create=False)
    r.check_version(PLAN); r.check_version(AI)
    z4 = np.load(PLAN / "R4.npz")
    R4 = [{"Rs": z4["Rs"][i], "Rh": z4["Rh"][i]} for i in range(len(z4["Rs"]))]
    z3 = np.load(PLAN / "R3.npz")
    R3 = {k: z3[k] for k in z3.files}
    Rai = None
    oos = json.load(open(PLAN / "oos.json")) if (PLAN / "oos.json").exists() else None
    f = AI / "R_ai1000.npz"
    if f.exists():
        za = np.load(f); Rai = [{"Rs": za["Rs"][i].astype(float), "Rh": za["Rh"][i].astype(float)} for i in range(len(za["Rs"]))]
    vel = pd.read_csv(HERE / "data" / "insar_points_velocity.csv").sample(15000, random_state=0)
    oil = json.load(open(HERE / "data" / "oil_fields_in_box.geojson"))
    cells_ll = lambda ij: (r.LAT0 + (ij[0] + 0.5) * r.MCELL / 1000 / r.KM_LAT, r.LON0 + (ij[1] + 0.5) * r.MCELL / 1000 / r.KM_LON)
    return dict(site_aq=site["aq_pts"], pts=[cells_ll(q) for q in pts], wells=[cells_ll(w) for w in wells],
                R4=R4, R3=R3, Rai=Rai, vel=vel, oil=oil, oos=oos)


D = load()
MAP = getattr(go, "Scattermap", None) or go.Scattermapbox          # plotly 6: Scattermap, 이전: Scattermapbox
MAPKEY = "map" if hasattr(go, "Scattermap") else "mapbox"
W = len(p.WELL_COST)

st.title("AquaLimit — 실제 지역 취수 계획")
st.caption("California Aqueduct MP 195~215 (Lost Hills 주변). 땅속 성질은 위성(InSAR)·관측정·USGS CVHM2로 역산했고 GPS로 검증했다. "
           "사업자 우물 6개의 위치·용량·비용은 설명용 가정이다.")

with st.sidebar:
    st.header("입력 (2026년 5~7월, 3개월)")
    demand = st.slider("필요 수량 (천 m³)", 300, 2000, 900, 50) * 1000.0
    limit = st.slider("수로 추가 침하 기준 (mm/3개월)", 0.5, 4.0, 2.0, 0.1)
    price = st.number_input("외부 용수 단가 ($/m³)", 0.2, 2.0, 0.81, 0.01)
    pcap = st.number_input("외부 구매 한도 (천 m³)", 0, 2000, 300, 50) * 1000.0
    st.header("불확실성")
    mode = st.radio("계획에 쓰는 모델", ["E3 추정값 1개", "E4 MODFLOW 앙상블 30개", "E4 AI 앙상블 1,000개"],
                    index=2 if D["Rai"] else 1)
    use_head = st.checkbox("임계수위(과거 최저 수위) 유지", value=False)
    ai_margin = 0.0
    if mode.startswith("E4 AI"):
        try:
            mg = json.load(open(r.result_dir("ai", create=False) / "results.json")).get("ai_margin_mm") or 0.0
        except FileNotFoundError:
            mg = 0.0
        if st.checkbox(f"AI 오차 여유 두기 ({mg:.2f} mm)", value=True,
                       help="AI 대리모델 오차만큼 기준을 낮춰 계획한다. 끄면 MODFLOW 확인에서 기준을 조금 넘을 수 있다."):
            ai_margin = mg

p.DEMAND, p.LIMIT_MM, p.PRICE, p.PURCHASE_CAP = demand, limit - ai_margin, price, pcap
R_plan = {"E3 추정값 1개": [D["R3"]], "E4 MODFLOW 앙상블 30개": D["R4"], "E4 AI 앙상블 1,000개": D["Rai"] or D["R4"]}[mode]
if use_head:   # 수위 여유는 E3 모델 값 사용
    R_plan = [dict(R, base_head_end=D["R3"]["base_head_end"], hist_min=D["R3"]["hist_min"]) for R in R_plan]
plan = p.solve(R_plan, use_head=use_head)
base = p.baseline()

c1, c2 = st.columns([1.1, 1])
with c1:
    st.subheader("1. 지역 지도")
    v = D["vel"]
    fig = go.Figure()
    fig.add_trace(MAP(lat=v.LAT, lon=v.LON, mode="markers", name="InSAR 침하 속도(mm/년)",
                                   marker=dict(size=3, color=v.VEL_V, colorscale="RdBu", cmin=-60, cmax=60,
                                               colorbar=dict(title="mm/년", x=1.0))))
    for f in D["oil"]["features"]:
        g = f["geometry"]; polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ring = np.array(poly[0])
            fig.add_trace(MAP(lat=ring[:, 1], lon=ring[:, 0], mode="lines", line=dict(color="#555", width=1),
                                           name="유전", showlegend=False, hoverinfo="skip"))
    aq = np.array(D["site_aq"])
    fig.add_trace(MAP(lat=aq[:, 1], lon=aq[:, 0], mode="markers", marker=dict(size=4, color="black"), name="California Aqueduct"))
    pts = np.array(D["pts"]); wl = np.array(D["wells"])
    fig.add_trace(MAP(lat=pts[:, 0], lon=pts[:, 1], mode="markers", marker=dict(size=7, color="#2166ac"), name="관리지점"))
    fig.add_trace(MAP(lat=wl[:, 0], lon=wl[:, 1], mode="markers+text", text=[chr(65 + i) for i in range(W)],
                                   textposition="top right", marker=dict(size=14, color="#b2182b"), name="사업자 우물(가정)"))
    fig.update_layout(**{MAPKEY: dict(style="open-street-map", center=dict(lat=35.64, lon=-119.75), zoom=9.2)},
                      height=520, margin=dict(t=0, b=0, l=0, r=0), legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.7)"))
    st.plotly_chart(fig, width="stretch")

with c2:
    st.subheader("2. 계획")
    if not plan["feasible"]:
        st.error(f"실행 불가 — {plan['reason']}")
        p.PURCHASE_CAP = demand
        need = p.solve(R_plan, use_head=use_head)
        p.PURCHASE_CAP = pcap
        if need["feasible"]:
            st.info(f"기준을 지키려면 외부 용수 **{need['purchase'] / 1e3:,.0f}천 m³**가 필요합니다(비용 {need['cost'] / 1e3:,.0f}천$).")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("총비용", f"{plan['cost'] / 1e3:,.0f}천$", f"{(plan['cost'] - base['cost']) / 1e3:+,.0f}천$ (기존 방식 대비)", delta_color="inverse")
        m2.metric("외부 구매", f"{plan['purchase'] / 1e3:,.0f}천 m³")
        p.LIMIT_MM = limit                                            # 평가는 원래 기준으로
        rk = p.risk(plan, D["R4"])
        n4 = len(D["R4"]); k4 = int(round(rk["exceed_prob"] * n4))
        m3.metric("기존 30개 시나리오 내 초과 비율", f"{k4}/{n4}",
                  help="E4 MODFLOW 앙상블 30개로 잰 값. 'E4 MODFLOW 앙상블 30개'로 계획하면 바로 이 30개에서 기준을 지키도록 만들기 때문에 0/30은 정의상 나온다.")
        tbl = pd.DataFrame({"우물": [chr(65 + i) for i in range(W)], "비용($/m³)": p.WELL_COST,
                            "계획 취수(m³/일)": np.round(plan["rates"]).astype(int),
                            "기존 방식(m³/일)": np.round(base["rates"]).astype(int)})
        st.dataframe(tbl, hide_index=True, width="stretch")
        rb = p.risk(base, D["R4"])
        st.caption(f"기존 방식(용량 비례): 비용 {base['cost'] / 1e3:,.0f}천$, 기존 30개 시나리오 내 초과 {int(round(rb['exceed_prob'] * n4))}/{n4}")
        if mode.startswith("E4 MODFLOW"):
            st.warning("이 계획은 위 30개 시나리오 모두에서 기준을 지키도록 만든 것이라, 같은 30개로 잰 0/30은 실제 위험이 0%라는 뜻이 아니다.")
        o = D.get("oos")
        if o and "E4" in o["plans"]:
            e4 = o["plans"]["E4"]["oos"]; n = o["n_new_members"]
            c = lambda x: int(round(x * n))
            st.info(f"참고 — 기본 조건(수요 900천 m³, 기준 2mm)의 E4 계획을 **계획에 안 쓴 {n}개**로 다시 잰 값: "
                    f"평년 {c(e4['평년']['x1.0'])}/{n}, 가뭄 해 {c(e4['가뭄 해']['x1.0'])}/{n}, "
                    f"실제 땅이 25% 더 반응하면 평년 {c(e4['평년']['x1.25'])}/{n}. "
                    f"0/{n}도 위험이 0이라는 증거는 아니다(통계적으로 대략 10% 미만). results_plan/RESULTS_oos.md")

st.subheader("3. 모델은 믿을 만한가? — 보정에 쓰지 않은 GPS로 검증")
st.image(str(r.result_dir("model", create=False) / "real-3-gps-validation.png"), width="stretch")
st.caption("2015.7~2022.12 위성·관측정으로 보정하고, 2023.1~2026.4 GPS(검정 점)로 채점. 위성을 넣은 E3의 GPS 오차는 관측정만 쓴 E2보다 크게 작다(results/RESULTS.md).")
