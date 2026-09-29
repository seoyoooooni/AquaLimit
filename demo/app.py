import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from evaluate import subsidence, summarize
from optimizer import solve
from scenario import build_site, sample_responses, weekly_demand

st.set_page_config(page_title="AquaLimit 데모", layout="wide")
st.title("AquaLimit — 불확실성을 고려한 취수 계획 데모")
st.caption("모든 수치는 가상 데이터입니다. 실제 지역·위성 자료는 사용하지 않았습니다.")

with st.sidebar:
    st.header("입력")
    monthly_demand = st.slider("월 필요 수량 (톤)", 10000, 50000, 30000, 1000)
    limit_mm = st.slider("침하 관리기준 (mm/월)", 2.0, 15.0, 6.0, 0.5)
    purchase_cost = st.number_input("외부 용수 단가 (원/톤)", 500, 5000, 1500, 100)
    purchase_cap = st.number_input("외부 구매 한도 (톤/주)", 0, 10000, 3000, 500)
    st.header("불확실성")
    sigma = st.slider("땅속 특성 불확실성", 0.1, 0.6, 0.3, 0.05)
    n_plan = st.slider("계획에 쓰는 시나리오 수", 10, 200, 100, 10)
    n_test = 1000

site = build_site()
demand = weekly_demand(monthly_demand)
plan_scenarios = sample_responses(site, n_plan, sigma, seed=1)
test_scenarios = sample_responses(site, n_test, sigma, seed=2)

mean_plan = solve(site, demand, plan_scenarios.mean(axis=0, keepdims=True), limit_mm, purchase_cost, purchase_cap)
robust_plan = solve(site, demand, plan_scenarios, limit_mm, purchase_cost, purchase_cap)

st.subheader("1. 가상 지역")
fig_map = go.Figure()
fig_map.add_trace(go.Scatter(
    x=site.well_xy[:, 0], y=site.well_xy[:, 1], mode="markers+text",
    text=site.well_names, textposition="top center",
    marker=dict(size=16, symbol="circle"), name="우물",
))
fig_map.add_trace(go.Scatter(
    x=site.point_xy[:, 0], y=site.point_xy[:, 1], mode="markers+text",
    text=site.point_names, textposition="bottom center",
    marker=dict(size=14, symbol="triangle-up"), name="침하 관리지점",
))
fig_map.update_layout(height=360, xaxis_title="km", yaxis_title="km", margin=dict(t=20, b=20))
st.plotly_chart(fig_map, width="stretch")

wells = pd.DataFrame({
    "우물": site.well_names,
    "주간 용량 (톤)": site.well_capacity.astype(int),
    "취수 단가 (원/톤)": site.well_cost.astype(int),
})
st.dataframe(wells, hide_index=True, width="stretch")

st.subheader("2. 계획 비교")
st.write(
    f"평균 기반 계획은 땅속 특성의 평균값 하나만 사용합니다. "
    f"불확실성 반영 계획은 {n_plan}개 시나리오 모두에서 기준을 지키도록 계산합니다. "
    f"두 계획 모두 계획에 쓰지 않은 {n_test}개 시나리오로 평가합니다."
)

plans = {"평균 기반 계획": mean_plan, "불확실성 반영 계획": robust_plan}
colors = {"평균 기반 계획": "#E8743B", "불확실성 반영 계획": "#2E6FD8"}
summaries = {}
cols = st.columns(2)
for col, (name, plan) in zip(cols, plans.items()):
    with col:
        st.markdown(f"#### {name}")
        if not plan.feasible:
            st.error(f"실행 불가: {plan.reason}")
            continue
        s = summarize(plan, test_scenarios, limit_mm)
        summaries[name] = s
        m1, m2, m3 = st.columns([4, 3, 3])
        m1.metric("총비용", f"{plan.cost / 1e4:,.0f}만원")
        m2.metric("외부 구매", f"{plan.purchase.sum():,.0f}톤")
        m3.metric("기준 초과 확률", f"{s['exceed_rate'] * 100:.1f}%")

        monthly = plan.pumping.sum(axis=0)
        table = pd.DataFrame({
            "항목": site.well_names + ["외부 구매"],
            "월 취수·구매량 (톤)": np.append(monthly, plan.purchase.sum()).round(0).astype(int),
            "가동률": [f"{m / (c * len(demand)) * 100:.0f}%" for m, c in zip(monthly, site.well_capacity)] + ["-"],
        })
        st.dataframe(table, hide_index=True, width="stretch")

if summaries:
    st.subheader("3. 시나리오별 최대 침하")
    fig_hist = go.Figure()
    for name, s in summaries.items():
        fig_hist.add_trace(go.Histogram(x=s["worst_by_scenario"], name=name, opacity=0.6, nbinsx=40, marker_color=colors[name]))
    fig_hist.add_vline(x=limit_mm, line_dash="dash", annotation_text="관리기준")
    fig_hist.update_layout(
        barmode="overlay", height=380, xaxis_title="관리지점 중 최대 침하 (mm/월)",
        yaxis_title="시나리오 수", margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_hist, width="stretch")

    if robust_plan.feasible:
        s = subsidence(robust_plan, test_scenarios)
        point_table = pd.DataFrame({
            "관리지점": site.point_names,
            "침하 중앙값 (mm)": np.median(s, axis=0).round(2),
            "침하 95% 상한 (mm)": np.percentile(s, 95, axis=0).round(2),
        })
        st.markdown("불확실성 반영 계획의 관리지점별 예상 침하 범위")
        st.dataframe(point_table, hide_index=True, width="stretch")
