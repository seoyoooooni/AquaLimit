"""AquaLimit 최종 데모 (California Aqueduct MP 195~215, 운영 모델 v3ops).

실행: cd real && streamlit run app_final.py
3분 시연 흐름: ① 문제(지도·금지 후에도 침하) → ② 관리기관: 전면 금지 vs AquaLimit → ③ 사업자: 지도에서 우물 고르기 → 계획(AI)
→ ④ 믿을 만한가(보정에 안 쓴 GPS, 자연 실험)
미리 계산한 결과를 읽는다(MODFLOW를 다시 돌리지 않음). 위치 입력 AI(results_ai_v3ops/ai_loc.pt)로 우물 위치를 바꿔도 바로 계산한다.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

os.environ.setdefault("AQ_VARIANT", "v3ops")
os.environ.setdefault("AQ_PUMPING", "et")
import real_model as r  # noqa: E402
import plan_real as p  # noqa: E402

HERE = Path(__file__).resolve().parent
VAL = HERE / "results_v3"                     # 검증 모델(예측 성능)
OPS = r.result_dir("model", create=False)     # 운영 모델
POL = HERE / f"results_policy_{r.VARIANT}"
PLAN = r.result_dir("plan", create=False)
AI = r.result_dir("ai", create=False)
VALR = HERE / "results_validate_v3"
VALO = HERE / f"results_validate_{r.VARIANT}"
REG = 12.7
st.set_page_config(page_title="AquaLimit", layout="wide")
MAP = getattr(go, "Scattermap", None) or go.Scattermapbox
MAPKEY = "map" if hasattr(go, "Scattermap") else "mapbox"


def jload(f):
    f = Path(f)
    return json.load(open(f)) if f.exists() else None


@st.cache_resource
def load():
    site = r.load_site()
    pts = p.mgmt_points(site)
    wells = p.operator_wells(site)                                      # 실제 신고 우물(금지구역 밖) 6개, CAP·WELL_COST 설정
    import policy_real as pol
    import validate_response as V
    dist = V.aq_dist_grid(site)
    ll = lambda iy, ix: (r.LAT0 + (iy + 0.5) * r.MCELL / 1000 / r.KM_LAT, r.LON0 + (ix + 0.5) * r.MCELL / 1000 / r.KM_LON)
    ag = np.argwhere(site["ag"])
    wcr = pol.load_wells(site)
    wcr_cap = {}
    for _, w in wcr.iterrows():
        wcr_cap[(int(w.iy), int(w.ix))] = wcr_cap.get((int(w.iy), int(w.ix)), 0) + w.yield_gpm * pol.GPM_TO_M3D * pol.DUTY
    res = jload(OPS / "results.json")
    th3, cov3 = np.array(res["params"]["E3"]), np.array(res["params"]["E3_cov"])
    rng = np.random.default_rng(1)
    L = np.linalg.cholesky(cov3 + 1e-12 * np.eye(len(th3)))
    M_big = np.clip(th3 + rng.normal(size=(1000, len(th3))) @ L.T, r.PRIOR_MEAN - 3 * r.PRIOR_SD, r.PRIOR_MEAN + 3 * r.PRIOR_SD)
    net = None
    if (AI / "ai_loc.pt").exists():
        from surrogate_real import Net
        net = Net.load(AI / "ai_loc.pt")
    vel = pd.read_csv(HERE / "data" / "insar_points_velocity.csv").sample(12000, random_state=0)
    oil = json.load(open(HERE / "data" / "oil_fields_in_box.geojson"))
    gps = site["gps"].groupby("station").agg(lat=("lat", "first"), lon=("lon", "first")).reset_index()
    obs = V.gps_rates(site)
    gps["rate_post"] = [obs[s][list(V.WINDOWS)[2]] for s in gps.station]
    return dict(site_aq=site["aq_pts"], pts=pts, pts_ll=[ll(*q) for q in pts], wells=wells, ag=ag,
                ag_ll=np.array([ll(*c) for c in ag]), ag_dist=np.array([dist[a, b] for a, b in ag]),
                buffer_ll=np.array([ll(*c) for c in ag if dist[c[0], c[1]] <= pol.BUFFER_KM]),
                wcr_cap=wcr_cap, M_big=M_big, net=net, vel=vel, oil=oil, gps=gps, cap0=p.CAP.copy(), cost0=p.WELL_COST.copy(),
                feats=lambda cells: __import__("surrogate_loc").loc_features(site, cells),
                default_cap=2000 * pol.GPM_TO_M3D * pol.DUTY,
                default_cost=float(np.median(p.WELL_COST)))


D = load()
st.title("AquaLimit — 위성으로 지하수 취수를 처방한다")
st.caption("California Aqueduct MP 195~215 (Lost Hills, 미국 캘리포니아). 위성 InSAR·Landsat·기상·관측정·USGS 지층모델로 보정한 "
           "MODFLOW 6 지하수·침하 모델(불확실성 앙상블)과 위치 입력 AI. 수로 기준: 연 0.5인치(12.7mm, 이웃 관리기관 실제 기준).")
t1, t2, t3, t4 = st.tabs(["① 문제", "② 관리기관: 전면 금지 vs AquaLimit", "③ 사업자: 우물별 계획", "④ 믿을 만한가"])


def base_map(height=520):
    fig = go.Figure()
    v = D["vel"]
    fig.add_trace(MAP(lat=v.LAT, lon=v.LON, mode="markers", name="InSAR 침하 속도 (mm/년, 2015~2022)",
                      marker=dict(size=3, color=v.VEL_V, colorscale="RdBu", cmin=-60, cmax=60, colorbar=dict(title="mm/년", x=1.0))))
    for f in D["oil"]["features"]:
        g = f["geometry"]; polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ring = np.array(poly[0])
            fig.add_trace(MAP(lat=ring[:, 1], lon=ring[:, 0], mode="lines", line=dict(color="#555", width=1), showlegend=False, hoverinfo="skip"))
    b = D["buffer_ll"]
    if len(b):
        fig.add_trace(MAP(lat=b[:, 0], lon=b[:, 1], mode="markers", marker=dict(size=9, color="rgba(214,96,77,0.25)"),
                          name="취수 금지구역(2024.12~, 수로 양쪽 2.5마일)", hoverinfo="skip"))
    aq = np.array(D["site_aq"])
    fig.add_trace(MAP(lat=aq[:, 1], lon=aq[:, 0], mode="markers", marker=dict(size=4, color="black"), name="California Aqueduct"))
    g = D["gps"]
    fig.add_trace(MAP(lat=g.lat, lon=g.lon, mode="markers+text", text=[f"{s} {x:.0f}mm/년" if np.isfinite(x) else s for s, x in zip(g.station, g.rate_post)],
                      textposition="middle left", marker=dict(size=12, color="#ffd400"), name="GPS (금지 후 침하 속도)"))
    fig.update_layout(**{MAPKEY: dict(style="open-street-map", center=dict(lat=35.64, lon=-119.75), zoom=9.2)},
                      height=height, margin=dict(t=0, b=0, l=0, r=0), legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.75)"))
    return fig


with t1:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.plotly_chart(base_map(), width="stretch")
    with c2:
        st.markdown("#### 주정부 수로가 지하수 때문에 꺼지고 있다")
        g = D["gps"].dropna(subset=["rate_post"])
        st.metric("수로 GPS 침하 속도 (2024.12~2026.4)", f"연 {g.rate_post.min():.0f}~{g.rate_post.max():.0f}mm",
                  f"기준(연 {REG}mm)의 최대 {g.rate_post.max() / REG:.1f}배", delta_color="inverse")
        st.markdown("- 가뭄 해에 수로 물이 줄면(2021·2022년 배분율 5%) 농장이 지하수를 뽑는다 → 땅이 꺼진다 → 수로가 휘어 물을 덜 보낸다.\n"
                    "- 관리기관은 **2024.12.9부터 수로 양쪽 2.5마일 취수를 전면 금지**했다(위반 시 $500/AF).\n"
                    "- **그런데 금지 뒤에도 GPS는 계속, 오히려 더 빨리 꺼지고 있다.** 원인을 모르면 다음 조치도 정할 수 없다.")
        img = VALO / "validate-2-ban.png"
        if img.exists():
            st.image(str(img), width="stretch")
            dec = (jload(VALO / "results.json") or {}).get("decomposition", {}).get("aqueduct")
            if dec:
                tot = max(dec["ban"]["med"], 1e-9)
                st.info(f"AquaLimit 원인 분석(운영 모델): 금지를 지켜도 수로 침하의 **{dec['legacy']['med'] / tot * 100:.0f}%는 과거 취수로 이미 진행 중인 침하**, "
                        f"**{dec['outside']['med'] / tot * 100:.0f}%는 금지구역 밖 취수**다.")

with t2:
    o = jload(POL / "results.json") or {}
    blocks = {}
    if (POL / "results_retro.json").exists():
        blocks["2021~22 대가뭄 재현"] = jload(POL / "results_retro.json")
    for k, v in (jload(POL / "results_fwd.json") or {}).items():
        blocks[f"다음 관개기 2026.5~7: {k}"] = v
    if not blocks:
        st.warning("정책 비교 결과가 없습니다: AQ_VARIANT=v3ops python3 policy_real.py")
    else:
        which = st.selectbox("상황", list(blocks))
        b = blocks[which]
        rows = []
        for k, v in b["policies"].items():
            if v.get("feasible"):
                rows.append({"정책": k, "대체 용수 (백만 m³)": round(v["replace_m3"] / 1e6, 2), "비용 (백만 $)": round(v["cost"] / 1e6, 2),
                             "수로 최대 침하 (mm, MODFLOW 확인)": round(v["ver_sub_med"], 1),
                             "기준 지키는 수로 지점": f"{v.get('reg_ok_points', np.nan) * 100:.0f}%",
                             "모든 지점 기준 확률": f"{v.get('reg_ok_prob', np.nan) * 100:.0f}%"})
            else:
                rows.append({"정책": k, "대체 용수 (백만 m³)": None, "비용 (백만 $)": None, "수로 최대 침하 (mm, MODFLOW 확인)": None,
                             "기준 지키는 수로 지점": "실행 불가: " + v.get("reason", "")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        ban = b["policies"].get("4km 전면 금지(2025 실제 정책)", {})
        aqr = b["policies"].get("AquaLimit(규제 기준 0.5인치/년)", {})
        aqb = b["policies"].get("AquaLimit(금지와 같은 예산)", {})
        m0, m1, m2 = st.columns(3)
        if ban.get("feasible"):
            m0.metric("전면 금지: 기준 지키는 수로 지점", f"{ban.get('reg_ok_points', 0) * 100:.0f}%", f"비용 ${ban['cost'] / 1e6:.1f}M", delta_color="off")
        if ban.get("feasible") and aqb.get("feasible"):
            m1.metric("같은 예산, AquaLimit 배분", f"{aqb.get('reg_ok_points', 0) * 100:.0f}%",
                      f"{(aqb.get('reg_ok_points', 0) - ban.get('reg_ok_points', 0)) * 100:+.0f}%p vs 전면 금지")
        if ban.get("feasible") and aqr.get("feasible"):
            m2.metric("모든 지점 기준 달성에 필요한 비용", f"${aqr['cost'] / 1e6:.1f}M", f"{(aqr['cost'] - ban['cost']) / 1e6:+.1f}M vs 전면 금지", delta_color="off")
        st.info("전면 금지는 같은 보호 수준에서는 효율적인 편이지만, 평년·가뭄에는 수로 기준을 다 지키지 못한다. "
                "AquaLimit은 기준을 지키는 데 필요한 추가 조치(금지구역 밖 우물별 감량)를 최소 비용으로 정한다.")
        cv = [c for c in b["curve"] if c["feasible"]]
        if cv:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[c["protect_vs_ban"] for c in cv], y=[c["cost"] / 1e6 for c in cv], mode="lines+markers", name="AquaLimit"))
            if ban.get("feasible"):
                fig.add_trace(go.Scatter(x=[1.0], y=[ban["cost"] / 1e6], mode="markers", marker=dict(size=14, symbol="square", color="#d6604d"), name="4km 전면 금지"))
            fig.update_layout(xaxis_title="수로 보호 수준 (전면 금지 = 1)", yaxis_title="비용 (백만 $)", height=340, margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
        st.caption("비용 = 우물 양수 전력·유지비 + 대체 용수(외부 구매·휴경 보상 $0.81/m³ ≈ $1,000/AF). 모든 정책은 MODFLOW 앙상블로 다시 확인했다. "
                   "우물: DWR 신고 농업용 우물 56개(12개 묶음).")

with t3:
    st.markdown("#### 지도에서 우물을 고르면(드래그·클릭으로 선택) 땅속 상태 1,000개에서 수로 기준을 지키는 우물별 계획을 바로 계산한다")
    if D["net"] is None:
        st.warning("위치 입력 AI 결과가 없습니다: AQ_VARIANT=v3ops python3 surrogate_loc.py")
    else:
        with st.sidebar:
            st.header("③ 사업자 계획 입력 (3개월)")
            limit = st.slider("수로 추가 침하 한도 (mm/3개월)", 0.5, 4.0, 2.0, 0.1, help="연 0.5인치 기준(3개월 3.2mm) 중 이 사업자 몫")
            dem_frac = st.slider("필요 수량 (우물 용량 합 대비 %)", 20, 120, 75, 5)
            price = st.number_input("외부 용수 단가 ($/m³)", 0.2, 2.0, 0.81, 0.01)
            margin = st.select_slider("구조 오차 여유", options=[0.0, 0.25, 0.5, 1.0], value=0.25,
                                      format_func=lambda x: f"{x * 100:.0f}% (실제 땅이 모델보다 {x * 100:.0f}% 더 반응해도 안전)")
        fig = go.Figure()
        agl = D["ag_ll"]
        fig.add_trace(MAP(lat=agl[:, 0], lon=agl[:, 1], mode="markers", marker=dict(size=9, color="rgba(44,160,44,0.35)"),
                          name="농경지 칸(선택 가능)", customdata=np.arange(len(agl)),
                          text=[f"수로까지 {d:.1f}km" for d in D["ag_dist"]], hoverinfo="text"))
        aq = np.array(D["site_aq"])
        fig.add_trace(MAP(lat=aq[:, 1], lon=aq[:, 0], mode="markers", marker=dict(size=4, color="black"), name="수로"))
        fig.update_layout(**{MAPKEY: dict(style="open-street-map", center=dict(lat=35.64, lon=-119.72), zoom=9.4)},
                          height=460, margin=dict(t=0, b=0, l=0, r=0), dragmode="lasso", legend=dict(x=0, y=1))
        ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=("points", "box", "lasso"), key="agmap")
        sel = [pt.get("customdata") for pt in (ev.selection.points if ev and ev.selection else []) if pt.get("curve_number", 0) == 0]
        sel = [int(s if not isinstance(s, (list, tuple)) else s[0]) for s in sel if s is not None]
        wcr_opts = {f"신고 우물 칸 ({c[0]},{c[1]}) · 수로까지 {D['ag_dist'][[tuple(a) for a in D['ag']].index(c)]:.1f}km": c
                    for c in D["wcr_cap"] if c in set(map(tuple, D["ag"]))}
        pick = st.multiselect("또는 실제 신고 우물에서 고르기 (지도 선택이 없을 때 사용)", list(wcr_opts), max_selections=12)
        if sel:
            cells = [tuple(D["ag"][i]) for i in sel][:12]
            src = "지도에서 고른 칸"
        elif pick:
            cells = [wcr_opts[k] for k in pick]
            src = "목록에서 고른 신고 우물"
        else:
            cells = [tuple(w) for w in D["wells"]]
            src = "기본값: 실제 신고 우물 6개(금지구역 밖)"
        caps = np.array([D["wcr_cap"].get(c, D["default_cap"]) for c in cells])
        costs = np.array([D["cost0"][[tuple(w) for w in D["wells"]].index(c)] if c in [tuple(w) for w in D["wells"]] else D["default_cost"] for c in cells])
        F = D["feats"](cells)
        X = np.c_[np.repeat((D["M_big"] - r.PRIOR_MEAN) / r.PRIOR_SD, len(cells), 0), np.tile(F, (len(D["M_big"]), 1))]
        import time as _t
        t0 = _t.time()
        Y = D["net"].predict(X)
        dt = _t.time() - t0
        P = len(D["pts"])
        Y = Y.reshape(len(D["M_big"]), len(cells), -1)
        R = [{"Rs": Y[i, :, :P].T / 1000.0, "Rh": Y[i, :, P:].T / 1000.0} for i in range(len(D["M_big"]))]
        p.CAP, p.WELL_COST, p.PRICE, p.LIMIT_MM = caps, costs, price, limit
        p.DEMAND = float(dem_frac / 100 * caps.sum() * p.DAYS); p.PURCHASE_CAP = p.DEMAND
        mg = (jload(AI / "results_loc.json") or {}).get("ai_margin_mm") or 0.0
        p.LIMIT_MM = limit - mg
        plan = p.solve(R, inflate=margin)
        p.LIMIT_MM = limit
        base = p.baseline()
        st.caption(f"{src} · AI 계산 {len(cells)}개 우물 × 땅속 상태 1,000개 = {dt * 1000:.0f}ms (MODFLOW였다면 약 {(len(cells) + 1) * 1000 * 15 / 3600:.0f}시간) · AI 오차 여유 {mg:.2f}mm 적용")
        if not plan["feasible"]:
            st.error(plan["reason"])
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("총비용", f"${plan['cost'] / 1e3:,.0f}천", f"{(plan['cost'] - base['cost']) / 1e3:+,.0f}천 vs 용량 비례 배분", delta_color="inverse")
            m2.metric("외부 구매", f"{plan['purchase'] / 1e3:,.0f}천 m³")
            rk = p.risk(plan, R); rb = p.risk(base, R)
            m3.metric("기준 초과 확률 (땅속 상태 1,000개)", f"{rk['exceed_prob'] * 100:.1f}%", f"용량 비례 배분은 {rb['exceed_prob'] * 100:.0f}%", delta_color="off")
            st.dataframe(pd.DataFrame({"우물": [chr(65 + i) for i in range(len(cells))],
                                       "수로까지 (km)": [round(float(D["ag_dist"][[tuple(a) for a in D["ag"]].index(c)]), 1) for c in cells],
                                       "용량 (m³/일)": np.round(caps).astype(int), "양수 비용 ($/m³)": np.round(costs, 3),
                                       "AquaLimit 계획 (m³/일)": np.round(plan["rates"]).astype(int),
                                       "용량 비례 (m³/일)": np.round(base["rates"]).astype(int)}), hide_index=True, width="stretch")
            st.caption("수로에 가깝고 점토가 두꺼운 칸의 우물은 줄이고, 영향이 작은 우물로 옮긴다. 최종 계획은 MODFLOW로 다시 확인한다(결과: results_plan_v3ops/).")

with t4:
    v = jload(VAL / "results.json")
    if v:
        b = v.get("baselines", {})
        g_ = lambda k: v.get(k, {}).get("val", {}).get("gps_rmse")
        rows = [("E2 관측정만", g_("E2")), ("E3 관측정 + 위성", g_("E3")),
                ("E4 + 불확실성", g_("E4")), ("(비교) 앞으로 안 꺼진다", b.get("zero_change_rmse")),
                ("(비교) 과거 위성 추세 연장", b.get("trend_rmse"))]
        st.markdown("#### 2022년까지 자료로 보정 → 보정에 안 쓴 2023~2026 GPS로 채점")
        st.dataframe(pd.DataFrame(rows, columns=["방식", "GPS 침하 오차 (mm)"]).round(1), hide_index=True)
        e4 = v.get("E4", {}).get("val", {})
        if e4.get("gps_cov90_with_model_error") is not None:
            st.caption(f"E4 90% 예측 범위가 실제 GPS를 포함한 비율 {e4['gps_cov90_with_model_error'] * 100:.0f}% (평균 폭 {e4.get('gps_band90_mean_width_mm', 0):.0f}mm)")
        if (VAL / "real-3-gps-validation.png").exists():
            st.image(str(VAL / "real-3-gps-validation.png"), width="stretch")
    if (VALR / "validate-1-rates.png").exists():
        st.markdown("#### 자연 실험: 수로 물이 늘고 줄 때 침하 속도 변화를 맞히는가")
        st.image(str(VALR / "validate-1-rates.png"), width="stretch")
    st.caption("운영 모델(계획·정책에 쓰는 모델)은 검증이 끝난 뒤 2023~2026 GPS까지 넣어 다시 보정했다. 실제 예보 시스템처럼 새 위성·GPS 자료가 들어올 때마다 갱신한다.")
