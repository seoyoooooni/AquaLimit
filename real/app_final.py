"""AquaLimit 최종 데모 (California Aqueduct MP 195~215, 운영 모델 v3ops).

실행: cd real && streamlit run app_final.py
시연 흐름: ① 문제(지도·금지 후에도 침하) → ② 관리기관: 전면 금지 vs AquaLimit
→ ③ 사업자: 과거 고정 계획과 시작 월·기간을 고르는 미래 계획 → ④ 검증(GPS, 자연 실험)
⑤ 지난 일주일 모델 개선 실험 기록(app_trials.py, 학교 발표용 — 고객 화면 ①~④와 따로)
③ 기본 조건은 미리 계산한 MODFLOW 응답을 읽는다. 새 기간·날씨·우물 조합은 MODFLOW로 다시 계산한다.
이전 2026년 5~7월 위치 입력 AI 계획도 ③에서 선택할 수 있다.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cloud_mf6 import configure_mf6

os.environ.setdefault("AQ_VARIANT", "v3ops")
os.environ.setdefault("AQ_PUMPING", "et")
configure_mf6()
import real_model as r  # noqa: E402
import plan_real as p  # noqa: E402
import future_simulator as future  # noqa: E402

HERE = Path(__file__).resolve().parent
VAL = HERE / "results_v3"                     # 검증 모델(예측 성능)
OPS = r.result_dir("model", create=False)     # 운영 모델
POL = HERE / f"results_policy_{r.TAG}"
PLAN = r.result_dir("plan", create=False)
AI = r.result_dir("ai", create=False)
VALR = HERE / "results_validate_v3"
VALO = HERE / f"results_validate_{r.TAG}"
REG = 12.7
st.set_page_config(page_title="AquaLimit", layout="wide", initial_sidebar_state="collapsed")
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
    # 수로 GPS는 여름에 올라가고 겨울에 내려가는 계절 변동이 있어서 같은 달 → 같은 달 1년 단위로 잰다(최근 1년 2025.3→2026.3).
    gm = site["gps"].pivot_table(index="month", columns="station", values="up_mm")
    def annual(st_, end="2026-03"):
        e = pd.Period(end, "M"); sub = lambda m: gm.reindex([m - 1, m, m + 1])[st_]
        a, b = sub(e - 12), sub(e)
        return float(-(b.mean() - a.mean())) if a.notna().sum() >= 2 and b.notna().sum() >= 2 else np.nan
    gps["rate_post"] = [annual(s) for s in gps.station]
    return dict(site=site, site_aq=site["aq_pts"], pts=pts, pts_ll=[ll(*q) for q in pts], wells=wells, ag=ag,
                ag_ll=np.array([ll(*c) for c in ag]), ag_dist=np.array([dist[a, b] for a, b in ag]),
                buffer_ll=np.array([ll(*c) for c in ag if dist[c[0], c[1]] <= pol.BUFFER_KM]),
                wcr_cap=wcr_cap, M_big=M_big, net=net, vel=vel, oil=oil, gps=gps, cap0=p.CAP.copy(), cost0=p.WELL_COST.copy(),
                e4=np.load(OPS / "E4_members.npy"),
                feats=lambda cells: __import__("surrogate_loc").loc_features(site, cells),
                default_cap=2000 * pol.GPM_TO_M3D * pol.DUTY,
                default_cost=float(np.median(p.WELL_COST)))


D = load()
st.title("AquaLimit")
st.caption("California Aqueduct MP 195~215 · 위성·관측정·GPS로 보정한 지하수 모델 · 관리 기준 연 12.7mm")
t1, t2, t3, t4, t5 = st.tabs(["① 현장 문제", "② 정책 비교", "③ 우물별 시뮬레이터", "④ 검증 결과", "⑤ 최근 1주 실험"])


@st.cache_data(show_spinner=False, max_entries=8)
def future_physics(start, count, scenario, well_ids, member_count):
    idx = list(well_ids)
    members = D["e4"][[0, 10, 20, 5, 15][:member_count]]
    return future.cached_responses(D["site"], members, D["pts"],
                                   [tuple(D["wells"][i]) for i in idx],
                                   D["cap0"][idx], start, count, scenario)


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
                      textposition="middle left", marker=dict(size=12, color="#ffd400"), name="GPS (최근 1년 침하 속도)"))
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
        st.metric("수로 GPS 침하 속도 (최근 1년, 2025.3→2026.3)", f"연 {g.rate_post.min():.0f}~{g.rate_post.max():.0f}mm",
                  f"기준(연 {REG}mm)의 최대 {g.rate_post.max() / REG:.1f}배", delta_color="inverse")
        st.markdown("가뭄에는 수로 물 공급이 줄어 농업용 지하수 취수가 늘어난다. 관리기관은 2024년 12월부터 수로 양쪽 2.5마일의 취수를 금지했다. "
                    "**그 후 1년에도 GPS에서 침하 감소는 확인되지 않았다.**")
        img = VALO / "validate-2-ban.png"
        if img.exists():
            st.image(str(img), width="stretch")
            dec = (jload(VALO / "results.json") or {}).get("decomposition", {}).get("aqueduct")
            if dec:
                # 금지 준수 시 침하가 0 근처라 비율(%)로 나누면 값이 폭주한다 → mm/년으로 보여 준다
                with st.expander("모델의 원인 분해와 한계"):
                    st.write(f"금지 후 수로 침하 추정값 {dec['asis']['med']:.1f}mm/년 = 과거 취수 영향 {dec['legacy']['med']:+.1f} "
                             f"+ 금지구역 밖 취수 {dec['outside']['med']:+.1f} + 금지구역 안 취수로 추정한 몫 {dec['inside_if_continued']['med']:+.1f}.")
                    st.caption("유전 생산에 따른 침하가 마지막 몫에 섞였을 수 있다. 관정별 취수 여부는 이 결과만으로 판단할 수 없다.")
    op = jload(HERE / "results_opera" / "results.json")
    if op and (HERE / "results_opera" / "opera-2-ban-map.png").exists():
        st.markdown("#### NASA 위성으로 본 금지 전후 (모델 없음)")
        st.image(str(HERE / "results_opera" / "opera-2-ban-map.png"), width="stretch")
        d_ = op["did"]
        ck = (jload(HERE / "results_opera" / "checks.json") or {}).get("placebo")   # opera_checks.py(11/2): 이중차분 위약 검사
        msg = (f"금지 전 1년 → 후 1년 침하 속도 변화: 금지구역 안 **{d_['in_change_med']:+.1f}**, 밖 **{d_['out_change_med']:+.1f}**mm/년. "
               f"금지가 지켜졌다면 안쪽이 더 좋아졌어야 하는데 **더 좋아지지 않았다**(안 − 밖 {d_['did_med']:+.1f}, 90% 범위 {d_['p5']:+.1f}~{d_['p95']:+.1f}). ")
        if ck:
            msg += (f"금지가 없던 해에도 안 − 밖은 수로 배분율에 따라 움직였다. 이번 배분율 변화만으로도 약 {ck['expected_ban_pair']:+.1f}mm/년이 예상된다. "
                    f"남는 차이 {ck['excess']:+.1f}mm/년은 연도별 변동({ck['resid_sd']:.1f}) 수준이다. **위성에서 금지 효과는 확인되지 않는다.** ")
        st.caption(msg + "초록 원은 현장 확인 후보이며, 취수 여부는 유량계로 확인해야 한다.")

with t2:
    o = jload(POL / "results.json") or {}
    blocks = {}
    if (POL / "results_retro.json").exists():
        blocks["2021~22 대가뭄 재현"] = jload(POL / "results_retro.json")
    for k, v in (jload(POL / "results_fwd.json") or {}).items():
        blocks[f"2026년 5~7월 시나리오: {k}"] = v
    if not blocks:
        st.warning("정책 비교 결과가 없습니다: AQ_VARIANT=v3ops python3 policy_real.py")
    else:
        choices = list(blocks)
        which = st.selectbox("상황", choices, index=choices.index("2026년 5~7월 시나리오: 평년") if "2026년 5~7월 시나리오: 평년" in choices else 0)
        b = blocks[which]
        ref = jload(POL / "refine.json") or {}             # refine_plan.py: MODFLOW로 확인·보정한 AquaLimit 계획 + MODFLOW로 찾은 금지 반경
        rtag = "retro" if which.startswith("2021") else which.split(": ", 1)[-1]
        rb = ref.get(rtag) or {}
        aqm = rb.get("aqualimit")
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
        if aqm:
            rows.append({"정책": "AquaLimit(규제 기준, MODFLOW로 확인·보정)", "대체 용수 (백만 m³)": round(aqm["replace_m3"] / 1e6, 2),
                         "비용 (백만 $)": round(aqm["cost"] / 1e6, 2), "수로 최대 침하 (mm, MODFLOW 확인)": round(aqm["worst_med"], 1),
                         "기준 지키는 수로 지점": f"{aqm['ok_points'] * 100:.0f}%", "모든 지점 기준 확률": f"{aqm['ok_all_prob'] * 100:.0f}%"})
        ban = b["policies"].get("4km 전면 금지(2025 실제 정책)", {})
        aqr = b["policies"].get("AquaLimit(규제 기준 0.5인치/년)", {})
        m0, m2 = st.columns(2)
        if ban.get("feasible"):
            m0.metric("전면 금지: 기준 지키는 수로 지점", f"{ban.get('reg_ok_points', 0) * 100:.0f}%", f"비용 ${ban['cost'] / 1e6:.1f}M", delta_color="off")
        if ban.get("feasible") and aqm:
            m2.metric("모든 지점 기준 달성에 필요한 비용 (MODFLOW 확인)", f"${aqm['cost'] / 1e6:.1f}M", f"{(aqm['cost'] - ban['cost']) / 1e6:+.1f}M vs 전면 금지", delta_color="off")
        elif ban.get("feasible") and aqr.get("feasible"):
            m2.metric("모든 지점 기준 달성에 필요한 비용", f"${aqr['cost'] / 1e6:.1f}M", f"{(aqr['cost'] - ban['cost']) / 1e6:+.1f}M vs 전면 금지", delta_color="off")
        st.caption("현재 4km 금지는 평년·가뭄에 모든 수로 지점을 보호하지 못한다. 계획안은 금지구역 밖 우물의 감량을 조정해 기준 달성 비용을 비교한다.")
        with st.expander("정책별 계산 결과와 MODFLOW 확인 과정"):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            if aqm:
                st.caption("선형 응답으로 푼 첫 계획은 MODFLOW에서 일부 구성원이 기준을 넘었다(모든 지점 확률 60~97%). "
                           "MODFLOW로 확인하고 감량을 조정하는 과정을 3~4회 반복해 모든 구성원·수로 지점을 기준 안으로 맞췄다.")
        names = {"retro": "2021~22 대가뭄 재현", "습한 해": "2026.5~7 습한 해", "평년": "2026.5~7 평년", "가뭄·폭염": "2026.5~7 가뭄·폭염"}
        if any(k != "_meta" and v.get("aqualimit") for k, v in ref.items()):
            fxr = (ref.get("_meta") or {}).get("fixed_radius_km")
            st.markdown("##### 그럼 금지 반경을 넓히면? — 같은 신뢰도(MODFLOW로 모든 구성원·모든 수로 지점 기준 달성)에 드는 비용")
            rr = []
            for k, v in ref.items():
                if k == "_meta" or not v.get("aqualimit"):
                    continue
                a, bm, fx = v["aqualimit"]["cost"], v.get("ban_min"), (v.get("ban_fixed") or {}).get("cost")
                rr.append({"상황": names.get(k, k), f"금지: 고정 규칙 {fxr:.1f}km": f"${fx / 1e6:.1f}M" if fx else "-",
                           "금지: 그해 최소 반경": f"{bm['radius_km']:.1f}km ${bm['cost'] / 1e6:.1f}M" if bm else "-",
                           "AquaLimit": f"${a / 1e6:.1f}M",
                           "절약(고정 규칙 대비)": f"${(fx - a) / 1e6:.1f}M ({(fx - a) / fx * 100:.0f}%)" if fx else "-",
                           "절약(최소 반경 대비)": f"${(bm['cost'] - a) / 1e6:.1f}M ({(bm['cost'] - a) / bm['cost'] * 100:.0f}%)" if bm else "-"})
            st.dataframe(pd.DataFrame(rr), hide_index=True, width="stretch")
            st.caption(f"반경은 한 번 정하면 해마다 바꾸기 어렵다. 가뭄에도 지키려면 {fxr:.1f}km가 필요하고, 평년·습한 해엔 필요 이상으로 멈춘다. "
                       "'그해 최소 반경'은 날씨를 미리 알고 0.1km 단위로 가장 유리하게 고른 경우다. 금지·AquaLimit 모두 MODFLOW 앙상블로 직접 확인했다"
                       "(results_policy_v3ops/RESULTS_refine.md).")
        else:
            rad = jload(POL / "radius_summary.json")
            if rad:
                st.markdown("##### 그럼 금지 반경을 넓히면? — 같은 기준(모든 수로 지점)을 지키는 비용 (선형 응답)")
                rr = []
                for k, v in rad.items():
                    fx = v.get("fixed_ban_cost")
                    rr.append({"상황": names.get(k, k), "금지: 그해 최소 반경": f"{v['min_radius_km']:.1f}km ${v['min_ban_cost'] / 1e6:.1f}M",
                               f"금지: 고정 {v.get('fixed_radius_km') or 0:.0f}km": f"${fx / 1e6:.1f}M" if fx else "-",
                               "AquaLimit": f"${v['aqualimit_cost'] / 1e6:.1f}M",
                               "절약(고정 규칙 대비)": f"${(fx - v['aqualimit_cost']) / 1e6:.1f}M ({(fx - v['aqualimit_cost']) / fx * 100:.0f}%)" if fx else "-"})
                st.dataframe(pd.DataFrame(rr), hide_index=True, width="stretch")
                st.caption("선형 응답(MODFLOW 미분)으로 계산했다(results_policy_v3ops/RESULTS_radius.md). MODFLOW 확인: AQ_VARIANT=v3ops python3 refine_plan.py")
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
    mode = st.radio("시뮬레이터 선택", ["미래 월별 계획", "2026년 5~7월 과거 실험"], horizontal=True)
    if mode == "미래 월별 계획":
        st.markdown("#### 앞으로 몇 달, 어느 우물에서 얼마나 뽑을까?")
        st.caption("기준 상태: 2026년 4월까지 반영한 지하수 모델. 시작 월 전까지의 빈 구간과 계획 기간은 선택한 날씨·수로물 배분 시나리오로 이어 계산합니다. "
                   "실제 관측과 미래 가정을 섞어 '현재 실측'이라고 부르지 않습니다.")
        first_month = max(pd.Period(pd.Timestamp.today(), "M"), D["site"]["months"][-1] + 1)
        starts = pd.period_range(first_month, pd.Period("2028-12", "M"), freq="M")
        c1, c2, c3 = st.columns(3)
        with c1:
            future_start = st.selectbox("시작 월", [str(x) for x in starts])
            future_count = st.slider("계획 기간 (개월)", 1, 6, 3)
        with c2:
            scenario = st.selectbox("미래 날씨·수로 배분", list(future.SCENARIOS))
            member_count = st.selectbox("물리 모델 구성원", [3, 1, 5], help="3개를 기본으로 사용합니다. 5개는 더 오래 걸립니다.")
        with c3:
            well_ids = st.multiselect("계획할 우물", list(range(len(D["wells"]))),
                                      default=list(range(len(D["wells"]))),
                                      format_func=lambda i: f"{chr(65+i)} · {D['cap0'][i]:,.0f} m³/일")
            margin_f = st.slider("모델 반응 여유 (%)", 0, 100, 25, 5) / 100
        future_months, bridge = future.forecast_months(future_start, future_count, D["site"]["months"][-1])
        sc = future.SCENARIOS[scenario]
        st.caption(f"2026년 5월부터 시작 전까지 {bridge}개월은 우물 용량의 75%로 운영한다고 가정합니다. "
                   f"시나리오: 강수 평년의 {sc['pfac']:.0%}, 기온 {sc['dT']:+.1f}°C, 수로물 배분 {sc['alloc']:.0%}. "
                   "우물 위치·용량은 DWR 신고 자료, 기준 초과는 사업자 계획의 추가 침하만으로 판단합니다.")
        with st.expander("월별 수요와 비용 설정"):
            dcols = st.columns(future_count)
            demand_f = [dcols[i].slider(f"{m} 수요 (%)", 20, 120, 75, 5, key=f"dem_{i}") / 100
                        for i, m in enumerate(future_months)]
            lc, pc, xc = st.columns(3)
            limit_f = lc.number_input("추가 침하 한도 (mm / 92일)", 0.1, 10.0, 2.0, 0.1)
            price_f = pc.number_input("외부 용수 단가 ($/m³)", 0.2, 3.0, 0.81, 0.01)
            purchase_f = xc.slider("월 수요 중 외부 구매 가능 비율 (%)", 0, 100, 100, 5) / 100
        key_f = (future_start, future_count, scenario, tuple(well_ids), member_count)
        if st.session_state.get("future_result", (None,))[0] != key_f and well_ids:
            indices = list(well_ids)
            ready = future.load_cached_responses(D["site"], D["e4"][[0, 10, 20, 5, 15][:member_count]],
                                                 [tuple(D["wells"][i]) for i in indices], D["cap0"][indices],
                                                 future_start, future_count, scenario)
            if ready is not None:
                st.session_state["future_result"] = (key_f, ready)
        if st.button("선택한 기간의 미래 계획 계산", type="primary", disabled=not well_ids):
            with st.spinner("MODFLOW로 월별 우물 응답을 계산하고 있습니다. 선택한 기간·우물 수에 따라 몇 분 걸릴 수 있습니다."):
                try:
                    st.session_state["future_result"] = (key_f, future_physics(*key_f))
                except Exception as e:
                    st.error(f"물리 모델 계산 실패: {e}")
        saved = st.session_state.get("future_result")
        if saved and saved[0] == key_f:
            F = saved[1]
            chosen = list(well_ids)
            opt = future.optimize(F, D["cost0"][chosen], price_f, demand_f, limit_f, margin_f, purchase_f)
            if not opt["feasible"]:
                st.error(opt["reason"])
            else:
                base_f = future.proportional_plan(F, demand_f)
                risk = (opt["max_extra"] > opt["limit"][None] + 1e-6).any(axis=1)
                base_risk = (base_f["max_extra"] > opt["limit"][None] + 1e-6).any(axis=1)
                a, b, c = st.columns(3)
                a.metric("계획 기간 총비용", f"${opt['cost']/1e3:,.0f}천")
                b.metric("외부 용수", f"{opt['purchase'].sum()/1e3:,.0f}천 m³")
                c.metric("추가 침하 한도 초과", f"{risk.sum()}/{len(risk)}개 모델",
                         f"용량 비례 계획 {base_risk.sum()}/{len(base_risk)}개", delta_color="off")
                table = pd.DataFrame(np.round(opt["rates"]).astype(int),
                                     columns=[f"우물 {chr(65+i)} (m³/일)" for i in chosen])
                table.insert(0, "월", F["months"])
                table["외부 용수 (m³/월)"] = np.round(opt["purchase"]).astype(int)
                table["월 수요 (m³)"] = np.round(opt["demand"]).astype(int)
                st.dataframe(table, hide_index=True, width="stretch")
                fig_f = go.Figure()
                for i, vals in enumerate(opt["max_extra"]):
                    fig_f.add_trace(go.Scatter(x=F["months"], y=vals, mode="lines+markers",
                                               name=f"지층 상태 {i+1}"))
                fig_f.add_trace(go.Scatter(x=F["months"], y=opt["limit"], mode="lines+markers",
                                           name="허용 추가 침하", line=dict(color="black", dash="dash")))
                fig_f.update_layout(yaxis_title="계획으로 생기는 최대 추가 침하 (mm)", height=330,
                                    margin=dict(t=15, b=15))
                st.plotly_chart(fig_f, width="stretch")
                st.caption(f"다른 농가와 날씨에 따른 배경 수로 변위: {np.median(F['baseline'][:, -1].max(axis=1)):+.1f}mm "
                           f"(마지막 달까지, 모델 {len(F['baseline'])}개 중앙값; 침하 +, 융기 −). 위 선은 이 사업자 계획의 추가 침하입니다. "
                           "표의 0/N은 계산에 사용한 구성원에서의 결과이며, 실제 초과 확률이나 2026년 11월 실측값이 아닙니다.")
                verify_key = (key_f, tuple(np.round(opt["rates"].ravel(), 3)))
                members = D["e4"][[0, 10, 20, 5, 15][:member_count]]
                checked = future.cached_verification(D["site"], members, D["pts"], F, opt["rates"])
                if st.button("이 월별 계획을 MODFLOW로 직접 검증"):
                    with st.spinner("선택한 월별 취수량을 물리 모델에 다시 넣어 검증하고 있습니다."):
                        try:
                            checked = future.cached_verification(D["site"], members, D["pts"], F, opt["rates"], compute=True)
                            st.session_state["future_verification"] = (verify_key, checked)
                        except Exception as e:
                            st.error(f"직접 검증 실패: {e}")
                checked_saved = st.session_state.get("future_verification")
                if checked is None and checked_saved and checked_saved[0] == verify_key:
                    checked = checked_saved[1]
                if checked is not None:
                    actual_max = checked["extra"].max(axis=2)
                    actual_fail = (actual_max > opt["limit"][None] + 1e-6).any(axis=1)
                    st.success(f"MODFLOW 직접 검증: 한도 초과 {actual_fail.sum()}/{len(actual_fail)}개 모델 · "
                               f"선형 응답과의 최대 차이 {checked['max_abs_linear_error_mm']:.3f}mm")
        elif saved:
            st.info("시작 월·기간·날씨·우물·모델 수가 바뀌었습니다. 계산 버튼을 눌러 새 조건으로 계산하세요.")
        else:
            st.info("시작 월과 기간을 선택한 뒤 계산 버튼을 누르면 월별 계획이 나옵니다.")
    else:
        st.markdown("#### 2026년 5~7월 우물별 취수 계획")
        st.caption("2026년 4월까지의 관측을 바탕으로 만든 과거 실험. 우물마다 정한 하루 취수량을 5·6·7월에 동일하게 적용한다. "
                   "지도에서 우물 위치를 바꾸면 이 기간의 계획을 다시 계산한다. 미래의 월별 계획은 위에서 '미래 월별 계획'을 선택하면 볼 수 있다.")
        if D["net"] is None:
            st.warning("위치 입력 AI 결과가 없습니다: AQ_VARIANT=v3ops python3 surrogate_loc.py")
        else:
            with st.expander("계획 조건 바꾸기"):
                a, b = st.columns(2)
                with a:
                    limit = st.slider("수로 추가 침하 한도 (mm/3개월)", 0.5, 4.0, 2.0, 0.1, help="연 0.5인치 기준(3개월 3.2mm) 중 이 사업자 몫")
                    dem_frac = st.slider("필요 수량 (우물 용량 합 대비 %)", 20, 120, 75, 5)
                with b:
                    price = st.number_input("외부 용수 단가 ($/m³)", 0.2, 2.0, 0.81, 0.01)
                    margin = st.select_slider("구조 오차 여유", options=[0.0, 0.25, 0.5, 1.0], value=0.25,
                                              format_func=lambda x: f"{x * 100:.0f}%")
                    st.caption("여유 25%는 실제 땅이 모델보다 25% 더 반응해도 안전하도록 계산한다는 뜻이다.")
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
            if src.startswith("기본값"):                 # 기본 사업자 우물 6개는 계획 계산(run_plan.py)과 같은 용량을 쓴다(같은 칸의 다른 신고 우물은 합치지 않음)
                caps = np.array(D["cap0"], float)
            else:
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
    ann = jload(VALR / "annual.json")
    if ann:
        st.markdown("#### 자연 실험: 수로 물이 늘고 줄 때 침하 속도 변화를 맞히는가 (검증 모델, 같은 달 → 같은 달 1년 단위)")
        st.dataframe(pd.DataFrame([{"GPS": x["station"], "사건": x["event"], "GPS 변화 (mm/년)": round(x["gps_change"], 1),
                                    "모델 변화 (중앙값)": round(x["model_change_med"], 1),
                                    "모델 90% 범위": f"{x['model_change_p5']:+.1f} ~ {x['model_change_p95']:+.1f}",
                                    "일치": "예" if x["agree"] else "아니오"} for x in ann["rows"]]), hide_index=True, width="stretch")
        st.caption("습한 해의 침하 감속은 크기까지 맞혔다. 금지 후에는 모델이 예측한 감속이 GPS에서 확인되지 않았다. 원인 판단에는 실제 취수량과 유전 생산 자료가 더 필요하다.")
    elif (VALR / "validate-1-rates.png").exists():
        st.markdown("#### 자연 실험: 수로 물이 늘고 줄 때 침하 속도 변화를 맞히는가")
        st.image(str(VALR / "validate-1-rates.png"), width="stretch")
    st.caption("정책·계획 계산에는 2023~2026 GPS를 추가해 다시 보정한 운영 모델을 사용했다.")


with t5:
    try:
        import app_trials
        app_trials.render(r)
    except Exception as e:                        # 실험 기록 탭이 실패해도 ①~④ 탭은 그대로 보인다
        st.error(f"실험 기록 탭을 그리지 못했습니다: {e}")
