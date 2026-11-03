"""⑤ 개발 과정: 실험 기록 탭 (app_final.py가 부른다). 미리 계산한 결과 파일만 읽는다(MODFLOW를 다시 돌리지 않음).

읽는 것
  results_*/results_dev_rescore.json·results_dev.json·calib_cache.json·model_version.json  — 연습 시험(scoreboard.py, rescore_practice.py)
  results_opera/checks.json                                                                — 위성 점검(opera_checks.py)
  data/cvhm2_box.npz, data/aqueduct_mp185_218.geojson, data/oil_fields_in_box.geojson, data/gps_monthly.csv — 압축성 지도
기존 ①~④ 탭(고객 시뮬레이터)은 건드리지 않는다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# 10/28~11/2에 시험한 후보 11개와 비교 기준 v3
# (결과 폴더 이름의 변형, 이름, 구분, 바꾼 것)
CANDS = [
    ("v3", "v3 (현재 모델)", "기준", "지연 압밀·과거 취수 이력·관개 회귀수, 3×3 구역 압축 배수"),
    ("v4-oil", "유전 침하 성분", "이번 주", "Lost Hills 유전 침하가 경계 밖으로 번진 몫(일정 속도, 거리 따라 감소)"),
    ("v4-oil-hw", "유전 + 수위 가중치", "이번 주", "보정 때 관측정 수위 σ 0.5 → 0.2m"),
    ("v4-oil-aq", "유전 + 수로 띠 배수", "이번 주", "유전 성분 + 수로 띠 압축 배수"),
    ("v3-opera", "v3, OPERA로 보정", "이번 주", "보정 위성을 DWR → NASA OPERA"),
    ("v4-oil-opera", "유전, OPERA로 보정", "이번 주", "유전 성분 + OPERA 보정"),
    ("v4-pp", "파일럿 포인트", "이번 주", "3×3 구역 9개 → 파일럿 포인트 33개(이웃끼리 비슷하게)"),
    ("v4-oil-pp", "유전 + 파일럿", "이번 주", "유전 성분 + 파일럿 포인트"),
    ("v4-pp-opera", "파일럿, OPERA로 보정", "이번 주", "파일럿 포인트 + OPERA 보정"),
    ("v4-oil-pp-opera", "유전 + 파일럿, OPERA", "이번 주", "유전 + 파일럿 + OPERA 보정"),
    ("v4-sw2", "수로물 의존도 안·밖", "이번 주", "수로물 공급 깊이를 금지구역 안·밖 두 개로"),
    ("v4-oil-pp-sw2", "유전 + 파일럿 + 수로물", "이번 주", "유전 + 파일럿 + 수로물 안·밖"),
]
FAM_COLOR = {"현재 모델 v3": "#111111",
             "유전 침하 성분": "#ff7f0e", "OPERA로 보정": "#2ca02c", "파일럿 포인트": "#1f77b4", "수로물 안·밖": "#9467bd"}


# 그림에 이름을 붙일 대표 후보(나머지는 마우스를 올리면 보임)와 글자 위치
LABEL_DEC = {"v3": "bottom center", "v3-opera": "top left", "v4-oil": "top right", "v4-pp": "bottom center",
             "v4-sw2": "top right"}
LABEL_FIT = {"v3": "bottom center", "v4-oil": "top center", "v4-pp": "top center", "v4-oil-aq": "bottom center"}


def md(text):
    """스트림릿 마크다운에서 '2015~2019'의 ~ 두 개가 취소선이 되지 않게."""
    return text.replace("~", "\\~")


def family(v):
    if v == "v3":
        return "현재 모델 v3"
    if "pp" in v:
        return "파일럿 포인트"
    if "sw2" in v:
        return "수로물 안·밖"
    if "oil" in v:
        return "유전 침하 성분"
    if v == "v3-opera":
        return "OPERA로 보정"
    return "이번 주 구조 후보"


def jl(f):
    f = Path(f)
    return json.load(open(f, encoding="utf-8")) if f.exists() else None


@st.cache_data
def load_trials():
    rows, params = [], {}
    for v, name, wave, what in CANDS:
        d = HERE / f"results_{v}"
        R, c, mv, dev = jl(d / "results_dev_rescore.json"), jl(d / "calib_cache.json"), jl(d / "model_version.json"), jl(d / "results_dev.json")
        if not (R and c and mv and "H2019" in c):
            continue
        n, a = R["opera"]["near"], R["opera"]["all"]
        bias = n["bias_model"]
        params[v] = dict(zip(mv["param_names"], c["H2019"]["th"]))
        rows.append({"key": v, "후보": name, "차수": wave, "바꾼 것": what, "묶음": family(v),
                     "보정 위성": "OPERA" if mv.get("insar") == "opera" else "DWR", "파라미터": len(mv["param_names"]),
                     "보정 적합도": c["H2019"]["hist"][-1]["insar_rmse_mm"], "수로 2km": n["model"], "편향": bias,
                     "칸별 분포 오차": float(np.sqrt(max(n["model"] ** 2 - bias ** 2, 0.0))), "지역 전체": a["model"],
                     "GPS P544": R["p544"]["model"], "2022.12 예측": R["end"]["model"],
                     "범위 끝 파라미터": len(dev.get("at_bound", [])) if dev else None})
    base = jl(HERE / "results_v3" / "results_dev_rescore.json") or {}
    tc = jl(HERE / "results_v3-opera" / "results_dev_rescore.json") or {}
    ref = {"trend": base.get("opera", {}).get("near", {}).get("trend"), "trend_all": base.get("opera", {}).get("all", {}).get("trend"),
           "trend_end": base.get("end", {}).get("trend"), "obs_end": base.get("end", {}).get("opera_obs"),
           "trend_p544": base.get("p544", {}).get("trend"), "trend_opera": tc.get("opera", {}).get("near", {}).get("trend_cal")}
    return pd.DataFrame(rows), params, ref


@st.cache_data
def geo(_r):
    """압축성 지도용: 격자 크기, 파일럿 포인트 위치·가중치, 수로·유전·GPS 위치(km, 격자 남서 모서리 기준)."""
    r = _r
    xy, (ny, nx) = r._pilot_points()
    km = r.MCELL / 1000
    cx, cy = np.meshgrid((np.arange(nx) + 0.5) * km, (np.arange(ny) + 0.5) * km)
    d2 = (cx.ravel()[:, None] - xy[None, :, 0]) ** 2 + (cy.ravel()[:, None] - xy[None, :, 1]) ** 2
    w = np.exp(-d2 / (2 * r.PP_LEN ** 2)); w = w / w.sum(1, keepdims=True)
    to_km = lambda lon, lat: ((np.asarray(lon) - r.LON0) * r.KM_LON, (np.asarray(lat) - r.LAT0) * r.KM_LAT)
    aq = json.load(open(DATA / "aqueduct_mp185_218.geojson")); pts = []
    for f in aq["features"]:
        g = f["geometry"]
        for part in (g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]):
            pts += [c[:2] for c in part]
    pts = np.array(pts); pts = pts[np.argsort(pts[:, 1])]
    oil = []
    for f in json.load(open(DATA / "oil_fields_in_box.geojson"))["features"]:
        g = f["geometry"]
        for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            ring = np.array(poly[0])
            oil.append(to_km(ring[:, 0], ring[:, 1]))
    gps = pd.read_csv(DATA / "gps_monthly.csv").groupby("station").agg(lat=("lat", "first"), lon=("lon", "first")).reset_index()
    gx, gy = to_km(gps.lon, gps.lat)
    return {"ny": ny, "nx": nx, "km": km, "xy": xy, "w": w, "aq": to_km(pts[:, 0], pts[:, 1]), "oil": oil,
            "gps": list(zip(gps.station, gx, gy))}


def fig_decompose(df, ref):
    fig = go.Figure()
    t = np.linspace(0, np.pi, 200)
    rv3 = float(df.loc[df.key == "v3", "수로 2km"].iloc[0])
    fig.add_trace(go.Scatter(x=rv3 * np.cos(t), y=rv3 * np.sin(t), fill="toself", fillcolor="rgba(44,160,44,0.10)",
                             line=dict(color="#2ca02c", width=2), name=f"v3 수준({rv3:.1f}mm) — 이 안에 들어와야 v3를 이김", hoverinfo="skip"))
    if ref.get("trend"):
        fig.add_trace(go.Scatter(x=ref["trend"] * np.cos(t), y=ref["trend"] * np.sin(t), line=dict(color="#d62728", width=1.5, dash="dot"),
                                 name=f"위성 추세 연장({ref['trend']:.1f}mm)", hoverinfo="skip"))
    for fam, g in df.groupby("묶음", sort=False):
        fig.add_trace(go.Scatter(x=g["편향"], y=g["칸별 분포 오차"], mode="markers+text",
                                 text=[n if k in LABEL_DEC else "" for k, n in zip(g.key, g["후보"])],
                                 textposition=[LABEL_DEC.get(k, "top center") for k in g.key], textfont=dict(size=11), name=fam,
                                 marker=dict(size=12 if fam == "현재 모델 v3" else 10, color=FAM_COLOR.get(fam, "#333"),
                                             symbol=["diamond" if s == "OPERA" else "circle" for s in g["보정 위성"]],
                                             line=dict(width=1, color="white")),
                                 customdata=np.c_[g["수로 2km"], g["후보"]],
                                 hovertemplate="%{customdata[1]}<br>편향 %{x:+.1f}mm<br>칸별 분포 오차 %{y:.1f}mm<br>수로 2km 오차 %{customdata[0]:.1f}mm<extra></extra>"))
    fig.update_layout(height=620, xaxis=dict(title="편향: 수로 칸 전체를 얼마나 넘치게 예측했나 (mm, +는 과대)", range=[-4, 25], zeroline=True,
                                             constrain="domain"),
                      yaxis=dict(title="칸별 분포 오차: 편향을 빼고 남는 오차 (mm)", range=[0, 21], scaleanchor="x", scaleratio=1, constrain="domain"),
                      legend=dict(orientation="h", y=-0.22), margin=dict(t=10, b=10, l=10, r=10))
    return fig


def fig_overfit(df):
    fig = go.Figure()
    for fam, g in df.groupby("묶음", sort=False):
        fig.add_trace(go.Scatter(x=g["보정 적합도"], y=g["수로 2km"], mode="markers+text",
                                 text=[n if k in LABEL_FIT else "" for k, n in zip(g.key, g["후보"])],
                                 textposition=[LABEL_FIT.get(k, "top center") for k in g.key], textfont=dict(size=11), name=fam,
                                 marker=dict(size=10, color=FAM_COLOR.get(fam, "#333"), symbol=["diamond" if s == "OPERA" else "circle" for s in g["보정 위성"]]),
                                 customdata=np.c_[g["후보"]],
                                 hovertemplate="%{customdata[0]}<br>보정 적합도 %{x:.1f}mm<br>연습 시험 %{y:.1f}mm<extra></extra>"))
    fig.update_layout(height=460, xaxis=dict(title="보정 적합도: 2015~2019 위성을 얼마나 잘 맞췄나 (mm, 왼쪽일수록 잘 맞춤)", autorange="reversed"),
                      yaxis=dict(title="연습 시험: 2020~22 수로 2km 오차 (mm)"), showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
    return fig


def fig_end(df, ref):
    g = df.sort_values("2022.12 예측")
    fig = go.Figure(go.Bar(x=g["2022.12 예측"], y=g["후보"], orientation="h", marker_color=[FAM_COLOR.get(f, "#333") for f in g["묶음"]],
                           text=[f"{v:.0f}" for v in g["2022.12 예측"]], textposition="outside",
                           hovertemplate="%{y}: %{x:.1f}mm<extra></extra>"))
    if ref.get("obs_end"):                                # 기준선 이름은 그림 위(막대와 겹치지 않게)
        fig.add_vline(x=ref["obs_end"], line=dict(color="#d62728", width=3))
        fig.add_annotation(x=ref["obs_end"], y=1.0, yref="paper", yanchor="bottom", xanchor="left", showarrow=False,
                           text=f" 정답(OPERA) {ref['obs_end']:.0f}mm", font=dict(color="#d62728"))
    if ref.get("trend_end"):
        fig.add_vline(x=ref["trend_end"], line=dict(color="#d62728", width=1.5, dash="dot"))
        fig.add_annotation(x=ref["trend_end"], y=1.0, yref="paper", yanchor="bottom", xanchor="right", showarrow=False,
                           text=f"추세 연장 {ref['trend_end']:.0f} ", font=dict(color="#d62728"))
    fig.update_layout(height=60 + 26 * len(g), xaxis=dict(title="2019.12 → 2022.12 수로 칸 누적 침하 중앙값 (mm)"), margin=dict(t=40, b=10, l=10, r=40))
    return fig


def fig_compress(G, p, key):
    ny, nx, km = G["ny"], G["nx"], G["km"]
    pp = [f"log10_mSk_p{i}" for i in range(len(G["xy"]))]
    fig = go.Figure()
    if all(k in p for k in pp):
        vals = np.array([p[k] for k in pp])
        field = (G["w"] @ vals).reshape(ny, nx)
    else:
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
        zone = np.minimum(gy * 3 // ny, 2) * 3 + np.minimum(gx * 3 // nx, 2)
        field = np.array([p.get(f"log10_mSk_z{i}", 0.0) for i in range(9)])[zone]
        vals = None
    fig.add_trace(go.Heatmap(z=field, x=(np.arange(nx) + 0.5) * km, y=(np.arange(ny) + 0.5) * km, zmin=-1.5, zmax=1.5, colorscale="RdBu_r",
                             colorbar=dict(title="압축성<br>(CVHM2 대비)", tickvals=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5],
                                           ticktext=["×0.03", "×0.1", "×0.3", "×1", "×3", "×10", "×30"]),
                             hovertemplate="x %{x:.1f}km, y %{y:.1f}km<br>압축성 log10 배수 %{z:+.2f}<extra></extra>"))
    for ox, oy in G["oil"]:
        fig.add_trace(go.Scatter(x=ox, y=oy, mode="lines", line=dict(color="#555", width=1), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=G["aq"][0], y=G["aq"][1], mode="lines", line=dict(color="black", width=3), name="California Aqueduct"))
    if vals is not None:
        bound = np.abs(vals) >= 1.48
        fig.add_trace(go.Scatter(x=G["xy"][~bound, 0], y=G["xy"][~bound, 1], mode="markers", name="파일럿 포인트",
                                 marker=dict(size=8, color="white", line=dict(color="black", width=1))))
        fig.add_trace(go.Scatter(x=G["xy"][bound, 0], y=G["xy"][bound, 1], mode="markers", name=f"범위 끝에 붙은 점 ({int(bound.sum())}개)",
                                 marker=dict(size=13, symbol="x", color="#d62728", line=dict(width=2))))
    fig.add_trace(go.Scatter(x=[g[1] for g in G["gps"]], y=[g[2] for g in G["gps"]], mode="markers+text", text=[g[0] for g in G["gps"]],
                             textposition="middle right", marker=dict(size=11, color="#ffd400", line=dict(color="black", width=1)), name="수로 GPS"))
    fig.update_layout(height=600, xaxis=dict(title="동쪽 (km)", range=[0, nx * km], constrain="domain"),
                      yaxis=dict(title="북쪽 (km)", range=[0, ny * km], scaleanchor="x", scaleratio=1),
                      legend=dict(orientation="h", y=-0.12), margin=dict(t=10, b=10, l=10, r=10))
    return fig


def fig_placebo(ck):
    rows = ck["rows"]
    lab = [f"{r_['pair']}<br>({r_['alloc']})" for r_ in rows]
    did = np.array([r_["did"] for r_ in rows]); lo = did - np.array([r_["p5"] for r_ in rows]); hi = np.array([r_["p95"] for r_ in rows]) - did
    exp = ck["intercept"] + ck["slope_per_pct"] * np.array([r_["d_alloc"] for r_ in rows])
    fig = go.Figure(go.Bar(x=lab, y=did, error_y=dict(type="data", array=hi, arrayminus=lo), name="안 − 밖 (위성 관측)",
                           marker_color=["#d62728" if r_["ban"] else "#9ecae1" for r_ in rows],
                           hovertemplate="%{x}<br>안 − 밖 %{y:+.1f}mm/년<extra></extra>"))
    fig.add_trace(go.Scatter(x=lab, y=exp, mode="markers", marker=dict(symbol="line-ew-open", size=26, color="black", line=dict(width=3)),
                             name="배분율 변화만으로 예상한 값", hovertemplate="예상 %{y:+.1f}<extra></extra>"))
    fig.update_layout(height=420, yaxis=dict(title="안 − 밖 속도 변화 (mm/년)"),
                      legend=dict(orientation="h", y=-0.3), margin=dict(t=10, b=10, l=10, r=10))
    return fig


def fig_distance(ck):
    bd = ck["by_distance"]
    names = list(bd)
    fig = go.Figure(go.Bar(x=names, y=[bd[n]["drought_ratio"] for n in names], marker_color=["#fdae6b", "#fd8d3c", "#e6550d", "#a63603"],
                           text=[f"{bd[n]['drought_ratio']:.2f}배" for n in names], textposition="outside"))
    fig.update_layout(height=360, yaxis=dict(title="2020~22 가뭄 침하 속도 ÷ 2017~19 (배)", range=[0, 3.6]), margin=dict(t=10, b=10, l=10, r=10))
    return fig


def render(r):
    df, params, ref = load_trials()
    if df.empty:
        st.warning("연습 시험 결과가 없습니다: bash run_dev.sh … → rescore_practice.py")
        return
    base = df.loc[df.key == "v3"].iloc[0]
    best = df.loc[df.key != "v3"].sort_values("수로 2km").iloc[0]
    pilot = df.loc[df.key == "v4-pp"].iloc[0]
    st.markdown("#### 10/28~11/2 · 모델 개선 실험 11개")
    st.write("유전 침하, 점토의 공간 분포, 수로 물 의존도, 보정 위성을 바꿔 시험했다. 결과를 보기 전에 정한 기준으로 비교했다.")
    a, b, c = st.columns(3)
    a.metric("기존 모델 v3 · 수로 오차", f"{base['수로 2km']:.1f} mm")
    b.metric("이번 주 최고 후보 · 수로 오차", f"{best['수로 2km']:.1f} mm")
    c.metric("칸별 분포 오차", f"{base['칸별 분포 오차']:.1f} → {pilot['칸별 분포 오차']:.1f} mm")
    st.markdown(f"**결론:** 11개 후보 중 수로 예측에서 v3를 이긴 모델은 없었다. 최고 후보도 {best['후보']}({best['수로 2km']:.1f}mm)였다. "
                "칸별 차이는 개선됐지만 수로 근처의 가뭄 반응을 크게 예측했다.")
    chart = go.Figure(go.Bar(
        x=[base["수로 2km"], best["수로 2km"], df.loc[df.key == "v4-oil", "수로 2km"].iloc[0], pilot["수로 2km"]],
        y=["기존 v3", "이번 주 최고", "유전 침하 반영", "점토 분포 세분화"], orientation="h",
        marker_color=["#316b73", "#7891a5", "#b98259", "#7891a5"],
        text=[f"{v:.1f} mm" for v in [base["수로 2km"], best["수로 2km"], df.loc[df.key == "v4-oil", "수로 2km"].iloc[0], pilot["수로 2km"]]],
        textposition="outside", hoverinfo="skip"))
    chart.update_layout(height=260, margin=dict(t=10, b=20, l=0, r=70), xaxis=dict(title="2020~22 수로 2km 예측 오차 · 낮을수록 좋음", range=[0, 19]), yaxis=dict(autorange="reversed"))
    st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
    if not st.toggle("실험 과정과 전체 점수표 보기", value=False):
        return

    st.markdown("##### 실험 설계")
    st.markdown(md(
        "- **연습 시험:** 2015~2019 자료로 보정 → 2020~2022 침하 예측 → NASA OPERA 위성으로 채점(수로 2km 칸). 최종 시험(2023~26 GPS)은 이긴 후보만 본다.\n"
        "- **판정 기준(결과 보기 전에 정함):** 수로 2km 오차가 v3보다 작고, 독립 GPS(P544) 오차가 크게 나빠지지 않을 것.\n"
        f"- **결과:** 이번 주 후보 11개 모두 v3({base['수로 2km']:.1f}mm)를 이기지 못해 v3 확정."))

    st.markdown("##### ① 오차를 두 부분으로 나누면: 전체 수준(편향)과 칸별 분포")
    st.plotly_chart(fig_decompose(df, ref), width="stretch")
    st.caption(md("원점에서 거리 = 수로 2km 오차. 파일럿 포인트(파랑)는 칸별 분포 오차를 줄였지만(아래로) 수로 근처를 크게 예측해 편향이 커졌다(오른쪽으로). "
               "유전 성분(주황)도 같은 방향. ◆ = OPERA로 보정."))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### ② 보정을 잘 맞출수록 예측이 나빠졌다")
        st.plotly_chart(fig_overfit(df), width="stretch")
        st.caption(md("유연한 구조일수록 2015~19를 잘 맞추지만(왼쪽) 2020~22 예측은 나빠진다 → 맞추는 방법이 아니라 입력이 부족하다는 신호."))
    with c2:
        st.markdown(md("##### ③ 2020~22 가뭄 뒤 수로 칸 침하: 예측 vs 정답"))
        st.plotly_chart(fig_end(df, ref), width="stretch")
        st.caption(md("v3는 모자라고(일정하게 꺼지는 몫이 없음), 고친 후보들은 넘친다(가뭄 반응을 크게 잡음) → v3의 점수에는 두 오차가 서로 상쇄된 몫이 있다."))

    st.markdown("##### ④ 모델이 찾은 점토 압축성 지도")
    opts = [k for k in df.key if k in params]
    names = dict(zip(df.key, df["후보"]))
    pick = st.selectbox("후보", opts, index=opts.index("v4-pp") if "v4-pp" in opts else 0, format_func=lambda k: names.get(k, k))
    try:
        st.plotly_chart(fig_compress(geo(r), params[pick], pick), width="stretch")
    except Exception as e:
        st.info(f"지도를 그리지 못했습니다: {e}")
    st.caption(md("빨강일수록 잘 눌리는 점토. 3×3 구역(v3)은 경계가 사각형으로 끊기고, 파일럿 포인트는 칸별로 부드럽게 바뀐다. "
               "어느 위성으로 보정하든 수로 남쪽 점들(Q208·Q213 근처)이 상한(×30, 빨간 X)으로 간다 — 모델이 수로 근처의 큰 침하를 '무른 점토'로만 설명할 수 있어서다."))

    ck = jl(HERE / "results_opera" / "checks.json")
    if ck:
        st.markdown("##### ⑤ 위성으로 다시 확인한 것 (모델 없이, opera_checks.py)")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**수로 근처는 가뭄에 둔하고, 바깥은 민감하다**")
            st.plotly_chart(fig_distance(ck), width="stretch")
            st.caption(md("모델 입력 취수는 같은 해 모든 농경지가 같은 비율로 늘어(수로 근처 1.37배·바깥 1.24배) 이 차이를 못 담는다. 실제 취수 자료가 필요한 이유."))
        with c4:
            pl = ck["placebo"]
            st.markdown("**금지 효과 점검(이중차분)을 금지 없던 해에도 해 보면**")
            st.plotly_chart(fig_placebo(pl), width="stretch")
            st.caption(md(f"금지 없던 해에도 안 − 밖이 배분율 변화를 따라 움직였다. 2024→2025(빨강)는 금지 없이도 {pl['expected_ban_pair']:+.1f}mm/년이 예상된다. "
                       f"남는 차이 {pl['excess']:+.1f}mm/년은 연도별 변동({pl['resid_sd']:.1f}) 수준이다. 위성에서 금지 효과는 확인되지 않는다."))

    st.markdown("##### 점수표 (연습 시험, mm)")
    show = df.drop(columns=["key", "묶음"]).sort_values("수로 2km").round(1)
    extra = []
    if ref.get("trend"):
        extra.append({"후보": "위성 추세 연장(DWR 2015.7~2019.12)", "차수": "기준선", "바꾼 것": "모델 없음", "보정 위성": "-", "수로 2km": round(ref["trend"], 1),
                      "지역 전체": round(ref["trend_all"], 1) if ref.get("trend_all") else None,
                      "GPS P544": round(ref["trend_p544"], 1) if ref.get("trend_p544") else None,
                      "2022.12 예측": round(ref["trend_end"], 1) if ref.get("trend_end") else None})
    if ref.get("trend_opera"):
        extra.append({"후보": "위성 추세 연장(OPERA 2016.7~2019.12)", "차수": "참고", "바꾼 것": "모델 없음", "보정 위성": "-", "수로 2km": round(ref["trend_opera"], 1)})
    table = pd.concat([show, pd.DataFrame(extra)], ignore_index=True).drop(columns=["묶음"], errors="ignore")
    st.dataframe(table, hide_index=True, width="stretch", height=36 * (len(table) + 1) + 4)

    st.markdown("##### 그래서 배운 것")
    st.markdown(md(
        "1. **판정을 먼저 정하고, 기준선과 같이 채점한다.** 이번 주 후보 11개 모두 미리 정한 기준에서 v3를 이기지 못했다.\n"
        "2. **오차를 나눠 보면 원인이 보인다.** 칸별 분포는 파일럿 포인트로 줄어들고(13.9 → 11.7mm), 남은 문제는 수로 근처 가뭄 반응이다.\n"
        "3. **수로 근처 침하 일부는 유전 생산 침하일 수 있다**(유전 경계 연 14mm, 1.3km마다 37%로 감소; 습한 해에도 계속 꺼진 GPS 4곳 모두 유전 2.5km 안).\n"
        "4. **다음 개선은 모델 구조가 아니라 입력 자료:** 가뭄 해 실제 취수(관리기관 계량), 유전 생산량, 수로 근처 관측정 → 첫 고객과의 자료 협력이 기술 로드맵 1단계."))
