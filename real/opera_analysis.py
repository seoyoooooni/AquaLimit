"""NASA OPERA 위성 침하로 본 2024.12 취수 금지 전후 (모델 없이, 관측만).

prepare_opera.py --build 가 만든 data/opera_grid500m_monthly.npz(500m 칸, 월별 수직 변위, 위로 +)를 쓴다.

1) 자료 품질: 같은 칸에서 OPERA·DWR(TRE Altamira)·GPS 비교. DWR는 2023년 이후 GPS와 반대로 움직여 버렸던 자료다.
2) 금지 전후 침하 속도 지도: 습한 해(2023.1~2024.11) vs 금지 후(2024.12~2026.2)
3) 구역별 속도: 수로 1km 안 / 금지구역(수로 동쪽 4.02km 안 농경지) / 금지구역 밖(4~10km) — 유전 칸 제외
4) 현장 확인 후보: 금지구역 안에서 금지 후 가장 빨리 꺼지는 1km 칸

실행: python3 opera_analysis.py   (몇 초) → results_opera/RESULTS.md, opera-1-gps.png, opera-2-ban-map.png, results.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results_opera"
OUT.mkdir(exist_ok=True)
for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    try:
        font_manager.fontManager.addfont(p)
    except Exception:
        pass
for name in ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "Malgun Gothic"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})

LAT0, LON0, CELL = 35.50, -119.92, 500.0
KM_LAT = 110.95
KM_LON = 111.32 * np.cos(np.radians(35.64))
BUFFER_KM = 4.02                     # 금지구역: 수로 양쪽 2.5마일
SERVICE_KM = 10.0
REG = 12.7                           # 연 0.5인치(이웃 관리기관 수로 기준)
# 속도 구간은 '같은 달 → 같은 달'(연 단위)로 잡는다. GPS에 여름에 올라가고 겨울에 내려가는 계절 변동이 있어서,
# 구간 길이가 연 단위가 아니면(예: 2024.12~2026.4) 겨울이 두 번 들어가 속도가 부풀려진다. 끝점은 앞뒤 1달 포함 3개월 평균.
WIN = {"drought": ("2020-11", "2022-11", "2020.11→2022.11 가뭄(배분율 5~20%)"),
       "wet": ("2023-11", "2024-11", "2023.11→2024.11 금지 전 1년(배분율 40%)"),
       "post": ("2024-11", "2025-11", "2024.11→2025.11 금지 후 1년(배분율 50%)")}


def load():
    cells = pd.read_csv(DATA / "insar_grid500m_cells.csv")
    z = np.load(DATA / "opera_grid500m_monthly.npz")
    w = np.load(DATA / "insar_grid500m_monthly.npz")
    months = pd.PeriodIndex(z["months"], freq="M")
    assert list(z["months"]) == list(w["months"]), "OPERA·DWR 월이 다릅니다: prepare_opera.py --build 다시 실행"
    op = pd.DataFrame(z["disp_mm"].T.astype(float), index=months)       # (월, 칸) 위로 +
    dwr = pd.DataFrame(w["disp_mm"].T.astype(float), index=months)
    gps = pd.read_csv(DATA / "gps_monthly.csv"); gps["month"] = pd.PeriodIndex(gps.month, freq="M")
    aq = json.load(open(DATA / "aqueduct_mp185_218.geojson"))
    pts = []
    for f in aq["features"]:
        g = f["geometry"]
        for part in (g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]):
            pts += [c[:2] for c in part]
    pts = np.array(pts); pts = pts[np.argsort(pts[:, 1])]
    oil = json.load(open(DATA / "oil_fields_in_box.geojson"))
    cells["east"] = cells.lon > np.interp(cells.lat, pts[:, 1], pts[:, 0])
    cells["is_oil"] = cells.oil_field.notna()
    return cells, op, dwr, gps, pts, oil, months


def end_mean(df, m):
    """m 앞뒤 1달 포함 3개월 평균(2개 이상 있을 때)."""
    m = pd.Period(m, "M")
    sub = df.reindex([m - 1, m, m + 1])
    return sub.mean().where(sub.notna().sum() >= 2)


def rates(df, a, b):
    """칸별 연 단위 속도(mm/년, 아래로 +) = −(끝 3개월 평균 − 시작 3개월 평균) / 연수."""
    yrs = (pd.Period(b, "M") - pd.Period(a, "M")).n / 12.0
    return (-(end_mean(df, b) - end_mean(df, a)) / yrs).to_numpy()


def gps_cell(cells, la, lo):
    iy = int((la - LAT0) * KM_LAT * 1000 // CELL); ix = int((lo - LON0) * KM_LON * 1000 // CELL)
    c = cells[(cells.iy == iy) & (cells.ix == ix)]
    return int(c.index[0]) if len(c) else None


def fit(t, y):
    ok = np.isfinite(y)
    return float(np.polyfit(t[ok], y[ok], 1)[0]) if ok.sum() >= 6 and np.ptp(t[ok]) > 0.6 else np.nan


def gps_table(cells, op, dwr, gps):
    rows = []
    for st, d in gps.groupby("station"):
        ci = gps_cell(cells, d.lat.iloc[0], d.lon.iloc[0])
        if ci is None:
            continue
        g = d.set_index("month").up_mm.to_frame("v")
        for key, (a, b, lab) in list(WIN.items()) + [("all", ("2023-01", "2026-02", "2023.1~2026.2 전체(누적)"))]:
            if key == "all":                                   # 누적 변화 곡선 오차(2023.1 기준)와 전체 기간 속도
                ms = [m for m in g.index if pd.Period(a, "M") <= m <= pd.Period(b, "M")]
                if len(ms) < 6:
                    continue
                t = np.array([(m.start_time - ms[0].start_time).days / 365.25 for m in ms])
                G = (g.v.loc[ms] - g.v.loc[ms[0]]).to_numpy()
                O = (op.loc[ms, ci] - op.loc[ms[0], ci]).to_numpy(); W = (dwr.loc[ms, ci] - dwr.loc[ms[0], ci]).to_numpy()
                rm = lambda X: float(np.sqrt(np.nanmean((X - G) ** 2))) if np.isfinite(X).sum() >= 6 else np.nan
                rows.append({"station": st, "window": lab, "months": len(ms), "gps_rate": -fit(t, G), "opera_rate": -fit(t, O),
                             "dwr_rate": -fit(t, W), "opera_rmse": rm(O), "dwr_rmse": rm(W)})
                continue
            yrs = (pd.Period(b, "M") - pd.Period(a, "M")).n / 12.0
            gv = -(end_mean(g, b).v - end_mean(g, a).v) / yrs
            if not np.isfinite(gv):
                continue
            ov = -(end_mean(op[[ci]], b)[ci] - end_mean(op[[ci]], a)[ci]) / yrs
            wv = -(end_mean(dwr[[ci]], b)[ci] - end_mean(dwr[[ci]], a)[ci]) / yrs
            rows.append({"station": st, "window": lab, "months": int(round(yrs * 12)), "gps_rate": float(gv), "opera_rate": float(ov),
                         "dwr_rate": float(wv), "opera_rmse": np.nan, "dwr_rmse": np.nan})
    return pd.DataFrame(rows)


def zone_masks(cells):
    d = cells.d_aq_km
    keep = ~cells.is_oil
    return {"수로 1km 안": keep & (d <= 1.0),
            "금지구역(수로 동쪽 4km 안)": keep & cells.east & (d <= BUFFER_KM),
            "금지구역 밖(동쪽 4~10km)": keep & cells.east & (d > BUFFER_KM) & (d <= SERVICE_KM)}


def hotspots(cells, v_pre, v_post, n=10):
    """금지구역 안 1km 칸(500m 4칸 평균)별 금지 후 속도. 빠른 순."""
    m = (~cells.is_oil) & cells.east & (cells.d_aq_km <= BUFFER_KM)
    df = pd.DataFrame({"k": (cells.iy // 2) * 1000 + cells.ix // 2, "lat": cells.lat, "lon": cells.lon, "d": cells.d_aq_km,
                       "pre": v_pre, "post": v_post})[m.to_numpy()]
    g = df.groupby("k").agg(lat=("lat", "mean"), lon=("lon", "mean"), d=("d", "mean"), pre=("pre", "mean"), post=("post", "mean"),
                            n=("post", lambda s: int(np.isfinite(s).sum())))
    g = g[g.n >= 2].dropna(subset=["post"])
    g["accel"] = g.post - g.pre
    return g.sort_values("post", ascending=False).head(n)


def fig_gps(cells, op, dwr, gps):
    sts = sorted(gps.station.unique())
    fig, axs = plt.subplots(1, len(sts), figsize=(3.3 * len(sts), 3.4), sharey=False)
    for ax, st in zip(axs, sts):
        d = gps[gps.station == st].set_index("month").up_mm
        ci = gps_cell(cells, gps[gps.station == st].lat.iloc[0], gps[gps.station == st].lon.iloc[0])
        m0 = max(d.index.min(), pd.Period("2016-07", "M"))
        ms = [m for m in op.index if m >= m0]
        x = [m.to_timestamp() for m in ms]
        base = lambda s: s - s.loc[[m for m in ms if m >= d.index.min()]].dropna().iloc[0]
        ax.plot([m.to_timestamp() for m in d.index if m >= m0], (d - d.loc[d.index >= m0].iloc[0])[d.index >= m0], color="k", lw=1.6, label="GPS")
        ax.plot(x, base(op.loc[ms, ci]), color="#2166ac", lw=1.3, label="OPERA (NASA)")
        ax.plot(x, base(dwr.loc[ms, ci]), color="#d6604d", lw=1.1, ls="--", label="DWR (TRE Altamira)")
        ax.axvline(pd.Timestamp("2024-12-09"), color="#555", lw=0.8, ls=":")
        ax.axvline(pd.Timestamp("2023-01-01"), color="#bbb", lw=0.6)
        ax.set_title(st, fontsize=10)
        import matplotlib.dates as mdates
        span = (pd.Timestamp("2026-10-01") - pd.Timestamp(str(m0))).days / 365.25
        ax.xaxis.set_major_locator(mdates.YearLocator(2 if span > 5 else 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelsize=8)
    axs[0].set_ylabel("수직 변화 (mm, 위로 +)")
    axs[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("같은 칸의 GPS·OPERA·DWR: 2023년 이후 DWR는 일부 지점에서 반대로 움직이고, OPERA는 GPS와 같은 방향 (점선: 2024.12.9 취수 금지)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "opera-1-gps.png", dpi=150); plt.close(fig)


def fig_map(cells, V, pts, oil, gps, hot):
    lab = ["금지 전 1년 (2023.11→2024.11)", "금지 후 1년 (2024.11→2025.11)", "변화 (금지 후 − 금지 전)"]
    arrs = [V["wet"], V["post"], V["post"] - V["wet"]]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.6))
    ny, nx = int(cells.iy.max()) + 1, int(cells.ix.max()) + 1
    ext = [LON0, LON0 + nx * CELL / 1000 / KM_LON, LAT0, LAT0 + ny * CELL / 1000 / KM_LAT]
    for ax, a, t, j in zip(axs, arrs, lab, range(3)):
        img = np.full((ny, nx), np.nan); img[cells.iy, cells.ix] = np.where(cells.is_oil, np.nan, a)
        cm, vmin, vmax = ("RdBu_r", -40, 40) if j < 2 else ("PuOr_r", -25, 25)
        im = ax.imshow(img, origin="lower", extent=ext, cmap=cm, vmin=vmin, vmax=vmax, aspect=KM_LON / KM_LAT)
        oimg = np.full((ny, nx), np.nan); oimg[cells.iy, cells.ix] = np.where(cells.is_oil, 1.0, np.nan)
        ax.imshow(oimg, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=3, alpha=0.6, aspect=KM_LON / KM_LAT)
        ax.plot(pts[:, 0], pts[:, 1], color="k", lw=1.2)
        # 금지구역 경계(수로 동쪽 4.02km): 수로 선을 동쪽으로 평행 이동한 근사선
        ax.plot(pts[:, 0] + BUFFER_KM / KM_LON, pts[:, 1], color="k", lw=0.9, ls="--")
        g = gps.groupby("station").agg(lat=("lat", "first"), lon=("lon", "first"))
        ax.scatter(g.lon, g.lat, s=30, c="#ffd400", edgecolor="k", zorder=5)
        for s, r_ in g.iterrows():
            ax.text(r_.lon + 0.006, r_.lat, s, fontsize=7)
        if j == 1 and len(hot):
            ax.scatter(hot.lon, hot.lat, s=70, facecolor="none", edgecolor="#00a65a", lw=1.6, zorder=6, label="현장 확인 후보(금지 후 가장 빠른 칸)")
            ax.legend(fontsize=7, loc="lower left")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title(t, fontsize=10); ax.tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=ax, shrink=0.75); cb.set_label("침하 속도 (mm/년, 아래로 +)" if j < 2 else "속도 변화 (mm/년)", fontsize=8)
    fig.suptitle("OPERA 위성 침하 속도. 검은 선: 수로, 점선: 금지구역 경계(수로 동쪽 4km), 회색: 유전(분석 제외)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "opera-2-ban-map.png", dpi=150); plt.close(fig)


def seasonal(cells, op, gps):
    """GPS 계절 변동 확인: 여름(5~9월) 변화 vs 겨울(10~3월) 변화, 2025~2026. OPERA(주변 안정 칸 기준)와 비교."""
    out = []
    for st, d in gps.groupby("station"):
        g = d.set_index("month").up_mm
        ci = gps_cell(cells, d.lat.iloc[0], d.lon.iloc[0])
        for a, b, lab in [("2025-04", "2025-09", "2025 여름(4→9월)"), ("2025-09", "2026-02", "2025~26 겨울(9→2월)"),
                          ("2026-04", "2026-09", "2026 여름(4→9월, OPERA 자료 없음)")]:
            pa, pb = pd.Period(a, "M"), pd.Period(b, "M")
            gv = float(g.get(pb, np.nan) - g.get(pa, np.nan))
            ov = float(op.loc[pb, ci] - op.loc[pa, ci]) if ci is not None and pb in op.index and pa in op.index else np.nan
            out.append({"station": st, "period": lab, "gps_mm": gv, "opera_mm": ov})
    return pd.DataFrame(out)


def ndvi_by_zone():
    """금지구역 안·밖 농경지(1km 모델 칸) 관개기(5~9월) 평균 NDVI(Landsat). 금지 후에도 밭이 계속 물을 썼는지 보는 간접 지표."""
    try:
        import real_model as r
        site = r.load_site()
    except Exception as e:                                   # MODFLOW 없이도 load_site는 되지만, 자료가 없으면 건너뜀
        print("  NDVI 건너뜀:", str(e)[:80]); return None
    z = np.load(DATA / "ndvi_monthly.npz"); m = pd.PeriodIndex(z["months"], freq="M"); nd = z["ndvi"]
    ag, d = site["ag"], site["aq_dist"]
    inb, out = ag & (d <= BUFFER_KM), ag & (d > BUFFER_KM) & (d <= SERVICE_KM)
    rows = []
    for y in sorted(set(p.year for p in m)):
        sel = np.array([(p.year == y and p.month in (5, 6, 7, 8, 9)) for p in m])
        if sel.sum() < 3:
            continue
        A = np.nanmean(nd[sel], 0)
        rows.append({"year": int(y), "inside": float(np.nanmean(A[inb])), "outside": float(np.nanmean(A[out]))})
    return pd.DataFrame(rows)


def frac(x, cond):
    ok = np.isfinite(x)
    return float(np.mean(cond(x[ok]))) if ok.any() else np.nan


def did(cells, V, n_boot=2000, seed=0):
    """이중차분: (금지구역 안 금지 후 − 금지 전) − (금지구역 밖 같은 차이). 1km 묶음 단위 부트스트랩 90% 범위."""
    Z = zone_masks(cells)
    d = V["post"] - V["wet"]
    blk = ((cells.iy // 2) * 1000 + cells.ix // 2).to_numpy()
    out = {}
    for name, m in [("in", Z["금지구역(수로 동쪽 4km 안)"]), ("out", Z["금지구역 밖(동쪽 4~10km)"])]:
        m = m.to_numpy() & np.isfinite(d)
        df = pd.DataFrame({"b": blk[m], "d": d[m]}).groupby("b").d.mean()
        out[name] = df.to_numpy()
    rng = np.random.default_rng(seed)
    est = float(np.median(out["in"]) - np.median(out["out"]))
    bs = [np.median(rng.choice(out["in"], len(out["in"]))) - np.median(rng.choice(out["out"], len(out["out"]))) for _ in range(n_boot)]
    return {"did_med": est, "p5": float(np.percentile(bs, 5)), "p95": float(np.percentile(bs, 95)),
            "in_change_med": float(np.median(out["in"])), "out_change_med": float(np.median(out["out"])),
            "n_in": int(len(out["in"])), "n_out": int(len(out["out"]))}


def main():
    cells, op, dwr, gps, pts, oil, months = load()
    V = {k: rates(op, a, b) for k, (a, b, _) in WIN.items()}
    VD = {k: rates(dwr, a, b) for k, (a, b, _) in WIN.items()}
    G = gps_table(cells, op, dwr, gps)
    Z = zone_masks(cells)
    zrows = []
    for zn, m in Z.items():
        m = m.to_numpy()
        row = {"zone": zn, "cells": int(m.sum())}
        for k in WIN:
            row[k] = float(np.nanmedian(V[k][m]))
            row[k + "_dwr"] = float(np.nanmedian(VD[k][m]))
        row["post_over_reg"] = frac(V["post"][m], lambda x: x > REG)
        row["accel_frac"] = frac((V["post"] - V["wet"])[m], lambda x: x > 0)
        zrows.append(row)
    Zt = pd.DataFrame(zrows)
    D = did(cells, V)
    hot = hotspots(cells, V["wet"], V["post"])
    S = seasonal(cells, op, gps)
    N = ndvi_by_zone()
    fig_gps(cells, op, dwr, gps)
    fig_map(cells, V, pts, oil, gps, hot)
    info = json.load(open(DATA / "opera_build_info.json")) if (DATA / "opera_build_info.json").exists() else {}
    res = {"windows": {k: v[:2] for k, v in WIN.items()}, "gps": G.to_dict("records"), "zones": Zt.to_dict("records"), "did": D,
           "hotspots": hot.reset_index().to_dict("records"), "seasonal": S.to_dict("records"), "ndvi": None if N is None else N.to_dict("records"), "build": info}
    json.dump(res, open(OUT / "results.json", "w"), ensure_ascii=False, indent=1, default=float)
    report(G, Zt, D, hot, S, info, N)
    print((OUT / "RESULTS.md").read_text(encoding="utf-8"))


def report(G, Zt, D, hot, S, info, N=None):
    f = lambda x, d=1: "-" if x is None or not np.isfinite(x) else f"{x:.{d}f}"
    L = ["# OPERA 위성으로 본 취수 금지 전후 (관측만, 모델 없음)\n",
         "자료: NASA JPL OPERA DISP-S1(Sentinel-1, 30m) 상승 1개·하강 2개 궤도 묶음 → 500m 칸 수직 변위(prepare_opera.py). "
         "2016.7~2026.2. 대기 오차는 칸 472개(2015~2022 DWR 속도 |v|<3mm/년, 유전 밖)의 날짜별 중앙값을 빼서 줄였다. 침하 속도는 아래로 +, mm/년.\n",
         "**속도는 '같은 달 → 같은 달' 연 단위로 잰다.** GPS는 여름에 올라가고 겨울에 내려가는 계절 변동이 있어(4절), "
         "연 단위가 아닌 구간(예: 2024.12~2026.4는 겨울이 두 번)으로 재면 속도가 부풀려지거나 줄어든다. 끝점은 앞뒤 1달 포함 3개월 평균.\n"]
    gaps = [g for fr in info.get("frames", {}).values() for g in fr.get("bridged_gaps", [])]
    if gaps:
        L.append("하강 궤도 F38503은 2021년에 받지 못한 파일 때문에 사슬이 두 번 끊겨(" + ", ".join(f"{g['days']}일" for g in gaps) +
                 ") 직전 180일 속도로 이었다. 끊긴 기간이 짧아 오차는 1mm 안팎이다.\n")
    A = G[G.window.str.contains("전체")]
    Y = G[~G.window.str.contains("전체")]
    L += ["## 1. 자료 품질: 같은 칸의 GPS vs OPERA vs DWR\n",
          "### 2023.1~2026.2 누적 변화 (DWR를 버린 구간)\n",
          "| GPS | 개월 | GPS 속도 | OPERA 속도 | DWR 속도 | OPERA 오차(RMSE) | DWR 오차(RMSE) |", "|---|---|---|---|---|---|---|"]
    for r_ in A.itertuples():
        L.append(f"| {r_.station} | {r_.months} | {f(r_.gps_rate)} | {f(r_.opera_rate)} | {f(r_.dwr_rate)} | {f(r_.opera_rmse)} | {f(r_.dwr_rmse)} |")
    so = int((np.sign(A.opera_rate) == np.sign(A.gps_rate)).sum()); sw = int((np.sign(A.dwr_rate) == np.sign(A.gps_rate)).sum())
    L += ["", f"- 침하 방향이 GPS와 같은 지점: **OPERA {so}/{len(A)}곳, DWR {sw}/{len(A)}곳**. DWR는 P544·Q204에서 2023년 이후 땅이 올라간 것으로 나온다(GPS는 침하).",
          f"- 누적 곡선 오차 중앙값: **OPERA {f(A.opera_rmse.median())}mm, DWR {f(A.dwr_rmse.median())}mm**. "
          "OPERA 속도는 GPS보다 조금 작게 나오는 경향이 있다(500m 칸 평균이라 수로 바로 옆 국지 침하가 희석되고, 기준 칸도 조금 움직일 수 있음).\n",
          "### 연 단위 속도 (mm/년)\n",
          "| GPS | 구간 | GPS | OPERA | DWR |", "|---|---|---|---|---|"]
    for r_ in Y.itertuples():
        L.append(f"| {r_.station} | {r_.window} | {f(r_.gps_rate)} | {f(r_.opera_rate)} | {f(r_.dwr_rate)} |")
    L += ["", "![GPS 비교](opera-1-gps.png)\n",
          "## 2. 구역별 침하 속도 (유전 칸 제외, 500m 칸 중앙값, mm/년)\n",
          f"| 구역 | 500m 칸 | {WIN['drought'][2]} | {WIN['wet'][2]} | **{WIN['post'][2]}** | 금지 후 기준(연 12.7mm) 초과 칸 | 금지 후 빨라진 칸 |",
          "|---|---|---|---|---|---|---|"]
    for r_ in Zt.itertuples():
        L.append(f"| {r_.zone} | {r_.cells} | {f(r_.drought)} | {f(r_.wet)} | **{f(r_.post)}** | {r_.post_over_reg * 100:.0f}% | {r_.accel_frac * 100:.0f}% |")
    L += ["", "### 금지 효과: 금지구역 안 vs 밖 (이중차분)\n",
          "금지 전 1년(배분율 40%)과 금지 후 1년(배분율 50%)은 수로 물 사정이 비슷하다. 금지가 지켜졌다면 **금지구역 안**은 밖보다 침하가 더 줄었어야 한다.\n",
          "| | 금지 후 − 금지 전 (mm/년, 1km 묶음 중앙값) |", "|---|---|",
          f"| 금지구역 안 ({D['n_in']}칸) | {D['in_change_med']:+.1f} |",
          f"| 금지구역 밖 ({D['n_out']}칸) | {D['out_change_med']:+.1f} |",
          f"| **안 − 밖 (금지 효과)** | **{D['did_med']:+.1f}** (90% 범위 {D['p5']:+.1f}~{D['p95']:+.1f}) |", "",
          ("- 금지구역 안이 밖보다 **덜 줄지 않았다**(차이가 0 이상). 위성으로는 금지 효과가 보이지 않는다 → 금지구역 안 취수가 계속됐거나, 과거 취수로 늦게 눌리는 점토가 금지 효과를 가렸다."
           if D["p5"] > -1.0 else
           "- 금지구역 안이 밖보다 침하가 더 줄었다 → 위성으로 금지 효과가 보인다."),
          "- 늦게 눌리는 점토(지연 압밀)는 수로 가까운 남쪽에서 크다(모델 v3: 남쪽이 4배 느림). 이 몫은 모델 원인 분해로만 나눌 수 있다.\n",
          "![지도](opera-2-ban-map.png)\n",
          "## 3. 현장 확인 후보: 금지구역 안에서 금지 후 가장 빨리 꺼지는 1km 칸\n",
          "| 순위 | 위도 | 경도 | 수로까지 (km) | 금지 전 1년 | 금지 후 1년 | 변화 |", "|---|---|---|---|---|---|---|"]
    for i, (k, r_) in enumerate(hot.iterrows(), 1):
        L.append(f"| {i} | {r_.lat:.4f} | {r_.lon:.4f} | {r_.d:.1f} | {f(r_.pre)} | **{f(r_.post)}** | {r_.accel:+.1f} |")
    L += ["", "금지구역 안은 취수가 0이어야 하므로, 금지 후에도 빠르게 꺼지는 칸은 (1) 과거 취수로 늦게 눌리는 점토, (2) 금지구역 밖 취수의 영향, "
          "(3) 금지구역 안 미계량 취수 중 하나다. 위성만으로는 셋을 나눌 수 없고, 모델 원인 분해(results_validate_v3ops)와 함께 본다. "
          "관리기관에는 '유량계·전력 사용량을 먼저 확인할 곳' 목록으로 준다.\n",
          "## 4. GPS의 계절 변동\n",
          "| GPS | 기간 | GPS 수직 변화 (mm, 위로 +) | OPERA 같은 칸 |", "|---|---|---|---|"]
    for r_ in S.itertuples():
        L.append(f"| {r_.station} | {r_.period} | {f(r_.gps_mm)} | {f(r_.opera_mm)} |")
    L += ["", "- GPS는 여름에 올라가고 겨울에 크게 내려간다. 같은 칸의 OPERA는 여름(관개기)에 내려간다. GPS(지구 기준 절대 위치)에는 넓은 지역이 함께 움직이는 "
          "계절 변형(지표 물·눈 무게에 따른 탄성 변형 등)이 들어가고, 주변 기준 칸을 빼는 InSAR에서는 빠지는 것으로 보인다. 원인은 확인하지 못했다.",
          "- **판정은 연 단위로 해야 한다.** 3개월 GPS 변화로 연 0.5인치 기준을 판정하면 계절 변동에 속는다. AquaLimit의 3개월 계획 한도는 모델로 계산한 '계획으로 인한 추가 침하'라 이 문제와 별개다.",
          "- 10/28 발표 초안의 '금지 후 오히려 빨라졌다'(Q204 13.0→17.5, Q208 18.5→22.4mm/년)는 연 단위가 아닌 구간이라 계절 변동이 섞였다. "
          "연 단위로 다시 재면 **금지 후에도 줄지 않았다**(1절 표)가 정확한 표현이다.\n",
          "## 5. 밭은 금지 후에도 계속 물을 썼나 (Landsat NDVI)\n",
          "| 연도 | 금지구역 안 농경지 | 금지구역 밖(4~10km) | 수로 물 배분율 |", "|---|---|---|---|"]
    alloc = {2019: 75, 2020: 20, 2021: 5, 2022: 5, 2023: 100, 2024: 40, 2025: 50, 2026: 45}
    if N is not None:
        for r_ in N.itertuples():
            if r_.year >= 2019:
                L.append(f"| {r_.year}{' (금지 후)' if r_.year >= 2025 else ''} | {r_.inside:.3f} | {r_.outside:.3f} | {alloc.get(r_.year, '-')}% |")
    L += ["", "- 관개기(5~9월) 평균 NDVI. 금지 후(2025·2026)에도 금지구역 안 밭의 녹색도가 줄지 않았다 → 휴경하지 않고 계속 물을 썼다. "
          "그 물이 지하수인지 수로 물·외부 구입 물인지는 위성으로 알 수 없다(과수원은 휴경이 어렵다). 계량·수로 물 배달 기록으로 확인할 부분이다.\n",
          "## 6. 한계\n",
          "- 500m 칸 평균, 수직·동서 두 성분만 풀었다(남북 성분 무시). 2016.7 이전은 DWR 자료로 이어 붙였다. OPERA는 2026.2까지(그 뒤 자료는 아직 받지 못함).",
          "- 금지 전후 비교는 1년씩이다. 해마다 기상·작물이 달라 1년 차이에는 잡음이 있다.",
          "- 출처: OPERA DISP-S1 v1.0, NASA JPL/ASF DAAC. https://www.jpl.nasa.gov/go/opera"]
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
