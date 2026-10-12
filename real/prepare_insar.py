"""InSAR 포인트 자료 → 500m 격자 월별 누적 침하 + 그림.

입력: raw/insar_tile*.csv (fetch_data.py), data/wells.csv, data/gwl_measurements.csv, data/aqueduct_*.geojson
출력: data/insar_points_velocity.csv, data/insar_grid500m_monthly.npz, data/insar_grid500m_cells.csv,
      figures/real-1-velocity-map.png, figures/real-2-timeseries.png, data/SUMMARY.json
단위: 변위 mm(아래로 꺼지면 음수), 속도 mm/yr. 기준일 2015-06-13 이후 공통 구간 사용.
"""
import glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    try: font_manager.fontManager.addfont(p)
    except Exception: pass
for name in ["Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name; break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})

CELL = 500.0
LAT0, LON0 = 35.50, -119.92
KM_LAT = 110.95; KM_LON = 111.32 * np.cos(np.radians(35.64))

df = pd.concat([pd.read_csv(f, na_values=["NULL"]) for f in sorted(glob.glob("raw/insar_tile*.csv"))], ignore_index=True)
dcols = [c for c in df.columns if c.startswith("D")]
dates = pd.to_datetime([c[1:] for c in dcols])
df[["CODE", "LAT", "LON", "VEL_V"]].round(6).to_csv("data/insar_points_velocity.csv", index=False)

# 500m 격자 번호
x = (df.LON - LON0) * KM_LON * 1000; y = (df.LAT - LAT0) * KM_LAT * 1000
df["ix"] = (x // CELL).astype(int); df["iy"] = (y // CELL).astype(int)
nx, ny = df.ix.max() + 1, df.iy.max() + 1

# 월말 값: 각 점의 해당 월 마지막 관측
D = df[dcols].to_numpy(float)
month = dates.to_period("M"); months = month.unique()
M = np.full((len(df), len(months)), np.nan)
for k, m in enumerate(months):
    sub = D[:, month == m]
    ok = ~np.isnan(sub)
    last = np.where(ok.any(1), sub[np.arange(len(sub)), np.where(ok, np.arange(sub.shape[1]), -1).max(1)], np.nan)
    M[:, k] = last
# 기준월(2015-06)을 0으로. 기준월 값이 없는 점(늦게 관측 시작)은 시계열 평균에서 뺀다(속도 지도에는 포함).
k0 = list(months).index(pd.Period("2015-06", "M"))
M = M - M[:, [k0]]
ts_ok = ~np.isnan(M[:, k0])
M[~ts_ok] = np.nan

cell_id = df.iy.to_numpy() * nx + df.ix.to_numpy()
cells = pd.DataFrame({"cell": cell_id, "vel": df.VEL_V}).groupby("cell").agg(npts=("vel", "size"), vel_mm_yr=("vel", "mean"))
G = np.full((len(cells), len(months)), np.nan)
idx = {c: i for i, c in enumerate(cells.index)}
gi = np.array([idx[c] for c in cell_id])
for k in range(len(months)):
    v = M[:, k]; ok = ~np.isnan(v)
    s = np.bincount(gi[ok], v[ok], len(cells)); n = np.bincount(gi[ok], minlength=len(cells))
    G[:, k] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
cells = cells.reset_index()
cells["iy"] = cells.cell // nx; cells["ix"] = cells.cell % nx
cells["lat"] = LAT0 + (cells.iy + 0.5) * CELL / 1000 / KM_LAT
cells["lon"] = LON0 + (cells.ix + 0.5) * CELL / 1000 / KM_LON
cells["cum_mm_last"] = G[:, -1]
cells.round(5).to_csv("data/insar_grid500m_cells.csv", index=False)
np.savez_compressed("data/insar_grid500m_monthly.npz", disp_mm=G.astype(np.float32),
                    months=np.array([str(m) for m in months]), cell=cells.cell.to_numpy(),
                    lat=cells.lat.to_numpy(), lon=cells.lon.to_numpy())

# 유전 경계: 석유 생산으로 인한 침하는 지하수 침하와 구분해야 함
from shapely.geometry import shape, Point
from shapely.prepared import prep
oil = json.load(open("data/oil_fields_in_box.geojson"))
oilg = [(f["properties"]["NAME"], prep(shape(f["geometry"]))) for f in oil["features"]]
def oil_name(la, lo):
    pt = Point(lo, la)
    for n, g in oilg:
        if g.contains(pt): return n
    return ""
cells["oil_field"] = [oil_name(a, b) for a, b in zip(cells.lat, cells.lon)]
cells.round(5).to_csv("data/insar_grid500m_cells.csv", index=False)

# 수로 선형
aq = json.load(open("data/aqueduct_mp185_218.geojson"))
lines = []
for f in aq["features"]:
    g = f["geometry"]; parts = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
    lines += [(np.array(p)[:, 0], np.array(p)[:, 1], f["properties"]) for p in parts]
aq_pts = np.concatenate([np.c_[a, b] for a, b, _ in lines])

wells = pd.read_csv("data/wells.csv"); gwl = pd.read_csv("data/gwl_measurements.csv", parse_dates=["msmt_date"])
cnt = gwl.groupby("site_code").size(); wells["n"] = wells.site_code.map(cnt)

# 그림 1: 침하 속도 지도
fig, ax = plt.subplots(figsize=(7.2, 6.4))
sc = ax.scatter(df.LON, df.LAT, c=df.VEL_V, s=0.4, cmap="RdBu", vmin=-60, vmax=60, rasterized=True)
for n, g in oilg:
    pass
for f in oil["features"]:
    gg = shape(f["geometry"]); polys = getattr(gg, "geoms", [gg])
    for pg in polys:
        xx, yy = pg.exterior.xy; ax.plot(xx, yy, color="#555", lw=0.8, ls="--")
ax.plot([], [], color="#555", lw=0.8, ls="--", label="유전 경계(CalGEM)")
for a, b, p in lines: ax.plot(a, b, color="k", lw=2)
ax.scatter(wells.longitude, wells.latitude, marker="^", s=40, facecolor="gold", edgecolor="k", zorder=5, label=f"관측정 {len(wells)}곳")
ax.annotate("California Aqueduct\nMP 185~218", (aq_pts[:, 0].mean() - 0.02, aq_pts[:, 1].mean()), fontsize=9, ha="right")
cb = plt.colorbar(sc, ax=ax, shrink=0.8); cb.set_label("수직 변위 속도 (mm/년, 음수=침하)")
ax.set_xlabel("경도"); ax.set_ylabel("위도"); ax.set_title("검증 지역 InSAR 침하 속도 (DWR TRE Altamira, 2015~2026)")
ax.legend(loc="lower left", fontsize=8); ax.set_aspect(1 / np.cos(np.radians(35.64)))
fig.tight_layout(); fig.savefig("figures/real-1-velocity-map.png", dpi=160); plt.close(fig)

# 그림 2: 관측정 옆 500m 격자의 InSAR 누적 변위 vs 깊이별 수위
KL = KM_LON
def nearest_cell(la, lo):
    c = cells[(cells.oil_field == "") & (cells.npts >= 5) & cells.cum_mm_last.notna()]
    d = np.hypot((c.lon - lo) * KL, (c.lat - la) * KM_LAT)
    j = d.idxmin(); return cells.loc[j], float(d[j])
def dist_to_aq(la, lo):
    return float(np.hypot((aq_pts[:, 0] - lo) * KL, (aq_pts[:, 1] - la) * KM_LAT).min())
cells["d_aq_km"] = [dist_to_aq(a, b) for a, b in zip(cells.lat, cells.lon)]
cells.round(5).to_csv("data/insar_grid500m_cells.csv", index=False)
t = months.to_timestamp(how="end")
PANELS = [("355816N1196775", "수로 옆 다층 관측정(2023~)"), ("355944N1195814", "장기 관측정(2014~)")]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
picked = []
for ax1, (stem, title) in zip(axes, PANELS):
    ws = wells[wells.site_code.str.startswith(stem)]
    la, lo = ws.latitude.mean(), ws.longitude.mean()
    c, dkm = nearest_cell(la, lo)
    ax1.plot(t, G[int(c.name)], color="#b2182b", lw=2.2, label=f"InSAR 누적 변위 (격자까지 {dkm:.1f}km)")
    ax1.set_ylabel("누적 수직 변위 (mm, 2015.6 기준)", color="#b2182b")
    ax2 = ax1.twinx()
    for sc_, g in gwl[gwl.site_code.str.startswith(stem)].groupby("site_code"):
        dep = wells.set_index("site_code").well_depth.get(sc_)
        lab = f"수위 {sc_[-4:]}" + (f" (깊이 {dep:.0f}ft)" if pd.notna(dep) else "")
        ax2.plot(g.msmt_date, g.gwe, "o-", ms=2.5, lw=1, label=lab)
    ax2.set_ylabel("지하수위 표고 (ft)"); ax2.spines["right"].set_visible(True)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower left")
    ax1.set_title(f"{title} · 수로까지 {dist_to_aq(la, lo):.1f}km", fontsize=11)
    picked.append({"well": stem, "cell_lat": round(float(c.lat), 4), "cell_lon": round(float(c.lon), 4),
                   "cell_vel_mm_yr": round(float(c.vel_mm_yr), 1), "cell_cum_mm_last": round(float(c.cum_mm_last), 0),
                   "well_to_cell_km": round(dkm, 2), "well_to_aqueduct_km": round(dist_to_aq(la, lo), 2)})
fig.suptitle("침하(InSAR)와 지하수위 — 같은 장소, 같은 기간", fontsize=13)
fig.tight_layout(); fig.savefig("figures/real-2-timeseries.png", dpi=160); plt.close(fig)

along_all = cells[(cells.d_aq_km < 2) & (cells.npts >= 10)]
along = along_all[along_all.oil_field == ""]
summ = {
    "points": int(len(df)), "cells_500m": int(len(cells)), "months": [str(months[0]), str(months[-1])],
    "vel_mm_yr": {"median": round(float(df.VEL_V.median()), 1), "p5": round(float(df.VEL_V.quantile(0.05)), 1)},
    "aqueduct_2km_cells_all": int(len(along_all)), "aqueduct_2km_cells_in_oilfield": int((along_all.oil_field != "").sum()),
    "aqueduct_2km_oilfield_vel_mm_yr_median": round(float(along_all[along_all.oil_field != ""].vel_mm_yr.median()), 1),
    "aqueduct_2km_cells": int(len(along)),
    "aqueduct_2km_vel_mm_yr": {"median": round(float(along.vel_mm_yr.median()), 1), "min": round(float(along.vel_mm_yr.min()), 1)},
    "aqueduct_2km_cum_mm_last": {"median": round(float(np.nanmedian(along.cum_mm_last)), 0), "min": round(float(np.nanmin(along.cum_mm_last)), 0)},
    "note": "aqueduct_2km_* 통계는 유전 밖 격자만 사용",
    "timeseries_points_with_2015_06_ref": int(ts_ok.sum()),
    "figure2_panels": picked,
    "wells": int(len(wells)), "wells_ge20": int((wells.n >= 20).sum()), "gwl_measurements": int(len(gwl)),
    "gwl_period": [str(gwl.msmt_date.min().date()), str(gwl.msmt_date.max().date())],
}
json.dump(summ, open("data/SUMMARY.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(summ, ensure_ascii=False, indent=1))
