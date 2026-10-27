"""OPERA DISP-S1(30m, 위성 시선 방향) → 우리 500m 격자 월별 수직 침하.

단계
  1) reduce_file: 내려받은 30m 파일 하나를 500m 칸 평균(품질 마스크 적용)으로 줄여 저장(수십 KB). 원본은 지워도 된다.
  2) stitch: 기준 날짜가 바뀌는 묶음(ministack)을 이어 붙여 칸별 누적 시선 변위 시계열을 만든다.
  3) 기준점: 해마다 거의 안 움직이는 칸(DWR 장기 속도 |v| < 3mm/년, 유전 밖)의 중앙값을 날짜마다 빼서 대기 잡음(넓은 범위로 같이 움직이는 오차)을 줄인다.
  4) 상승(36541)·하강(38502·38503) 궤도를 합쳐 칸마다 수직(위로 +)·동서 성분을 푼다(남북 성분은 무시: 위성 궤도가 남북이라 거의 안 보임).
  5) DWR InSAR와 같은 형식으로 저장: 2016.7 전은 DWR, 이후는 OPERA 변화량을 2016.7~12 평균으로 DWR에 맞춰 붙인다.
     → data/opera_grid500m_monthly.npz (real_model: AQ_INSAR=opera)
  6) GPS와 비교표: data/opera_vs_gps.md

실행: python3 prepare_opera.py          (내려받은 파일 줄이기 + 만들기)
      python3 prepare_opera.py --build  (줄인 파일로 다시 만들기만)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OP = DATA / "opera"
RED = OP / "reduced"
LAT0, LON0, CELL = 35.50, -119.92, 500.0
KM_LAT = 110.95
KM_LON = 111.32 * np.cos(np.radians(35.64))
FRAMES = {"36541": "asc", "38502": "desc", "38503": "desc"}
MIN_PIX = 30                       # 500m 칸(30m 화소 약 278개) 중 품질 통과 화소가 이보다 적으면 비움
COH_MIN = 0.5
NAME = re.compile(r"_F(\d+)_VV_(\d{8}T\d{6})Z_(\d{8}T\d{6})Z_")


def grid_shape():
    c = pd.read_csv(DATA / "insar_grid500m_cells.csv")
    return int(c.iy.max()) + 1, int(c.ix.max()) + 1


_MAP = {}


def pixel_cells(frame, x, y, crs_wkt):
    """UTM 화소 → 500m 칸 번호(평탄화). 범위 밖은 -1."""
    key = (frame, len(x), len(y), float(x[0]), float(y[0]))
    if key in _MAP:
        return _MAP[key]
    from pyproj import Transformer
    tr = Transformer.from_crs(crs_wkt, "EPSG:4326", always_xy=True)
    X, Y = np.meshgrid(x, y)
    lon, lat = tr.transform(X, Y)
    ny, nx = grid_shape()
    iy = np.floor((lat - LAT0) * KM_LAT * 1000 / CELL).astype(int)
    ix = np.floor((lon - LON0) * KM_LON * 1000 / CELL).astype(int)
    ok = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    idx = np.where(ok, iy * nx + ix, -1)
    _MAP[key] = idx
    return idx


def reduce_file(path):
    """30m 파일 → 칸별 (합, 개수). 저장하고 저장 경로를 돌려준다."""
    import h5py
    path = Path(path)
    m = NAME.search(path.name)
    frame, ref, sec = m.group(1), m.group(2), m.group(3)
    out = RED / f"F{frame}" / (path.stem + ".npz")
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as f:
        d = f["displacement"][()].astype(np.float64) * 1000.0           # mm, 시선 방향
        ok = np.isfinite(d)
        for k, cond in [("recommended_mask", lambda a: a > 0.5), ("water_mask", lambda a: a > 0.5),
                        ("temporal_coherence", lambda a: a >= COH_MIN)]:
            if k in f:
                ok &= cond(f[k][()])
        crs = f["spatial_ref"].attrs["crs_wkt"]
        crs = crs.decode() if isinstance(crs, bytes) else crs
        idx = pixel_cells(frame, f["x"][()], f["y"][()], crs)
    ny, nx = grid_shape()
    sel = ok & (idx >= 0)
    s = np.bincount(idx[sel], weights=d[sel], minlength=ny * nx)
    n = np.bincount(idx[sel], minlength=ny * nx)
    np.savez_compressed(out, frame=frame, ref=ref, sec=sec, sum=s.astype(np.float32), cnt=n.astype(np.int32))
    return out


def los_cells(frame):
    """정적 시선 방향(동·북·위) → 칸 평균."""
    import rasterio
    with rasterio.open(OP / "static" / f"F{frame}_los_enu.tif") as src:
        a = src.read().astype(np.float64)
        T = src.transform
        cols, rows = np.meshgrid(np.arange(src.width), np.arange(src.height))
        x, y = T * (cols + 0.5, rows + 0.5)
        from pyproj import Transformer
        lon, lat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(x, y)
    ny, nx = grid_shape()
    iy = np.floor((lat - LAT0) * KM_LAT * 1000 / CELL).astype(int); ix = np.floor((lon - LON0) * KM_LON * 1000 / CELL).astype(int)
    ok = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx) & (np.abs(a[2]) > 0.1)
    idx = iy * nx + ix
    out = []
    for b in range(3):
        s = np.bincount(idx[ok], weights=a[b][ok], minlength=ny * nx); n = np.bincount(idx[ok], minlength=ny * nx)
        out.append(np.where(n > 0, s / np.maximum(n, 1), np.nan))
    return np.array(out)                                                  # (3, ny*nx)


def stitch(frame):
    """칸별 누적 시선 변위(mm) 시계열: {날짜: (ny*nx,)}."""
    files = sorted((RED / f"F{frame}").glob("*.npz"))
    recs = []
    for f in files:
        z = np.load(f)
        v = np.where(z["cnt"] >= MIN_PIX, z["sum"] / np.maximum(z["cnt"], 1), np.nan)
        recs.append((str(z["ref"]), str(z["sec"]), v))
    recs.sort(key=lambda r: (r[0], r[1]))
    cum = {}
    first = recs[0][0] if recs else None
    if first:
        cum[first] = np.zeros_like(recs[0][2])
    t = lambda s: pd.Timestamp(s[:8])
    bridged = []
    for ref, sec, v in recs:
        if ref not in cum:
            # 기준 날짜(ref)가 앞 묶음에 없다 = 받지 못한 파일 때문에 사슬이 끊겼다.
            # 끊긴 구간은 직전 180일 칸별 속도로 이어 붙인다(구간이 짧으면 오차 몇 mm). 길이를 기록한다.
            known = sorted((d for d in cum if t(d) < t(ref)), key=t)
            if not known:
                continue
            d0 = known[-1]
            win = [d for d in known if (t(d0) - t(d)).days <= 180]
            gap = (t(ref) - t(d0)).days
            if len(win) >= 3:
                x = np.array([(t(d) - t(d0)).days for d in win], float)
                Y = np.array([cum[d] for d in win])
                ok = np.isfinite(Y)
                xm = np.where(ok, x[:, None], np.nan)
                xbar, ybar = np.nanmean(xm, 0), np.nanmean(np.where(ok, Y, np.nan), 0)
                rate = np.nansum((xm - xbar) * (Y - ybar), 0) / np.maximum(np.nansum((xm - xbar) ** 2, 0), 1e-9)
                rate = np.where(ok.sum(0) >= 3, rate, 0.0)
            else:
                rate = 0.0
            cum[ref] = cum[d0] + rate * gap
            bridged.append((d0[:8], ref[:8], gap))
        cum[sec] = cum[ref] + v
    for a, b, g in bridged:
        print(f"  F{frame}: 사슬 끊김 {a}→{b} ({g}일)을 직전 속도로 이어 붙임")
    return cum, bridged


def build():
    ny, nx = grid_shape()
    cells = pd.read_csv(DATA / "insar_grid500m_cells.csv")
    flat = (cells.iy * nx + cells.ix).to_numpy()
    stable = flat[(cells.vel_mm_yr.abs() < 3.0).to_numpy() & cells.oil_field.isna().to_numpy()]
    print(f"기준(안정) 칸 {len(stable)}개")
    series, gaps = {}, {}
    for fr in FRAMES:
        cum, gaps[fr] = stitch(fr)
        if not cum:
            continue
        dates = sorted(cum)
        A = np.array([cum[d] for d in dates])                             # (날짜, 칸)
        A = A - np.nanmedian(A[:, stable], axis=1, keepdims=True)         # 기준점(안정 칸) 빼기
        mon = pd.PeriodIndex([pd.Timestamp(d[:8]).to_period("M") for d in dates], freq="M")
        M = pd.DataFrame(A).groupby(mon).mean()                           # 월 평균
        series[fr] = M
        print(f"  F{fr}({FRAMES[fr]}): 날짜 {len(dates)}개 {dates[0][:8]}~{dates[-1][:8]}, 월 {len(M)}개")
    # 하강 두 묶음 평균
    desc = [series[f] for f in series if FRAMES[f] == "desc"]
    asc = [series[f] for f in series if FRAMES[f] == "asc"]
    if not desc or not asc:
        sys.exit("상승·하강 자료가 모두 있어야 수직을 풀 수 있습니다.")
    D = pd.concat(desc).groupby(level=0).mean() if len(desc) > 1 else desc[0]
    Aa = asc[0]
    common = Aa.index.intersection(D.index)
    La = los_cells([f for f in FRAMES if FRAMES[f] == "asc"][0])
    Ld = np.nanmean([los_cells(f) for f in FRAMES if FRAMES[f] == "desc"], axis=0)
    # 칸마다 2×2: [ea ua; ed ud] [E; U] = [la; ld]
    det = La[0] * Ld[2] - Ld[0] * La[2]
    la = Aa.loc[common].to_numpy(); ld = D.loc[common].to_numpy()
    U = (La[0] * ld - Ld[0] * la) / det                                   # 위로 + (mm)
    E = (Ld[2] * la - La[2] * ld) / det
    # DWR 형식으로: 2016.7 전은 DWR, 이후는 OPERA 변화량을 DWR에 맞춰 붙임
    z = np.load(DATA / "insar_grid500m_monthly.npz")
    dmon = pd.PeriodIndex(z["months"], freq="M")
    dwr = pd.DataFrame(z["disp_mm"].T, index=dmon)                        # (월, 셀) 위로 +
    Up = pd.DataFrame(U[:, flat], index=common)
    anchor = [m for m in Up.index if pd.Period("2016-07", "M") <= m <= pd.Period("2016-12", "M") and m in dwr.index]
    off = (dwr.loc[anchor] - Up.loc[anchor]).mean()
    Up = Up + off
    months = dwr.index.union(Up.index)
    out = dwr.reindex(months).astype(np.float64)
    out.loc[out.index >= pd.Period("2016-07", "M")] = np.nan          # 2016.7 이후는 OPERA만(빈 달은 비움)
    out.loc[Up.index] = Up.to_numpy()
    np.savez_compressed(DATA / "opera_grid500m_monthly.npz", disp_mm=out.to_numpy().T.astype(np.float32),
                        months=np.array([str(m) for m in months]), cell=z["cell"], lat=z["lat"], lon=z["lon"],
                        east_mm=pd.DataFrame(E[:, flat], index=common).reindex(months).to_numpy().T.astype(np.float32))
    print(f"저장: data/opera_grid500m_monthly.npz 칸 {out.shape[1]}개, 월 {len(months)}개({months[0]}~{months[-1]})")
    cover = {fr: int(np.isfinite(series[fr].to_numpy()).any(0).sum()) for fr in series}
    json.dump({"frames": {fr: {"orbit": FRAMES[fr], "months": len(series[fr]), "cells_with_data": cover[fr],
                               "bridged_gaps": [{"from": a, "to": b, "days": g} for a, b, g in gaps.get(fr, [])]} for fr in series},
               "anchor": "2016-07~12 평균을 DWR에 맞춤", "stable_cells": int(len(stable))},
              open(DATA / "opera_build_info.json", "w"), ensure_ascii=False, indent=1)
    compare_gps(out, dwr)


def compare_gps(op, dwr):
    """GPS 5곳: 같은 500m 칸 OPERA·DWR 수직 변화 vs GPS (기간별 RMSE, 속도)."""
    g = pd.read_csv(DATA / "gps_monthly.csv"); g["month"] = pd.PeriodIndex(g.month, freq="M")
    cells = pd.read_csv(DATA / "insar_grid500m_cells.csv")
    L = ["# OPERA vs DWR vs GPS (수직, mm, 위로 +)\n",
         "같은 500m 칸. 기간 첫 달 기준 변화로 비교. RMSE가 작을수록 GPS와 맞는다.\n",
         "| GPS | 기간 | 개월 | OPERA RMSE | DWR RMSE | GPS 속도 | OPERA 속도 | DWR 속도 (mm/년) |", "|---|---|---|---|---|---|---|---|"]
    rows = []
    for st, d in g.groupby("station"):
        la, lo = d.lat.iloc[0], d.lon.iloc[0]
        iy = int((la - LAT0) * KM_LAT * 1000 // CELL); ix = int((lo - LON0) * KM_LON * 1000 // CELL)
        c = cells[(cells.iy == iy) & (cells.ix == ix)]
        if not len(c):
            continue
        ci = int(c.index[0])
        gs = d.set_index("month").up_mm
        for a, b in [("2016-07", "2022-12"), ("2023-01", "2026-04")]:
            ms = [m for m in gs.index if pd.Period(a, "M") <= m <= pd.Period(b, "M") and m in op.index]
            if len(ms) < 6:
                continue
            G = gs.loc[ms] - gs.loc[ms[0]]
            O = op.loc[ms, ci] - op.loc[ms[0], ci]
            W = dwr.reindex(ms)[ci] - dwr.reindex(ms)[ci].iloc[0]
            t = np.array([(m.start_time - ms[0].start_time).days / 365.25 for m in ms])
            rate = lambda s: np.polyfit(t[np.isfinite(s)], s[np.isfinite(s)], 1)[0] if np.isfinite(s).sum() > 5 else np.nan
            r = lambda s: float(np.sqrt(np.nanmean((s.to_numpy() - G.to_numpy()) ** 2)))
            rows.append((st, f"{a}~{b}", len(ms), r(O), r(W), rate(G.to_numpy()), rate(O.to_numpy()), rate(W.to_numpy())))
    for x in rows:
        L.append(f"| {x[0]} | {x[1]} | {x[2]} | {x[3]:.1f} | {x[4]:.1f} | {x[5]:.1f} | {x[6]:.1f} | {x[7]:.1f} |")
    (DATA / "opera_vs_gps.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def reduce_all(delete=False):
    n = 0
    for f in sorted(OP.glob("**/*.nc")):
        reduce_file(f); n += 1
        if delete:
            f.unlink()
    print(f"줄인 파일 {n}개")


if __name__ == "__main__":
    if "--build" not in sys.argv:
        reduce_all(delete="--delete" in sys.argv)
    build()
