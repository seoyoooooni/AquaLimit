"""위성(Landsat 8·9) 식생지수 NDVI → 1km 모델 격자 월별 값.

- 출처: Microsoft Planetary Computer, Landsat Collection 2 Level-2 지표 반사도(무료, 계정 불필요)
- 이 지역은 Landsat 경로 042, 행 035·036 두 장면에 걸친다. 같은 날짜 두 장면을 붙여 쓴다.
- 달마다 구름이 가장 적은 날짜 1개를 골라 NDVI = (근적외 - 적색) / (근적외 + 적색)을 계산하고,
  구름·그림자 화소는 뺀다(QA_PIXEL). 90m로 읽어 1km 칸 평균을 낸다.
- 영상이 없는 달은 앞뒤 달로 보간한다.
출력: data/ndvi_monthly.npz (months, ndvi[달, ny, nx], scene_dates)
"""
import concurrent.futures as cf
import time

import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.warp import transform_bounds

import real_model as r

BBOX = [-119.93, 35.49, -119.55, 35.79]
OUT = r.DATA / "ndvi_monthly.npz"
DOWN = 3                                   # 30m → 90m로 읽기


def search(cat, month):
    t0 = month.start_time.strftime("%Y-%m-%d"); t1 = month.end_time.strftime("%Y-%m-%d")
    items = list(cat.search(collections=["landsat-c2-l2"], bbox=BBOX, datetime=f"{t0}/{t1}",
                            query={"platform": {"in": ["landsat-8", "landsat-9"]}, "eo:cloud_cover": {"lt": 40}}).items())
    by_date = {}
    for it in items:
        if it.properties.get("landsat:wrs_path") != "042":
            continue
        by_date.setdefault(it.datetime.date(), []).append(it)
    best = [(np.mean([i.properties["eo:cloud_cover"] for i in v]), d, v) for d, v in by_date.items()
            if {i.properties["landsat:wrs_row"] for i in v} >= {"035", "036"}]
    return sorted(best, key=lambda x: x[0])[0] if best else None


def read_scene(item, ny, nx):
    """장면 1개 → (합, 개수) 격자."""
    s = np.zeros((ny, nx)); n = np.zeros((ny, nx))
    with rasterio.open(item.assets["red"].href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *BBOX)
        win = rasterio.windows.from_bounds(*b, src.transform).round_offsets().round_lengths()
        shp = (max(int(win.height / DOWN), 1), max(int(win.width / DOWN), 1))
        red = src.read(1, window=win, out_shape=shp, boundless=True, fill_value=0).astype(float)
        tr = src.window_transform(win) * rasterio.Affine.scale(win.width / shp[1], win.height / shp[0])
        crs = src.crs
    with rasterio.open(item.assets["nir08"].href) as src:
        nir = src.read(1, window=win, out_shape=shp, boundless=True, fill_value=0).astype(float)
    with rasterio.open(item.assets["qa_pixel"].href) as src:
        qa = src.read(1, window=win, out_shape=shp, boundless=True, fill_value=1, resampling=rasterio.enums.Resampling.nearest)
    bad = (qa & (1 << 0)) | (qa & (1 << 1)) | (qa & (1 << 3)) | (qa & (1 << 4)) | (qa & (1 << 5))   # 채움·확장구름·구름·그림자·눈
    red = red * 2.75e-5 - 0.2; nir = nir * 2.75e-5 - 0.2
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
    ok = (bad == 0) & np.isfinite(ndvi) & (red > 0) & (nir > 0)
    rows, cols = np.nonzero(ok)
    xs, ys = rasterio.transform.xy(tr, rows, cols)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(np.array(xs), np.array(ys))
    iy = ((lat - r.LAT0) * r.KM_LAT * 1000 // r.MCELL).astype(int)
    ix = ((lon - r.LON0) * r.KM_LON * 1000 // r.MCELL).astype(int)
    k = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    np.add.at(s, (iy[k], ix[k]), np.clip(ndvi[rows, cols][k], -1, 1))
    np.add.at(n, (iy[k], ix[k]), 1.0)
    return s, n


def month_grid(cat, m, ny, nx):
    for attempt in range(3):
        try:
            hit = search(cat, m)
            if hit is None:
                return m, None, None
            cc, d, items = hit
            S = np.zeros((ny, nx)); N = np.zeros((ny, nx))
            for it in items:
                if it.properties["landsat:wrs_row"] in ("035", "036"):
                    s, n = read_scene(pc.sign(it), ny, nx); S += s; N += n
            g = np.where(N > 20, S / np.maximum(N, 1), np.nan)
            return m, g, str(d)
        except Exception as e:  # 네트워크 오류는 재시도
            print(f"  {m} 재시도 ({e})"); time.sleep(5)
    return m, None, None


def main():
    site = r.load_site()
    ny, nx = site["ny"], site["nx"]
    months = pd.period_range("2015-06", "2026-09", freq="M")
    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    out = {}
    t0 = time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        for m, g, d in ex.map(lambda m: month_grid(cat, m, ny, nx), months):
            out[m] = (g, d)
            print(f"  {m}: {'영상 ' + d if d else '영상 없음'}  ({time.time() - t0:.0f}초)", flush=True)
    arr = np.full((len(months), ny, nx), np.nan); dates = []
    for i, m in enumerate(months):
        g, d = out[m]; dates.append(d or "")
        if g is not None:
            arr[i] = g
    # 빈 달·빈 칸: 시간 방향 선형 보간
    t = np.arange(len(months))
    for iy in range(ny):
        for ix in range(nx):
            v = arr[:, iy, ix]; ok = np.isfinite(v)
            if ok.sum() >= 2:
                arr[:, iy, ix] = np.interp(t, t[ok], v[ok])
    arr = np.nan_to_num(arr, nan=0.1)
    np.savez_compressed(OUT, months=np.array([str(m) for m in months]), ndvi=arr.astype(np.float32), scene_dates=np.array(dates))
    print(f"저장 {OUT}: 영상 있는 달 {sum(1 for d in dates if d)}/{len(months)}, {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
