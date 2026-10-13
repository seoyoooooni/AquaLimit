"""USGS SSEBop 위성 실제 증발산(MODIS 기반, 월별, 약 1km) → 검증 지역만 잘라 저장.

출처: USGS FEWS NET / EROS, SSEBop ETa (Senay et al.)
https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/monthly/downloads/
SSEBop은 위성 지표면 온도와 기온(gridMET)·기준 증발산을 써서 실제 증발산을 계산한다. 그래서 '덥고 건조할수록 작물이 물을 더 쓴다'가 이 자료 안에 들어 있다.

출력: data/et_ssebop_monthly.npz  (months, lat, lon, et_mm[달, 위도, 경도])
"""
import io
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BASE = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/monthly/downloads"
BOX = (-119.97, 35.45, -119.50, 35.83)          # 서, 남, 동, 북 (모델 범위 + 여유)


def one(m):
    import rasterio
    from rasterio.windows import from_bounds
    url = f"{BASE}/m{m.year}{m.month:02d}.zip"
    for _ in range(3):
        try:
            raw = urllib.request.urlopen(url, timeout=120).read()
            break
        except Exception as e:                          # noqa: BLE001
            err = e
    else:
        print("실패", m, err, flush=True)
        return m, None, None
    z = zipfile.ZipFile(io.BytesIO(raw))
    name = [n for n in z.namelist() if n.endswith(".tif")][0]
    with rasterio.MemoryFile(z.read(name)) as mf, mf.open() as r:
        w = from_bounds(*BOX, transform=r.transform).round_offsets().round_lengths()
        a = r.read(1, window=w).astype(float)
        a[a == r.nodata] = np.nan
        t = r.window_transform(w)
    lon = t.c + (np.arange(a.shape[1]) + 0.5) * t.a
    lat = t.f + (np.arange(a.shape[0]) + 0.5) * t.e
    print("완료", m, flush=True)
    return m, a, (lat, lon)


def main(start="2014-10", end="2026-08"):
    months = pd.period_range(start, end, freq="M")
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, months))
    ok = [x for x in res if x[1] is not None]
    lat, lon = ok[0][2]
    et = np.full((len(months), len(lat), len(lon)), np.nan)
    for i, (m, a, _) in enumerate(res):
        if a is not None:
            et[i] = a
    np.savez_compressed(DATA / "et_ssebop_monthly.npz", months=np.array([str(m) for m in months]), lat=lat, lon=lon, et_mm=et)
    print("저장", et.shape, "결측 달", int(np.isnan(et).all((1, 2)).sum()))


if __name__ == "__main__":
    main(*sys.argv[1:])
