"""USGS CVHM2(센트럴밸리 지하수 모델)에서 검증 지역 층 구조·지질을 뽑아 data/cvhm2_box.npz로 저장.

CVHM2: Traum, Faunt, Boyce (2024), MODFLOW-OWHM Used to Characterize the Groundwater Flow System of the
Central Valley, California. USGS data release, https://doi.org/10.5066/P14N8VFT

- model.zip(334MB)을 내려받아 필요한 배열만 읽는다(한 번만 실행하면 됨, 결과 npz는 약 100KB).
- CVHM2 격자: 441행 × 98열 × 13층, 1마일(1,609m), 회전 격자. Modelgeoref.txt의 네 모서리로 위치를 계산.
- 우리 모델 격자(1km) 각 칸에 가장 가까운 CVHM2 칸 값을 쓴다.

뽑는 것 (층 묶음 4개로 합침)
  G1 상부 대수층   = CVHM2 4~5층 (코코란 점토 위)
  G2 코코란 점토   = CVHM2 6~8층
  G3 하부 대수층 위 = CVHM2 9~10층
  G4 하부 대수층 아래 = CVHM2 11~13층
  각 묶음: 윗면·아랫면 표고(m), 조립질 비율, 압축 가능한 세립질 두께(m)

CVHM2 침하 계수(남부 구역 'S', MULT.txt): 세립질 비탄성 비저류 4.415e-4 /m, 탄성 2.5e-5 /m,
조립질 탄성 9.42e-7 /m, 코코란 점토 비탄성 2.0e-4, 탄성 8.25e-6.
"""
import io
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
URL = ("https://www.sciencebase.gov/catalog/file/get/65bd367fd34e18c6baf32758?f=__disk__"
       "af%2Fcf%2F2b%2Fafcf2bc619f00665f243fc167d782a3ebc563515")
NR, NC, NL = 441, 98, 13
CORNERS = {"ul": (-123.793, 39.978), "ur": (-122.209, 40.729), "lr": (-118.065, 35.271), "ll": (-119.577, 34.572)}
GROUPS = {"G1": [4, 5], "G2": [6, 7, 8], "G3": [9, 10], "G4": [11, 12, 13]}
COEF = dict(SSFV=4.4153541e-4, SSFE=2.5e-5, SSCE=9.4227193e-07, SSFV_CC=2.0e-4, SSFE_CC=8.25e-6)


def cvhm_lonlat():
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4269", "EPSG:26911", always_xy=True)
    inv = Transformer.from_crs("EPSG:26911", "EPSG:4269", always_xy=True)
    U = {k: np.array(tr.transform(*v)) for k, v in CORNERS.items()}
    r, c = np.meshgrid(np.arange(NR) + 0.5, np.arange(NC) + 0.5, indexing="ij")
    u, v = (c / NC)[..., None], (r / NR)[..., None]
    P = (1 - u) * (1 - v) * U["ul"] + u * (1 - v) * U["ur"] + u * v * U["lr"] + (1 - u) * v * U["ll"]
    lon, lat = inv.transform(P[..., 0], P[..., 1])
    return np.asarray(lon), np.asarray(lat)


def main(zip_path=None):
    import real_model as r
    if zip_path is None:
        zip_path = HERE / "raw" / "cvhm2_model.zip"
        if not zip_path.exists():
            import urllib.request
            zip_path.parent.mkdir(exist_ok=True)
            print("CVHM2 model.zip 내려받는 중 (334MB)...")
            urllib.request.urlretrieve(URL, zip_path)
    z = zipfile.ZipFile(zip_path)

    def rd(name):
        return np.array(z.read("model/model_arrays/" + name).split(), float).reshape(NR, NC)

    tops = [rd("lay4_tms.txt") if k == 4 else rd(f"lay{k}t_m.txt") for k in range(1, NL + 1)] + [rd("lay13b_m.txt")]
    lon, lat = cvhm_lonlat()

    # 우리 1km 격자 중심
    ny, nx = 32, 31
    gy, gx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    mlat = r.LAT0 + (gy + 0.5) * r.MCELL / 1000 / r.KM_LAT
    mlon = r.LON0 + (gx + 0.5) * r.MCELL / 1000 / r.KM_LON
    near = (np.abs(lat - 35.64) < 0.5) & (np.abs(lon + 119.75) < 0.5)
    ci = np.argwhere(near)
    d = (lat[near][None, :] - mlat.ravel()[:, None]) ** 2 + ((lon[near][None, :] - mlon.ravel()[:, None]) * 0.81) ** 2
    pick = ci[d.argmin(1)]                                       # 우리 칸마다 가장 가까운 CVHM2 (행, 열)
    R, C = pick[:, 0].reshape(ny, nx), pick[:, 1].reshape(ny, nx)

    out = {"cvhm_row": R, "cvhm_col": C}
    for g, lays in GROUPS.items():
        top = tops[lays[0] - 1][R, C]; bot = tops[lays[-1]][R, C]
        thk_c = np.zeros((ny, nx)); thk_f_nd = np.zeros((ny, nx)); thk_f_d = np.zeros((ny, nx)); thk = np.zeros((ny, nx))
        for L in lays:
            t = np.maximum(tops[L - 1] - tops[L], 0)[R, C]
            act = rd(f"lay{L}.txt")[R, C] != 0
            pc = rd(f"pc_lyr{L:02d}.txt")[R, C] / 100.0
            beq = rd(f"Beq{L}.txt")[R, C] * 0.3048; neq = rd(f"Neq{L}.txt")[R, C]
            t = np.where(act, t, 0.0)
            fine = (1 - pc) * t
            dly = np.minimum(beq * neq, fine) if L not in (6, 7, 8) else 0 * fine
            thk += t; thk_c += pc * t; thk_f_d += np.where(act, dly, 0); thk_f_nd += np.where(act, fine - dly, 0)
        out[f"{g}_top"] = top; out[f"{g}_bot"] = bot; out[f"{g}_thk"] = thk
        out[f"{g}_pc"] = np.where(thk > 0, thk_c / np.maximum(thk, 1e-9), 0.5)
        out[f"{g}_fine"] = thk_f_nd + thk_f_d                         # 압축 가능한 세립질 총두께(지연+비지연)
    crit = rd("IC_CHD_lo.txt")[R, C]
    out["crit_head_lo"] = np.where(crit > -9000, crit, np.nan)
    for k, v in COEF.items():
        out[k] = v
    np.savez_compressed(HERE / "data" / "cvhm2_box.npz", **out)
    print("저장: data/cvhm2_box.npz")
    for g in GROUPS:
        print(f"  {g}: 두께 중앙값 {np.median(out[g + '_thk']):.0f}m, 조립질 {np.median(out[g + '_pc']) * 100:.0f}%, "
              f"세립질 {np.median(out[g + '_fine']):.0f}m")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
