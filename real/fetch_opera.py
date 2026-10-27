"""NASA OPERA DISP-S1(Sentinel-1 지표 변위, 30m) 자료를 우리 지역만 잘라 받는다.

왜: DWR InSAR(TRE Altamira)는 2023년 이후 GPS와 반대로 나와서 버렸다. OPERA는 같은 Sentinel-1을 NASA JPL이 따로 처리한
     자료로, 2016.7~2026.2까지 있고 상승·하강 두 궤도가 있어 수직(침하)과 동서 움직임을 나눌 수 있다.
우리 지역을 덮는 묶음(frame): 36541(상승, 현지 19시경), 38502·38503(하강, 현지 6시경). 전체는 수백 GB라 범위만 잘라 받는다.

준비(한 번만)
  1) https://urs.earthdata.nasa.gov 무료 가입
  2) ~/.netrc 에 한 줄:  machine urs.earthdata.nasa.gov login 아이디 password 비밀번호   (chmod 600 ~/.netrc)
  3) pip install "opera-utils[disp]" rasterio

실행
  python3 fetch_opera.py --test     # 2016년 몇 장만 받아 로그인·형식 확인(몇 분)
  python3 fetch_opera.py            # 전체 기간(시간이 걸림, 멈추면 다시 실행하면 이어 받음)
결과: data/opera/reduced/F<번호>/*.npz (500m 칸으로 줄인 변위, 원본 .nc는 받는 즉시 지움 — 맥 저장 공간 절약),
      data/opera/static/F<번호>_los_enu.tif (위성 시선 방향)
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "opera"
BBOX = (-119.95, 35.48, -119.55, 35.81)            # 서, 남, 동, 북 (모델 범위보다 조금 넓게)
FRAMES = {"36541": "상승", "38502": "하강", "38503": "하강"}
STATIC = "https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L3_DISP-S1-STATIC_V1/OPERA_L3_DISP-S1-STATIC_F{f}_20140403_S1A_v1.0/OPERA_L3_DISP-S1-STATIC_F{f}_20140403_S1A_v1.0_line_of_sight_enu.tif"


def check_login():
    netrc = Path.home() / ".netrc"
    if not netrc.exists() or "urs.earthdata.nasa.gov" not in netrc.read_text():
        sys.exit("~/.netrc 에 Earthdata 로그인이 없습니다. 파일 맨 위 '준비'를 보세요.")


def fetch_static(f):
    """위성 시선 방향(동·북·위 성분) 래스터: 통째로 받아(curl + Earthdata 로그인) 우리 범위만 잘라 저장하고 원본은 지운다."""
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
    dst = OUT / "static" / f"F{f}_los_enu.tif"
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    full = dst.parent / f"_full_F{f}.tif"
    ck = str(Path.home() / ".urs_cookies")
    subprocess.run(["curl", "-n", "-L", "-f", "-c", ck, "-b", ck, "-o", str(full), STATIC.format(f=f)], check=True)
    with rasterio.open(full) as src:
        b = transform_bounds("EPSG:4326", src.crs, *BBOX)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(window=win)
        prof = src.profile.copy()
        prof.update(width=arr.shape[2], height=arr.shape[1], transform=src.window_transform(win), driver="GTiff")
        with rasterio.open(dst, "w", **prof) as o:
            o.write(arr)
    full.unlink()
    print(f"  F{f} 시선 방향 저장: {dst.name} {arr.shape}")


def fetch_disp(f, test=False):
    """3개월씩 받아서 바로 500m 칸으로 줄이고(prepare_opera.reduce_file) 원본은 지운다. 끝난 기간은 표시 파일로 건너뛴다."""
    import pandas as pd
    import prepare_opera as po
    tmp = OUT / "_tmp" / f"F{f}"
    tmp.mkdir(parents=True, exist_ok=True)
    done_dir = OUT / "_done"; done_dir.mkdir(parents=True, exist_ok=True)
    for old in list((OUT / f"F{f}").glob("*.nc")) + list(tmp.glob("*.nc")):   # 예전에 통째로 받아 둔 파일·끊긴 파일
        try:
            po.reduce_file(old); old.unlink()
        except Exception as e:
            print(f"  (줄이기 실패, 지우고 다시 받음: {old.name}: {str(e)[:80]})"); old.unlink()
    starts = pd.date_range("2016-07-01", "2026-12-31", freq="3MS")
    if test:
        starts = starts[:2]
    for a in starts:
        b = a + pd.DateOffset(months=3)
        mark = done_dir / f"F{f}_{a:%Y%m}"
        if mark.exists():
            continue
        cmd = ["opera-utils", "disp-s1-download", "--bbox", *map(str, BBOX), "--frame-id", f, "--output-dir", str(tmp),
               "--num-workers", "4", "--start-datetime", f"{a:%Y-%m-%d}", "--end-datetime", f"{b:%Y-%m-%d}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        files = sorted(tmp.glob("*.nc"))
        if r.returncode != 0 and not files:
            msg = (r.stderr or r.stdout or "")[-300:]
            if "No products" in msg or "Found 0" in msg or "no results" in msg.lower():
                mark.touch(); continue
            print(f"  F{f} {a:%Y-%m}: 실패, 다음에 다시 — {msg}"); continue
        for x in files:
            po.reduce_file(x); x.unlink()
        if r.returncode == 0:
            mark.touch()
        n = len(list((po.RED / f"F{f}").glob("*.npz")))
        print(f"  F{f}({FRAMES[f]}) {a:%Y-%m}~: {len(files)}개 받아 줄임 (누적 {n}개)", flush=True)


if __name__ == "__main__":
    test = "--test" in sys.argv
    check_login()
    for f in FRAMES:
        try:
            fetch_static(f)
        except Exception as e:                      # 보조 자료라 실패해도 계속(나중에 다시 실행하면 이어 받음)
            print(f"  F{f} 시선 방향 받기 실패(나중에 다시): {str(e)[:200]}")
        fetch_disp(f, test)
    print("끝. 다음: python3 prepare_opera.py --build")
