"""실제 검증 지역(California Aqueduct MP 195~215) MODFLOW 6 + CSUB 모델과 ES-MDA 역산.

흐름
  1) 자료 읽기: 500m 격자 InSAR 월별 누적 침하, 관측정 수위, 수로·유전, 강수(NOAA 관측소 2곳)
  2) 모델 만들기: 62×56 격자(500m), 3개 층(상부 대수층 / 점토 차수층 / 하부 취수 대수층 + 압축층)
  3) 입력: 강수 → 함양, 농경지(수로 동쪽) 취수 = 기본량 × 관개기 계절 × 가뭄 계수. 취수량은 공개 자료가 없어 역산으로 추정
  4) 역산: ES-MDA(앙상블 다중 자료동화). E2 = 관측정 수위만, E3 = 수위 + InSAR
  5) 검증: 2015.7~2022.12 InSAR + 관측정으로 보정, 2023.1~2026.4는 독립 자료(GPS 수직 변위, 관측정 수위)와 비교
     (DWR InSAR는 2023년 이후 GPS와 크게 어긋나 검증 자료로 쓰지 않는다)

단위: m, 일, 침하 mm(아래로 양수). 수위는 관측정별 첫 측정 대비 변화(m)로 비교한다(절대 표고 기준 차이를 피함).
"""
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MF6 = os.environ.get("MF6_EXE") or shutil.which("mf6") or str(Path.home() / "bin" / "mf6")

# prepare_insar.py와 같은 격자 정의
CELL = 500.0               # InSAR 격자
F = 2                      # 모델 격자 = InSAR 격자 2×2 (1km). Colab에서 1회 실행 약 5초
MCELL = CELL * F
LAT0, LON0 = 35.50, -119.92
KM_LAT = 110.95
KM_LON = 111.32 * np.cos(np.radians(35.64))

REF_MONTH = "2015-06"      # InSAR 기준월(누적 침하 0)
CAL_END = "2022-12"        # 보정 기간 끝. 2023년 이후 InSAR는 GPS와 어긋나서(아래 GPS 비교) 보정에 쓰지 않는다
GPS_STATIONS = ["P544", "Q198", "Q204", "Q208", "Q213"]   # 수로를 따라 있는 GPS(NGL 처리, IGS20)
END = "2026-04"            # 자료 끝
OBS_EVERY = 3              # InSAR는 3개월마다 사용(자료 수 줄이기)

# 층 구조(가정값, CVHM2 반영 전 단순화)
TOP, BOT = 0.0, [-60.0, -80.0, -300.0]
K1, SS1 = 5.0, 1.5e-3        # 상부층: 비피압 저류(약 0.09)를 피압 저류로 근사
SS3 = 5e-6
IB_THICK = 60.0              # 하부층 안 압축층(점토) 두께
INELASTIC_RATIO = 10.0
SIG_INSAR = 5.0              # mm, 관측 잡음 + 모델 오차(1km 평균값 기준)
SIG_HEAD = 0.5               # m

STRUCTURE = os.environ.get("AQ_STRUCTURE", "cvhm2")   # "cvhm2"(USGS CVHM2 층 구조, 기본) 또는 "simple"(10/12 가정 3층)

# 역산 파라미터: 이름, 사전 평균, 사전 표준편차
SK_ZONES = 3                 # 3×3 구역
PARAMS_SIMPLE = ([(f"log10_Sk_z{i}", np.log10(0.02), 0.5) for i in range(SK_ZONES * SK_ZONES)] + [
    ("log10_K3", np.log10(5.0), 0.5),          # 하부층 수평 투수계수 m/일
    ("log10_K2v", np.log10(1e-4), 1.0),        # 차수층 연직 투수계수 m/일
    ("log10_pump", np.log10(0.03), 0.5),       # 농경지 평균 지하수 취수 깊이 m/년 (서부 Kern은 수로 물 의존이 커서 작게 시작)
    ("dry_beta", 0.5, 0.5),                    # 가뭄 해 취수 증가 계수
    ("log10_rch", np.log10(0.1), 0.4),         # 강수 중 함양 비율
])
# CVHM2 구조: 압축 계수는 CVHM2 값 × 구역별 보정 배수. 투수계수는 조립질 비율로 계산(조립질·세립질 K를 추정)
PARAMS_CVHM2 = ([(f"log10_mSk_z{i}", 0.0, 0.5) for i in range(SK_ZONES * SK_ZONES)] + [
    ("log10_Kc", np.log10(20.0), 0.5),         # 조립질(모래·자갈) 투수계수 m/일
    ("log10_Kf", np.log10(0.05), 0.7),         # 세립질(점토·실트) 투수계수 m/일
    ("log10_Kcorc", np.log10(1.01e-5), 1.0),   # 코코란 점토 연직 투수계수 m/일 (CVHM2 VK_CORC)
    ("log10_pump", np.log10(0.1), 0.6),        # 농경지 평균 지하수 취수 깊이 m/년
    ("dry_beta", 0.5, 0.5),                    # 가뭄 해 취수 증가 계수
    ("log10_rch", np.log10(0.1), 0.4),         # 강수 중 함양 비율
])
PARAMS_CVHM2_BASE = list(PARAMS_CVHM2)
# 취수량 추정 방식: "et"(위성 증발산 기반, 기본) 또는 "uniform"(농경지 균일 × 계절 × 가뭄 계수, 이전 방식)
PUMPING = os.environ.get("AQ_PUMPING", "et")
if STRUCTURE == "cvhm2" and PUMPING == "et":
    # 칸별 지하수 취수 = 배수 × max(위성 증발산 − 유효강수, 0) × (1 − 수로물 비율 × 그해 SWP 배분율)
    # 지하수 취수 = (위성 증발산으로 설명되는 밭별 취수) + (지역 공통 취수: 증발산으로 설명 안 되는 부분, 이전 방식)
    PARAMS_CVHM2 = PARAMS_CVHM2[:14] + [
        ("log10_pmult", 0.0, 0.3),             # 관개 수요/순증발산 배수(관개 효율·위성 오차 보정)
        ("log10_sw", np.log10(0.3), 0.4),      # SWP 100% 배분일 때 수로물 공급 깊이 m/년(관리 지역 농경지 평균)
        ("log10_rch", np.log10(0.1), 0.4),
    ]
# 모델 변형(새 버전 et에서만): v2 = 10/13 버전(results/), v3 = 최종 후보(지연 압밀 + 과거 이력 spin-up + 관개 회귀수 함양 + 현실적 사전범위)
#   입력별 기여 실험: v3_nolandsat(밭별 차이 없음), v3_noweather(해마다 같은 기상), v3_noalloc(해마다 같은 수로 배분율)
VARIANT = os.environ.get("AQ_VARIANT", "v3") if (STRUCTURE == "cvhm2" and PUMPING == "et") else "base"
V3 = VARIANT.startswith("v3")
OPS = VARIANT.endswith("ops")   # 운영 모델(v3ops): 2023~2026 GPS까지 보정에 쓴다 → 계획·정책·AI용. 검증 점수는 v3(2022년까지 보정)로 말한다
SIG_GPS_CAL = 2.0               # mm, (GPS가 없을 때 기본값)
GPS_WEIGHT = 0.5                # 운영 모델 보정: GPS 전체 가중치 = InSAR 전체 가중치 × 0.5
ABLATE = VARIANT.split("_", 1)[1] if "_" in VARIANT else ""
BAN_MONTH = "2024-12"        # 수로 양쪽 2.5마일 취수 금지 시행(2024.12.9)
BUFFER_KM = 4.02
SPIN_YEARS = 10              # v3: 2005.7~2015.6 과거 취수 이력(반년 단위)을 먼저 돌려 2015년 시작 상태에 '진행 중인 침하'를 담는다
DELAY_BEQ = 10.0             # v3: 지연 압밀 등가 점토층 두께(m). 코코란 점토(G2)는 층 전체를 한 덩어리로 둔다
if V3:
    PARAMS_CVHM2 = PARAMS_CVHM2_BASE[:10] + [
        ("log10_Kf", np.log10(0.02), 0.4),     # 세립질 투수계수: 3σ 상한 0.3m/일(점토·실트로 현실적인 범위)
        PARAMS_CVHM2_BASE[11], PARAMS_CVHM2_BASE[12], PARAMS_CVHM2_BASE[13],
        ("log10_pmult", 0.0, 0.3),
        ("log10_sw", np.log10(0.3), 0.4),
        ("log10_rch", np.log10(0.05), 0.25),   # 강수 함양 비율: 3σ 상한 약 0.28(연 강수 150mm 반건조 지역)
        ("log10_rf", np.log10(0.15), 0.15),    # 관개 회귀수 비율(뿌린 물 중 땅속으로 스며드는 비율): 3σ 범위 0.05~0.42
        ("log10_Kvdb", np.log10(1e-5), 0.7),   # 지연 압밀 점토층 연직 투수계수 m/일(북쪽 절반) → 침하가 늦게 따라오는 시간
        ("log10_Kvdb_s", np.log10(1e-5), 0.7), # 같은 값, 남쪽 절반(Q204·Q208·Q213 쪽). GPS: 북쪽은 2023년에 바로 반등, 남쪽은 계속 침하
        ("log10_spin", 0.0, 0.3),              # 2005~2015 과거 취수 = 2015~2022 계절 평균 × 이 배수
    ]
    if VARIANT.endswith("ops"):
        # 운영 모델: 2024.12 취수 금지 이후 실제 취수가 위성 추정과 얼마나 달랐는지 GPS로 거꾸로 추정한다(계량 기록 비공개)
        PARAMS_CVHM2 = PARAMS_CVHM2 + [
            ("log10_post_in", 0.0, 0.5),       # 금지구역(수로 양쪽 2.5마일) 안 취수 배수, 2024.12~
            ("log10_post_out", 0.0, 0.5),      # 금지구역 밖 취수 배수, 2024.12~
        ]
PARAMS = PARAMS_CVHM2 if STRUCTURE == "cvhm2" else PARAMS_SIMPLE
NAMES = [p[0] for p in PARAMS]
PRIOR_MEAN = np.array([p[1] for p in PARAMS])
PRIOR_SD = np.array([p[2] for p in PARAMS])

# ---------------------------------------------------------------- 모델 버전과 결과 폴더
# 취수 추정 방식(AQ_PUMPING)마다 결과를 다른 폴더에 둔다. 서로 다른 버전 결과가 섞이지 않게 폴더마다 model_version.json을 남기고 확인한다.
#   et(새 버전, 기본)    : results/          results_plan_et/  results_ai_et/
#   uniform(이전 버전)   : results_uniform/  results_plan/     results_ai/
RESULT_DIRS = {"et": {"model": "results", "plan": "results_plan_et", "ai": "results_ai_et"},
               "uniform": {"model": "results_uniform", "plan": "results_plan", "ai": "results_ai"}}


def model_version():
    return {"structure": STRUCTURE, "pumping": PUMPING, "et_source": ET_SOURCE if PUMPING == "et" else None,
            "variant": VARIANT if PUMPING == "et" else None, "param_names": NAMES}


def result_dir(kind, create=True):
    """kind: model / plan / ai. 지금 버전(AQ_PUMPING)에 맞는 결과 폴더."""
    if STRUCTURE != "cvhm2":
        d = HERE / {"model": "results_simple", "plan": "results_plan_simple", "ai": "results_ai_simple"}[kind]
    else:
        d = HERE / RESULT_DIRS[PUMPING][kind]
        if PUMPING == "et" and VARIANT != "v2":
            d = HERE / {"model": f"results_{VARIANT}", "plan": f"results_plan_{VARIANT}", "ai": f"results_ai_{VARIANT}"}[kind]
    if create:
        d.mkdir(exist_ok=True)
    return d


def write_version(d):
    json.dump(model_version(), open(Path(d) / "model_version.json", "w"), ensure_ascii=False, indent=1)


def check_version(d):
    """폴더의 결과가 지금 모델 버전으로 만든 것인지 확인. 다르면 어떻게 실행해야 하는지 알려 주고 멈춘다."""
    d = Path(d); f = d / "model_version.json"
    if f.exists():
        v = json.load(open(f)); names = v.get("param_names")
    elif (d / "results.json").exists():
        names = json.load(open(d / "results.json")).get("params", {}).get("names"); v = {}
    else:
        return
    cur = model_version()
    diff = [k for k in ("structure", "pumping", "et_source", "variant") if k in v and v[k] != cur[k]]
    if names is not None and list(names) != list(NAMES):
        diff.append("param_names")
    if diff:
        other = v.get("pumping") or ("uniform" if names is not None and len(names) == len(PARAMS_CVHM2_BASE) else "et")
        env = f"AQ_PUMPING={other}" + (f" AQ_ET={v['et_source']}" if v.get("et_source") else "") + \
              (f" AQ_VARIANT={v['variant']}" if v.get("variant") else (" AQ_VARIANT=v2" if other == "et" else ""))
        raise RuntimeError(f"{d.name}/ 결과는 다른 모델 버전으로 만든 것입니다(다른 항목: {', '.join(diff)}). "
                           f"폴더: {dict((k, v.get(k)) for k in ('structure', 'pumping', 'et_source'))}, "
                           f"지금: {dict((k, cur[k]) for k in ('structure', 'pumping', 'et_source'))}. "
                           f"'{env} python ...'로 실행하거나 결과를 다시 만드세요.")


# ---------------------------------------------------------------- 자료
def load_site():
    cells = pd.read_csv(DATA / "insar_grid500m_cells.csv")
    z = np.load(DATA / "insar_grid500m_monthly.npz")
    months = pd.PeriodIndex(z["months"], freq="M")
    k0, k1 = list(months).index(pd.Period(REF_MONTH, "M")), list(months).index(pd.Period(END, "M"))
    disp = -z["disp_mm"][:, k0:k1 + 1].astype(float)              # 아래로 양수
    months = months[k0:k1 + 1]
    nx, ny = (int(cells.ix.max()) + 1 + F - 1) // F, (int(cells.iy.max()) + 1 + F - 1) // F   # 모델 격자 크기
    oil = cells.oil_field.fillna("").astype(str).str.len() > 0
    use = (~oil) & (cells.npts >= 5)

    wells = pd.read_csv(DATA / "wells.csv")
    gwl = pd.read_csv(DATA / "gwl_measurements.csv", parse_dates=["msmt_date"])
    wells["ix"] = ((wells.longitude - LON0) * KM_LON * 1000 // MCELL).astype(int)
    wells["iy"] = ((wells.latitude - LAT0) * KM_LAT * 1000 // MCELL).astype(int)
    wells["layer"] = np.where(wells.well_depth.fillna(400) < 200, 0, 2)
    cv = load_cvhm2() if STRUCTURE == "cvhm2" else None
    if cv is not None:                                             # 관측정 바닥 표고가 들어가는 층 묶음
        bot_el = (wells.gse.fillna(300) - wells.well_depth.fillna(400)) * 0.3048
        iy = wells.iy.clip(0, ny - 1).to_numpy(); ix = wells.ix.clip(0, nx - 1).to_numpy()
        bots = np.stack([cv[f"G{g}_bot"][iy, ix] for g in range(1, 5)], 1)
        wells["layer"] = np.minimum((bot_el.to_numpy()[:, None] < bots).sum(1), 3)
    rows = []
    for sc, g in gwl.groupby("site_code"):
        g = g[(g.msmt_date >= months[1].start_time) & (g.msmt_date <= months[-1].end_time)].sort_values("msmt_date")
        if len(g) < 3 or sc not in set(wells.site_code):
            continue
        g = g.assign(m=g.msmt_date.dt.to_period("M")).groupby("m").gwe.mean()   # 월 평균
        ref = g.index[0]
        for m, v in g.items():
            if m == ref:
                continue
            rows.append({"site_code": sc, "month": m, "ref_month": ref, "dh_m": (v - g.iloc[0]) * 0.3048})
    hobs = pd.DataFrame(rows).merge(wells[["site_code", "ix", "iy", "layer"]], on="site_code")

    # 수로 동쪽 = 농경지(취수 지역). 유전 격자는 제외
    aq = json.load(open(DATA / "aqueduct_mp185_218.geojson"))
    pts = []
    for f in aq["features"]:
        g = f["geometry"]
        for part in (g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]):
            pts += [c[:2] for c in part]
    pts = np.array(pts); pts = pts[np.argsort(pts[:, 1])]
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    lat_c = LAT0 + (gy + 0.5) * MCELL / 1000 / KM_LAT
    lon_c = LON0 + (gx + 0.5) * MCELL / 1000 / KM_LON
    aq_lon = np.interp(lat_c, pts[:, 1], pts[:, 0])
    oil_frac = np.zeros((ny, nx)); cnt = np.zeros((ny, nx))
    np.add.at(oil_frac, (cells.iy // F, cells.ix // F), oil.to_numpy().astype(float))
    np.add.at(cnt, (cells.iy // F, cells.ix // F), 1.0)
    ag = (lon_c > aq_lon + 0.005) & (oil_frac < 0.5 * np.maximum(cnt, 1))
    aq_dist = np.array([np.min(np.hypot((pts[:, 0] - a) * KM_LON, (pts[:, 1] - b) * KM_LAT))
                        for a, b in zip(lon_c.ravel(), lat_c.ravel())]).reshape(lat_c.shape)

    return dict(cells=cells, disp=disp, months=months, nx=nx, ny=ny, use=use.to_numpy(), hobs=hobs,
                ag=ag, aq_dist=aq_dist, buf=ag & (aq_dist <= BUFFER_KM), precip=load_precip(months), gps=load_gps(months), aq_pts=pts, lat_c=lat_c, lon_c=lon_c, cv=cv,
                **load_et(months, lat_c, lon_c))


P_EFF = 0.7                  # 강수 중 작물이 쓰는 비율(유효강수)
ET_BASE_YEARS = (2023, 2025)  # 미래 달 증발산: 이 기간 같은 달 평균(최근 작부 반영)


ET_SOURCE = os.environ.get("AQ_ET", "landsat")   # "landsat"(Landsat NDVI × gridMET 기준증발산, 기본) 또는 "ssebop"


def kc_from_ndvi(ndvi):
    """작물계수 Kcb = 1.44·NDVI − 0.1 (Johnson & Trout 2012, NASA SIMS 방식)."""
    return np.clip(1.44 * ndvi - 0.1, 0.0, 1.2)


def load_et(months, lat_c, lon_c):
    """작물 증발산 → 모델 격자, SWP 배분율, 기온.
    landsat: 작물 증발산 = 작물계수(Landsat NDVI, fetch_landsat.py) × 기준 증발산(gridMET: 기온·습도·바람·일사로 계산)
    ssebop : USGS SSEBop 실제 증발산(MODIS, fetch_et.py)"""
    if ET_SOURCE == "landsat":
        zn = np.load(DATA / "ndvi_monthly.npz"); ze = np.load(DATA / "eto_gridmet_monthly.npz")
        nm = list(pd.PeriodIndex(zn["months"], freq="M")); em = pd.PeriodIndex(ze["months"], freq="M")
        em = [m for m in em if m in set(nm)]
        eidx = {m: i for i, m in enumerate(pd.PeriodIndex(ze["months"], freq="M"))}
        KC = np.array([kc_from_ndvi(zn["ndvi"][nm.index(m)]) for m in em])
        ETO = np.array([ze["eto_mm"][eidx[m]] for m in em])
        if ABLATE == "nolandsat":       # 위성 밭별 정보 없음: 달마다 지역 평균 작물계수 하나
            KC = np.repeat(np.nanmean(KC, axis=(1, 2), keepdims=True), KC.shape[1], 1).repeat(KC.shape[2], 2)
        if ABLATE == "noweather":       # 기상(기온 등) 해마다 같음: 같은 달 평균 기준증발산
            mo = np.array([m.month for m in em])
            ETO = np.array([ETO[mo == k].mean(0) for k in mo])
        E = KC * ETO
        em = pd.PeriodIndex(em, freq="M")
    else:
        z = np.load(DATA / "et_ssebop_monthly.npz")
        em = pd.PeriodIndex(z["months"], freq="M")
        iy = np.abs(z["lat"][:, None, None] - lat_c[None]).argmin(0)
        ix = np.abs(z["lon"][:, None, None] - lon_c[None]).argmin(0)
        E = np.nan_to_num(z["et_mm"][:, iy, ix])                      # (달, ny, nx) mm/달
    et = pd.Series(list(E), index=em)
    clim = {m: np.mean([et[pd.Period(f"{y}-{m:02d}", "M")] for y in range(ET_BASE_YEARS[0], ET_BASE_YEARS[1] + 1)], 0)
            for m in range(1, 13)}
    al = pd.read_csv(DATA / "swp_allocation.csv").set_index("year").alloc_pct / 100.0
    if ABLATE == "noalloc":             # 수로 배분율 정보 없음: 해마다 평균값
        al = al * 0 + float(al.mean())
    t = pd.read_csv(DATA / "temp_monthly.csv", index_col=0)
    t.index = pd.PeriodIndex(t.index, freq="M")
    tclim = t.groupby(t.index.month).mean()
    return dict(et=np.array([et[m] for m in months]), et_clim=clim, et_all=et, alloc=al, temp=t, temp_clim=tclim)


def hargreaves_factor(month, dT):
    """기온이 dT(°C) 오를 때 기준 증발산(Hargreaves) 증가 배수. 최고·최저 기온이 같이 오른다고 본다."""
    return 1.0 + dT / (T_CLIM_MEAN[month] + 17.8)


T_CLIM_MEAN = {}


def load_precip(months):
    """NOAA 일 강수(Lost Hills, Buttonwillow) → 월 강수 mm. 관측일 20일 미만인 달은 결측, 두 관측소 평균, 남은 결측은 월별 평년값."""
    cache = DATA / "precip_monthly.csv"
    if cache.exists():
        p = pd.read_csv(cache, index_col=0).squeeze("columns")
        p.index = pd.PeriodIndex(p.index, freq="M")
        return p.reindex(months).to_numpy()
    out = []
    for st in ["USC00045151", "USC00041244"]:
        url = ("https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries"
               f"&stations={st}&startDate=2014-10-01&endDate=2026-08-31&dataTypes=PRCP&format=csv&units=metric")
        d = pd.read_csv(url, parse_dates=["DATE"])
        m = d.set_index("DATE").PRCP.resample("MS").agg(["sum", "count"])
        s = m["sum"].where(m["count"] >= 20)
        s.index = s.index.to_period("M"); out.append(s)
    p = pd.concat(out, axis=1).mean(axis=1)
    clim = p.groupby(p.index.month).mean()
    p = p.fillna(pd.Series(p.index.month, index=p.index).map(clim))
    p.rename("precip_mm").to_csv(cache)
    return p.reindex(months).to_numpy()


def load_cvhm2():
    """fetch_cvhm2.py가 만든 data/cvhm2_box.npz (CVHM2 층 구조·지질을 우리 1km 격자에 옮긴 것)."""
    f = DATA / "cvhm2_box.npz"
    if not f.exists():
        raise FileNotFoundError("data/cvhm2_box.npz 없음: python3 fetch_cvhm2.py 먼저 실행")
    z = np.load(f)
    return {k: z[k] for k in z.files}


def load_gps(months):
    """NGL(네바다대) GPS 수직 위치 → 월 중앙값(mm, 위로 양수). 캐시: data/gps_monthly.csv"""
    cache = DATA / "gps_monthly.csv"
    if not cache.exists():
        llh = pd.read_csv("https://geodesy.unr.edu/NGLStationPages/llh.out", sep=r"\s+", header=None,
                          names=["st", "lat", "lon", "h"])
        rows = []
        for st in GPS_STATIONS:
            g = pd.read_csv(f"https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/{st}.tenv3", sep=r"\s+")
            up = pd.Series((g["u0(m)"] + g["____up(m)"]).to_numpy() * 1000,
                           index=pd.to_datetime(g["YYMMMDD"], format="%y%b%d")).resample("ME").median().dropna()
            la, lo = llh.set_index("st").loc[st, ["lat", "lon"]]
            rows += [{"station": st, "month": str(t.to_period("M")), "up_mm": v, "lat": la, "lon": lo} for t, v in up.items()]
        pd.DataFrame(rows).to_csv(cache, index=False)
    g = pd.read_csv(cache)
    g["month"] = pd.PeriodIndex(g.month, freq="M")
    return g[g.month.isin(months)]


# ---------------------------------------------------------------- 모델
def zone_index(site):
    gx, gy = np.meshgrid(np.arange(site["nx"]), np.arange(site["ny"]))
    zx = np.minimum(gx * SK_ZONES // site["nx"], SK_ZONES - 1)
    zy = np.minimum(gy * SK_ZONES // site["ny"], SK_ZONES - 1)
    return zy * SK_ZONES + zx                                   # (ny, nx), iy 기준(남→북), 모델 격자


def forcing(site, theta, extra_months=0, future_dry=0.0, future=None):
    """월별 함양(m/일)과 취수(m3/일). extra_months: 자료 끝(2026.4) 뒤로 더 계산할 달 수(계획용).
    PUMPING="et": 취수는 칸별 배열 (달, ny, nx). future: {"dT": 기온 변화 °C, "alloc": SWP 배분율, "pfac": 강수 배수}
    PUMPING="uniform": 취수는 농경지 칸 공통 값 (달,). future_dry로 가뭄 계수."""
    p = dict(zip(NAMES, theta))
    future = future or {}
    months = site["months"][1:]
    prcp = site["precip"][1:]
    if extra_months:
        fut = pd.period_range(site["months"][-1] + 1, periods=extra_months, freq="M")
        clim = pd.Series(site["precip"], index=[m.month for m in site["months"]]).groupby(level=0).mean()
        prcp = np.concatenate([prcp, clim.reindex([m.month for m in fut]).to_numpy() * future.get("pfac", 1.0)])
        months = months.append(fut)
    days = np.array([m.days_in_month for m in months], float)
    rch = 10 ** p["log10_rch"] * prcp / 1000.0 / days
    if PUMPING == "et" and STRUCTURE == "cvhm2":
        n_hist = len(site["months"]) - 1
        E = list(site["et"][1:])
        if extra_months:
            if not T_CLIM_MEAN:
                tc = site["temp_clim"]; T_CLIM_MEAN.update({m: float((tc.tmax[m] + tc.tmin[m]) / 2) for m in tc.index})
            dT = future.get("dT", 0.0)
            E += [site["et_clim"][m.month] * hargreaves_factor(m.month, dT) for m in months[n_hist:]]
        E = np.array(E)                                               # (달, ny, nx) mm/달
        al = site["alloc"]
        a = np.array([al.get(m.year, al.iloc[-1]) for m in months], float)
        if extra_months and "alloc" in future:
            a[n_hist:] = future["alloc"]
        net = np.maximum(E - P_EFF * prcp[:, None, None], 0.0) * site["ag"][None]   # 관개로 채워야 하는 증발산 mm/달
        need = 10 ** p["log10_pmult"] * net                           # 관개 수요 mm/달
        # 수로물은 물 공급구역 단위로 정해진 양(계약량 × 그해 배분율)이 들어오고, 달별로는 수요에 비례해 나눠 쓴다.
        # 수로물이 수요보다 적은 만큼을 지하수로 채운다 → 배분율이 낮은 가뭄 해에 지하수 취수가 크게 는다.
        yr = np.array([m.year for m in months])
        tot = need.sum((1, 2))
        sw = np.zeros(len(months))
        n_ag = site["ag"].sum()
        # 같은 해, 같은 배분율인 달끼리 묶는다. 과거 달은 해마다 배분율이 하나라 해별 묶음과 같다.
        # 미래 시나리오에서 배분율을 바꾸면(future["alloc"]) 그해 과거 달과 미래 달이 따로 묶여 미래 설정값이 반영된다.
        # 과거 달과 미래(계획) 달도 따로 묶는다 → 계획 기간을 붙여도 과거 달의 취수가 바뀌지 않는다.
        fut_flag = np.arange(len(months)) >= n_hist
        grp = [(int(y), float(v), bool(f)) for y, v, f in zip(yr, a, fut_flag)]
        for g in dict.fromkeys(grp):
            k = np.array([x == g for x in grp])
            sw_g = 10 ** p["log10_sw"] * 1000.0 * g[1] * n_ag * (k.sum() / 12.0)   # mm × 칸
            sw[k] = sw_g * tot[k] / max(tot[k].sum(), 1e-9)
        frac_gw = np.clip(1 - sw / np.maximum(tot, 1e-9), 0.0, 1.0)
        gw = need * frac_gw[:, None, None]                            # 지하수로 채우는 양 mm/달
        q = gw / 1000.0 / days[:, None, None] * MCELL * MCELL         # m3/일/칸
        q = q + base_uniform(site, p, months, extra_months, future.get("dry", future_dry))[:, None, None]
        q = q * site["ag"][None]
        if "log10_post_in" in p:                                      # 운영 모델: 금지 이후 취수 배수(안/밖)
            kb = np.array([m >= pd.Period(BAN_MONTH, "M") for m in months])
            mult = np.where(site["buf"], 10 ** p["log10_post_in"], 10 ** p["log10_post_out"])
            q = q.copy(); q[kb] = q[kb] * mult[None]
        if V3:                                                        # 관개 회귀수: 뿌린 물(수로물+지하수)의 일부가 다시 스며든다
            rch = rch[:, None, None] + 10 ** p["log10_rf"] * need / 1000.0 / days[:, None, None]
        return days, np.maximum(rch, 0), q
    return days, np.maximum(rch, 0), base_uniform(site, p, months, extra_months, future_dry)


def base_uniform(site, p, months, extra_months=0, future_dry=0.0):
    """농경지 칸 공통 취수(m3/일/칸) = 기본량 × 관개기 계절 × (1 + 가뭄 계수 × 그해 가뭄 정도)."""
    season = np.where(np.isin([m.month for m in months], [4, 5, 6, 7, 8, 9]), 1.6, 0.4)
    wy = np.array([m.year + (1 if m.month >= 10 else 0) for m in site["months"][1:]])
    allp = pd.Series(site["precip"][1:], index=wy).groupby(level=0).sum()
    dry = np.clip((allp.mean() - allp) / allp.mean(), -1, 1)
    wy_all = np.array([m.year + (1 if m.month >= 10 else 0) for m in months])
    dval = dry.reindex(wy_all).to_numpy().copy()
    if extra_months:
        dval[-extra_months:] = future_dry
    q = (10 ** p["log10_pump"] / 365.25) * season * (1 + p["dry_beta"] * np.nan_to_num(dval))
    return np.maximum(q, 0) * MCELL * MCELL


def run_model(site, theta, ws=None, keep=False, plan=None):
    """MODFLOW 6 실행 → (침하 mm (T,ny,nx), 층별 수위 (T,nlay,ny,nx)). 행은 iy(남→북) 순서.
    plan: {"months": 계획 달 수, "wells": [(iy, ix), ...], "rates": (달, 우물) m3/일, "dry": 미래 가뭄 계수}
          자료 끝 뒤로 계획 기간을 이어 붙이고, 계획 우물 취수를 더한다(하부 대수층 위 60% / 아래 40%)."""
    if STRUCTURE == "cvhm2":
        return run_model_cvhm2(site, theta, ws, keep, plan)
    if plan is not None:
        raise NotImplementedError("계획 계산은 CVHM2 구조에서만 지원")
    import flopy
    p = dict(zip(NAMES, theta))
    ny, nx = site["ny"], site["nx"]
    own = ws is None
    ws = ws or tempfile.mkdtemp(prefix="aq_real_")
    days, rch, q = forcing(site, theta)
    sim = flopy.mf6.MFSimulation(sim_name="aq", sim_ws=ws, exe_name=MF6, verbosity_level=0)
    flopy.mf6.ModflowTdis(sim, nper=len(days), perioddata=[(d, 1, 1.0) for d in days], time_units="days")
    flopy.mf6.ModflowIms(sim, complexity="moderate", outer_dvclose=1e-5, inner_dvclose=1e-6, linear_acceleration="bicgstab")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="aq")
    flopy.mf6.ModflowGwfdis(gwf, nlay=3, nrow=ny, ncol=nx, delr=MCELL, delc=MCELL, top=TOP, botm=BOT)
    k3 = 10 ** p["log10_K3"]; k2v = 10 ** p["log10_K2v"]
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=[K1, k2v * 10, k3], k33=[K1 / 10, k2v, k3 / 10])
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.0, transient={0: True})
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    edge = [(0, r, c) for r in range(ny) for c in range(nx) if r in (0, ny - 1) or c in (0, nx - 1)]
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data={0: [[e, 0.0] for e in edge]})
    flopy.mf6.ModflowGwfrcha(gwf, recharge={k: float(rch[k]) for k in range(len(days))})
    ag = np.argwhere(site["ag"])                                  # (iy, ix)
    wel = {k: [[(2, ny - 1 - iy, ix), -float(q[k])] for iy, ix in ag] for k in range(len(days))}
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel)
    sk = 10 ** np.array([p[f"log10_Sk_z{i}"] for i in range(SK_ZONES ** 2)])[zone_index(site)]   # (iy, ix)
    sskv = sk / IB_THICK
    pdata, n = [], 0
    for r in range(ny):
        for c in range(nx):
            v = sskv[ny - 1 - r, c]
            pdata.append([n, (2, r, c), "nodelay", 0.0, IB_THICK, 1.0, v, v / INELASTIC_RATIO, 0.3, 999.0, 0.0]); n += 1
    flopy.mf6.ModflowGwfcsub(gwf, head_based=True, initial_preconsolidation_head=True, ninterbeds=n,
                             cg_ske_cr=[SS1, 1e-5, SS3], cg_theta=0.3, packagedata=pdata, compaction_filerecord="aq.cmp")
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="aq.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    ok, buff = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError("MODFLOW 실패: " + " ".join(buff[-10:]))
    hf = flopy.utils.HeadFile(os.path.join(ws, "aq.hds"))
    H = np.array(hf.get_alldata())[:, :, ::-1, :]                 # (T,3,iy,ix)
    cf = flopy.utils.HeadFile(os.path.join(ws, "aq.cmp"), text="CSUB-COMPACTION")
    S = np.array(cf.get_alldata()).sum(axis=1)[:, ::-1, :] * 1000.0
    if own and not keep:
        shutil.rmtree(ws, ignore_errors=True)
    return S, H


def run_model_cvhm2(site, theta, ws=None, keep=False, plan=None, robust=False):
    """CVHM2 층 구조 4개 묶음(상부 대수층 / 코코란 점토 / 하부 대수층 위·아래) 모델.
    먼저 빠른 해법(moderate)으로 풀고, 수렴하지 않으면 강건한 해법(complex)으로 다시 푼다."""
    if robust is False:
        try:
            return run_model_cvhm2(site, theta, ws, keep, plan, robust=None)
        except RuntimeError:
            return run_model_cvhm2(site, theta, None, keep, plan, robust=True)
    import flopy
    p = dict(zip(NAMES, theta)); cv = site["cv"]
    ny, nx = site["ny"], site["nx"]
    own = ws is None
    ws = ws or tempfile.mkdtemp(prefix="aq_cv_")
    # plan["start"]가 있으면 과거 기간 안에서 계획(과거 재현), 없으면 자료 끝 뒤로 계획 기간을 붙인다
    in_hist = bool(plan) and plan.get("start") is not None
    nplan = plan["months"] if plan and not in_hist else 0
    days, rch, q = forcing(site, theta, nplan, plan.get("dry", 0.0) if plan else 0.0, plan.get("future") if plan else None)
    if plan:
        n0 = plan["start"] if in_hist else len(days) - plan["months"]
        plan_k = range(n0, n0 + plan["months"])
        if plan.get("replace_cells") and q.ndim == 3:                # 계획 대상 칸의 기존 취수는 계획으로 대체
            q = q.copy()
            for iy, ix in plan["replace_cells"]:
                q[list(plan_k), iy, ix] = 0.0
        if plan.get("zero_from") is not None and q.ndim == 3:        # 시나리오: 어느 달부터 지정 칸 취수를 멈춤(금지 효과 등)
            q = q.copy()
            k0z = plan["zero_from"]
            for iy, ix in plan.get("zero_cells", []):
                q[k0z:, iy, ix] = 0.0
    ns = 0
    if V3:                                                           # 과거 취수 이력 spin-up(2005.7~2015.6, 반년 단위)
        days, rch, q, ns = add_spinup(site, p, days, rch, q)
        if plan:
            plan_k = range(plan_k.start + ns, plan_k.stop + ns)
    flip = lambda a: a[::-1, :]                                    # iy(남→북) → MODFLOW 행(북→남)
    G = ["G1", "G2", "G3", "G4"]
    thk = np.stack([cv[f"{g}_thk"] for g in G])
    active = thk.sum(0) > 1.0
    top = cv["G1_top"].copy()
    th = np.where(active, np.maximum(thk, 1.0), 1.0)               # 최소 두께 1m (얇은 코코란 등)
    botm = top[None] - np.cumsum(th, 0)
    Kc, Kf = 10 ** p["log10_Kc"], 10 ** p["log10_Kf"]
    pc = np.stack([cv[f"{g}_pc"] for g in G])
    kh = pc * Kc + (1 - pc) * Kf
    kv = 1.0 / (pc / Kc + (1 - pc) / Kf)
    corc_present = cv["G2_thk"] >= 1.0
    kv[1] = np.where(corc_present, 10 ** p["log10_Kcorc"], kv[2])   # 코코란 점토가 없는 곳은 아래층 값
    kh[1] = np.where(corc_present, np.minimum(kh[1], 0.05), kh[2])
    idom = np.repeat(active[None], 4, 0).astype(int)
    sim = flopy.mf6.MFSimulation(sim_name="aq", sim_ws=ws, exe_name=MF6, verbosity_level=0)
    flopy.mf6.ModflowTdis(sim, nper=len(days), perioddata=[(d, (NSTP_SPIN if k < ns else 1), 1.0) for k, d in enumerate(days)],
                          time_units="days")
    if robust:
        flopy.mf6.ModflowIms(sim, complexity="complex", outer_dvclose=(2e-3 if V3 else 1e-4), inner_dvclose=1e-5, outer_maximum=200,
                             inner_maximum=300, linear_acceleration="bicgstab")
    else:
        flopy.mf6.ModflowIms(sim, complexity="moderate", outer_dvclose=(1e-3 if V3 else 1e-5), inner_dvclose=(1e-4 if V3 else 1e-6),
                             outer_maximum=(100 if V3 else 25), linear_acceleration="bicgstab")   # v3: 지연 압밀은 수위 1mm 수렴으로 충분
    gwf = flopy.mf6.ModflowGwf(sim, modelname="aq")
    flopy.mf6.ModflowGwfdis(gwf, nlay=4, nrow=ny, ncol=nx, delr=MCELL, delc=MCELL, top=flip(top),
                            botm=[flip(b) for b in botm], idomain=[flip(i) for i in idom])
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=[flip(a) for a in kh], k33=[flip(a) for a in kv])
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.0, transient={0: True})
    h_off = float(np.nanmax(top)) + 10.0 if V3 else 0.0             # v3: 수위 기준을 지표 위로(지연 압밀 점토가 포화 상태로 계산되게). 출력은 변화량이라 영향 없음
    flopy.mf6.ModflowGwfic(gwf, strt=h_off)
    edge = [(0, ny - 1 - iy, ix) for iy in range(ny) for ix in range(nx)
            if (iy in (0, ny - 1) or ix in (0, nx - 1)) and active[iy, ix]]
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data={0: [[e, h_off] for e in edge]})
    flopy.mf6.ModflowGwfrcha(gwf, recharge={k: (float(rch[k]) if np.ndim(rch[k]) == 0 else flip(rch[k]))
                                            for k in range(len(days))})
    ag = np.argwhere(site["ag"] & active)
    qc = (lambda k, iy, ix: float(q[k, iy, ix])) if q.ndim == 3 else (lambda k, iy, ix: float(q[k]))
    wel = {k: [[(L, ny - 1 - iy, ix), -qc(k, iy, ix) * f] for iy, ix in ag for L, f in ((2, 0.6), (3, 0.4))
               if qc(k, iy, ix) > 0] for k in range(len(days))}
    if plan:
        for j, k in enumerate(plan_k):
            for w, (iy, ix) in enumerate(plan["wells"]):
                rate = float(plan["rates"][j][w])
                if rate > 0 and active[iy, ix]:
                    wel[k] += [[(L, ny - 1 - iy, ix), -rate * f] for L, f in ((2, 0.6), (3, 0.4))]
    wel = {k: (v if v else [[(2, ny - 1 - int(ag[0][0]), int(ag[0][1])), 0.0]]) for k, v in wel.items()}
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel)
    mult = 10 ** np.array([p[f"log10_mSk_z{i}"] for i in range(SK_ZONES ** 2)])[zone_index(site)]
    pdata, n = [], 0
    for L, g in enumerate(G):
        fine = cv[f"{g}_fine"]
        ssv, sse = (cv["SSFV_CC"], cv["SSFE_CC"]) if g == "G2" else (cv["SSFV"], cv["SSFE"])
        for iy in range(ny):
            for ix in range(nx):
                if not active[iy, ix] or fine[iy, ix] < 0.5:
                    continue
                m = float(mult[iy, ix])
                if V3 and USE_DELAY != "0" and not (USE_DELAY == "2" and g == "G2"):   # 지연 압밀: 두꺼운 점토는 수위가 내려간 뒤에도 몇 년에 걸쳐 눌린다
                    b = float(fine[iy, ix]) if g == "G2" else min(float(fine[iy, ix]), DELAY_BEQ)
                    kvdb = 10 ** p["log10_Kvdb_s" if iy < ny // 2 else "log10_Kvdb"]
                    pdata.append([n, (L, ny - 1 - iy, ix), "delay", 0.0, b, max(float(fine[iy, ix]) / b, 1.0),
                                  float(ssv) * m, float(sse) * m, 0.3, kvdb, 0.0]); n += 1
                    continue
                pdata.append([n, (L, ny - 1 - iy, ix), "nodelay", 0.0, float(fine[iy, ix]), 1.0,
                              float(ssv) * m, float(sse) * m, 0.3, 999.0, 0.0]); n += 1
    flopy.mf6.ModflowGwfcsub(gwf, head_based=True, initial_preconsolidation_head=True, ninterbeds=n,
                             ndelaycells=(7 if V3 else None),                 # 지연 압밀 점토층 안을 7칸으로 나눠 계산(속도)
                             cg_ske_cr=float(cv["SSCE"]) + 1e-6, cg_theta=0.3, packagedata=pdata,
                             compaction_filerecord="aq.cmp")
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="aq.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    ok, buff = _run_mf6(sim, ws)
    if not ok:
        if own:
            shutil.rmtree(ws, ignore_errors=True)
        raise RuntimeError("MODFLOW 실패: " + " ".join(buff[-10:]))
    hf = flopy.utils.HeadFile(os.path.join(ws, "aq.hds"), precision="double")
    H = np.array(_last_steps(hf))[:, :, ::-1, :]
    H = np.where(np.abs(H) > 1e20, np.nan, H)
    cf = flopy.utils.HeadFile(os.path.join(ws, "aq.cmp"), text="CSUB-COMPACTION", precision="double")
    C = np.array(_last_steps(cf))
    C = np.where(np.abs(C) > 1e20, 0.0, C)
    S = C.sum(axis=1)[:, ::-1, :] * 1000.0
    if ns:                                                           # spin-up 끝(2015.6) 기준 변화로 바꾼다
        H = H[ns:] - H[ns - 1]; S = S[ns:] - S[ns - 1]
    if own and not keep:
        shutil.rmtree(ws, ignore_errors=True)
    return S, H


USE_DELAY = os.environ.get("AQ_DELAY", "2")   # 2(기본): 대수층 안 점토층만 지연 압밀, 코코란 점토는 비지연(지연으로 두면 수렴 실패). 0: 모두 비지연
NSTP_SPIN = int(os.environ.get("AQ_NSTP", "6"))                # spin-up 반년 기간은 6단계로 나눠 푼다(지연 압밀 수렴)


def _last_steps(f):
    """기간마다 마지막 시간 단계 값만(기간 수 = 출력 수)."""
    kk = f.get_kstpkper()
    last = {}
    for kstp, kper in kk:
        last[kper] = (kstp, kper)
    return [f.get_data(kstpkper=last[k]) for k in sorted(last)]


MF6_TIMEOUT = float(os.environ.get("AQ_MF6_TIMEOUT", 180))   # 초. 극단 파라미터에서 수렴이 끝없이 길어지는 실행은 실패로 처리


def _run_mf6(sim, ws):
    """MODFLOW 6을 시간 제한을 두고 실행. (성공 여부, 마지막 출력 줄)"""
    try:
        cp = subprocess.run([MF6], cwd=ws, capture_output=True, text=True, timeout=MF6_TIMEOUT)
        out = (cp.stdout or "").splitlines()
        return ("Normal termination" in (cp.stdout or "")), out[-10:]
    except subprocess.TimeoutExpired:
        return False, [f"시간 초과({MF6_TIMEOUT:.0f}초)"]


def add_spinup(site, p, days, rch, q):
    """2015.7 앞에 SPIN_YEARS년(관개기 4~9월 / 비관개기 10~3월 반년 단위)을 붙인다.
    과거 취수 = 2015.7~2022.12 같은 계절 평균 × 10**log10_spin, 함양 = 같은 계절 평균.
    2015년에 이미 수위가 내려가 있고 점토가 아직 눌리는 중인 상태(잔류 침하)에서 시작하게 한다."""
    months = site["months"][1:]
    n_hist = min(len(site["months"]) - 1, list(site["months"]).index(pd.Period(CAL_END, "M")))
    irr = np.isin([m.month for m in months[:n_hist]], [4, 5, 6, 7, 8, 9])
    mult = 10 ** p["log10_spin"]
    qi, qn = q[:n_hist][irr].mean(0) * mult, q[:n_hist][~irr].mean(0) * mult
    ri, rn = rch[:n_hist][irr].mean(0), rch[:n_hist][~irr].mean(0)
    ds, rs, qs = [], [], []
    for _ in range(SPIN_YEARS):
        ds += [182.0, 183.0]; rs += [rn, ri]; qs += [qn, qi]           # 10~3월, 4~9월 (2005.10부터)
    ns = len(ds)
    return (np.r_[ds, days], np.concatenate([np.array(rs), rch]), np.concatenate([np.array(qs), q]), ns)


def mix_band(P, sd, lo_q=5, hi_q=95, K=40, seed=0):
    """예측 분포의 90% 범위 = 앙상블 구성원 각각에 오차 sd(정규분포)를 더한 혼합 분포의 5~95% 분위수.
    (10/13 버전은 앙상블 5~95% 끝에 1.645σ를 다시 더해 실제 90%보다 넓었다 → 수정)"""
    P = np.atleast_2d(np.asarray(P, float))
    N, n = P.shape
    sdv = np.broadcast_to(np.asarray(sd, float), (n,))
    rng = np.random.default_rng(seed)
    lo, hi = np.empty(n), np.empty(n)
    for a in range(0, n, 2000):
        b = min(n, a + 2000)
        Z = (P[:, None, a:b] + rng.normal(size=(N, K, b - a)) * sdv[a:b]).reshape(-1, b - a)
        lo[a:b], hi[a:b] = np.percentile(Z, [lo_q, hi_q], axis=0)
    return lo, hi


# ---------------------------------------------------------------- 관측 연산자
def obs_index(site, period, use_insar=True, use_gps=None, cal_end=None, val_end=None):
    """관측 벡터 정의. period: 'cal'(보정) 또는 'val'(검증).
    검증 기간에는 InSAR 대신 GPS를 쓴다(use_gps 기본: 검증=True, 보정=False)."""
    if use_gps is None:
        use_gps = period == "val"
    if period == "val" and val_end is None:
        use_insar = False                                         # 2023년 이후 InSAR는 쓰지 않음(되돌아보기 검증은 예외)
    months = site["months"]
    cal_k = list(months).index(pd.Period(cal_end or CAL_END, "M"))
    ks = np.arange(1, len(months))
    ks = ks[ks <= cal_k] if period == "cal" else ks[ks > cal_k]
    if val_end is not None:
        ks = ks[ks <= list(months).index(pd.Period(val_end, "M"))]
    ins = []
    if use_insar:
        kk = ks[::OBS_EVERY] if period == "cal" else ks
        cells = site["cells"]
        u = np.where(site["use"])[0]
        D = site["disp"][u][:, kk]                                   # (셀, 시점)
        key = (cells.iy.to_numpy()[u] // F) * 1000 + cells.ix.to_numpy()[u] // F
        df = pd.DataFrame(D, columns=kk); df["key"] = key
        G = df.groupby("key").mean()                                 # 모델 격자(1km) 평균
        for kc, row in G.iterrows():
            for k, v in row.items():
                if np.isfinite(v):
                    ins.append((int(kc // 1000), int(kc % 1000), int(k), float(v)))
    h = site["hobs"]
    kmap = {m: i for i, m in enumerate(months)}
    hh = []
    for r in h.itertuples():
        k, kr = kmap.get(r.month), kmap.get(r.ref_month)
        if k is None or kr is None or k not in set(ks):
            continue
        hh.append((r.iy, r.ix, r.layer, k, kr, r.dh_m))
    gp = []
    if use_gps:
        for st, g in site["gps"].groupby("station"):
            g = g[g.month.map(kmap).isin(set(ks))].sort_values("month")
            if len(g) < 3:
                continue
            ref = g.iloc[0]
            iy = int((ref.lat - LAT0) * KM_LAT * 1000 // MCELL); ix = int((ref.lon - LON0) * KM_LON * 1000 // MCELL)
            for r in g.iloc[1:].itertuples():
                gp.append((iy, ix, kmap[r.month], kmap[ref.month], -(r.up_mm - ref.up_mm)))   # 침하(아래로 양수)
    sig_gp = SIG_INSAR
    if OPS and period == "cal" and cal_end is None and use_insar:   # 운영 모델: 보정 기간 뒤 GPS도 보정에 쓴다
        ks2 = np.arange(1, len(months)); ks2 = ks2[ks2 > cal_k]
        for st, g in site["gps"].groupby("station"):
            g = g[g.month.map(kmap).isin(set(ks2))].sort_values("month")
            if len(g) < 3:
                continue
            ref = g.iloc[0]
            iy = int((ref.lat - LAT0) * KM_LAT * 1000 // MCELL); ix = int((ref.lon - LON0) * KM_LON * 1000 // MCELL)
            for r_ in g.iloc[1:].itertuples():
                gp.append((iy, ix, kmap[r_.month], kmap[ref.month], -(r_.up_mm - ref.up_mm)))
        # GPS 가중치: InSAR 관측은 수천 개지만 서로 겹치는 정보가 많다. GPS 전체 가중치가 InSAR 전체의 GPS_WEIGHT배가 되게 σ를 정한다
        n_ins_ = len(ins)
        sig_gp = float(np.sqrt(len(gp) * SIG_INSAR ** 2 / (GPS_WEIGHT * max(n_ins_, 1)))) if gp else SIG_GPS_CAL
        # 수위도 보정 기간 뒤까지
        for r_ in h.itertuples():
            k, kr = kmap.get(r_.month), kmap.get(r_.ref_month)
            if k is None or kr is None or k not in set(ks2):
                continue
            hh.append((r_.iy, r_.ix, r_.layer, k, kr, r_.dh_m))
    ins = np.array(ins) if ins else np.zeros((0, 4))
    hh = np.array(hh) if hh else np.zeros((0, 6))
    gp = np.array(gp) if gp else np.zeros((0, 5))
    d = np.concatenate([ins[:, 3] if len(ins) else [], hh[:, 5] if len(hh) else [], gp[:, 4] if len(gp) else []])
    sd = np.concatenate([np.full(len(ins), SIG_INSAR), np.full(len(hh), SIG_HEAD), np.full(len(gp), sig_gp)])
    return dict(ins=ins, hh=hh, gp=gp, d=d, sd=sd, n_ins=len(ins), n_hh=len(hh))


def simulate_obs(out, idx):
    S, H = out
    ins, hh = idx["ins"], idx["hh"]
    vals = []
    if len(ins):
        # 모델 시간축: out[0]은 첫 응력기간 끝 = months[1]
        vals.append(S[ins[:, 2].astype(int) - 1, ins[:, 0].astype(int), ins[:, 1].astype(int)])
    if len(hh):
        iy, ix, k, kr = hh[:, 0].astype(int), hh[:, 1].astype(int), hh[:, 3].astype(int), hh[:, 4].astype(int)
        L = hh[:, 2].astype(int)
        h_now = H[k - 1, L, iy, ix]
        h_ref = np.where(kr > 0, H[np.maximum(kr - 1, 0), L, iy, ix], 0.0)
        vals.append(np.nan_to_num(h_now - h_ref))
    gp = idx.get("gp", np.zeros((0, 5)))
    if len(gp):
        iy, ix, k, kr = gp[:, 0].astype(int), gp[:, 1].astype(int), gp[:, 2].astype(int), gp[:, 3].astype(int)
        vals.append(S[k - 1, iy, ix] - np.where(kr > 0, S[np.maximum(kr - 1, 0), iy, ix], 0.0))
    return np.concatenate(vals) if vals else np.zeros(0)


# ---------------------------------------------------------------- ES-MDA
_SITE = None


def _init(site):
    global _SITE
    _SITE = site


def _one(theta):
    try:
        return run_model(_SITE, theta)
    except Exception as e:                      # 수렴 실패 구성원은 제외
        print("  (구성원 실행 실패:", str(e)[:120], ")", flush=True)
        return None


def _one_plan(args):
    theta, plan = args
    try:
        return run_model(_SITE, theta, plan=plan)
    except Exception as e:
        print("  (계획 실행 실패:", str(e)[:120], ")", flush=True)
        return None


def run_ensemble_plans(site, thetas, plans, workers=None):
    workers = workers or os.cpu_count() or 2
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(site,)) as ex:
        return list(ex.map(_one_plan, list(zip(thetas, plans))))


def run_ensemble(site, thetas, workers=None):
    workers = workers or os.cpu_count() or 2
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(site,)) as ex:
        return list(ex.map(_one, list(thetas)))


def esmda(site, use_insar, n_ens=32, n_iter=4, seed=0, workers=None, log=print):
    """ES-MDA. 반환: 사후 앙상블 파라미터 (N,P), 마지막 모델 출력 목록."""
    rng = np.random.default_rng(seed)
    idx = obs_index(site, "cal", use_insar)
    M = PRIOR_MEAN + rng.normal(size=(n_ens, len(NAMES))) * PRIOR_SD
    # 팽창 계수: 앞 반복은 크게(조금만 고침), 뒤로 갈수록 작게. 역수 합 = 1 (Emerick 2016 방식)
    g = np.geomspace(1.0, 6.0, n_iter)[::-1]
    alphas = list(g * np.sum(1.0 / g))
    lo, hi = PRIOR_MEAN - 3 * PRIOR_SD, PRIOR_MEAN + 3 * PRIOR_SD
    for it, a in enumerate(alphas):
        outs = run_ensemble(site, M, workers)
        ok = [i for i, o in enumerate(outs) if o is not None]
        D = np.array([simulate_obs(outs[i], idx) for i in ok])      # (n, Nd)
        Mk = M[ok]
        rmse = np.sqrt(np.mean(((D - idx["d"]) / idx["sd"]) ** 2, axis=1))
        log(f"  반복 {it + 1}/{n_iter}: 구성원 {len(ok)}/{len(M)}, 정규화 RMSE 중앙값 {np.median(rmse):.2f}")
        n = len(ok)
        dM = (Mk - Mk.mean(0)) / np.sqrt(n - 1)
        dD = (D - D.mean(0)) / np.sqrt(n - 1)
        R = a * idx["sd"] ** 2                                       # 대각
        Dp = idx["d"] + np.sqrt(a) * idx["sd"] * rng.normal(size=D.shape)
        # (dD dD^T + R)^-1 (Dp - D)^T 를 Woodbury로: 관측 수가 많아도 앙상블 크기 행렬만 역행렬
        Rinv_dD = dD / R                                             # (n, Nd)
        small = np.eye(n) + Rinv_dD @ dD.T
        resid = (Dp - D) / R                                         # (n, Nd)  = R^-1 (Dp-D)
        x = resid - (np.linalg.solve(small, Rinv_dD @ resid.T)).T @ Rinv_dD
        M = np.clip(Mk + (x @ dD.T) @ dM, lo, hi)                    # (n,P), 사전범위 ±3σ로 제한
    outs = run_ensemble(site, M, workers)
    keep = [i for i, o in enumerate(outs) if o is not None]
    return M[keep], [outs[i] for i in keep]


def _fd_jacobian(site, theta, idx, base, step=0.05, workers=None):
    """유한차분 야코비안(정규화 잔차 기준). 파라미터 13개를 병렬로 한 번씩 흔들어 실행."""
    thetas = [theta + step * PRIOR_SD * np.eye(len(theta))[j] for j in range(len(theta))]
    outs = run_ensemble(site, thetas, workers)
    J = np.zeros((len(base), len(theta)))
    for j, o in enumerate(outs):
        if o is None:
            continue
        J[:, j] = ((simulate_obs(o, idx) - base) / idx["sd"]) / (step * PRIOR_SD[j])
    return J


def calibrate(site, use_insar, n_iter=8, workers=None, log=print, cal_end=None, ckpt=None, th0=None):
    """가우스-뉴턴(레벤버그-마쿼트) 최소제곱 역산.

    목적함수 = Σ((모델-관측)/σ)² + Σ((θ-사전평균)/사전σ)².
    반환: 추정값 θ, 사후 공분산(선형화), 반복 기록.
    """
    idx = obs_index(site, "cal", use_insar, cal_end=cal_end)
    th = PRIOR_MEAN.copy() if th0 is None else np.clip(np.asarray(th0, float), PRIOR_MEAN - 3 * PRIOR_SD, PRIOR_MEAN + 3 * PRIOR_SD)
    lam = 1.0
    hist = []
    it0 = 0
    if ckpt is not None and Path(ckpt).exists():                    # 중간 저장에서 이어 하기(컴퓨터가 꺼져도 반복 단위로 보존)
        ck = json.load(open(ckpt))
        th, lam, hist, it0 = np.array(ck["th"]), ck["lam"], ck["hist"], ck["it"]
        log(f"  중간 저장에서 이어 함: 반복 {it0}/{n_iter} 완료 상태")
    def cost(o, t):
        r = (simulate_obs(o, idx) - idx["d"]) / idx["sd"]
        return float(r @ r + (((t - PRIOR_MEAN) / PRIOR_SD) ** 2).sum()), r
    out = run_model(site, th)
    c, r = cost(out, th)
    for it in range(it0, n_iter):
        base = simulate_obs(out, idx)
        J = _fd_jacobian(site, th, idx, base, workers=workers)
        Jp = np.diag(1 / PRIOR_SD)
        A = J.T @ J + Jp.T @ Jp
        g = J.T @ r + (th - PRIOR_MEAN) / PRIOR_SD ** 2
        improved = False
        for _ in range(4):                                         # 감쇠 계수 조정
            step = -np.linalg.solve(A + lam * np.diag(np.diag(A)), g)
            step = np.clip(step, -1.5 * PRIOR_SD, 1.5 * PRIOR_SD)
            t_new = np.clip(th + step, PRIOR_MEAN - 3 * PRIOR_SD, PRIOR_MEAN + 3 * PRIOR_SD)
            try:
                o_new = run_model(site, t_new)
                c_new, r_new = cost(o_new, t_new)
            except Exception:
                c_new = np.inf
            if c_new < c:
                th, out, c, r, lam, improved = t_new, o_new, c_new, r_new, max(lam / 3, 1e-3), True
                break
            lam *= 4
        n = idx["n_ins"]
        rm = np.sqrt(np.mean((r * idx["sd"])[:n] ** 2)) if n else float("nan")
        rh = np.sqrt(np.mean((r * idx["sd"])[n:] ** 2)) if len(r) > n else float("nan")
        hist.append({"iter": it + 1, "cost": c, "insar_rmse_mm": rm, "head_rmse_m": rh})
        log(f"  반복 {it + 1}/{n_iter}: 목적함수 {c:,.0f}, InSAR RMSE {rm:.1f} mm, 수위 RMSE {rh:.2f} m" + ("" if improved else " (개선 없음)"))
        if ckpt is not None:
            json.dump({"th": th.tolist(), "lam": lam, "hist": hist, "it": it + 1}, open(ckpt, "w"), default=float)
        if not improved and lam > 1e3:
            break
    base = simulate_obs(out, idx)
    J = _fd_jacobian(site, th, idx, base, workers=workers)
    cov = np.linalg.inv(J.T @ J + np.diag(1 / PRIOR_SD ** 2))
    # 잔차가 가정한 오차(σ)보다 크면 모델 구조 오차가 있다는 뜻 → 공분산을 그만큼 키운다
    scale = max(1.0, float((r @ r) / max(len(r) - len(th), 1)))
    return th, cov * scale, hist


def posterior_cov(site, th, use_insar, cal_end=None, workers=None):
    """추정값 θ에서 선형화 사후 공분산(calibrate 마지막 단계와 같은 계산)."""
    idx = obs_index(site, "cal", use_insar, cal_end=cal_end)
    out = run_model(site, th)
    base = simulate_obs(out, idx)
    r = (base - idx["d"]) / idx["sd"]
    J = _fd_jacobian(site, th, idx, base, workers=workers)
    cov = np.linalg.inv(J.T @ J + np.diag(1 / PRIOR_SD ** 2))
    scale = max(1.0, float((r @ r) / max(len(r) - len(th), 1)))
    return cov * scale


def posterior_ensemble(site, th, cov, n=30, seed=0, workers=None):
    """선형화 사후분포에서 n개를 뽑아 실행(E4 불확실성)."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(th)))
    M = th + rng.normal(size=(n, len(th))) @ L.T
    M = np.clip(M, PRIOR_MEAN - 3 * PRIOR_SD, PRIOR_MEAN + 3 * PRIOR_SD)
    outs = run_ensemble(site, M, workers)
    keep = [i for i, o in enumerate(outs) if o is not None]
    print(f"  앙상블 {len(keep)}/{n} 성공")
    return M[keep], [outs[i] for i in keep]


def aqueduct_cells(site, km=2.0):
    """수로에서 km 이내 모델 칸 (iy, ix) 집합 — 의사결정·GPS 검증이 이루어지는 곳."""
    pts = site["aq_pts"]
    out = set()
    for iy in range(site["ny"]):
        for ix in range(site["nx"]):
            la, lo = site["lat_c"][iy, ix], site["lon_c"][iy, ix]
            d = np.min(np.hypot((pts[:, 0] - lo) * KM_LON, (pts[:, 1] - la) * KM_LAT))
            if d <= km:
                out.add((iy, ix))
    return out


def forecast_error(site, th, cal_end="2019-12", val_end="2022-12", near_km=2.0):
    """되돌아보기(hindcast) 예측 오차: cal_end까지로 보정한 모델(th)로 그 뒤 val_end까지 InSAR를 예측했을 때
    오차가 예측 기간(lead, 년)에 따라 어떻게 커지는지 재서 sd(lead) = sqrt(s0² + (b·lead)²)로 맞춘다.
    모델 구조 오차를 앙상블에 더하는 데 쓴다(관측 잡음·파라미터 불확실성만으로는 범위가 너무 좁았음)."""
    out = run_model(site, th)
    idx = obs_index(site, "val", True, use_gps=False, cal_end=cal_end, val_end=val_end)
    sim = simulate_obs(out, idx)
    n = idx["n_ins"]
    months = site["months"]; k0 = list(months).index(pd.Period(cal_end, "M"))
    lead = (idx["ins"][:, 2] - k0) / 12.0
    # 보정 끝 시점 대비 변화로 비교(같은 시점 기준이 아니면 누적 편차가 섞임)
    base_obs = {}; base_sim = {}
    ins = idx["ins"]
    S = out[0]
    e = (sim[:n] - S[k0 - 1, ins[:, 0].astype(int), ins[:, 1].astype(int)]) - \
        (idx["d"][:n] - _obs_at(site, ins, k0))
    ok = np.isfinite(e)
    if near_km:
        near = aqueduct_cells(site, near_km)
        ok &= np.array([(int(a), int(b)) in near for a, b in ins[:, :2]])
    bins = np.arange(0, lead.max() + 0.5, 0.5)
    rows = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = ok & (lead > a) & (lead <= b)
        if m.sum() > 20:
            rows.append(((a + b) / 2, float(np.sqrt(np.mean(e[m] ** 2)))))
    L, R = np.array(rows).T
    A = np.c_[np.ones_like(L), L ** 2]
    coef, *_ = np.linalg.lstsq(A, R ** 2, rcond=None)
    s0 = float(np.sqrt(max(coef[0], 1.0))); b = float(np.sqrt(max(coef[1], 0.0)))
    return {"s0_mm": s0, "b_mm_per_yr": b, "near_aqueduct_km": near_km, "table": [[float(x), float(y)] for x, y in rows]}


def _obs_at(site, ins, k):
    """InSAR 관측(1km 평균)을 시점 k에서 같은 칸끼리 뽑기."""
    cells = site["cells"]; u = np.where(site["use"])[0]
    key = (cells.iy.to_numpy()[u] // F) * 1000 + cells.ix.to_numpy()[u] // F
    v = pd.Series(site["disp"][u, k]).groupby(key).mean()
    kk = ins[:, 0].astype(int) * 1000 + ins[:, 1].astype(int)
    return v.reindex(kk).to_numpy()


def evaluate(site, outs, err=None):
    """보정·검증 기간 오차. insar/gps는 mm, head는 m. cov90 = 앙상블 90% 구간(+관측 잡음) 안에 관측이 들어간 비율."""
    res = {}
    for period in ["cal", "val"]:
        idx = obs_index(site, period, True)
        P = np.array([simulate_obs(o, idx) for o in outs])
        mean = P.mean(0)
        ni, nh = idx["n_ins"], idx["n_hh"]
        e = mean - idx["d"]
        sl = {"insar": slice(0, ni), "head": slice(ni, ni + nh), "gps": slice(ni + nh, len(e))}
        r = {}
        for k, s_ in sl.items():
            n = s_.stop - s_.start
            r[f"n_{k}"] = int(n)
            if n:
                r[f"{k}_rmse"] = float(np.sqrt(np.mean(e[s_] ** 2)))
                if len(outs) > 1:
                    sig = SIG_HEAD if k == "head" else SIG_INSAR
                    lo, hi = mix_band(P[:, s_], sig)
                    r[f"{k}_cov90"] = float(np.mean((idx["d"][s_] >= lo) & (idx["d"][s_] <= hi)))
                    if err is not None and k == "gps" and period == "val":
                        # 구조 오차 추가: 예측 기간(보정 끝으로부터 년)에 따라 커지는 오차
                        k0 = list(site["months"]).index(pd.Period(CAL_END, "M"))
                        lead = (idx["gp"][:, 2] - k0) / 12.0
                        sd = np.sqrt(err["s0_mm"] ** 2 + (err["b_mm_per_yr"] * lead) ** 2 + SIG_INSAR ** 2)
                        lo2, hi2 = mix_band(P[:, s_], sd)
                        r["gps_cov90_with_model_error"] = float(np.mean((idx["d"][s_] >= lo2) & (idx["d"][s_] <= hi2)))
                        r["gps_band90_mean_width_mm"] = float(np.mean(hi2 - lo2))
        res[period] = r
    return res
