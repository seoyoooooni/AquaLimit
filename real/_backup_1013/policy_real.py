"""실제 우물·실제 정책으로 하는 취수 계획 (위성 증발산 기반 취수 모델 위에서).

고객 상황(실제)
- Westside District Water Authority(Lost Hills·Belridge·Berrenda Mesa 물 공급구역)는 2025년 1월
  수로(California Aqueduct) 양쪽 2.5마일(4km) 안 농업용 우물 취수를 전면 금지했다(SJV Water, 2025.1).
- 같은 관리기관은 "침하가 지하수 때문인지 증명할 자료가 없다"고 했다. 주정부 검토(2025.9)도
  "수로 주변, 특히 Lost Hills 근처 침하를 원인별로 나누지 못한다"를 부족한 점으로 꼽았다.

이 파일이 하는 일
1) 원인 분석: 2015.7~2022.12 수로 침하 중 '수로 10km 안 농업 취수' 몫은 얼마인가 (앙상블)
2) 과거 재현(2021.4~2022.9 대가뭄): 실제 취수 / 4km 전면 금지 / AquaLimit 최적 배분 비교
   - AquaLimit: 모든 앙상블 구성원·모든 수로 지점에서 '금지만큼' 수로를 지키면서 대체 용수 비용 최소
   - 계획은 선형 응답(현재 취수 주변 미분)으로 풀고, 결과는 MODFLOW로 다시 돌려 확인한다
3) 다음 관개기(2026.5~7) 우물별 계획: 날씨(기온)·SWP 배분율 시나리오별

우물: DWR 우물 신고 자료(Well Completion Reports, 공개)의 농업용 신규 우물. 위치는 섹션(1.6km) 중심 좌표.
취수량: 위성 증발산(USGS SSEBop)으로 칸별 추정(real_model.forcing). 실제 계량 기록은 비공개.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import real_model as r

HERE = Path(__file__).resolve().parent
OUT = HERE / ("results_policy" if r.PUMPING == "et" else "results_policy_uniform")
OUT.mkdir(exist_ok=True)

SERVICE_KM = 10.0          # 관리 대상: 수로 동쪽 10km 안 농경지
BUFFER_KM = 4.02           # 실제 정책: 수로 양쪽 2.5마일 취수 금지
DEFAULT_GPM = 2000.0       # 신고 양수량이 없는 우물(신고 자료 중앙값)
GPM_TO_M3D = 5.451
DUTY = 0.8                 # 우물 가동률(정비·교대)
PRICE = 0.81               # 대체 용수(외부 구매·휴경 보상) $/m3 ≈ $1,000/AF (가뭄기 Kern 시세 수준, 가정)
KWH_PRICE = 0.20           # $/kWh (캘리포니아 농업용 전기요금 수준, 가정)
PUMP_EFF = 0.6
N_GROUPS_IN, N_GROUPS_OUT = 4, 8
RETRO = ("2021-04", "2022-09")
# 다음 관개기 날씨 시나리오: 기온 변화(°C, 증발산 수요), SWP 배분율, 지역 공통 취수의 가뭄 계수
SCENARIOS = {"습한 해": {"dT": -1.5, "alloc": 0.45, "dry": -1.0},
             "평년": {"dT": 0.0, "alloc": 0.30, "dry": 0.0},
             "가뭄·폭염": {"dT": 2.5, "alloc": 0.15, "dry": 1.0}}
FWD_MONTHS = 3             # 2026.5~7
N_FWD = 15                 # 다음 관개기 계획에 쓰는 앙상블 구성원 수(계산 시간)


# ---------------------------------------------------------------- 우물·관리 지역
def aq_dist_km(site, lon, lat):
    pts = site["aq_pts"]
    lon, lat = np.atleast_1d(lon), np.atleast_1d(lat)
    return np.array([np.min(np.hypot((pts[:, 0] - a) * r.KM_LON, (pts[:, 1] - b) * r.KM_LAT)) for a, b in zip(lon, lat)])


def load_wells(site):
    d = json.load(open(r.DATA / "wcr_raw.json"))
    df = pd.DataFrame([f["attributes"] for f in d["features"]])
    w = df[df.PlannedUseFormerUse.str.contains("Irrigation - Agri", na=False) & (df.RecordType == "New")
           & (df.TotalCompletedDepth.fillna(0) >= 150)].copy()
    w["iy"] = ((w.DecimalLatitude - r.LAT0) * r.KM_LAT * 1000 // r.MCELL).astype(int)
    w["ix"] = ((w.DecimalLongitude - r.LON0) * r.KM_LON * 1000 // r.MCELL).astype(int)
    w = w[(w.iy >= 0) & (w.iy < site["ny"]) & (w.ix >= 0) & (w.ix < site["nx"])]
    w["dist_km"] = aq_dist_km(site, w.DecimalLongitude, w.DecimalLatitude)
    pts = site["aq_pts"]
    w["east"] = w.DecimalLongitude > np.interp(w.DecimalLatitude, pts[:, 1], pts[:, 0])
    w = w[w.east & (w.dist_km <= SERVICE_KM) & np.array([site["ag"][a, b] for a, b in zip(w.iy, w.ix)])]
    y = pd.to_numeric(w.WellYield, errors="coerce")
    w["yield_gpm"] = y.where(y > 0, DEFAULT_GPM)
    w["yield_reported"] = y.notna() & (y > 0)
    swl = pd.to_numeric(w.StaticWaterLevel, errors="coerce")
    w["lift_m"] = swl.where(swl > 0, np.nan).fillna(swl[swl > 0].median() if (swl > 0).any() else 300.0) * 0.3048
    w["in_buffer"] = w.dist_km <= BUFFER_KM
    return w.reset_index(drop=True)


def group_wells(w, seed=0):
    """우물을 위치로 묶는다(금지구역 안 4개, 밖 8개). 같은 묶음은 같은 응답을 쓴다."""
    from scipy.cluster.vq import kmeans2
    w = w.copy(); w["group"] = -1
    xy = np.c_[w.ix + 0.0, w.iy + 0.0]
    g0 = 0
    for inb, k in ((True, N_GROUPS_IN), (False, N_GROUPS_OUT)):
        m = (w.in_buffer == inb).to_numpy()
        k = min(k, int(m.sum()))
        if k == 0:
            continue
        _, lab = kmeans2(xy[m], k, seed=seed, minit="++")
        w.loc[m, "group"] = lab + g0
        g0 += k
    w["group"] = pd.factorize(w.group, sort=True)[0]
    groups = []
    for g, d in w.groupby("group"):
        cells = sorted(set(zip(d.iy, d.ix)))
        cap = float((d.yield_gpm * GPM_TO_M3D * DUTY).sum())
        lift = float(d.lift_m.mean())
        cost = 0.02 + KWH_PRICE * 9.81 * lift / 3600.0 / PUMP_EFF      # $/m3 = 유지비 + 양정 전력비
        groups.append({"group": int(g), "cells": cells, "n_wells": len(d), "cap": cap, "cost": cost,
                       "in_buffer": bool(d.in_buffer.all()), "dist_km": float(d.dist_km.mean()),
                       "wcr": d.WCRNumber.tolist(), "lat": float(d.DecimalLatitude.mean()), "lon": float(d.DecimalLongitude.mean())})
    return w, groups


def service_cells(site):
    lat, lon = site["lat_c"], site["lon_c"]
    d = aq_dist_km(site, lon.ravel(), lat.ravel()).reshape(lat.shape)
    m = site["ag"] & (d <= SERVICE_KM)
    return [tuple(c) for c in np.argwhere(m)], d


def assign_cells(cells, groups, peak=None):
    """관리 지역 농경지 칸을 우물 묶음에 배정(현재 물 공급 가정). 가까운 묶음부터, 묶음 용량(최대 달 기준)을 넘지 않게."""
    gc = [np.array(g["cells"], float) for g in groups]
    dist = np.array([[np.min(np.hypot(c[:, 0] - iy, c[:, 1] - ix)) for c in gc] for iy, ix in cells])
    if peak is None:
        return dist.argmin(1)
    left = np.array([g["cap"] for g in groups], float)
    out = np.full(len(cells), -1)
    for i in np.argsort(-np.asarray(peak)):
        for g in np.argsort(dist[i]):
            if left[g] >= peak[i]:
                out[i] = g; left[g] -= peak[i]; break
        if out[i] < 0:
            out[i] = int(dist[i].argmin())
    return out


def peak_cell_demand(site, theta, cells):
    _, _, q = r.forcing(site, theta)
    return np.array([q[:, iy, ix].max() for iy, ix in cells])


# ---------------------------------------------------------------- 계획 실행 단위
def k_of(site, month):
    """응력기간 인덱스: forcing의 기간 k는 site['months'][k+1]."""
    return list(site["months"]).index(pd.Period(month, "M")) - 1


def demand_matrix(site, theta, cells, assign, G, window=None, future=None):
    """우물 묶음별 현재 취수(m3/일), (달, G). window=(시작, 끝) 과거 기간 또는 None(미래 FWD_MONTHS)."""
    if window:
        _, _, q = r.forcing(site, theta)
        k0, k1 = k_of(site, window[0]), k_of(site, window[1])
        qq = q[k0:k1 + 1]
    else:
        _, _, q = r.forcing(site, theta, FWD_MONTHS, future=future)
        qq = q[-FWD_MONTHS:]
    D = np.zeros((len(qq), G))
    for (iy, ix), g in zip(cells, assign):
        D[:, g] += qq[:, iy, ix]
    return D


def make_plan(groups, cells, rates_mg, window, site, future=None):
    """rates_mg: (달, G) m3/일 → MODFLOW 계획(관리 지역 칸의 기존 취수는 지우고 우물 칸에서 뽑는다)."""
    wells, cols = [], []
    for g, gr in enumerate(groups):
        for c in gr["cells"]:
            wells.append(c); cols.append((g, len(gr["cells"])))
    R = np.zeros((rates_mg.shape[0], len(wells)))
    for j, (g, n) in enumerate(cols):
        R[:, j] = rates_mg[:, g] / n
    p = {"months": rates_mg.shape[0], "wells": wells, "rates": R, "replace_cells": cells, "future": future}
    if window:
        p["start"] = k_of(site, window[0])
    return p


def outputs(site, out, window, pts, svc):
    """창 기간 수로 지점 침하(mm), 관리 지역 평균 침하, 2026.4까지 남는 침하."""
    S, H = out
    iy, ix = np.array(pts).T
    if window:
        a, b = k_of(site, window[0]) - 1, k_of(site, window[1])
    else:
        a, b = len(site["months"]) - 2, S.shape[0] - 1
    sub = S[b, iy, ix] - S[a, iy, ix]
    lasting = S[-1, iy, ix] - S[a, iy, ix]
    siy, six = np.array(svc).T
    area = float(np.mean(S[b, siy, six] - S[a, siy, six]))
    return {"sub": sub, "lasting": lasting, "area": area}


def run_plans(site, thetas, plans):
    return r.run_ensemble_plans(site, thetas, plans)


# ---------------------------------------------------------------- 1) 원인 분석
def attribution(site, M, cells, pts, log=print):
    """2015.7~2022.12 수로 침하: 실제 vs 관리 지역(수로 10km 안) 농업 취수를 0으로 했을 때."""
    n = k_of(site, "2022-12") + 1
    base = {"months": n, "wells": [], "rates": np.zeros((n, 0)), "start": 0}
    off = dict(base, replace_cells=cells)
    outs = run_plans(site, list(M) * 2, [base] * len(M) + [off] * len(M))
    iy, ix = np.array(pts).T
    k = n - 1
    res = []
    for i in range(len(M)):
        a, b = outs[i], outs[len(M) + i]
        if a is None or b is None:
            continue
        sa, sb = a[0][k, iy, ix], b[0][k, iy, ix]
        res.append((float(np.median(sa)), float(np.median(sb))))
    res = np.array(res)
    frac = 1 - res[:, 1] / res[:, 0]
    # InSAR 실제 관측(수로 칸 중앙값)과 비교
    obs = obs_aqueduct(site, pts, "2022-12")
    log(f"  원인 분석: 수로 침하(중앙 지점) 모델 {np.median(res[:, 0]):.0f}mm, 관측 {obs:.0f}mm, "
        f"농업 취수 몫 {np.median(frac) * 100:.0f}% (90% 범위 {np.percentile(frac, 5) * 100:.0f}~{np.percentile(frac, 95) * 100:.0f}%)")
    return {"model_mm": res[:, 0].tolist(), "no_ag_mm": res[:, 1].tolist(), "frac": frac.tolist(), "obs_insar_mm": obs}


def obs_aqueduct(site, pts, month):
    cells = site["cells"]; k = list(site["months"]).index(pd.Period(month, "M"))
    key = set(iy * 1000 + ix for iy, ix in pts)
    kk = (cells.iy // r.F) * 1000 + cells.ix // r.F
    m = kk.isin(key).to_numpy() & site["use"]
    return float(np.nanmedian(site["disp"][m, k]))


# ---------------------------------------------------------------- 2·3) 정책 비교
def responses(site, M, groups, cells, D_sq, window, pts, svc, future=None, rel=0.3):
    """현재 취수(status quo) 주변 선형 응답. 구성원마다 1 + G회 실행."""
    G = len(groups)
    shape = D_sq.sum(1) / D_sq.sum(1).mean()                         # 달별 모양(관리 지역 전체)
    x_sq = D_sq.mean(0)                                               # 묶음별 평균 m3/일
    rates_sq = np.outer(shape, x_sq)
    cap = np.array([g["cap"] for g in groups])
    dx = np.maximum(rel * x_sq, 0.1 * cap)
    plans, th = [], []
    for m in M:
        plans.append(make_plan(groups, cells, rates_sq, window, site, future)); th.append(m)
        for g in range(G):
            x = x_sq.copy(); x[g] += dx[g]
            plans.append(make_plan(groups, cells, np.outer(shape, x), window, site, future)); th.append(m)
    outs = run_plans(site, th, plans)
    R = []
    for i in range(len(M)):
        o = outs[i * (G + 1): (i + 1) * (G + 1)]
        if any(v is None for v in o):
            continue
        base = outputs(site, o[0], window, pts, svc)
        Rs = np.zeros((len(pts), G)); Ra = np.zeros(G); Rl = np.zeros((len(pts), G))
        for g in range(G):
            y = outputs(site, o[g + 1], window, pts, svc)
            Rs[:, g] = (y["sub"] - base["sub"]) / dx[g]
            Rl[:, g] = (y["lasting"] - base["lasting"]) / dx[g]
            Ra[g] = (y["area"] - base["area"]) / dx[g]
        R.append({"sub0": base["sub"], "last0": base["lasting"], "area0": base["area"], "Rs": Rs, "Rl": Rl, "Ra": Ra})
    return R, x_sq, shape


def total_days(shape, site, window):
    if window:
        ms = pd.period_range(window[0], window[1], freq="M")
    else:
        ms = pd.period_range(site["months"][-1] + 1, periods=FWD_MONTHS, freq="M")
    days = np.array([m.days_in_month for m in ms], float)
    return float((shape * days).sum())


def solve(R, x_sq, groups, T, limit, no_worse_area=True):
    """비용 최소: 우물 묶음별 평균 취수 x(m3/일) + 대체 용수 v(m3).
    수요: T·Σx + v = T·Σx_sq.  수로: 모든 구성원·지점에서 sub0 + Rs(x − x_sq) ≤ limit[구성원, 지점]."""
    G = len(groups)
    cost = np.array([g["cost"] for g in groups]); cap = np.array([g["cap"] for g in groups])
    c = np.r_[cost * T, PRICE]
    A_eq = np.r_[np.full(G, T), 1.0][None]; b_eq = [T * x_sq.sum()]
    A, b = [], []
    for i, Ri in enumerate(R):
        for p in range(Ri["Rs"].shape[0]):
            A.append(np.r_[Ri["Rs"][p], 0.0]); b.append(limit[i][p] - Ri["sub0"][p] + Ri["Rs"][p] @ x_sq)
        if no_worse_area:
            A.append(np.r_[Ri["Ra"], 0.0]); b.append(Ri["Ra"] @ x_sq)
    res = linprog(c, A_ub=np.array(A), b_ub=b, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, cg) for cg in cap] + [(0, None)], method="highs")
    if res.status != 0:
        return {"feasible": False, "reason": "수로 기준을 지키는 배분이 없습니다(우물 용량 부족)."}
    x = res.x[:G]
    return {"feasible": True, "x": x.tolist(), "replace_m3": float(res.x[G]), "cost": float(res.fun)}


def policy_ban(x_sq, groups, T):
    x = np.array([0.0 if g["in_buffer"] else xs for g, xs in zip(groups, x_sq)])
    cost = np.array([g["cost"] for g in groups])
    v = T * (x_sq.sum() - x.sum())
    return {"feasible": True, "x": x.tolist(), "replace_m3": float(v), "cost": float((cost * x).sum() * T + PRICE * v)}


def policy_sq(x_sq, groups, T):
    cost = np.array([g["cost"] for g in groups])
    return {"feasible": True, "x": list(map(float, x_sq)), "replace_m3": 0.0, "cost": float((cost * x_sq).sum() * T)}


def predict(R, x, x_sq):
    x = np.asarray(x)
    sub = np.array([Ri["sub0"] + Ri["Rs"] @ (x - x_sq) for Ri in R])
    last = np.array([Ri["last0"] + Ri["Rl"] @ (x - x_sq) for Ri in R])
    return sub, last


def verify(site, M, groups, cells, plan, shape, window, pts, svc, future=None):
    """계획을 MODFLOW로 다시 돌린다(선형 근사 확인)."""
    rates = np.outer(shape, np.array(plan["x"]))
    outs = run_plans(site, list(M), [make_plan(groups, cells, rates, window, site, future)] * len(M))
    ys = [outputs(site, o, window, pts, svc) for o in outs if o is not None]
    return np.array([y["sub"] for y in ys]), np.array([y["lasting"] for y in ys]), np.array([y["area"] for y in ys])


def compare(site, M, th3, groups, cells, assign, pts, svc, window, future=None, tag="retro", log=print):
    G = len(groups)
    D_sq = demand_matrix(site, th3, cells, assign, G, window, future)
    R, x_sq, shape = responses(site, M, groups, cells, D_sq, window, pts, svc, future)
    T = total_days(shape, site, window)
    cap = np.array([g["cap"] for g in groups])
    np.savez_compressed(OUT / f"R_{tag}.npz", Rs=np.array([q["Rs"] for q in R]), Rl=np.array([q["Rl"] for q in R]),
                        Ra=np.array([q["Ra"] for q in R]), sub0=np.array([q["sub0"] for q in R]),
                        last0=np.array([q["last0"] for q in R]), area0=np.array([q["area0"] for q in R]),
                        x_sq=x_sq, shape=shape, T=T, D_sq=D_sq)
    log(f"  [{tag}] 응답 계산 {len(R)}/{len(M)}개 구성원. 현재 취수 {x_sq.sum() * T / 1e6:.1f}백만 m3, "
        f"용량 대비 {x_sq.sum() / cap.sum() * 100:.0f}%")
    pol = {"현재 방식(실제 취수)": policy_sq(x_sq, groups, T), "4km 전면 금지(2025 실제 정책)": policy_ban(x_sq, groups, T)}
    sub_ban, last_ban = predict(R, pol["4km 전면 금지(2025 실제 정책)"]["x"], x_sq)
    pol["AquaLimit(금지와 같은 수로 보호)"] = solve(R, x_sq, groups, T, sub_ban)
    # 보호 수준을 바꿔 가며 비용 곡선: 현재 대비 수로 침하(지점별) 감소율
    sub_sq, _ = predict(R, x_sq, x_sq)
    curve = []
    for f in np.linspace(0, 1.0, 11):
        lim = sub_sq - f * (sub_sq - sub_ban) * 1.6                    # 금지의 0~1.6배 보호
        s = solve(R, x_sq, groups, T, lim)
        curve.append({"protect_vs_ban": float(f * 1.6), "feasible": s["feasible"],
                      "replace_m3": s.get("replace_m3"), "cost": s.get("cost")})
    out = {"T_days": T, "x_sq": x_sq.tolist(), "shape": shape.tolist(), "policies": {}, "curve": curve}
    for k, p in pol.items():
        if not p["feasible"]:
            out["policies"][k] = p; continue
        sub, last = predict(R, p["x"], x_sq)
        vs, vl, va = verify(site, M, groups, cells, p, shape, window, pts, svc, future)
        out["policies"][k] = dict(p, pred_sub_med=float(np.median(sub.max(1))), ver_sub_med=float(np.median(vs.max(1))),
                                  ver_sub_p95=float(np.percentile(vs.max(1), 95)), ver_last_med=float(np.median(vl.max(1))),
                                  ver_aqmed_med=float(np.median(np.median(vs, 1))), ver_area_med=float(np.median(va)),
                                  ver_sub_members=vs.max(1).tolist(), ver_sub_points_med=np.median(vs, 0).tolist())
        log(f"    {k}: 대체 용수 {p['replace_m3'] / 1e6:.2f}백만 m3, 비용 ${p['cost'] / 1e6:.2f}M, "
            f"수로 최대 지점 침하 예측 {np.median(sub.max(1)):.1f} / MODFLOW {np.median(vs.max(1)):.1f}mm")
    return out, R


def main(only=None):
    t0 = time.time()
    site = r.load_site()
    MODEL = r.result_dir("model", create=False); r.check_version(MODEL)
    res = json.load(open(MODEL / "results.json"))
    th3 = np.array(res["params"]["E3"])
    M = np.load(MODEL / "E4_members.npy")
    import plan_real as pr
    pts = pr.mgmt_points(site)
    w, groups = group_wells(load_wells(site))
    cells, dist = service_cells(site)
    assign = assign_cells(cells, groups, peak_cell_demand(site, th3, cells))
    print(f"우물 {len(w)}개(신고 양수량 있음 {int(w.yield_reported.sum())}개) → 묶음 {len(groups)}개, "
          f"관리 지역 농경지 {len(cells)}칸, 수로 지점 {len(pts)}개", flush=True)
    w.to_csv(OUT / "wells_used.csv", index=False)
    json.dump({"groups": groups, "cells": cells, "assign": assign.tolist(), "pts": pts}, open(OUT / "setup.json", "w"),
              ensure_ascii=False, default=float)
    out = {"setup": {"n_wells": len(w), "n_yield_reported": int(w.yield_reported.sum()), "n_groups": len(groups),
                     "n_cells": len(cells), "buffer_km": BUFFER_KM, "service_km": SERVICE_KM, "price": PRICE}}
    if only in (None, "attr"):
        print("[1] 원인 분석", flush=True)
        out["attribution"] = attribution(site, M, cells, pts)
        json.dump(out, open(OUT / "results_attr.json", "w"), ensure_ascii=False, default=float)
    if only in (None, "retro"):
        print("[2] 과거 재현 2021.4~2022.9", flush=True)
        out["retro"], R = compare(site, M, th3, groups, cells, assign, pts, cells, RETRO, tag="retro")
        json.dump(out["retro"], open(OUT / "results_retro.json", "w"), ensure_ascii=False, default=float)
    if only in (None, "fwd"):
        print("[3] 다음 관개기 2026.5~7", flush=True)
        out["fwd"] = {}
        for name, fut in SCENARIOS.items():
            o, _ = compare(site, M[:N_FWD], th3, groups, cells, assign, pts, cells, None, future=fut, tag=name)
            out["fwd"][name] = o
            json.dump(out["fwd"], open(OUT / "results_fwd.json", "w"), ensure_ascii=False, default=float)
    print(f"끝 {(time.time() - t0) / 60:.1f}분", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
