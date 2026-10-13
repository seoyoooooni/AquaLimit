"""실제 검증 지역 취수 계획 E1~E4 (2026년 5~7월 관개기, 3개월).

상황(설명용 가정)
- 수로 동쪽 3km에 사업자 우물 6개(용량 각 4,000 m3/일). 실제 관정별 기록이 공개돼 있지 않아 위치·용량은 가정.
- 3개월 수요 900,000 m3(용량의 약 4분의 3). 부족분은 외부 용수(수로 물 이전 구매)로 채운다.
- 비용: 우물별 양수 비용 0.10~0.17 달러/m3(깊이·위치 차이 가정), 외부 용수 0.81 달러/m3(약 1,000달러/AF), 구매 한도 300,000 m3.
- 관리 기준: 계획으로 인한 수로 칸 추가 침하 ≤ LIMIT_MM (3개월). 다른 농가 취수(배경)는 역산한 패턴대로 계속된다.

방식
- E1: 모델 없이 우물 용량 비례 배분
- E2·E3: 각 모델(추정값 1개)의 응답 행렬로 비용 최소 계획(선형계획)
- E4: E3 불확실성 앙상블 모든 구성원에서 기준을 지키는 계획(강건)
- E4+임계수위: E4 + 수로 칸 하부 대수층 수위가 과거 최저(선행압밀수위)보다 내려가지 않게
평가: 실제 정답이 없으므로 E4 앙상블 전체(구조 오차 포함)에서 기준 초과 확률로 비교한다(시뮬레이션 결과).
"""
import json
import time

import numpy as np
from scipy.optimize import linprog

import real_model as r

MONTHS = 3
DAYS = 92.0
CAP = 4000.0                     # m3/일
DEMAND = 900_000.0               # m3 / 3개월
PRICE = 0.81                     # 달러/m3
PURCHASE_CAP = 300_000.0
WELL_COST = np.array([0.12, 0.10, 0.15, 0.13, 0.17, 0.11])
LIMIT_MM = 2.0                   # 3개월 계획 추가 침하 기준(mm). 배경 침하(수로 칸 중앙값 약 2.7mm/3개월)보다 낮게


def mgmt_points(site):
    cells = site["cells"]
    oil = cells.oil_field.fillna("").astype(str).str.len() > 0
    frac = cells.assign(oil=oil, k=(cells.iy // r.F) * 1000 + cells.ix // r.F).groupby("k").oil.mean()
    pts = set()
    for lo, la in site["aq_pts"]:
        iy = int((la - r.LAT0) * r.KM_LAT * 1000 // r.MCELL); ix = int((lo - r.LON0) * r.KM_LON * 1000 // r.MCELL)
        if 0 <= iy < site["ny"] and 0 <= ix < site["nx"] and frac.get(iy * 1000 + ix, 1.0) < 0.5:
            pts.add((iy, ix))
    return sorted(pts)


def operator_wells(site, n=6, offset_km=3):
    pts = sorted(set((int((la - r.LAT0) * r.KM_LAT * 1000 // r.MCELL), int((lo - r.LON0) * r.KM_LON * 1000 // r.MCELL))
                     for lo, la in site["aq_pts"]))
    pts = [p for p in pts if 2 <= p[0] < site["ny"] - 2]
    pick = [pts[int(i)] for i in np.linspace(0, len(pts) - 1, n)]
    wells = []
    for iy, ix in pick:
        jx = min(ix + offset_km, site["nx"] - 2)
        while not site["ag"][iy, jx] and jx < site["nx"] - 2:
            jx += 1
        wells.append((iy, jx))
    return wells


def responses(site, theta, wells, pts, workers=None, dry=0.0):
    """응답 행렬: 우물 w를 용량으로 3개월 뽑을 때 관리지점 추가 침하(mm)·하부 대수층 수위 변화(m), 용량 1 m3/일당.
    기준 실행(계획 우물 0) 1회 + 우물별 1회."""
    W = len(wells)
    plans = [None] + [w for w in range(W)]
    thetas = [theta] * (W + 1)

    def mk(w):
        rates = np.zeros((MONTHS, W))
        if w is not None:
            rates[:, w] = CAP
        return {"months": MONTHS, "wells": wells, "rates": rates, "dry": dry}
    outs = r.run_ensemble_plans(site, thetas, [mk(w) for w in plans], workers)
    if any(o is None for o in outs):
        return None
    S0, H0 = outs[0]
    iy, ix = np.array(pts).T
    n0 = len(site["months"]) - 2                    # 자료 끝 시점 인덱스(출력 기준)
    base_sub = S0[-1, iy, ix] - S0[n0, iy, ix]
    Rs = np.zeros((len(pts), W)); Rh = np.zeros((len(pts), W))
    for w in range(W):
        S, H = outs[w + 1]
        Rs[:, w] = ((S[-1, iy, ix] - S[n0, iy, ix]) - base_sub) / CAP
        Rh[:, w] = (H[-1, 2, iy, ix] - H0[-1, 2, iy, ix]) / CAP
    hist_min = np.nanmin(H0[: n0 + 1, 2, iy, ix], axis=0)           # 과거 최저 수위(하부 대수층 위)
    return {"Rs": Rs, "Rh": Rh, "base_sub": base_sub, "base_head_end": H0[-1, 2, iy, ix], "hist_min": hist_min}


def solve(resp_list, use_head=False):
    """변수: 우물별 평균 취수(m3/일) W개 + 외부 구매(m3). 비용 최소."""
    W = len(WELL_COST)
    c = np.concatenate([WELL_COST * DAYS, [PRICE]])
    A_eq = np.concatenate([np.full(W, DAYS), [1.0]])[None]; b_eq = [DEMAND]
    A_ub, b_ub = [], []
    for R in resp_list:
        for p in range(R["Rs"].shape[0]):
            A_ub.append(np.concatenate([R["Rs"][p], [0]])); b_ub.append(LIMIT_MM)
            if use_head:   # 기준 끝 수위 + 계획 영향 ≥ 과거 최저  →  -Rh·x ≤ base_end - hist_min
                A_ub.append(np.concatenate([-R["Rh"][p], [0]])); b_ub.append(max(R["base_head_end"][p] - R["hist_min"][p], 0.0))
    bounds = [(0, CAP)] * W + [(0, PURCHASE_CAP)]
    res = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        reason = ("우물 용량과 외부 구매 한도를 모두 써도 수요를 채울 수 없습니다." if W * CAP * DAYS + PURCHASE_CAP < DEMAND
                  else "침하·수위 기준을 지키면서 수요를 채울 수 없습니다. 추가 용수 확보가 필요합니다.")
        return {"feasible": False, "reason": reason}
    return {"feasible": True, "rates": res.x[:W].tolist(), "purchase": float(res.x[W]), "cost": float(res.fun)}


def baseline():
    x = np.full(len(WELL_COST), min(DEMAND / DAYS / len(WELL_COST), CAP))
    purchase = DEMAND - x.sum() * DAYS
    return {"feasible": True, "rates": x.tolist(), "purchase": float(purchase), "cost": float((WELL_COST * x * DAYS).sum() + PRICE * purchase)}


def risk(plan, resp_list, err_sd=0.0, use_head=False):
    """평가 앙상블에서 기준 초과 확률. err_sd: 응답 구조 오차(곱셈, 로그 표준편차)."""
    if not plan["feasible"]:
        return None
    x = np.array(plan["rates"])
    sub = np.array([R["Rs"] @ x for R in resp_list])                    # (구성원, 지점)
    worst = sub.max(1)
    out = {"exceed_prob": float(np.mean(worst > LIMIT_MM + 1e-9)), "worst_mean_mm": float(worst.mean()),
           "worst_p95_mm": float(np.percentile(worst, 95))}
    if use_head:
        h = np.array([R["base_head_end"] + R["Rh"] @ x - R["hist_min"] for R in resp_list])
        out["below_crit_prob"] = float(np.mean(h.min(1) < -1e-6))
    return out
