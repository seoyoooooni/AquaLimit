"""실제 검증 지역 취수 계획 E1~E4 (2026년 5~7월 관개기, 3개월).

상황
- 사업자 우물: DWR 우물 신고 자료(공개)의 실제 농업용 우물 중 취수 금지구역(수로 양쪽 2.5마일) 밖 6개.
  위치는 신고 좌표, 용량은 신고 양수량 × 가동률(신고값이 없으면 신고 중앙값 2,000gpm), 양수 비용은 정수위 + 양수 강하 깊이로 계산.
- 3개월 수요 = 우물 용량 합의 75%. 부족분은 외부 용수(수로 물 이전 구매, 0.81달러/m3 ≈ 1,000달러/AF)로 채운다. 구매 한도 = 수요의 3분의 1.
- 관리 기준: 계획으로 인한 수로 칸 추가 침하 ≤ LIMIT_MM (3개월). 수로 침하 기준(연 0.5인치 = 3개월 3.2mm, 이웃 관리기관 실제 기준) 중
  이 사업자에게 배정된 몫(가정). 다른 농가 취수(배경)는 역산한 패턴대로 계속된다.

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
CAP = np.full(6, 4000.0)         # m3/일 (setup_operator가 실제 우물 값으로 바꾼다)
DEMAND = 900_000.0               # m3 / 3개월
PRICE = 0.81                     # 달러/m3
PURCHASE_CAP = 300_000.0
WELL_COST = np.array([0.12, 0.10, 0.15, 0.13, 0.17, 0.11])
LIMIT_MM = 2.0                   # 3개월 계획 추가 침하 기준(mm): 0.5인치/년 기준(3개월 3.2mm) 중 이 사업자 몫(가정)
WELL_INFO = []


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


def setup_operator(site, n=6):
    """실제 신고 우물 중 금지구역 밖 n개를 사업자 우물로 쓴다. CAP·WELL_COST·DEMAND·PURCHASE_CAP를 그에 맞게 바꾼다."""
    global CAP, WELL_COST, DEMAND, PURCHASE_CAP, WELL_INFO
    import policy_real as pol
    w = pol.load_wells(site)
    w = w[~w.in_buffer].copy()
    w["cap"] = w.yield_gpm * pol.GPM_TO_M3D * pol.DUTY
    w["cost"] = 0.02 + pol.KWH_PRICE * 9.81 * (w.lift_m + pol.DRAWDOWN_M) / 3600.0 / pol.PUMP_EFF
    w = w.drop_duplicates(subset=["iy", "ix"])                       # 같은 1km 칸은 하나만
    # 수로에서 가까운 순서로 고르되 남북으로 퍼지게: 위도 구간 n개에서 가장 가까운 우물
    w = w.sort_values("DecimalLatitude")
    picks = []
    for chunk in np.array_split(w.index.to_numpy(), n):
        if len(chunk):
            picks.append(w.loc[chunk].sort_values("dist_km").index[0])
    w = w.loc[picks]
    CAP = w.cap.to_numpy(float); WELL_COST = w.cost.to_numpy(float)
    DEMAND = float(0.75 * CAP.sum() * DAYS); PURCHASE_CAP = DEMAND / 3
    WELL_INFO = [{"name": chr(65 + i), "wcr": str(x.WCRNumber), "lat": float(x.DecimalLatitude), "lon": float(x.DecimalLongitude),
                  "dist_km": float(x.dist_km), "cap": float(x.cap), "cost": float(x.cost), "yield_reported": bool(x.yield_reported),
                  "depth_ft": float(x.TotalCompletedDepth)} for i, (_, x) in enumerate(w.iterrows())]
    return [(int(a), int(b)) for a, b in zip(w.iy, w.ix)]


def operator_wells(site, n=6, offset_km=3):
    """사업자 우물. v3 모델: 실제 신고 우물(금지구역 밖, setup_operator). 이전 버전: 수로 동쪽 3km 가정 우물(10/13 결과 재현용)."""
    if r.V3:
        return setup_operator(site, n)
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


def responses(site, theta, wells, pts, workers=None, dry=0.0, future=None):
    """응답 행렬: 우물 w를 용량으로 3개월 뽑을 때 관리지점 추가 침하(mm)·하부 대수층 수위 변화(m), 용량 1 m3/일당.
    기준 실행(계획 우물 0) 1회 + 우물별 1회."""
    W = len(wells)
    plans = [None] + [w for w in range(W)]
    thetas = [theta] * (W + 1)

    def mk(w):
        rates = np.zeros((MONTHS, W))
        if w is not None:
            rates[:, w] = CAP[w]
        return {"months": MONTHS, "wells": wells, "rates": rates, "dry": dry, "future": future}
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
        Rs[:, w] = ((S[-1, iy, ix] - S[n0, iy, ix]) - base_sub) / CAP[w]
        Rh[:, w] = (H[-1, 2, iy, ix] - H0[-1, 2, iy, ix]) / CAP[w]
    hist_min = np.nanmin(H0[: n0 + 1, 2, iy, ix], axis=0)           # 과거 최저 수위(하부 대수층 위)
    return {"Rs": Rs, "Rh": Rh, "base_sub": base_sub, "base_head_end": H0[-1, 2, iy, ix], "hist_min": hist_min}


def solve(resp_list, use_head=False, inflate=0.0):
    """inflate: 모델 구조 오차 여유. 응답을 (1+inflate)배로 키워 계획한다(실제 땅이 모델보다 그만큼 더 반응해도 기준을 지키게)."""
    """변수: 우물별 평균 취수(m3/일) W개 + 외부 구매(m3). 비용 최소."""
    W = len(WELL_COST)
    if use_head:
        # 계획 우물을 하나도 안 돌려도(배경 취수만으로) 이미 과거 최저 수위 아래면, 이 사업자 계획으로는 임계수위를 지킬 수 없다.
        # (예전 코드는 이 경우 제약을 0으로 바꿔 '취수 0 + 전량 구매'를 가능으로 냈다 → 수정)
        short = [float(np.min(R["base_head_end"] - R["hist_min"])) for R in resp_list]
        if min(short) < 0:
            frac = float(np.mean([v < 0 for v in short]))
            return {"feasible": False, "background_below_crit": True, "background_below_crit_frac": frac,
                    "reason": f"다른 농가 취수만으로도 이미 과거 최저 수위 아래입니다(앙상블 {frac * 100:.0f}%). "
                              "이 사업자의 계획만으로는 임계수위를 지킬 수 없고, 지역 전체 취수 조정이 필요합니다."}
    c = np.concatenate([WELL_COST * DAYS, [PRICE]])
    A_eq = np.concatenate([np.full(W, DAYS), [1.0]])[None]; b_eq = [DEMAND]
    A_ub, b_ub = [], []
    for R in resp_list:
        for p in range(R["Rs"].shape[0]):
            A_ub.append(np.concatenate([R["Rs"][p] * (1 + inflate), [0]])); b_ub.append(LIMIT_MM)
            if use_head:   # 기준 끝 수위 + 계획 영향 ≥ 과거 최저  →  -Rh·x ≤ base_end - hist_min
                A_ub.append(np.concatenate([-R["Rh"][p], [0]])); b_ub.append(R["base_head_end"][p] - R["hist_min"][p])
    bounds = [(0, float(c_)) for c_ in CAP] + [(0, PURCHASE_CAP)]
    res = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        reason = ("우물 용량과 외부 구매 한도를 모두 써도 수요를 채울 수 없습니다." if CAP.sum() * DAYS + PURCHASE_CAP < DEMAND
                  else "침하·수위 기준을 지키면서 수요를 채울 수 없습니다. 추가 용수 확보가 필요합니다.")
        return {"feasible": False, "reason": reason}
    return {"feasible": True, "rates": res.x[:W].tolist(), "purchase": float(res.x[W]), "cost": float(res.fun)}


def baseline():
    x = np.minimum(CAP * DEMAND / DAYS / CAP.sum(), CAP)                 # 용량 비례
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
