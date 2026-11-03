"""Month-by-month forward planning with the calibrated MODFLOW model.

The model state ends in April 2026. All following months, including the bridge
to the selected start, use an explicit weather and water-allocation scenario.
Only the operator's incremental compaction is constrained; total background
subsidence is reported separately.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import real_model as r


SCENARIOS = {
    "평년": {"pfac": 1.0, "dT": 0.0, "alloc": 0.30, "dry": 0.0},
    "가뭄·폭염": {"pfac": 0.6, "dT": 2.5, "alloc": 0.15, "dry": 1.0},
    "습한 해": {"pfac": 1.4, "dT": -1.5, "alloc": 0.45, "dry": -1.0},
}

_SITE = None
_PTS = None
_START = None
_COUNT = None


def _init(site, pts, start_index, count):
    global _SITE, _PTS, _START, _COUNT
    _SITE, _PTS, _START, _COUNT = site, np.array(pts, int), start_index, count


def _run_case(case):
    theta, plan = case
    S, _ = r.run_model(_SITE, theta, plan=plan)
    iy, ix = _PTS.T
    # One output per historical month from July 2015 onward. The selected
    # period starts after the historical output plus bridge months.
    end_hist = len(_SITE["months"]) - 2
    first = end_hist + _START
    return S[first:first + _COUNT, iy, ix] - S[first - 1, iy, ix]


def forecast_months(start, count, last_observed=None):
    start = pd.Period(start, freq="M")
    last = pd.Period(last_observed or r.END, freq="M")
    bridge = start.ordinal - last.ordinal - 1
    if bridge < 0:
        raise ValueError(f"시작 월은 모델 기준월 {last} 이후여야 합니다.")
    if count < 1 or count > 6:
        raise ValueError("계획 기간은 1~6개월입니다.")
    return pd.period_range(start, periods=count, freq="M"), bridge


def build_responses(site, thetas, pts, wells, caps, start, count, scenario, workers=4):
    """Physical impulse responses for every (month, well), at every month end.

    The bridge assumes this operator pumps at 75% of well capacity. During the
    plan period, the baseline operator pumps nothing. Each perturbation pumps
    one well for one month, with all other months unchanged.
    """
    months, bridge = forecast_months(start, count, site["months"][-1])
    extra = bridge + count
    W = len(wells)
    caps = np.asarray(caps, float)
    bridge_rates = np.zeros((extra, W))
    bridge_rates[:bridge] = 0.75 * caps
    common = {"months": extra, "wells": list(wells), "future": SCENARIOS[scenario],
              "replace_cells": list(wells)}
    plans = []
    for rates in [bridge_rates] + [None] * (count * W):
        if rates is not None:
            plans.append(dict(common, rates=rates))
            continue
        j = len(plans) - 1
        m, w = divmod(j, W)
        x = bridge_rates.copy()
        x[bridge + m, w] = caps[w]
        plans.append(dict(common, rates=x))
    baseline, response = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(site, pts, bridge + 1, count)) as pool:
        for i, out in enumerate(pool.map(_run_case,
                                         ((theta, plan) for theta in thetas for plan in plans))):
            k = i % len(plans)
            if k == 0:
                baseline.append(out)
                response.append(np.empty((count, len(pts), count * W)))
            else:
                response[-1][:, :, k - 1] = (out - baseline[-1]) / caps[(k - 1) % W]
    return {"months": [str(m) for m in months], "days": np.array([m.days_in_month for m in months]),
            "baseline": np.array(baseline), "response": np.array(response), "bridge": bridge,
            "scenario": scenario, "wells": list(wells), "caps": caps}


def response_cache_path(site, thetas, wells, caps, start, count, scenario,
                        cache_dir=None):
    cache_dir = Path(cache_dir or Path(__file__).parent / f"results_future_{r.TAG}")
    key = {"version": 2, "model": r.model_version(), "end": str(site["months"][-1]),
           "start": start, "count": count, "scenario": scenario, "wells": wells,
           "caps": np.asarray(caps).tolist(),
           "thetas": hashlib.sha256(np.asarray(thetas).tobytes()).hexdigest()}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]
    return cache_dir / f"response_{digest}.npz"


def load_cached_responses(site, thetas, wells, caps, start, count, scenario,
                          cache_dir=None):
    path = response_cache_path(site, thetas, wells, caps, start, count,
                               scenario, cache_dir)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].item() if k in ("bridge", "scenario") else z[k]
                    for k in z.files}
    return None


def cached_responses(site, thetas, pts, wells, caps, start, count, scenario,
                     cache_dir=None, workers=4):
    """Keep completed physical responses available for a live demo restart."""
    path = response_cache_path(site, thetas, wells, caps, start, count,
                               scenario, cache_dir)
    cached = load_cached_responses(site, thetas, wells, caps, start, count,
                                   scenario, cache_dir)
    if cached is not None:
        return cached
    path.parent.mkdir(exist_ok=True)
    out = build_responses(site, thetas, pts, wells, caps, start, count, scenario, workers)
    np.savez_compressed(path, **out)
    return out


def verify_schedule(site, thetas, pts, data, rates, workers=4):
    """Run the optimized monthly schedule directly through MODFLOW."""
    M, W = np.asarray(rates).shape
    bridge = int(data["bridge"])
    all_rates = np.zeros((bridge + M, W))
    all_rates[:bridge] = 0.75 * np.asarray(data["caps"])
    all_rates[bridge:] = rates
    wells = [tuple(map(int, w)) for w in data["wells"]]
    plan = {"months": bridge + M, "wells": wells, "rates": all_rates,
            "future": SCENARIOS[str(data["scenario"])], "replace_cells": wells}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(site, pts, bridge + 1, M)) as pool:
        full = np.array(list(pool.map(_run_case, ((th, plan) for th in thetas))))
    extra = full - data["baseline"]
    linear = np.einsum("empc,c->emp", data["response"], np.asarray(rates).ravel())
    return {"extra": extra, "total": full,
            "max_abs_linear_error_mm": float(np.max(np.abs(extra - linear)))}


def cached_verification(site, thetas, pts, data, rates, cache_dir=None,
                        compute=False):
    """Load a direct MODFLOW check, or compute and save it on request."""
    cache_dir = Path(cache_dir or Path(__file__).parent / f"results_future_{r.TAG}")
    key = (np.asarray(thetas).tobytes() + np.asarray(rates).tobytes()
           + np.asarray(data["wells"]).tobytes() + str(data["months"]).encode()
           + str(data["scenario"]).encode())
    path = cache_dir / f"verified_{hashlib.sha256(key).hexdigest()[:20]}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].item() if k == "max_abs_linear_error_mm" else z[k]
                    for k in z.files}
    if not compute:
        return None
    out = verify_schedule(site, thetas, pts, data, rates)
    cache_dir.mkdir(exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def optimize(data, costs, price, demand_fraction, limit_per_92days, margin,
             purchase_fraction=1.0):
    """LP: monthly pumping and purchases; cumulative limits at every point/date/member."""
    days = np.asarray(data["days"], float)
    caps = np.asarray(data["caps"], float)
    costs = np.asarray(costs, float)
    M, W = len(days), len(caps)
    demand = np.asarray(demand_fraction, float) * caps.sum() * days
    c = np.r_[np.repeat(costs[None], M, axis=0).ravel() * np.repeat(days, W),
              np.full(M, price)]
    eq = np.zeros((M, M * W + M))
    for m in range(M):
        eq[m, m * W:(m + 1) * W] = days[m]
        eq[m, M * W + m] = 1.0
    R = np.asarray(data["response"])
    # The operator's 2 mm / 92 d allowance accrues with elapsed calendar days.
    limit = limit_per_92days * np.cumsum(days) / 92.0
    A = np.pad(R.reshape(-1, M * W) * (1 + margin), ((0, 0), (0, M)))
    b = np.broadcast_to(limit[None, :, None], R.shape[:3]).ravel()
    bounds = [(0, float(cap)) for _ in range(M) for cap in caps]
    bounds += [(0, float(purchase_fraction * d)) for d in demand]
    res = linprog(c, A_ub=A, b_ub=b, A_eq=eq, b_eq=demand,
                  bounds=bounds, method="highs")
    if not res.success:
        return {"feasible": False, "reason": "이 조건에서는 월별 수요와 침하 한도를 함께 만족할 수 없습니다. 외부 용수 한도나 침하 한도를 조정해 주세요."}
    rates = res.x[:M * W].reshape(M, W)
    purchase = res.x[M * W:]
    extra = np.einsum("empc,c->emp", R, rates.ravel())
    total = np.asarray(data["baseline"]) + extra
    return {"feasible": True, "rates": rates, "purchase": purchase, "demand": demand,
            "cost": float(res.fun), "limit": limit, "extra": extra, "total": total,
            "max_extra": extra.max(axis=2), "max_total": total.max(axis=2)}


def proportional_plan(data, demand_fraction):
    """Comparison: pump each well at the same fraction of capacity each month."""
    days = np.asarray(data["days"], float)
    caps = np.asarray(data["caps"], float)
    rates = np.minimum(np.asarray(demand_fraction, float)[:, None], 1.0) * caps[None]
    extra = np.einsum("empc,c->emp", data["response"], rates.ravel())
    return {"rates": rates, "extra": extra, "max_extra": extra.max(axis=2)}
