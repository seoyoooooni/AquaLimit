from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class Plan:
    feasible: bool
    pumping: np.ndarray = None
    purchase: np.ndarray = None
    cost: float = None
    reason: str = None


def solve(site, demand, responses, limit_mm, purchase_cost, purchase_cap):
    n_wells = len(site.well_names)
    n_weeks = len(demand)
    n_q = n_wells * n_weeks
    n_var = n_q + n_weeks

    c = np.concatenate([np.tile(site.well_cost, n_weeks), np.full(n_weeks, purchase_cost)])

    a_eq = np.zeros((n_weeks, n_var))
    for t in range(n_weeks):
        a_eq[t, t * n_wells:(t + 1) * n_wells] = 1.0
        a_eq[t, n_q + t] = 1.0

    rows = []
    for r in responses:
        for p in range(r.shape[0]):
            row = np.zeros(n_var)
            row[:n_q] = np.tile(r[p] / 1000.0, n_weeks)
            rows.append(row)
    a_ub = np.array(rows)
    b_ub = np.full(len(rows), limit_mm)

    bounds = [(0, cap) for _ in range(n_weeks) for cap in site.well_capacity]
    bounds += [(0, purchase_cap)] * n_weeks

    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=demand, bounds=bounds, method="highs")
    if res.status != 0:
        return Plan(feasible=False, reason=_infeasible_reason(site, demand, purchase_cap))

    x = res.x
    return Plan(
        feasible=True,
        pumping=x[:n_q].reshape(n_weeks, n_wells),
        purchase=x[n_q:],
        cost=float(res.fun),
    )


def _infeasible_reason(site, demand, purchase_cap):
    if np.any(demand > site.well_capacity.sum() + purchase_cap):
        return "우물 용량과 외부 구매 한도를 모두 써도 수요를 채울 수 없습니다."
    return "침하 기준을 지키면서 수요를 채울 수 없습니다. 추가 용수 확보 없이는 수요를 충족할 수 없습니다."
