"""취수 계획을 '계획에 쓰지 않은' 땅속 상태로 다시 채점한다 (표본 밖 평가).

왜 필요한가
- run_plan.py의 E4 계획은 E4 앙상블 30개 모두에서 기준을 지키도록 만든 것이다. 그래서 같은 30개로 채점하면 기준 초과 0%가 정의상 나온다.
- 여기서는 같은 사후분포에서 **새로 30개**를 뽑아(계획에 안 씀) 각 계획을 다시 채점한다.
- 응답 행렬(선형 근사)만 믿지 않도록, E4 평년 계획은 새 30개 각각에 **MODFLOW로 직접** 넣어 돌린 값도 같이 낸다.
- 모델 구조 오차는 계획 응답에 바로 옮길 근거가 없어서, 응답이 모델보다 25%·50% 클 때(스트레스 시험)를 따로 보인다.

실행: python eval_plan_oos.py   (이전 버전 모델 기준, MODFLOW 약 660회. 10코어 맥 약 5~10분)
출력: results_plan/RESULTS_oos.md, oos.json
"""
import json
import os
import time

os.environ.setdefault("AQ_PUMPING", "uniform")     # 계획·날씨 결과(results_plan)를 만든 이전 버전 모델

import numpy as np  # noqa: E402

import plan_real as p  # noqa: E402
import real_model as r  # noqa: E402

MODEL = r.result_dir("model", create=False)
PLAN = r.result_dir("plan", create=False)
N_NEW = int(os.environ.get("AQ_OOS_N", 30))
SEED = 2026                                         # E4 앙상블(seed 0)과 다른 뽑기
SCEN = {"평년": 0.0, "가뭄 해": 1.0, "습한 해": -1.0}
STRESS = [1.0, 1.25, 1.5]


def draw_members(th, cov, n, seed):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(th)))
    M = th + rng.normal(size=(n, len(th))) @ L.T
    return np.clip(M, r.PRIOR_MEAN - 3 * r.PRIOR_SD, r.PRIOR_MEAN + 3 * r.PRIOR_SD)


def main():
    t0 = time.time()
    r.check_version(MODEL); r.check_version(PLAN)
    site = r.load_site()
    res = json.load(open(MODEL / "results.json"))
    th3, cov3 = np.array(res["params"]["E3"]), np.array(res["params"]["E3_cov"])
    pts, wells = p.mgmt_points(site), p.operator_wells(site)
    W = len(wells)
    plan_res = json.load(open(PLAN / "results.json"))
    wx = json.load(open(PLAN / "weather.json")) if (PLAN / "weather.json").exists() else {"plans": {}}
    plans = {k: v for k, v in plan_res["plans"].items() if v.get("feasible")}
    plans.update({k: v for k, v in wx["plans"].items() if v.get("feasible")})
    M = draw_members(th3, cov3, N_NEW, SEED)
    print(f"새 땅속 상태 {len(M)}개, 계획 {len(plans)}개, 시나리오 {len(SCEN)}개", flush=True)

    # 한 번에 모아서 병렬 실행: 구성원 × 시나리오 × (기준 1 + 우물 6) + E4 평년 계획 직접 실행
    def mk(w, dry, rates=None):
        x = np.zeros((p.MONTHS, W))
        if rates is not None:
            x[:] = rates
        elif w is not None:
            x[:, w] = p.CAP
        return {"months": p.MONTHS, "wells": wells, "rates": x, "dry": dry}
    jobs, keys = [], []
    for i, th in enumerate(M):
        for sc, d in SCEN.items():
            for w in [None] + list(range(W)):
                jobs.append((th, mk(w, d))); keys.append((i, sc, w))
        jobs.append((th, mk(None, 0.0, np.array(plans["E4"]["rates"])))); keys.append((i, "평년", "E4직접"))
    print(f"MODFLOW {len(jobs)}회 실행", flush=True)
    outs = r.run_ensemble_plans(site, [j[0] for j in jobs], [j[1] for j in jobs])
    O = dict(zip(keys, outs))

    iy, ix = np.array(pts).T
    n0 = len(site["months"]) - 2
    R = {sc: [] for sc in SCEN}; direct = []
    for i in range(len(M)):
        for sc in SCEN:
            o = [O[(i, sc, w)] for w in [None] + list(range(W))]
            if any(v is None for v in o):
                continue
            S0 = o[0][0]; base = S0[-1, iy, ix] - S0[n0, iy, ix]
            Rs = np.stack([((o[w + 1][0][-1, iy, ix] - o[w + 1][0][n0, iy, ix]) - base) / p.CAP for w in range(W)], 1)
            R[sc].append({"i": i, "Rs": Rs, "base": base})
        od = O[(i, "평년", "E4직접")]
        b0 = O[(i, "평년", None)]
        if od is not None and b0 is not None:
            d = (od[0][-1, iy, ix] - od[0][n0, iy, ix]) - (b0[0][-1, iy, ix] - b0[0][n0, iy, ix])
            direct.append(float(d.max()))

    out = {"n_new_members": int(len(M)), "seed": SEED, "n_ok": {sc: len(v) for sc, v in R.items()}, "plans": {}}
    for k, pl in plans.items():
        x = np.array(pl["rates"]); row = {}
        for sc in SCEN:
            worst = np.array([(q["Rs"] @ x).max() for q in R[sc]])
            row[sc] = {f"x{f}": float(np.mean(worst * f > p.LIMIT_MM + 1e-9)) for f in STRESS}
            row[sc]["worst_mean_mm"] = float(worst.mean()); row[sc]["worst_p95_mm"] = float(np.percentile(worst, 95))
        out["plans"][k] = {"cost": pl["cost"], "purchase": pl["purchase"], "oos": row,
                           "in_sample": (plan_res.get("risk", {}).get(k) or {}).get("exceed_prob",
                                         ((wx.get("risk", {}).get(k) or {}).get("평년") or {}).get("exceed_prob"))}
    x4 = np.array(plans["E4"]["rates"])
    lin4 = np.array([(q["Rs"] @ x4).max() for q in R["평년"]])
    out["E4_direct"] = {"n": len(direct), "exceed_prob": float(np.mean(np.array(direct) > p.LIMIT_MM + 1e-9)),
                        "worst_mean_mm": float(np.mean(direct)), "worst_max_mm": float(np.max(direct)),
                        "linear_worst_mean_mm": float(lin4.mean())}
    out["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(out, open(PLAN / "oos.json", "w"), ensure_ascii=False, indent=1, default=float)
    report(out)


NAMES = {"E1": "E1 기존 방식(용량 비례)", "E2": "E2 관측정만", "E3": "E3 관측정 + 위성", "E4": "E4 + 불확실성(강건)"}


def report(o):
    L = ["# 취수 계획 표본 밖 평가 (계획에 안 쓴 땅속 상태 30개)\n",
         f"같은 사후분포에서 새로 뽑은 {o['n_new_members']}개(seed {o['seed']})로 각 계획을 다시 채점했다. "
         f"계획은 results_plan/에 있는 것 그대로(이전 버전 모델). 기준: 계획 추가 침하 ≤ {p.LIMIT_MM}mm/3개월.\n",
         "## 1. 평년 기준 초과 확률\n",
         "| 계획 | 비용(천$) | 계획에 쓴 30개(표본 내) | **새 30개(표본 밖)** | 응답 1.25배 | 응답 1.5배 | 최대 추가 침하 평균(mm) |",
         "|---|---|---|---|---|---|---|"]
    for k, v in o["plans"].items():
        s = v["oos"]["평년"]; ins = v["in_sample"]
        L.append(f"| {NAMES.get(k, k)} | {v['cost'] / 1e3:,.0f} | {'-' if ins is None else f'{ins * 100:.0f}%'} | "
                 f"**{s['x1.0'] * 100:.0f}%** | {s['x1.25'] * 100:.0f}% | {s['x1.5'] * 100:.0f}% | {s['worst_mean_mm']:.2f} |")
    d = o["E4_direct"]
    L += ["", f"E4 계획을 새 {d['n']}개 각각에 **MODFLOW로 직접** 넣어 돌린 결과: 기준 초과 {d['exceed_prob'] * 100:.0f}%, "
          f"최대 추가 침하 평균 {d['worst_mean_mm']:.2f}mm(최대 {d['worst_max_mm']:.2f}mm). 같은 구성원의 응답 행렬 예측 평균은 {d['linear_worst_mean_mm']:.2f}mm.\n",
          f"## 2. 날씨 시나리오별 (새 {o['n_new_members']}개)\n",
          "| 계획 | 습한 해 | 평년 | 가뭄 해 |", "|---|---|---|---|"]
    for k, v in o["plans"].items():
        L.append(f"| {NAMES.get(k, k)} | " + " | ".join(f"{v['oos'][sc]['x1.0'] * 100:.0f}%" for sc in ["습한 해", "평년", "가뭄 해"]) + " |")
    L += ["", "## 3. 읽는 법\n",
          "- '계획에 쓴 30개' 값은 계획을 만든 자료로 채점한 것이라 E4는 정의상 0%다. 실제 위험에 가까운 값은 '새 30개' 열이다.",
          "- '응답 1.25배·1.5배'는 모델이 실제 땅의 반응을 25%·50% 작게 본 경우(모델 구조 오차)를 가정한 스트레스 시험이다. "
          "구조 오차의 실제 크기는 계획 응답 단위로 측정하지 못했다.",
          "- 모든 값은 실제 지역 자료로 보정한 모델 안에서의 시뮬레이션 결과다. 사업자 우물 6개는 가정이다.",
          f"\n계산 시간 {o['minutes']}분."]
    (PLAN / "RESULTS_oos.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
