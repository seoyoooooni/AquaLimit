"""합성 대수층 E1~E4 비교 실험: 역산 복원 -> 검증기간 예측 -> 취수 계획 -> 참값으로 평가.

실행: python3 synthetic/run_experiment.py [--trials 10] [--members 30]
결과: synthetic/results/RESULTS.md, results.json (시행별 캐시: results/trials/, 중단 후 다시 실행하면 이어서 계산)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "demo"))

import aquifer as a  # noqa: E402
import calibrate as c  # noqa: E402
from optimizer import solve  # noqa: E402
from scenario import build_site, weekly_demand  # noqa: E402

# 데모 기본값과 같은 계획 조건
MONTHLY_DEMAND = 30000.0
LIMIT_MM = 6.0
PURCHASE_COST = 1500.0
PURCHASE_CAP = 3000.0
METHODS = ["E1", "E2", "E3", "E4"]
LABEL = {
    "E1": "E1 기존 계획(용량 비례 배분)",
    "E2": "E2 지상 관측만",
    "E3": "E3 위성 + 단일 추정",
    "E4": "E4 위성 + 불확실성(앙상블)",
}


def rel_err(est, true):
    return np.abs(est - true) / true


def baseline_plan(site, demand):
    """E1: 모델 없이 수요를 우물 용량 비례로 배분, 부족분만 외부 구매."""
    from optimizer import Plan
    share = site.well_capacity / site.well_capacity.sum()
    pumping = np.minimum(demand[:, None] * share[None, :], site.well_capacity[None, :])
    purchase = demand - pumping.sum(axis=1)
    cost = float((pumping * site.well_cost).sum() + purchase.sum() * PURCHASE_COST)
    return Plan(True, pumping, purchase, cost)


def one_trial(seed, n_members):
    truth = a.make_truth(seed)
    obs = a.make_observations(truth, seed)
    t0 = time.time()
    models = c.calibrate_all(obs, n_members, seed)
    calib_sec = time.time() - t0
    e4 = models["E4"]
    out = {"seed": seed, "calib_sec": calib_sec}

    # 1) 참값 복원 오차
    true_T, true_S = 10 ** truth.theta[0], 10 ** truth.theta[1]
    true_sk_px = truth.sk(obs.insar_xy)
    true_sk_pt = truth.sk(a.POINT_XY)
    rec = {}
    for k in ["E2", "E3"]:
        m = models[k]
        rec[k] = {
            "T": rel_err(10 ** m.log_ts[0], true_T), "S": rel_err(10 ** m.log_ts[1], true_S),
            "Sk_px": float(np.median(rel_err(m.sk_px, true_sk_px))),
            "Sk_pt": float(np.max(rel_err(m.sk(a.POINT_XY), true_sk_pt))),
        }
    ens_ts = np.array([m.log_ts for m in e4])
    ens_px = np.array([m.sk_px for m in e4])
    ens_pt = np.array([m.sk(a.POINT_XY) for m in e4])
    rec["E4"] = {
        "T": rel_err(10 ** ens_ts[:, 0].mean(), true_T), "S": rel_err(10 ** ens_ts[:, 1].mean(), true_S),
        "Sk_px": float(np.median(rel_err(ens_px.mean(0), true_sk_px))),
        "Sk_pt": float(np.max(rel_err(ens_pt.mean(0), true_sk_pt))),
    }
    lo, hi = np.percentile(ens_px, 5, axis=0), np.percentile(ens_px, 95, axis=0)
    rec["E4"]["coverage90"] = float(((true_sk_px >= lo) & (true_sk_px <= hi)).mean())
    out["recovery"] = rec

    # 2) 검증기간(540~720일) 예측 오차 — 관측으로 확인하는 성능
    mi, mh = obs.insar_times > a.SPLIT_DAY, obs.head_times > a.SPLIT_DAY
    valid = ~np.isnan(obs.insar_mm[mi])

    def pred_err(m_list):
        subs, heads = zip(*[a.forward(m.theta, obs, m.sk) for m in m_list])
        sub, hd = np.mean(subs, 0), np.mean(heads, 0)
        return (float(np.sqrt(np.mean((sub[mi][valid] - obs.insar_mm[mi][valid]) ** 2))),
                float(np.sqrt(np.mean((hd[mh] - obs.head_drawdown_m[mh]) ** 2))))
    out["prediction"] = {"E2": pred_err([models["E2"]]), "E3": pred_err([models["E3"]]), "E4": pred_err(e4)}

    # 3) 취수 계획 -> 참값 대수층에서 평가 (실행하지 않은 계획이므로 시뮬레이션 결과)
    site = build_site()
    demand = weekly_demand(MONTHLY_DEMAND)
    resp = {
        "E2": c_resp([models["E2"]]), "E3": c_resp([models["E3"]]), "E4": c_resp(e4),
    }
    true_resp = a.response_matrix(truth.theta, sk_fn=truth.sk)
    plans = {"E1": baseline_plan(site, demand)}
    for k in ["E2", "E3", "E4"]:
        plans[k] = solve(site, demand, resp[k], LIMIT_MM, PURCHASE_COST, PURCHASE_CAP)
    pl = {}
    for k, p in plans.items():
        if not p.feasible:
            pl[k] = {"feasible": False, "reason": p.reason}
            continue
        sub = true_resp @ (p.pumping.sum(axis=0) / 1000.0)
        pl[k] = {"feasible": True, "cost": p.cost, "purchase": float(p.purchase.sum()),
                 "true_worst_mm": float(sub.max()), "exceed": bool(sub.max() > LIMIT_MM + 1e-6)}
    out["planning"] = pl
    return out


def c_resp(models):
    return np.array([a.response_matrix(m.theta, sk_fn=m.sk) for m in models])


def infeasible_test():
    """수요를 크게 올리고 기준을 강하게 하면 해 대신 실행 불가와 원인을 반환하는지 확인."""
    truth = a.make_truth(0)
    obs = a.make_observations(truth, 0)
    e3 = c.calibrate_e3(c.subset(obs, a.SPLIT_DAY))
    site = build_site()
    r = c_resp([e3])
    cases = {
        "수요 40,000톤 · 기준 1mm (침하 기준 원인)": (40000.0, 1.0),
        "수요 60,000톤 · 기준 6mm (용량 초과)": (60000.0, 6.0),
        "수요 30,000톤 · 기준 6mm (정상)": (30000.0, 6.0),
    }
    res = {}
    for name, (dem, lim) in cases.items():
        p = solve(site, weekly_demand(dem), r, lim, PURCHASE_COST, PURCHASE_CAP)
        res[name] = "계획 산출" if p.feasible else f"실행 불가 — {p.reason}"
    return res


def fmt_pct(x):
    return f"{x * 100:.1f}%"


def write_report(trials, infeas, args, path):
    n = len(trials)
    L = []
    L.append("# 합성 대수층 E1~E4 비교 결과\n")
    L.append(f"참값을 아는 가상 대수층 {n}개(seed 0~{n - 1})에서 같은 실험을 반복했다. "
             f"앙상블 크기 {args.members}. 계획 조건: 월 수요 {MONTHLY_DEMAND:,.0f}톤, 침하 기준 {LIMIT_MM}mm/월, "
             f"외부 용수 {PURCHASE_COST:,.0f}원/톤·주 {PURCHASE_CAP:,.0f}톤 한도.\n")
    L.append("모든 수치는 합성 자료에서 나온 값이며 실제 지역 성능을 뜻하지 않는다.\n")

    L.append("## 1. 역산 복원 오차 (R2 관문 판단 자료)\n")
    L.append("| 방식 | T 오차 | S 오차 | 화소별 Sk 오차(중앙값) | 관리지점 Sk 최대 오차 |")
    L.append("|---|---|---|---|---|")
    for k in ["E2", "E3", "E4"]:
        r = [t["recovery"][k] for t in trials]
        L.append(f"| {LABEL[k]} | {fmt_pct(np.mean([x['T'] for x in r]))} | {fmt_pct(np.mean([x['S'] for x in r]))} | "
                 f"{fmt_pct(np.mean([x['Sk_px'] for x in r]))} | {fmt_pct(np.mean([x['Sk_pt'] for x in r]))} |")
    cov = np.mean([t["recovery"]["E4"]["coverage90"] for t in trials])
    L.append(f"\n{n}회 평균. E4는 앙상블 평균 기준. E4 90% 구간이 참값 Sk를 포함한 화소 비율: {fmt_pct(cov)} (이상적이면 90%).\n")

    L.append("## 2. 검증기간 예측 오차 (관측으로 확인하는 성능)\n")
    L.append("0~540일 자료로만 보정하고 540~720일 관측과 비교했다.\n")
    L.append("| 방식 | InSAR 침하 RMSE (mm) | 관측정 수위 RMSE (m) |")
    L.append("|---|---|---|")
    for k in ["E2", "E3", "E4"]:
        p = np.array([t["prediction"][k] for t in trials])
        L.append(f"| {LABEL[k]} | {p[:, 0].mean():.2f} | {p[:, 1].mean():.3f} |")
    L.append(f"\n관측 잡음 자체가 InSAR {a.SIGMA_INSAR_MM}mm, 수위 {a.SIGMA_HEAD_M}m이므로 이 값 근처면 잡음 수준까지 예측한 것이다.\n")

    L.append("## 3. 취수 계획을 참값 대수층에 적용 (시뮬레이션 결과)\n")
    L.append("| 방식 | 실행 가능 | 기준 초과 빈도 | 초과 시 평균 초과량 (mm) | 참값 최대 침하 평균 (mm) | 평균 총비용 (만원) | 평균 외부 구매 (톤) |")
    L.append("|---|---|---|---|---|---|---|")
    for k in METHODS:
        ps = [t["planning"][k] for t in trials]
        ok = [p for p in ps if p["feasible"]]
        if not ok:
            L.append(f"| {LABEL[k]} | 0/{n} | - | - | - | - | - |")
            continue
        over = [p["true_worst_mm"] - LIMIT_MM for p in ok if p["exceed"]]
        over_s = f"{np.mean(over):.2f}" if over else "-"
        L.append(f"| {LABEL[k]} | {len(ok)}/{n} | {sum(p['exceed'] for p in ok)}/{len(ok)} | {over_s} | "
                 f"{np.mean([p['true_worst_mm'] for p in ok]):.2f} | {np.mean([p['cost'] for p in ok]) / 1e4:,.0f} | "
                 f"{np.mean([p['purchase'] for p in ok]):,.0f} |")
    L.append("\n기준 초과 빈도 = 참값 대수층에서 관리지점 중 최대 침하가 기준을 넘은 실험 수.\n")

    L.append("## 4. 실행 불가 판정 테스트\n")
    L.append("| 입력 | 결과 |")
    L.append("|---|---|")
    for k, v in infeas.items():
        L.append(f"| {k} | {v} |")

    L.append("\n## 5. 계산 시간\n")
    L.append(f"보정(E2+E3+E4 {args.members}개) 1회 평균 {np.mean([t['calib_sec'] for t in trials]):.1f}초.\n")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--members", type=int, default=30)
    args = ap.parse_args()
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    cache = out_dir / "trials"
    cache.mkdir(exist_ok=True)
    trials = []
    for s in range(args.trials):
        f = cache / f"seed{s}_m{args.members}.json"
        if f.exists():
            t = json.loads(f.read_text(encoding="utf-8"))
        else:
            t = one_trial(s, args.members)
            f.write_text(json.dumps(t, ensure_ascii=False, default=float), encoding="utf-8")
        trials.append(t)
        pl = t["planning"]
        print(f"seed {s}: " + ", ".join(
            f"{k} {'불가' if not pl[k]['feasible'] else ('초과' if pl[k]['exceed'] else '준수')}" for k in METHODS),
            flush=True)
    infeas = infeasible_test()
    (out_dir / "results.json").write_text(json.dumps({"trials": trials, "infeasible_test": infeas},
                                                     ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_report(trials, infeas, args, out_dir / "RESULTS.md")
    print((out_dir / "RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
