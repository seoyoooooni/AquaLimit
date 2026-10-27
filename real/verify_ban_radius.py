"""금지 반경 확대안과 AquaLimit 계획을 MODFLOW 앙상블로 직접 다시 돌려 확인한다(policy_ban_radius.py는 선형 응답 계산).

확인하는 계획(상황마다)
- 금지 6km 고정(가뭄에도 지키는 규칙), 그해 기준을 맞추는 최소 반경 금지
- AquaLimit 규제 기준(선형 응답으로 모든 구성원·지점 한도), AquaLimit 여유 20%(한도 × 0.8로 다시 푼 계획)
4km 전면 금지와 AquaLimit 규제 기준의 MODFLOW 값은 policy_real.py 결과(results.json)를 그대로 읽는다.

실행: AQ_VARIANT=v3ops python3 verify_ban_radius.py   (MODFLOW 약 200회, 10코어 맥 약 5분, 2코어 약 25분)
출력: results_policy_<버전>/RESULTS_radius_verify.md, radius_verify.json
"""
import json
import time

import numpy as np
import pandas as pd

import real_model as r
import policy_real as pol
import policy_ban_radius as pbr

OUT = pol.OUT
BLOCKS = {"retro": ("2021.4~2022.9 대가뭄 재현", pol.RETRO, None),
          "습한 해": ("2026.5~7 습한 해", None, pol.SCENARIOS["습한 해"]),
          "평년": ("2026.5~7 평년", None, pol.SCENARIOS["평년"]),
          "가뭄·폭염": ("2026.5~7 가뭄·폭염", None, pol.SCENARIOS["가뭄·폭염"])}
TIGHT = 0.8


def metrics(vs, reg):
    return {"ok_points": float(np.mean(np.median(vs, 0) <= reg + 1e-6)), "ok_all_prob": float(np.mean(vs.max(1) <= reg + 1e-6)),
            "worst_med": float(np.median(vs.max(1))), "worst_p95": float(np.percentile(vs.max(1), 95)), "n": int(len(vs))}


def main():
    t0 = time.time()
    site = r.load_site()
    MODEL = r.result_dir("model", create=False); r.check_version(MODEL)
    M = np.load(MODEL / "E4_members.npy")
    setup = json.load(open(OUT / "setup.json"))
    ti = lambda c: tuple(int(v) for v in c)                     # setup.json은 칸 번호를 실수로 저장했다(json default=float)
    groups = [dict(g, cells=[ti(c) for c in g["cells"]]) for g in setup["groups"]]
    cells = [ti(c) for c in setup["cells"]]
    pts = [ti(p) for p in setup["pts"]]
    wells = pd.read_csv(OUT / "wells_used.csv")
    cost = np.array([g["cost"] for g in groups])
    summ = json.load(open(OUT / "radius_summary.json"))
    prev = json.load(open(OUT / "results.json"))
    out = {}
    for tag, (name, window, future) in BLOCKS.items():
        z = np.load(OUT / f"R_{tag}.npz")
        R = [{"Rs": z["Rs"][i], "sub0": z["sub0"][i]} for i in range(z["Rs"].shape[0])]
        x_sq, shape, T = z["x_sq"], z["shape"], float(z["T"])
        reg = pbr.REG_MM_PER_YR * len(shape) / 12.0
        Mb = M if window else M[:pol.N_FWD]
        cap = np.maximum(np.array([g["cap"] for g in groups]), x_sq)
        plans = {}
        s = summ[tag]
        for lab, rad in [(f"금지 {s['fixed_radius_km']:.1f}km 고정", s["fixed_radius_km"]), (f"금지 {s['min_radius_km']:.1f}km(그해 최소 반경)", s["min_radius_km"])]:
            frac = np.array([wells[(wells.group == g) & (wells.dist_km <= rad + 1e-9)].yield_gpm.sum() / max(wells[wells.group == g].yield_gpm.sum(), 1e-9)
                             for g in range(len(groups))])
            x = pbr.ban_x(x_sq, frac)
            if any(np.allclose(x, p["x"]) for p in plans.values()):
                continue
            c, v = pbr.cost_of(x, x_sq, cost, T)
            plans[lab] = {"x": x, "cost": c, "radius_km": rad}
        xa, ca, va = pbr.solve_reg(R, x_sq, cost, cap, T, reg * TIGHT)
        if xa is not None:
            plans[f"AquaLimit 여유 {round((1 - TIGHT) * 100)}%"] = {"x": xa, "cost": ca}
        res = {}
        for lab, p in plans.items():
            vs, vl, va_ = pol.verify(site, Mb, groups, cells, {"x": list(map(float, p["x"]))}, shape, window, pts, cells, future)
            res[lab] = dict(cost=p["cost"], radius_km=p.get("radius_km"), **metrics(vs, reg))
            print(f"[{name}] {lab}: ${p['cost'] / 1e6:.1f}M, MODFLOW 기준 지키는 지점 {res[lab]['ok_points'] * 100:.0f}%, "
                  f"모든 지점 확률 {res[lab]['ok_all_prob'] * 100:.0f}%, 최악 지점 {res[lab]['worst_med']:.1f}mm", flush=True)
        blk = prev["retro"] if tag == "retro" else prev["fwd"][tag]
        for k, lab in [("4km 전면 금지(2025 실제 정책)", "금지 4km(실제, policy_real)"), ("AquaLimit(규제 기준 0.5인치/년)", "AquaLimit 규제 기준(policy_real)")]:
            p = blk["policies"][k]
            res[lab] = {"cost": p["cost"], "ok_points": p["reg_ok_points"], "ok_all_prob": p["reg_ok_prob"], "worst_med": p["ver_sub_med"],
                        "worst_p95": p["ver_sub_p95"], "n": len(p["ver_sub_members"])}
        out[tag] = {"name": name, "reg_mm": reg, "policies": res}
        json.dump(out, open(OUT / "radius_verify.json", "w"), ensure_ascii=False, indent=1, default=float)
    report(out, round((time.time() - t0) / 60, 1))
    print((OUT / "RESULTS_radius_verify.md").read_text(encoding="utf-8"))


def report(out, minutes):
    L = ["# 금지 반경 확대안 MODFLOW 확인\n",
         "policy_ban_radius.py(선형 응답) 결과를 MODFLOW 앙상블로 직접 다시 돌렸다. '기준 지키는 지점' = 지점별 앙상블 중앙값이 기준 안인 비율, "
         "'모든 지점 확률' = 구성원 중 수로 38개 지점이 모두 기준 안인 비율. 과거 재현은 앙상블 30개, 다음 관개기는 15개.\n"]
    for tag, o in out.items():
        L += [f"## {o['name']} (기준 {o['reg_mm']:.1f}mm)\n",
              "| 계획 | 비용 (백만 $) | 기준 지키는 지점 | 모든 지점 확률 | 최악 지점 침하 (mm, 중앙값 / 상위 5%) |", "|---|---|---|---|---|"]
        for lab, p in sorted(o["policies"].items(), key=lambda kv: kv[1]["cost"]):
            L.append(f"| {lab} | {p['cost'] / 1e6:.1f} | {p['ok_points'] * 100:.0f}% | {p['ok_all_prob'] * 100:.0f}% | {p['worst_med']:.1f} / {p['worst_p95']:.1f} |")
        L.append("")
    L.append(f"계산 시간 {minutes}분.")
    (OUT / "RESULTS_radius_verify.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
