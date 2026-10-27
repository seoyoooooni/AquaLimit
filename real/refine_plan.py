"""AquaLimit 계획을 MODFLOW로 확인하고 고쳐 다시 푼다(순차 선형 계획) + 금지 반경도 MODFLOW로 직접 찾는다 → 같은 신뢰도에서 비용 비교.

왜: 선형 응답으로 '모든 구성원·지점 기준 안'이 되게 푼 AquaLimit 계획을 MODFLOW로 돌리면 모든 지점 기준 확률이
    평년 60%, 가뭄 60%, 습한 해 80%로 떨어졌다(verify_ban_radius.py). 선형 응답은 취수를 '늘려서' 잰 기울기인데 계획은 '줄이는' 쪽이라 비선형이 생긴다.
    금지 반경 확대안도 선형으로 고른 최소 반경이 MODFLOW에선 80~87%였다. 신뢰도가 다른 계획끼리 비용을 비교하면 공정하지 않다.
방법
  AquaLimit: 계획 x_k를 MODFLOW 앙상블로 돌려 구성원·지점별 차이 off = MODFLOW − 선형 예측을 구하고,
             '선형(x) + off ≤ 기준'으로 다시 푼다(최대 4번). 모든 구성원·모든 지점이 MODFLOW로 기준 안이 되면 멈춘다.
  금지 반경: MODFLOW로 모든 구성원·모든 지점이 기준 안이 되는 가장 작은 반경을 0.1km 단위로 찾는다(그해 날씨를 미리 안다고 가정 = 금지 쪽에 가장 유리).
  고정 금지 규칙: 2026년 세 상황 중 가장 큰 최소 반경(가뭄에도 지키는 규칙)을 모든 해에 쓴다.
실행: AQ_VARIANT=v3ops python3 refine_plan.py   (MODFLOW 약 400~600회, 10코어 맥 약 15분)
      AQ_VARIANT=v3ops python3 refine_plan.py --aq-only   (금지 반경 탐색은 지난 결과 재사용, AquaLimit 보정만, 약 7분)
      AQ_VARIANT=v3ops python3 refine_plan.py --report    (계산 없이 표·그림만)
출력: results_policy_<버전>/RESULTS_refine.md, refine.json, policy-4-refine.png
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import real_model as r
import policy_real as pol
import policy_ban_radius as pbr

OUT = pol.OUT
BLOCKS = {"retro": ("2021.4~2022.9 대가뭄 재현", pol.RETRO, None),
          "습한 해": ("2026.5~7 습한 해", None, pol.SCENARIOS["습한 해"]),
          "평년": ("2026.5~7 평년", None, pol.SCENARIOS["평년"]),
          "가뭄·폭염": ("2026.5~7 가뭄·폭염", None, pol.SCENARIOS["가뭄·폭염"])}
N_IT = 6
TARGET = 1.0           # 모든 구성원에서 수로 모든 지점이 기준 안
AQ_ONLY = "--aq-only" in sys.argv   # 금지 반경 탐색은 지난 refine.json을 그대로 쓰고 AquaLimit 보정만 다시
MARGIN_MM = 0.05      # 보정 계획은 기준에 딱 붙으므로 MODFLOW 수치 잡음·반복 사이 변화만큼 여유(보정 k회째 0.05×k mm)


def solve_corr(R, x_sq, cost, cap, T, reg, alpha, beta):
    """pbr.solve_reg와 같은 선형계획 + 구성원·지점별 MODFLOW 보정.
    침하 예측 = sub0 + alpha × (선형 응답 변화) + beta. alpha = 'MODFLOW 변화 ÷ 선형 변화'(비율 보정), beta = 더하는 보정."""
    G = len(x_sq)
    c = np.r_[cost * T, pbr.PRICE]
    A, b = [], []
    for i, Ri in enumerate(R):
        for p in range(Ri["Rs"].shape[0]):
            a = alpha[i, p] * Ri["Rs"][p]
            A.append(np.r_[a, 0.0]); b.append(reg - beta[i, p] - Ri["sub0"][p] + a @ x_sq)
    res = linprog(c, A_ub=np.array(A), b_ub=b, A_eq=np.r_[np.full(G, T), 1.0][None], b_eq=[T * x_sq.sum()],
                  bounds=[(0, cg) for cg in cap] + [(0, None)], method="highs")
    return (res.x[:G], float(res.fun)) if res.status == 0 else (None, np.nan)


def correction(R, x, x_sq, vs, min_change=0.2):
    """MODFLOW 결과 vs(구성원, 지점)로 보정값을 만든다. 선형 예측 변화가 충분히 큰 곳은 비율(alpha), 아니면 차이(beta)."""
    sub0 = np.array([Ri["sub0"] for Ri in R])
    lin = pbr.predict(R, x, x_sq)
    if len(vs) != len(lin):                                   # 실패한 구성원이 있으면 지점별 최대 차이를 모두에게(보수적)
        return np.ones_like(lin), np.broadcast_to(np.max(vs, 0) - np.max(lin, 0), lin.shape).copy()
    dL, dM = lin - sub0, vs - sub0
    big = np.abs(dL) > min_change
    alpha = np.where(big, np.clip(dM / np.where(big, dL, 1.0), 0.3, 3.0), 1.0)
    beta = np.where(big, 0.0, vs - lin)
    return alpha, beta


def metrics(vs, reg):
    return {"ok_points": float(np.mean(np.median(vs, 0) <= reg + 1e-6)), "ok_all_prob": float(np.mean(vs.max(1) <= reg + 1e-6)),
            "worst_med": float(np.median(vs.max(1))), "worst_p95": float(np.percentile(vs.max(1), 95)), "n": int(len(vs))}


def main():
    t0 = time.time()
    site = r.load_site()
    MODEL = r.result_dir("model", create=False); r.check_version(MODEL)
    M = np.load(MODEL / "E4_members.npy")
    setup = json.load(open(OUT / "setup.json"))
    ti = lambda c: tuple(int(v) for v in c)
    groups = [dict(g, cells=[ti(c) for c in g["cells"]]) for g in setup["groups"]]
    cells = [ti(c) for c in setup["cells"]]
    pts = [ti(p) for p in setup["pts"]]
    wells = pd.read_csv(OUT / "wells_used.csv")
    cost = np.array([g["cost"] for g in groups])
    summ = json.load(open(OUT / "radius_summary.json"))
    prev = json.load(open(OUT / "refine.json")) if AQ_ONLY else None
    out = {}
    for tag, (name, window, future) in BLOCKS.items():
        z = np.load(OUT / f"R_{tag}.npz")
        R = [{"Rs": z["Rs"][i], "sub0": z["sub0"][i]} for i in range(z["Rs"].shape[0])]
        x_sq, shape, T = z["x_sq"], z["shape"], float(z["T"])
        reg = pbr.REG_MM_PER_YR * len(shape) / 12.0
        Mb = M if window else M[:pol.N_FWD]
        cap = np.maximum(np.array([g["cap"] for g in groups]), x_sq)
        n_mf = [0]

        def mf(x):
            n_mf[0] += 1
            vs = pol.verify(site, Mb, groups, cells, {"x": list(map(float, x))}, shape, window, pts, cells, future)[0]
            return vs

        # ---- AquaLimit: 선형 계획 → MODFLOW 확인 → 보정해 다시 풀기
        ones, zeros = np.ones((len(R), len(pts))), np.zeros((len(R), len(pts)))
        x, _ = solve_corr(R, x_sq, cost, cap, T, reg, ones, zeros)
        its = []
        for it in range(N_IT + 1):
            if x is None:
                its.append({"it": it, "feasible": False}); break
            vs = mf(x)
            c, v = pbr.cost_of(x, x_sq, cost, T)
            m = metrics(vs, reg); its.append({"it": it, "feasible": True, "cost": c, "replace_m3": v, "x": list(map(float, x)), **m})
            print(f"[{name}] AquaLimit 보정 {it}회: ${c / 1e6:.1f}M, MODFLOW 모든 지점 확률 {m['ok_all_prob'] * 100:.0f}% "
                  f"(최악 지점 {m['worst_med']:.1f} / 상위 5% {m['worst_p95']:.1f}mm, 기준 {reg:.1f})", flush=True)
            if m["ok_all_prob"] >= TARGET or it == N_IT:
                break
            alpha, beta = correction(R, x, x_sq, vs)
            x_new, _ = solve_corr(R, x_sq, cost, cap, T, reg - MARGIN_MM * (it + 1), alpha, beta)
            if x_new is None:                             # 비율 보정으로 풀 수 없으면 차이 더하기로
                lin = pbr.predict(R, x, x_sq)
                off = vs - lin if len(vs) == len(lin) else np.broadcast_to(np.max(vs, 0) - np.max(lin, 0), lin.shape)
                x_new, _ = solve_corr(R, x_sq, cost, cap, T, reg - MARGIN_MM * (it + 1), np.ones_like(lin), off)
            x = x_new
        aq = next((q for q in reversed(its) if q.get("feasible") and q["ok_all_prob"] >= TARGET), None)
        best = max((q for q in its if q.get("feasible")), key=lambda q: (q["ok_all_prob"], -q["cost"]), default=None)
        if AQ_ONLY:                                       # 금지 반경은 지난 실행 결과를 그대로 쓴다
            out[tag] = dict(prev[tag], aqualimit_iters=its, aqualimit=aq, aqualimit_best=best,
                            aqualimit_iters_additive=prev[tag].get("aqualimit_iters_additive", prev[tag]["aqualimit_iters"]),
                            n_modflow_plans=len(prev[tag]["ban_tested"]) + len(prev[tag].get("aqualimit_iters_additive", prev[tag]["aqualimit_iters"])) + n_mf[0])
            json.dump(dict(prev, **out), open(OUT / "refine.json", "w"), ensure_ascii=False, indent=1, default=float)
            continue

        # ---- 금지 반경: MODFLOW로 기준을 지키는 가장 작은 반경(0.1km)
        tested = {}

        def ban(rad):
            rad = round(float(rad), 1)
            if rad not in tested:
                frac = np.array([wells[(wells.group == g) & (wells.dist_km <= rad + 1e-9)].yield_gpm.sum() / max(wells[wells.group == g].yield_gpm.sum(), 1e-9)
                                 for g in range(len(groups))])
                xb = pbr.ban_x(x_sq, frac)
                c, v = pbr.cost_of(xb, x_sq, cost, T)
                m = metrics(mf(xb), reg)
                tested[rad] = {"radius_km": rad, "cost": c, "replace_m3": v, **m}
                print(f"[{name}] 금지 {rad:.1f}km: ${c / 1e6:.1f}M, MODFLOW 모든 지점 확률 {m['ok_all_prob'] * 100:.0f}%", flush=True)
            return tested[rad]["ok_all_prob"] >= TARGET

        r0 = round(summ[tag]["min_radius_km"], 1)       # 선형 응답으로 찾은 최소 반경에서 시작
        lo = hi = None                                    # lo: 기준 못 지킨 반경, hi: 지킨 반경
        if ban(r0):
            hi = r0
            while lo is None and hi - 0.4 >= 4.0:          # 0.4km씩 줄여 못 지키는 반경을 찾는다
                step = round(hi - 0.4, 1)
                if ban(step):
                    hi = step
                else:
                    lo = step
            lo = 4.0 if lo is None else lo
        else:
            lo = r0
            while hi is None and lo + 0.4 <= 10.0:         # 0.4km씩 넓혀 지키는 반경을 찾는다
                step = round(lo + 0.4, 1)
                if ban(step):
                    hi = step
                else:
                    lo = step
        while hi is not None and round(hi - lo, 1) > 0.1:  # 0.1km까지 이분 탐색
            mid = round((hi + lo) / 2, 1)
            if mid in (round(lo, 1), round(hi, 1)):
                break
            if ban(mid):
                hi = mid
            else:
                lo = mid
        bmin = tested.get(round(hi, 1)) if hi is not None else None
        out[tag] = {"name": name, "reg_mm": reg, "aqualimit_iters": its, "aqualimit": aq, "aqualimit_best": best, "ban_tested": sorted(tested.values(), key=lambda q: q["radius_km"]),
                    "ban_min": bmin, "n_modflow_plans": n_mf[0], "n_members": len(Mb)}
        json.dump(out, open(OUT / "refine.json", "w"), ensure_ascii=False, indent=1, default=float)

    # ---- 고정 금지 규칙: 2026 세 상황의 최소 반경 중 최댓값을 모든 해에
    fut = [t for t in out if t != "retro"]
    rads = [out[t]["ban_min"]["radius_km"] for t in fut if out[t]["ban_min"]]
    fixed = max(rads) if len(rads) == len(fut) else None
    for t in fut:
        if fixed is None:
            continue
        z = np.load(OUT / f"R_{t}.npz")
        frac = np.array([wells[(wells.group == g) & (wells.dist_km <= fixed + 1e-9)].yield_gpm.sum() / max(wells[wells.group == g].yield_gpm.sum(), 1e-9)
                         for g in range(len(groups))])
        c, v = pbr.cost_of(pbr.ban_x(z["x_sq"], frac), z["x_sq"], cost, float(z["T"]))
        out[t]["ban_fixed"] = {"radius_km": fixed, "cost": c, "replace_m3": v}
    out["_meta"] = {"fixed_radius_km": fixed, "minutes": round((time.time() - t0) / 60, 1), "variant": r.TAG}
    json.dump(out, open(OUT / "refine.json", "w"), ensure_ascii=False, indent=1, default=float)
    report(out); figure(out)
    print((OUT / "RESULTS_refine.md").read_text(encoding="utf-8"))


def figure(out):
    """같은 신뢰도에서 드는 비용: 금지 고정 규칙 / 그해 최소 반경 / AquaLimit (가로 막대, 상황별)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import run_real  # noqa: F401  글꼴
    except Exception:
        pass
    meta = out["_meta"]; fxr = meta.get("fixed_radius_km")
    tags = [t for t in ["습한 해", "평년", "가뭄·폭염", "retro"] if t in out and out[t].get("aqualimit")]
    fig, axs = plt.subplots(1, len(tags), figsize=(3.6 * len(tags), 3.0), squeeze=False)
    col = {"fixed": "#b9b6ad", "min": "#7d7a72", "aq": "#2a78d6"}
    for ax, t in zip(axs[0], tags):
        o = out[t]
        bars = []
        if o.get("ban_fixed"):
            bars.append((f"금지 고정 {fxr:.1f}km", o["ban_fixed"]["cost"], col["fixed"]))
        if o.get("ban_min"):
            bars.append((f"금지 그해 최소 {o['ban_min']['radius_km']:.1f}km", o["ban_min"]["cost"], col["min"]))
        bars.append(("AquaLimit", o["aqualimit"]["cost"], col["aq"]))
        ys = np.arange(len(bars))[::-1]
        for y, (lab, c, cl) in zip(ys, bars):
            ax.barh(y, c / 1e6, height=0.55, color=cl, edgecolor="white", linewidth=2)
            ax.text(c / 1e6 * 1.01, y, f"${c / 1e6:.1f}M", va="center", fontsize=8, color="#0b0b0b")
        ax.set_yticks(ys); ax.set_yticklabels([b[0] for b in bars], fontsize=8)
        ref = bars[0][1]
        sv = (ref - bars[-1][1]) / ref * 100
        ax.set_title(f"{o['name']}\nAquaLimit {sv:.0f}% 절약 ({'고정 규칙' if o.get('ban_fixed') else '최소 반경'} 대비)", fontsize=9)
        ax.set_xlim(0, max(b[1] for b in bars) / 1e6 * 1.3); ax.set_xlabel("비용 (백만 $)", fontsize=8)
        ax.tick_params(axis="x", labelsize=7); ax.grid(axis="x", color="#e6e4df", linewidth=0.6); ax.set_axisbelow(True)
    fig.suptitle("같은 신뢰도(MODFLOW 앙상블 모든 구성원·수로 38개 지점이 연 0.5인치 비례 기준 안)를 지키는 비용", fontsize=9.5)
    fig.tight_layout(); fig.savefig(OUT / "policy-4-refine.png", dpi=170); plt.close(fig)


def report(out):
    meta = out["_meta"]
    L = ["# MODFLOW로 고친 AquaLimit 계획 vs MODFLOW로 찾은 금지 반경 (같은 신뢰도)\n",
         "기준: 연 0.5인치(12.7mm)를 기간 길이로 나눈 값. **신뢰도 = MODFLOW 앙상블 모든 구성원에서 수로 38개 지점이 모두 기준 안**(과거 재현 30개, 다음 관개기 15개 구성원). "
         "AquaLimit은 선형 계획을 MODFLOW로 확인하고, 구성원·지점별로 'MODFLOW 침하 변화 ÷ 선형 예측 변화' 비율만큼 응답을 고쳐 다시 풀었다(순차 선형 계획, 최대 6회). "
         "처음 쓴 방식(차이를 더하기)은 4회 보정 뒤에도 평년·가뭄에서 15개 중 1개 구성원이 기준을 넘어(93%) 비율 보정으로 바꿨다(refine.json의 aqualimit_iters_additive). "
         "금지 반경은 MODFLOW로 직접 돌려 기준을 지키는 가장 작은 반경을 0.1km 단위로 찾았다.\n",
         "## 요약: 같은 신뢰도(MODFLOW 모든 지점 기준 달성)에 드는 비용 (백만 $)\n",
         f"| 상황 | 금지: 그해 최소 반경 | 금지: 고정 규칙 {meta['fixed_radius_km'] or '-'}km | **AquaLimit (MODFLOW 보정)** | 절약 (최소 반경 대비) | 절약 (고정 규칙 대비) |",
         "|---|---|---|---|---|---|"]
    for t, o in out.items():
        if t == "_meta":
            continue
        a, b, f = o.get("aqualimit"), o.get("ban_min"), o.get("ban_fixed")
        ac = a["cost"] if a else np.nan
        bst = o.get("aqualimit_best")
        a_txt = f"${ac / 1e6:.1f}M" if a else (f"${bst['cost'] / 1e6:.1f}M에서 {bst['ok_all_prob'] * 100:.0f}%" if bst else "풀 수 없음")
        bs = f"{b['radius_km']:.1f}km ${b['cost'] / 1e6:.1f}M" if b else "찾지 못함(10km까지)"
        fs = f"${f['cost'] / 1e6:.1f}M" if f else "-"
        s1 = f"${(b['cost'] - ac) / 1e6:.1f}M ({(b['cost'] - ac) / b['cost'] * 100:.0f}%)" if (a and b) else "-"
        s2 = f"${(f['cost'] - ac) / 1e6:.1f}M ({(f['cost'] - ac) / f['cost'] * 100:.0f}%)" if (a and f) else "-"
        L.append(f"| {o['name']} | {bs} | {fs} | **{a_txt}** | {s1} | {s2} |")
    L += ["", "- '그해 최소 반경'은 그해 날씨를 미리 알고 반경을 0.1km 단위로 가장 유리하게 고른 경우다. 현실의 금지 규칙은 해마다 바꾸기 어려워 '고정 규칙'에 가깝다.",
          "- 고정 규칙 비용은 반경이 그해 최소 반경 이상이라 기준을 지킨다(반경이 넓을수록 침하가 작다).", "", "![비용](policy-4-refine.png)", ""]
    for t, o in out.items():
        if t == "_meta":
            continue
        L += [f"## {o['name']} (기준 {o['reg_mm']:.1f}mm, MODFLOW 계획 {o['n_modflow_plans']}개 × 구성원 {o['n_members']}개)\n",
              "| 계획 | 비용 (백만 $) | 멈춘·대체 용수 (백만 m³) | MODFLOW 모든 지점 확률 | 지점별 중앙값 기준 안 | 최악 지점 (mm, 중앙값 / 상위 5%) |", "|---|---|---|---|---|---|"]
        for q in o["aqualimit_iters"]:
            if not q.get("feasible"):
                L.append(f"| AquaLimit 보정 {q['it']}회 | 풀 수 없음(관리 지역 우물을 모두 줄여도 부족) | | | | |"); continue
            L.append(f"| AquaLimit {'선형 계획' if q['it'] == 0 else '보정 %d회' % q['it']} | {q['cost'] / 1e6:.1f} | {q['replace_m3'] / 1e6:.1f} | "
                     f"{q['ok_all_prob'] * 100:.0f}% | {q['ok_points'] * 100:.0f}% | {q['worst_med']:.1f} / {q['worst_p95']:.1f} |")
        for q in o["ban_tested"]:
            L.append(f"| 금지 {q['radius_km']:.1f}km | {q['cost'] / 1e6:.1f} | {q['replace_m3'] / 1e6:.1f} | {q['ok_all_prob'] * 100:.0f}% | "
                     f"{q['ok_points'] * 100:.0f}% | {q['worst_med']:.1f} / {q['worst_p95']:.1f} |")
        L.append("")
    L.append(f"계산 시간 {meta['minutes']}분.")
    (OUT / "RESULTS_refine.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if "--report" in sys.argv:                   # 계산 없이 refine.json으로 표·그림만 다시
        o = json.load(open(OUT / "refine.json")); report(o); figure(o)
    else:
        main()
