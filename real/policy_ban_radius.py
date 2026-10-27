"""금지 반경을 넓히는 방식 vs AquaLimit 우물별 배분: 같은 기준을 지키는 데 드는 비용 비교.

policy_real.py가 저장한 응답(results_policy_<버전>/R_*.npz: 앙상블 구성원별 '우물 묶음 취수 → 수로 지점 침하' 선형 응답)을 그대로 쓴다.
MODFLOW를 다시 돌리지 않으므로 몇 초 걸린다. 선형 응답의 정확도는 policy_real.py가 MODFLOW로 확인했다(RESULTS.md: 예측 vs MODFLOW).

- 금지 반경 R km: 우물 묶음마다 '수로에서 R km 안에 있는 우물의 양수량 비율'만큼 취수를 멈추고 대체 용수로 바꾼다(2025 실제 정책 = 4.02km).
- AquaLimit: 같은 기준(모든 앙상블 구성원·모든 수로 지점에서 연 0.5인치 비례 한도)을 최소 비용으로 지키는 우물 묶음별 취수(선형계획).
실행: AQ_VARIANT=v3ops python3 policy_ban_radius.py → results_policy_<버전>/RESULTS_radius.md, radius.json, policy-3-radius.png
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
TAG = os.environ.get("AQ_VARIANT", "v3ops") + ("-opera" if os.environ.get("AQ_INSAR") == "opera" else "")
POL = HERE / f"results_policy_{TAG}"
PRICE = 0.81
REG_MM_PER_YR = 12.7
RADII = [4.02, 5.0, 6.0, 7.0, 8.0, 10.0]
BLOCKS = {"retro": "2021.4~2022.9 대가뭄 재현", "습한 해": "2026.5~7 습한 해", "평년": "2026.5~7 평년", "가뭄·폭염": "2026.5~7 가뭄·폭염"}


def load(tag):
    z = np.load(POL / f"R_{tag}.npz")
    R = [{"Rs": z["Rs"][i], "sub0": z["sub0"][i]} for i in range(z["Rs"].shape[0])]
    return R, z["x_sq"], float(z["T"]), len(z["shape"])


def predict(R, x, x_sq):
    return np.array([Ri["sub0"] + Ri["Rs"] @ (np.asarray(x) - x_sq) for Ri in R])      # (구성원, 지점)


def cost_of(x, x_sq, cost, T):
    v = T * (x_sq.sum() - np.sum(x))
    return float((cost * x).sum() * T + PRICE * v), float(v)


def ban_x(x_sq, frac_in):
    return x_sq * (1 - frac_in)


def solve_reg(R, x_sq, cost, cap, T, reg):
    G = len(x_sq)
    c = np.r_[cost * T, PRICE]
    A, b = [], []
    for Ri in R:
        for p in range(Ri["Rs"].shape[0]):
            A.append(np.r_[Ri["Rs"][p], 0.0]); b.append(reg - Ri["sub0"][p] + Ri["Rs"][p] @ x_sq)
    res = linprog(c, A_ub=np.array(A), b_ub=b, A_eq=np.r_[np.full(G, T), 1.0][None], b_eq=[T * x_sq.sum()],
                  bounds=[(0, cg) for cg in cap] + [(0, None)], method="highs")
    return (res.x[:G], float(res.fun), float(res.x[G])) if res.status == 0 else (None, np.nan, np.nan)


def metrics(sub, reg):
    return {"ok_points": float(np.mean(np.median(sub, 0) <= reg + 1e-6)),   # 지점별 중앙값이 기준 안
            "ok_all_prob": float(np.mean(sub.max(1) <= reg + 1e-6)),        # 모든 지점이 기준 안일 확률(구성원 비율)
            "worst_med": float(np.median(sub.max(1)))}


def main():
    setup = json.load(open(POL / "setup.json"))
    groups = setup["groups"]
    wells = pd.read_csv(POL / "wells_used.csv")
    cost = np.array([g["cost"] for g in groups])
    out = {}
    for tag, name in BLOCKS.items():
        if not (POL / f"R_{tag}.npz").exists():
            continue
        R, x_sq, T, n_mon = load(tag)
        reg = REG_MM_PER_YR * n_mon / 12.0
        cap = np.maximum(np.array([g["cap"] for g in groups]), x_sq)
        rows = []
        for rad in RADII:
            w = wells.assign(q=wells.yield_gpm)
            frac = np.array([w[(w.group == g) & (w.dist_km <= rad)].q.sum() / max(w[w.group == g].q.sum(), 1e-9) for g in range(len(groups))])
            x = ban_x(x_sq, frac)
            c, v = cost_of(x, x_sq, cost, T)
            m = metrics(predict(R, x, x_sq), reg)
            rows.append({"policy": f"금지 {rad:.0f}km" if rad != 4.02 else "금지 4km(실제)", "radius_km": rad, "cost": c, "replace_m3": v,
                         "banned_share": float(1 - x.sum() / x_sq.sum()), **m})
        # 같은 기준(모든 구성원·지점)을 지키는 가장 작은 금지 반경을 0.1km 단위로 찾는다(금지 쪽에 가장 유리하게)
        for rad in np.arange(4.1, 10.01, 0.1):
            w = wells.assign(q=wells.yield_gpm)
            frac = np.array([w[(w.group == g) & (w.dist_km <= rad)].q.sum() / max(w[w.group == g].q.sum(), 1e-9) for g in range(len(groups))])
            x = ban_x(x_sq, frac)
            m = metrics(predict(R, x, x_sq), reg)
            if m["ok_all_prob"] >= 0.999:
                c, v = cost_of(x, x_sq, cost, T)
                rows.append({"policy": f"금지 {rad:.1f}km(기준 맞추는 최소 반경)", "radius_km": round(float(rad), 1), "cost": c, "replace_m3": v,
                             "banned_share": float(1 - x.sum() / x_sq.sum()), **m, "min_radius": True})
                break
        xa, ca, va = solve_reg(R, x_sq, cost, cap, T, reg)
        if xa is not None:
            m = metrics(predict(R, xa, x_sq), reg)
            rows.append({"policy": "AquaLimit 규제 기준", "radius_km": np.nan, "cost": ca, "replace_m3": va,
                         "banned_share": float(1 - xa.sum() / x_sq.sum()), **m, "x": xa.tolist()})
        df = pd.DataFrame(rows)
        # 같은 기준(모든 구성원·모든 지점)을 지키는 가장 작은 금지 반경
        full = df[(df.radius_km.notna()) & (df.ok_all_prob >= 0.999)].sort_values("cost")
        best_ban = full.iloc[0].to_dict() if len(full) else None
        aq = df[df.policy == "AquaLimit 규제 기준"].iloc[0].to_dict() if (df.policy == "AquaLimit 규제 기준").any() else None
        out[tag] = {"name": name, "reg_mm": reg, "months": n_mon, "rows": df.drop(columns=["x"], errors="ignore").to_dict("records"),
                    "min_ban_meeting_reg": best_ban, "aqualimit": aq,
                    "saving_vs_ban": (best_ban["cost"] - aq["cost"]) if (best_ban and aq) else None}
    json.dump(out, open(POL / "radius.json", "w"), ensure_ascii=False, indent=1, default=float)
    figure(out); report(out)
    print((POL / "RESULTS_radius.md").read_text(encoding="utf-8"))


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import run_real  # noqa: F401  글꼴
    except Exception:
        pass
    tags = [t for t in out if t != "retro"] + (["retro"] if "retro" in out else [])
    fig, axs = plt.subplots(1, len(tags), figsize=(4.2 * len(tags), 3.8), squeeze=False)
    for ax, t in zip(axs[0], tags):
        df = pd.DataFrame(out[t]["rows"])
        b = df[df.radius_km.notna() & df.get("min_radius", pd.Series(False, index=df.index)).isna()].sort_values("radius_km") \
            if "min_radius" in df else df[df.radius_km.notna()].sort_values("radius_km")
        ax.plot(b.cost / 1e6, b.ok_all_prob * 100, "o-", color="#d6604d", label="금지 반경 넓히기")
        for _, r_ in b.iterrows():
            ax.annotate(f"{r_.radius_km:.0f}km", (r_.cost / 1e6, r_.ok_all_prob * 100), fontsize=7, xytext=(2, -10), textcoords="offset points")
        a = df[df.policy == "AquaLimit 규제 기준"]
        if len(a):
            ax.plot(a.cost / 1e6, a.ok_all_prob * 100, "*", ms=15, color="#2166ac", label="AquaLimit (같은 기준)")
            ax.annotate(f"${a.cost.iloc[0] / 1e6:.1f}M", (a.cost.iloc[0] / 1e6, 100), fontsize=8, color="#2166ac", xytext=(-10, 8), textcoords="offset points")
        ax.set_title(out[t]["name"], fontsize=10); ax.set_xlabel("비용 (백만 $)"); ax.set_ylim(0, 112)
    axs[0][0].set_ylabel("수로 38개 지점이 모두 기준 안일 확률 (%)"); axs[0][0].legend(fontsize=7, loc="lower right")
    fig.suptitle("같은 수로 기준을 지키는 데 드는 비용: 금지 반경을 넓히는 방식 vs AquaLimit 우물별 배분 (선형 응답, 앙상블)", fontsize=10)
    fig.tight_layout(); fig.savefig(POL / "policy-3-radius.png", dpi=150); plt.close(fig)


def report(out):
    f = lambda x: "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.1f}"
    L = ["# 금지 반경 넓히기 vs AquaLimit (같은 기준, 비용 비교)\n",
         "policy_real.py의 앙상블 선형 응답(구성원별 '우물 묶음 취수 → 수로 38개 지점 침하')으로 계산했다. 기준: 연 0.5인치(12.7mm)를 기간 길이로 나눈 값. "
         "'모든 지점 기준 확률' = 앙상블 구성원 중 수로 38개 지점이 모두 기준 안인 비율. AquaLimit 계획은 모든 구성원·모든 지점에서 기준을 지키게 만든 것이라 이 값이 100%다.\n",
         "**주의:** 선형 응답 값이다. policy_real.py의 MODFLOW 확인에서 'AquaLimit 규제 기준' 계획의 모든 지점 기준 확률은 평년 60%, 가뭄 60%, 습한 해 80%, 대가뭄 97%로 선형 예측(100%)보다 낮았다(비선형). "
         "금지 반경 안들도 같은 방향으로 낮아질 것이므로 비교(누가 더 싼가)는 유지되지만, 금지 확대 안의 MODFLOW 확인은 맥에서 policy_real.py에 추가해 돌려야 한다.\n"]
    # 요약: (1) 해마다 가장 작은 반경을 알고 고른 금지(금지 쪽에 가장 유리) (2) 가뭄 해에도 기준을 지키게 고정한 반경(현실적 규칙)
    fixed = None
    if "가뭄·폭염" in out and out["가뭄·폭염"]["min_ban_meeting_reg"]:
        fixed = out["가뭄·폭염"]["min_ban_meeting_reg"]["radius_km"]
    L += ["## 요약: 모든 수로 지점에서 기준을 지키는 데 드는 비용 (백만 $)\n",
          f"| 상황 | 금지: 그 해에 맞는 최소 반경 | 금지: 고정 반경 {fixed:.1f}km(가뭄에도 지키는 규칙) | **AquaLimit** | 절약 (최소 반경 대비) | 절약 (고정 규칙 대비) |" if fixed else
          "| 상황 | 금지: 최소 반경 | - | AquaLimit | 절약 | - |", "|---|---|---|---|---|---|"]
    summ = {}
    for t, o in out.items():
        b, a = o["min_ban_meeting_reg"], o["aqualimit"]
        if not (b and a):
            continue
        fx = None
        if fixed:
            wells_fx = [r_ for r_ in o["rows"] if abs(r_["radius_km"] - fixed) < 1e-6] if True else []
            fx = wells_fx[0]["cost"] if wells_fx else None
        summ[t] = {"min_radius_km": b["radius_km"], "min_ban_cost": b["cost"], "fixed_radius_km": fixed, "fixed_ban_cost": fx, "aqualimit_cost": a["cost"]}
        L.append(f"| {o['name']} | {b['radius_km']:.1f}km ${b['cost'] / 1e6:.1f}M | " + (f"${fx / 1e6:.1f}M" if fx else "-") +
                 f" | **${a['cost'] / 1e6:.1f}M** | ${(b['cost'] - a['cost']) / 1e6:.1f}M ({(b['cost'] - a['cost']) / b['cost'] * 100:.0f}%) | " +
                 (f"${(fx - a['cost']) / 1e6:.1f}M ({(fx - a['cost']) / fx * 100:.0f}%)" if fx else "-") + " |")
    L += ["", "- 금지 반경은 한 번 정하면 해마다 바꾸기 어렵다. 가뭄 해에도 기준을 지키려면 반경을 가뭄에 맞춰야 하고, 그러면 평년·습한 해에는 필요 이상으로 멈춘다.",
          "- AquaLimit은 그해 날씨·수로 배분율에 맞춰 우물별로 줄일 양을 다시 계산한다. 같은 기준을 지키면서 위 표만큼 덜 쓴다.",
          "- '최소 반경' 열은 그해 날씨를 미리 알고 반경을 0.1km 단위로 가장 유리하게 고른 경우다(현실에서는 모델 없이 고를 수 없다).\n"]
    json.dump(summ, open(POL / "radius_summary.json", "w"), ensure_ascii=False, indent=1, default=float)
    for t, o in out.items():
        L += [f"## {o['name']} (기준 {o['reg_mm']:.1f}mm/{o['months']}개월)\n",
              "| 정책 | 멈춘 취수 비율 | 대체 용수 (백만 m³) | 비용 (백만 $) | 기준 지키는 지점 | 모든 지점 기준 확률 | 최악 지점 침하 (mm, 중앙값) |", "|---|---|---|---|---|---|---|"]
        for r_ in o["rows"]:
            L.append(f"| {r_['policy']} | {r_['banned_share'] * 100:.0f}% | {r_['replace_m3'] / 1e6:.1f} | {r_['cost'] / 1e6:.1f} | {r_['ok_points'] * 100:.0f}% | "
                     f"{r_['ok_all_prob'] * 100:.0f}% | {f(r_['worst_med'])} |")
        b, a = o["min_ban_meeting_reg"], o["aqualimit"]
        if b and a:
            L.append(f"\n- 기준을 모든 지점에서 지키는 가장 작은 금지: **{b['policy']}**, ${b['cost'] / 1e6:.1f}M. AquaLimit: **${a['cost'] / 1e6:.1f}M** → "
                     f"**${(b['cost'] - a['cost']) / 1e6:.1f}M 절약({(b['cost'] - a['cost']) / b['cost'] * 100:.0f}%)**.\n")
        elif a:
            L.append(f"\n- 금지 반경을 10km(관리 지역 전체)까지 넓혀도 모든 지점 기준을 지키지 못한다. AquaLimit은 ${a['cost'] / 1e6:.1f}M로 지킨다"
                     "(관리 지역 우물을 다 멈추는 대신, 영향이 작은 우물은 오히려 더 뽑게 하고 큰 우물을 줄인다).\n")
    L.append("![반경](policy-3-radius.png)")
    (POL / "RESULTS_radius.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
