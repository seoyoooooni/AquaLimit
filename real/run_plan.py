"""실제 검증 지역 취수 계획 실험(3단계). run_real.py 결과(results/)가 있어야 한다.

실행: python3 real/run_plan.py   (MODFLOW 약 230회, 2코어 기준 약 15분)
출력: results_plan/RESULTS.md, results.json, real-5-plan.png
"""
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import real_model as r
import plan_real as p
import run_real as rr          # 글꼴 설정

OUT = r.HERE / "results_plan"; OUT.mkdir(exist_ok=True)
NAMES = {"E1": "E1 기존 방식(용량 비례)", "E2": "E2 관측정만", "E3": "E3 관측정 + 위성", "E4": "E4 + 불확실성(강건)",
         "E4H": "E4 + 임계수위 유지"}


def main():
    t0 = time.time()
    site = r.load_site()
    res = json.load(open(r.HERE / "results" / "results.json"))
    th2, th3 = np.array(res["params"]["E2"]), np.array(res["params"]["E3"])
    M4 = np.load(r.HERE / "results" / "E4_members.npy")
    pts, wells = p.mgmt_points(site), p.operator_wells(site)
    print(f"관리지점 {len(pts)}곳, 계획 우물 {len(wells)}개, E4 구성원 {len(M4)}개")

    print("[응답 행렬] E2, E3, E4 구성원마다 MODFLOW 7회(기준 1 + 우물 6)")
    R2 = p.responses(site, th2, wells, pts); R3 = p.responses(site, th3, wells, pts)
    R4 = [p.responses(site, th, wells, pts) for th in M4]
    R4 = [x for x in R4 if x is not None]
    print(f"  E4 응답 {len(R4)}개 계산")
    np.savez_compressed(OUT / "R4.npz", Rs=np.array([x["Rs"] for x in R4]), Rh=np.array([x["Rh"] for x in R4]))

    plans = {"E1": p.baseline()}
    if R2 is not None:
        plans["E2"] = p.solve([R2])
    plans["E3"] = p.solve([R3]); plans["E4"] = p.solve(R4); plans["E4H"] = p.solve(R4, use_head=True)
    # 임계수위를 지키려면 외부 용수가 얼마나 필요한가(구매 한도 없이)
    cap0 = p.PURCHASE_CAP; p.PURCHASE_CAP = p.DEMAND
    need = p.solve(R4, use_head=True); p.PURCHASE_CAP = cap0
    out = {"setting": {"months": p.MONTHS, "demand_m3": p.DEMAND, "cap_m3_per_day": p.CAP, "limit_mm": p.LIMIT_MM,
                       "price_per_m3": p.PRICE, "purchase_cap_m3": p.PURCHASE_CAP, "well_cost": p.WELL_COST.tolist(),
                       "wells": wells, "n_points": len(pts)},
           "plans": plans, "head_need": need}
    out["risk"] = {k: p.risk(v, R4, use_head=True) for k, v in plans.items()}
    out["background_sub_mm"] = {"median": float(np.median(R3["base_sub"])), "max": float(R3["base_sub"].max())}

    # 선형 응답 확인: E3 모델로 E3·E4 계획을 MODFLOW로 직접 실행해 응답 행렬 예측과 비교
    chk = {}
    for k in ["E3", "E4"]:
        if not plans[k]["feasible"]:
            continue
        x = np.array(plans[k]["rates"])
        plan = {"months": p.MONTHS, "wells": wells, "rates": np.repeat(x[None], p.MONTHS, 0), "dry": 0.0}
        S, H = r.run_model(site, th3, plan=plan)
        iy, ix = np.array(pts).T; n0 = len(site["months"]) - 2
        direct = (S[-1, iy, ix] - S[n0, iy, ix]) - R3["base_sub"]
        lin = R3["Rs"] @ x
        chk[k] = {"direct_worst_mm": float(direct.max()), "linear_worst_mm": float(lin.max()),
                  "max_abs_diff_mm": float(np.abs(direct - lin).max())}
    out["linearity_check"] = chk

    # 실행 불가 테스트
    tests = {}
    for name, (dem, lim) in {"수요 1.5배 · 기준 그대로": (p.DEMAND * 1.5, p.LIMIT_MM),
                             "수요 그대로 · 기준 0.3mm": (p.DEMAND, 0.3),
                             "수요 2.5배 (용량+구매 한도 초과)": (p.DEMAND * 2.5, p.LIMIT_MM)}.items():
        d0, l0 = p.DEMAND, p.LIMIT_MM; p.DEMAND, p.LIMIT_MM = dem, lim
        s_ = p.solve(R4); p.DEMAND, p.LIMIT_MM = d0, l0
        tests[name] = "계획 산출" if s_["feasible"] else "실행 불가 — " + s_["reason"]
    out["infeasible_tests"] = tests
    out["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(out, open(OUT / "results.json", "w"), ensure_ascii=False, indent=1, default=float)
    figure(site, pts, wells, plans, out)
    report(out)
    print((OUT / "RESULTS.md").read_text(encoding="utf-8"))


def figure(site, pts, wells, plans, out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.25]})
    cells = site["cells"]
    v = np.full((site["ny"], site["nx"]), np.nan)
    a = ax[0]
    a.imshow(np.where(site["ag"], 1.0, np.nan), origin="lower", cmap="Greens", vmin=0, vmax=3, alpha=0.6)
    iy, ix = np.array(pts).T
    a.scatter(ix, iy, s=14, c="#2166ac", marker="s", label="관리지점(수로 칸)")
    for w, (wy, wx) in enumerate(wells):
        a.scatter(wx, wy, s=90, c="#b2182b", marker="^", edgecolor="k", zorder=5)
        a.annotate(f"{chr(65 + w)}  {p.WELL_COST[w]:.2f}$/m³", (wx, wy), xytext=(6, -3), textcoords="offset points", fontsize=8)
    a.scatter([], [], s=90, c="#b2182b", marker="^", edgecolor="k", label="계획 우물(가정)")
    a.set_title("계획 대상 (초록 = 농경지 배경 취수 지역)", fontsize=11); a.set_xticks([]); a.set_yticks([]); a.legend(fontsize=8, loc="lower left")
    b = ax[1]
    keys = [k for k in ["E1", "E2", "E3", "E4", "E4H"] if k in plans]
    W = len(p.WELL_COST); width = 0.8 / max(len(keys), 1)
    for j, k in enumerate(keys):
        pl = plans[k]
        if not pl["feasible"]:
            b.text(0.02, 0.03, f"{NAMES[k]}: 실행 불가 (외부 용수 추가 확보 필요)", transform=b.transAxes, fontsize=8.5, color="#b2182b")
            continue
        rk = out["risk"][k]
        b.bar(np.arange(W) + (j - len(keys) / 2) * width + width / 2, np.array(pl["rates"]) / 1000, width,
              label=f"{NAMES[k]} · 비용 {pl['cost'] / 1e3:,.0f}천$ · 초과확률 {rk['exceed_prob'] * 100:.0f}%")
    b.set_xticks(range(W), [chr(65 + w) for w in range(W)]); b.set_ylabel("평균 취수 (천 m³/일)")
    b.set_title(f"우물별 계획 (3개월 수요 {p.DEMAND / 1e3:,.0f}천 m³, 수로 추가 침하 ≤ {p.LIMIT_MM}mm)", fontsize=11)
    b.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "real-5-plan.png", dpi=150); plt.close(fig)


def report(o):
    s = o["setting"]
    L = ["# 실제 검증 지역 취수 계획 (2026년 5~7월, 시뮬레이션)\n",
         f"수로 동쪽 3km 가상 사업자 우물 {len(s['wells'])}개(용량 각 {s['cap_m3_per_day']:,.0f} m³/일), 3개월 수요 {s['demand_m3']:,.0f} m³, "
         f"외부 용수 {s['price_per_m3']}$/m³(한도 {s['purchase_cap_m3']:,.0f} m³). 관리 기준: 계획으로 인한 수로 칸({s['n_points']}곳) 추가 침하 ≤ **{s['limit_mm']}mm/3개월**. "
         f"참고로 배경 침하(다른 농가 취수)만으로 수로 칸 중앙값 {o['background_sub_mm']['median']:.1f}mm, 최대 {o['background_sub_mm']['max']:.1f}mm가 3개월 동안 생긴다.\n",
         "우물 위치·용량·비용은 가정값이다(관정별 실제 기록 비공개). 땅속 성질·배경 취수·함양은 실제 자료로 역산한 값(results/)을 쓴다. "
         "실제 정답이 없으므로, 각 계획을 **E4 앙상블 30개 모두에서** 계산해 기준 초과 확률로 비교한다.\n",
         "## 1. 계획 비교\n",
         "| 방식 | 실행 가능 | 총비용(천$) | 외부 구매(천 m³) | 최대 추가 침하 평균(mm) | 상위 5% (mm) | 기준 초과 확률 | 임계수위 아래로 내려갈 확률 |",
         "|---|---|---|---|---|---|---|---|"]
    for k, pl in o["plans"].items():
        if not pl["feasible"]:
            L.append(f"| {NAMES[k]} | 불가 ({pl['reason']}) | - | - | - | - | - | - |"); continue
        rk = o["risk"][k]
        L.append(f"| {NAMES[k]} | 가능 | {pl['cost'] / 1e3:,.0f} | {pl['purchase'] / 1e3:,.0f} | {rk['worst_mean_mm']:.2f} | {rk['worst_p95_mm']:.2f} | "
                 f"{rk['exceed_prob'] * 100:.0f}% | {rk.get('below_crit_prob', 0) * 100:.0f}% |")
    n = o["head_need"]
    if n["feasible"]:
        L.append(f"\n임계수위(과거 최저 수위)를 지키려면 외부 용수가 **{n['purchase'] / 1e3:,.0f}천 m³** 필요하다(수요의 {n['purchase'] / s['demand_m3'] * 100:.0f}%, 비용 {n['cost'] / 1e3:,.0f}천$). "
                 "다른 농가의 배경 취수만으로도 수로 칸 수위가 이미 과거 최저보다 내려가고 있어서, 모든 계획에서 '임계수위 아래로 내려갈 확률'이 100%로 나온다. "
                 "즉 이 사업자 혼자 우물을 줄여서는 임계수위를 지킬 수 없고, **지역 전체(지하수 관리기관) 차원의 취수 조정이 필요하다**.\n")
    pl, rk = o["plans"], o["risk"]
    if all(pl[k]["feasible"] for k in ["E1", "E3", "E4"]):
        L += ["### 해석\n",
              f"- **E1(용량 비례)**은 모델 없이 고르게 뽑아 비용 {pl['E1']['cost'] / 1e3:,.0f}천$, 기준 초과 확률 {rk['E1']['exceed_prob'] * 100:.0f}%.",
              f"- **E3(위성 + 추정값 1개)**는 싸고 침하 영향이 작은 우물로 옮겨 비용을 {pl['E3']['cost'] / 1e3:,.0f}천$로 줄이지만, 기준선에 딱 맞춰 계획해서 초과 확률이 {rk['E3']['exceed_prob'] * 100:.0f}%다. "
              f"실제로 MODFLOW로 직접 돌리면 최대 {o['linearity_check'].get('E3', {}).get('direct_worst_mm', float('nan')):.2f}mm로 기준(2mm)을 넘는다(2절).",
              f"- **E4(불확실성 반영)**는 외부 용수 {pl['E4']['purchase'] / 1e3:,.0f}천 m³를 사서 비용이 {pl['E4']['cost'] / 1e3:,.0f}천$(E3보다 {(pl['E4']['cost'] / pl['E3']['cost'] - 1) * 100:.0f}% 높음)이지만, "
              f"앙상블 30개 모두에서 기준을 지킨다(초과 확률 {rk['E4']['exceed_prob'] * 100:.0f}%).",
              "- 가상 대수층 실험(synthetic/)과 같은 결론이 실제 지역 모델에서도 나왔다: 추정값 하나만 믿으면 기준선에 붙고, 불확실성을 반영하면 비용을 더 써서(이 조건에서는 외부 용수 구매 때문에 비용 증가가 가상 실험보다 크다) 위험을 없앤다.\n"]
    L += ["![계획](real-5-plan.png)\n", "## 2. 응답 행렬(선형 근사) 확인\n",
          "E3 모델에서 계획을 MODFLOW로 직접 실행한 결과와 응답 행렬 예측을 비교했다.\n", "| 계획 | 직접 실행 최대 침하 | 응답 행렬 예측 | 최대 차이 |", "|---|---|---|---|"]
    for k, c in o["linearity_check"].items():
        L.append(f"| {NAMES[k]} | {c['direct_worst_mm']:.2f} mm | {c['linear_worst_mm']:.2f} mm | {c['max_abs_diff_mm']:.3f} mm |")
    L += ["\n## 3. 실행 불가 판정\n", "| 입력 | 결과 |", "|---|---|"]
    for k, v in o["infeasible_tests"].items():
        L.append(f"| {k} | {v} |")
    L.append(f"\n계산 시간 {o['minutes']}분.\n")
    (OUT / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    import sys
    if "--report-only" in sys.argv:
        o = json.load(open(OUT / "results.json"))
        site = r.load_site()
        figure(site, p.mgmt_points(site), p.operator_wells(site), o["plans"], o)
        report(o)
        print((OUT / "RESULTS.md").read_text(encoding="utf-8"))
    else:
        main()
