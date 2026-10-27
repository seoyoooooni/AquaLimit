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

OUT = r.result_dir("plan")
MODEL = r.result_dir("model", create=False)
NAMES = {"E1": "E1 기존 방식(용량 비례)", "E2": "E2 관측정만", "E3": "E3 관측정 + 위성", "E4": "E4 + 불확실성(강건)",
         "E4S": "E4 + 구조 오차 여유", "E4H": "E4 + 임계수위 유지"}
MARGINS = [0.0, 0.25, 0.5, 1.0]          # 구조 오차 여유: 실제 땅이 모델보다 이만큼 더 반응해도 기준을 지키게


def structural_margin():
    """응답 구조 오차 크기: 자연 실험(validate_response.py)에서 GPS 속도 변화 / 모델 속도 변화의 비율로 잰다. 없으면 0.25."""
    f = r.HERE / f"results_validate_{r.TAG.replace('ops', '')}" / "results.json"
    if not f.exists():
        f = r.HERE / f"results_validate_{r.VARIANT.replace('ops', '')}" / "results.json"
    if f.exists():
        ch = json.load(open(f))["changes"]
        ratio = [c["gps_change"] / c["model_change_med"] for c in ch
                 if c["same_sign"] and abs(c["model_change_med"]) > 1.0]
        if ratio:
            return float(np.clip(np.median(ratio) - 1.0, 0.1, 1.0)), "자연 실험(GPS 속도 변화 / 모델 속도 변화) 중앙값"
    return 0.25, "기본값(자연 실험 결과 없음)"


def main():
    t0 = time.time()
    site = r.load_site()
    r.check_version(MODEL); r.check_version(OUT); r.write_version(OUT)
    res = json.load(open(MODEL / "results.json"))
    th3 = np.array(res["params"]["E3"])
    if "E2" in res["params"]:
        th2 = np.array(res["params"]["E2"])
    else:                                              # 운영 모델에는 E2가 없다 → 검증 모델(v3)의 '관측정만' 보정 결과를 쓴다
        v = r.HERE / "results_v3" / "results.json"
        pv = json.load(open(v))["params"] if v.exists() else None
        if pv:                                         # 이름으로 맞추고, 운영 모델에만 있는 값(금지 후 취수 배수)은 사전 평균
            m2 = dict(zip(pv["names"], pv["E2"]))
            th2 = np.array([m2.get(n, mu) for n, mu in zip(r.NAMES, r.PRIOR_MEAN)])
        else:
            th2 = None
    M4 = np.load(MODEL / "E4_members.npy")
    pts, wells = p.mgmt_points(site), p.operator_wells(site)
    print(f"관리지점 {len(pts)}곳, 계획 우물 {len(wells)}개, E4 구성원 {len(M4)}개")

    print("[응답 행렬] E2, E3, E4 구성원마다 MODFLOW 7회(기준 1 + 우물 6)")
    R2 = p.responses(site, th2, wells, pts) if th2 is not None else None; R3 = p.responses(site, th3, wells, pts)
    R4 = [p.responses(site, th, wells, pts) for th in M4]
    R4 = [x for x in R4 if x is not None]
    print(f"  E4 응답 {len(R4)}개 계산")
    np.savez_compressed(OUT / "R4.npz", Rs=np.array([x["Rs"] for x in R4]), Rh=np.array([x["Rh"] for x in R4]))

    plans = {"E1": p.baseline()}
    if R2 is not None:
        plans["E2"] = p.solve([R2])
    plans["E3"] = p.solve([R3]); plans["E4"] = p.solve(R4)
    delta, delta_src = structural_margin()
    plans["E4S"] = p.solve(R4, inflate=delta); plans["E4H"] = p.solve(R4, use_head=True)
    margin_curve = []
    pc0 = p.PURCHASE_CAP; p.PURCHASE_CAP = p.DEMAND     # 여유 곡선은 외부 구매 한도 없이(여유를 키우면 구매가 늘어난다)
    for d in MARGINS:                                   # 여유를 얼마나 둘지 = 비용
        pl = p.solve(R4, inflate=d)
        row = {"margin": d, "feasible": pl["feasible"], "cost": pl.get("cost"), "purchase": pl.get("purchase")}
        if pl["feasible"]:
            for f in [1.0, 1.25, 1.5]:
                row[f"exceed_x{f}"] = float(np.mean([(f * (R_["Rs"] @ np.array(pl["rates"]))).max() > p.LIMIT_MM + 1e-9 for R_ in R4]))
        margin_curve.append(row)
    p.PURCHASE_CAP = pc0
    if not plans["E4S"]["feasible"]:                     # 구매 한도 안에서 불가능하면 한도 없이 다시(외부 구매로 채움)
        p.PURCHASE_CAP = p.DEMAND; plans["E4S"] = p.solve(R4, inflate=delta); p.PURCHASE_CAP = pc0
    # 임계수위를 지키려면 외부 용수가 얼마나 필요한가(구매 한도 없이)
    cap0 = p.PURCHASE_CAP; p.PURCHASE_CAP = p.DEMAND
    need = p.solve(R4, use_head=True); p.PURCHASE_CAP = cap0
    out = {"setting": {"months": p.MONTHS, "demand_m3": p.DEMAND, "cap_m3_per_day": p.CAP.tolist(), "limit_mm": p.LIMIT_MM,
                       "well_info": p.WELL_INFO, "variant": r.VARIANT,
                       "price_per_m3": p.PRICE, "purchase_cap_m3": p.PURCHASE_CAP, "well_cost": p.WELL_COST.tolist(),
                       "wells": wells, "n_points": len(pts)},
           "plans": plans, "head_need": need, "margin": {"delta": delta, "source": delta_src, "curve": margin_curve}}
    out["risk"] = {k: p.risk(v, R4, use_head=True) for k, v in plans.items()}
    out["background_sub_mm"] = {"median": float(np.median(R3["base_sub"])), "max": float(R3["base_sub"].max())}

    # 선형 응답 확인: E3 모델로 E3·E4 계획을 MODFLOW로 직접 실행해 응답 행렬 예측과 비교
    chk = {}
    for k in ["E3", "E4", "E4S"]:
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
    a.scatter([], [], s=90, c="#b2182b", marker="^", edgecolor="k", label="사업자 우물(실제 신고 우물)" if r.V3 else "계획 우물(가정)")
    a.set_title("계획 대상 (초록 = 농경지 배경 취수 지역)", fontsize=11); a.set_xticks([]); a.set_yticks([]); a.legend(fontsize=8, loc="lower left")
    b = ax[1]
    keys = [k for k in ["E1", "E2", "E3", "E4", "E4S", "E4H"] if k in plans]
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
         (f"사업자 우물 {len(s['wells'])}개 = DWR 신고 농업용 우물 중 취수 금지구역(수로 양쪽 2.5마일) 밖 실제 우물(용량 합 {np.sum(s['cap_m3_per_day']):,.0f} m³/일), "
          if s.get("well_info") else f"수로 동쪽 3km 가상 사업자 우물 {len(s['wells'])}개(용량 각 {np.mean(s['cap_m3_per_day']):,.0f} m³/일), ")
         + f"3개월 수요 {s['demand_m3']:,.0f} m³, "
         f"외부 용수 {s['price_per_m3']}$/m³(한도 {s['purchase_cap_m3']:,.0f} m³). 관리 기준: 계획으로 인한 수로 칸({s['n_points']}곳) 추가 침하 ≤ **{s['limit_mm']}mm/3개월**. "
         f"참고로 배경 침하(다른 농가 취수)만으로 수로 칸 중앙값 {o['background_sub_mm']['median']:.1f}mm, 최대 {o['background_sub_mm']['max']:.1f}mm가 3개월 동안 생긴다.\n",
         ("우물 위치·깊이는 실제 신고 자료, 양수량은 신고값(없으면 신고 중앙값), 비용은 정수위 + 양수 강하 깊이의 전력비다. 관리 기준(3개월 추가 침하 한도)은 "
          "연 0.5인치 수로 기준 중 이 사업자 몫으로 둔 가정값이다. " if s.get("well_info") else
          "우물 위치·용량·비용은 가정값이다(관정별 실제 기록 비공개). ") +
         "땅속 성질·배경 취수·함양은 실제 자료로 역산한 값을 쓴다. "
         "실제 정답이 없으므로, 각 계획을 **E4 앙상블 30개 모두에서** 계산해 기준 초과 확률로 비교한다. "
         "주의: E4 계획은 바로 이 30개 모두에서 기준을 지키도록 만든 것이라 E4의 0%는 정의상 0%다(표본 내 값). "
         "계획에 쓰지 않은 땅속 상태로 다시 잰 값은 `eval_plan_oos.py` 결과(RESULTS_oos.md)를 본다.\n",
         "## 1. 계획 비교\n",
         "| 방식 | 실행 가능 | 총비용(천$) | 외부 구매(천 m³) | 최대 추가 침하 평균(mm) | 상위 5% (mm) | 기준 초과 확률(계획에 쓴 30개) | 임계수위 아래로 내려갈 확률 |",
         "|---|---|---|---|---|---|---|---|"]
    for k, pl in o["plans"].items():
        if not pl["feasible"]:
            L.append(f"| {NAMES[k]} | 불가 ({pl['reason']}) | - | - | - | - | - | - |"); continue
        rk = o["risk"][k]
        L.append(f"| {NAMES[k]} | 가능 | {pl['cost'] / 1e3:,.0f} | {pl['purchase'] / 1e3:,.0f} | {rk['worst_mean_mm']:.2f} | {rk['worst_p95_mm']:.2f} | "
                 f"{rk['exceed_prob'] * 100:.0f}% | {rk.get('below_crit_prob', 0) * 100:.0f}% |")
    n = o["head_need"]
    if n.get("background_below_crit"):
        L.append(f"\n**임계수위(과거 최저 수위)는 이 사업자의 계획으로 지킬 수 없다.** 계획 우물을 하나도 안 돌리고 수요 전량을 외부에서 사도, "
                 f"다른 농가의 배경 취수만으로 수로 칸 수위가 이미 과거 최저보다 낮다(앙상블 {n['background_below_crit_frac'] * 100:.0f}%). "
                 "그래서 모든 계획에서 '임계수위 아래로 내려갈 확률'이 100%로 나온다. **지역 전체(지하수 관리기관) 차원의 취수 조정이 필요하다**.\n")
    elif n.get("feasible"):
        L.append(f"\n임계수위를 지키려면 외부 용수가 **{n['purchase'] / 1e3:,.0f}천 m³** 필요하다(수요의 {n['purchase'] / s['demand_m3'] * 100:.0f}%, 비용 {n['cost'] / 1e3:,.0f}천$).\n")
    pl, rk = o["plans"], o["risk"]
    if all(pl[k]["feasible"] for k in ["E1", "E3", "E4"]):
        L += ["### 해석\n",
              f"- **E1(용량 비례)**은 모델 없이 고르게 뽑아 비용 {pl['E1']['cost'] / 1e3:,.0f}천$, 기준 초과 확률 {rk['E1']['exceed_prob'] * 100:.0f}%.",
              f"- **E3(위성 + 추정값 1개)**는 싸고 침하 영향이 작은 우물로 옮겨 비용을 {pl['E3']['cost'] / 1e3:,.0f}천$로 줄이지만, 기준선에 딱 맞춰 계획해서 초과 확률이 {rk['E3']['exceed_prob'] * 100:.0f}%다. "
              f"실제로 MODFLOW로 직접 돌리면 최대 {o['linearity_check'].get('E3', {}).get('direct_worst_mm', float('nan')):.2f}mm로 기준(2mm)을 넘는다(2절).",
              f"- **E4(불확실성 반영)**는 외부 용수 {pl['E4']['purchase'] / 1e3:,.0f}천 m³를 사서 비용이 {pl['E4']['cost'] / 1e3:,.0f}천$(E3보다 {(pl['E4']['cost'] / pl['E3']['cost'] - 1) * 100:.0f}% 높음)이지만, "
              f"앙상블 30개 모두에서 기준을 지킨다(초과 확률 {rk['E4']['exceed_prob'] * 100:.0f}%).",
              "- 가상 대수층 실험(synthetic/)과 같은 결론이 실제 지역 모델에서도 나왔다: 추정값 하나만 믿으면 기준선에 붙고, 불확실성을 반영하면 비용을 더 써서(이 조건에서는 외부 용수 구매 때문에 비용 증가가 가상 실험보다 크다) 위험을 없앤다.\n"]
    if s.get("well_info"):
        L += ["### 사업자 우물 (DWR 우물 신고 자료)\n", "| 우물 | 신고 번호 | 수로까지 | 깊이 | 용량 (m³/일) | 양수 비용 ($/m³) |", "|---|---|---|---|---|---|"]
        for w_ in s["well_info"]:
            L.append(f"| {w_['name']} | {w_['wcr']} | {w_['dist_km']:.1f} km | {w_['depth_ft']:.0f} ft | {w_['cap']:,.0f}{'' if w_['yield_reported'] else ' (중앙값)'} | {w_['cost']:.3f} |")
        L.append("")
    if o.get("margin"):
        mg = o["margin"]
        L += ["### 구조 오차 여유: 여유를 얼마나 둘지 = 비용\n",
              f"E4S는 응답을 {1 + mg['delta']:.2f}배로 키워 계획한다(여유 {mg['delta'] * 100:.0f}%, 출처: {mg['source']}).\n",
              "| 여유 | 비용(천$) | 외부 구매(천 m³) | 초과 확률: 모델대로 | 실제 반응 1.25배 | 1.5배 |", "|---|---|---|---|---|---|"]
        for c in mg["curve"]:
            if not c["feasible"]:
                L.append(f"| {c['margin'] * 100:.0f}% | 실행 불가 | - | - | - | - |"); continue
            L.append(f"| {c['margin'] * 100:.0f}% | {c['cost'] / 1e3:,.0f} | {c['purchase'] / 1e3:,.0f} | {c['exceed_x1.0'] * 100:.0f}% | "
                     f"{c['exceed_x1.25'] * 100:.0f}% | {c['exceed_x1.5'] * 100:.0f}% |")
        L.append("")
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
