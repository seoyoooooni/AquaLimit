"""날씨 시나리오: 다음 관개기(2026년 5~7월)가 가뭄 해 / 평년 / 습한 해일 때 계획이 어떻게 달라지나.

가뭄 계수 dry: +1 = 아주 건조(역산한 '가뭄 해 취수 증가'가 최대로 걸림, 강수는 평년값), 0 = 평년, -1 = 아주 습함.
다른 농가의 배경 취수가 바뀌어 배경 침하와 수위 상태(과거 최저와의 거리)가 달라지고, 그에 따라 같은 계획의 추가 침하도 달라진다.
실행: python3 run_weather.py  (MODFLOW 약 420회, 약 25분) → results_plan/weather.json, RESULTS_weather.md
"""
import json, time
import numpy as np
import real_model as r
import plan_real as p

OUT = r.result_dir("plan")
MODEL = r.result_dir("model", create=False)


def main():
    t0 = time.time()
    site = r.load_site()
    r.check_version(MODEL); r.check_version(OUT)
    M4 = np.load(MODEL / "E4_members.npy")
    pts, wells = p.mgmt_points(site), p.operator_wells(site)
    z = np.load(OUT / "R4.npz")
    R = {"평년": [{"Rs": z["Rs"][i], "Rh": z["Rh"][i]} for i in range(len(z["Rs"]))]}
    base = {}
    for name, d in [("가뭄 해", 1.0), ("습한 해", -1.0)]:
        print(f"[{name}] E4 구성원 {len(M4)}개 응답 계산")
        lst = [p.responses(site, th, wells, pts, dry=d) for th in M4]
        lst = [x for x in lst if x is not None]
        R[name] = lst
        base[name] = float(np.median([np.median(x["base_sub"]) for x in lst]))
    res0 = json.load(open(OUT / "results.json"))
    out = {"background_median_mm": base, "plans": {}, "risk": {}}
    out["background_median_mm"]["평년"] = res0["background_sub_mm"]["median"]
    plans = {"E4 평년 기준 계획": p.solve(R["평년"]), "E4 가뭄 해 기준 계획": p.solve(R["가뭄 해"]),
             "E4 세 시나리오 모두 기준": p.solve(R["평년"] + R["가뭄 해"] + R["습한 해"])}
    for k, pl in plans.items():
        out["plans"][k] = pl
        out["risk"][k] = {sc: (p.risk(pl, R[sc]) if pl["feasible"] else None) for sc in R}
    out["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(out, open(OUT / "weather.json", "w"), ensure_ascii=False, indent=1, default=float)
    L = ["# 날씨 시나리오 (2026년 5~7월)\n",
         "다음 관개기가 가뭄 해 / 평년 / 습한 해일 때. 가뭄 해에는 다른 농가가 지하수를 더 뽑는다(역산한 가뭄 계수). 기준: 계획 추가 침하 ≤ 2mm/3개월.\n",
         "| 시나리오 | 배경 침하(수로 칸 중앙값, 3개월) |", "|---|---|"]
    for sc in ["습한 해", "평년", "가뭄 해"]:
        L.append(f"| {sc} | {out['background_median_mm'][sc]:.2f} mm |")
    L += ["\n| 계획 | 비용(천$) | 외부 구매(천 m³) | 초과 확률(계획에 쓴 30개): 습한 해 | 평년 | 가뭄 해 |", "|---|---|---|---|---|---|"]
    for k, pl in plans.items():
        if not pl["feasible"]:
            L.append(f"| {k} | 불가 | - | - | - | - |"); continue
        rk = out["risk"][k]
        L.append(f"| {k} | {pl['cost'] / 1e3:,.0f} | {pl['purchase'] / 1e3:,.0f} | {rk['습한 해']['exceed_prob'] * 100:.0f}% | "
                 f"{rk['평년']['exceed_prob'] * 100:.0f}% | {rk['가뭄 해']['exceed_prob'] * 100:.0f}% |")
    L.append(f"\n계산 시간 {out['minutes']}분. 미래 강수는 월 평년값으로 두고 취수 쪽만 바꾼 단순 시나리오다.\n")
    (OUT / "RESULTS_weather.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
