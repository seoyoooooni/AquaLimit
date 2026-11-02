"""11/2 추가 위성 점검(모델 없이 관측만, 그림 없음) → results_opera/RESULTS_checks.md, checks.json

왜: 10/22 위성 이중차분(금지 후 − 금지 전, 금지구역 안 − 밖 = +2.9mm/년)을 '금지구역 안 취수가 계속됐을 가능성'의 근거로 썼다.
    11/2 모델 개선 시도(MODEL_TRIALS.md) 중 수로 근처 침하가 수로 배분율과 거의 상관없이 일정하고,
    그 일정한 침하가 Lost Hills 유전에 가까울수록 크다는 것을 봤다. 그래서 이중차분이
    (a) 금지가 없던 해에도 얼마나 흔들리는지(위약 검사), (b) 유전 영향을 빼도 그대로인지 확인한다.
1) 위약 검사: 같은 이중차분을 금지 전 해들(11월→11월 1년 속도, 다음 해 − 앞 해)에 계산 → 배분율 변화와의 관계로
   2024→2025(배분율 40 → 50%)에 금지가 없어도 예상되는 값을 구해 실제와 비교
2) 유전 영향: Lost Hills 유전에서 X km 안 칸을 빼고 이중차분 다시
3) 수로 거리별 연 침하 속도와 배분율 상관(수로 동쪽, 유전 칸 제외)
4) 수로 2km 안: Lost Hills 유전까지 거리별 습한 해(2023) 속도와 가뭄(2022) → 습한 해 감속
5) P544: GPS·OPERA·DWR 연 속도 — 보정 기간(2017~2019) 위성 두 개의 연별 차이를 GPS로 판정
실행: python3 opera_checks.py   (몇 초, opera_analysis.py와 같은 자료·같은 속도 계산)
"""
import json

import numpy as np
import pandas as pd

import opera_analysis as oa

ALLOC = pd.read_csv(oa.DATA / "swp_allocation.csv").set_index("year").alloc_pct


def yr_rate(df, y):
    """y년 관개기를 담는 1년 속도: (y−1).11 → y.11, mm/년(아래로 +)."""
    return oa.rates(df, f"{y - 1}-11", f"{y}-11")


def did_pair(cells, op, y):
    """y년 속도 → y+1년 속도 변화의 이중차분(금지구역 안 − 밖). y=2024가 실제 금지(2024.12)."""
    V = {"wet": yr_rate(op, y), "post": yr_rate(op, y + 1)}
    return None if not np.isfinite(V["post"]).any() else oa.did(cells, V)


def main():
    cells, op, dwr, gps, pts, oil, months = oa.load()
    lh = cells[cells.oil_field == "Lost Hills"]
    d_lh = np.array([np.min(np.hypot((lh.lon.values - a) * oa.KM_LON, (lh.lat.values - b) * oa.KM_LAT))
                     for a, b in zip(cells.lon.values, cells.lat.values)])
    out = {}
    L = ["# 위성 점검 추가 (11/2, 관측만)\n",
         "opera_checks.py. 자료·속도 계산은 opera_analysis.py와 같다(OPERA 500m 칸, 11월 → 11월 1년 속도, 끝점 3개월 평균, 유전 칸 제외). 아래로 +, mm/년.\n"]

    # 1) 위약 검사
    rows = []
    for y in range(2017, 2025):
        D = did_pair(cells, op, y)
        if D is None:
            continue
        rows.append({"pair": f"{y}→{y + 1}", "alloc": f"{ALLOC[y]}→{ALLOC[y + 1]}%", "d_alloc": int(ALLOC[y + 1] - ALLOC[y]),
                     "did": D["did_med"], "p5": D["p5"], "p95": D["p95"], "ban": y == 2024})
    pre = [r for r in rows if not r["ban"]]
    x = np.array([r["d_alloc"] for r in pre], float); z = np.array([r["did"] for r in pre])
    b, a = np.polyfit(x, z, 1)
    sd = float(np.std(z - (a + b * x), ddof=2))
    ban = [r for r in rows if r["ban"]][0]
    exp = float(a + b * ban["d_alloc"])
    out["placebo"] = {"rows": rows, "slope_per_pct": float(b), "intercept": float(a), "resid_sd": sd, "r": float(np.corrcoef(x, z)[0, 1]),
                      "expected_ban_pair": exp, "actual_ban_pair": ban["did"], "excess": ban["did"] - exp}
    L += ["## 1. 이중차분 위약 검사: 금지가 없던 해에도 같은 계산을 하면\n",
          "금지구역 안(수로 동쪽 4km) − 밖(4~10km)의 '다음 해 속도 − 앞 해 속도' 차이. 1km 묶음 중앙값, 90% 범위는 묶음 부트스트랩.\n",
          "| 해 | 수로 배분율 | 안 − 밖 (90% 범위) |", "|---|---|---|"]
    for r in rows:
        L.append(f"| {r['pair']}{' **(금지)**' if r['ban'] else ''} | {r['alloc']} | {r['did']:+.1f} ({r['p5']:+.1f}~{r['p95']:+.1f}) |")
    L += ["",
          f"- 금지 전 해들의 '안 − 밖'은 배분율 변화와 함께 움직인다(상관 {out['placebo']['r']:+.2f}, 배분율 10%p당 {b * 10:+.1f}mm/년). "
          "물이 늘면 바깥이 더 크게 느려지고, 수로 근처는 덜 반응하기 때문이다(3절).",
          f"- 2024→2025는 배분율이 {ban['alloc']}로 올라서 금지가 없어도 약 **{exp:+.1f}**가 예상된다. 실제 {ban['did']:+.1f}에서 남는 몫은 "
          f"**{ban['did'] - exp:+.1f}**로, 금지 전 해들이 이 관계에서 벗어나는 정도(표준편차 {sd:.1f})와 비슷하다.",
          "- 결론: '금지구역 안이 밖보다 더 좋아지지 않았다'(금지 효과가 위성에 안 보인다)는 그대로다. "
          "하지만 '안쪽이 덜 좋아졌다 = 금지구역 안 취수가 계속됐다'의 근거로 쓰기엔 약하다. 현장 확인 후보 목록은 그대로 쓸 수 있다.",
          f"- 한계: 금지 전 해가 {len(pre)}쌍뿐인 회귀다.\n"]

    # 2) 유전 영향
    orig = cells.is_oil.copy()
    V = {"wet": yr_rate(op, 2024), "post": yr_rate(op, 2025)}
    oil_rows = [{"x_km": 0, **oa.did(cells, V)}]
    for X in (1, 2, 3):
        cells["is_oil"] = orig | (d_lh <= X)
        oil_rows.append({"x_km": X, **oa.did(cells, V)})
    cells["is_oil"] = orig
    out["oil_excluded_did"] = oil_rows
    L += ["## 2. Lost Hills 유전 가까운 칸을 빼고 이중차분\n", "| 뺀 범위 | 이중차분 (90% 범위) | 1km 묶음 수 (안 / 밖) |", "|---|---|---|"]
    for r in oil_rows:
        lab = "유전 칸만(기존)" if r["x_km"] == 0 else "유전 %dkm 안까지" % r["x_km"]
        L.append(f"| {lab} | {r['did_med']:+.1f} ({r['p5']:+.1f}~{r['p95']:+.1f}) | {r['n_in']} / {r['n_out']} |")
    L += ["", "- 유전 근처를 빼도 이중차분은 줄지 않는다 → 이중차분은 유전 탓이 아니다. 일정한 유전 침하는 '후 − 전' 차이에서 빠지기 때문이다.\n"]

    # 3) 수로 거리별 연 속도와 배분율
    d = cells.d_aq_km.values; e = cells.east.values; keep = ~cells.is_oil.values
    years = list(range(2017, 2026))
    R = np.array([yr_rate(op, y) for y in years]).T
    A = np.array([ALLOC[y] for y in years], float)
    bands = [("수로 0~1km", 0, 1), ("수로 1~2km", 1, 2), ("2~4km", 2, 4), ("4~10km", 4, 10)]
    out["by_distance"] = {}
    L += ["## 3. 수로 거리별 연 침하 속도와 수로 배분율 (수로 동쪽, 유전 칸 제외, 칸 중앙값)\n",
          "| 구역 | 칸 | " + " | ".join(str(y) for y in years) + " | 배분율과 상관 | 가뭄(2020~22) ÷ 앞(2017~19) |",
          "|---|---|" + "---|" * len(years) + "---|---|",
          "| 배분율(%) | | " + " | ".join(str(int(v)) for v in A) + " | | |"]
    for name, lo, hi in bands:
        m = keep & e & (d > lo) & (d <= hi)
        med = np.nanmedian(R[m], 0)
        c = float(np.corrcoef(A, med)[0, 1]); ratio = float(med[3:6].mean() / med[0:3].mean())
        out["by_distance"][name] = {"cells": int(m.sum()), "rates": med.tolist(), "corr_alloc": c, "drought_ratio": ratio}
        L.append(f"| {name} | {int(m.sum())} | " + " | ".join(f"{v:.1f}" for v in med) + f" | {c:+.2f} | {ratio:.2f}배 |")
    L += ["", "- 바깥(4~10km)은 배분율이 낮은 해에 크게 빨라지고(가뭄 3배), 수로 바로 옆은 해마다 거의 같다. "
          "물리모델 후보들이 수로 근처 가뭄 반응을 크게 예측한 이유(MODEL_TRIALS.md)와 이중차분이 배분율 변화에 끌려가는 이유가 같다.\n"]

    # 4) 수로 2km 안: 유전 거리별
    w23, w22 = yr_rate(op, 2023), yr_rate(op, 2022)
    out["oil_distance"] = []
    L += ["## 4. 수로 2km 안: Lost Hills 유전까지 거리별 (수로 동쪽, 유전 칸 제외)\n",
          "| 유전까지 | 칸 | 습한 해 2023 속도 | 가뭄 2022 → 2023 감속 | 위도 중앙값 |", "|---|---|---|---|---|"]
    for lo, hi in [(0, 1), (1, 3), (3, 6), (6, 40)]:
        m = keep & e & (d <= 2) & (d_lh > lo) & (d_lh <= hi) & np.isfinite(w23) & np.isfinite(w22)
        if m.sum() < 5:
            continue
        r_ = {"from_km": lo, "to_km": hi, "cells": int(m.sum()), "wet2023": float(np.median(w23[m])), "slowdown": float(np.median(w22[m] - w23[m])),
              "lat": float(np.median(cells.lat.values[m]))}
        out["oil_distance"].append(r_)
        L.append(f"| {lo}~{hi}km | {r_['cells']} | {r_['wet2023']:.1f} | {r_['slowdown']:.1f} | {r_['lat']:.3f} |")
    gl = []
    for st, g in gps.groupby("station"):
        dd = float(np.min(np.hypot((lh.lon.values - g.lon.iloc[0]) * oa.KM_LON, (lh.lat.values - g.lat.iloc[0]) * oa.KM_LAT)))
        gl.append(f"{st} {dd:.1f}km")
    out["gps_to_lost_hills_km"] = gl
    L += ["", f"- 수로 GPS에서 Lost Hills 유전까지: {', '.join(gl)}. 습한 해·금지 후에도 계속 꺼진 Q198·Q204·Q208·Q213이 모두 2.5km 안이고, 5km 떨어진 P544만 습한 해에 반등했다.",
          "- 유전 거리와 위도가 겹쳐 있어(먼 칸은 대부분 북쪽) 상관일 뿐 원인 확정은 아니다. 모델 후보 v4-oil은 유전 경계 속도 연 14.4mm, 감소 거리 1.3km로 추정했다.\n"]

    # 5) P544: GPS vs OPERA vs DWR
    g = gps[gps.station == "P544"]
    ci = oa.gps_cell(cells, g.lat.iloc[0], g.lon.iloc[0])
    gg = g.set_index("month").up_mm.to_frame("v")
    rows5 = []
    for y in range(2017, 2023):
        a_, b_ = f"{y - 1}-11", f"{y}-11"
        gv = -(oa.end_mean(gg, b_).v - oa.end_mean(gg, a_).v)
        ov = -(oa.end_mean(op[[ci]], b_)[ci] - oa.end_mean(op[[ci]], a_)[ci])
        dv = -(oa.end_mean(dwr[[ci]], b_)[ci] - oa.end_mean(dwr[[ci]], a_)[ci])
        rows5.append({"year": y, "alloc": int(ALLOC[y]), "gps": float(gv), "opera": float(ov), "dwr": float(dv)})
    G = np.array([[r["gps"], r["opera"], r["dwr"]] for r in rows5])
    stats = {n: {"corr_gps": float(np.corrcoef(G[:, 0], G[:, i])[0, 1]), "rmse": float(np.sqrt(np.mean((G[:, 0] - G[:, i]) ** 2)))}
             for i, n in [(1, "opera"), (2, "dwr")]}
    out["p544"] = {"rows": rows5, "stats": stats}
    L += ["## 5. P544: GPS·OPERA·DWR 연 속도 (같은 500m 칸)\n", "| 해 | 배분율 | GPS | OPERA | DWR |", "|---|---|---|---|---|"]
    for r in rows5:
        L.append(f"| {r['year']} | {r['alloc']}% | {r['gps']:.1f} | {r['opera']:.1f} | {r['dwr']:.1f} |")
    L += ["", f"- GPS와 상관·RMSE: OPERA {stats['opera']['corr_gps']:+.2f}·{stats['opera']['rmse']:.1f}mm, DWR {stats['dwr']['corr_gps']:+.2f}·{stats['dwr']['rmse']:.1f}mm. "
          "보정 기간(2017~2019)에도 해마다의 움직임은 OPERA가 GPS에 더 가깝지만, 한 지점·6년이라 흔들림이 크다.\n"]

    json.dump(out, open(oa.OUT / "checks.json", "w"), ensure_ascii=False, indent=1, default=float)
    (oa.OUT / "RESULTS_checks.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
