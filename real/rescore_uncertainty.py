"""E4 불확실성 범위의 '모델 구조 오차'를 OPERA 기준으로 다시 재고, 최종 시험 GPS 포함률·범위 폭을 다시 계산한다.

왜: 구조 오차 √(17.2² + (10.6 × 예측 연수)²)는 2019년까지 보정한 모델로 2020~22 DWR InSAR를 예측한 오차로 쟀다(real_model.forecast_error).
    그런데 그 기간 DWR는 GPS P544 침하의 절반만 잡는다(rescore_practice.py 설명). 정답의 오류까지 '모델 오차'로 세었으니 범위가 지나치게 넓을 수 있다
    (90% 범위인데 GPS 포함률 99%, 평균 폭 97mm).
방법: 같은 H2019 모델로 오차를 DWR / OPERA 두 정답으로 재고, 같은 E4 앙상블(검증 모델 v3, 30개)에 각각 더해 2023~2026 GPS(보정에 안 쓴 자료)로 채점한다.
      90% 범위라면 포함률이 90% 근처이면서 폭이 좁을수록 좋다.
실행: AQ_VARIANT=v3 python3 rescore_uncertainty.py   (MODFLOW 32회, 10코어 맥 약 1~2분)
출력: results_v3/RESULTS_uncertainty.md, uncertainty.json
"""
import json
import os
import sys
import time

import numpy as np

import real_model as r
from rescore_practice import opera_site


def main():
    if os.environ.get("AQ_INSAR", "dwr") != "dwr" or r.VARIANT != "v3":
        sys.exit("AQ_VARIANT=v3 python3 rescore_uncertainty.py 로 실행(AQ_INSAR 없이)")
    t0 = time.time()
    d = r.result_dir("model", create=False); r.check_version(d)
    th = np.array(json.load(open(d / "calib_cache.json"))["H2019"]["th"])
    site = r.load_site(); so = opera_site(site, r)
    errs = {"DWR 정답으로 잰 구조 오차(기존)": r.forecast_error(site, th), "OPERA 정답으로 잰 구조 오차": r.forecast_error(so, th)}
    for k, e in errs.items():
        print(k, {q: round(v, 1) for q, v in e.items() if q in ("s0_mm", "b_mm_per_yr")}, flush=True)
    M = np.load(d / "E4_members.npy")
    outs = [o for o in r.run_ensemble(site, list(M)) if o is not None]
    rows = {"파라미터 불확실성만": r.evaluate(site, outs)["val"]}
    rows["파라미터 불확실성만"]["s0_mm"] = rows["파라미터 불확실성만"]["b_mm_per_yr"] = None
    for k, e in errs.items():
        v = r.evaluate(site, outs, err=e)["val"]; v.update({"s0_mm": e["s0_mm"], "b_mm_per_yr": e["b_mm_per_yr"], "table": e["table"]})
        rows[k] = v
    res = {"variant": r.TAG, "n_members": len(outs), "rows": rows, "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(res, open(d / "uncertainty.json", "w"), ensure_ascii=False, indent=1, default=float)
    L = ["# E4 불확실성 범위 다시 재기 (검증 모델 v3, 2023~2026.4 GPS로 채점)\n",
         "구조 오차 = 2019년까지 보정한 모델의 2020~22 예측 오차(수로 2km). 정답을 DWR(2020년 무렵부터 GPS와 어긋남)에서 OPERA(GPS와 일치)로 바꿔 다시 쟀다. "
         "90% 범위는 포함률이 90% 근처이면서 폭이 좁을수록 좋다.\n",
         "| 범위에 넣은 오차 | 구조 오차 식 (mm) | GPS 90% 범위 포함률 | 평균 범위 폭 (mm) |", "|---|---|---|---|"]
    for k, v in rows.items():
        f = f"√({v['s0_mm']:.1f}² + ({v['b_mm_per_yr']:.1f} × 연수)²)" if v.get("s0_mm") else "-"
        cov = v.get("gps_cov90_with_model_error", v.get("gps_cov90"))
        w = v.get("gps_band90_mean_width_mm")
        L.append(f"| {k} | {f} | {cov * 100:.0f}% | {w:.0f} |" if w else f"| {k} | {f} | {cov * 100:.0f}% | - |")
    L += ["", f"앙상블 {len(outs)}개, 계산 시간 {res['minutes']}분."]
    (d / "RESULTS_uncertainty.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
