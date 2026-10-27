"""연습 시험 다시 채점: 정답을 DWR InSAR 대신 GPS로 확인된 NASA OPERA로 바꿔 본다(재보정 없음, 변형마다 MODFLOW 1회).

왜: 연습 시험 기간(2019.12 → 2022.12)에 GPS가 있는 유일한 곳(P544, 반경 0.75km 위성 칸 평균)에서
      GPS 27.8mm, OPERA 24.9mm, DWR 13.6mm 침하 — DWR가 절반만 잡는다.
    수로 2km 안 500m 칸 중앙값: 2021~22 대가뭄(수로 배분율 5%)인데 DWR는 침하 속도가 연 14.0 → 7.2mm로 줄었다고,
      OPERA는 12.0 → 17.1mm로 빨라졌다고 본다. 수로 GPS 4곳 칸에서 DWR는 2020~22에 이미 느려지고 2023년 이후엔 상승(GPS와 반대).
    → DWR의 2023년 이후 오류가 2020년 무렵부터 시작됐을 가능성이 크다. 기존 연습 시험에서 '변화 0'(17.0mm)이 1등이었던 것도 이 때문일 수 있다.
방법: 각 변형의 H2019(2015.7~2019.12 DWR로 보정, run_real.py --dev가 저장)를 그대로 쓰고, 같은 예측을 세 정답으로 채점한다.
      ① DWR InSAR(기존 정답) ② OPERA(새 정답) ③ GPS P544(한 곳뿐이지만 위성과 독립)
      비교: 추세 연장(2015.7~2019.12 DWR 칸 추세 — 모델 보정과 같은 자료), 변화 0, 결합(추세 + 모델이 예측한 추세에서 벗어나는 만큼, hybrid_forecast.py)
      ①은 scoreboard.py의 연습 시험과 같은 계산이다(같은 숫자가 나와야 함 = 확인용).
판정 규칙(결과 보기 전에 정함): ② 수로 2km RMSE가 가장 작은 방식을 고르고, ③ P544에서 크게 틀리지 않는지 본다.
      10/27 추가 후보 '칸별 보정'(hybrid_forecast.py의 ratio): 보정 기간(2015.7~2019.12) 1km 칸마다 '위성 누적 침하 ÷ 모델 누적 침하'를
      하나 구해(원점 통과 회귀, 0.2~5배) 예측에 곱한다 = 칸마다 점토 압축성 배수 1개. 10/21 DWR 정답으로만 채점해 탈락(37.1mm)했던 방식이다.
      같은 규칙: ② 수로 2km에서 추세 연장보다 작을 때만 최종 시험(hybrid_forecast.py)을 한 번 돌린다.
      고른 방식이 ②에서 추세 연장보다 작을 때만 최종 시험(2023~2026 GPS)을 한 번 돌린다.
실행: for v in v3 v4 v4-corc v4-emult v4-pfrac v4-kv9 v4-aq; do AQ_VARIANT=$v python3 rescore_practice.py; done
      python3 rescore_practice.py --table      → RESCORE_PRACTICE.md (변형마다 1분 안쪽)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def opera_site(site, r):
    """site(DWR로 읽은 것)의 위성 자료만 OPERA로 바꾼 사본(같은 500m 칸·같은 달 범위, load_site와 같은 방식). 모델 입력은 그대로."""
    z = np.load(r.DATA / "opera_grid500m_monthly.npz")
    zm = pd.PeriodIndex(z["months"], freq="M")
    a, b = list(zm).index(pd.Period(r.REF_MONTH, "M")), list(zm).index(pd.Period(r.END, "M"))
    so = dict(site)
    so["disp"] = -z["disp_mm"][:, a:b + 1].astype(float)
    oil = site["cells"].oil_field.fillna("").astype(str).str.len() > 0
    so["use"] = ((~oil) & (np.isfinite(so["disp"]).sum(1) >= 24)).to_numpy()
    return so


def table():
    rows = []
    for f in sorted(HERE.glob("results_*/results_dev_rescore.json")):
        o = json.load(open(f)); rows.append(o)
    if not rows:
        print("results_*/results_dev_rescore.json 없음"); return
    L = ["# 연습 시험 다시 채점 (정답: DWR → OPERA)\n",
         "2015.7~2019.12로 보정한 모델(H2019)이 2020.1~2022.12 침하를 예측. 수로 2km 안 1km 칸, 2019.12 대비 누적 침하 RMSE(mm, 작을수록 좋음). "
         "P544는 GPS 월 값 36개. 정답을 바꾼 이유는 rescore_practice.py 맨 위 설명 참고.\n",
         "| 변형 | 방식 | ① DWR 정답 (기존) | **② OPERA 정답** | ③ GPS P544 | 2022.12 수로 칸 중앙값 (DWR / OPERA / 예측) |",
         "|---|---|---|---|---|---|"]
    best = []
    for o in rows:
        for meth in [m for m in ["model", "ratio", "hybrid", "trend", "zero"] if m in o["opera"]["near"]]:
            if meth in ("trend", "zero") and o is not rows[0]:
                continue                                      # 추세·변화 0은 변형과 무관 → 한 번만
            lab = {"model": "물리모델", "ratio": "칸별 보정(위성 비율 × 모델)", "hybrid": "결합(추세 + 모델 변화)", "trend": "추세 연장", "zero": "변화 0"}[meth]
            v = o["variant"] if meth in ("model", "ratio", "hybrid") else "-"
            e = o["end"]
            L.append(f"| {v} | {lab} | {o['dwr']['near'][meth]:.1f} | **{o['opera']['near'][meth]:.1f}** | {o['p544'][meth]:.1f} | "
                     f"{e['dwr_obs']:.0f} / {e['opera_obs']:.0f} / {e[meth]:.0f} |")
            best.append((o["opera"]["near"][meth], f"{v} {lab}", o["dwr"]["near"][meth], o["p544"][meth]))
    best.sort()
    L += ["", "## ② OPERA 정답 순위 (수로 2km)\n", "| 순위 | 방식 | ② OPERA | ① DWR | ③ P544 |", "|---|---|---|---|---|"]
    for i, (a, lab, b, c) in enumerate(best, 1):
        L.append(f"| {i} | {lab} | {a:.1f} | {b:.1f} | {c:.1f} |")
    L += ["", "지역 전체(수로 2km 밖 포함) 값은 각 폴더 results_dev_rescore.json의 'all'에 있다."]
    (HERE / "RESCORE_PRACTICE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    if os.environ.get("AQ_INSAR", "dwr") != "dwr":
        sys.exit("AQ_INSAR 없이 실행(모델은 DWR로 보정한 H2019를 쓴다)")
    import real_model as r
    import scoreboard as sb
    from hybrid_forecast import cell_rates, cell_ratios, tyears
    t0 = time.time()
    d = r.result_dir("model", create=False)
    r.check_version(d)
    cache = json.load(open(d / "calib_cache.json"))
    if "H2019" not in cache:
        sys.exit(f"{d.name}에 H2019 없음: AQ_VARIANT={r.VARIANT} python3 run_real.py --dev 먼저")
    th = np.array(cache["H2019"]["th"])
    site = r.load_site()
    months = list(site["months"]); t = tyears(site["months"])
    k0 = months.index(pd.Period(sb.DEV_CAL_END, "M")); k_end = months.index(pd.Period(sb.DEV_VAL_END, "M"))
    S = r.run_model(site, th)[0]

    so = opera_site(site, r)

    ro = cell_rates(site, k0)                                # 추세 연장: DWR 2015.7~2019.12 칸 추세(모델 보정과 같은 자료)
    rm_ = cell_rates(site, k0, S)                            # 같은 기간 모델 칸 추세
    rat = cell_ratios(site, k0, S)                           # 칸별 보정: 보정 기간 '위성 누적 ÷ 모델 누적'(칸마다 점토 압축성 배수 1개)
    near = r.aqueduct_cells(site, 2.0)

    def preds(iy, ix, kk):
        dt = t[kk] - t[k0]
        mod = S[kk - 1, iy, ix] - S[k0 - 1, iy, ix]
        trend = np.array([ro.get(int(p) * 1000 + int(q), np.nan) for p, q in zip(iy, ix)]) * dt
        ratio = np.array([rat.get(int(p) * 1000 + int(q), 1.0) for p, q in zip(iy, ix)]) * mod
        return {"model": mod, "trend": trend, "hybrid": trend + (mod - rm_[iy, ix] * dt), "ratio": ratio, "zero": np.zeros_like(mod)}

    f = lambda e: float(np.sqrt(np.mean(e ** 2)))
    out = {"variant": r.TAG}
    ends = {}
    for name, st in [("dwr", site), ("opera", so)]:
        idx = r.obs_index(st, "val", True, use_gps=False, cal_end=sb.DEV_CAL_END, val_end=sb.DEV_VAL_END)
        ins = idx["ins"]; n = idx["n_ins"]
        iy, ix, kk = ins[:, 0].astype(int), ins[:, 1].astype(int), ins[:, 2].astype(int)
        obs = idx["d"][:n] - r._obs_at(st, ins, k0)
        P = preds(iy, ix, kk)
        ok = np.isfinite(obs) & np.isfinite(P["trend"])
        m_near = np.array([(int(p), int(q)) in near for p, q in zip(iy, ix)])
        res = {}
        for lab, m in [("near", ok & m_near), ("all", ok)]:
            res[lab] = {k: f(v[m] - obs[m]) for k, v in P.items()}
            res[lab]["bias_model"] = float(np.mean(P["model"][m] - obs[m])); res[lab]["n"] = int(m.sum())
        out[name] = res
        last = ok & m_near & (kk == k_end)
        ends[f"{name}_obs"] = float(np.median(obs[last]))
        if name == "opera":                                  # 예측 중앙값은 OPERA 정답 칸 기준
            ends.update({k: float(np.median(v[last])) for k, v in P.items()})
    out["end"] = ends

    # GPS P544: 2019.12 대비 2020.1~2022.12 월 값(아래로 +)
    g = site["gps"]; g = g[g.station == "P544"]
    iy0 = int((g.lat.iloc[0] - r.LAT0) * r.KM_LAT * 1000 // r.MCELL); ix0 = int((g.lon.iloc[0] - r.LON0) * r.KM_LON * 1000 // r.MCELL)
    gs = (-g.set_index("month").up_mm)
    kk = np.array([k for k in range(k0 + 1, k_end + 1) if months[k] in gs.index])
    obs = np.array([gs[months[k]] for k in kk]) - gs[months[k0]]
    P = preds(np.full(len(kk), iy0), np.full(len(kk), ix0), kk)
    out["p544"] = {k: f(v - obs) for k, v in P.items()}
    out["p544"].update({"n": int(len(kk)), "gps_end": float(obs[-1]), "model_end": float(P["model"][-1]), "trend_end": float(P["trend"][-1])})
    out["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(out, open(d / "results_dev_rescore.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(f"[{r.TAG}] 수로 2km RMSE  ① DWR 정답: 모델 {out['dwr']['near']['model']:.1f} / 결합 {out['dwr']['near']['hybrid']:.1f} / "
          f"추세 {out['dwr']['near']['trend']:.1f} / 변화0 {out['dwr']['near']['zero']:.1f}")
    print(f"[{r.TAG}]               ② OPERA 정답: 모델 {out['opera']['near']['model']:.1f} / 칸별 보정 {out['opera']['near']['ratio']:.1f} / 결합 {out['opera']['near']['hybrid']:.1f} / "
          f"추세 {out['opera']['near']['trend']:.1f} / 변화0 {out['opera']['near']['zero']:.1f}")
    print(f"[{r.TAG}]               ③ GPS P544: 모델 {out['p544']['model']:.1f} / 칸별 보정 {out['p544']['ratio']:.1f} / 결합 {out['p544']['hybrid']:.1f} / 추세 {out['p544']['trend']:.1f} / "
          f"변화0 {out['p544']['zero']:.1f}  (2022.12 GPS {out['p544']['gps_end']:.0f}, 모델 {out['p544']['model_end']:.0f}, 추세 {out['p544']['trend_end']:.0f}mm) "
          f"({out['minutes']}분)")


if __name__ == "__main__":
    table() if "--table" in sys.argv else main()
