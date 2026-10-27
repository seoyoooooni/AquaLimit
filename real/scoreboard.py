"""채점표: 모델 구조를 바꿀 때마다 같은 기준으로 기록한다.

두 단계 채점(정답에 맞춰 고르지 않기 위해)
  연습 시험(dev): 2015.7~2019.12 자료로 보정 → 2020.1~2022.12 위성 침하(InSAR)를 예측. 개선 여부는 이것으로만 판단한다.
  최종 시험(test): 2015.7~2022.12로 보정 → 2023~2026 GPS. 모델을 확정(동결)할 때만 본다.

연습 시험 지표
  수로 2km 안 칸의 위성 침하 변화(2019.12 기준) RMSE, 평균 편향. 정답 위성: 10/23부터 OPERA(rescore_practice.py), 그 전엔 DWR
  비교: 같은 칸 2015.7~2019.12 DWR 추세를 그대로 연장한 예측, 변화 0 예측

실행
  python3 scoreboard.py            # 결과 폴더들을 모아 SCOREBOARD.md 만들기(계산 없음)
  AQ_VARIANT=... python3 run_real.py --dev   # 새 구조의 연습 시험(보정 1번, 10코어 맥 약 5분)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEV_CAL_END, DEV_VAL_END = "2019-12", "2022-12"


def dev_score(site, th, r):
    """연습 시험 점수. r = real_model 모듈(지금 변형 설정을 따름)."""
    out = r.run_model(site, th)
    idx = r.obs_index(site, "val", True, use_gps=False, cal_end=DEV_CAL_END, val_end=DEV_VAL_END)
    ins = idx["ins"]; n = idx["n_ins"]
    months = site["months"]; k0 = list(months).index(pd.Period(DEV_CAL_END, "M"))
    S = out[0]
    iy, ix, kk = ins[:, 0].astype(int), ins[:, 1].astype(int), ins[:, 2].astype(int)
    obs = idx["d"][:n] - r._obs_at(site, ins, k0)
    mod = S[kk - 1, iy, ix] - S[k0 - 1, iy, ix]
    # 추세 연장 기준(같은 1km 칸 평균 InSAR, 2015.7~2019.12)
    t = np.array([(m.start_time - months[0].start_time).days / 365.25 for m in months])
    cells = site["cells"]; u = np.where(site["use"])[0]
    key = (cells.iy.to_numpy()[u] // r.F) * 1000 + cells.ix.to_numpy()[u] // r.F
    D = pd.DataFrame(site["disp"][u][:, 1:k0 + 1]).groupby(key).mean()
    rate = {}
    for kc, row in D.iterrows():
        y = row.to_numpy(); ok = np.isfinite(y)
        rate[int(kc)] = np.polyfit(t[1:k0 + 1][ok], y[ok], 1)[0] if ok.sum() > 12 else np.nan
    trend = np.array([rate.get(int(a) * 1000 + int(b), np.nan) for a, b in zip(iy, ix)]) * (t[kk] - t[k0])
    near = r.aqueduct_cells(site, 2.0)
    m_near = np.array([(int(a), int(b)) in near for a, b in zip(iy, ix)])
    ok = np.isfinite(obs) & np.isfinite(trend)
    rm = lambda e, m: float(np.sqrt(np.mean(e[m] ** 2)))
    res = {}
    for name, m in [("near", ok & m_near), ("all", ok)]:
        res[name] = {"model_rmse": rm(mod - obs, m), "trend_rmse": rm(trend - obs, m), "zero_rmse": rm(-obs, m),
                     "model_bias": float(np.mean((mod - obs)[m])), "n": int(m.sum())}
    # 2022.12 시점(예측 3년) 수로 칸 변화량 비교
    last = ok & m_near & (kk == list(months).index(pd.Period(DEV_VAL_END, "M")))
    res["near_at_end"] = {"obs_med": float(np.median(obs[last])), "model_med": float(np.median(mod[last])),
                          "trend_med": float(np.median(trend[last]))}
    return res


def params_at_bound(names, th, mean, sd, tol=0.02):
    lo, hi = np.array(mean) - 3 * np.array(sd), np.array(mean) + 3 * np.array(sd)
    return [n.replace("log10_", "") for n, v, a, b in zip(names, th, lo, hi) if min(v - a, b - v) < tol]


def collect():
    rows = []
    for d in sorted(HERE.glob("results_*")):
        mv = d / "model_version.json"
        if not mv.exists() or d.name.startswith(("results_plan", "results_ai", "results_policy", "results_validate")):
            continue
        v = json.load(open(mv))
        row = {"폴더": d.name, "변형": (v.get("variant") or v.get("pumping")) + ("-opera" if v.get("insar") == "opera" else ""),
               "파라미터 수": len(v.get("param_names", []))}
        dev = json.load(open(d / "results_dev.json")) if (d / "results_dev.json").exists() else None
        rs = json.load(open(d / "results_dev_rescore.json")) if (d / "results_dev_rescore.json").exists() else None
        if rs:                                     # 10/23~: 연습 시험 정답 = OPERA(GPS로 확인된 위성), rescore_practice.py
            row.update({"연습(OPERA): 수로 RMSE": rs["opera"]["near"]["model"], "연습(OPERA): 추세 연장": rs["opera"]["near"]["trend"],
                        "연습(OPERA): 편향": rs["opera"]["near"]["bias_model"], "연습: GPS P544": rs["p544"]["model"]})
        if dev:
            row.update({"연습(DWR, 10/22까지): 수로 RMSE": dev["near"]["model_rmse"], "연습(DWR): 추세 연장": dev["near"]["trend_rmse"],
                        "연습(DWR): 편향": dev["near"]["model_bias"], "범위 끝 파라미터": ", ".join(dev.get("at_bound", []))})
        res = json.load(open(d / "results.json")) if (d / "results.json").exists() else None
        e3o = json.load(open(d / "results_e3only.json")) if (d / "results_e3only.json").exists() else None
        src = res or e3o
        if src and src.get("E3", {}).get("val", {}).get("gps_rmse") is not None and not str(v.get("variant") or "").endswith("ops"):
            row["최종: GPS E3"] = src["E3"]["val"]["gps_rmse"]
            e4 = (res or {}).get("E4", {}).get("val", {})
            if e4.get("gps_cov90_with_model_error") is not None:
                row["최종: 90% 포함률"] = e4["gps_cov90_with_model_error"] * 100
                row["최종: 범위 폭"] = e4.get("gps_band90_mean_width_mm")
        vf = HERE / f"results_validate_{row['변형']}" / "results.json"
        if vf.exists():
            ch = json.load(open(vf)).get("changes", [])
            row["자연 실험 방향"] = f"{sum(c['same_sign'] for c in ch)}/{len(ch)}"
        rows.append(row)
    df = pd.DataFrame(rows)
    L = ["# 채점표\n",
         "개선 여부는 **연습 시험**(2019년까지 보정 → 2020~2022 위성 예측, 수로 2km 칸)으로만 판단한다. "
         "**최종 시험**(2022년까지 보정 → 2023~2026 GPS)은 모델을 확정할 때만 본다. 단위 mm.\n",
         "같은 연습 시험에서 '추세 연장'(2015~2019 위성 추세를 그대로 늘림)보다 작아야 물리모델이 데이터 추세보다 낫다고 말할 수 있다.\n",
         "**10/23 연습 시험 정답 변경: DWR InSAR → NASA OPERA.** 연습 기간(2020~22) GPS가 있는 유일한 곳 P544에서 GPS 27.8mm, OPERA 24.9mm, DWR 13.6mm 침하 — "
         "DWR가 절반만 잡는다(2023년 이후 GPS와 반대로 가는 문제가 2020년 무렵부터 시작). 같은 보정 결과(H2019)를 다시 채점만 했다(rescore_practice.py, RESCORE_PRACTICE.md). "
         "DWR 정답 열은 기록으로 남긴다. 'GPS P544' = 연습 기간 P544 GPS 월 값 36개 RMSE(추세 연장 4.1mm, 변화 0 12.5mm).\n"]
    if len(df):                                   # tabulate 없이 마크다운 표
        fmt = lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.1f}" if isinstance(x, float) else str(x))
        L.append("| " + " | ".join(df.columns) + " |"); L.append("|" + "---|" * len(df.columns))
        for _, rw in df.iterrows():
            L.append("| " + " | ".join(fmt(x) for x in rw) + " |")
    (HERE / "SCOREBOARD.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return df


if __name__ == "__main__":
    print(collect().to_string())
