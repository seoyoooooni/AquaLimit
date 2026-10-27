"""물리모델 + 위성 추세 결합 예측(하이브리드)을 연습 시험·최종 시험으로 채점한다.

왜: 물리모델(v3)은 '취수가 바뀌면 침하가 얼마나 바뀌나(응답)'는 맞히지만(2023 습한 해 감속을 크기까지 맞힘),
    칸마다 장기 침하 속도 자체는 틀린다(남쪽은 작게, P544는 크게). 단순 추세 연장은 그 반대다.
방법: 예측 = (보정 기간 위성 추세 그대로 연장) + (물리모델이 예측한 '자기 추세에서 벗어나는 만큼')
      = 물리모델 예측 + (보정 기간 위성 속도 − 같은 기간 모델 속도) × 경과 시간
      칸마다 모델의 속도 오차를 보정 기간 위성으로 고쳐 주고, 날씨·취수 변화에 따른 가속·감속은 물리모델에서 가져온다.
      쓰는 정보는 단순 추세 연장과 같다(보정 기간 자료만). 새로 맞춘 숫자(계수)는 없다.

채점
  연습 시험: 2019.12까지 보정(H2019) → 2020~2022 위성(수로 2km 칸, 지역 전체)
  최종 시험: 2022.12까지 보정(E3) → 2023~2026.4 수로 GPS (run_real.py와 같은 관측·같은 방식)
실행: AQ_VARIANT=v3 python3 hybrid_forecast.py   (MODFLOW 2회 + 앙상블 30회, 10코어 맥 1분)
출력: results_v3/RESULTS_hybrid.md, hybrid.json
"""
import json
import sys
import time

import numpy as np
import pandas as pd

import real_model as r
import scoreboard as sb

MODEL = r.result_dir("model", create=False)


def tyears(months):
    return np.array([(m.start_time - months[0].start_time).days / 365.25 for m in months])


def cell_rates(site, k_end, model_S=None):
    """1km 칸별 속도(mm/년, 아래로 +), 기간 months[1]~months[k_end]. model_S가 있으면 모델, 없으면 위성(사용 칸 평균)."""
    months = site["months"]; t = tyears(months)
    tt = t[1:k_end + 1]
    if model_S is not None:
        Y = model_S[0:k_end]                                            # S[k-1] = months[k]
        X = tt - tt.mean()
        return (np.tensordot(X, Y - Y.mean(0), axes=(0, 0)) / (X @ X))  # (ny, nx)
    cells = site["cells"]; u = np.where(site["use"])[0]
    key = (cells.iy.to_numpy()[u] // r.F) * 1000 + cells.ix.to_numpy()[u] // r.F
    D = pd.DataFrame(site["disp"][u][:, 1:k_end + 1]).groupby(key).mean()
    rate = {}
    for kc, row in D.iterrows():
        y = row.to_numpy(); ok = np.isfinite(y)
        rate[int(kc)] = np.polyfit(tt[ok], y[ok], 1)[0] if ok.sum() > 12 else np.nan
    return rate


def cell_ratios(site, k_end, S, lo=0.2, hi=5.0):
    """1km 칸별 '위성 누적 침하 / 모델 누적 침하' 기울기(원점 통과 회귀), 기간 months[1]~months[k_end]. 칸별 압축성 보정 배수."""
    cells = site["cells"]; u = np.where(site["use"])[0]
    key = (cells.iy.to_numpy()[u] // r.F) * 1000 + cells.ix.to_numpy()[u] // r.F
    D = pd.DataFrame(site["disp"][u][:, 1:k_end + 1]).groupby(key).mean()
    out = {}
    for kc, row in D.iterrows():
        y = row.to_numpy(); a, b = int(kc) // 1000, int(kc) % 1000
        x = S[0:k_end, a, b]; ok = np.isfinite(y)
        if ok.sum() > 12 and (x[ok] @ x[ok]) > 1e-6:
            out[int(kc)] = float(np.clip((x[ok] @ y[ok]) / (x[ok] @ x[ok]), lo, hi))
    return out


def practice(site, th):
    k0 = list(site["months"]).index(pd.Period(sb.DEV_CAL_END, "M"))
    S = r.run_model(site, th)[0]
    idx = r.obs_index(site, "val", True, use_gps=False, cal_end=sb.DEV_CAL_END, val_end=sb.DEV_VAL_END)
    ins = idx["ins"]; n = idx["n_ins"]; t = tyears(site["months"])
    iy, ix, kk = ins[:, 0].astype(int), ins[:, 1].astype(int), ins[:, 2].astype(int)
    obs = idx["d"][:n] - r._obs_at(site, ins, k0)
    mod = S[kk - 1, iy, ix] - S[k0 - 1, iy, ix]
    ro = cell_rates(site, k0); rm_ = cell_rates(site, k0, S)
    dt = t[kk] - t[k0]
    trend = np.array([ro.get(a * 1000 + b, np.nan) for a, b in zip(iy, ix)]) * dt
    hyb = trend + (mod - rm_[iy, ix] * dt)
    rat = cell_ratios(site, k0, S)
    mult = np.array([rat.get(a * 1000 + b, 1.0) for a, b in zip(iy, ix)]) * mod
    near = r.aqueduct_cells(site, 2.0)
    m_near = np.array([(int(a), int(b)) in near for a, b in zip(iy, ix)])
    ok = np.isfinite(obs) & np.isfinite(trend)
    f = lambda e, m: float(np.sqrt(np.mean(e[m] ** 2)))
    out = {}
    for name, m in [("near", ok & m_near), ("all", ok)]:
        out[name] = {"model": f(mod - obs, m), "trend": f(trend - obs, m), "hybrid": f(hyb - obs, m), "ratio": f(mult - obs, m),
                     "zero": f(-obs, m), "n": int(m.sum())}
    return out


def final(site, th, members=None):
    k_cal = list(site["months"]).index(pd.Period(r.CAL_END, "M")); t = tyears(site["months"])
    idx = r.obs_index(site, "val", True)
    gp = idx["gp"]; d = gp[:, 4]
    iy, ix, k, kr = (gp[:, i].astype(int) for i in range(4))
    dt = t[k] - t[kr]
    ro = cell_rates(site, k_cal)
    trend = np.array([ro.get(a * 1000 + b, np.nan) for a, b in zip(iy, ix)]) * dt

    def preds(S):
        mod = S[k - 1, iy, ix] - np.where(kr > 0, S[np.maximum(kr - 1, 0), iy, ix], 0.0)
        rm_ = cell_rates(site, k_cal, S)
        rat = cell_ratios(site, k_cal, S)                              # 칸별 보정: 보정 기간 위성 누적 ÷ 모델 누적
        ratio = np.array([rat.get(a * 1000 + b, 1.0) for a, b in zip(iy, ix)]) * mod
        return mod, trend + (mod - rm_[iy, ix] * dt), ratio

    S = r.run_model(site, th)[0]
    mod, hyb, rto = preds(S)
    f = lambda e: float(np.sqrt(np.mean(e ** 2)))
    out = {"model": f(mod - d), "trend": f(trend - d), "hybrid": f(hyb - d), "ratio": f(rto - d), "zero": f(d), "n": int(len(d))}
    st = r.load_gps(site["months"]).groupby("station").first()
    by = {}
    for s_, row in st.iterrows():
        a = int((row.lat - r.LAT0) * r.KM_LAT * 1000 // r.MCELL); b = int((row.lon - r.LON0) * r.KM_LON * 1000 // r.MCELL)
        m = (iy == a) & (ix == b)
        if m.any():
            by[s_] = {"n": int(m.sum()), "model": f((mod - d)[m]), "trend": f((trend - d)[m]), "hybrid": f((hyb - d)[m]), "ratio": f((rto - d)[m]),
                      "gps_end": float(d[m][-1]), "model_end": float(mod[m][-1]), "ratio_end": float(rto[m][-1]), "trend_end": float(trend[m][-1])}
    out["by_station"] = by
    if members is not None:                                            # 앙상블 + 구조 오차로 90% 범위 포함률
        outs = r.run_ensemble(site, list(members))
        PP = [preds(o[0]) for o in outs if o is not None]
        H = np.array([q[1] for q in PP]); HR = np.array([q[2] for q in PP])
        err = json.load(open(MODEL / "results.json")).get("model_error") or {}
        a0, b0 = err.get("s0_mm", 0.0), err.get("b_mm_per_yr", 0.0)
        sig = np.sqrt(a0 ** 2 + (b0 * dt) ** 2) if (a0 or b0) else np.zeros_like(dt)
        rng = np.random.default_rng(0)
        samp = H[rng.integers(0, len(H), 4000)] + rng.normal(size=(4000, len(d))) * sig
        lo, hi = np.percentile(samp, 5, 0), np.percentile(samp, 95, 0)
        sampR = HR[rng.integers(0, len(HR), 4000)] + rng.normal(size=(4000, len(d))) * sig
        loR, hiR = np.percentile(sampR, 5, 0), np.percentile(sampR, 95, 0)
        out["ensemble"] = {"n_members": int(len(H)), "hybrid_med_rmse": f(np.median(H, 0) - d),
                           "cov90": float(np.mean((d >= lo) & (d <= hi))), "width90": float(np.mean(hi - lo)), "model_error": err,
                           "ratio_med_rmse": f(np.median(HR, 0) - d), "ratio_cov90": float(np.mean((d >= loR) & (d <= hiR))),
                           "ratio_width90": float(np.mean(hiR - loR))}
    return out


def main():
    t0 = time.time()
    site = r.load_site()
    r.check_version(MODEL)
    cache = json.load(open(MODEL / "calib_cache.json"))
    res = json.load(open(MODEL / "results.json"))
    P = practice(site, np.array(cache["H2019"]["th"]))
    print("연습 시험:", json.dumps(P, ensure_ascii=False), flush=True)
    if "--practice-only" in sys.argv:
        return
    M = np.load(MODEL / "E4_members.npy") if (MODEL / "E4_members.npy").exists() else None
    Fn = final(site, np.array(res["params"]["E3"]), M)
    print("최종 시험:", json.dumps({k: v for k, v in Fn.items() if k not in ("ensemble", "by_station")}, ensure_ascii=False), flush=True)
    out = {"variant": r.TAG, "practice": P, "final": Fn, "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(out, open(MODEL / "hybrid.json", "w"), ensure_ascii=False, indent=1, default=float)
    report(out)
    print((MODEL / "RESULTS_hybrid.md").read_text(encoding="utf-8"))


def report(o):
    P, Fn = o["practice"], o["final"]
    L = [f"# 물리모델 + 위성 추세 결합 예측 ({o['variant']})\n",
         "예측 = 보정 기간 위성 추세 연장 + 물리모델이 예측한 '자기 추세에서 벗어나는 만큼'(가뭄·습한 해·취수 변화 효과). "
         "단순 추세 연장과 같은 정보만 쓰고, 새로 맞춘 계수는 없다. 단위 mm(작을수록 좋음).\n",
         "## 연습 시험 (2019.12까지 보정 → 2020~2022 위성)\n",
         "| 칸 | 물리모델만 | 추세 연장 | **결합** | 변화 0 |", "|---|---|---|---|---|"]
    for k, lab in [("near", "수로 2km 안"), ("all", "지역 전체")]:
        x = P[k]; L.append(f"| {lab} ({x['n']}개) | {x['model']:.1f} | {x['trend']:.1f} | **{x['hybrid']:.1f}** | {x['zero']:.1f} |")
    L.append("(연습 시험은 DWR 정답 기준. OPERA 정답 기준 연습 시험은 rescore_practice.py / RESCORE_PRACTICE.md)")
    L += ["", "## 최종 시험 (2022.12까지 보정 → 2023~2026.4 수로 GPS)\n",
          "| 방식 | GPS 침하 오차 |", "|---|---|",
          f"| 물리모델만 (E3) | {Fn['model']:.1f} |", f"| 칸별 보정(위성 비율 × 모델) | {Fn['ratio']:.1f} |", f"| 추세 연장 | {Fn['trend']:.1f} |",
          f"| 결합 | {Fn['hybrid']:.1f} |", f"| 변화 0 | {Fn['zero']:.1f} |", "",
          "| GPS | 월 값 | 물리모델만 | 칸별 보정 | 추세 연장 | 결합 | 마지막 달 GPS / 모델 / 칸별 보정 / 추세 (mm) |", "|---|---|---|---|---|---|---|"]
    for s_, x in Fn.get("by_station", {}).items():
        L.append(f"| {s_} | {x['n']} | {x['model']:.1f} | {x['ratio']:.1f} | {x['trend']:.1f} | {x['hybrid']:.1f} | "
                 f"{x['gps_end']:.0f} / {x['model_end']:.0f} / {x['ratio_end']:.0f} / {x['trend_end']:.0f} |")
    e = Fn.get("ensemble")
    if e:
        L += ["", f"앙상블 {e['n_members']}개(구조 오차 포함) 90% 범위: 결합 GPS 포함률 {e['cov90'] * 100:.0f}%·평균 폭 {e['width90']:.0f}mm, "
              f"칸별 보정 포함률 {e['ratio_cov90'] * 100:.0f}%·평균 폭 {e['ratio_width90']:.0f}mm."]
    L.append(f"\n계산 시간 {o['minutes']}분.")
    (MODEL / "RESULTS_hybrid.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
