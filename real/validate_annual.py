"""자연 실험을 1년 단위 구간으로 다시 채점한다(validate_response.py의 구간은 1년 단위가 아니라 GPS 계절 변동이 섞였다).

GPS·모델 모두 '같은 달 → 같은 달' 1년 변화로 침하 속도를 잰다(끝점은 앞뒤 1달 포함 3개월 평균).
사건 1 (습한 해): 2021.11→2022.11 (배분율 5%) vs 2022.11→2023.11 (배분율 100%) — P544만 자료 있음
사건 2 (취수 금지): 2023.11→2024.11 (금지 전, 배분율 40%) vs 2024.11→2025.11 (금지 후, 배분율 50%) — P544·Q204·Q208

실행: AQ_VARIANT=v3 python3 validate_annual.py   (검증 모델, 독립 채점, MODFLOW 30회)
      AQ_VARIANT=v3ops python3 validate_annual.py (운영 모델, 적합도)
출력: results_validate_<버전>/RESULTS_annual.md, annual.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import real_model as r

HERE = Path(__file__).resolve().parent
OUT = HERE / f"results_validate_{r.TAG}"
OUT.mkdir(exist_ok=True)
EVENTS = [("습한 해(배분율 5% → 100%)", ("2021-11", "2022-11"), ("2022-11", "2023-11")),
          ("취수 금지(배분율 40% → 50%)", ("2023-11", "2024-11"), ("2024-11", "2025-11"))]
N_MEMBERS = 30


def end_mean_series(s, m):
    m = pd.Period(m, "M")
    v = s.reindex([m - 1, m, m + 1])
    return float(v.mean()) if v.notna().sum() >= 2 else np.nan


def rate(s, a, b):
    """연 침하 속도(mm/년, 아래로 +). s: 월 인덱스 Series, 아래로 + 누적 침하."""
    return end_mean_series(s, b) - end_mean_series(s, a)


def main():
    t0 = time.time()
    site = r.load_site()
    MODEL = r.result_dir("model", create=False); r.check_version(MODEL)
    M = np.load(MODEL / "E4_members.npy")[:N_MEMBERS]
    outs = r.run_ensemble(site, list(M))
    outs = [o for o in outs if o is not None]
    months = site["months"][1:]                                       # S[k] = months[k+1] 끝
    g = site["gps"]
    st_cells = {}
    for st, d in g.groupby("station"):
        st_cells[st] = (int((d.lat.iloc[0] - r.LAT0) * r.KM_LAT * 1000 // r.MCELL), int((d.lon.iloc[0] - r.LON0) * r.KM_LON * 1000 // r.MCELL))
    rows = []
    for st, (iy, ix) in sorted(st_cells.items()):
        gs = -g[g.station == st].set_index("month").up_mm                 # 아래로 +
        ms = [pd.Series(o[0][:len(months), iy, ix], index=months[:min(len(months), o[0].shape[0])]) for o in outs]
        for name, w0, w1 in EVENTS:
            g0, g1 = rate(gs, *w0), rate(gs, *w1)
            if not (np.isfinite(g0) and np.isfinite(g1)):
                continue
            dm = np.array([rate(s, *w1) - rate(s, *w0) for s in ms])
            m0 = np.array([rate(s, *w0) for s in ms]); m1 = np.array([rate(s, *w1) for s in ms])
            rows.append({"station": st, "event": name, "gps_before": g0, "gps_after": g1, "gps_change": g1 - g0,
                         "model_before": float(np.median(m0)), "model_after": float(np.median(m1)),
                         "model_change_med": float(np.median(dm)), "model_change_p5": float(np.percentile(dm, 5)),
                         "model_change_p95": float(np.percentile(dm, 95))})
    df = pd.DataFrame(rows)
    # 방향 일치: GPS 변화가 ±1mm/년 안이면 '거의 변화 없음'으로 보고, 모델 90% 범위가 그 안에 걸치면 일치로 본다
    def agree(r_):
        if abs(r_.gps_change) <= 1.0:
            return bool(r_.model_change_p5 <= 1.0 and r_.model_change_p95 >= -1.0)
        return bool(np.sign(r_.gps_change) == np.sign(r_.model_change_med))
    df["agree"] = [agree(x) for x in df.itertuples()]
    res = {"variant": r.TAG, "n_members": len(outs), "rows": df.to_dict("records"), "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(res, open(OUT / "annual.json", "w"), ensure_ascii=False, indent=1, default=float)
    ops = r.OPS
    L = [f"# 자연 실험 1년 단위 재채점 ({r.TAG})\n",
         ("운영 모델은 2023~2026 GPS를 보정에 썼으므로 **적합도**다." if ops else "검증 모델(2022년까지 보정)이라 사건 2는 **독립 채점**이다. 사건 1(2023 습한 해)도 보정 기간 뒤다."),
         " GPS·모델 모두 같은 달 → 같은 달 1년 변화(끝점 3개월 평균), mm/년, 아래로 +. GPS 변화가 ±1mm/년 안이면 '변화 거의 없음'으로 보고, "
         "모델 90% 범위가 ±1 안에 걸치면 일치로 센다.\n",
         "| GPS | 사건 | GPS 전 → 후 | GPS 변화 | 모델 전 → 후 (중앙값) | 모델 변화 (90% 범위) | 일치 |", "|---|---|---|---|---|---|---|"]
    for x in df.itertuples():
        L.append(f"| {x.station} | {x.event} | {x.gps_before:.1f} → {x.gps_after:.1f} | {x.gps_change:+.1f} | {x.model_before:.1f} → {x.model_after:.1f} | "
                 f"{x.model_change_med:+.1f} ({x.model_change_p5:+.1f}~{x.model_change_p95:+.1f}) | {'예' if x.agree else '아니오'} |")
    L += ["", f"{len(df)}개 중 {int(df.agree.sum())}개 일치. 앙상블 {len(outs)}개, 계산 시간 {res['minutes']}분."]
    (OUT / "RESULTS_annual.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
