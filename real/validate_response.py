"""응답 검증(자연 실험)과 2024.12 취수 금지 이후 침하 원인 나누기.

계획·정책 비교는 '취수를 바꾸면 수로 침하가 얼마나 바뀌나(응답)'로 결정된다. 전체 침하량 GPS 검증만으로는
이 응답이 맞는지 알 수 없어서, 실제로 취수가 크게 바뀐 두 사건을 '자연 실험'으로 쓴다.

1) 2023년 습한 해: 수로 물 배분율 2021·2022년 5% → 2023년 100%. 지하수를 덜 뽑았으니 침하가 느려졌어야 한다.
   → GPS 침하 속도 변화(2020~22 → 2023~24.11)와 모델이 예측한 속도 변화를 비교(보정에 안 쓴 기간).
2) 2024.12.9 취수 금지(수로 양쪽 2.5마일): 금지 뒤에도 GPS 침하는 오히려 빨라졌다.
   → 모델로 금지 후(2024.12~2026.4) 침하를 원인별로 나눈다(앙상블):
      · 과거 취수로 이미 진행 중인 침하(잔류) = 2024.12부터 모든 농업 취수를 멈춰도 남는 침하
      · 금지구역 밖 취수 몫 = (금지만 했을 때) − (모두 멈췄을 때)
      · 금지구역 안 취수가 계속됐다면 더해졌을 침하 = (금지 안 했을 때) − (금지했을 때)

실행: python3 validate_response.py   (MODFLOW 약 90회, 10코어 맥 약 5~10분)
출력: results_validate_<버전>/RESULTS.md, results.json, validate-1-rates.png, validate-2-ban.png
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import real_model as r
import run_real as rr          # 글꼴 설정
import plan_real as pr

HERE = Path(__file__).resolve().parent
OUT = HERE / f"results_validate_{r.TAG if r.PUMPING == 'et' else 'uniform'}"
OUT.mkdir(exist_ok=True)
BUFFER_KM = 4.02               # 2.5마일
BAN_MONTH = "2024-12"
WINDOWS = {"2020~2022 (가뭄, 배분율 5~20%)": ("2020-01", "2022-12"),
           "2023~2024.11 (습한 해, 배분율 40~100%)": ("2023-01", "2024-11"),
           "2024.12~2026.4 (취수 금지 후, 배분율 30~50%)": ("2024-12", "2026-04")}
N_MEMBERS = 30


def k_of(site, month):
    """출력 인덱스: S[k]는 site['months'][k+1] 끝 시점."""
    return list(site["months"]).index(pd.Period(month, "M")) - 1


def aq_dist_grid(site):
    pts = site["aq_pts"]; lat, lon = site["lat_c"], site["lon_c"]
    d = np.array([np.min(np.hypot((pts[:, 0] - a) * r.KM_LON, (pts[:, 1] - b) * r.KM_LAT)) for a, b in zip(lon.ravel(), lat.ravel())])
    return d.reshape(lat.shape)


def station_cells(site):
    out = {}
    for st, g in site["gps"].groupby("station"):
        la, lo = g.lat.iloc[0], g.lon.iloc[0]
        out[st] = (int((la - r.LAT0) * r.KM_LAT * 1000 // r.MCELL), int((lo - r.LON0) * r.KM_LON * 1000 // r.MCELL))
    return out


def fit_rate(t, y):
    ok = np.isfinite(y)
    if ok.sum() < 6 or (t[ok].max() - t[ok].min()) < 0.6:
        return np.nan
    return float(np.polyfit(t[ok], y[ok], 1)[0])                         # mm/년 (침하 +)


def gps_rates(site):
    g = site["gps"].copy()
    g["t"] = [(m.start_time - pd.Timestamp("2015-01-01")).days / 365.25 for m in g.month]
    out = {}
    for st, d in g.groupby("station"):
        out[st] = {}
        for w, (a, b) in WINDOWS.items():
            x = d[(d.month >= pd.Period(a, "M")) & (d.month <= pd.Period(b, "M"))]
            out[st][w] = fit_rate(x.t.to_numpy(), -x.up_mm.to_numpy())
    return out


def model_rates(site, S, cell):
    months = site["months"][1:]
    t = np.array([(m.start_time - pd.Timestamp("2015-01-01")).days / 365.25 for m in months])
    out = {}
    for w, (a, b) in WINDOWS.items():
        ka, kb = k_of(site, a), k_of(site, b)
        out[w] = fit_rate(t[ka:kb + 1], S[ka:kb + 1, cell[0], cell[1]])
    return out


def main():
    t0 = time.time()
    site = r.load_site()
    MODEL = r.result_dir("model", create=False); r.check_version(MODEL)
    res = json.load(open(MODEL / "results.json"))
    th3 = np.array(res["params"]["E3"])
    M = np.load(MODEL / "E4_members.npy")[:N_MEMBERS]
    pts = pr.mgmt_points(site)
    st_cells = station_cells(site)
    dist = aq_dist_grid(site)
    buffer_cells = [tuple(c) for c in np.argwhere(site["ag"] & (dist <= BUFFER_KM))]
    all_ag = [tuple(c) for c in np.argwhere(site["ag"])]
    kb = k_of(site, BAN_MONTH) + 1                                          # forcing 기간 인덱스(2024.12)
    base = {"months": 0, "wells": [], "rates": np.zeros((0, 0)), "start": 0}
    scen = {"A": None, "B": dict(base, zero_from=kb, zero_cells=buffer_cells), "C": dict(base, zero_from=kb, zero_cells=all_ag)}
    print(f"금지구역 농경지 {len(buffer_cells)}칸 / 전체 농경지 {len(all_ag)}칸, 수로 지점 {len(pts)}곳, 앙상블 {len(M)}개", flush=True)
    thetas, plans, tags = [], [], []
    for i, m in enumerate(M):
        for s, pl in scen.items():
            thetas.append(m); plans.append(pl); tags.append((i, s))
    outs = r.run_ensemble_plans(site, thetas, plans)
    runs = {}
    for (i, s), o in zip(tags, outs):
        if o is not None:
            runs.setdefault(i, {})[s] = o[0]
    runs = {i: v for i, v in runs.items() if len(v) == 3}
    print(f"  성공 {len(runs)}/{len(M)}", flush=True)

    # 1) 속도 비교(GPS vs 모델 A)
    obs = gps_rates(site)
    mod = {st: {w: [] for w in WINDOWS} for st in st_cells}
    for i, v in runs.items():
        for st, c in st_cells.items():
            for w, x in model_rates(site, v["A"], c).items():
                mod[st][w].append(x)
    rate_rows = []
    for st in sorted(st_cells):
        for w in WINDOWS:
            arr = np.array(mod[st][w], float)
            rate_rows.append({"station": st, "window": w, "gps": obs[st][w], "model_med": float(np.nanmedian(arr)),
                              "model_p5": float(np.nanpercentile(arr, 5)), "model_p95": float(np.nanpercentile(arr, 95))})
    ws = list(WINDOWS)
    changes = []
    for st in sorted(st_cells):
        for a, b in [(ws[0], ws[1]), (ws[1], ws[2])]:
            go = obs[st][b] - obs[st][a]
            if not np.isfinite(go):
                continue
            dm = np.array(mod[st][b], float) - np.array(mod[st][a], float)
            changes.append({"station": st, "from": a, "to": b, "gps_change": float(go), "model_change_med": float(np.nanmedian(dm)),
                            "model_change_p5": float(np.nanpercentile(dm, 5)), "model_change_p95": float(np.nanpercentile(dm, 95)),
                            "same_sign": bool(np.sign(go) == np.sign(np.nanmedian(dm)))})

    # 2) 금지 후 원인 나누기 (2024.12~2026.4, 연 환산)
    ka, kz = k_of(site, BAN_MONTH) - 1, k_of(site, "2026-04")
    yrs = (kz - ka) / 12.0
    iy, ix = np.array(pts).T
    def win(S, cells_iy, cells_ix):
        return (S[kz, cells_iy, cells_ix] - S[ka, cells_iy, cells_ix]) / yrs
    dec = {"aqueduct": [], **{st: [] for st in st_cells}}
    for i, v in runs.items():
        for key, (cy, cx) in [("aqueduct", (iy, ix))] + [(st, (np.array([c[0]]), np.array([c[1]]))) for st, c in st_cells.items()]:
            A, B, C = (np.median(win(v[s], cy, cx)) for s in "ABC")
            dec[key].append({"asis": A, "ban": B, "legacy": C, "outside": B - C, "inside_if_continued": A - B})
    decomp = {}
    for key, lst in dec.items():
        df = pd.DataFrame(lst)
        decomp[key] = {c: {"med": float(df[c].median()), "p5": float(df[c].quantile(0.05)), "p95": float(df[c].quantile(0.95))} for c in df}
        if key != "aqueduct":
            decomp[key]["gps_rate"] = obs[key][ws[2]]
    post = {}
    for nm, lab in [("log10_post_in", "in"), ("log10_post_out", "out")]:
        if nm in res["params"]["names"]:
            j = res["params"]["names"].index(nm); v = res["params"]["E3"][j]; sd = res["params"]["E3_sd"][j]
            post[lab] = {"med": 10 ** v, "p5": 10 ** (v - 1.645 * sd), "p95": 10 ** (v + 1.645 * sd)}
    out = {"variant": r.VARIANT, "n_members": len(runs), "post_mult": post, "buffer_cells": len(buffer_cells), "all_ag_cells": len(all_ag),
           "rates": rate_rows, "changes": changes, "decomposition": decomp, "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(out, open(OUT / "results.json", "w"), ensure_ascii=False, indent=1, default=float)
    figures(out)
    report(out)
    print((OUT / "RESULTS.md").read_text(encoding="utf-8"))


def figures(o):
    rows = pd.DataFrame(o["rates"])
    sts = [s for s in ["P544", "Q204", "Q208"] if s in set(rows.station)]
    fig, axes = plt.subplots(1, len(sts), figsize=(4.6 * len(sts), 4), squeeze=False)
    for ax, st in zip(axes[0], sts):
        d = rows[rows.station == st]
        x = np.arange(len(d))
        ax.bar(x - 0.18, d.gps, 0.36, color="#333", label="GPS 관측")
        ax.bar(x + 0.18, d.model_med, 0.36, color="#2166ac", label="모델(E4 중앙값)",
               yerr=[d.model_med - d.model_p5, d.model_p95 - d.model_med], capsize=3)
        ax.set_xticks(x, ["2020~22\n가뭄", "2023~24.11\n습한 해", "2024.12~\n금지 후"], fontsize=8.5)
        ax.axhline(12.7, color="#b2182b", lw=1, ls="--"); ax.text(len(d) - 0.5, 12.7, " 기준 0.5in/년", color="#b2182b", fontsize=7.5, va="bottom", ha="right")
        ax.set_title(f"GPS {st}", fontsize=11); ax.set_ylabel("침하 속도 (mm/년)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("자연 실험: 수로 물이 늘고 줄 때 침하 속도가 바뀌는가 (보정에 안 쓴 기간)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "validate-1-rates.png", dpi=150); plt.close(fig)

    keys = ["aqueduct"] + [s for s in ["Q204", "Q208", "P544"] if s in o["decomposition"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for j, k in enumerate(keys):
        d = o["decomposition"][k]
        leg, outs = d["legacy"]["med"], d["outside"]["med"]
        ax.bar(j, leg, 0.55, color="#9e9e9e", label="과거 취수로 이미 진행 중인 침하" if j == 0 else None)
        ax.bar(j, outs, 0.55, bottom=leg, color="#f4a582", label="금지구역 밖 취수" if j == 0 else None)
        if "gps_rate" in d and np.isfinite(d["gps_rate"]):
            ax.plot(j, d["gps_rate"], "k*", ms=12, label="GPS 실제 속도" if j == 1 else None)
    ax.axhline(12.7, color="#b2182b", lw=1, ls="--", label="이웃 기관 수로 기준(0.5인치/년)")
    ax.set_xticks(range(len(keys)), ["수로 전 구간\n(중앙값)" if k == "aqueduct" else f"GPS {k}" for k in keys])
    ax.set_ylabel("2024.12~2026.4 침하 속도 (mm/년)")
    ax.set_title("취수 금지 뒤에도 수로가 꺼지는 이유 (모델, 금지 준수 가정)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "validate-2-ban.png", dpi=150); plt.close(fig)


def report(o):
    L = ["# 응답 검증(자연 실험)과 취수 금지 후 침하 원인\n",
         (f"모델 변형 {o['variant']}, E4 앙상블 {o['n_members']}개. " +
          ("**운영 모델은 2023~2026 GPS를 보정에 썼으므로 1·2절은 독립 검증이 아니라 적합도다.** 독립 검증은 검증 모델(results_validate_v3)로 말한다.\n"
           if str(o["variant"]).endswith("ops") else "모든 기간이 보정(2015.7~2022.12) 뒤이고, 보정에 GPS를 쓰지 않았다(독립 검증).\n")),
         "## 1. 침하 속도 (mm/년, 아래로 +)\n", "| GPS | 기간 | GPS 관측 | 모델 중앙값 (90% 범위) |", "|---|---|---|---|"]
    f = lambda v: "-" if v is None or not np.isfinite(v) else f"{v:.1f}"
    for x in o["rates"]:
        if not np.isfinite(x["gps"]):
            continue
        L.append(f"| {x['station']} | {x['window']} | {f(x['gps'])} | {f(x['model_med'])} ({f(x['model_p5'])}~{f(x['model_p95'])}) |")
    L += ["\n## 2. 취수가 바뀔 때 침하 속도 변화 — 응답 검증\n", "| GPS | 변화 | GPS 관측 | 모델 (90% 범위) | 방향 일치 |", "|---|---|---|---|---|"]
    for c in o["changes"]:
        L.append(f"| {c['station']} | {c['from'].split(' ')[0]} → {c['to'].split(' ')[0]} | {c['gps_change']:+.1f} | "
                 f"{c['model_change_med']:+.1f} ({c['model_change_p5']:+.1f}~{c['model_change_p95']:+.1f}) | {'예' if c['same_sign'] else '아니오'} |")
    n_ok = sum(c["same_sign"] for c in o["changes"])
    L.append(f"\n{len(o['changes'])}개 변화 중 {n_ok}개에서 모델이 방향을 맞혔다.\n")
    if o.get("post_mult"):
        pm = o["post_mult"]
        L += ["## 금지 이후 실제 취수: 위성 침하·GPS로 거꾸로 추정 (운영 모델)\n",
              "관정별 계량 기록은 비공개다. 운영 모델은 2024.12 이후 취수가 위성 증발산 추정의 몇 배였는지를 GPS 침하로 역산했다.\n",
              "| 구역 | 위성 추정 대비 실제 취수 배수 (90% 범위) |", "|---|---|"]
        for k, nm in [("in", "금지구역 안(수로 양쪽 2.5마일)"), ("out", "금지구역 밖")]:
            if k in pm:
                L.append(f"| {nm} | {pm[k]['med']:.2f}배 ({pm[k]['p5']:.2f}~{pm[k]['p95']:.2f}) |")
        L.append("\n1배보다 크면 위성으로 보이는 작물 물 사용보다 더 뽑았다는 뜻이다(미계량·이전 취수, 또는 모델 구조 오차). "
                 "관리기관이 현장 확인할 구역을 고르는 근거로 쓴다.\n")
    L += ["## 3. 2024.12 취수 금지 뒤에도 꺼지는 이유 (2024.12~2026.4, mm/년)\n",
          "금지구역(수로 양쪽 2.5마일) 농업 취수가 2024.12부터 완전히 멈췄다고 가정하고 모델로 나눴다.\n",
          "| 위치 | GPS 실제 | 금지 준수 시 모델 | └ 과거 취수로 진행 중인 침하 | └ 금지구역 밖 취수 | 금지 안 했으면 추가 |", "|---|---|---|---|---|---|"]
    for k, d in o["decomposition"].items():
        nm = "수로 전 구간(지점 중앙값)" if k == "aqueduct" else f"GPS {k}"
        g = f(d.get("gps_rate")) if "gps_rate" in d else "-"
        L.append(f"| {nm} | {g} | {d['ban']['med']:.1f} | {d['legacy']['med']:.1f} ({d['legacy']['p5']:.1f}~{d['legacy']['p95']:.1f}) | "
                 f"{d['outside']['med']:.1f} ({d['outside']['p5']:.1f}~{d['outside']['p95']:.1f}) | {d['inside_if_continued']['med']:.1f} |")
    a = o["decomposition"]["aqueduct"]
    L += ["", f"- 모델이 본 실제(금지 후) 수로 침하 연 {a['asis']['med']:.1f}mm = 과거 취수의 영향 {a['legacy']['med']:+.1f} "
              f"(음수면 수위 회복으로 되올라오는 몫) + 금지구역 밖 취수 {a['outside']['med']:+.1f} + **금지구역 안에서 계속된 취수 {a['inside_if_continued']['med']:+.1f}** (mm/년, 중앙값).",
          f"- 금지가 완전히 지켜졌다면 수로 침하는 연 {a['ban']['med']:.1f}mm로 거의 멈췄을 것이다(모델). "
          "실제로 계속 꺼진다면 금지구역 안 취수가 계속됐거나(아래 '보이지 않는 취수'), 모델이 아직 못 담은 원인이 있다는 뜻이다.",
          "- 관리기관이 할 일: 금지구역 안 미계량 취수 확인 + 금지구역 밖까지 포함한 관정별 배분.",
          "- 한계: 금지가 실제로 얼마나 지켜졌는지(계량 기록)는 비공개다. 2025~26년 수로 물 배분율(50%·30%)이 2023년(100%)보다 낮아 금지구역 밖 취수가 늘었을 수 있다.\n",
          "![속도](validate-1-rates.png)\n", "![금지](validate-2-ban.png)\n", f"계산 시간 {o['minutes']}분.\n"]
    (OUT / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        o = json.load(open(OUT / "results.json")); figures(o); report(o)
        print((OUT / "RESULTS.md").read_text(encoding="utf-8"))
    else:
        main()
