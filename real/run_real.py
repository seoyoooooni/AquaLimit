"""실제 검증 지역 E2·E3·E4 실행 → results/ 에 결과 표·그림 저장.

실행: python3 real/run_real.py  (MODFLOW 6 필요, 2코어 기준 약 10~15분)
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import real_model as r

OUT = r.result_dir("model")
for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    try: font_manager.fontManager.addfont(p)
    except Exception: pass
for name in ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "Malgun Gothic"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name; break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def main(n_iter=6, n_post=30, e3_only=False):
    t0 = time.time()
    site = r.load_site()
    res = {}
    cache_f = OUT / "calib_cache.json"
    r.check_version(OUT)
    r.write_version(OUT)
    cache = json.load(open(cache_f)) if cache_f.exists() else {}
    th_start = warm_start()
    def cal(key, msg, **kw):                      # 역산 결과를 바로 저장(중간에 멈춰도 다시 계산하지 않게)
        if key in cache:
            print(f"{msg} (저장된 결과 사용)")
            c = cache[key]; return np.array(c["th"]), np.array(c["cov"]), c["hist"]
        print(msg)
        ck = OUT / f"ckpt_{key}.json"
        if r.V3 and key == "E2" and ck.exists() and json.load(open(ck))["it"] >= 3:
            # E2(관측정만)는 점토 성질이 정해지지 않아 극단값으로 가고, 그 근처에서 MODFLOW가 수렴하지 못한다(시간 초과).
            # 비교용이므로 마지막으로 저장된 반복 결과를 해로 쓰고, 공분산은 사전분포(관측정만으로는 거의 줄지 않음)로 둔다.
            c = json.load(open(ck))
            th, cov, h = np.array(c["th"]), np.diag(r.PRIOR_SD ** 2), c["hist"]
            print(f"  E2: 저장된 반복 {c['it']}회 결과를 해로 사용(이후 계산은 시간 초과로 생략), 공분산 = 사전분포")
        else:
            th, cov, h = r.calibrate(site, n_iter=(3 if (r.V3 and key == "E2") else n_iter), ckpt=ck, th0=th_start, **kw)
        cache[key] = {"th": th.tolist(), "cov": cov.tolist(), "hist": h}
        json.dump(cache, open(cache_f, "w"), default=float)
        return th, cov, h
    th3, cov3, h3 = cal("E3", "[E3] 관측정 수위 + 위성(InSAR)으로 역산", use_insar=True)
    if e3_only:                                   # 입력별 기여 실험: E3 보정 + GPS 채점만
        o3pt = r.run_model(site, th3)
        res = {"E3": r.evaluate(site, [o3pt]), "baselines": baselines(site), "variant": r.VARIANT,
               "params": {"names": r.NAMES, "E3": th3.tolist(), "E3_sd": np.sqrt(np.diag(cov3)).tolist()}, "history": {"E3": h3},
               "minutes": round((time.time() - t0) / 60, 1)}
        json.dump(res, open(OUT / "results_e3only.json", "w"), ensure_ascii=False, indent=1, default=float)
        v = res["E3"]
        print(f"[{r.VARIANT}] E3 보정 InSAR RMSE {v['cal']['insar_rmse']:.1f}mm, 검증 GPS RMSE {v['val']['gps_rmse']:.1f}mm, "
              f"수위 RMSE {v['val']['head_rmse']:.2f}m ({res['minutes']}분)")
        return
    if r.OPS:                                     # 운영 모델: E3(모든 자료) + E4 앙상블. 구조 오차는 검증 모델(v3)에서 잰 값
        print("[E4] 운영 모델 사후분포 앙상블"); M4, o4 = r.posterior_ensemble(site, th3, cov3, n=n_post)
        vf = r.HERE / "results_v3" / "results.json"
        err = json.load(open(vf)).get("model_error") if vf.exists() else None
        o3pt = r.run_model(site, th3)
        np.save(OUT / "E4_members.npy", M4)
        res = {"E3": r.evaluate(site, [o3pt]), "E4": r.evaluate(site, o4, err=err), "model_error": err, "baselines": baselines(site),
               "n_members": {"E4": len(o4)}, "variant": r.VARIANT,
               "params": {"names": r.NAMES, "prior": r.PRIOR_MEAN.tolist(), "E3": th3.tolist(), "E3_sd": np.sqrt(np.diag(cov3)).tolist(),
                          "E3_cov": cov3.tolist()},
               "history": {"E3": h3}, "minutes": round((time.time() - t0) / 60, 1)}
        json.dump(res, open(OUT / "results.json", "w"), ensure_ascii=False, indent=1, default=float)
        figures(site, o3pt, o3pt, o4, o4, err)
        v = res["E3"]
        L = ["# 운영 모델 (v3ops): 2015.7~2026.4 모든 자료로 보정\n",
             "계획·정책 비교·AI에 쓰는 모델이다. InSAR(2015.7~2022.12) + 관측정 수위(전 기간) + 수로 GPS(2023~2026.4)로 보정했다. "
             "GPS를 보정에 썼으므로 아래 GPS 오차는 **검증 점수가 아니라 적합도**다. 예측 성능은 검증 모델(results_v3/)로 말한다.\n",
             "| 항목 | 값 |", "|---|---|",
             f"| InSAR 적합 RMSE (2015~2022) | {v['cal'].get('insar_rmse', float('nan')):.1f} mm |",
             f"| GPS 적합 RMSE (2023~2026.4) | {v['val'].get('gps_rmse', float('nan')):.1f} mm |",
             f"| 수위 RMSE (2023~2026.4) | {v['val'].get('head_rmse', float('nan')):.2f} m |", "",
             "| 파라미터 | 사전값 | E3 (±1σ) |", "|---|---|---|"]
        for i, n in enumerate(r.NAMES):
            f = (lambda x: f"{10 ** x:.3g}") if n.startswith("log10") else (lambda x: f"{x:.2f}")
            L.append(f"| {n.replace('log10_', '')} | {f(r.PRIOR_MEAN[i])} | {f(th3[i])} (±{np.sqrt(cov3[i, i]):.2f}) |")
        L += ["", "![GPS](real-3-gps-validation.png)", f"\n계산 시간 {res['minutes']}분.\n"]
        (OUT / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
        print((OUT / "RESULTS.md").read_text(encoding="utf-8"))
        return
    th2, cov2, h2 = cal("E2", "[E2] 관측정 수위만으로 역산", use_insar=False)
    print("[E4] E3 사후분포에서 앙상블 실행"); M4, o4 = r.posterior_ensemble(site, th3, cov3, n=n_post)
    if r.V3:                                      # E2 앙상블은 시간 초과가 많아 생략(비교표에 '-')
        o2 = None
    else:
        print("[E2 앙상블] 비교용"); M2, o2 = r.posterior_ensemble(site, th2, cov2, n=n_post, seed=1)
    thh, _, _ = cal("H2019", "[E4 구조 오차] 2015.7~2019.12로 보정 → 2020~2022 InSAR 예측 오차 측정", use_insar=True, cal_end="2019-12")
    err = r.forecast_error(site, thh)
    err_all = r.forecast_error(site, thh, near_km=None)
    print("  구조 오차(수로 2km):", err, "\n  (지역 전체):", err_all)
    o2pt, o3pt = r.run_model(site, th2), r.run_model(site, th3)
    res["E2"] = r.evaluate(site, [o2pt]); res["E3"] = r.evaluate(site, [o3pt])
    res["E4"] = r.evaluate(site, o4, err=err); res["E2_ens"] = r.evaluate(site, o2) if o2 else None
    res["model_error"] = err; res["model_error_all_region"] = err_all
    res["n_members"] = {"E4": len(o4), "E2_ens": len(o2) if o2 else 0}
    np.save(OUT / "E4_members.npy", M4)
    res["params"] = {"names": r.NAMES, "prior": r.PRIOR_MEAN.tolist(), "E2": th2.tolist(), "E3": th3.tolist(),
                     "E2_sd": np.sqrt(np.diag(cov2)).tolist(), "E3_sd": np.sqrt(np.diag(cov3)).tolist()}
    res["baselines"] = baselines(site)
    print("  단순 예측(같은 GPS 채점):", res["baselines"])
    res["params"]["E3_cov"] = cov3.tolist(); res["params"]["E2_cov"] = cov2.tolist(); res["params"]["hindcast_2019"] = thh.tolist()
    res["history"] = {"E2": h2, "E3": h3}
    res["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(OUT / "results.json", "w"), ensure_ascii=False, indent=1, default=float)
    figures(site, o2pt, o3pt, o4, o2 or [o2pt], err)
    report(res)
    print((OUT / "RESULTS.md").read_text(encoding="utf-8"))


def warm_start():
    """역산 시작점: 이전 버전 E3 해(같은 이름 파라미터)를 쓰고, 새 파라미터는 사전 평균. 목적함수(사전분포 포함)는 그대로라 결과의 뜻은 같고 빨라진다."""
    for d in [r.HERE / "results_v3", r.HERE / "results"]:
        f = d / "results.json"
        if d != OUT and f.exists():
            p = json.load(open(f))["params"]
            m = dict(zip(p["names"], p["E3"]))
            print(f"  역산 시작점: {d.name}/ E3 해")
            return np.array([m.get(n, mu) for n, mu in zip(r.NAMES, r.PRIOR_MEAN)])
    return None


def baselines(site):
    """모델 없는 단순 예측을 같은 GPS 검증 자료·같은 방식(기준월 대비 변화)으로 채점.
    - 변화 0: 앞으로 더 안 꺼진다
    - 추세 연장: 2015.7~2022.12 InSAR(같은 1km 칸) 침하 속도를 그대로 연장"""
    idx = r.obs_index(site, "val", True)
    gp = idx["gp"]; d = gp[:, 4]
    months = site["months"]; t = np.array([(m.start_time - months[0].start_time).days / 365.25 for m in months])
    k_cal = list(months).index(pd.Period(r.CAL_END, "M"))
    cells = site["cells"]; u = np.where(site["use"])[0]
    rate = {}
    for iy, ix in set(map(tuple, gp[:, :2].astype(int))):
        sel = u[(cells.iy.to_numpy()[u] // r.F == iy) & (cells.ix.to_numpy()[u] // r.F == ix)]
        y = np.nanmean(site["disp"][sel][:, 1:k_cal + 1], 0)
        ok = np.isfinite(y)
        rate[(iy, ix)] = np.polyfit(t[1:k_cal + 1][ok], y[ok], 1)[0] if ok.sum() > 12 else np.nan
    pred = np.array([rate[(int(a), int(b))] * (t[int(k)] - t[int(kr)]) for a, b, k, kr in gp[:, :4]])
    ok = np.isfinite(pred)
    return {"zero_change_rmse": float(np.sqrt(np.mean(d ** 2))),
            "trend_rmse": float(np.sqrt(np.mean((pred[ok] - d[ok]) ** 2))), "n_gps": int(len(d)), "n_trend": int(ok.sum())}


ERR = None


def figures(site, o2, o3, o4, o2e, err=None):
    global ERR
    ERR = err
    months = site["months"]; t = months[1:].to_timestamp(how="end")
    gps = site["gps"]
    sts = [s for s in ["P544", "Q204", "Q208"] if s in set(gps.station)]
    fig, axes = plt.subplots(1, len(sts), figsize=(5.2 * len(sts), 4.2), squeeze=False)
    cal_end = pd.Period(r.CAL_END, "M").to_timestamp(how="end")
    cells = site["cells"]
    for ax, st in zip(axes[0], sts):
        g = gps[gps.station == st].sort_values("month")
        la, lo = g.lat.iloc[0], g.lon.iloc[0]
        iy = int((la - r.LAT0) * r.KM_LAT * 1000 // r.MCELL); ix = int((lo - r.LON0) * r.KM_LON * 1000 // r.MCELL)
        ref = g.month.iloc[0]; kr = list(months).index(ref)
        def model(o): S = o[0][:, iy, ix]; return S - (S[kr - 1] if kr > 0 else 0)
        E4 = np.array([model(o) for o in o4])
        k0 = list(months).index(pd.Period(r.CAL_END, "M"))
        lead = np.maximum((np.arange(1, len(months)) - k0) / 12.0, 0)
        sd = np.sqrt(ERR["s0_mm"] ** 2 + (ERR["b_mm_per_yr"] * lead) ** 2) if ERR else 0 * lead
        sd = np.where(np.arange(1, len(months)) > k0, sd, 0)
        blo, bhi = r.mix_band(E4, np.sqrt(sd ** 2 + np.where(sd > 0, r.SIG_INSAR ** 2, 0)))
        ax.fill_between(t, blo, bhi, color="#2166ac", alpha=0.12, label="E4 90% 예측 범위(구조 오차 포함)")
        ax.fill_between(t, np.percentile(E4, 5, 0), np.percentile(E4, 95, 0), color="#2166ac", alpha=0.25, label="E4 90% 범위(파라미터만)")
        ax.plot(t, model(o2), color="#999", lw=1.6, ls="--", label="E2 수위만(범위 밖은 잘림)")
        ax.plot(t, model(o3), color="#2166ac", lw=2, label="E3 수위+위성")
        ax.plot(g.month.dt.to_timestamp(how="end"), -(g.up_mm - g.up_mm.iloc[0]), "k.", ms=4, label="GPS 관측(검증용)")
        # InSAR(같은 1km 격자 평균)
        sel = np.where((cells.iy // r.F == iy) & (cells.ix // r.F == ix) & site["use"])[0]
        if len(sel):
            ins = np.nanmean(site["disp"][sel], 0); ins = ins - (ins[kr] if kr < len(ins) else 0)
            ax.plot(months.to_timestamp(how="end"), ins, color="#d6604d", lw=1, alpha=0.8, label="InSAR(DWR)")
        ref_vals = np.concatenate([-(g.up_mm - g.up_mm.iloc[0]).to_numpy(), model(o3), blo, bhi])
        lo_, hi_ = np.nanmin(ref_vals), np.nanmax(ref_vals); pad = 0.1 * (hi_ - lo_ + 1)
        ax.set_ylim(lo_ - pad, hi_ + pad)
        ax.axvline(cal_end, color="#555", lw=0.8, ls=":"); ax.text(cal_end, ax.get_ylim()[1], " 보정 | 검증", fontsize=8, va="top")
        ax.set_title(f"GPS {st}", fontsize=11); ax.set_ylabel("침하 (mm, 아래로 +)")
        ax.invert_yaxis() if False else None
    axes[0][0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("실제 지역: 모델 예측 vs 독립 GPS 관측 (2023년부터 검증 기간)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "real-3-gps-validation.png", dpi=150); plt.close(fig)

    # 보정 끝(2022.12) 침하 지도: 관측 vs E3
    k = list(months).index(pd.Period(r.CAL_END, "M"))
    obs = np.full((site["ny"], site["nx"]), np.nan); cnt = np.zeros_like(obs)
    u = np.where(site["use"])[0]; v = site["disp"][u, k]; ok = np.isfinite(v)
    acc = np.zeros_like(obs); np.add.at(acc, (cells.iy.to_numpy()[u][ok] // r.F, cells.ix.to_numpy()[u][ok] // r.F), v[ok])
    np.add.at(cnt, (cells.iy.to_numpy()[u][ok] // r.F, cells.ix.to_numpy()[u][ok] // r.F), 1)
    obs = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    for a, g, ttl in [(ax[0], obs, "InSAR 관측 (2015.6→2022.12)"), (ax[1], np.where(np.isfinite(obs), o3[0][k - 1], np.nan), "E3 모델"),
                      (ax[2], np.where(np.isfinite(obs), o2[0][k - 1], np.nan), "E2 모델 (위성 없이)")]:
        im = a.imshow(g, origin="lower", cmap="RdBu_r", vmin=-60, vmax=200); a.set_title(ttl, fontsize=11); a.set_xticks([]); a.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8, label="누적 침하 (mm)")
    fig.savefig(OUT / "real-4-map-fit.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fmt(v, d=1):
    return "-" if v is None else f"{v:.{d}f}"


def report(res):
    struct = ("USGS CVHM2 층 구조·지질(4개 층 묶음, 압축 계수는 CVHM2 값 × 구역별 보정 배수)" if r.STRUCTURE == "cvhm2"
              else "가정한 단순 3개 층") + (f", 모델 변형 **{r.VARIANT}**" if r.PUMPING == "et" else "")
    L = ["# 실제 검증 지역 결과 (California Aqueduct MP 195~215)\n",
         f"모델 구조: **{struct}**.\n",
         "MODFLOW 6 + CSUB 모델(1km 격자)을 실제 자료로 역산했다. 보정: 2015.7~2022.12 (InSAR + 관측정), "
         "검증: 2023.1~2026.4 (**독립 자료**: 수로 GPS 수직 변위 + 관측정 수위). "
         "2023년 이후 DWR InSAR는 GPS와 크게 어긋나 검증에 쓰지 않았다(아래 3절).\n",
         "## 1. 검증 기간 예측 오차 (관측으로 확인한 성능)\n",
         "| 방식 | GPS 침하 RMSE (mm) | 관측정 수위 RMSE (m) | GPS 90% 범위 포함률 | 수위 90% 범위 포함률 |", "|---|---|---|---|---|"]
    pct = lambda x: "-" if x is None else f"{x * 100:.0f}%"
    rows = [("E2 관측정만 (추정값 1개)", res["E2"]), ("E3 관측정 + 위성 (추정값 1개)", res["E3"]),
            ("E2 앙상블", res["E2_ens"]), ("E4 관측정 + 위성 + 불확실성(앙상블)", res["E4"])]
    for name, v in rows:
        if v is None:
            continue
        x = v["val"]
        L.append(f"| {name} | {fmt(x.get('gps_rmse'))} | {fmt(x.get('head_rmse'), 2)} | "
                 f"{fmt(x.get('gps_cov90') and x['gps_cov90'] * 100, 0)}{'%' if x.get('gps_cov90') is not None else ''} | "
                 f"{fmt(x.get('head_cov90') and x['head_cov90'] * 100, 0)}{'%' if x.get('head_cov90') is not None else ''} |")
    if res.get("baselines"):
        b = res["baselines"]
        L.append(f"| (비교) 변화 0 — 앞으로 더 안 꺼진다 | {b['zero_change_rmse']:.1f} | - | - | - |")
        L.append(f"| (비교) 2015~2022 InSAR 추세 그대로 연장 | {b['trend_rmse']:.1f} | - | - | - |")
    v = res["E3"]["val"]
    L.append(f"\nGPS {v['n_gps']}개(월 값), 수위 {v['n_head']}개. 앙상블 성공 구성원: {res.get('n_members', {})}. "
             "90% 범위 포함률은 구성원 예측에 오차를 더한 예측 분포의 5~95% 분위수로 계산했다(10/13 버전보다 정확한 방식).\n")
    if res.get("model_error"):
        me = res["model_error"]; e4 = res["E4"]["val"]
        L += ["### E4 범위에 모델 구조 오차 더하기\n",
              "2015.7~2019.12 자료로만 보정한 모델로 2020~2022 InSAR를 예측해 보고(되돌아보기 검증), 예측 기간이 길어질수록 오차가 커지는 정도를 쟀다: "
              f"오차 표준편차 ≈ √({me['s0_mm']:.1f}² + ({me['b_mm_per_yr']:.1f} × 예측 연수)²) mm. 이 오차를 E4 앙상블 범위에 더했다.\n",
              "| E4 90% 범위 | GPS 포함률 | 평균 범위 폭 |", "|---|---|---|",
              f"| 파라미터 불확실성만 | {pct(e4.get('gps_cov90'))} | - |",
              f"| + 모델 구조 오차 | **{pct(e4.get('gps_cov90_with_model_error'))}** | {e4.get('gps_band90_mean_width_mm', 0):.0f} mm |", ""]
    L += ["## 2. 보정 기간 적합도\n", "| 방식 | InSAR RMSE (mm) | 수위 RMSE (m) |", "|---|---|---|"]
    for name, k in [("E2 관측정만", "E2"), ("E3 관측정 + 위성", "E3")]:
        x = res[k]["cal"]
        ins = x.get("insar_rmse")
        L.append(f"| {name} | {fmt(ins)} | {fmt(x.get('head_rmse'), 2)} |")
    L.append("\nE2는 보정에 InSAR를 쓰지 않으므로 표의 E2 InSAR 칸은 비어 있다(지도 비교는 그림 real-4).\n")
    L += ["## 3. 추정한 땅속 성질\n", "| 파라미터 | 사전값 | E2 | E3 (±1σ) |", "|---|---|---|---|"]
    p = res["params"]
    for i, n in enumerate(p["names"]):
        f = (lambda x: f"{10 ** x:.3g}") if n.startswith("log10") else (lambda x: f"{x:.2f}")
        L.append(f"| {n.replace('log10_', '')} | {f(p['prior'][i])} | {f(p['E2'][i])} | {f(p['E3'][i])} (±{p['E3_sd'][i]:.2f}{' log' if n.startswith('log10') else ''}) |")
    if res.get("minutes"):
        L.append(f"\n계산 시간 {res['minutes']}분.\n")
    e2, e3, e4 = res["E2"]["val"], res["E3"]["val"], res["E4"]["val"]
    L += ["## 4. 해석\n",
          f"- **위성을 넣으면 실제 예측이 좋아졌다.** 검증 기간 GPS 침하 오차 {e2['gps_rmse']:.1f}mm(E2) → {e3['gps_rmse']:.1f}mm(E3), "
          f"관측정 수위 오차 {e2['head_rmse']:.2f}m → {e3['head_rmse']:.2f}m. 보정에 쓰지 않은 독립 GPS로 확인한 결과다.",
          "- E2(수위만)는 관측정이 적어(2023년 이전 장기 관측정 7곳) 땅속 성질을 정하지 못하고 사전범위 끝값으로 간다(3절 E2 열). "
          "수위는 어느 정도 맞추지만 침하 공간 분포는 전혀 맞지 않는다(그림 real-4).",
          f"- **파라미터 불확실성만으로는 E4 범위가 너무 좁다.** 90% 범위가 GPS를 {e4.get('gps_cov90', 0) * 100:.0f}%만 포함했다(가상 실험과 같은 문제). "
          + (f"되돌아보기 검증으로 잰 모델 구조 오차를 더하면 {e4['gps_cov90_with_model_error'] * 100:.0f}%가 된다." if e4.get("gps_cov90_with_model_error") is not None else ""),
          ("- v3 모델: 2005~2015 과거 취수 이력으로 시작 상태를 만들고(spin-up), 대수층 안 점토층을 지연 압밀로 계산해 "
           "'수위가 내려간 뒤 몇 년에 걸쳐 계속 꺼지는' 침하를 넣었다. 관개 회귀수 함양을 더하고, 세립질 투수계수·함양 비율의 사전범위를 현실적으로 좁혔다.\n"
           if r.V3 else
           "- E3는 수로 남쪽(Q204·Q208)의 2023년 이후 침하 속도를 실제보다 작게 예측했다. 취수 분포를 농경지 전체에 고르게 둔 가정 때문일 가능성이 크다.\n"),
          "## 5. 자료 품질: 2023년 이후 InSAR와 GPS 불일치\n",
          "- 2015~2022년에는 DWR InSAR와 GPS P544가 잘 맞는다(2022.6 기준 GPS 47mm, InSAR 38mm 침하).",
          "- 2023년 이후 InSAR는 지반이 올라가는 것으로 나오지만 GPS는 계속 침하한다(P544 2025.12: GPS 72mm, InSAR 33mm. Q204·Q208: GPS 36~50mm 침하, InSAR 30~36mm 상승).",
          "- InSAR 처리 갱신(최근 2년치 재처리) 과정의 문제로 보이며 원인은 확인하지 못했다. 그래서 2023년 이후 InSAR는 보정·검증 모두에서 뺐다.",
          "- 사업 관점: 공개 InSAR도 GPS로 교차 검증하는 **자료 품질 검사**가 서비스 기능으로 필요하다.\n",
          "## 6. 가정과 한계\n",
          ("- 층 구조와 지질은 USGS CVHM2(Traum 외, 2024)에서 가져왔다: 13개 층을 4개 묶음(상부 대수층 / 코코란 점토 / 하부 대수층 위·아래)으로 합치고, "
           "각 칸의 조립질 비율로 투수계수를, 세립질 두께와 CVHM2 압축 계수(비탄성 4.4e-4/m, 탄성 2.5e-5/m)로 압축층을 만들었다. "
           "역산은 구역별 보정 배수(3×3)와 투수계수·취수·함양만 추정한다. "
           + ("대수층 안 점토층은 지연 압밀(등가 두께 10m)로, 코코란 점토는 비지연으로 계산했다(코코란을 지연으로 두면 수렴하지 않음)." if r.V3
              else "CVHM2의 지연 배수 압축층은 비지연으로 단순화했다.")
           if r.STRUCTURE == "cvhm2" else
           "- 층 구조(상부 60m / 차수층 20m / 하부 220m, 압축층 60m)는 단순화한 가정값이다."),
          "- 관정별 취수 기록이 없어 '수로 동쪽 농경지 전체에 고르게, 관개기(4~9월)에 많이, 가뭄 해에 더' 취수한다고 두고 그 크기를 역산했다.",
          "- 함양 = 강수 × 비율(역산). 강수는 NOAA Lost Hills·Buttonwillow 관측소 월 강수(결측은 월 평년값).",
          "- 관측정 수위는 관측정별 첫 측정 대비 변화로 비교했다. 장기 관측정의 큰 계절 변동(수십 m)은 모델이 재현하지 못한다.",
          "- 유전 격자(유전 면적 50% 이상)는 역산·취수에서 뺐다."]
    (OUT / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    import sys
    if "--report-only" in sys.argv:          # 계산 없이 results.json으로 표만 다시 만들기
        report(json.load(open(OUT / "results.json")))
        print((OUT / "RESULTS.md").read_text(encoding="utf-8"))
    else:
        main(e3_only="--e3-only" in sys.argv)
