"""위치 입력 AI 대리모델: '어느 칸에 우물을 놓고 하루 1,000 m³ 뽑으면 수로 38곳이 3개월 동안 얼마나 더 꺼지나'.

10/13 버전(surrogate_real.py)은 우물 6개 위치가 고정이라, 우물이 바뀌면 다시 학습해야 했다.
이 버전은 우물 위치를 입력으로 받는다 → 지도에서 아무 농경지 칸이나 골라도 바로 계산된다(데모).

입력  땅속 성질 파라미터(v3: 21개, 표준화) + 우물 위치(동서·남북 좌표, 수로까지 거리)
출력  수로 관리지점 38곳의 추가 침하(mm)와 하부 대수층 수위 변화(m), 우물 1개가 하루 1,000 m³ 뽑을 때(단위 응답)
      여러 우물은 단위 응답 × 취수량을 더한다(선형 중첩. run_plan.py의 '응답 행렬 확인'에서 MODFLOW 직접 계산과 비교).

학습 자료  땅속 상태 N_THETA개(E4 사후분포 ×2 폭에서 라틴 하이퍼큐브) × (기준 1회 + 무작위 농경지 칸 K_LOC곳에 우물 1개씩)
          → MODFLOW N_THETA × (1 + K_LOC)회. 학습 80% / 검증 20%(땅속 상태 단위로 나눔: 처음 보는 땅속 상태로 채점)
확인      ① 처음 보는 땅속 상태·위치에서 오차  ② 실제 사업자 우물 6곳에서 MODFLOW 응답 행렬(run_plan.py의 R4)과 비교
          ③ AI로 앙상블 1,000개 강건 계획 → MODFLOW 30개로 다시 확인

실행: python3 surrogate_loc.py   (기본 200 × 5 = MODFLOW 1,000회, 10코어 맥 약 20~40분)
출력: results_ai_<버전>/RESULTS_loc.md, ai_loc.pt, loc_train.npz
"""
import json
import os
import sys
import time

import numpy as np

import real_model as r
import plan_real as p
from surrogate_real import Net, lhs

OUT = r.result_dir("ai")
MODEL = r.result_dir("model", create=False)
PLAN = r.result_dir("plan", create=False)
N_THETA = int(os.environ.get("AQ_AI_THETA", 200))
K_LOC = int(os.environ.get("AQ_AI_LOC", 4))
Q0 = 1000.0                      # m3/일, 단위 응답 기준 취수량(학습 실행에서는 Q_RUN으로 뽑고 나눈다)
Q_RUN = 8000.0                   # 학습용 실행의 우물 취수량(실제 우물 1개 용량 수준)


def loc_features(site, cells):
    import validate_response as V
    d = V.aq_dist_grid(site)
    c = np.asarray(cells, float)
    return np.c_[c[:, 1] / site["nx"], c[:, 0] / site["ny"], [d[int(a), int(b)] / 10.0 for a, b in c]]


def ai_input(theta, feats):
    z = (np.asarray(theta) - r.PRIOR_MEAN) / r.PRIOR_SD
    z = np.atleast_2d(z); f = np.atleast_2d(feats)
    if len(z) == 1 and len(f) > 1:
        z = np.repeat(z, len(f), 0)
    if len(f) == 1 and len(z) > 1:
        f = np.repeat(f, len(z), 0)
    return np.c_[z, f]


def make_data(site, th3, cov3, pts, seed=0):
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    U = lhs(N_THETA, len(th3), rng)
    L = np.linalg.cholesky(cov3 * 4.0 + 1e-12 * np.eye(len(th3)))
    TH = np.clip(th3 + norm.ppf(np.clip(U, 1e-4, 1 - 1e-4)) @ L.T, r.PRIOR_MEAN - 3 * r.PRIOR_SD, r.PRIOR_MEAN + 3 * r.PRIOR_SD)
    ag = np.argwhere(site["ag"])
    jobs, keys = [], []
    for i, th in enumerate(TH):
        locs = ag[rng.choice(len(ag), K_LOC, replace=False)]
        jobs.append((th, {"months": p.MONTHS, "wells": [], "rates": np.zeros((p.MONTHS, 0)), "dry": 0.0})); keys.append((i, None))
        for c in locs:
            jobs.append((th, {"months": p.MONTHS, "wells": [tuple(c)], "rates": np.full((p.MONTHS, 1), Q_RUN), "dry": 0.0}))
            keys.append((i, (int(c[0]), int(c[1]))))
    print(f"[학습 자료] MODFLOW {len(jobs)}회 (땅속 상태 {N_THETA}개 × (기준 1 + 위치 {K_LOC}곳))", flush=True)
    outs = r.run_ensemble_plans(site, [j[0] for j in jobs], [j[1] for j in jobs])
    iy, ix = np.array(pts).T
    n0 = len(site["months"]) - 2
    base = {}
    for (i, c), o in zip(keys, outs):
        if c is None and o is not None:
            base[i] = o
    TI, C, Y = [], [], []
    for (i, c), o in zip(keys, outs):
        if c is None or o is None or i not in base:
            continue
        S, H = o; S0, H0 = base[i]
        rs = ((S[-1, iy, ix] - S[n0, iy, ix]) - (S0[-1, iy, ix] - S0[n0, iy, ix])) / Q_RUN * Q0
        rh = (H[-1, 2, iy, ix] - H0[-1, 2, iy, ix]) / Q_RUN * Q0
        TI.append(i); C.append(c); Y.append(np.r_[rs, rh])
    return TH, np.array(TI), np.array(C), np.array(Y)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import run_real  # noqa: F401  글꼴
    t0 = time.time()
    site = r.load_site()
    r.check_version(MODEL); r.check_version(OUT); r.write_version(OUT)
    res = json.load(open(MODEL / "results.json"))
    th3, cov3 = np.array(res["params"]["E3"]), np.array(res["params"]["E3_cov"])
    pts = p.mgmt_points(site); P = len(pts)
    cache = OUT / "loc_train.npz"
    if cache.exists():
        z = np.load(cache); TH, TI, C, Y = z["TH"], z["TI"], z["C"], z["Y"]
        print(f"학습 자료 {len(Y)}개 (저장된 것 사용)")
    else:
        TH, TI, C, Y = make_data(site, th3, cov3, pts)
        np.savez_compressed(cache, TH=TH, TI=TI, C=C, Y=Y)
    X = ai_input(TH[TI], loc_features(site, C))
    rng = np.random.default_rng(0)
    th_ids = rng.permutation(len(TH)); te_ids = set(th_ids[: len(TH) // 5].tolist())
    te = np.array([t in te_ids for t in TI]); tr = ~te
    print(f"[학습] 신경망: 학습 {tr.sum()}개 / 검증 {te.sum()}개(처음 보는 땅속 상태)", flush=True)
    net = Net(hidden=256, epochs=4000).fit(X[tr], Y[tr], X[te], Y[te])
    pred = net.predict(X[te])
    e = pred[:, :P] - Y[te][:, :P]
    rmse = float(np.sqrt(np.mean(e ** 2))); r2 = float(1 - np.sum(e ** 2) / np.sum((Y[te][:, :P] - Y[te][:, :P].mean()) ** 2))
    scale = float(np.mean(np.abs(Y[te][:, :P].max(1))))
    net.save(OUT / "ai_loc.pt")

    # 실제 사업자 우물 6곳: MODFLOW 응답 행렬(R4)과 비교
    wells = p.operator_wells(site); W = len(wells)
    feats_w = loc_features(site, wells)
    M4 = np.load(MODEL / "E4_members.npy")

    def ai_resp(th):
        F = net.predict(ai_input(th, feats_w))                          # (W, 2P), 1,000 m3/일당
        return {"Rs": (F[:, :P] / Q0).T, "Rh": (F[:, P:] / Q0).T}
    comp = None
    if (PLAN / "R4.npz").exists():
        Rs_mf = np.load(PLAN / "R4.npz")["Rs"]                          # (구성원, 지점, 우물), 1 m3/일당
        Rs_ai = np.array([ai_resp(th)["Rs"] for th in M4[: len(Rs_mf)]])
        d = (Rs_ai - Rs_mf) * p.CAP[None, None, :]
        comp = {"resp_rmse_mm_at_cap": float(np.sqrt(np.mean(d ** 2))),
                "resp_mean_abs_mm_at_cap": float(np.mean(np.abs(Rs_mf * p.CAP[None, None, :])))}
    # 속도
    t = time.time(); r.run_model(site, th3); t_mf = time.time() - t
    big = ai_input(th3, loc_features(site, np.argwhere(site["ag"])))
    t = time.time(); net.predict(big); t_map = time.time() - t
    # 앙상블 1,000개 강건 계획
    rng2 = np.random.default_rng(1)
    Lc = np.linalg.cholesky(cov3 + 1e-12 * np.eye(len(th3)))
    M_big = np.clip(th3 + rng2.normal(size=(1000, len(th3))) @ Lc.T, r.PRIOR_MEAN - 3 * r.PRIOR_SD, r.PRIOR_MEAN + 3 * r.PRIOR_SD)
    t = time.time(); R_big = [ai_resp(th) for th in M_big]; t_big = time.time() - t
    p.PURCHASE_CAP = p.DEMAND                     # 땅속 상태 1,000개 모두에서 지키려면 외부 구매가 한도(수요의 1/3)를 넘을 수 있다
    plan_ai = p.solve(R_big)
    out = {"n_samples": int(len(Y)), "n_theta": int(len(TH)), "k_loc": K_LOC, "n_test": int(te.sum()),
           "test_rmse_mm_per_1000": rmse, "test_r2": r2, "resp_scale_mm_per_1000": scale,
           "sec_modflow": t_mf, "sec_ai_all_cells": t_map, "n_ag_cells": int(site["ag"].sum()),
           "sec_resp_1000_ai": t_big, "sec_resp_1000_modflow_est": 1000 * (W + 1) * t_mf,
           "response_check": comp, "plan_ai_1000": plan_ai, "minutes": None}
    if comp is not None and plan_ai["feasible"]:
        R_mf = [{"Rs": Rs_mf[i]} for i in range(len(Rs_mf))]
        out["risk_ai_plan_on_modflow30"] = p.risk(plan_ai, R_mf)
        dd = Rs_ai - Rs_mf
        err_plan = np.einsum("mpw,w->mp", dd, np.array(plan_ai["rates"]))
        margin = float(1.645 * np.sqrt(np.mean(err_plan ** 2)))
        lim0 = p.LIMIT_MM; p.LIMIT_MM = lim0 - margin
        pm = p.solve(R_big); p.LIMIT_MM = lim0
        out["ai_margin_mm"] = margin; out["plan_ai_margin"] = pm
        if pm["feasible"]:
            out["risk_ai_margin_plan_on_modflow30"] = p.risk(pm, R_mf)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(out, open(OUT / "results_loc.json", "w"), ensure_ascii=False, indent=1, default=float)

    # 그림: 정확도 + 위치별 영향 지도(평균 땅속 상태)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].scatter(Y[te][:, :P].ravel(), pred[:, :P].ravel(), s=3, alpha=0.3, color="#2166ac")
    lim = [min(Y[te][:, :P].min(), pred[:, :P].min()), max(Y[te][:, :P].max(), pred[:, :P].max())]
    ax[0].plot(lim, lim, "k--", lw=1); ax[0].set_xlabel("MODFLOW (mm, 하루 1,000m³당)"); ax[0].set_ylabel("AI (mm)")
    ax[0].set_title(f"처음 보는 땅속 상태·위치 {te.sum()}개: RMSE {rmse:.3f}mm, R² {r2:.3f}", fontsize=10.5)
    cells = np.argwhere(site["ag"])
    Fm = net.predict(big)[:, :P].max(1)
    g = np.full((site["ny"], site["nx"]), np.nan); g[cells[:, 0], cells[:, 1]] = Fm
    im = ax[1].imshow(g, origin="lower", cmap="magma_r")
    iy, ix = np.array(pts).T; ax[1].plot(ix, iy, "s", ms=3, color="#2166ac", label="수로 관리지점")
    wy, wx = np.array(wells).T; ax[1].plot(wx, wy, "^", ms=8, color="#2ca02c", mec="k", label="사업자 우물(실제)")
    plt.colorbar(im, ax=ax[1], shrink=0.8, label="수로 최대 추가 침하 (mm, 하루 1,000m³·3개월)")
    ax[1].set_title(f"우물 위치별 수로 영향 지도 (AI, 농경지 {len(cells)}칸 {t_map * 1000:.0f}ms)", fontsize=10.5)
    ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "real-7-ai-loc.png", dpi=150); plt.close(fig)
    report(out)
    print((OUT / "RESULTS_loc.md").read_text(encoding="utf-8"))


def report(o):
    L = ["# 위치 입력 AI 대리모델\n",
         "입력: 땅속 성질 + **우물 위치**. 출력: 그 위치에서 하루 1,000m³를 3개월 뽑을 때 수로 38곳 추가 침하·수위 변화. "
         "우물 위치가 바뀌어도 다시 학습할 필요가 없다.\n",
         "## 1. 정확도\n", "| 항목 | 값 |", "|---|---|",
         f"| 학습 자료 | MODFLOW {o['n_samples'] + o['n_theta']}회 (땅속 상태 {o['n_theta']}개 × 위치 {o['k_loc']}곳 + 기준) |",
         f"| 처음 보는 땅속 상태·위치 오차 | {o['test_rmse_mm_per_1000']:.4f} mm (최대 응답 평균 {o['resp_scale_mm_per_1000']:.3f} mm, R² {o['test_r2']:.3f}) |"]
    if o.get("response_check"):
        c = o["response_check"]
        L.append(f"| 실제 사업자 우물 6곳, MODFLOW 응답 행렬과 차이(우물 용량 기준) | {c['resp_rmse_mm_at_cap']:.3f} mm (응답 크기 평균 {c['resp_mean_abs_mm_at_cap']:.3f} mm) |")
    L += ["\n## 2. 속도\n", "| 계산 | 시간 |", "|---|---|",
          f"| MODFLOW 1회 | {o['sec_modflow']:.1f} 초 |",
          f"| AI: 농경지 {o['n_ag_cells']}칸 전부의 영향 지도 | {o['sec_ai_all_cells'] * 1000:.0f} ms (MODFLOW로는 {o['n_ag_cells'] + 1}회 ≈ {(o['n_ag_cells'] + 1) * o['sec_modflow'] / 60:.0f}분) |",
          f"| 땅속 상태 1,000개 × 우물 6곳 응답 | AI {o['sec_resp_1000_ai']:.1f} 초 / MODFLOW 약 {o['sec_resp_1000_modflow_est'] / 3600:.1f} 시간 |",
          "\n## 3. AI로 땅속 상태 1,000개 강건 계획 → MODFLOW 30개로 확인\n"]
    pl = o["plan_ai_1000"]
    if pl["feasible"]:
        L.append(f"- AI 계획: 비용 {pl['cost'] / 1e3:,.0f}천$, 외부 구매 {pl['purchase'] / 1e3:,.0f}천 m³")
        if o.get("risk_ai_plan_on_modflow30"):
            L.append(f"- MODFLOW 확인 기준 초과 확률 {o['risk_ai_plan_on_modflow30']['exceed_prob'] * 100:.0f}%")
        pm = o.get("plan_ai_margin")
        if pm and pm.get("feasible") and o.get("risk_ai_margin_plan_on_modflow30"):
            L.append(f"- AI 오차 여유({o['ai_margin_mm']:.2f}mm)를 둔 계획: 비용 {pm['cost'] / 1e3:,.0f}천$, MODFLOW 확인 기준 초과 확률 "
                     f"**{o['risk_ai_margin_plan_on_modflow30']['exceed_prob'] * 100:.0f}%**")
    else:
        L.append(f"- 실행 불가: {pl.get('reason')}")
    L += ["\n![AI](real-7-ai-loc.png)\n", "## 4. 주의\n",
          "- AI는 MODFLOW를 흉내 낸다. 학습 범위(E4 사후분포 2배 폭, 평년 조건, 농경지 칸) 밖은 보장하지 않는다. 최종 계획은 MODFLOW로 다시 확인한다.",
          "- 여러 우물은 단위 응답을 더한다(선형 중첩). 우물이 많고 취수가 크면 MODFLOW 확인이 더 중요하다.",
          f"\n계산 시간 {o['minutes']}분.\n"]
    (OUT / "RESULTS_loc.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report(json.load(open(OUT / "results_loc.json"))); print((OUT / "RESULTS_loc.md").read_text(encoding="utf-8"))
    else:
        main()
