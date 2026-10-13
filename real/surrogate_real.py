"""AI 대리모델: MODFLOW 6 + CSUB 실제 지역 모델을 흉내 내는 신경망.

목적
  계획 계산과 불확실성 평가는 물리모델을 수백~수천 번 돌려야 한다(1회 약 6초). 그래서 MODFLOW 결과로 신경망을 학습해
  같은 계산을 1ms 안팎으로 한다. 신경망은 땅속을 직접 아는 모델이 아니라 '물리모델을 빠르게 흉내 내는 모델'이다.

입력(21개)  땅속 성질 파라미터 15개(E4 사후분포 근처에서 뽑음) + 계획 우물 6개 취수량(0~용량)
출력(76개)  관리지점 38곳의 계획 기간(3개월) 침하 증가(mm, 배경 + 계획) + 하부 대수층 수위 변화(m)
            계획 때문에 생긴 부분 = f(θ, x) − f(θ, 0)

학습 자료  MODFLOW를 N번 실행(라틴 하이퍼큐브로 파라미터·취수량 뽑기). 학습 80% / 검증 20%.
신경망     다층 퍼셉트론(21 → 256 → 256 → 76, SiLU), 입력·출력 표준화, Adam
"""
import json
import time
from pathlib import Path

import numpy as np

import real_model as r
import plan_real as p

HERE = Path(__file__).resolve().parent
OUT = r.result_dir("ai")
MODEL = r.result_dir("model", create=False)
PLAN = r.result_dir("plan", create=False)


def lhs(n, d, rng):
    u = (np.argsort(rng.random((n, d)), axis=0) + rng.random((n, d))) / n
    return u


def sample_inputs(n, th_mean, cov, seed=0, widen=2.0):
    """파라미터: E4 사후분포(공분산 × widen²)에서, 취수량: 0~용량 균등(라틴 하이퍼큐브)."""
    rng = np.random.default_rng(seed)
    from scipy.stats import norm
    U = lhs(n, len(th_mean) + len(p.WELL_COST), rng)
    L = np.linalg.cholesky(cov * widen ** 2 + 1e-12 * np.eye(len(th_mean)))
    Z = norm.ppf(np.clip(U[:, :len(th_mean)], 1e-4, 1 - 1e-4))
    TH = np.clip(th_mean + Z @ L.T, r.PRIOR_MEAN - 3 * r.PRIOR_SD, r.PRIOR_MEAN + 3 * r.PRIOR_SD)
    X = U[:, len(th_mean):] * p.CAP
    X[: n // 10] = 0.0                                  # 10%는 계획 취수 0(배경만) — 차이 계산 정확도용
    return TH, X


def simulate(site, TH, X, wells, pts, workers=None):
    plans = [{"months": p.MONTHS, "wells": wells, "rates": np.repeat(x[None], p.MONTHS, 0), "dry": 0.0} for x in X]
    outs = r.run_ensemble_plans(site, list(TH), plans, workers)
    iy, ix = np.array(pts).T
    n0 = len(site["months"]) - 2
    Y, ok = [], []
    for i, o in enumerate(outs):
        if o is None:
            continue
        S, H = o
        Y.append(np.concatenate([S[-1, iy, ix] - S[n0, iy, ix], H[-1, 2, iy, ix] - H[n0, 2, iy, ix]]))
        ok.append(i)
    return np.array(ok), np.array(Y)


class Net:
    """표준화 + MLP. torch가 없으면 sklearn MLPRegressor로 대신한다."""

    def __init__(self, hidden=256, epochs=3000, lr=2e-3, seed=0):
        self.h, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed

    def fit(self, Xin, Y, Xv=None, Yv=None, log=print):
        self.mx, self.sx = Xin.mean(0), Xin.std(0) + 1e-9
        self.my, self.sy = Y.mean(0), Y.std(0) + 1e-9
        a, b = (Xin - self.mx) / self.sx, (Y - self.my) / self.sy
        try:
            import torch
            torch.manual_seed(self.seed)
            self.torch = True
            self.m = torch.nn.Sequential(torch.nn.Linear(a.shape[1], self.h), torch.nn.SiLU(),
                                         torch.nn.Linear(self.h, self.h), torch.nn.SiLU(),
                                         torch.nn.Linear(self.h, b.shape[1]))
            opt = torch.optim.Adam(self.m.parameters(), lr=self.lr, weight_decay=1e-6)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
            A, B = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
            self.hist = []
            for ep in range(self.epochs):
                opt.zero_grad(); loss = ((self.m(A) - B) ** 2).mean(); loss.backward(); opt.step(); sch.step()
                if (ep + 1) % 500 == 0:
                    msg = f"  학습 {ep + 1}/{self.epochs}: 학습 오차 {loss.item():.4f}"
                    if Xv is not None:
                        e = self.predict(Xv) - Yv
                        msg += f", 검증 침하 RMSE {np.sqrt(np.mean(e[:, :Y.shape[1] // 2] ** 2)):.3f} mm"
                    log(msg); self.hist.append(float(loss.item()))
        except ImportError:
            from sklearn.neural_network import MLPRegressor
            self.torch = False
            self.m = MLPRegressor((self.h, self.h), activation="relu", max_iter=self.epochs, random_state=self.seed).fit(a, b)
        return self

    def save(self, path):
        """학습한 신경망과 표준화 값을 저장(torch 사용 시)."""
        if getattr(self, "torch", False):
            import torch
            torch.save({"state": self.m.state_dict(), "mx": self.mx, "sx": self.sx, "my": self.my, "sy": self.sy,
                        "hidden": self.h, "n_in": len(self.mx), "n_out": len(self.my)}, path)

    @classmethod
    def load(cls, path):
        import torch
        d = torch.load(path, weights_only=False)
        n = cls(hidden=d["hidden"]); n.torch = True
        n.mx, n.sx, n.my, n.sy = d["mx"], d["sx"], d["my"], d["sy"]
        n.m = torch.nn.Sequential(torch.nn.Linear(d["n_in"], n.h), torch.nn.SiLU(), torch.nn.Linear(n.h, n.h), torch.nn.SiLU(),
                                  torch.nn.Linear(n.h, d["n_out"]))
        n.m.load_state_dict(d["state"]); n.m.eval()
        return n

    def predict(self, Xin):
        a = (np.atleast_2d(Xin) - self.mx) / self.sx
        if self.torch:
            import torch
            with torch.no_grad():
                b = self.m(torch.tensor(a, dtype=torch.float32)).numpy()
        else:
            b = self.m.predict(a)
        return b * self.sy + self.my


# ---------------------------------------------------------------- 실행
def main(n_train=600, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import run_real  # noqa: F401  (한글 글꼴 설정)
    t0 = time.time()
    site = r.load_site()
    r.check_version(MODEL); r.check_version(OUT); r.write_version(OUT)
    res = json.load(open(MODEL / "results.json"))
    th3, cov3 = np.array(res["params"]["E3"]), np.array(res["params"]["E3_cov"])
    pts, wells = p.mgmt_points(site), p.operator_wells(site)
    P = len(pts)

    cache = OUT / "train_data.npz"
    if cache.exists():
        z = np.load(cache); TH, X, Y = z["TH"], z["X"], z["Y"]
        if TH.shape[1] != len(r.NAMES):
            raise RuntimeError(f"저장된 학습 자료는 파라미터 {TH.shape[1]}개 버전입니다(지금 {len(r.NAMES)}개). "
                               f"{cache} 를 지우고 다시 실행하거나 AQ_PUMPING을 맞추세요.")
        print(f"학습 자료 {len(Y)}개 (저장된 것 사용)")
    else:
        print(f"[학습 자료] MODFLOW {n_train}회 실행 (파라미터: E4 사후분포 ×2 폭, 취수량: 0~용량)")
        TH, X = sample_inputs(n_train, th3, cov3, seed=seed)
        ok, Y = simulate(site, TH, X, wells, pts)
        TH, X = TH[ok], X[ok]
        np.savez_compressed(cache, TH=TH, X=X, Y=Y)
        print(f"  성공 {len(Y)}/{n_train}, {(time.time() - t0) / 60:.1f}분")
    t = time.time(); r.run_model(site, th3, plan={"months": p.MONTHS, "wells": wells, "rates": np.zeros((p.MONTHS, len(wells))), "dry": 0.0})
    t_mf = time.time() - t                                         # MODFLOW 1회(순차 실행) 시간

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Y)); ntr = int(0.8 * len(Y))
    tr, te = idx[:ntr], idx[ntr:]
    Xin = np.c_[TH, X / p.CAP]
    print("[학습] 신경망")
    net = Net().fit(Xin[tr], Y[tr], Xin[te], Y[te])
    pred = net.predict(Xin[te])
    e_sub = pred[:, :P] - Y[te][:, :P]; e_h = pred[:, P:] - Y[te][:, P:]
    rmse_sub = float(np.sqrt(np.mean(e_sub ** 2))); rmse_h = float(np.sqrt(np.mean(e_h ** 2)))
    r2 = float(1 - np.sum(e_sub ** 2) / np.sum((Y[te][:, :P] - Y[te][:, :P].mean()) ** 2))
    t = time.time(); [net.predict(Xin[te][:1]) for _ in range(200)]; t_ai = (time.time() - t) / 200
    t = time.time(); net.predict(np.repeat(Xin[te][:1], 7000, 0)); t_batch = (time.time() - t) / 7000

    # AI로 만든 응답 행렬 vs MODFLOW 응답 행렬(run_plan의 E4 구성원)
    M4 = np.load(MODEL / "E4_members.npy")
    def ai_resp(th):
        base = np.r_[th, np.zeros(len(wells))]
        Z = np.array([base] + [np.r_[th, np.eye(len(wells))[w]] for w in range(len(wells))])
        F = net.predict(Z)
        return {"Rs": ((F[1:, :P] - F[0, :P]) / p.CAP).T, "Rh": ((F[1:, P:] - F[0, P:]) / p.CAP).T}
    comp = None
    plan_res = PLAN / "results.json"
    if (PLAN / "R4.npz").exists():
        z = np.load(PLAN / "R4.npz")
        Rs_mf = z["Rs"]                                             # (구성원, 지점, 우물)
        Rs_ai = np.array([ai_resp(th)["Rs"] for th in M4[: len(Rs_mf)]])
        d = (Rs_ai - Rs_mf) * p.CAP
        comp = {"resp_rmse_mm_at_cap": float(np.sqrt(np.mean(d ** 2))),
                "resp_mean_abs_mm_at_cap": float(np.mean(np.abs(Rs_mf * p.CAP)))}

    # AI로 앙상블 1,000개 강건 계획
    print("[AI 활용] E4 앙상블 1,000개로 강건 계획")
    rng2 = np.random.default_rng(1)
    L = np.linalg.cholesky(cov3 + 1e-12 * np.eye(len(th3)))
    M_big = np.clip(th3 + rng2.normal(size=(1000, len(th3))) @ L.T, r.PRIOR_MEAN - 3 * r.PRIOR_SD, r.PRIOR_MEAN + 3 * r.PRIOR_SD)
    t = time.time()
    R_big = [ai_resp(th) for th in M_big]
    t_resp_ai = time.time() - t
    np.savez_compressed(OUT / "R_ai1000.npz", Rs=np.array([x["Rs"] for x in R_big]).astype(np.float32),
                        Rh=np.array([x["Rh"] for x in R_big]).astype(np.float32))
    net.save(OUT / "ai_model.pt")
    plan_ai = p.solve(R_big)
    risk_ai_on_ai = p.risk(plan_ai, R_big) if plan_ai["feasible"] else None
    risk_ai_on_mf = None; plan_ai_m = None; risk_ai_m_on_mf = None; margin = None
    if comp is not None and plan_ai["feasible"]:
        R_mf = [{"Rs": Rs_mf[i]} for i in range(len(Rs_mf))]
        risk_ai_on_mf = p.risk(plan_ai, R_mf)
        # AI 오차만큼 여유를 두고 다시 계획: 기준 - 1.645 × (AI 응답 오차, 계획 규모로 환산)
        d = (Rs_ai - Rs_mf)                                          # (구성원, 지점, 우물), 1 m3/일당
        x = np.array(plan_ai["rates"])
        err_plan = np.einsum("mpw,w->mp", d, x)                       # 그 계획에서 AI−MODFLOW 침하 차이
        margin = float(1.645 * np.sqrt(np.mean(err_plan ** 2)))
        lim0 = p.LIMIT_MM; p.LIMIT_MM = lim0 - margin
        plan_ai_m = p.solve(R_big); p.LIMIT_MM = lim0
        if plan_ai_m["feasible"]:
            risk_ai_m_on_mf = p.risk(plan_ai_m, R_mf)

    out = {"n_samples": int(len(Y)), "n_train": int(ntr), "n_test": int(len(te)),
           "test_rmse_sub_mm": rmse_sub, "test_rmse_head_m": rmse_h, "test_r2_sub": r2,
           "sec_per_modflow": t_mf, "sec_per_ai_single": t_ai, "sec_per_ai_batch": t_batch,
           "speedup_single": t_mf / t_ai, "speedup_batch": t_mf / t_batch,
           "response_check": comp, "plan_ai_1000": plan_ai, "risk_ai_plan_on_ai1000": risk_ai_on_ai,
           "risk_ai_plan_on_modflow30": risk_ai_on_mf, "ai_margin_mm": margin, "plan_ai_margin": plan_ai_m,
           "risk_ai_margin_plan_on_modflow30": risk_ai_m_on_mf, "sec_responses_1000_ai": t_resp_ai,
           "sec_responses_1000_modflow_est": 1000 * 7 * t_mf, "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(out, open(OUT / "results.json", "w"), ensure_ascii=False, indent=1, default=float)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(Y[te][:, :P].ravel(), pred[:, :P].ravel(), s=3, alpha=0.3, color="#2166ac")
    lim = [min(Y[te][:, :P].min(), pred[:, :P].min()), max(Y[te][:, :P].max(), pred[:, :P].max())]
    ax[0].plot(lim, lim, "k--", lw=1); ax[0].set_xlabel("MODFLOW 계산 (mm)"); ax[0].set_ylabel("AI 예측 (mm)")
    ax[0].set_title(f"3개월 침하: 검증 자료 {len(te)}개 (RMSE {rmse_sub:.2f}mm, R² {r2:.3f})", fontsize=11)
    names = ["MODFLOW 1회", "AI 1회", "AI (묶음 처리, 1건당)"]
    vals = [t_mf, t_ai, t_batch]
    ax[1].barh(names, vals, color=["#999", "#2166ac", "#2166ac"]); ax[1].set_xscale("log")
    for i, v in enumerate(vals):
        ax[1].text(v, i, f"  {v * 1000:.2f} ms" if v < 1 else f"  {v:.1f} s", va="center", fontsize=10)
    ax[1].set_title(f"계산 시간 (AI가 약 {t_mf / t_batch:,.0f}배 빠름, 묶음 기준)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "real-6-ai.png", dpi=150); plt.close(fig)
    report(out)
    print((OUT / "RESULTS.md").read_text(encoding="utf-8"))


def report(o):
    L = ["# AI 대리모델 결과\n",
         "실제 지역 MODFLOW 6 + CSUB 모델(CVHM2 구조)을 흉내 내는 신경망. 입력: 땅속 성질 15개 + 계획 우물 6개 취수량, "
         "출력: 수로 관리지점 38곳의 3개월 침하 증가와 수위 변화.\n",
         "## 1. 정확도 (학습에 쓰지 않은 MODFLOW 결과와 비교)\n",
         "| 항목 | 값 |", "|---|---|",
         f"| 학습 자료 | MODFLOW {o['n_samples']}회 (학습 {o['n_train']} / 검증 {o['n_test']}) |",
         f"| 침하 예측 오차(RMSE) | {o['test_rmse_sub_mm']:.3f} mm |",
         f"| 침하 설명력(R²) | {o['test_r2_sub']:.4f} |",
         f"| 수위 예측 오차(RMSE) | {o['test_rmse_head_m']:.4f} m |"]
    if o["response_check"]:
        c = o["response_check"]
        L.append(f"| 계획 응답(우물 1개 최대 취수 시 추가 침하) 오차 | {c['resp_rmse_mm_at_cap']:.3f} mm (응답 크기 평균 {c['resp_mean_abs_mm_at_cap']:.2f} mm) |")
    L += ["\n## 2. 속도\n", "| 계산 | 시간 |", "|---|---|",
          f"| MODFLOW 1회 (순차 실행) | {o['sec_per_modflow']:.1f} 초 |",
          f"| AI 1회 | {o['sec_per_ai_single'] * 1000:.2f} ms (약 {o['speedup_single']:,.0f}배) |",
          f"| AI 묶음 처리(1건당) | {o['sec_per_ai_batch'] * 1e6:.1f} μs (약 {o['speedup_batch']:,.0f}배) |",
          f"| 앙상블 1,000개 응답 행렬 | AI {o['sec_responses_1000_ai']:.1f} 초 / MODFLOW였다면 약 {o['sec_responses_1000_modflow_est'] / 3600:.1f} 시간 |",
          "\n## 3. AI로 앙상블 1,000개 강건 계획\n"]
    pl = o["plan_ai_1000"]
    if pl["feasible"]:
        L.append(f"- 비용 {pl['cost'] / 1e3:,.0f}천$, 외부 구매 {pl['purchase'] / 1e3:,.0f}천 m³, 우물별 평균 취수(m³/일): "
                 + ", ".join(f"{chr(65 + i)} {v:,.0f}" for i, v in enumerate(pl["rates"])))
        if o["risk_ai_plan_on_modflow30"]:
            rk = o["risk_ai_plan_on_modflow30"]
            L.append(f"- 이 계획을 **MODFLOW 응답(E4 구성원 30개)**으로 다시 확인: 기준 초과 확률 {rk['exceed_prob'] * 100:.0f}%, 최대 추가 침하 평균 {rk['worst_mean_mm']:.2f} mm")
        pm = o.get("plan_ai_margin")
        if pm and pm["feasible"] and o.get("risk_ai_margin_plan_on_modflow30"):
            rk2 = o["risk_ai_margin_plan_on_modflow30"]
            L.append(f"- AI 오차가 있어서 기준선에 딱 맞춘 계획은 MODFLOW에서 조금 넘을 수 있다. 그래서 AI 오차(계획 규모 환산 {o['ai_margin_mm']:.2f} mm, 90% 수준)만큼 "
                     f"기준을 낮춰 다시 계획: 비용 {pm['cost'] / 1e3:,.0f}천$, MODFLOW 확인 기준 초과 확률 **{rk2['exceed_prob'] * 100:.0f}%**.")
    else:
        L.append(f"- 실행 불가: {pl['reason']}")
    L += ["\n![AI](real-6-ai.png)\n", "## 4. 주의\n",
          "- AI는 땅속을 직접 아는 모델이 아니라 MODFLOW를 빠르게 흉내 내는 모델이다. 학습 범위(E4 사후분포의 2배 폭, 취수 0~용량, 평년 강수) 밖에서는 정확도를 보장하지 않는다.",
          "- 그래서 AI로 고른 최종 계획은 MODFLOW로 다시 확인한다(위 3절).",
          f"\n계산 시간 {o['minutes']}분(학습 자료 생성 포함).\n"]
    (OUT / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
