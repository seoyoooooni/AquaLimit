"""발표용 그림 생성: docs/progress/1007/images/*.png

실행: python3 synthetic/make_figures.py  (run_experiment.py 실행 후, results/trials 캐시 사용)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aquifer as a  # noqa: E402

OUT = HERE.parent / "docs" / "progress" / "1007" / "images"
OUT.mkdir(parents=True, exist_ok=True)

for name in ["Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "Noto Sans CJK JP"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({
    "axes.unicode_minus": False, "font.size": 12,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0"
COLOR = {"E1": "#b4b3ad", "E2": "#8d8c86", "E3": "#eb6834", "E4": "#2a78d6"}
LABEL = {"E1": "E1 기존 계획", "E2": "E2 수위만", "E3": "E3 위성\n단일 추정", "E4": "E4 위성\n+ 불확실성"}
NOTE = "※ 가상 자료(합성 대수층 10개) 기준"


def load_trials(members=30):
    files = sorted((HERE / "results" / "trials").glob(f"seed*_m{members}.json"),
                   key=lambda p: int(p.stem.split("_")[0][4:]))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def fig_setup():
    truth = a.make_truth(0)
    n = truth.log_field.shape[0]
    g = np.arange(n) * a.FIELD_CELL
    gx, gy = np.meshgrid(g, g)
    sk = truth.sk(np.column_stack([gx.ravel(), gy.ravel()])).reshape(n, n)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(sk, origin="lower", extent=[0, 6, 0, 6], cmap="Blues", vmin=0, vmax=sk.max())
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("정답 압축성 Sk (진할수록 잘 꺼짐)", color=INK2)
    cb.outline.set_visible(False)
    px = np.arange(0, 6.01, 0.5)
    pxx, pyy = np.meshgrid(px, px)
    ax.scatter(pxx, pyy, s=4, c=MUTED, label="위성 화소 (500m 간격)")
    ax.axhline(3, color="white", lw=1, ls="--")
    ax.axvline(3, color="white", lw=1, ls="--")
    w, p, o = a.WELL_XY / 1000, a.POINT_XY / 1000, a.OBS_WELL_XY / 1000
    ax.scatter(w[:, 0], w[:, 1], s=140, marker="o", c="#fcfcfb", edgecolors=INK, linewidths=2, label="취수 우물", zorder=3)
    ax.scatter(p[:, 0], p[:, 1], s=150, marker="^", c="#eb6834", edgecolors="#fcfcfb", linewidths=2, label="침하 관리지점", zorder=3)
    ax.scatter(o[:, 0], o[:, 1], s=90, marker="s", c=INK2, edgecolors="#fcfcfb", linewidths=2, label="수위 관측정", zorder=3)
    for (x, y), t in zip(w, "ABCD"):
        ax.annotate(t, (x, y), xytext=(9, 6), textcoords="offset points", fontsize=12, color=INK, weight="bold")
    for (x, y), t in zip(p, "123"):
        ax.annotate(t, (x, y), xytext=(9, -14), textcoords="offset points", fontsize=12, color=INK, weight="bold")
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.set_title("가상 지역과 정답 압축성 지도 (예시 1개)", color=INK, loc="left", fontsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=4, frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig1-setup.png", dpi=200)
    plt.close(fig)


def _bars(ax, keys, vals, fmt, title, ylabel):
    x = np.arange(len(keys))
    ax.bar(x, vals, width=0.55, color=[COLOR[k] for k in keys])
    for xi, v in zip(x, vals):
        ax.annotate(fmt(v), (xi, v), xytext=(0, 4), textcoords="offset points", ha="center", color=INK, fontsize=12)
    ax.set_xticks(x, [LABEL[k] for k in keys])
    ax.set_title(title, color=INK, loc="left", fontsize=13)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)


def fig_satellite(trials):
    keys = ["E2", "E3", "E4"]
    err = [np.mean([t["recovery"][k]["Sk_px"] for t in trials]) * 100 for k in keys]
    rmse = [np.mean([t["prediction"][k][0] for t in trials]) for k in keys]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    _bars(axs[0], keys, err, lambda v: f"{v:.1f}%", "땅이 얼마나 잘 꺼지는지(압축성) 추정 오차", "오차 중앙값 (%)")
    _bars(axs[1], keys, rmse, lambda v: f"{v:.1f}mm", "6개월 뒤 침하 예측 오차", "RMSE (mm)")
    axs[1].axhline(a.SIGMA_INSAR_MM, color=INK2, ls="--", lw=1)
    axs[1].annotate("위성 관측\n잡음 3mm", (0.5, a.SIGMA_INSAR_MM), xytext=(0, -6), textcoords="offset points",
                    ha="center", va="top", color=INK2, fontsize=10)
    fig.text(0.99, 0.01, NOTE, ha="right", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig2-satellite.png", dpi=200)
    plt.close(fig)


def fig_robust(trials, limit=6.0):
    keys = ["E1", "E2", "E3", "E4"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    rng = np.random.default_rng(0)
    for i, k in enumerate(keys):
        ys = np.array([t["planning"][k]["true_worst_mm"] for t in trials])
        xs = i + rng.uniform(-0.15, 0.15, len(ys))
        ax.scatter(xs, ys, s=60, c=COLOR[k], edgecolors="#fcfcfb", linewidths=2, zorder=3)
        n_ex = int(sum(t["planning"][k]["exceed"] for t in trials))
        ax.annotate(f"초과 {n_ex}/{len(trials)}", (i, ys.max()), xytext=(0, 10), textcoords="offset points",
                    ha="center", color=INK, fontsize=12, weight="bold")
    ax.axhline(limit, color=INK, ls="--", lw=1.2)
    ax.annotate("침하 기준 6mm", (1.5, limit), xytext=(0, 5), textcoords="offset points", ha="center", color=INK, fontsize=10)
    ax.set_xticks(range(4), [LABEL[k] for k in keys])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("정답 대수층에서 실제 최대 침하 (mm/월)")
    ax.set_title("각 계획을 정답 땅에 적용했을 때 (점 하나 = 가상 지역 하나)", color=INK, loc="left", fontsize=13)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_ylim(top=ax.get_ylim()[1] + 0.8)
    cost = [np.mean([t["planning"][k]["cost"] for t in trials]) / 1e4 for k in keys]
    _bars(axs[1], keys, cost, lambda v: f"{v:,.0f}만원", "평균 총비용", "만원")
    fig.text(0.99, 0.01, NOTE + " · 계획은 실행하지 않은 시뮬레이션 결과", ha="right", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig3-robust.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    trials = load_trials()
    fig_setup()
    fig_satellite(trials)
    fig_robust(trials)
    print("saved:", sorted(p.name for p in OUT.glob("*.png")))
