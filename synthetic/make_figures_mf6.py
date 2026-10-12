"""10/7(해석해 참값) vs 10/12(MODFLOW 6 참값) 기준 초과 비교 그림.

실행: python3 synthetic/make_figures_mf6.py  →  synthetic/results_mf6/fig-compare.png
두 결과 폴더의 results.json을 읽는다.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    try:
        font_manager.fontManager.addfont(p)
    except Exception:
        pass
for name in ["Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                     "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
INK = "#2b2a27"
METHODS = ["E1", "E2", "E3", "E4"]
NAMES = ["E1\n기존 방식", "E2\n관측정만", "E3\n위성+추정값 1개", "E4\n위성+불확실성"]


def exceed_counts(path, fallback):
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return [sum(t["planning"][k]["exceed"] for t in d["trials"] if t["planning"][k]["feasible"]) for k in METHODS]
    except FileNotFoundError:
        return fallback


theis = exceed_counts(HERE / "results" / "results.json", [9, 9, 7, 0])   # 10/7 RESULTS.md 값
mf6 = exceed_counts(HERE / "results_mf6" / "results.json", None)

fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(4); w = 0.38
b1 = ax.bar(x - w / 2, theis, w, color="#c9c8c2", label="10/7  정답 땅 = 교과서 공식(Theis)")
b2 = ax.bar(x + w / 2, mf6, w, color="#b2182b", label="10/12 정답 땅 = 물리모델(MODFLOW 6)")
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{int(b.get_height())}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", color=INK, fontsize=12)
ax.set_xticks(x, NAMES); ax.set_ylim(0, 10.8); ax.set_ylabel("침하 기준 초과 (가상 땅 10곳 중)")
ax.set_title("정답 땅을 물리모델로 바꾸면: 차이는 줄고, E4도 완벽하지 않다", loc="left", color=INK, fontsize=13)
ax.legend(frameon=False, fontsize=10, loc="upper right")
fig.text(0.01, 0.01, "기준: 10/7 월 6mm, 10/12 월 1mm(물리모델은 한 달 침하가 작아 실제 수로 주변 수준으로 낮춤). 모두 가상 땅 시뮬레이션 결과.",
         fontsize=8.5, color="#7a7974")
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = HERE / "results_mf6" / "fig-compare.png"
fig.savefig(out, dpi=160)
print(out)
