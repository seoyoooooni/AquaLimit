"""MODFLOW 6 + CSUB 참값 대수층 (10/12 단계).

10/7 실험은 참값과 역산 모델이 둘 다 Theis 해석해였다. 이번에는 참값(정답 땅)을 물리모델로 바꾼다.
역산·계획에 쓰는 해석해 모델은 그대로 두고 '대리모델'로 취급한다. 대리모델로 만든 계획을
물리모델로 다시 돌려 기준을 지키는지 확인한다(검증 계획 3절 5번).

참값 물리모델에 새로 들어간 것 (해석해에는 없는 것)
- 2개 층: 상부 대수층(50m) + 취수 대수층(100m). 층 사이 누수.
- 경계: 상부층 바깥 경계는 수위 고정(지역 지하수면). 6km 관심 지역 바깥으로 약 7km 완충 격자.
- 압축층(interbed): 취수 대수층 안 점토층. 선행압밀수위보다 수위가 내려가면 비탄성(영구) 압축,
  그 위에서는 탄성 압축. 비탄성 저류계수 = 탄성의 10배. 그래서 과거 최저 수위까지 다시 내려가는 것은
  침하가 작고, 과거 최저보다 더 내려갈 때 침하가 크게 생긴다.
- 압축층 저류가 수위 계산에 반영된다(해석해는 침하를 수위강하 × Sk로 따로 계산).

단위: m, 일, m3/일. 침하 mm(아래로 양수, aquifer.py와 같은 부호).
"""
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import flopy

import aquifer as a

MF6 = os.environ.get("MF6_EXE") or shutil.which("mf6") or str(Path.home() / "bin" / "mf6")

CORE_CELL = 100.0
BUFFER = [150.0 * 1.5 ** k for k in range(8)]          # 바깥으로 갈수록 커지는 완충 격자 (합 약 7.4km)
TOP, BOT1, BOT2 = 0.0, -50.0, -150.0
K1, K33_1, SS1 = 10.0, 2e-4, 1e-4                       # 상부층: 누수를 제한하는 낮은 연직 투수계수
K2 = a.TRUE_T / (BOT1 - BOT2)                           # 취수층 T = 300 m2/일
SS2 = a.TRUE_S / (BOT1 - BOT2)                          # 취수층 조립질 저류 S = 0.002
IB_THICK = 20.0                                         # 압축층 두께
INELASTIC_RATIO = 10.0
PCS_OFFSET = 0.0                                        # 선행압밀수위 = 초기수위 (처음 내려가는 수위는 비탄성 압축)
NSTP = 5                                                # 월 응력기간당 6일 시간간격


def _axis():
    core = np.full(int(a.DOMAIN / CORE_CELL), CORE_CELL)
    d = np.concatenate([BUFFER[::-1], core, BUFFER])
    edges = np.concatenate([[0.0], np.cumsum(d)]) - sum(BUFFER)   # 관심 지역 원점(0,0) 기준 좌표
    return d, edges


DELR, X_EDGES = _axis()
NC = len(DELR)
CENTERS = 0.5 * (X_EDGES[:-1] + X_EDGES[1:])


def cell_of(xy):
    """관심 지역 좌표 (P,2) -> (row, col). 행은 y가 큰 쪽(북쪽)부터."""
    col = np.clip(np.searchsorted(X_EDGES, xy[:, 0], side="right") - 1, 0, NC - 1)
    rr = np.clip(np.searchsorted(X_EDGES, xy[:, 1], side="right") - 1, 0, NC - 1)
    return NC - 1 - rr, col


def cell_xy():
    gx, gy = np.meshgrid(CENTERS, CENTERS[::-1])
    return np.column_stack([gx.ravel(), gy.ravel()])


def true_sk_grid(truth):
    """참값 비탄성 Sk(무차원, = 비탄성 비저류 × 압축층 두께)를 격자에 배치. 관심 지역 밖은 가장 가까운 값."""
    xy = np.clip(cell_xy(), 0.0, a.DOMAIN)
    return truth.sk(xy).reshape(NC, NC)


def run(truth, rates, extra_days=None, extra_rate=None, ws=None, sk_grid=None):
    """참값 물리모델 실행.

    rates: (K,W) 월별 취수량 m3/일. extra_days/extra_rate: 이력 뒤에 이어지는 계획 기간(일)과 우물별 m3/일.
    반환: times (N,), head2 (N,NC,NC) 취수층 수위강하 m, sub (N,NC,NC) 누적 침하 mm.
    """
    own = ws is None
    ws = ws or tempfile.mkdtemp(prefix="aq_mf6_")
    sk = true_sk_grid(truth) if sk_grid is None else sk_grid
    perlen = [a.STEP_DAYS] * rates.shape[0]
    nstp = [NSTP] * rates.shape[0]
    if extra_days:
        perlen.append(float(extra_days)); nstp.append(4)
    sim = flopy.mf6.MFSimulation(sim_name="aq", sim_ws=ws, exe_name=MF6, verbosity_level=0)
    flopy.mf6.ModflowTdis(sim, nper=len(perlen), perioddata=[(p, n, 1.0) for p, n in zip(perlen, nstp)], time_units="days")
    flopy.mf6.ModflowIms(sim, complexity="moderate", outer_dvclose=1e-6, inner_dvclose=1e-7, linear_acceleration="bicgstab")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="aq", save_flows=False)
    flopy.mf6.ModflowGwfdis(gwf, nlay=2, nrow=NC, ncol=NC, delr=DELR, delc=DELR[::-1].copy(),
                            top=TOP, botm=[BOT1, BOT2], length_units="meters")
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=[K1, K2], k33=[K33_1, K2])
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.0, transient={0: True})   # CSUB 사용 시 STO 저류는 0, 조립질 저류는 CSUB cg_ske_cr로
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    edge = [(0, r, c) for r in range(NC) for c in range(NC) if r in (0, NC - 1) or c in (0, NC - 1)]
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data={0: [[cid, 0.0] for cid in edge]})
    wr, wc = cell_of(a.WELL_XY)
    spd = {k: [[(1, wr[i], wc[i]), -float(rates[k, i])] for i in range(len(wr))] for k in range(rates.shape[0])}
    if extra_days:
        spd[rates.shape[0]] = [[(1, wr[i], wc[i]), -float(extra_rate[i])] for i in range(len(wr))]
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=spd)
    sskv = sk / IB_THICK
    pdata = []
    n = 0
    for r in range(NC):
        for c in range(NC):
            pdata.append([n, (1, r, c), "nodelay", -PCS_OFFSET, IB_THICK, 1.0, sskv[r, c],
                          sskv[r, c] / INELASTIC_RATIO, 0.3, 999.0, 0.0])
            n += 1
    flopy.mf6.ModflowGwfcsub(gwf, head_based=True, initial_preconsolidation_head=True, ninterbeds=n,
                             cg_ske_cr=[SS1, SS2], cg_theta=0.3, packagedata=pdata,
                             compaction_filerecord="aq.cmp", boundnames=False)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="aq.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    ok, buff = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError("MODFLOW 6 실행 실패:\n" + "\n".join(buff[-20:]))
    hf = flopy.utils.HeadFile(os.path.join(ws, "aq.hds"))
    times = np.array(hf.get_times())
    head2 = -np.array([hf.get_data(totim=t)[1] for t in times])          # 수위강하(양수)
    cf = flopy.utils.HeadFile(os.path.join(ws, "aq.cmp"), text="CSUB-COMPACTION")
    sub = np.array([cf.get_data(totim=t).sum(axis=0) for t in times]) * 1000.0   # 층 합계 압축 = 지표 침하
    if own:
        shutil.rmtree(ws, ignore_errors=True)
    return times, head2, sub


def sample(times, grid, xy, at):
    """격자 시계열 (N,NC,NC)을 위치 xy, 시각 at에서 뽑기(시간 선형 보간)."""
    r, c = cell_of(xy)
    series = grid[:, r, c]                                   # (N,P)
    t = np.concatenate([[0.0], times]); s = np.vstack([np.zeros((1, series.shape[1])), series])
    return np.array([np.interp(at, t, s[:, p]) for p in range(series.shape[1])]).T


def make_observations(truth, seed=0):
    """aquifer.make_observations와 같은 형식의 관측 자료를 물리모델로 생성."""
    rng = np.random.default_rng(seed + 100)
    g = np.arange(0.0, a.DOMAIN + 1, a.INSAR_SPACING)
    gx, gy = np.meshgrid(g, g)
    insar_xy = np.column_stack([gx.ravel(), gy.ravel()])
    insar_times = np.arange(a.INSAR_REVISIT, a.N_STEPS * a.STEP_DAYS + 0.1, a.INSAR_REVISIT)
    head_times = np.arange(a.STEP_DAYS, a.N_STEPS * a.STEP_DAYS + 0.1, a.STEP_DAYS)
    rates = a.pumping_history(seed)
    times, head2, sub = run(truth, rates)
    clean_sub = sample(times, sub, insar_xy, insar_times)
    clean_head = sample(times, head2, a.OBS_WELL_XY, head_times)
    obs = a.Observations(insar_xy, insar_times, None, a.OBS_WELL_XY, head_times, None, rates)
    obs.extras["clean_sub"], obs.extras["clean_head"] = clean_sub.copy(), clean_head.copy()
    s = clean_sub + rng.normal(0, a.SIGMA_INSAR_MM, clean_sub.shape)
    s[rng.random(s.shape) < a.DECORRELATION_RATE] = np.nan
    obs.insar_mm = s
    obs.head_drawdown_m = clean_head + rng.normal(0, a.SIGMA_HEAD_M, clean_head.shape)
    return obs


def plan_subsidence(truth, rates, monthly_total, horizon_days=28.0):
    """이력 24개월 뒤 계획 기간에 우물별 총 취수량(톤)을 균등하게 뽑을 때, 관리지점의 '계획으로 인한 추가 침하'(mm).

    추가 침하 = (계획대로 뽑은 경우 침하) - (계획 기간에 하나도 안 뽑은 경우 침하), 계획 기간 끝 시점.
    비탄성 압축 때문에 선형 중첩이 정확히 성립하지 않으므로 응답 행렬 대신 물리모델을 두 번 돌린다.
    """
    q = np.asarray(monthly_total, float) / horizon_days
    t1, _, s1 = run(truth, rates, horizon_days, q)
    t0, _, s0 = run(truth, rates, horizon_days, np.zeros_like(q))
    r, c = cell_of(a.POINT_XY)
    return s1[-1][r, c] - s0[-1][r, c]
