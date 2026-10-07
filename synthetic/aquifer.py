"""합성 대수층: 참값 파라미터를 알고 있는 가상 지역과 관측 자료 생성.

물리 가정 (MODFLOW 6·CSUB로 교체하기 전 단계의 해석해 모델)
- 수위강하: 균질·무한 피압대수층 Theis 해. 취수량은 월별 계단 함수, 시간 중첩.
- 침하: 압축층 침하 = 골격 저류계수(Sk) x 수위강하. 탄성·선형 가정, 지연 배수 없음.
- 역산 모델은 Sk를 구역 4개의 상수로 보지만, 참값 대수층은 구역 안에서도 Sk가 공간적으로
  변한다(로그정규 불균질장, 상관거리 약 1 km). 생성 모델과 역산 모델이 같으면 복원이
  지나치게 쉬워지므로(inverse crime) 의도적으로 모델 오차를 넣었다.
- InSAR: 시선방향(LOS) 대신 수직 변위를 직접 관측한다고 가정 (수평 변위 무시).
단위: 거리 m, 시간 일, 취수량 m3/일 (1 m3 = 1톤으로 취급), 침하 mm, 수위 m.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.special import exp1

DOMAIN = 6000.0
ZONE_SPLIT = 3000.0
ZONE_NAMES = ["남서", "남동", "북서", "북동"]
WELL_NAMES = ["우물 A", "우물 B", "우물 C", "우물 D"]
WELL_XY = np.array([[1000.0, 1000.0], [4000.0, 1500.0], [2000.0, 4000.0], [5000.0, 4500.0]])
POINT_NAMES = ["관리지점 1", "관리지점 2", "관리지점 3"]
POINT_XY = np.array([[1500.0, 2000.0], [4500.0, 3000.0], [3000.0, 4500.0]])
OBS_WELL_XY = np.array([[3000.0, 3000.0], [1000.0, 3500.0], [5000.0, 2000.0], [2500.0, 1000.0]])

STEP_DAYS = 30.0
N_STEPS = 24                 # 24개월 취수 이력
SPLIT_DAY = 540.0            # 0~540일 보정, 540~720일 검증
INSAR_REVISIT = 12.0         # Sentinel-1 재방문 주기
INSAR_SPACING = 500.0
R_MIN = 150.0                # 화소 평균 효과: 우물과 겹친 화소의 특이점 방지

SIGMA_INSAR_MM = 3.0
SIGMA_HEAD_M = 0.05
DECORRELATION_RATE = 0.10    # 결맞음 저하로 빠지는 화소 비율

# 참값. 파라미터 벡터 = [log10 T, log10 S, log10 Sk(구역 4개)]
TRUE_T = 300.0
TRUE_S = 2e-3
TRUE_SK = np.array([0.030, 0.018, 0.040, 0.012])
PRIOR_SK = 0.02              # 문헌 기반 평균값 (지상 관측만 쓸 때 사용)
PRIOR_LOG_SD = np.array([0.5, 0.5, 0.3, 0.3, 0.3, 0.3])
HETERO_SD = 0.25             # 참값 Sk 불균질의 로그 표준편차
HETERO_CORR_M = 1000.0
FIELD_CELL = 100.0


def true_theta():
    return np.concatenate([[np.log10(TRUE_T), np.log10(TRUE_S)], np.log10(TRUE_SK)])


def prior_mean():
    return np.concatenate([[np.log10(500.0), np.log10(1e-3)], np.full(4, np.log10(PRIOR_SK))])


def zone_of(xy):
    return (xy[:, 0] >= ZONE_SPLIT).astype(int) + 2 * (xy[:, 1] >= ZONE_SPLIT).astype(int)


def pumping_history(seed=0):
    """월별 우물 취수량 (m3/일), 계절성 + 임의 변동."""
    rng = np.random.default_rng(seed)
    base = np.array([250.0, 200.0, 180.0, 220.0])
    month = np.arange(N_STEPS)
    season = 1.0 + 0.35 * np.sin(2 * np.pi * (month - 3) / 12)
    noise = rng.uniform(0.8, 1.2, size=(N_STEPS, len(base)))
    return base[None, :] * season[:, None] * noise


def drawdown(xy, times, rates, T, S):
    """관측 위치 xy (P,2), 시각 times (N,), 월별 취수량 rates (K,W) -> 수위강하 (N,P) m."""
    r = np.linalg.norm(xy[:, None, :] - WELL_XY[None, :, :], axis=2)
    r = np.maximum(r, R_MIN)                                       # (P,W)
    d_rates = np.diff(np.vstack([np.zeros((1, rates.shape[1])), rates]), axis=0)  # (K,W)
    t_start = np.arange(rates.shape[0]) * STEP_DAYS                 # (K,)
    dt = times[:, None] - t_start[None, :]                          # (N,K)
    active = dt > 0
    dt_safe = np.where(active, dt, 1.0)
    u = (r[None, None, :, :] ** 2) * S / (4.0 * T * dt_safe[:, :, None, None])  # (N,K,P,W)
    w = np.where(active[:, :, None, None], exp1(u), 0.0)
    return np.einsum("nkpw,kw->np", w, d_rates) / (4.0 * np.pi * T)


@dataclass
class Observations:
    insar_xy: np.ndarray
    insar_times: np.ndarray
    insar_mm: np.ndarray        # (N,P), 결맞음 저하는 nan
    head_xy: np.ndarray
    head_times: np.ndarray
    head_drawdown_m: np.ndarray  # (N,P)
    rates: np.ndarray
    extras: dict = field(default_factory=dict)


def zone_sk(theta):
    sk = 10 ** np.asarray(theta)[2:]
    return lambda xy: sk[zone_of(xy)]


@dataclass
class Truth:
    theta: np.ndarray
    log_field: np.ndarray  # 100 m 격자 위 Sk 곱셈 계수(자연로그)

    def sk(self, xy):
        idx = np.clip(np.round(xy / FIELD_CELL).astype(int), 0, self.log_field.shape[0] - 1)
        return zone_sk(self.theta)(xy) * np.exp(self.log_field[idx[:, 1], idx[:, 0]])


def make_truth(seed=0):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed + 200)
    n = int(DOMAIN / FIELD_CELL) + 1
    f = gaussian_filter(rng.normal(size=(n, n)), HETERO_CORR_M / FIELD_CELL / 2, mode="wrap")
    f = (f - f.mean()) / f.std() * HETERO_SD
    return Truth(true_theta(), f)


def forward(theta, obs_like, sk_fn=None):
    """파라미터로 InSAR 침하(mm)와 관측정 수위강하(m)를 계산. sk_fn이 있으면 위치별 Sk 사용."""
    T, S = 10 ** theta[0], 10 ** theta[1]
    sk_fn = sk_fn or zone_sk(theta)
    s_px = drawdown(obs_like.insar_xy, obs_like.insar_times, obs_like.rates, T, S)
    sub = s_px * sk_fn(obs_like.insar_xy)[None, :] * 1000.0
    head = drawdown(obs_like.head_xy, obs_like.head_times, obs_like.rates, T, S)
    return sub, head


def make_observations(truth, seed=0):
    rng = np.random.default_rng(seed + 100)
    g = np.arange(0.0, DOMAIN + 1, INSAR_SPACING)
    gx, gy = np.meshgrid(g, g)
    insar_xy = np.column_stack([gx.ravel(), gy.ravel()])
    insar_times = np.arange(INSAR_REVISIT, N_STEPS * STEP_DAYS + 0.1, INSAR_REVISIT)
    head_times = np.arange(STEP_DAYS, N_STEPS * STEP_DAYS + 0.1, STEP_DAYS)
    obs = Observations(insar_xy, insar_times, None, OBS_WELL_XY, head_times, None, pumping_history(seed))
    sub, head = forward(truth.theta, obs, truth.sk)
    obs.extras["clean_sub"], obs.extras["clean_head"] = sub.copy(), head.copy()
    sub = sub + rng.normal(0, SIGMA_INSAR_MM, sub.shape)
    sub[rng.random(sub.shape) < DECORRELATION_RATE] = np.nan
    obs.insar_mm = sub
    obs.head_drawdown_m = head + rng.normal(0, SIGMA_HEAD_M, head.shape)
    return obs


def response_matrix(theta, horizon_days=28.0, sk_fn=None):
    """계획 기간 동안 우물 i에서 1,000톤을 균등 취수할 때 관리지점 j의 추가 침하 (mm). (P,W)"""
    T, S = 10 ** theta[0], 10 ** theta[1]
    sk_at_points = (sk_fn or zone_sk(theta))(POINT_XY)
    rate = np.eye(len(WELL_NAMES)) * 1000.0 / horizon_days      # 우물별 단위 취수
    out = np.zeros((len(POINT_NAMES), len(WELL_NAMES)))
    for i in range(len(WELL_NAMES)):
        s = drawdown(POINT_XY, np.array([horizon_days]), rate[i:i + 1], T, S)[0]
        out[:, i] = s * sk_at_points * 1000.0
    return out
