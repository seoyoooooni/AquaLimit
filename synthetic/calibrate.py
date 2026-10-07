"""역산(모델 보정).

- E2 지상 관측만: 관측정 수위로 T, S 보정. 압축성 Sk는 문헌 평균값(0.02) 하나.
- E3 위성 + 단일 추정: 수위 + InSAR로 T, S와 화소별 Sk를 한 번 추정.
- E4 위성 + 불확실성: E3와 같은 모델을 Randomized Maximum Likelihood로 30번 보정한 앙상블.

침하는 주어진 T, S에서 Sk에 선형이므로, 화소별 Sk는 닫힌 식(사전분포 포함 선형 회귀)으로 풀고
T, S만 비선형 최소제곱으로 찾는다(variable projection).
"""
import copy
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

import aquifer as a

SK_PRIOR_MEAN = a.PRIOR_SK
SK_PRIOR_SD = 0.02


@dataclass
class Model:
    log_ts: np.ndarray          # [log10 T, log10 S]
    sk_px: np.ndarray           # InSAR 화소별 Sk (E2는 상수 배열)
    px_xy: np.ndarray

    @property
    def theta(self):
        return np.concatenate([self.log_ts, np.zeros(4)])   # Sk 자리는 sk_fn이 대신함

    def sk(self, xy):
        d = np.linalg.norm(xy[:, None, :] - self.px_xy[None, :, :], axis=2)
        return self.sk_px[d.argmin(axis=1)]


def subset(obs, until_day):
    o = copy.copy(obs)
    mi, mh = obs.insar_times <= until_day, obs.head_times <= until_day
    o.insar_times, o.insar_mm = obs.insar_times[mi], obs.insar_mm[mi]
    o.head_times, o.head_drawdown_m = obs.head_times[mh], obs.head_drawdown_m[mh]
    return o


def _pixel_sk(s_px, insar, valid, prior_mean):
    g = np.where(valid, s_px * 1000.0, 0.0)
    y = np.where(valid, insar, 0.0)
    w, wp = 1 / a.SIGMA_INSAR_MM**2, 1 / SK_PRIOR_SD**2
    sk = ((g * y).sum(0) * w + prior_mean * wp) / ((g * g).sum(0) * w + wp)
    return np.maximum(sk, 1e-5)


def _fit(obs, use_insar, x0, ts_center, sk_center=SK_PRIOR_MEAN, insar=None, head=None):
    insar = obs.insar_mm if insar is None else insar
    head = obs.head_drawdown_m if head is None else head
    valid = ~np.isnan(insar)

    def solve_inner(x):
        T, S = 10 ** x[0], 10 ** x[1]
        hd = a.drawdown(obs.head_xy, obs.head_times, obs.rates, T, S)
        if not use_insar:
            return hd, None, None
        s_px = a.drawdown(obs.insar_xy, obs.insar_times, obs.rates, T, S)
        sk = _pixel_sk(s_px, insar, valid, sk_center)
        return hd, s_px, sk

    def resid(x):
        hd, s_px, sk = solve_inner(x)
        r = [((hd - head) / a.SIGMA_HEAD_M).ravel(), (x - ts_center) / a.PRIOR_LOG_SD[:2]]
        if use_insar:
            r.append(((s_px * sk * 1000.0)[valid] - insar[valid]) / a.SIGMA_INSAR_MM)
        return np.concatenate(r)

    x = least_squares(resid, x0, method="trf").x
    sk = solve_inner(x)[2] if use_insar else np.full(len(obs.insar_xy), a.PRIOR_SK)
    return Model(x, sk, obs.insar_xy)


def calibrate_e2(obs):
    pm = a.prior_mean()[:2]
    return _fit(obs, False, pm, pm)


def calibrate_e3(obs):
    pm = a.prior_mean()[:2]
    return _fit(obs, True, pm, pm)


def calibrate_e4(obs, n_members=30, seed=0):
    rng = np.random.default_rng(seed)
    pm, sd = a.prior_mean()[:2], a.PRIOR_LOG_SD[:2]
    members = []
    for _ in range(n_members):
        members.append(_fit(
            obs, True,
            x0=pm + rng.normal(0, sd),
            ts_center=pm + rng.normal(0, sd),
            sk_center=SK_PRIOR_MEAN + rng.normal(0, SK_PRIOR_SD),
            insar=obs.insar_mm + rng.normal(0, a.SIGMA_INSAR_MM, obs.insar_mm.shape),
            head=obs.head_drawdown_m + rng.normal(0, a.SIGMA_HEAD_M, obs.head_drawdown_m.shape),
        ))
    return members


def calibrate_all(obs, n_members=30, seed=0):
    cal = subset(obs, a.SPLIT_DAY)
    return {"E2": calibrate_e2(cal), "E3": calibrate_e3(cal), "E4": calibrate_e4(cal, n_members, seed)}
