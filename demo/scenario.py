from dataclasses import dataclass

import numpy as np

WEEK_WEIGHTS = np.array([0.22, 0.24, 0.26, 0.28])


@dataclass
class Site:
    well_names: list
    well_xy: np.ndarray
    well_capacity: np.ndarray
    well_cost: np.ndarray
    point_names: list
    point_xy: np.ndarray
    base_response: np.ndarray


def build_site():
    well_xy = np.array([[1.0, 1.0], [4.0, 1.5], [2.0, 4.0], [5.0, 4.5]])
    point_xy = np.array([[1.5, 2.0], [4.5, 3.0], [3.0, 4.5]])
    compressibility = np.array([1.0, 0.6, 1.3, 0.5])
    dist = np.linalg.norm(point_xy[:, None, :] - well_xy[None, :, :], axis=2)
    base_response = compressibility[None, :] * np.exp(-dist / 1.5)
    return Site(
        well_names=["우물 A", "우물 B", "우물 C", "우물 D"],
        well_xy=well_xy,
        well_capacity=np.array([3000.0, 2500.0, 2000.0, 2500.0]),
        well_cost=np.array([300.0, 350.0, 250.0, 400.0]),
        point_names=["관리지점 1", "관리지점 2", "관리지점 3"],
        point_xy=point_xy,
        base_response=base_response,
    )


def sample_responses(site, n, sigma, seed):
    rng = np.random.default_rng(seed)
    factors = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=(n, 1, site.base_response.shape[1]))
    return site.base_response[None, :, :] * factors


def weekly_demand(monthly_demand):
    return monthly_demand * WEEK_WEIGHTS
