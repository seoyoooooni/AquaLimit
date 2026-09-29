import numpy as np


def subsidence(plan, responses):
    total_pumping = plan.pumping.sum(axis=0)
    return responses @ (total_pumping / 1000.0)


def summarize(plan, responses, limit_mm):
    s = subsidence(plan, responses)
    worst = s.max(axis=1)
    return {
        "worst_by_scenario": worst,
        "exceed_rate": float((worst > limit_mm + 1e-9).mean()),
        "worst_p50": float(np.median(worst)),
        "worst_max": float(worst.max()),
    }
