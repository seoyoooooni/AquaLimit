import unittest

import numpy as np

import future_simulator as f


class FutureSimulatorTest(unittest.TestCase):
    def test_bridge_and_month_selection(self):
        months, bridge = f.forecast_months("2027-02", 3, "2026-04")
        self.assertEqual([str(m) for m in months], ["2027-02", "2027-03", "2027-04"])
        self.assertEqual(bridge, 9)

    def test_monthly_balance_capacity_and_cumulative_risk(self):
        # The first well is cheaper but has a larger, persistent response.
        R = np.zeros((1, 2, 1, 4))
        R[0, 0, 0, :2] = [0.10, 0.01]
        R[0, 1, 0, :2] = [0.20, 0.02]
        R[0, 1, 0, 2:] = [0.10, 0.01]
        data = {"days": np.array([30.0, 31.0]), "caps": np.array([10.0, 10.0]),
                "response": R, "baseline": np.zeros((1, 2, 1))}
        out = f.optimize(data, [0.1, 0.2], 1.0, [0.75, 0.50], 2.0, 0.25)
        self.assertTrue(out["feasible"])
        self.assertTrue(np.all(out["rates"] <= data["caps"] + 1e-8))
        supplied = (out["rates"] * data["days"][:, None]).sum(1) + out["purchase"]
        np.testing.assert_allclose(supplied, out["demand"], atol=1e-6)
        self.assertTrue(np.all(out["max_extra"] <= out["limit"][None] + 1e-6))
        self.assertNotAlmostEqual(out["rates"][0, 0], out["rates"][1, 0])

    def test_infeasible_without_enough_water(self):
        data = {"days": np.array([30.0]), "caps": np.array([10.0]),
                "response": np.zeros((1, 1, 1, 1)), "baseline": np.zeros((1, 1, 1))}
        out = f.optimize(data, [0.1], 1.0, [1.2], 2.0, 0.0, purchase_fraction=0.0)
        self.assertFalse(out["feasible"])


if __name__ == "__main__":
    unittest.main()
