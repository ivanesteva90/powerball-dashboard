import unittest

import numpy as np
import pandas as pd

from powerball_core import build_powerball_forecast, build_white_forecast, current_matrix_draws, walk_forward_backtest


class PowerballCoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        rows = []
        for index in range(140):
            whites = rng.choice(np.arange(1, 70), size=5, replace=False)
            rows.append(
                {
                    "draw_date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=index * 3),
                    "num1": int(whites[0]),
                    "num2": int(whites[1]),
                    "num3": int(whites[2]),
                    "num4": int(whites[3]),
                    "num5": int(whites[4]),
                    "powerball": int(rng.integers(1, 27)),
                    "white_pool_max": 69,
                    "powerball_pool_max": 26,
                }
            )
        self.df = pd.DataFrame(rows)

    def test_current_matrix_uses_latest_matrix(self):
        older = self.df.head(3).copy()
        older["white_pool_max"] = 59
        older["powerball_pool_max"] = 35
        combined = pd.concat([older, self.df], ignore_index=True)
        active = current_matrix_draws(combined)
        self.assertEqual(len(active), len(self.df))
        self.assertEqual(int(active["white_pool_max"].iloc[-1]), 69)

    def test_white_forecast_weights_sum_to_one(self):
        forecast = build_white_forecast(self.df, strength=0.025)
        self.assertEqual(len(forecast), 69)
        self.assertAlmostEqual(float(forecast["draw_weight"].sum()), 1.0, places=10)
        self.assertTrue(forecast["number"].between(1, 69).all())

    def test_powerball_forecast_weights_sum_to_one(self):
        forecast = build_powerball_forecast(self.df, strength=0.025)
        self.assertEqual(len(forecast), 26)
        self.assertAlmostEqual(float(forecast["draw_weight"].sum()), 1.0, places=10)
        self.assertTrue(forecast["number"].between(1, 26).all())

    def test_walk_forward_backtest_uses_later_holdout(self):
        detail, summary, yearly, config = walk_forward_backtest(self.df)
        self.assertFalse(detail.empty)
        self.assertFalse(summary.empty)
        self.assertFalse(yearly.empty)
        self.assertGreater(config.calibration_draws, config.holdout_draws)
        self.assertEqual(pd.Timestamp(detail["draw_date"].min()), config.holdout_start)


if __name__ == "__main__":
    unittest.main()
