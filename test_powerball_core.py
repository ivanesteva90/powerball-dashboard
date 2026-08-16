import unittest
from itertools import combinations

import numpy as np
import pandas as pd

from powerball_core import (
    build_powerball_forecast,
    build_white_forecast,
    calculate_play_plan,
    conditional_bernoulli_inclusion_probabilities,
    conditional_bernoulli_subset_probability,
    current_matrix_draws,
    forecast_pop_intervals,
    sample_conditional_bernoulli,
    walk_forward_backtest,
)


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
        self.assertAlmostEqual(float(forecast["pop_next_draw"].sum()), 5.0, places=10)
        self.assertTrue(np.allclose(forecast["uniform_pop_next_draw"], 5 / 69))
        self.assertTrue(forecast["number"].between(1, 69).all())

    def test_zero_model_weight_returns_uniform_pop(self):
        white = build_white_forecast(self.df, strength=0.20, model_weight=0.0)
        powerball = build_powerball_forecast(self.df, strength=0.20, model_weight=0.0)
        self.assertTrue(np.allclose(white["pop_next_draw"], 5 / 69))
        self.assertTrue(np.allclose(powerball["pop_next_draw"], 1 / 26))

    def test_conditional_bernoulli_probabilities_are_exact(self):
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        inclusion = conditional_bernoulli_inclusion_probabilities(weights, sample_size=2)
        subset_probabilities = [
            conditional_bernoulli_subset_probability(weights, subset, sample_size=2)
            for subset in combinations(range(4), 2)
        ]
        self.assertAlmostEqual(float(inclusion.sum()), 2.0, places=12)
        self.assertAlmostEqual(float(sum(subset_probabilities)), 1.0, places=12)

    def test_conditional_bernoulli_sampler_matches_inclusion(self):
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        expected = conditional_bernoulli_inclusion_probabilities(weights, sample_size=2)
        rng = np.random.default_rng(987)
        counts = np.zeros(4, dtype=float)
        repetitions = 20000
        for _ in range(repetitions):
            counts[sample_conditional_bernoulli(weights, sample_size=2, rng=rng)] += 1
        self.assertTrue(np.allclose(counts / repetitions, expected, atol=0.015))

    def test_powerball_forecast_weights_sum_to_one(self):
        forecast = build_powerball_forecast(self.df, strength=0.025)
        self.assertEqual(len(forecast), 26)
        self.assertAlmostEqual(float(forecast["draw_weight"].sum()), 1.0, places=10)
        self.assertAlmostEqual(float(forecast["pop_next_draw"].sum()), 1.0, places=10)
        self.assertTrue(np.allclose(forecast["uniform_pop_next_draw"], 1 / 26))
        self.assertTrue(forecast["number"].between(1, 26).all())

    def test_walk_forward_backtest_uses_later_holdout(self):
        detail, summary, yearly, config = walk_forward_backtest(self.df)
        self.assertFalse(detail.empty)
        self.assertFalse(summary.empty)
        self.assertFalse(yearly.empty)
        self.assertGreater(config.calibration_draws, config.holdout_draws)
        self.assertEqual(pd.Timestamp(detail["draw_date"].min()), config.holdout_start)
        self.assertIn(config.white_evidence, {"Sin mejora", "Mejora incierta", "Evidencia de mejora"})
        self.assertTrue({"improvement_ci_low", "improvement_ci_high", "model_win_rate", "evidence"}.issubset(summary.columns))

    def test_forecast_intervals_cover_every_number(self):
        white, powerball = forecast_pop_intervals(
            self.df,
            white_strength=0.025,
            powerball_strength=0.025,
            white_model_weight=0.25,
            powerball_model_weight=0.25,
            repetitions=20,
            block_size=7,
        )
        self.assertEqual(len(white), 69)
        self.assertEqual(len(powerball), 26)
        self.assertTrue((white["pop_ci_low"] <= white["pop_ci_high"]).all())
        self.assertTrue((powerball["pop_ci_low"] <= powerball["pop_ci_high"]).all())

    def test_play_plan_costs_and_probability_are_exact(self):
        plan = calculate_play_plan(5, ticket_cost=2, draws_per_week=3, years=10)
        self.assertEqual(plan["total_combinations"], 292_201_338)
        self.assertAlmostEqual(plan["probability_per_draw"], 5 / 292_201_338, places=15)
        self.assertEqual(plan["cost_per_draw"], 10)
        self.assertEqual(plan["cost_per_week"], 30)
        self.assertEqual(plan["cost_per_year"], 1560)
        self.assertEqual(plan["cost_horizon"], 15600)
        self.assertGreater(plan["probability_horizon"], plan["probability_per_year"])

    def test_zero_ticket_play_plan_has_no_cost_or_probability(self):
        plan = calculate_play_plan(0, ticket_cost=2, draws_per_week=3, years=10)
        self.assertEqual(plan["probability_horizon"], 0)
        self.assertEqual(plan["cost_horizon"], 0)
        self.assertTrue(np.isinf(plan["one_in_horizon"]))


if __name__ == "__main__":
    unittest.main()
