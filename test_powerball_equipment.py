import unittest

import numpy as np
import pandas as pd

from powerball_equipment import (
    build_hypothetical_weight_table,
    equipment_walk_forward_backtest,
    join_winning_draws_with_equipment,
    parse_pretest_text,
    pretest_draw_comparison,
)


class PowerballEquipmentTests(unittest.TestCase):
    def test_parser_handles_missing_power_play_and_repairs_pdf_year_typo(self):
        text = """
        07/23/16 67 65 31 16 20 11 38 15 -- 09 23 Pre-test
        07/23/16 39 05 35 07 23 11 38 11 2 09 23 Draw
        12/27/25 63 60 45 37 56 15 42 04 -- 16 29 Pre-test
        12/27/27 46 56 27 08 41 15 42 07 16 29 Pre-test
        """
        result = parse_pretest_text(text)
        self.assertEqual(len(result), 4)
        self.assertTrue(pd.isna(result.loc[result["white_1"].eq(46), "power_play"].iloc[0]))
        self.assertEqual(result["draw_date"].max(), pd.Timestamp("2025-12-27"))
        self.assertEqual(result["draw_type"].value_counts().to_dict(), {"Pre-test": 3, "Draw": 1})

    def test_join_matches_white_numbers_regardless_of_draw_order(self):
        winning = pd.DataFrame(
            [
                {
                    "draw_date": pd.Timestamp("2026-08-12"),
                    "num1": 4,
                    "num2": 26,
                    "num3": 66,
                    "num4": 67,
                    "num5": 69,
                    "powerball": 9,
                }
            ]
        )
        equipment = parse_pretest_text(
            "08/12/26 26 67 69 66 04 15 46 09 02 16 30 Draw"
        )
        merged, quality = join_winning_draws_with_equipment(winning, equipment)
        self.assertTrue(bool(merged.loc[0, "white_numbers_match"]))
        self.assertTrue(bool(merged.loc[0, "powerball_matches"]))
        quality_map = quality.set_index("metric")["value"].to_dict()
        self.assertEqual(quality_map["coverage_pct"], 100.0)

    def test_pretest_comparison_counts_actual_numbers_seen(self):
        rows = parse_pretest_text(
            """
            08/12/26 26 01 02 03 04 15 46 09 -- 16 30 Pre-test
            08/12/26 67 05 06 07 08 15 46 10 -- 16 30 Pre-test
            08/12/26 26 67 69 66 04 15 46 09 02 16 30 Draw
            """
        )
        comparison = pretest_draw_comparison(rows)
        self.assertEqual(len(comparison), 1)
        self.assertEqual(int(comparison.iloc[0]["white_draw_numbers_seen_in_pretests"]), 3)
        self.assertEqual(int(comparison.iloc[0]["powerball_seen_in_pretests"]), 1)

    def test_equipment_backtest_is_walk_forward_and_returns_evidence(self):
        rng = np.random.default_rng(2026)
        rows = []
        for index in range(180):
            whites = rng.choice(np.arange(1, 70), size=5, replace=False)
            rows.append(
                {
                    "draw_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=index * 3),
                    "white_1": int(whites[0]),
                    "white_2": int(whites[1]),
                    "white_3": int(whites[2]),
                    "white_4": int(whites[3]),
                    "white_5": int(whites[4]),
                    "white_machine_id": 13 + index % 4,
                    "white_ball_set_id": 42 + index % 5,
                    "powerball": int(rng.integers(1, 27)),
                    "powerball_machine_id": 13 + index % 4,
                    "powerball_ball_set_id": 27 + index % 5,
                    "draw_type": "Draw",
                    "white_pool_max": 69,
                    "powerball_pool_max": 26,
                }
            )
        detail, summary, yearly = equipment_walk_forward_backtest(
            pd.DataFrame(rows), warmup_draws=40, prior_draws=50
        )
        self.assertEqual(len(detail), 140)
        self.assertEqual(len(summary), 2)
        self.assertFalse(yearly.empty)
        self.assertTrue(
            set(summary["evidence"]).issubset({"Sin mejora", "Mejora incierta", "Evidencia de mejora"})
        )

    def test_hypothetical_weight_table_changes_only_selected_numbers(self):
        weights = build_hypothetical_weight_table(10, [3, 7], relative_delta_pct=-0.375)
        changed = weights.loc[weights["weight_delta_pct"].abs() > 1e-9, "number"].tolist()
        self.assertEqual(changed, [3, 7])
        self.assertAlmostEqual(float(weights.loc[weights["number"].eq(3), "weight"].iloc[0]), 79.7)


if __name__ == "__main__":
    unittest.main()
