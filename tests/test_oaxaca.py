"""Minimal regression tests for Oaxaca-Blinder decomposition outputs.

The aim is to guard the business-centred presentation behaviour:
    * Row-level gap columns remain intuitive (rate/mix decomposition) while
      still summing mathematically to the topline gap.
    * Supporting tables expose the new Net/Rate/Mix split without breaking
      previous consumers that expect percent-point strings.

The tests operate with the public API (`run_oaxaca_analysis`) so they can be
invoked either via `pytest` or simply `python tests/test_oaxaca.py`.
"""

import unittest

import numpy as np
import pandas as pd

from rca_package.oaxaca_blinder import run_oaxaca_analysis


class OaxacaDecompositionTests(unittest.TestCase):
    """Focused checks on decomposition math and presentation columns."""

    @classmethod
    def setUpClass(cls):
        # Two-category dummy portfolio with mix and rate differences.
        data = pd.DataFrame(
            [
                {"region": "TARGET", "category": "A", "wins": 45, "total": 100},  # 45%
                {"region": "TARGET", "category": "B", "wins": 90, "total": 300},  # 30%
                {"region": "BASE", "category": "A", "wins": 200, "total": 400},  # 50%
                {"region": "BASE", "category": "B", "wins": 50, "total": 200},   # 25%
            ]
        )

        cls.result = run_oaxaca_analysis(
            df=data,
            region_column="region",
            numerator_column="wins",
            denominator_column="total",
            category_columns=["category"],
        )

    def test_row_net_gap_sums_to_topline(self):
        """Row-level net gaps must aggregate exactly to the regional topline gap."""
        region_rows = self.result.decomposition_df[self.result.decomposition_df["region"] == "TARGET"]
        topline = self.result.regional_gaps.loc[
            self.result.regional_gaps["region"] == "TARGET", "total_net_gap"
        ].iloc[0]

        self.assertAlmostEqual(region_rows["net_gap"].sum(), topline, places=12)

    def test_component_sums_match_net_gap(self):
        """Rate + mix contributions should equal the pooled net gap per row."""
        region_rows = self.result.decomposition_df[self.result.decomposition_df["region"] == "TARGET"]
        component_sum = (
            region_rows["display_rate_contribution"]
            + region_rows["display_mix_contribution"]
        )

        diff = region_rows["net_display"] - component_sum
        self.assertLess(np.abs(diff).max(), 1e-12)

    def test_supporting_table_exposes_net_rate_mix(self):
        """Narrative supporting tables should contain the new Net/Rate/Mix columns."""
        decision = self.result.narrative_decisions["TARGET"]
        cols = decision.supporting_table_df.columns.tolist()
        for required in ["Net Impact_pp", "Rate Impact_pp", "Mix Impact_pp"]:
            self.assertIn(required, cols)

    def test_mix_for_display_includes_directional_adjustment(self):
        """Displayed mix impact should reconcile net minus rate for every row."""
        rows = self.result.decomposition_df[self.result.decomposition_df["region"] == "TARGET"]
        mix_display = rows["display_mix_contribution"]
        rate_display = rows["display_rate_contribution"]
        net = rows["net_display"]
        residual = (mix_display + rate_display) - net
        self.assertLess(np.abs(residual).max(), 1e-12)

    def test_summary_top_contributors_uses_net_gap(self):
        """Formatted summary output should include net gap percent-point strings."""
        stats = self.result.get_summary_stats(format_for_display=True)
        top = stats["top_category_contributors"]
        for col in ["net_gap_pp", "mix_gap_pp", "rate_gap_pp"]:
            self.assertIn(col, top.columns)
        # Every value should end with the pp suffix to stay presentation friendly
        self.assertTrue(top["net_gap_pp"].str.endswith("pp").all())


if __name__ == "__main__":
    unittest.main()
