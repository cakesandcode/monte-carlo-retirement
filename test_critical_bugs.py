"""
Tests for critical simulation bugs.

Bug 1: FRA_TABLE birth years 1947-1951 incorrectly have graduated values.
        Per SSA, 1943-1954 ALL = 66.0.
Bug 2: IRMAA thresholds and surcharges are wrong for 2025.
Bug 3: calculate_safe_withdrawal_rate() mutates self.config (reference not copy).
Bug 3 Bonus: Binary search direction is inverted in safe withdrawal rate calc.
Bug 5: run_deterministic() applies inflation on year 0 (no year_idx > 0 guard).

Written BEFORE fixes so that tests fail, confirming each bug exists.
"""

import pytest
import copy
from config.defaults import SimulationConfig
from models.social_security import SocialSecurityModel
from models.tax import TaxCalculator
from models.simulation import MonteCarloSimulator


# ============================================================
# Bug 1: FRA Table — 1943-1954 should ALL be 66.0
# ============================================================

class TestFRATable:
    """FRA_TABLE must match SSA rules: 1943-1954 all 66.0, graduated 1955-1959."""

    def test_fra_birth_year_1947_through_1954_all_66(self):
        """Birth years 1943-1954 should ALL have FRA = 66.0 per SSA."""
        fra_table = SocialSecurityModel.FRA_TABLE
        for year in range(1943, 1955):
            assert fra_table.get(year) == 66.0, (
                f"FRA for birth year {year} should be 66.0, got {fra_table.get(year)}"
            )

    def test_fra_graduated_only_1955_to_1959(self):
        """Only 1955-1959 have graduated FRA values between 66 and 67."""
        fra_table = SocialSecurityModel.FRA_TABLE
        expected = {
            1955: 66 + 2 / 12,
            1956: 66 + 4 / 12,
            1957: 66 + 6 / 12,
            1958: 66 + 8 / 12,
            1959: 66 + 10 / 12,
        }
        for year, expected_fra in expected.items():
            actual = fra_table.get(year)
            assert actual == pytest.approx(expected_fra, abs=1e-6), (
                f"FRA for birth year {year}: expected {expected_fra}, got {actual}"
            )

    def test_fra_1960_is_67(self):
        """Birth year 1960+ should have FRA = 67."""
        assert SocialSecurityModel.FRA_TABLE.get(1960) == 67

    def test_fra_table_completeness(self):
        """FRA_TABLE covers 1943-1960."""
        fra_table = SocialSecurityModel.FRA_TABLE
        for year in range(1943, 1961):
            assert year in fra_table, f"Missing birth year {year} in FRA_TABLE"

    def test_fra_monotonic_nondecreasing(self):
        """FRA values non-decreasing across birth years."""
        fra_table = SocialSecurityModel.FRA_TABLE
        years = sorted(fra_table.keys())
        for i in range(1, len(years)):
            assert fra_table[years[i]] >= fra_table[years[i - 1]], (
                f"FRA decreases from {years[i-1]} to {years[i]}: "
                f"{fra_table[years[i-1]]} -> {fra_table[years[i]]}"
            )


# ============================================================
# Bug 2: IRMAA Thresholds — wrong for 2025
# ============================================================

class TestIRMAA:
    """IRMAA thresholds must match 2025 CMS values."""

    def test_irmaa_zero_below_first_threshold(self):
        """MAGI $105K single should have zero IRMAA (threshold is $106K)."""
        tc = TaxCalculator()
        # The first threshold should be 106000 for single
        single_thresholds = tc.IRMAA_THRESHOLDS_SINGLE
        first_threshold = single_thresholds[0][0]
        assert first_threshold == 106000, (
            f"First single IRMAA threshold should be $106,000, got ${first_threshold:,}"
        )

    def test_irmaa_has_six_tiers(self):
        """IRMAA should have 6 tiers including zero tier."""
        tc = TaxCalculator()
        assert len(tc.IRMAA_THRESHOLDS_SINGLE) == 6, (
            f"Expected 6 IRMAA tiers for single, got {len(tc.IRMAA_THRESHOLDS_SINGLE)}"
        )
        assert len(tc.IRMAA_THRESHOLDS_MFJ) == 6, (
            f"Expected 6 IRMAA tiers for MFJ, got {len(tc.IRMAA_THRESHOLDS_MFJ)}"
        )

    def test_irmaa_top_surcharge_exceeds_5000(self):
        """Top IRMAA surcharge > $5,000/yr per 2025 CMS."""
        tc = TaxCalculator()
        top_single = tc.IRMAA_THRESHOLDS_SINGLE[-1][1]
        assert top_single > 5000, (
            f"Top single IRMAA surcharge should exceed $5,000, got ${top_single:,}"
        )
        top_mfj = tc.IRMAA_THRESHOLDS_MFJ[-1][1]
        assert top_mfj > 5000, (
            f"Top MFJ IRMAA surcharge should exceed $5,000, got ${top_mfj:,}"
        )

    def test_irmaa_thresholds_ascending(self):
        """Thresholds strictly ascending."""
        tc = TaxCalculator()
        for label, thresholds in [
            ("single", tc.IRMAA_THRESHOLDS_SINGLE),
            ("MFJ", tc.IRMAA_THRESHOLDS_MFJ),
        ]:
            for i in range(1, len(thresholds)):
                assert thresholds[i][0] > thresholds[i - 1][0], (
                    f"IRMAA {label} thresholds not ascending at index {i}"
                )

    def test_irmaa_surcharges_ascending(self):
        """Surcharges non-decreasing."""
        tc = TaxCalculator()
        for label, thresholds in [
            ("single", tc.IRMAA_THRESHOLDS_SINGLE),
            ("MFJ", tc.IRMAA_THRESHOLDS_MFJ),
        ]:
            for i in range(1, len(thresholds)):
                assert thresholds[i][1] >= thresholds[i - 1][1], (
                    f"IRMAA {label} surcharges not ascending at index {i}"
                )

    def test_irmaa_mfj_greater_than_single(self):
        """MFJ thresholds should be greater than single at every tier."""
        tc = TaxCalculator()
        single = tc.IRMAA_THRESHOLDS_SINGLE
        mfj = tc.IRMAA_THRESHOLDS_MFJ
        # Compare finite thresholds only
        for i in range(min(len(single), len(mfj))):
            s_thresh = single[i][0]
            m_thresh = mfj[i][0]
            if s_thresh == float('inf') or m_thresh == float('inf'):
                continue
            assert m_thresh > s_thresh, (
                f"Tier {i}: MFJ threshold ${m_thresh:,} should exceed "
                f"single ${s_thresh:,}"
            )
            # Surcharges should be identical across filing statuses
            assert single[i][1] == mfj[i][1], (
                f"Tier {i}: surcharges should match: "
                f"single={single[i][1]}, MFJ={mfj[i][1]}"
            )


# ============================================================
# Bug 3: Config mutation in calculate_safe_withdrawal_rate()
# ============================================================

class TestConfigMutation:
    """calculate_safe_withdrawal_rate() must not mutate the original config."""

    def test_safe_withdrawal_rate_does_not_mutate_config(self):
        """Config must not change after calling calculate_safe_withdrawal_rate()."""
        config = SimulationConfig()
        config.n_simulations = 10
        original_rate = config.withdrawal_rate
        original_method = config.withdrawal_method
        original_sims = config.n_simulations

        sim = MonteCarloSimulator(config)
        sim.calculate_safe_withdrawal_rate()

        assert config.withdrawal_rate == original_rate, (
            f"withdrawal_rate mutated: {original_rate} -> {config.withdrawal_rate}"
        )
        assert config.withdrawal_method == original_method, (
            f"withdrawal_method mutated: {original_method} -> {config.withdrawal_method}"
        )
        assert config.n_simulations == original_sims, (
            f"n_simulations mutated: {original_sims} -> {config.n_simulations}"
        )

    def test_calculate_success_at_withdrawal_does_not_mutate_config(self):
        """calculate_success_at_withdrawal() must not mutate config either."""
        config = SimulationConfig()
        config.n_simulations = 10
        original_method = config.withdrawal_method
        original_withdrawal = getattr(config, 'annual_withdrawal_real', None)

        sim = MonteCarloSimulator(config)
        sim.calculate_success_at_withdrawal(50000.0)

        assert config.withdrawal_method == original_method, (
            f"withdrawal_method mutated: {original_method} -> {config.withdrawal_method}"
        )


# ============================================================
# Bug 3 Bonus: Inverted binary search direction
# ============================================================

class TestBinarySearchDirection:
    """Binary search should lower the rate when success is too low."""

    def test_binary_search_direction_correct(self):
        """When success < target, binary search should lower the rate (set high_rate)."""
        import inspect
        source = inspect.getsource(MonteCarloSimulator.calculate_safe_withdrawal_rate)
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'success < target_success' in line:
                for j in range(i + 1, min(i + 4, len(lines))):
                    stripped = lines[j].strip()
                    if not stripped or stripped.startswith('#'):
                        continue
                    assert 'high_rate' in stripped and 'mid_rate' in stripped, (
                        f"Binary search inverted: expected 'high_rate = mid_rate' after "
                        f"'success < target_success', got: {stripped}"
                    )
                    break
                break


# ============================================================
# Bug 5: Inflation timing in run_deterministic()
# ============================================================

class TestDeterministicInflation:
    """run_deterministic() must not apply inflation on year 0."""

    def test_deterministic_year0_no_inflation(self):
        """Year 0 spending should match nominal target (no inflation applied)."""
        config = SimulationConfig()
        config.inflation_method = 'fixed'
        config.fixed_inflation_rate = 0.05  # 5% — large enough to detect
        config.n_simulations = 1
        sim = MonteCarloSimulator(config)
        det = sim.run_deterministic()
        # Year 0 and year 1 spending: year 0 should be uninflated,
        # year 1 should be ~5% higher
        if 'total_spending' in det and len(det['total_spending']) >= 2:
            yr0 = det['total_spending'][0]
            yr1 = det['total_spending'][1]
            if yr0 > 0:
                ratio = yr1 / yr0
                assert 1.03 < ratio < 1.08, (
                    f"Year 1/Year 0 spending ratio should be ~1.05, got {ratio:.4f}. "
                    f"If ratio ~1.10, inflation is being double-applied."
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
