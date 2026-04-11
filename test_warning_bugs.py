"""
Tests for 7 warning-level simulation bugs.

Bug 1: Dead code — duplicate retirement allocation check in glide path.
Bug 2: Glide path hardcodes 35-year career instead of using actual career span.
Bug 3: Healthcare deducted proportionally from Roth (should use ordered draw).
Bug 4: Senior deduction wrong for single filers ($1,550 should be $1,950).
Bug 5: LTCG stacking bug — remaining_fifteen computed incorrectly.
Bug 6: RMD uses current-year balance (should use prior-year-end balance).
Bug 7: ss_fra typed as int (should be float to support graduated FRA like 66.5).

Written BEFORE fixes so that tests fail, confirming each bug exists.
"""

import inspect
import pytest
import numpy as np
from config.defaults import SimulationConfig, ASSET_CLASSES
from models.portfolio import PortfolioMechanics
from models.tax import TaxCalculator


# ============================================================
# Bug 1: Dead code — duplicate retirement allocation check
# ============================================================

class TestBug1DuplicateRetirementCheck:
    """Lines 199-203 of portfolio.py have a duplicate `if current_age >= retirement_age` check."""

    def test_no_duplicate_retirement_check_in_glide_path(self):
        """The glide path method should not have a duplicate retirement allocation check."""
        source = inspect.getsource(PortfolioMechanics.calculate_glide_path_allocation)
        # Count occurrences of the retirement age check pattern
        count = source.count('current_age >= retirement_age')
        assert count == 1, (
            f"Found {count} occurrences of 'current_age >= retirement_age' in "
            f"calculate_glide_path_allocation — expected exactly 1 (duplicate dead code exists)"
        )


# ============================================================
# Bug 2: Glide path hardcodes 35-year career
# ============================================================

class TestBug2GlidePathHardcoded:
    """Glide path uses `retirement_age - (retirement_age - 35)` = always 35 years."""

    def test_different_career_start_gives_different_glide(self):
        """
        A person who starts working at 22 with retirement at 65 (43-year career)
        should have a different glide allocation at age 45 than someone who starts
        at 30 (35-year career) — but the hardcoded 35 makes them identical.
        After fix, calculate_glide_path_allocation should accept career_start_age.
        """
        pre_alloc = {ac: (0.90 if ac == 'us_large_cap' else 0.10 / 7)
                     for ac in ASSET_CLASSES}
        ret_alloc = {ac: (0.40 if ac == 'us_large_cap' else 0.60 / 7)
                     for ac in ASSET_CLASSES}

        # After fix: should accept career_start_age parameter
        # Person A: career_start_age=22, retirement=65 → 43-year career
        result_a = PortfolioMechanics.calculate_glide_path_allocation(
            current_age=45,
            retirement_age=65,
            pre_allocation=pre_alloc,
            retirement_allocation=ret_alloc,
            career_start_age=22,
        )

        # Person B: career_start_age=35, retirement=65 → 30-year career
        result_b = PortfolioMechanics.calculate_glide_path_allocation(
            current_age=45,
            retirement_age=65,
            pre_allocation=pre_alloc,
            retirement_allocation=ret_alloc,
            career_start_age=35,
        )

        # They should differ because career lengths differ
        diff = abs(result_a['us_large_cap'] - result_b['us_large_cap'])
        assert diff > 0.01, (
            f"Glide allocations should differ for different career starts, "
            f"but us_large_cap diff = {diff:.6f}"
        )

    def test_glide_path_signature_has_career_start_age(self):
        """After fix, calculate_glide_path_allocation should accept career_start_age."""
        sig = inspect.signature(PortfolioMechanics.calculate_glide_path_allocation)
        assert 'career_start_age' in sig.parameters, (
            "calculate_glide_path_allocation should have a career_start_age parameter"
        )


# ============================================================
# Bug 3: Healthcare deducted proportionally from Roth
# ============================================================

class TestBug3HealthcareFromRoth:
    """Healthcare costs should use ordered draw (taxable -> trad -> roth), not proportional."""

    def test_healthcare_does_not_touch_roth_when_others_sufficient(self):
        """Healthcare should use ordered draw, not proportional from Roth."""
        import inspect
        from models.simulation import MonteCarloSimulator
        source = inspect.getsource(MonteCarloSimulator.run)
        # The buggy pattern is: roth_bal -= roth_bal * _hc_frac
        # The fixed pattern draws from taxable first, then traditional, then roth
        assert 'roth_bal    -= roth_bal * _hc_frac' not in source, (
            "Healthcare is still deducted proportionally from Roth. "
            "Should use ordered: taxable -> traditional -> roth."
        )


# ============================================================
# Bug 4: Senior deduction wrong for single filers
# ============================================================

class TestBug4SeniorDeduction:
    """
    IRS 2025: Single filer 65+ gets $1,950 additional standard deduction.
    MFJ 65+ gets $1,550 per qualifying spouse.
    The code uses $1,550 for both — wrong for single.
    """

    def test_single_senior_deduction_is_1950(self):
        """Single filer 65+ should get $1,950 additional deduction, not $1,550."""
        tc = TaxCalculator()
        # Tax on income just above standard deduction to see the senior bonus.
        # Single std ded = 15000.  With age 65+: should be 15000 + 1950 = 16950.
        # Income = 16950 should yield $0 tax (exactly at deduction).
        # Income = 16550 (wrong deduction) would also yield $0.
        # Income = 17000 should yield tax on only $50 at 10% = $5.

        # If bug exists: deduction = 15000 + 1550 = 16550
        # Tax on 16950 with wrong deduction = (16950-16550)*0.10 = $40
        # If fixed: deduction = 15000 + 1950 = 16950
        # Tax on 16950 = $0
        tax = tc.calculate_federal_income_tax(16950, 'single', 66)
        assert tax == pytest.approx(0.0, abs=1.0), (
            f"Tax on $16,950 for single age 66 should be ~$0 (deduction = $16,950), "
            f"got ${tax:.2f}. Senior deduction likely $1,550 instead of $1,950."
        )

    def test_mfj_senior_deduction_is_1550_per_spouse(self):
        """MFJ 65+ should get $1,550 per qualifying spouse (unchanged)."""
        tc = TaxCalculator()
        # MFJ std ded = 30000. Both 65+: 30000 + 1550 + 1550 = 33100.
        tax = tc.calculate_federal_income_tax(33100, 'married_filing_jointly', 66, spouse_age=66)
        assert tax == pytest.approx(0.0, abs=1.0), (
            f"Tax on $33,100 for MFJ both 65+ should be ~$0, got ${tax:.2f}"
        )


# ============================================================
# Bug 5: LTCG stacking bug
# ============================================================

class TestBug5LTCGStacking:
    """
    remaining_fifteen should be:
      max(0, fifteen_threshold - ordinary_income - ltcg_in_zero)
    NOT:
      max(0, fifteen_threshold - max(ordinary_income, zero_threshold) - ltcg_in_zero)

    The max(ordinary_income, zero_threshold) inflates the "used" space when
    ordinary_income < zero_threshold, incorrectly shrinking the 15% band.
    """

    def test_ltcg_stacking_low_ordinary_income_mfj(self):
        """
        MFJ: zero_threshold=96950, fifteen_threshold=583750.
        If ordinary_income=50000 and LTCG=500000:
          - ltcg_in_zero = min(500000, 96950-50000) = 46950
          - CORRECT remaining_fifteen = 583750 - 50000 - 46950 = 486800
          - BUG remaining_fifteen = 583750 - max(50000,96950) - 46950 = 439850
          - ltcg_in_fifteen: CORRECT=453050, BUG=439850
          - ltcg_in_twenty: CORRECT=0, BUG=13200
          - BUG adds 20% tax on 13200 = $2,640 extra tax
        """
        tc = TaxCalculator()
        tax = tc.calculate_ltcg_tax(500000, 50000, 'married_filing_jointly')

        # Correct: all LTCG fits in 0% + 15% bands
        # ltcg_in_zero = 46950 (at 0%) = $0
        # ltcg_in_fifteen = 453050 (at 15%) = $67,957.50
        # ltcg_in_twenty = 0
        expected = 46950 * 0.0 + 453050 * 0.15 + 0 * 0.20
        assert tax == pytest.approx(expected, abs=10.0), (
            f"LTCG tax should be ${expected:,.0f}, got ${tax:,.0f}. "
            f"Stacking formula likely uses max(ordinary, zero_threshold)."
        )

    def test_ltcg_stacking_single(self):
        """
        Single: zero_threshold=47025, fifteen_threshold=518900.
        ordinary_income=20000, LTCG=480000.
          - ltcg_in_zero = min(480000, 47025-20000) = 27025
          - CORRECT remaining_fifteen = 518900 - 20000 - 27025 = 471875
          - BUG remaining_fifteen = 518900 - max(20000,47025) - 27025 = 444850
        """
        tc = TaxCalculator()
        tax = tc.calculate_ltcg_tax(480000, 20000, 'single')

        ltcg_in_zero = 27025
        correct_remaining_fifteen = 518900 - 20000 - 27025  # = 471875
        ltcg_in_fifteen = min(480000 - ltcg_in_zero, correct_remaining_fifteen)
        ltcg_in_twenty = max(0, 480000 - ltcg_in_zero - ltcg_in_fifteen)
        expected = ltcg_in_zero * 0.0 + ltcg_in_fifteen * 0.15 + ltcg_in_twenty * 0.20

        assert tax == pytest.approx(expected, abs=10.0), (
            f"Single LTCG tax should be ${expected:,.0f}, got ${tax:,.0f}. "
            f"Stacking formula likely uses max(ordinary, zero_threshold)."
        )


# ============================================================
# Bug 6: RMD uses current-year balance (should use prior-year-end)
# ============================================================

class TestBug6RMDCurrentYear:
    """
    IRS rules: RMD is based on Dec 31 balance of PRIOR year.
    Current code calls calculate_rmd(age, traditional) AFTER adding
    current-year contributions (Step 1 adds contributions, Step 2 computes RMD),
    so the RMD incorrectly includes this year's contributions in the balance.

    The fix: capture traditional balance BEFORE contributions, use that for RMD.
    """

    def test_rmd_invariant_to_current_year_contributions(self):
        """
        RMD at age 75 should be based on the balance PASSED IN (prior-year-end),
        not the balance after current-year contributions are added.

        Scenario: retired person age 75, trad balance = 500k.
        A spouse is contributing (not retired yet), adding 50k to traditional.
        RMD should be 500k / 24.6, NOT 540k / 24.6.
        """
        pm = PortfolioMechanics()
        returns = np.zeros(len(ASSET_CLASSES))

        # Config with SPOUSE contributions (primary is retired, spouse is not)
        config_spouse_contrib = SimulationConfig(
            current_age=74,
            retirement_age=65,  # Primary already retired
            life_expectancy=95,
            traditional_balance=500000.0,
            roth_balance=0.0,
            taxable_balance=0.0,
            total_portfolio_value=500000.0,
            annual_contribution=0.0,  # Primary not contributing (retired)
            annual_withdrawal_real=0.0,
            withdrawal_start_age=80,
            use_glide_path=False,
            expense_ratio=0.0,
            advisory_fee=0.0,
            spend_annual_real=0.0,
            inflation_mean=0.0,
            # Spouse is younger, still contributing
            spouse_age=60,
        )
        # Set spouse contribution fields via setattr (not in dataclass)
        config_spouse_contrib.spouse_retirement_age = 65
        config_spouse_contrib.spouse_annual_contribution = 50000.0
        config_spouse_contrib.spouse_contribution_growth_rate = 0.0

        result_with_spouse_contrib = pm.process_year(
            age=75,
            traditional=500000.0,
            roth=0.0,
            taxable=0.0,
            annual_returns=returns,
            inflation_rate=0.0,
            cum_inflation=1.0,
            config=config_spouse_contrib,
            is_retired=True,
        )

        # Config with NO contributions at all
        config_no_contrib = SimulationConfig(
            current_age=74,
            retirement_age=65,
            life_expectancy=95,
            traditional_balance=500000.0,
            roth_balance=0.0,
            taxable_balance=0.0,
            total_portfolio_value=500000.0,
            annual_contribution=0.0,
            annual_withdrawal_real=0.0,
            withdrawal_start_age=80,
            use_glide_path=False,
            expense_ratio=0.0,
            advisory_fee=0.0,
            spend_annual_real=0.0,
            inflation_mean=0.0,
        )

        result_no_contrib = pm.process_year(
            age=75,
            traditional=500000.0,
            roth=0.0,
            taxable=0.0,
            annual_returns=returns,
            inflation_rate=0.0,
            cum_inflation=1.0,
            config=config_no_contrib,
            is_retired=True,
        )

        rmd_no = result_no_contrib['rmd_amount']
        rmd_with = result_with_spouse_contrib['rmd_amount']

        # Both should yield same RMD: 500k / 24.6 = 20325.20
        expected_rmd = 500000.0 / 24.6
        assert rmd_no == pytest.approx(expected_rmd, rel=0.001), (
            f"RMD without contrib should be {expected_rmd:.2f}, got {rmd_no:.2f}"
        )
        assert rmd_with == pytest.approx(expected_rmd, rel=0.001), (
            f"RMD with spouse contrib should be {expected_rmd:.2f} (prior-year balance), "
            f"got {rmd_with:.2f}. Bug: RMD includes current-year contributions."
        )


# ============================================================
# Bug 7: ss_fra: int = 67 should be float
# ============================================================

class TestBug7SSFraType:
    """ss_fra should be float to support graduated FRA values like 66.5."""

    def test_ss_fra_default_is_float(self):
        """ss_fra default should be float (67.0), not int (67)."""
        config = SimulationConfig()
        assert isinstance(config.ss_fra, float), (
            f"ss_fra should be float, got {type(config.ss_fra).__name__}"
        )

    def test_ss_fra_accepts_fractional_values(self):
        """ss_fra should accept fractional values like 66.5 without truncation."""
        config = SimulationConfig(ss_fra=66.5)
        assert config.ss_fra == 66.5, (
            f"ss_fra should be 66.5, got {config.ss_fra}"
        )
        assert isinstance(config.ss_fra, float), (
            f"ss_fra should remain float, got {type(config.ss_fra).__name__}"
        )

    def test_ss_fra_type_annotation_is_float(self):
        """The type annotation for ss_fra should be float, not int."""
        import typing
        hints = typing.get_type_hints(SimulationConfig)
        assert hints['ss_fra'] is float, (
            f"ss_fra type annotation should be float, got {hints['ss_fra']}"
        )
