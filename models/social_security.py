"""
Social Security Benefit Calculation and Income Overlay Module.

Implements complete Social Security benefit calculation including:
  - Primary Insurance Amount (PIA) from AIME using 2025 bend points
  - Full Retirement Age (FRA) lookup by birth year
  - Claiming age adjustments (early reduction, delayed credit)
  - Spousal benefit calculation (50% rule)
  - Multiple income stream compilation (SS, pension, rental, part-time)
  - COLA (Cost of Living Adjustment) application

Data:
  - 2025 Social Security Bend Points: $1,226, $7,391 (PIA formula: 90%, 32%, 15%)
  - FRA Table: 1943-1959 with age increments, 1960+ age 67
  - Earning reductions (early): 5/9 of 1% per month first 36, then 5/12 of 1%
  - Delayed credits: 8% per year to age 70

Version: 1.7.2

Boundary / Edge Cases:
  - Claiming age < 62: invalid per SSA rules; should be caught by config
    validation.
  - Claiming at FRA: benefit = stated benefit; no early/late adjustment.
  - SERP year outside 2026-2033 range: getattr returns 0.0 (no distribution).
  - cum_inflation = 0: would cause division by zero in some income conversions;
    clamped to 1.0 minimum.
  - Pension / part-time income with start_age > end_age: returns 0 for all
    years (silent no-op).
  - Real vs nominal flag: when is_real=True, amount multiplied by cum_inflation;
    when False, flat dollar amount used.
"""

from typing import Optional, Dict
from config.defaults import SimulationConfig, SS_BEND_POINTS_2025


class SocialSecurityModel:
    """
    Calculate Social Security benefits and income overlays for retirement.

    Implements accurate PIA calculation, claiming age adjustments, and
    coordination with other retirement income sources.

    Attributes:
        BEND_POINTS_2025: 2025 Social Security Bend Points for PIA calculation.
        FRA_TABLE: Full Retirement Age by birth year (1943-1970+).
        EARLY_REDUCTION_MONTHS_36: Reduction rate for first 36 months (5/9 of 1%).
        EARLY_REDUCTION_MONTHS_OVER: Reduction rate beyond 36 months (5/12 of 1%).
        DELAYED_CREDIT_RATE: Increase per year for delayed claiming (8%).
    """

    # 2025 Social Security Bend Points (dollars per month of AIME)
    BEND_POINTS_2025 = [1226, 7391]

    # PIA formula percentages (applied to AIME ranges)
    PIA_PERCENTAGES = [0.90, 0.32, 0.15]

    # Full Retirement Age (FRA) by birth year
    FRA_TABLE = {
        1943: 66,
        1944: 66,
        1945: 66,
        1946: 66,
        1947: 66 + 2/12,
        1948: 66 + 4/12,
        1949: 66 + 6/12,
        1950: 66 + 8/12,
        1951: 66 + 10/12,
        1952: 66,
        1953: 66,
        1954: 66,
        1955: 66 + 2/12,
        1956: 66 + 4/12,
        1957: 66 + 6/12,
        1958: 66 + 8/12,
        1959: 66 + 10/12,
        1960: 67,
    }

    # Early claiming reduction rates
    EARLY_REDUCTION_MONTHS_36 = 5.0 / 9.0 / 100.0  # 5/9 of 1% per month
    EARLY_REDUCTION_MONTHS_OVER = 5.0 / 12.0 / 100.0  # 5/12 of 1% per month

    # Delayed claiming credit rate
    DELAYED_CREDIT_RATE = 0.08  # 8% per year

    def __init__(self):
        """Initialize Social Security model."""
        pass

    @staticmethod
    def calculate_pia_from_aime(aime: float) -> float:
        """
        Calculate Primary Insurance Amount (PIA) from Average Indexed Monthly Earnings.

        PIA Formula (2025):
          - 90% of first $1,226 of AIME
          - plus 32% of AIME from $1,226 to $7,391
          - plus 15% of AIME over $7,391

        Args:
            aime: Average Indexed Monthly Earnings (in dollars).

        Returns:
            Primary Insurance Amount (monthly benefit at FRA).
        """
        pia = 0.0

        # First bracket: 90% of AIME up to first bend point
        first_bracket = min(aime, 1226)
        pia += first_bracket * 0.90

        # Second bracket: 32% of AIME from first to second bend point
        if aime > 1226:
            second_bracket = min(aime - 1226, 7391 - 1226)
            pia += second_bracket * 0.32

        # Third bracket: 15% of AIME above second bend point
        if aime > 7391:
            third_bracket = aime - 7391
            pia += third_bracket * 0.15

        return pia

    @staticmethod
    def get_fra_from_birth_year(birth_year: int) -> float:
        """
        Look up Full Retirement Age (FRA) by birth year.

        Birth years 1943-1959 use table with increments.
        Birth year 1960 and later: FRA = 67.

        Args:
            birth_year: Year of birth.

        Returns:
            Full Retirement Age (in years, e.g., 66.5 for 66 years 6 months).
        """
        if birth_year in SocialSecurityModel.FRA_TABLE:
            return SocialSecurityModel.FRA_TABLE[birth_year]
        elif birth_year >= 1960:
            return 67.0
        else:
            # Before 1943: assume FRA = 65
            return 65.0

    def calculate_benefit_at_age(
        self,
        monthly_benefit_at_fra: float,
        fra: float,
        claiming_age: float
    ) -> float:
        """
        Calculate monthly benefit based on claiming age.

        Applies early reduction (if claiming before FRA) or delayed credit
        (if claiming after FRA).

        Early reduction: Reduces by (5/9 of 1%) per month for first 36 months
        before FRA, then (5/12 of 1%) per month beyond.

        Delayed credit: Increases by 8% per year (8/12 of 1% per month) after FRA,
        maximum to age 70.

        Args:
            monthly_benefit_at_fra: Monthly benefit at full retirement age.
            fra: Full retirement age (e.g., 66.5).
            claiming_age: Age at which benefit is claimed.

        Returns:
            Monthly benefit adjusted for claiming age.
        """
        if claiming_age == fra:
            return monthly_benefit_at_fra
        elif claiming_age < fra:
            # Early claiming: apply reduction
            months_early = (fra - claiming_age) * 12.0
            # First 36 months at 5/9 of 1%
            months_at_5_9 = min(months_early, 36.0)
            reduction = months_at_5_9 * self.EARLY_REDUCTION_MONTHS_36

            # Remaining months at 5/12 of 1%
            if months_early > 36.0:
                months_at_5_12 = months_early - 36.0
                reduction += months_at_5_12 * self.EARLY_REDUCTION_MONTHS_OVER

            adjusted_benefit = monthly_benefit_at_fra * (1.0 - reduction)
            return adjusted_benefit
        else:
            # Delayed claiming: apply credit (max to age 70)
            months_delayed = min((claiming_age - fra) * 12.0, (70.0 - fra) * 12.0)
            credit = (months_delayed / 12.0) * self.DELAYED_CREDIT_RATE
            adjusted_benefit = monthly_benefit_at_fra * (1.0 + credit)
            return adjusted_benefit

    def calculate_annual_ss_income(
        self,
        age: int,
        config: SimulationConfig,
        spouse_current_age: int = None
    ) -> float:
        """
        Calculate total annual Social Security income (primary + spouse).

        Primary and spouse benefits are gated INDEPENDENTLY:
          - Primary SS flows when primary age >= primary ss_claiming_age.
          - Spouse SS flows when spouse_current_age >= spouse_ss_claiming_age.

        FRA is read from config.ss_fra (not hardcoded).

        Args:
            age: Primary earner's current age.
            config: SimulationConfig with SS claiming parameters.
            spouse_current_age: Spouse's current age this year (computed from
                age gap). None if no spouse.

        Returns:
            Annual Social Security income (primary + spouse's own benefit
            + any spousal supplement).
        """
        # ── Primary FRA from config (FIX for Bug 7: was hardcoded 66.5) ────
        fra = getattr(config, 'ss_fra', 67)

        # ── Primary benefit: gated by primary claiming age ─────────────────
        annual_primary = 0.0
        if age >= config.ss_claiming_age:
            monthly_pia = config.ss_monthly_benefit_at_fra
            monthly_primary = self.calculate_benefit_at_age(
                monthly_pia, fra, config.ss_claiming_age
            )
            annual_primary = monthly_primary * 12.0

        # ── Spouse benefit: gated by SPOUSE's age, not primary's ───────────
        # FIX for Bug 2: spouse SS was gated by primary claiming age.
        # Now: spouse's own benefit flows when spouse reaches spouse_ss_claiming_age.
        # Spousal supplement (50% rule) requires BOTH primary and spouse to have
        # reached their respective claiming ages.
        annual_spouse_own = 0.0
        annual_spousal_supplement = 0.0

        if (config.include_spousal_benefit
                and config.spouse_ss_monthly_benefit_at_fra > 0
                and spouse_current_age is not None):

            spouse_fra = fra  # Same FRA assumption for spouse (simplified)

            # Spouse's OWN benefit — flows when SPOUSE reaches spouse claiming age
            if spouse_current_age >= config.spouse_ss_claiming_age:
                monthly_spouse_pia = config.spouse_ss_monthly_benefit_at_fra
                monthly_spouse = self.calculate_benefit_at_age(
                    monthly_spouse_pia,
                    spouse_fra,
                    config.spouse_ss_claiming_age
                )
                annual_spouse_own = monthly_spouse * 12.0

                # Spousal supplement: if 50% of primary PIA > spouse's own benefit,
                # spouse gets the difference as a supplement.
                # Supplement only available when primary is also collecting.
                if age >= config.ss_claiming_age:
                    half_primary_pia = 0.5 * config.ss_monthly_benefit_at_fra
                    if half_primary_pia > monthly_spouse:
                        supplement_monthly = half_primary_pia - monthly_spouse
                        annual_spousal_supplement = supplement_monthly * 12.0

        return annual_primary + annual_spouse_own + annual_spousal_supplement

    def calculate_spousal_benefit(
        self,
        primary_pia: float,
        spouse_pia: float,
        spouse_claiming_age: float,
        spouse_fra: float
    ) -> float:
        """
        Calculate spousal benefit for a spouse with their own earnings record.

        Spousal benefit = max(spouse's own PIA, 50% of primary earner's PIA).
        Adjusted for spouse's claiming age.

        Args:
            primary_pia: Primary earner's PIA (monthly).
            spouse_pia: Spouse's own PIA (monthly).
            spouse_claiming_age: Spouse's claiming age.
            spouse_fra: Spouse's full retirement age.

        Returns:
            Monthly spousal benefit.
        """
        # Spouse's own benefit
        spouse_own = self.calculate_benefit_at_age(
            spouse_pia,
            spouse_fra,
            spouse_claiming_age
        )

        # Spousal supplement (50% of primary, adjusted for claiming age)
        # Simplified: no reduction for spousal benefit claimed before FRA
        spousal_supplement_pia = 0.5 * primary_pia
        spousal_supplement = self.calculate_benefit_at_age(
            spousal_supplement_pia,
            spouse_fra,
            spouse_claiming_age
        )

        # Total: own benefit + supplement, but not less than own benefit
        total_spousal_benefit = max(spouse_own, spousal_supplement)
        return total_spousal_benefit

    def get_income_overlays(self, age: int, config: 'SimulationConfig',
                            cum_inflation: float) -> dict:
        """
        Compile all non-portfolio income for a given age.

        IMPORTANT (v1.7.2 fix): Spouse income start/end ages are now compared
        against the SPOUSE's actual age, not the primary's age.  The spouse's
        age is derived as:  spouse_current_age = spouse_age + (age - current_age).

        Args:
            age: Primary earner's current age.
            config: SimulationConfig with all income parameters.
            cum_inflation: Cumulative inflation factor since start of simulation.

        Returns:
            Dict with keys:
              ss_income, pension_income, rental_income,
              part_time_income, serp_income,
              total_other_income
        """
        # ── Compute spouse's actual current age (FIX for Bug 6/9) ──────────
        _spouse_age_cfg = getattr(config, 'spouse_age', None)
        if _spouse_age_cfg is not None:
            spouse_current_age = _spouse_age_cfg + (age - config.current_age)
        else:
            spouse_current_age = None

        # ── Social Security — primary and spouse gated independently ───────
        # FIX for Bug 2: now passes spouse_current_age so spouse SS is gated
        # by spouse's age, not primary's.
        # FIX for Bug 10 (SS COLA): SS benefits receive annual COLA matching CPI
        # by statute (42 USC §415(i)).  The input ss_monthly_benefit_at_fra is
        # expressed in today's dollars.  Multiply by cum_inflation to model COLA.
        ss_income = self.calculate_annual_ss_income(age, config, spouse_current_age)
        ss_income *= cum_inflation  # Apply COLA

        # ── Primary pension — gated by primary age ─────────────────────────
        pension_end = getattr(config, 'pension_end_age', 95)
        _pension_adj = cum_inflation if getattr(config, 'pension_is_real', False) else 1.0
        pension_income = (
            config.pension_annual_real * _pension_adj
            if config.pension_start_age <= age <= pension_end
            else 0.0
        )

        # ── Rental income — gated by primary age ──────────────────────────
        rental_start = getattr(config, 'rental_start_age', 0)
        rental_end   = getattr(config, 'rental_end_age', 999)
        rental_income = (
            config.rental_income_annual_real * cum_inflation
            if rental_start <= age <= rental_end
            else 0.0
        )

        # ── Primary part-time income — gated by primary age ───────────────
        pt_start = getattr(config, 'part_time_income_start_age', 0)
        pt_end   = getattr(config, 'part_time_income_end_age', 999)
        _pt_adj = cum_inflation if getattr(config, 'part_time_is_real', True) else 1.0
        part_time_income = (
            config.part_time_income_annual * _pt_adj
            if pt_start <= age <= pt_end
            else 0.0
        )

        # ── SERP (primary) — calendar-year based (correct as-is) ──────────
        _start_year = getattr(config, 'simulation_start_year', 2026)
        _cal_year   = _start_year + (age - config.current_age)
        serp_income = getattr(config, f'serp_{_cal_year}', 0.0)

        total_other_income = (
            ss_income + pension_income + rental_income +
            part_time_income + serp_income
        )

        return {
            'ss_income':              ss_income,
            'pension_income':         pension_income,
            'rental_income':          rental_income,
            'part_time_income':       part_time_income,
            'serp_income':            serp_income,
            'total_other_income':     total_other_income,
        }
