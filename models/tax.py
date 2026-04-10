"""
Tax Calculation Module for Retirement Simulations.

Implements federal and state income tax calculation, including:
  - 2025 federal income tax brackets (single and MFJ)
  - Standard deduction with senior adjustment (65+)
  - Long-term capital gains tax
  - Social Security taxable income rules (provisional income test)
  - IRMAA Medicare surcharge for high earners
  - RMD tax treatment (ordinary income)

Tax Year: 2025 (indexed to inflation)

Version: 1.0.0

Boundary / Edge Cases:
  - Zero income year: total tax = 0; effective rate = 0.
  - All income from Roth: tax = 0 (Roth withdrawals excluded from AGI).
  - Negative AGI: clamped to 0; no refundable credit modeled.
  - SS income + high ordinary income: up to 85% of SS is taxable (provisional
    income test).
  - state_tax_rate = 0 or include_state_tax=False: state tax component = 0.
  - IRMAA surcharges: triggered by high MAGI; hard-coded bracket thresholds
    must be updated for future years.
"""

from typing import Optional, Dict, Tuple
from config.defaults import SimulationConfig, TAX_BRACKETS_2025


class TaxCalculator:
    """
    Calculate federal, state, and effective tax liability for retirement income.

    Handles progressive tax brackets, capital gains rates, Social Security
    taxability thresholds, and IRMAA surcharges.

    Attributes:
        TAX_BRACKETS_2025_SINGLE: Federal brackets for single filers.
        TAX_BRACKETS_2025_MFJ: Federal brackets for married filing jointly.
        STANDARD_DEDUCTION_SINGLE: Single filer standard deduction (2025).
        STANDARD_DEDUCTION_MFJ: Married filing jointly standard deduction (2025).
        STANDARD_DEDUCTION_65_PLUS: Additional deduction per person age 65+ (2025).
        LTCG_RATE_ZERO: 0% LTCG income threshold for MFJ.
        LTCG_RATE_15: 15% LTCG income threshold (top of 15% bracket).
    """

    # 2025 Federal Income Tax Brackets (inflation-indexed)
    TAX_BRACKETS_2025_SINGLE = [
        (11925, 0.10),
        (48475, 0.12),
        (103350, 0.22),
        (197300, 0.24),
        (250525, 0.32),
        (626350, 0.35),
        (float('inf'), 0.37),
    ]

    TAX_BRACKETS_2025_MFJ = [
        (23850, 0.10),
        (96950, 0.12),
        (206700, 0.22),
        (394600, 0.24),
        (501050, 0.32),
        (751600, 0.35),
        (float('inf'), 0.37),
    ]

    # 2025 Standard Deductions
    STANDARD_DEDUCTION_SINGLE = 15000
    STANDARD_DEDUCTION_MFJ = 30000
    STANDARD_DEDUCTION_65_PLUS = 1550  # Per person age 65+

    # 2025 Long-Term Capital Gains Thresholds
    LTCG_RATE_ZERO_SINGLE = 47025
    LTCG_RATE_ZERO_MFJ = 96950
    LTCG_RATE_15_SINGLE = 518900
    LTCG_RATE_15_MFJ = 583750

    # 2025 Social Security Taxability Thresholds (Tier 1 / Tier 2)
    SS_THRESHOLD_SINGLE_TIER1 = 25000
    SS_THRESHOLD_SINGLE_TIER2 = 34000
    SS_THRESHOLD_MFJ_TIER1 = 32000
    SS_THRESHOLD_MFJ_TIER2 = 44000

    # 2025 IRMAA Medicare Surcharge Thresholds (Part B + D combined)
    # Format: (MAGI threshold, annual surcharge per person)
    # Source: 2025 CMS IRMAA brackets
    IRMAA_THRESHOLDS_SINGLE = [
        (106000, 0),           # Standard premium (no surcharge)
        (133000, 1052),        # Tier 1
        (167000, 2644),        # Tier 2
        (200000, 4232),        # Tier 3
        (500000, 5824),        # Tier 4
        (float('inf'), 6268),  # Tier 5 (highest)
    ]

    IRMAA_THRESHOLDS_MFJ = [
        (212000, 0),           # Standard premium (no surcharge)
        (266000, 1052),        # Tier 1
        (334000, 2644),        # Tier 2
        (400000, 4232),        # Tier 3
        (750000, 5824),        # Tier 4
        (float('inf'), 6268),  # Tier 5 (highest)
    ]

    def __init__(self):
        """Initialize tax calculator with 2025 parameters."""
        pass

    def calculate_federal_income_tax(
        self,
        taxable_income: float,
        filing_status: str,
        age: int,
        spouse_age: Optional[int] = None
    ) -> float:
        """
        Calculate federal income tax on taxable income.

        Applies 2025 tax brackets progressively. Standard deduction includes
        age 65+ bonus ($1,550 per person for single, $1,550 per person for MFJ).

        Args:
            taxable_income: Gross income in dollars.
            filing_status: 'single' or 'married_filing_jointly'.
            age: Primary taxpayer age.
            spouse_age: Spouse age (only used if filing_status is MFJ).

        Returns:
            Federal income tax owed in dollars.

        Raises:
            ValueError: If filing_status is invalid.
        """
        if filing_status not in ['single', 'married_filing_jointly']:
            raise ValueError(
                f"Invalid filing_status: {filing_status}. "
                f"Expected 'single' or 'married_filing_jointly'."
            )

        # Determine standard deduction
        if filing_status == 'single':
            std_ded = self.STANDARD_DEDUCTION_SINGLE
            if age >= 65:
                std_ded += self.STANDARD_DEDUCTION_65_PLUS
            brackets = self.TAX_BRACKETS_2025_SINGLE
        else:  # married_filing_jointly
            std_ded = self.STANDARD_DEDUCTION_MFJ
            if age >= 65:
                std_ded += self.STANDARD_DEDUCTION_65_PLUS
            if spouse_age is not None and spouse_age >= 65:
                std_ded += self.STANDARD_DEDUCTION_65_PLUS
            brackets = self.TAX_BRACKETS_2025_MFJ

        # Apply standard deduction
        agi = max(0.0, taxable_income - std_ded)

        # Calculate tax using progressive brackets
        tax = 0.0
        prev_limit = 0.0

        for limit, rate in brackets:
            if agi <= prev_limit:
                break
            # Income in this bracket
            bracket_income = min(agi, limit) - prev_limit
            tax += bracket_income * rate
            prev_limit = limit

        return max(0.0, tax)

    def calculate_social_security_taxable_portion(
        self,
        ss_income: float,
        other_income: float,
        filing_status: str
    ) -> float:
        """
        Calculate taxable portion of Social Security using provisional income test.

        Provisional Income = AGI (excluding SS) + 50% of SS benefits
        Thresholds vary by filing status.

        Args:
            ss_income: Annual Social Security benefit (gross, before tax).
            other_income: Other ordinary income (wages, pensions, RMD, etc.).
            filing_status: 'single' or 'married_filing_jointly'.

        Returns:
            Taxable portion of Social Security benefit (0.0 to ss_income).
        """
        if filing_status == 'single':
            tier1_thresh = self.SS_THRESHOLD_SINGLE_TIER1
            tier2_thresh = self.SS_THRESHOLD_SINGLE_TIER2
        elif filing_status == 'married_filing_jointly':
            tier1_thresh = self.SS_THRESHOLD_MFJ_TIER1
            tier2_thresh = self.SS_THRESHOLD_MFJ_TIER2
        else:
            # Default to single if invalid
            tier1_thresh = self.SS_THRESHOLD_SINGLE_TIER1
            tier2_thresh = self.SS_THRESHOLD_SINGLE_TIER2

        # Provisional income = AGI (ex-SS) + 50% of SS
        prov_income = other_income + 0.5 * ss_income

        if prov_income <= tier1_thresh:
            # No SS is taxable
            return 0.0
        elif prov_income <= tier2_thresh:
            # Tier 1: up to 50% of SS above first threshold
            excess = prov_income - tier1_thresh
            taxable = min(ss_income, 0.5 * excess)
            return taxable
        else:
            # Tier 2: 85% of amount above second threshold, plus tier1 portion
            excess_tier2 = prov_income - tier2_thresh
            tier2_taxable = 0.85 * excess_tier2
            tier1_portion = 0.5 * (tier2_thresh - tier1_thresh)
            total_taxable = min(0.85 * ss_income, tier1_portion + tier2_taxable)
            return total_taxable

    def calculate_ltcg_tax(
        self,
        ltcg_income: float,
        ordinary_income: float,
        filing_status: str
    ) -> float:
        """
        Calculate tax on long-term capital gains.

        LTCG are taxed at preferential rates: 0%, 15%, or 20%.
        LTCG are stacked on top of ordinary income; ordinary income "uses up"
        the 0% bracket first.

        Args:
            ltcg_income: Long-term capital gains income in dollars.
            ordinary_income: Ordinary taxable income (before LTCG).
            filing_status: 'single' or 'married_filing_jointly'.

        Returns:
            Tax owed on long-term capital gains.
        """
        if filing_status == 'single':
            zero_threshold = self.LTCG_RATE_ZERO_SINGLE
            fifteen_threshold = self.LTCG_RATE_15_SINGLE
        else:  # married_filing_jointly
            zero_threshold = self.LTCG_RATE_ZERO_MFJ
            fifteen_threshold = self.LTCG_RATE_15_MFJ

        # Ordinary income "fills up" the 0% bracket
        ordinary_in_zero = min(ordinary_income, zero_threshold)
        remaining_zero = max(0.0, zero_threshold - ordinary_in_zero)

        # LTCG stacked on top of ordinary income
        ltcg_in_zero = min(ltcg_income, remaining_zero)
        remaining_fifteen = max(0.0, fifteen_threshold - max(ordinary_income, zero_threshold) - ltcg_in_zero)
        ltcg_in_fifteen = min(ltcg_income - ltcg_in_zero, remaining_fifteen)
        ltcg_in_twenty = max(0.0, ltcg_income - ltcg_in_zero - ltcg_in_fifteen)

        tax = (
            ltcg_in_zero * 0.00 +
            ltcg_in_fifteen * 0.15 +
            ltcg_in_twenty * 0.20
        )

        return tax

    def calculate_rmd_tax_impact(
        self,
        rmd_amount: float,
        other_income: float,
        filing_status: str,
        age: int,
        spouse_age: Optional[int] = None
    ) -> float:
        """
        Calculate federal tax on RMD (treated as ordinary income).

        RMD are fully taxable as ordinary income. This method adds RMD to
        other income and computes marginal federal tax.

        Args:
            rmd_amount: Required minimum distribution amount.
            other_income: Other ordinary income.
            filing_status: 'single' or 'married_filing_jointly'.
            age: Primary taxpayer age.
            spouse_age: Spouse age (if MFJ).

        Returns:
            Federal tax attributable to RMD.
        """
        # Tax on (other_income + RMD)
        total_taxable = other_income + rmd_amount
        tax_with_rmd = self.calculate_federal_income_tax(
            total_taxable,
            filing_status,
            age,
            spouse_age
        )

        # Tax on other_income alone
        tax_without_rmd = self.calculate_federal_income_tax(
            other_income,
            filing_status,
            age,
            spouse_age
        )

        # Marginal tax on RMD
        return max(0.0, tax_with_rmd - tax_without_rmd)

    def calculate_annual_tax(
        self,
        age: int,
        filing_status: str,
        state_tax_rate: float,
        include_state_tax: bool,
        traditional_withdrawal: float,
        roth_withdrawal: float,
        taxable_withdrawal: float,
        ss_income: float,
        pension_income: float,
        rental_income: float,
        part_time_income: float,
        rmd_amount: float,
        ltcg_amount: float,
        serp_income: float = 0.0,
        spouse_age: Optional[int] = None,
        roth_conversion: float = 0.0,
    ) -> Dict:
        """
        Comprehensive annual tax calculation for retirement income.

        Combines all income sources, applies proper tax treatment (ordinary vs.
        capital gains vs. tax-free), and calculates total tax liability.

        Income Treatment:
          - Traditional withdrawal: ordinary income
          - Roth withdrawal: tax-free (not included in AGI)
          - Taxable withdrawal: capital gains (long-term, preferential rate)
          - SS income: taxable portion only (provisional income test)
          - Pension, rental, part-time: ordinary income
          - RMD: ordinary income

        Args:
            age: Primary taxpayer age.
            filing_status: 'single' or 'married_filing_jointly'.
            state_tax_rate: State income tax rate (e.g., 0.05 for 5%).
            include_state_tax: Whether to include state tax in calculation.
            traditional_withdrawal: Withdrawal from traditional (pre-tax) account.
            roth_withdrawal: Withdrawal from Roth (tax-free) account.
            taxable_withdrawal: Withdrawal from taxable account (gains are LTCG).
            ss_income: Social Security income (gross, before tax).
            pension_income: Pension distribution (ordinary income).
            rental_income: Rental/other passive income.
            part_time_income: Part-time employment income.
            rmd_amount: Required minimum distribution (ordinary income).
            ltcg_amount: Embedded capital gains in taxable account (can differ from withdrawal).
            serp_income: SERP / nonqualified deferred compensation distribution.
            spouse_age: Spouse age (if filing_status is MFJ).
            roth_conversion: Amount converted from Traditional to Roth (taxable as ordinary income).

        Returns:
            Dict with keys:
              - federal_tax: Federal income tax owed
              - state_tax: State income tax owed
              - total_tax: federal_tax + state_tax
              - effective_rate: total_tax / gross_income
              - agi: Adjusted gross income (pre-standard deduction)
              - taxable_income: Taxable income after standard deduction
              - serp_income: SERP income passed through (for report transparency)
              - tax_breakdown: dict showing each income component's tax contribution
        """
        # AGI = all ordinary income sources (excluding Roth, which is tax-free)
        ordinary_income = (
            rmd_amount +
            traditional_withdrawal +
            roth_conversion +
            serp_income +
            pension_income +
            part_time_income +
            rental_income
        )

        # Add taxable portion of Social Security
        ss_taxable = self.calculate_social_security_taxable_portion(
            ss_income,
            ordinary_income,
            filing_status
        )
        ordinary_income += ss_taxable

        # AGI (before standard deduction)
        agi = ordinary_income + ltcg_amount

        # Federal tax on ordinary income
        fed_ordinary = self.calculate_federal_income_tax(
            ordinary_income,
            filing_status,
            age,
            spouse_age
        )

        # Federal tax on LTCG (at preferential rates, stacked on ordinary income)
        fed_ltcg = self.calculate_ltcg_tax(
            ltcg_amount,
            ordinary_income,
            filing_status
        )

        # Total federal tax
        federal_tax = fed_ordinary + fed_ltcg

        # State tax (typically on AGI, simplified)
        state_tax = 0.0
        if include_state_tax:
            state_tax = agi * state_tax_rate

        # Total tax
        total_tax = federal_tax + state_tax

        # Effective rate
        gross_income = (
            traditional_withdrawal +
            roth_withdrawal +
            taxable_withdrawal +
            roth_conversion +
            ss_income +
            pension_income +
            serp_income +
            rental_income +
            part_time_income
        )
        effective_rate = total_tax / gross_income if gross_income > 0 else 0.0

        return {
            'federal_tax': federal_tax,
            'state_tax': state_tax,
            'total_tax': total_tax,
            'effective_rate': effective_rate,
            'agi': agi,
            'taxable_income': agi,  # Simplified (pre-standard deduction AGI)
            'serp_income': serp_income,
            'tax_breakdown': {
                'rmd': rmd_amount,
                'traditional_withdrawal': traditional_withdrawal,
                'serp_primary': serp_income,
                'pension_primary': pension_income,
                'part_time_primary': part_time_income,
                'rental': rental_income,
                'ss_taxable_portion': ss_taxable,
                'roth_withdrawal_tax_free': roth_withdrawal,
                'ltcg': ltcg_amount,
            },
        }

    def estimate_irmaa_surcharge(
        self,
        magi: float,
        filing_status: str
    ) -> float:
        """
        Estimate IRMAA (Income-Related Monthly Adjustment Amount) surcharge.

        IRMAA is an additional Medicare premium for high earners on Parts B and D.
        2025 thresholds provided.

        Args:
            magi: Modified Adjusted Gross Income.
            filing_status: 'single' or 'married_filing_jointly'.

        Returns:
            Annual IRMAA surcharge amount (0 if below first threshold).
        """
        if filing_status == 'married_filing_jointly':
            thresholds = self.IRMAA_THRESHOLDS_MFJ
        else:
            thresholds = self.IRMAA_THRESHOLDS_SINGLE

        # Find applicable tier
        for i, (threshold, surcharge) in enumerate(thresholds):
            if magi < threshold:
                return surcharge

        # Above all thresholds
        return thresholds[-1][1]
