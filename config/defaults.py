"""
Default configuration and data classes for retirement financial modeling.

Defines the core data structures (SimulationConfig, SimulationResults) and
constants (ASSET_CLASSES, tax brackets, Social Security bend points) used
across all model modules.

Boundary / Edge Cases:
  - Zero balances (all accounts empty): simulation runs but portfolio depletes
    immediately on any non-zero withdrawal.  Success rate should be 0%.
  - Negative cash flows: annual_contribution < 0 is invalid; catches in
    __post_init__ validation.
  - Life expectancy <= current_age: n_years = 0; simulation produces empty
    arrays.  Caught in validation.
  - withdrawal_start_age < current_age: withdrawals begin immediately; valid
    but unusual.
  - Allocation sums != 1.0: caught and normalised in portfolio.py; also
    validated here with a warning.
  - state_tax_rate > 0 but include_state_tax=False: state tax not applied.
  - ss_claiming_age < 62: invalid per SSA rules; caught in validation.
  - spouse_age = None with non-zero spouse balances: treated as single-earner
    household with spouse assets; may or may not be intentional.

Version: 1.8.0
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import logging
import numpy as np

logger = logging.getLogger(__name__)


# Asset class definitions
ASSET_CLASSES = [
    'us_large_cap',
    'us_small_cap',
    'international_dev',
    'emerging_markets',
    'us_bonds',
    'tips',
    'cash',
    'reits',
]

# 2025 Federal Tax Brackets (indexed to inflation)
TAX_BRACKETS_2025 = {
    'single': [
        (11925, 0.10),
        (48475, 0.12),
        (103350, 0.22),
        (197300, 0.24),
        (250525, 0.32),
        (626350, 0.35),
        (float('inf'), 0.37),
    ],
    'married_filing_jointly': [
        (23850, 0.10),
        (96950, 0.12),
        (206700, 0.22),
        (394600, 0.24),
        (501050, 0.32),
        (751600, 0.35),
        (float('inf'), 0.37),
    ],
}

# 2025 Social Security Bend Points (monthly amounts)
SS_BEND_POINTS_2025 = [1226, 7391]

# Social Security PIA formula percentages
SS_PIA_PERCENTAGES = [0.90, 0.32, 0.15]


@dataclass
class SimulationConfig:
    """
    Complete configuration for a retirement simulation.

    Includes demographics, financial parameters, account structure,
    withdrawal strategy, asset allocation, simulation method, tax parameters,
    Social Security, and other income sources.

    Attributes:
        # Demographics
        current_age: Current age in years.
        retirement_age: Target retirement age.
        life_expectancy: Planning horizon (age to model to).
        spouse_age: Spouse's current age (None if single).
        spouse_life_expectancy: Spouse's planning horizon.

        # Portfolio Structure
        total_portfolio_value: Sum of all account balances.
        traditional_balance: Pre-tax account balance (401k, IRA).
        roth_balance: Roth account balance (Roth IRA, Roth 401k).
        taxable_balance: Taxable brokerage account balance.

        # Contributions & Growth
        annual_contribution: Annual contribution amount (today's dollars).
        contribution_growth_rate: Annual growth rate of contributions.

        # Withdrawals
        annual_withdrawal_real: Net portfolio draw in today's dollars (fixed_real method).
        withdrawal_start_age: Age at which drawing from portfolio begins (explicit; default = retirement_age).
        withdrawal_method: 'fixed_real', 'percentage', or 'guardrails'.
        withdrawal_rate: Withdrawal rate as % of portfolio (percentage method).
        withdrawal_floor: Min withdrawal as % (guardrails method).
        withdrawal_ceiling: Max withdrawal as % (guardrails method).

        # Asset Allocation
        pre_retirement_allocation: Dict of asset weights before retirement.
        retirement_allocation: Dict of asset weights at/after retirement.
        use_glide_path: Whether to transition allocation over time.
        rebalance_threshold: Drift threshold for rebalancing (e.g., 0.05 for 5%).

        # Simulation Method
        simulation_method: 'bootstrap' or 'gbm'.
        n_simulations: Number of Monte Carlo paths.
        random_seed: Seed for RNG reproducibility.

        # Taxes
        filing_status: 'single' or 'married_filing_jointly'.
        state_tax_rate: State income tax rate (e.g., 0.05 for 5%).
        include_state_tax: Whether to include state tax calculation.

        # Social Security
        ss_monthly_benefit_at_fra: Primary earner's SS benefit at FRA (monthly).
        ss_fra: Primary earner's full retirement age.
        ss_claiming_age: Age at which primary earner claims SS.
        spouse_ss_monthly_benefit_at_fra: Spouse's SS benefit at FRA.
        spouse_ss_claiming_age: Age at which spouse claims SS.
        include_spousal_benefit: Whether to include spouse's benefit.

        # Other Income
        pension_annual_real: Annual pension in today's dollars.
        pension_start_age: Age pension begins.
        rental_income_annual_real: Annual rental income (today's dollars).
        part_time_income_annual: Annual part-time work income.
        part_time_income_end_age: Age part-time income ends.

        # Income — Start/End Ages
        pension_end_age: Age pension ends (default 95 = lifetime).
        rental_start_age: Age rental income begins.
        rental_end_age: Age rental income ends (e.g., property sold).
        part_time_income_start_age: Age primary earner begins part-time work.

        # SERP / Nonqualified Deferred Compensation
        # Tax treatment: ordinary income for federal + state income tax.
        # FICA NOT applicable at distribution — already paid at time of deferral.
        simulation_start_year: Calendar year corresponding to current_age (for SERP schedule).
        serp_2026..serp_2033: Per-year SERP distribution (nominal $, not inflation-adjusted).

        # Fees & Costs
        expense_ratio: Annual expense ratio (e.g., 0.001 for 0.1%).
        advisory_fee: Annual advisory fee (e.g., 0.005 for 0.5%).

        # Inflation
        inflation_method: 'bootstrap', 'fixed', or 'mean_reverting'.
        inflation_mean: Mean inflation rate (e.g., 0.03 for 3%).
        inflation_std: Std dev of inflation (mean_reverting method).
        healthcare_inflation_premium: Extra inflation on healthcare (e.g., 0.02).
        annual_healthcare_cost_real: Annual healthcare cost (today's dollars).
        include_healthcare_costs: Whether to include healthcare costs in withdrawals.
    """

    # Demographics
    current_age: int = 60
    retirement_age: int = 65
    life_expectancy: int = 95
    spouse_age: Optional[int] = None
    spouse_life_expectancy: int = 98

    # Portfolio Structure
    total_portfolio_value: float = 600000.0  # Computed: sum of 3 balances
    traditional_balance: float = 500000.0
    roth_balance: float = 100000.0
    taxable_balance: float = 0.0

    # Contributions & Growth
    annual_contribution: float = 0.0
    contribution_growth_rate: float = 0.02

    # Withdrawals
    # Primary earner: net portfolio draw in today's dollars.
    # Active from withdrawal_start_age (explicit; independent of retirement_age).
    annual_withdrawal_real: float = 50000.0
    withdrawal_start_age: int = 65          # Age primary begins drawing from portfolio.
    withdrawal_method: str = 'fixed_real'
    withdrawal_rate: float = 0.04
    withdrawal_floor: float = 0.03
    withdrawal_ceiling: float = 0.05

    # Asset Allocation
    pre_retirement_allocation: Dict[str, float] = field(default_factory=lambda: {
        'us_large_cap': 0.85,
        'us_small_cap': 0.00,
        'international_dev': 0.00,
        'emerging_markets': 0.00,
        'us_bonds': 0.00,
        'tips': 0.00,
        'cash': 0.15,
        'reits': 0.00,
    })
    retirement_allocation: Dict[str, float] = field(default_factory=lambda: {
        'us_large_cap': 0.70,
        'us_small_cap': 0.00,
        'international_dev': 0.00,
        'emerging_markets': 0.00,
        'us_bonds': 0.00,
        'tips': 0.10,
        'cash': 0.20,
        'reits': 0.00,
    })
    use_glide_path: bool = True
    rebalance_threshold: float = 0.05

    # Simulation Method
    simulation_method: str = 'gbm'
    n_simulations: int = 10000
    random_seed: Optional[int] = None

    # Taxes
    filing_status: str = 'single'
    state_tax_rate: float = 0.05
    include_state_tax: bool = False

    # Social Security
    ss_monthly_benefit_at_fra: float = 2500.0
    ss_fra: int = 67
    ss_claiming_age: int = 67
    spouse_ss_monthly_benefit_at_fra: float = 0.0
    spouse_ss_claiming_age: int = 67
    include_spousal_benefit: bool = True

    # Other Income
    pension_annual_real: float = 0.0
    pension_start_age: int = 67
    rental_income_annual_real: float = 0.0
    part_time_income_annual: float = 0.0
    part_time_income_end_age: int = 65

    # Pension end age
    pension_end_age: int = 95  # Default: lifetime pension

    # Rental income — start/end age
    rental_start_age: int = 69
    rental_end_age: int = 95   # Default: indefinite

    # Primary part-time — add start age (end age already exists)
    part_time_income_start_age: int = 55

    # Simulation start year — maps current_age to the first calendar year.
    # Used for SERP per-year distribution schedule (age→year lookup).
    simulation_start_year: int = 2026

    # SERP / Nonqualified Deferred Compensation (IRC §409A)
    # Per-year distribution schedule: nominal contractual dollars per calendar year.
    # NOT inflation-adjusted — these are fixed contractual amounts.
    # Federal + state ordinary income tax applies at distribution.
    # FICA (Social Security + Medicare) NOT applicable — already paid at deferral.
    serp_2026: float = 0.0
    serp_2027: float = 0.0
    serp_2028: float = 0.0
    serp_2029: float = 0.0
    serp_2030: float = 0.0
    serp_2031: float = 0.0
    serp_2032: float = 0.0
    serp_2033: float = 0.0

    # ── Spend (v1.8) ────────────────────────────────────────────────────────
    # Annual discretionary/lifestyle spending in today's dollars.
    # Simulation logic: net = total_income − spend.
    #   If net < 0 → shortfall added to portfolio withdrawal target.
    #   If net ≥ 0 → surplus handled per spend_surplus_mode.
    # Healthcare costs remain separate (own inflation premium).
    spend_annual_real: float = 80000.0
    spend_start_age: int = 60              # Age spend begins
    spend_surplus_mode: str = 'ignore'     # 'ignore' or 'reinvest' (surplus → taxable account)

    # ── Healthcare visibility flag (v1.8) ─────────────────────────────────
    # Controls whether annual_healthcare_cost_real is inflation-adjusted.
    # True  = real dollars (×cum_inflation × (1 + healthcare_inflation_premium)).
    # False = nominal / fixed dollar amount (no inflation adjustment).
    healthcare_is_real: bool = True

    # Real / Nominal Flags — controls whether income source is inflation-adjusted.
    # True  = real dollars (multiplied by cum_inflation each year).
    # False = nominal / fixed contractual dollars (no inflation adjustment).
    # Pensions default False (most private-sector pensions are fixed nominal).
    pension_is_real: bool = False
    part_time_is_real: bool = False

    # Fees & Costs
    expense_ratio: float = 0.0010
    advisory_fee: float = 0.0050

    # Inflation
    inflation_method: str = 'fixed'
    inflation_mean: float = 0.025
    inflation_std: float = 0.015
    healthcare_inflation_premium: float = 0.020
    annual_healthcare_cost_real: float = 12000.0
    include_healthcare_costs: bool = True

    # ── Input Validation (Req 2) ─────────────────────────────────────────────
    def __post_init__(self):
        """Validate types, ranges, and cross-field consistency on construction."""
        errors: List[str] = []
        warnings: List[str] = []

        # --- Demographics ---
        for fld, lo, hi in [
            ('current_age', 25, 75), ('retirement_age', 45, 85),
            ('life_expectancy', 60, 120),
        ]:
            v = getattr(self, fld)
            if not isinstance(v, (int, float)):
                errors.append(f"{fld} must be numeric, got {type(v).__name__}")
            elif not (lo <= v <= hi):
                errors.append(f"{fld}={v} outside valid range [{lo}, {hi}]")
        if self.life_expectancy <= self.current_age:
            errors.append(
                f"life_expectancy ({self.life_expectancy}) must exceed "
                f"current_age ({self.current_age})"
            )

        # --- Dollar amounts must be >= 0 ---
        dollar_fields = [
            'traditional_balance', 'roth_balance', 'taxable_balance',
            'annual_contribution', 'annual_withdrawal_real',
            'pension_annual_real',
            'rental_income_annual_real', 'part_time_income_annual',
            'annual_healthcare_cost_real',
            'expense_ratio', 'advisory_fee',
            'spend_annual_real',
        ]
        for fld in dollar_fields:
            v = getattr(self, fld, None)
            if v is not None and v < 0:
                errors.append(f"{fld}={v} must be >= 0")

        # --- Rates must be in [0, 1] ---
        rate_fields = [
            'contribution_growth_rate', 'withdrawal_rate', 'withdrawal_floor',
            'withdrawal_ceiling', 'state_tax_rate', 'inflation_mean',
            'inflation_std', 'healthcare_inflation_premium',
            'expense_ratio', 'advisory_fee',
        ]
        for fld in rate_fields:
            v = getattr(self, fld, None)
            if v is not None and not (0.0 <= v <= 1.0):
                errors.append(f"{fld}={v} outside [0.0, 1.0]")

        # --- SS ages ---
        if not (62 <= self.ss_claiming_age <= 70):
            errors.append(f"ss_claiming_age={self.ss_claiming_age} outside [62, 70]")
        if not (62 <= self.spouse_ss_claiming_age <= 70):
            errors.append(
                f"spouse_ss_claiming_age={self.spouse_ss_claiming_age} outside [62, 70]"
            )

        # --- Allocation sum check (warning, not hard error — portfolio.py normalises) ---
        for label, alloc in [
            ('pre_retirement_allocation', self.pre_retirement_allocation),
            ('retirement_allocation', self.retirement_allocation),
        ]:
            s = sum(alloc.values())
            if abs(s - 1.0) > 0.02:
                errors.append(
                    f"{label} weights sum to {s:.4f}, expected ~1.0 (±0.02)"
                )
            elif abs(s - 1.0) > 1e-6:
                warnings.append(
                    f"{label} weights sum to {s:.6f}; will be normalised to 1.0."
                )

        # --- Spend (v1.8) ---
        if self.spend_annual_real < 0:
            errors.append(f"spend_annual_real={self.spend_annual_real} must be >= 0")
        if not (self.current_age <= self.spend_start_age <= self.life_expectancy):
            warnings.append(
                f"spend_start_age={self.spend_start_age} outside "
                f"[{self.current_age}, {self.life_expectancy}]; will be clamped."
            )
            self.spend_start_age = max(self.current_age, min(self.spend_start_age, self.life_expectancy))
        if self.spend_surplus_mode not in ('ignore', 'reinvest'):
            errors.append(f"spend_surplus_mode='{self.spend_surplus_mode}' not in (ignore, reinvest)")

        # --- Simulation params ---
        if self.simulation_method not in ('bootstrap', 'gbm'):
            errors.append(f"simulation_method='{self.simulation_method}' not in (bootstrap, gbm)")
        if self.n_simulations < 1:
            errors.append(f"n_simulations={self.n_simulations} must be >= 1")
        if self.filing_status not in ('single', 'married_filing_jointly'):
            errors.append(f"filing_status='{self.filing_status}' not recognised")
        if self.withdrawal_method not in ('fixed_real', 'fixed_nominal', 'percentage', 'guardrails'):
            errors.append(f"withdrawal_method='{self.withdrawal_method}' not recognised")

        # --- Recompute total_portfolio_value from constituents ---
        computed_total = (
            self.traditional_balance + self.roth_balance + self.taxable_balance
        )
        if abs(self.total_portfolio_value - computed_total) > 1.0:
            warnings.append(
                f"total_portfolio_value ({self.total_portfolio_value:,.0f}) differs from "
                f"sum of 3 balances ({computed_total:,.0f}); overwriting with sum."
            )
            self.total_portfolio_value = computed_total

        # --- Log warnings and raise errors ---
        for w in warnings:
            logger.warning("SimulationConfig: %s", w)
        if errors:
            msg = "SimulationConfig validation failed:\n  " + "\n  ".join(errors)
            raise ValueError(msg)


@dataclass
class SimulationResults:
    """
    Results from a completed Monte Carlo simulation run.

    Contains full path data for all simulations and aggregated success statistics.

    Attributes:
        config: Original SimulationConfig used for this run.
        portfolio_values: Array shape (n_simulations, n_years) of total portfolio balance.
        traditional_values: Array shape (n_simulations, n_years) of traditional account.
        roth_values: Array shape (n_simulations, n_years) of Roth account.
        taxable_values: Array shape (n_simulations, n_years) of taxable account.
        withdrawals: Array shape (n_simulations, n_years) of annual withdrawals.
        taxes: Array shape (n_simulations, n_years) of annual taxes paid.
        ss_income: Array shape (n_simulations, n_years) of Social Security income.
        inflation_rates: Array shape (n_simulations, n_years) of inflation rates.
        portfolio_returns: Array shape (n_simulations, n_years) of portfolio returns.
        success_mask: Array shape (n_simulations,) bool. True if portfolio never depletes.
        success_rate: Fraction of simulations that succeeded (0.0 to 1.0).
        ages: Array of ages from current to life_expectancy.
        years: Array of year indices (0 to n_years-1).
    """

    config: SimulationConfig
    portfolio_values: np.ndarray
    traditional_values: np.ndarray
    roth_values: np.ndarray
    taxable_values: np.ndarray
    withdrawals: np.ndarray
    taxes: np.ndarray
    ss_income: np.ndarray
    inflation_rates: np.ndarray
    portfolio_returns: np.ndarray
    success_mask: np.ndarray
    success_rate: float
    ages: np.ndarray
    years: np.ndarray
    # Spend tracking (v1.8): annual spend amounts and shortfalls (shape: n_sims × n_years)
    spend_amounts: np.ndarray = field(default_factory=lambda: np.array([]))
    spend_shortfalls: np.ndarray = field(default_factory=lambda: np.array([]))
    # Per-bucket withdrawal breakdown (shape: n_sims × n_years)
    trad_withdrawals: np.ndarray = field(default_factory=lambda: np.array([]))
    roth_withdrawals: np.ndarray = field(default_factory=lambda: np.array([]))
    taxable_withdrawals: np.ndarray = field(default_factory=lambda: np.array([]))
    # Gross income tracking (shape: n_sims × n_years) — all taxable income sources
    gross_income: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary_statistics(self):
        """
        Return a dict of summary statistics for the simulation.

        Returns:
            Dict with portfolio statistics (mean, median, percentiles, etc.).
        """
        final_values = self.portfolio_values[:, -1]
        return {
            'success_rate': self.success_rate,
            'final_value_mean': np.mean(final_values),
            'final_value_median': np.median(final_values),
            'final_value_10pct': np.percentile(final_values, 10),
            'final_value_90pct': np.percentile(final_values, 90),
            'final_value_min': np.min(final_values),
            'final_value_max': np.max(final_values),
            'avg_withdrawal_mean': np.mean(self.withdrawals),
            'avg_withdrawal_median': np.median(self.withdrawals),
            'avg_tax_mean': np.mean(self.taxes),
            'avg_tax_median': np.median(self.taxes),
        }

    def compute_rich_statistics(self) -> dict:
        """
        Compute Portfolio Visualizer-style summary statistics table.

        Returns dict with 8 metrics × 5 percentile columns (10th, 25th, 50th, 75th, 90th):
          - twrr_nominal: Time-Weighted Rate of Return (nominal)
          - twrr_real: TWRR adjusted for inflation
          - end_balance_nominal: Final portfolio value (nominal)
          - end_balance_real: Final portfolio value (real / today's dollars)
          - max_drawdown: Maximum peak-to-trough drawdown
          - max_drawdown_ex_cf: Max drawdown excluding cash flows
          - safe_withdrawal_rate: Annual withdrawal / initial portfolio
          - perpetual_withdrawal_rate: Real yield that preserves principal

        Plus:
          - survival_rate: fraction of simulations that survived
        """
        n_sims, n_years = self.portfolio_values.shape
        percentiles = [10, 25, 50, 75, 90]
        initial_portfolio = self.portfolio_values[:, 0]

        # ── TWRR nominal: geometric mean of annual portfolio returns ──
        # portfolio_returns is (n_sims, n_years) of annual return rates
        # TWRR = (prod(1 + r_t))^(1/n) - 1
        cum_returns = np.prod(1.0 + self.portfolio_returns, axis=1)
        twrr_nominal = np.power(np.maximum(cum_returns, 1e-10), 1.0 / n_years) - 1.0

        # ── TWRR real: adjust for inflation ──
        cum_inflation = np.prod(1.0 + self.inflation_rates, axis=1)
        real_cum_returns = cum_returns / cum_inflation
        twrr_real = np.power(np.maximum(real_cum_returns, 1e-10), 1.0 / n_years) - 1.0

        # ── End balance nominal & real ──
        end_bal_nominal = self.portfolio_values[:, -1]
        end_bal_real = end_bal_nominal / np.maximum(cum_inflation, 1e-10)

        # ── Max drawdown (includes cash flows) ──
        max_dd = np.zeros(n_sims)
        for s in range(n_sims):
            path = self.portfolio_values[s, :]
            running_max = np.maximum.accumulate(path)
            dd = np.where(running_max > 0, (path - running_max) / running_max, 0.0)
            max_dd[s] = np.min(dd)  # most negative

        # ── Max drawdown excluding cash flows ──
        # Reconstruct returns-only path (no contributions/withdrawals)
        max_dd_ex_cf = np.zeros(n_sims)
        for s in range(n_sims):
            synthetic = np.ones(n_years)
            synthetic[0] = initial_portfolio[s] if initial_portfolio[s] > 0 else 1.0
            for t in range(1, n_years):
                synthetic[t] = synthetic[t - 1] * (1.0 + self.portfolio_returns[s, t])
            running_max = np.maximum.accumulate(synthetic)
            dd = np.where(running_max > 0, (synthetic - running_max) / running_max, 0.0)
            max_dd_ex_cf[s] = np.min(dd)

        # ── Safe Withdrawal Rate: mean annual withdrawal / initial portfolio ──
        mean_annual_withdrawal = np.mean(self.withdrawals, axis=1)
        swr = np.where(
            initial_portfolio > 0,
            mean_annual_withdrawal / initial_portfolio,
            0.0
        )

        # ── Perpetual Withdrawal Rate: real return that preserves principal ──
        # Approximated as TWRR_real (the geometric mean real return)
        pwr = np.clip(twrr_real, 0.0, None)

        # ── Build percentile table ──
        metrics = {
            'twrr_nominal': twrr_nominal,
            'twrr_real': twrr_real,
            'end_balance_nominal': end_bal_nominal,
            'end_balance_real': end_bal_real,
            'max_drawdown': max_dd,
            'max_drawdown_ex_cf': max_dd_ex_cf,
            'safe_withdrawal_rate': swr,
            'perpetual_withdrawal_rate': pwr,
        }

        table = {}
        for name, data in metrics.items():
            table[name] = {p: float(np.percentile(data, p)) for p in percentiles}

        table['survival_rate'] = self.success_rate

        return table
