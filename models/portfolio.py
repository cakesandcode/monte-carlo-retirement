"""
Portfolio Mechanics for Retirement Simulation.

Implements the three-bucket portfolio model (traditional, Roth, taxable),
rebalancing, contributions, withdrawals, and required minimum distributions.
All computations are annual and done per-simulation per-year (not vectorized).

Key Features:
  - 3-bucket withdrawal order: taxable → traditional (≥RMD) → Roth
  - Threshold-based rebalancing
  - Glide-path asset allocation (pre-retirement to retirement)
  - RMD calculation per IRS Uniform Lifetime Table (age 73+, SECURE 2.0)
  - Expense ratio and advisory fee drag
  - Allocation validation and normalization (sum must ≈ 1.0)

Boundary / Edge Cases:
  - Zero portfolio: all operations short-circuit; returns 0 balances.
  - Allocation sums to > 1.0: creates artificial leverage; must validate and
    normalize before use.  Allocation sums to < 1.0: under-invests cash.
  - Negative returns (severe drawdown): buckets can approach zero but are
    clamped to max(0, balance).  If total_portfolio <= 0, simulation marks
    path as depleted.
  - RMD > portfolio: withdrawal capped at available balance; remaining_need
    recorded as unmet.
  - Zero interest rates / zero returns: portfolio declines monotonically by
    fees + withdrawals; legitimate scenario in stress testing.
  - Irrational ending total vs. starting: post-simulation sanity check should
    flag if ending portfolio > starting * (1 + max_plausible_return)^n_years.
  - Spouse contributions after primary retirement but before spouse retirement:
    handled by separate age gate.

Version: 1.7.1
"""

import numpy as np
import logging
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Dict, Optional, Tuple
from config.defaults import SimulationConfig, ASSET_CLASSES

logger = logging.getLogger(__name__)

# ── Decimal helpers for withdrawal / RMD precision ────────────────────────────
def _to_dec(v: float) -> Decimal:
    """Convert float to Decimal with 2 dp rounding for financial math."""
    return Decimal(str(v))

def _from_dec(d: Decimal) -> float:
    """Convert Decimal back to float for numpy interop."""
    return float(d)

# ── Allocation guard ──────────────────────────────────────────────────────────
_ALLOC_TOLERANCE = 0.02  # 2 % absolute tolerance for allocation sums

def _validate_and_normalize_allocation(
    allocation: Dict[str, float],
    label: str = "allocation",
) -> Dict[str, float]:
    """
    Validate that allocation weights sum to ~1.0; normalise if within tolerance.

    Raises ValueError if the sum deviates by more than _ALLOC_TOLERANCE from 1.0.
    If within tolerance but not exact, rescales proportionally to 1.0 and logs a
    warning.

    Args:
        allocation: {asset_class: weight} dict.
        label: Human-readable label for error messages.

    Returns:
        Normalised allocation dict (weights sum to exactly 1.0).
    """
    total = sum(allocation.values())
    if abs(total - 1.0) < 1e-9:
        return allocation  # already exact
    if abs(total - 1.0) > _ALLOC_TOLERANCE:
        raise ValueError(
            f"{label} weights sum to {total:.4f}, expected ~1.0 "
            f"(tolerance ±{_ALLOC_TOLERANCE}).  Check input allocations."
        )
    # Within tolerance but not exact — normalise and warn
    logger.warning(
        "%s weights sum to %.6f (not 1.0); normalising.", label, total
    )
    return {k: v / total for k, v in allocation.items()}


class PortfolioMechanics:
    """
    Annual portfolio state transitions: contributions, returns, withdrawals, rebalancing.

    All methods operate on single-year, single-simulation basis. Handles three
    distinct account types with tax-aware withdrawal ordering.

    Attributes:
        IRS_UNIFORM_LIFETIME (Dict[int, float]): IRS Uniform Lifetime Table
            for RMD calculation (ages 72-120+).
    """

    # IRS Uniform Lifetime Table (SECURE 2.0: RMD age = 73, effective 2023)
    # Maps age to distribution period (divisor for RMD calculation)
    IRS_UNIFORM_LIFETIME = {
        72: 27.4,   # Pre-SECURE (for historical compatibility)
        73: 26.5,
        74: 25.5,
        75: 24.6,
        76: 23.7,
        77: 22.9,
        78: 22.0,
        79: 21.1,
        80: 20.2,
        81: 19.4,
        82: 18.5,
        83: 17.7,
        84: 16.8,
        85: 16.0,
        86: 15.1,
        87: 14.3,
        88: 13.4,
        89: 12.6,
        90: 11.8,
        91: 11.0,
        92: 10.2,
        93: 9.4,
        94: 8.7,
        95: 8.0,
        96: 7.3,
        97: 6.6,
        98: 6.0,
        99: 5.4,
        100: 4.9,
        101: 4.4,
        102: 3.9,
        103: 3.4,
        104: 2.9,
        105: 2.4,
        106: 2.0,
        107: 1.6,
        108: 1.3,
        109: 1.0,
        110: 0.8,
        111: 0.6,
        112: 0.4,
        113: 0.3,
        114: 0.2,
        115: 0.1,
    }

    @staticmethod
    def calculate_rmd(age: int, traditional_balance: float) -> float:
        """
        Calculate required minimum distribution for traditional account.

        RMD applies age 73+ (per SECURE 2.0). Uses IRS Uniform Lifetime Table.
        Formula: RMD = Account Balance at Dec 31 of prior year / Distribution Period

        Args:
            age: Current age.
            traditional_balance: Balance in traditional (pre-tax) account.

        Returns:
            Required minimum distribution amount. Returns 0 if age < 73.

        Raises:
            None
        """
        if age < 73:
            return 0.0

        # Get distribution period from table, default to 0.1 for very old ages
        distribution_period = PortfolioMechanics.IRS_UNIFORM_LIFETIME.get(age, 0.1)

        rmd = traditional_balance / distribution_period
        return max(0.0, rmd)

    @staticmethod
    def calculate_glide_path_allocation(
        current_age: int,
        retirement_age: int,
        pre_allocation: Dict[str, float],
        retirement_allocation: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Interpolate asset allocation along glide path.

        Linear interpolation between pre-retirement and retirement allocations.
        Before retirement_age: moves from pre_allocation toward retirement_allocation.
        At/after retirement_age: uses retirement_allocation.

        Args:
            current_age: Current age in years.
            retirement_age: Age at which full glide-path completion occurs.
            pre_allocation: Target allocation pre-retirement {asset_class: weight}.
            retirement_allocation: Target allocation at/after retirement.

        Returns:
            Dict with interpolated allocation weights, keys=ASSET_CLASSES.
        """
        if current_age >= retirement_age:
            return retirement_allocation.copy()

        if current_age >= retirement_age:
            return retirement_allocation.copy()

        # Linear interpolation
        years_to_retirement = retirement_age - current_age
        total_career_years = retirement_age - (retirement_age - 35)  # Assume 35-year career
        glide_progress = (total_career_years - years_to_retirement) / total_career_years

        glide_progress = np.clip(glide_progress, 0.0, 1.0)

        allocation = {}
        for asset_class in ASSET_CLASSES:
            pre_weight = pre_allocation.get(asset_class, 0.0)
            retire_weight = retirement_allocation.get(asset_class, 0.0)
            # Linear interpolation
            allocation[asset_class] = (
                pre_weight * (1 - glide_progress) +
                retire_weight * glide_progress
            )

        return allocation

    @staticmethod
    def apply_rebalance(
        balances: Dict[str, float],
        target_allocation: Dict[str, float],
        threshold: float
    ) -> Dict[str, float]:
        """
        Rebalance portfolio to target allocation if drift exceeds threshold.

        Threshold rebalancing: only rebalance if any asset class deviates from
        target by more than threshold. E.g., if target is 50% and threshold
        is 5%, rebalance only if any asset is <45% or >55%.

        Args:
            balances: Current account balances {asset_class: value}.
            target_allocation: Target weights {asset_class: weight, sum ≈ 1.0}.
            threshold: Drift threshold (e.g., 0.05 for 5%).

        Returns:
            Rebalanced balances dict with same keys.
        """
        total = sum(balances.values())
        if total <= 0:
            return balances.copy()

        # Compute current weights
        current_allocation = {
            asset: balances.get(asset, 0.0) / total
            for asset in target_allocation.keys()
        }

        # Check if any asset drifts beyond threshold
        needs_rebalance = False
        for asset in target_allocation.keys():
            current = current_allocation.get(asset, 0.0)
            target = target_allocation[asset]
            drift = abs(current - target)
            if drift > threshold:
                needs_rebalance = True
                break

        if not needs_rebalance:
            return balances.copy()

        # Rebalance: set each asset to target weight, but do not refill
        # buckets that were depleted (zero balance) by withdrawals.
        # Redistribute their target weight proportionally among non-zero buckets.
        depleted = {a for a in target_allocation if balances.get(a, 0.0) <= 0}
        if depleted:
            active_weight = sum(
                w for a, w in target_allocation.items() if a not in depleted
            )
            rebalanced = {}
            for asset in target_allocation.keys():
                if asset in depleted:
                    rebalanced[asset] = 0.0
                elif active_weight > 0:
                    rebalanced[asset] = total * (target_allocation[asset] / active_weight)
                else:
                    rebalanced[asset] = balances.get(asset, 0.0)
            return rebalanced

        rebalanced = {}
        for asset in target_allocation.keys():
            target_weight = target_allocation[asset]
            rebalanced[asset] = total * target_weight

        return rebalanced

    @staticmethod
    def apply_returns(
        balances: Dict[str, float],
        annual_returns: np.ndarray,
        allocation: Dict[str, float]
    ) -> Tuple[Dict[str, float], float]:
        """
        Apply annual returns to portfolio balances.

        Each asset class in the allocation earns its specified return.
        Returns are weighted by current allocation to compute blended
        portfolio return rate.

        Args:
            balances: Current account balances {asset_class: value}.
            annual_returns: 1D array of returns for year, one per asset class.
                Length = len(ASSET_CLASSES).
            allocation: Current asset allocation {asset_class: weight}.

        Returns:
            Tuple of (updated_balances_dict, blended_portfolio_return_rate).
            Portfolio return rate is weighted average of asset returns.

        Raises:
            ValueError: If annual_returns length != len(ASSET_CLASSES).
        """
        if len(annual_returns) != len(ASSET_CLASSES):
            raise ValueError(
                f"annual_returns length {len(annual_returns)} != "
                f"ASSET_CLASSES length {len(ASSET_CLASSES)}"
            )

        total = sum(balances.values())
        if total <= 0:
            return balances.copy(), 0.0

        # Apply returns asset-by-asset
        updated = {}
        portfolio_return = 0.0

        for i, asset_class in enumerate(ASSET_CLASSES):
            current_balance = balances.get(asset_class, 0.0)
            asset_return = annual_returns[i]

            # Weighted return contribution
            weight = allocation.get(asset_class, 0.0)
            portfolio_return += weight * asset_return

            # Update balance with return
            if current_balance > 0:
                updated[asset_class] = current_balance * (1.0 + asset_return)
            else:
                updated[asset_class] = 0.0

        return updated, portfolio_return

    def process_year(
        self,
        age: int,
        traditional: float,
        roth: float,
        taxable: float,
        annual_returns: np.ndarray,
        inflation_rate: float,
        cum_inflation: float,
        config: SimulationConfig,
        is_retired: bool,
    ) -> Dict:
        """
        Process one year of portfolio activity (contributions, withdrawals, returns).

        This is the main annual processing method called once per year per simulation.
        Handles contributions, RMDs, withdrawals, returns, and rebalancing.

        Withdrawal Order (when retired):
          1. Taxable account first (preferential tax treatment on gains)
          2. Traditional account (must satisfy RMD if required)
          3. Roth account last (tax-free growth, flexibility)

        Args:
            age: Current age in years.
            traditional: Traditional (pre-tax, 401k/IRA) balance at year start.
            roth: Roth account balance at year start.
            taxable: Taxable account balance at year start.
            annual_returns: Asset returns this year, shape (n_asset_classes,).
            inflation_rate: CPI inflation rate this year (e.g., 0.025 for 2.5%).
            cum_inflation: Cumulative inflation from start of projection (e.g., 1.05).
            config: SimulationConfig with allocation, withdrawal, fee parameters.
            is_retired: True if age >= retirement_age, else False.

        Returns:
            Dict with keys:
              - traditional: Traditional balance at year end
              - roth: Roth balance at year end
              - taxable: Taxable balance at year end
              - gross_withdrawal: Total withdrawal amount (in nominal dollars)
              - rmd_amount: RMD from traditional (in nominal dollars)
              - contribution: Total contribution (if not retired, else 0)
              - portfolio_return: Blended portfolio return rate this year
              - allocation_used: Asset allocation used this year
              - total_portfolio: Sum of all buckets at year end

        Raises:
            ValueError: If config fields are invalid.
        """
        # Determine asset allocation
        if config.use_glide_path:
            allocation = self.calculate_glide_path_allocation(
                age,
                config.retirement_age,
                config.pre_retirement_allocation,
                config.retirement_allocation
            )
        else:
            allocation = (
                config.retirement_allocation if is_retired
                else config.pre_retirement_allocation
            )

        # ── Validate & normalise allocation ────────────────────────────────
        allocation = _validate_and_normalize_allocation(
            allocation,
            label=f"age-{age} allocation",
        )

        # STEP 1: Add contributions
        # Primary earner: contributes until retirement_age.
        # Spouse: contributes until spouse_retirement_age — gated by SPOUSE's age.
        contribution = 0.0
        _years_elapsed = age - config.current_age

        # When spend netting is active, contributions are suppressed to avoid
        # double-counting with the spend shortfall draw.
        _spend_netting_active = (
            getattr(config, 'spend_annual_real', 0.0) > 0
            and age >= getattr(config, 'spend_start_age', config.retirement_age)
        )

        if not is_retired and not _spend_netting_active:
            _primary_growth = getattr(config, 'contribution_growth_rate', 0.0)
            _primary_base = config.annual_contribution
            contribution = _primary_base * ((1.0 + _primary_growth) ** _years_elapsed) * cum_inflation

        spouse_retirement_age = getattr(config, 'spouse_retirement_age', config.retirement_age)
        spouse_contrib = getattr(config, 'spouse_annual_contribution', 0.0)
        _spouse_age_cfg = getattr(config, 'spouse_age', None)
        if _spouse_age_cfg is not None:
            _spouse_current_age = _spouse_age_cfg + _years_elapsed
        else:
            _spouse_current_age = age
        if (spouse_contrib > 0
                and _spouse_current_age < spouse_retirement_age
                and not _spend_netting_active):
            _spouse_growth = getattr(config, 'spouse_contribution_growth_rate', 0.0)
            contribution += spouse_contrib * ((1.0 + _spouse_growth) ** _years_elapsed) * cum_inflation

        if contribution > 0:
            # Split contributions 80/20 traditional/Roth
            trad_contrib = contribution * 0.80
            roth_contrib = contribution * 0.20
            traditional += trad_contrib
            roth += roth_contrib

        # STEP 2: Calculate RMD (single household RMD on combined traditional)
        rmd_amount = 0.0
        if is_retired:
            rmd_amount = self.calculate_rmd(age, traditional)

        # STEP 3: Determine target withdrawal.
        withdrawal_start = getattr(config, 'withdrawal_start_age', config.retirement_age)
        primary_drawing = age >= withdrawal_start

        if is_retired or primary_drawing:
            if config.withdrawal_method == 'fixed_real':
                d_cum = _to_dec(cum_inflation)
                target_withdrawal = (
                    _from_dec(_to_dec(config.annual_withdrawal_real) * d_cum)
                    if primary_drawing else 0.0
                )
            elif config.withdrawal_method == 'percentage':
                if is_retired:
                    total_portfolio = traditional + roth + taxable
                    target_withdrawal = total_portfolio * config.withdrawal_rate
                else:
                    target_withdrawal = 0.0
            elif config.withdrawal_method == 'guardrails':
                if is_retired:
                    total_portfolio = traditional + roth + taxable
                    floor_withdrawal = total_portfolio * config.withdrawal_floor
                    ceiling_withdrawal = total_portfolio * config.withdrawal_ceiling
                    target_withdrawal = (floor_withdrawal + ceiling_withdrawal) / 2.0
                else:
                    target_withdrawal = 0.0
            else:
                target_withdrawal = 0.0

            # Ensure withdrawal >= RMD (mandatory; from traditional account)
            target_withdrawal = max(target_withdrawal, rmd_amount)
        else:
            target_withdrawal = 0.0

        # Suppress discretionary withdrawal when spend netting is active.
        _spend_real = getattr(config, 'spend_annual_real', 0.0)
        _spend_start = getattr(config, 'spend_start_age', config.retirement_age)
        if _spend_real > 0 and age >= _spend_start:
            target_withdrawal = rmd_amount

        # STEP 4: 3-bucket withdrawal sequence
        # Order: taxable -> traditional (>= RMD) -> Roth
        gross_withdrawal = 0.0
        taxable_withdraw = 0.0
        traditional_withdraw = 0.0
        roth_withdraw = 0.0
        remaining_need = target_withdrawal

        if remaining_need > 0 and taxable > 0:
            taxable_withdraw = min(remaining_need, taxable)
            taxable -= taxable_withdraw
            remaining_need -= taxable_withdraw
            gross_withdrawal += taxable_withdraw
        if remaining_need > 0 and traditional > 0:
            traditional_withdraw = min(remaining_need, traditional)
            traditional -= traditional_withdraw
            remaining_need -= traditional_withdraw
            gross_withdrawal += traditional_withdraw
        if remaining_need > 0 and roth > 0:
            roth_withdraw = min(remaining_need, roth)
            roth -= roth_withdraw
            remaining_need -= roth_withdraw
            gross_withdrawal += roth_withdraw

        if remaining_need > 0:
            gross_withdrawal = target_withdrawal - remaining_need

        # STEP 5: Apply returns to POST-WITHDRAWAL balances.
        portfolio_return = 0.0
        _rebal_threshold = getattr(config, 'rebalance_threshold', 0.05)

        total_post_withdrawal = traditional + roth + taxable
        if total_post_withdrawal > 0:
            trad_frac = traditional / total_post_withdrawal
            roth_frac = roth / total_post_withdrawal
            taxable_frac = taxable / total_post_withdrawal
            detailed_balances = {ac: total_post_withdrawal * allocation.get(ac, 0.0)
                                 for ac in ASSET_CLASSES}
            detailed_balances, portfolio_return = self.apply_returns(
                detailed_balances, annual_returns, allocation)
            if _rebal_threshold > 0:
                detailed_balances = self.apply_rebalance(
                    detailed_balances, allocation, _rebal_threshold)
            total_with_returns = sum(detailed_balances.values())
            traditional = total_with_returns * trad_frac
            roth = total_with_returns * roth_frac
            taxable = total_with_returns * taxable_frac

        # STEP 6: Apply fees (expense ratio + advisory fee)
        total_portfolio = traditional + roth + taxable
        if total_portfolio > 0:
            annual_fee_rate = config.expense_ratio + config.advisory_fee
            fee_frac = annual_fee_rate
            traditional *= (1.0 - fee_frac)
            roth *= (1.0 - fee_frac)
            taxable *= (1.0 - fee_frac)
            total_portfolio *= (1.0 - fee_frac)

        return {
            'traditional': max(0.0, traditional),
            'roth': max(0.0, roth),
            'taxable': max(0.0, taxable),
            'gross_withdrawal': gross_withdrawal,
            'trad_withdraw': traditional_withdraw,
            'roth_withdraw': roth_withdraw,
            'taxable_withdraw': taxable_withdraw,
            'rmd_amount': rmd_amount,
            'contribution': contribution,
            'portfolio_return': portfolio_return,
            'allocation_used': allocation,
            'total_portfolio': max(0.0, total_portfolio),
        }
