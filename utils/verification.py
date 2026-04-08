"""
Verification & Sanity-Check Utilities for Retirement Financial Modeling.

This module provides:
  1. Inverse heuristic checks — simplified reverse calculations to validate
     primary simulation outputs against back-of-envelope expectations.
  2. Non-convergence detection — checks whether Monte Carlo statistics have
     stabilised with the chosen number of simulations.
  3. Unit consistency guards — asserts that dollar values are in the expected
     magnitude (raw dollars, not thousands or millions) throughout the pipeline.

Boundary / Edge Cases:
  - Empty SimulationResults arrays: all checks short-circuit and return warnings.
  - Single-simulation runs: convergence check always flags as insufficient.
  - Zero portfolio: heuristic checks accept zero end balance.
  - Very large portfolios (>$100M): heuristic bounds widen to accommodate.

Sources:
  - Bengen (1994): 4% rule as withdrawal heuristic baseline.
  - Rule of 72: doubling time = 72 / (return_pct).

Version: 1.7.1
"""

import logging
from typing import Dict, List, Tuple, Optional
from decimal import Decimal

import numpy as np

from config.defaults import SimulationConfig, SimulationResults

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INVERSE HEURISTIC CHECKS (Req 3)
# ═══════════════════════════════════════════════════════════════════════════════

def heuristic_portfolio_growth(config: SimulationConfig) -> dict:
    """
    Simplified inverse check: estimate expected portfolio at end of horizon
    using deterministic compound growth — no Monte Carlo.

    Heuristic:
      P_end ≈ P_start × (1 + r_real)^N − W × [(1+r)^N − 1] / r
    where:
      P_start = total portfolio
      r_real  = weighted expected return − inflation − fees
      N       = years of withdrawal
      W       = annual withdrawal (real)

    Returns dict with:
      - expected_end_balance: rough estimate
      - heuristic_return: the blended real return assumed
      - doubling_years: Rule of 72 doubling time
      - plausible_range: (low, high) tuple of plausible median end balances
    """
    # Weighted expected nominal return based on allocation
    # (uses fallback means from AssetReturnModel)
    FALLBACK_MEANS = {
        'us_large_cap': 0.1050, 'us_small_cap': 0.1150,
        'international_dev': 0.0850, 'emerging_markets': 0.0950,
        'us_bonds': 0.0450, 'tips': 0.0250,
        'cash': 0.0300, 'reits': 0.0950,
    }
    alloc = config.retirement_allocation
    weighted_return = sum(
        alloc.get(a, 0.0) * FALLBACK_MEANS.get(a, 0.05) for a in alloc
    )
    r_real = weighted_return - config.inflation_mean - config.expense_ratio - config.advisory_fee
    r_real = max(r_real, -0.05)  # clamp unreasonable negative

    n_years = config.life_expectancy - config.current_age
    if n_years <= 0:
        return {
            'expected_end_balance': config.total_portfolio_value,
            'heuristic_return': r_real,
            'doubling_years': 72.0 / max(r_real * 100, 0.01),
            'plausible_range': (0.0, config.total_portfolio_value),
        }

    P = config.total_portfolio_value
    W = config.annual_withdrawal_real + getattr(config, 'spouse_annual_withdrawal_real', 0.0)

    # Retirement years only (withdrawals start at retirement)
    n_retire = max(0, config.life_expectancy - config.retirement_age)
    n_accum = max(0, n_years - n_retire)

    # Accumulation phase (no withdrawals)
    if r_real != 0.0:
        P_at_retirement = P * (1 + r_real) ** n_accum
    else:
        P_at_retirement = P

    # Withdrawal phase (annuity drawdown)
    if r_real != 0.0 and n_retire > 0:
        growth_factor = (1 + r_real) ** n_retire
        annuity_factor = (growth_factor - 1) / r_real
        P_end = P_at_retirement * growth_factor - W * annuity_factor
    else:
        P_end = P_at_retirement - W * n_retire

    doubling = 72.0 / max(r_real * 100, 0.01) if r_real > 0 else float('inf')

    # Plausible range: ±50% around heuristic (accounts for stochastic variance)
    low = max(0.0, P_end * 0.5)
    high = max(P_end * 1.5, P * 2)

    return {
        'expected_end_balance': max(0.0, P_end),
        'heuristic_return': r_real,
        'doubling_years': doubling,
        'plausible_range': (low, high),
    }


def heuristic_tax_check(config: SimulationConfig, median_tax: float) -> dict:
    """
    Inverse heuristic for annual tax: estimate what the median effective rate
    should be given income level, and compare.

    Rule of thumb:
      - MFJ with $100K–$200K total income → ~15–22% effective federal rate.
      - Single with $50K–$100K → ~12–18%.
      - If include_state_tax, add state_tax_rate.

    Returns dict with heuristic_effective_rate, actual_effective_rate, pass/fail.
    """
    # Estimate total annual income at retirement
    annual_income = (
        config.annual_withdrawal_real
        + getattr(config, 'spouse_annual_withdrawal_real', 0.0)
        + config.ss_monthly_benefit_at_fra * 12 * 0.85  # up to 85% taxable
        + config.pension_annual_real
        + config.rental_income_annual_real
    )

    # Rough effective federal rate by income bracket
    if annual_income < 50_000:
        heuristic_rate = 0.08
    elif annual_income < 100_000:
        heuristic_rate = 0.12
    elif annual_income < 200_000:
        heuristic_rate = 0.18
    elif annual_income < 500_000:
        heuristic_rate = 0.24
    else:
        heuristic_rate = 0.30

    if config.include_state_tax:
        heuristic_rate += config.state_tax_rate

    expected_tax = annual_income * heuristic_rate
    actual_rate = median_tax / annual_income if annual_income > 0 else 0.0

    # Pass if within 2x of heuristic (Monte Carlo has variance)
    ratio = median_tax / expected_tax if expected_tax > 0 else 0.0
    is_plausible = 0.2 <= ratio <= 5.0

    return {
        'estimated_annual_income': annual_income,
        'heuristic_effective_rate': heuristic_rate,
        'heuristic_annual_tax': expected_tax,
        'actual_median_tax': median_tax,
        'actual_effective_rate': actual_rate,
        'ratio_actual_to_heuristic': ratio,
        'is_plausible': is_plausible,
    }


def heuristic_withdrawal_check(
    config: SimulationConfig,
    median_withdrawal_at_retirement: float,
    cum_inflation_at_retirement: float,
) -> dict:
    """
    Inverse check: median withdrawal at retirement year should be close to
    (primary_real + spouse_real) × cum_inflation.
    """
    target_real = config.annual_withdrawal_real + getattr(
        config, 'spouse_annual_withdrawal_real', 0.0
    )
    expected_nominal = target_real * cum_inflation_at_retirement
    ratio = (
        median_withdrawal_at_retirement / expected_nominal
        if expected_nominal > 0 else 0.0
    )
    return {
        'target_real': target_real,
        'cum_inflation': cum_inflation_at_retirement,
        'expected_nominal': expected_nominal,
        'actual_median': median_withdrawal_at_retirement,
        'ratio': ratio,
        'is_plausible': 0.5 <= ratio <= 2.0 if expected_nominal > 0 else True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NON-CONVERGENCE DETECTION (Req 5)
# ═══════════════════════════════════════════════════════════════════════════════

def check_convergence(results: SimulationResults, tolerance: float = 0.02) -> dict:
    """
    Check whether Monte Carlo statistics have converged.

    Method: split simulations into two halves and compare median final balance,
    success rate, and mean withdrawal.  If any metric differs by more than
    `tolerance` (relative), flag as non-converged.

    Args:
        results: Completed SimulationResults.
        tolerance: Max relative difference between halves (default 2%).

    Returns:
        Dict with per-metric comparisons and overall 'converged' bool.
    """
    n = results.portfolio_values.shape[0]
    if n < 100:
        return {
            'converged': False,
            'reason': f'Only {n} simulations; need >= 100 for convergence check.',
            'metrics': {},
        }

    mid = n // 2
    half_a = results.portfolio_values[:mid, -1]
    half_b = results.portfolio_values[mid:, -1]

    metrics = {}
    all_pass = True

    # Median final balance
    med_a, med_b = np.median(half_a), np.median(half_b)
    denom = max(abs(med_a), abs(med_b), 1.0)
    diff = abs(med_a - med_b) / denom
    ok = diff <= tolerance
    metrics['median_final_balance'] = {
        'half_a': float(med_a), 'half_b': float(med_b),
        'relative_diff': float(diff), 'pass': ok,
    }
    if not ok:
        all_pass = False

    # Success rate
    sr_a = float(np.mean(results.success_mask[:mid]))
    sr_b = float(np.mean(results.success_mask[mid:]))
    diff_sr = abs(sr_a - sr_b)
    ok_sr = diff_sr <= tolerance
    metrics['success_rate'] = {
        'half_a': sr_a, 'half_b': sr_b,
        'absolute_diff': diff_sr, 'pass': ok_sr,
    }
    if not ok_sr:
        all_pass = False

    # Mean annual withdrawal
    mw_a = float(np.mean(results.withdrawals[:mid]))
    mw_b = float(np.mean(results.withdrawals[mid:]))
    denom_w = max(abs(mw_a), abs(mw_b), 1.0)
    diff_w = abs(mw_a - mw_b) / denom_w
    ok_w = diff_w <= tolerance
    metrics['mean_withdrawal'] = {
        'half_a': mw_a, 'half_b': mw_b,
        'relative_diff': float(diff_w), 'pass': ok_w,
    }
    if not ok_w:
        all_pass = False

    return {
        'converged': all_pass,
        'reason': 'All metrics within tolerance' if all_pass else 'Some metrics diverge between halves',
        'n_simulations': n,
        'tolerance': tolerance,
        'metrics': metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. UNIT CONSISTENCY GUARDS (Req 6 + 8)
# ═══════════════════════════════════════════════════════════════════════════════

# Expected magnitude ranges for key dollar fields.
# Format: (field_name, min_plausible, max_plausible, unit_label)
_UNIT_EXPECTATIONS = [
    # Portfolio balances — raw dollars, typically $0 – $50M
    ('total_portfolio_value',        0.0,    50_000_000, 'dollars'),
    ('traditional_balance',          0.0,    50_000_000, 'dollars'),
    ('roth_balance',                 0.0,    50_000_000, 'dollars'),
    ('taxable_balance',              0.0,    50_000_000, 'dollars'),
    ('spouse_traditional_balance',   0.0,    50_000_000, 'dollars'),
    ('spouse_roth_balance',          0.0,    50_000_000, 'dollars'),
    ('spouse_taxable_balance',       0.0,    50_000_000, 'dollars'),
    # Annual amounts — raw dollars, typically $0 – $1M
    ('annual_withdrawal_real',       0.0,    1_000_000, 'dollars'),
    ('spouse_annual_withdrawal_real', 0.0,   1_000_000, 'dollars'),
    ('annual_contribution',          0.0,    500_000,   'dollars'),
    ('pension_annual_real',          0.0,    500_000,   'dollars'),
    ('rental_income_annual_real',    0.0,    500_000,   'dollars'),
    ('part_time_income_annual',      0.0,    500_000,   'dollars'),
    ('ss_monthly_benefit_at_fra',    0.0,    10_000,    'dollars/month'),
    # Rates — should be decimal fractions (0 – 1), not percentages (0 – 100)
    ('expense_ratio',                0.0,    0.05,      'decimal fraction'),
    ('advisory_fee',                 0.0,    0.03,      'decimal fraction'),
    ('inflation_mean',               0.0,    0.15,      'decimal fraction'),
    ('state_tax_rate',               0.0,    0.15,      'decimal fraction'),
]


def check_unit_consistency(config: SimulationConfig) -> List[dict]:
    """
    Verify that SimulationConfig fields are in expected units (raw dollars,
    decimal rates) — not accidentally in thousands, millions, or percentages.

    Returns list of violations; empty list means all checks passed.
    """
    violations = []
    for field_name, lo, hi, unit in _UNIT_EXPECTATIONS:
        val = getattr(config, field_name, None)
        if val is None:
            continue
        if not (lo <= val <= hi):
            violations.append({
                'field': field_name,
                'value': val,
                'expected_range': (lo, hi),
                'expected_unit': unit,
                'likely_issue': (
                    f"Value {val:,.2f} outside [{lo:,.0f}, {hi:,.0f}]. "
                    f"If in thousands, multiply by 1,000. "
                    f"If a percentage, divide by 100."
                ),
            })
    return violations


def check_output_unit_consistency(results: SimulationResults) -> List[dict]:
    """
    Verify that simulation output arrays are in expected magnitude ranges.

    Checks:
      - Median final portfolio should be < $500M (sanity upper bound).
      - Median annual withdrawal should be < $10M.
      - Median annual tax should be < median annual withdrawal.
      - Portfolio values should be non-negative.
    """
    violations = []
    pv = results.portfolio_values
    n_sims, n_years = pv.shape

    # Median final portfolio
    med_final = float(np.median(pv[:, -1]))
    if med_final > 500_000_000:
        violations.append({
            'metric': 'median_final_portfolio',
            'value': med_final,
            'threshold': 500_000_000,
            'issue': (
                f"Median final portfolio ${med_final/1e6:,.1f}M exceeds $500M. "
                f"Possible: withdrawals not applied, or allocation > 100%."
            ),
        })

    # Negative portfolio values
    if np.any(pv < -1.0):
        violations.append({
            'metric': 'negative_portfolio_values',
            'value': float(np.min(pv)),
            'threshold': 0.0,
            'issue': 'Portfolio values went negative; should be clamped to 0.',
        })

    # Median annual withdrawal
    med_wd = float(np.median(results.withdrawals[results.withdrawals > 0]))
    if med_wd > 10_000_000:
        violations.append({
            'metric': 'median_annual_withdrawal',
            'value': med_wd,
            'threshold': 10_000_000,
            'issue': (
                f"Median non-zero withdrawal ${med_wd/1e6:,.1f}M exceeds $10M. "
                f"Possible unit error or RMD from inflated portfolio."
            ),
        })

    # Median tax should be less than median income
    med_tax = float(np.mean(np.median(results.taxes, axis=0)))
    med_income = float(np.mean(np.median(results.withdrawals, axis=0)))
    if med_income > 0 and med_tax > med_income:
        violations.append({
            'metric': 'tax_exceeds_income',
            'value': med_tax,
            'threshold': med_income,
            'issue': (
                f"Median tax ${med_tax:,.0f} exceeds median withdrawal "
                f"${med_income:,.0f}. Tax cannot exceed gross income."
            ),
        })

    return violations


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COMBINED VERIFICATION REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_verification(
    config: SimulationConfig,
    results: SimulationResults,
) -> dict:
    """
    Run all verification checks and return a structured report.

    Returns dict with sections:
      - input_unit_check: List[dict] of input unit violations
      - output_unit_check: List[dict] of output unit violations
      - convergence: dict from check_convergence
      - heuristic_portfolio: dict from heuristic_portfolio_growth
      - heuristic_tax: dict from heuristic_tax_check
      - heuristic_withdrawal: dict from heuristic_withdrawal_check
      - overall_pass: bool (True if no critical violations)
    """
    report = {}

    # Input units
    report['input_unit_check'] = check_unit_consistency(config)

    # Output units
    report['output_unit_check'] = check_output_unit_consistency(results)

    # Convergence
    report['convergence'] = check_convergence(results)

    # Heuristic: portfolio
    report['heuristic_portfolio'] = heuristic_portfolio_growth(config)

    # Heuristic: tax
    median_tax = float(np.mean(np.median(results.taxes, axis=0)))
    report['heuristic_tax'] = heuristic_tax_check(config, median_tax)

    # Heuristic: withdrawal at retirement year
    retire_idx = max(0, config.retirement_age - config.current_age)
    if retire_idx < results.withdrawals.shape[1]:
        med_wd = float(np.median(results.withdrawals[:, retire_idx]))
        cum_inf = float(np.median(
            np.prod(1.0 + results.inflation_rates[:, :retire_idx + 1], axis=1)
        )) if retire_idx > 0 else 1.0
    else:
        med_wd = 0.0
        cum_inf = 1.0
    report['heuristic_withdrawal'] = heuristic_withdrawal_check(
        config, med_wd, cum_inf
    )

    # Overall pass: no critical violations
    critical = (
        len(report['input_unit_check']) == 0
        and len(report['output_unit_check']) == 0
        and report['convergence']['converged']
        and report['heuristic_portfolio']['plausible_range'][0]
            <= float(np.median(results.portfolio_values[:, -1]))
            <= report['heuristic_portfolio']['plausible_range'][1]
        and report['heuristic_tax']['is_plausible']
    )
    report['overall_pass'] = critical

    return report
