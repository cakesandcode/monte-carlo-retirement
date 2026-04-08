"""
Shared utility functions for retirement financial modeling.

Provides currency formatting, percentage formatting, age formatting, real/nominal
conversions, inflation calculations, percentile analysis, and validation functions
used across the codebase.

Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional


def fmt_dollar(value: float) -> str:
    """
    Unified monetary formatter — auto-selects K or M based on magnitude.

    Thresholds:
      >= $1,000,000  → "$X.XXM"  (2 dp)
      >= $1,000      → "$X.XK"   (1 dp)
      < $1,000       → "$X"      (0 dp)

    Consistency rule: the SAME value always produces the SAME string
    regardless of whether it is an input or an output.

    Args:
        value: Dollar amount (raw, NOT pre-divided).

    Returns:
        Formatted string, e.g., "$2.45M", "$80.0K", "$500".
    """
    abs_v = abs(value)
    sign = "-" if value < 0 else ""
    if abs_v >= 1_000_000:
        return f"{sign}${abs_v / 1_000_000:,.2f}M"
    elif abs_v >= 1_000:
        return f"{sign}${abs_v / 1_000:,.1f}K"
    return f"{sign}${abs_v:,.0f}"


# Legacy aliases — point to unified formatter so old call-sites still work.
# Both were inconsistent: _fmt_k always divided by 1K (even for $2.45M inputs),
# _fmt_m switched to M at $100K (so $150K showed as "$0.15M").
# Now both use the same magnitude-aware logic.
fmt_k = fmt_dollar
fmt_m = fmt_dollar


def format_currency(value: float, decimals: int = 0) -> str:
    """
    Format a numeric value as a currency string.

    Converts to US dollar format with thousands separators and optional decimal places.
    Handles negative values with leading minus sign.

    Args:
        value: Numeric value to format (float).
        decimals: Number of decimal places (default 0). Typical use: 0 for $1,234,567
            or 2 for $1,234,567.89.

    Returns:
        Formatted currency string, e.g., "$1,234,567" or "$1,234,567.89".

    Examples:
        >>> format_currency(1234567.5)
        '$1,234,568'
        >>> format_currency(1234567.5, decimals=2)
        '$1,234,567.50'
        >>> format_currency(-50000)
        '-$50,000'
    """
    is_negative = value < 0
    abs_value = abs(value)

    if decimals == 0:
        formatted = f"{abs_value:,.0f}"
    else:
        formatted = f"{abs_value:,.{decimals}f}"

    result = f"${formatted}"
    if is_negative:
        result = f"-{result}"

    return result


def format_percent(value: float, decimals: int = 1) -> str:
    """
    Format a decimal value as a percentage string.

    Converts decimal (e.g., 0.042) to percentage format (e.g., "4.2%").

    Args:
        value: Decimal value to format (e.g., 0.042 for 4.2%).
        decimals: Number of decimal places in percentage (default 1).
            Typical use: 1 for "4.2%" or 0 for "4%".

    Returns:
        Formatted percentage string, e.g., "4.2%".

    Examples:
        >>> format_percent(0.042)
        '4.2%'
        >>> format_percent(0.95, decimals=0)
        '95%'
        >>> format_percent(0.009, decimals=2)
        '0.90%'
    """
    percent_value = value * 100
    return f"{percent_value:.{decimals}f}%"


def format_age(age: int) -> str:
    """
    Format an age as a human-readable string.

    Args:
        age: Age in years (integer).

    Returns:
        Formatted string, e.g., "Age 65".

    Examples:
        >>> format_age(65)
        'Age 65'
        >>> format_age(45)
        'Age 45'
    """
    return f"Age {age}"


def real_to_nominal(real_value: float, cum_inflation: float) -> float:
    """
    Convert a real (inflation-adjusted) value to nominal dollars.

    Given cumulative inflation multiplier (product of (1 + inflation_rate) over time),
    scales the real value to its nominal equivalent.

    Args:
        real_value: Value in today's dollars (real terms).
        cum_inflation: Cumulative inflation multiplier, e.g., 1.25 for 25% total
            inflation. Typically computed as running product of (1 + r).

    Returns:
        Nominal value in future dollars.

    Examples:
        >>> real_to_nominal(100000, 1.05)  # 5% cumulative inflation
        105000.0
        >>> real_to_nominal(50000, 1.20)  # 20% cumulative inflation
        60000.0
    """
    return real_value * cum_inflation


def nominal_to_real(nominal_value: float, cum_inflation: float) -> float:
    """
    Convert a nominal (future) value to real (inflation-adjusted) dollars.

    Given cumulative inflation multiplier, scales the nominal value back to
    today's dollars.

    Args:
        nominal_value: Value in future dollars (nominal terms).
        cum_inflation: Cumulative inflation multiplier, e.g., 1.25 for 25% total
            inflation.

    Returns:
        Real value in today's dollars.

    Examples:
        >>> nominal_to_real(105000, 1.05)  # 5% cumulative inflation
        100000.0
        >>> nominal_to_real(60000, 1.20)  # 20% cumulative inflation
        50000.0
    """
    if cum_inflation == 0:
        return 0.0
    return nominal_value / cum_inflation


def cumulative_inflation(annual_rates: np.ndarray) -> np.ndarray:
    """
    Compute cumulative inflation multipliers from annual inflation rates.

    Given an array of annual inflation rates, computes the running product of
    (1 + rate) to produce cumulative inflation multipliers. Useful for converting
    between real and nominal values over time.

    Args:
        annual_rates: Array of annual inflation rates (1D or 2D).
            If 1D shape (n_years,): returns 1D array of cumulative multipliers.
            If 2D shape (n_sims, n_years): returns 2D array.

    Returns:
        Cumulative inflation array with same shape as input.
        Each element is the product of (1 + rate) up to that year.
        First element is always 1.0 (year 0 has no inflation yet).

    Examples:
        >>> rates = np.array([0.03, 0.03, 0.02])
        >>> cumulative_inflation(rates)
        array([1.0, 1.03, 1.0609, 1.082927])
    """
    if annual_rates.ndim == 1:
        # 1D case: single path
        n_years = len(annual_rates)
        cum_inf = np.ones(n_years + 1)
        for i in range(n_years):
            cum_inf[i + 1] = cum_inf[i] * (1.0 + annual_rates[i])
        return cum_inf[1:]  # Return n_years elements
    elif annual_rates.ndim == 2:
        # 2D case: multiple paths
        n_sims, n_years = annual_rates.shape
        cum_inf = np.ones((n_sims, n_years))
        for sim_idx in range(n_sims):
            cumulative = 1.0
            for year_idx in range(n_years):
                cumulative *= (1.0 + annual_rates[sim_idx, year_idx])
                cum_inf[sim_idx, year_idx] = cumulative
        return cum_inf
    else:
        raise ValueError(f"annual_rates must be 1D or 2D, got shape {annual_rates.shape}")


def calculate_percentiles(data: np.ndarray, percentiles: List[int]) -> Dict[int, np.ndarray]:
    """
    Calculate percentiles for each time step in simulation data.

    Given data array (typically shape (n_sims, n_years)), computes requested
    percentiles along the simulation dimension, returning percentile bands
    across time.

    Args:
        data: Array of simulation results, shape (n_sims, n_years) or (n_sims, n_time_steps).
        percentiles: List of percentile levels (0-100), e.g., [10, 25, 50, 75, 90].

    Returns:
        Dict mapping percentile -> array of shape (n_years,).
        Each array contains the percentile value at each time step.

    Examples:
        >>> data = np.random.randn(1000, 30)
        >>> p = calculate_percentiles(data, [10, 50, 90])
        >>> p[50].shape
        (30,)
    """
    result = {}
    for perc in percentiles:
        result[perc] = np.percentile(data, perc, axis=0)
    return result


def years_to_retirement(current_age: int, retirement_age: int) -> int:
    """
    Calculate years remaining until retirement.

    Args:
        current_age: Current age in years.
        retirement_age: Target retirement age.

    Returns:
        Number of years until retirement (non-negative).
        Returns 0 if already at or past retirement_age.

    Examples:
        >>> years_to_retirement(55, 65)
        10
        >>> years_to_retirement(67, 65)
        0
    """
    return max(0, retirement_age - current_age)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default if denominator is zero.

    Prevents division-by-zero errors in financial calculations.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value to return if denominator is zero or very close to it
            (default 0.0).

    Returns:
        numerator / denominator, or default if denominator is too close to zero.

    Examples:
        >>> safe_divide(100, 5)
        20.0
        >>> safe_divide(100, 0)
        0.0
        >>> safe_divide(100, 0, default=1.0)
        1.0
    """
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def validate_allocation(allocation: Dict[str, float], tolerance: float = 0.01) -> bool:
    """
    Validate that an allocation dictionary sums to approximately 1.0 (100%).

    Used to check asset allocation percentages and other portfolio weights.

    Args:
        allocation: Dict mapping asset class names to weights (e.g., {
            'stocks': 0.60, 'bonds': 0.40 }).
        tolerance: Maximum absolute deviation from 1.0 to consider valid
            (default 0.01, i.e., +/-1%).

    Returns:
        True if sum of allocation values is within 1.0 +/- tolerance.
        False otherwise.

    Examples:
        >>> validate_allocation({'stocks': 0.60, 'bonds': 0.40})
        True
        >>> validate_allocation({'stocks': 0.70, 'bonds': 0.40})
        False
        >>> validate_allocation({'stocks': 0.605, 'bonds': 0.395}, tolerance=0.01)
        True
    """
    total = sum(allocation.values())
    return abs(total - 1.0) <= tolerance


def normalize_allocation(allocation: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize an allocation dictionary so weights sum to exactly 1.0.

    Rescales all weights proportionally so their sum equals 1.0. Useful for
    correcting minor rounding errors or user input that doesn't sum to 100%.

    Args:
        allocation: Dict mapping asset class names to weights.
            Must have at least one non-zero value.

    Returns:
        New dict with same keys and normalized weights summing to 1.0.
        Returns original dict unchanged if sum is zero.

    Examples:
        >>> normalize_allocation({'stocks': 60, 'bonds': 40})
        {'stocks': 0.6, 'bonds': 0.4}
        >>> normalize_allocation({'stocks': 0.605, 'bonds': 0.395})
        {'stocks': 0.6054622406639004, 'bonds': 0.3945377593360996}
    """
    total = sum(allocation.values())
    if total == 0:
        return allocation.copy()
    return {key: val / total for key, val in allocation.items()}
