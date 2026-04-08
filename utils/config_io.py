"""
Configuration I/O — CSV serialization and deserialization for SimulationConfig.

Provides round-trip save/load of all SimulationConfig fields to a two-column
CSV (parameter, value).  Dict fields (asset allocations) are flattened with
dot-notation keys (e.g. pre_retirement_allocation.us_large_cap).
Optional fields (spouse_age, random_seed) serialise as empty string when None.

Usage:
    from utils.config_io import save_config_csv, load_config_csv, config_to_csv_bytes

    # Save
    save_config_csv(config, "/path/to/config.csv")

    # Load
    config = load_config_csv("/path/to/config.csv")

    # In-memory bytes for st.download_button
    csv_bytes = config_to_csv_bytes(config)

Version: 1.0.0
"""

import csv
import io
import dataclasses
from typing import Optional, Dict, Tuple

from config.defaults import SimulationConfig, ASSET_CLASSES


# -- Field metadata for labeled CSV output ------------------------------------
# Each key = SimulationConfig field name (or dot-notation for allocation).
# Value = (label, valid_range, notes).
# Used by 5-column CSV export: parameter, value, label, valid_range, notes.

_FIELD_METADATA: Dict[str, Tuple[str, str, str]] = {
    # Demographics
    'current_age':              ('Current Age',                  '25-75',     'Primary earner age'),
    'retirement_age':           ('Retirement Age',               '45-80',     'Target retirement age'),
    'life_expectancy':          ('Life Expectancy',              '75-100',    'Planning horizon'),
    'spouse_age':               ('Spouse Current Age',           '25-75 or blank', 'Blank if no spouse'),
    'spouse_life_expectancy':   ('Spouse Life Expectancy',       '75-100',    'Spouse planning horizon'),
    # Portfolio
    'total_portfolio_value':    ('Total Portfolio Value ($)',     'computed',  'Read-only; sum of balances'),
    'traditional_balance':      ('Traditional Balance ($)',      '>= 0',      '401k/IRA pre-tax'),
    'roth_balance':             ('Roth Balance ($)',             '>= 0',      'Roth IRA/401k'),
    'taxable_balance':          ('Taxable Balance ($)',          '>= 0',      'Brokerage account'),
    # Contributions
    'annual_contribution':      ('Annual Contribution ($)',      '>= 0',      "Today's dollars"),
    'contribution_growth_rate': ('Contribution Growth Rate',     '0.0-0.10',  'Decimal (e.g. 0.02 = 2%)'),
    # Withdrawals
    'annual_withdrawal_real':   ('Primary Withdrawal ($)',       '>= 0',      "Real $/year"),
    'withdrawal_start_age':     ('Primary Withdrawal Start Age', '40-85',     'Age primary begins draw'),
    'withdrawal_method':        ('Withdrawal Method',            'fixed_real|fixed_nominal|percentage|guardrails', ''),
    'withdrawal_rate':          ('Withdrawal Rate',              '0.005-0.10','Decimal (percentage method)'),
    'withdrawal_floor':         ('Guardrails Floor',             '0.0-0.10',  'Min % (guardrails)'),
    'withdrawal_ceiling':       ('Guardrails Ceiling',           '0.0-0.10',  'Max % (guardrails)'),
    # Allocation (dot-notation keys added below)
    'use_glide_path':           ('Use Glide Path',              'True|False', 'Linear transition'),
    'rebalance_threshold':      ('Rebalance Threshold',         '0.0-0.20',  'Drift threshold (e.g. 0.05 = 5%)'),
    # Simulation
    'simulation_method':        ('Simulation Method',           'bootstrap|gbm', ''),
    'n_simulations':            ('Number of Simulations',       '100-100000', 'Monte Carlo paths'),
    'random_seed':              ('Random Seed',                 'int or blank', 'For reproducibility'),
    # Tax
    'filing_status':            ('Filing Status',               'single|married_filing_jointly', ''),
    'state_tax_rate':           ('State Tax Rate',              '0.0-0.15',  'Decimal (e.g. 0.05 = 5%)'),
    'include_state_tax':        ('Include State Tax',           'True|False', ''),
    # Social Security
    'ss_monthly_benefit_at_fra':    ('SS Monthly Benefit at FRA ($)', '>= 0', 'Primary earner'),
    'ss_fra':                       ('SS Full Retirement Age',       '62-70', 'Primary'),
    'ss_claiming_age':              ('SS Claiming Age',              '62-70', 'Primary'),
    'spouse_ss_monthly_benefit_at_fra': ('Spouse SS Monthly at FRA ($)', '>= 0', ''),
    'spouse_ss_claiming_age':       ('Spouse SS Claiming Age',       '62-70', ''),
    'include_spousal_benefit':      ('Include Spousal Benefit',      'True|False', ''),
    # Primary income
    'pension_annual_real':      ('Pension Annual ($)',           '>= 0',      'Primary pension'),
    'pension_start_age':        ('Pension Start Age',           '45-95',     ''),
    'pension_end_age':          ('Pension End Age',             '45-100',    'Default 95 = lifetime'),
    'rental_income_annual_real':('Rental Income Annual ($)',     '>= 0',      'Net Schedule E'),
    'rental_start_age':         ('Rental Start Age',            '25-100',    ''),
    'rental_end_age':           ('Rental End Age',              '25-100',    ''),
    'part_time_income_annual':  ('Part-time Income Annual ($)', '>= 0',      'Primary earner'),
    'part_time_income_start_age': ('Part-time Start Age',       '25-100',    ''),
    'part_time_income_end_age': ('Part-time End Age',           '25-100',    ''),
    # Simulation start year
    'simulation_start_year':    ('Simulation Start Year',       '2020-2040', 'Calendar year for current_age'),
    # SERP -- primary
    'serp_2026':  ('SERP 2026 ($)', '>= 0', 'Nominal contractual'),
    'serp_2027':  ('SERP 2027 ($)', '>= 0', 'Nominal contractual'),
    'serp_2028':  ('SERP 2028 ($)', '>= 0', 'Nominal contractual'),
    'serp_2029':  ('SERP 2029 ($)', '>= 0', 'Nominal contractual'),
    'serp_2030':  ('SERP 2030 ($)', '>= 0', 'Nominal contractual'),
    'serp_2031':  ('SERP 2031 ($)', '>= 0', 'Nominal contractual'),
    'serp_2032':  ('SERP 2032 ($)', '>= 0', 'Nominal contractual'),
    'serp_2033':  ('SERP 2033 ($)', '>= 0', 'Nominal contractual'),
    # Real/nominal flags
    'pension_is_real':          ('Pension Is Real (COLA)',         'True|False', 'False = nominal/fixed'),
    'part_time_is_real':        ('Part-time Is Real',             'True|False', 'True = inflation-adjusted'),
    # Spend
    'spend_annual_real':   ('Annual Spend (Real $)',      '>= 0',      "Discretionary/lifestyle in today's dollars"),
    'spend_start_age':     ('Spend Start Age',            '25-100',    'Age annual spend begins'),
    'spend_surplus_mode':  ('Spend Surplus Mode',         'ignore|reinvest', 'What to do when income > spend'),
    'healthcare_is_real':  ('Healthcare Is Real',         'True|False', 'True = inflation-adjusted'),
    # Fees
    'expense_ratio':  ('Expense Ratio',  '0.0-0.05', 'Decimal (e.g. 0.001 = 10 bps)'),
    'advisory_fee':   ('Advisory Fee',   '0.0-0.03', 'Decimal (e.g. 0.005 = 50 bps)'),
    # Inflation
    'inflation_method':             ('Inflation Method',            'bootstrap|fixed|mean_reverting', ''),
    'inflation_mean':               ('Inflation Mean',              '0.0-0.10',  'Decimal'),
    'inflation_std':                ('Inflation Std Dev',           '0.0-0.05',  'Decimal'),
    'healthcare_inflation_premium': ('Healthcare Inflation Premium', '0.0-0.05', 'Extra above CPI'),
    'annual_healthcare_cost_real':  ('Healthcare Annual Cost ($)',   '>= 0',     "Today's dollars"),
    'include_healthcare_costs':     ('Include Healthcare Costs',    'True|False', ''),
}

# Add dot-notation keys for asset allocation (primary only)
for _alloc_prefix in ('pre_retirement_allocation', 'retirement_allocation'):
    _label_prefix = 'Pre-Ret' if 'pre' in _alloc_prefix else 'Ret'
    for _ac in ASSET_CLASSES:
        _key = f'{_alloc_prefix}.{_ac}'
        _FIELD_METADATA[_key] = (
            f'{_label_prefix} Alloc: {_ac}',
            '0.0-1.0 or 0-100',
            'Decimal weight; all must sum to 1.0',
        )

# -- Field classification ----------------------------------------------------
# Fields whose values are Dict[str, float]; serialised with dot-notation keys.
_DICT_FIELDS = {
    'pre_retirement_allocation', 'retirement_allocation',
}

# Fields typed Optional[int]; serialised as '' when None.
_OPTIONAL_INT_FIELDS = {'spouse_age', 'random_seed'}


# -- Serialisation ------------------------------------------------------------

def _config_to_rows(config: SimulationConfig) -> list:
    """
    Convert SimulationConfig to a list of 5-element tuples:
        (parameter, value, label, valid_range, notes)

    Handles all field types:
      - bool   -> 'True' / 'False'
      - int    -> string integer
      - float  -> string float (full precision)
      - str    -> as-is
      - Optional[int] -> '' when None, else string integer
      - Dict[str,float] -> expanded to dotted keys, one row per asset class
    """
    rows = []
    for f in dataclasses.fields(config):
        name = f.name
        val = getattr(config, name)

        if name in _DICT_FIELDS:
            # Expand dict to per-asset rows in ASSET_CLASSES order
            for asset in ASSET_CLASSES:
                key = f'{name}.{asset}'
                meta = _FIELD_METADATA.get(key, ('', '', ''))
                rows.append((key, str(val.get(asset, 0.0)), meta[0], meta[1], meta[2]))

        elif name in _OPTIONAL_INT_FIELDS:
            meta = _FIELD_METADATA.get(name, ('', '', ''))
            rows.append((name, '' if val is None else str(int(val)), meta[0], meta[1], meta[2]))

        elif isinstance(val, bool):
            # Check bool before int -- bool is a subclass of int
            meta = _FIELD_METADATA.get(name, ('', '', ''))
            rows.append((name, str(val), meta[0], meta[1], meta[2]))

        elif val is None:
            meta = _FIELD_METADATA.get(name, ('', '', ''))
            rows.append((name, '', meta[0], meta[1], meta[2]))

        else:
            meta = _FIELD_METADATA.get(name, ('', '', ''))
            rows.append((name, str(val), meta[0], meta[1], meta[2]))

    return rows


def config_to_csv_bytes(config: SimulationConfig) -> bytes:
    """
    Serialise SimulationConfig to CSV bytes (UTF-8, CRLF line endings).

    Produces 5-column output: parameter, value, label, valid_range, notes.
    Suitable for st.download_button(data=...).

    Args:
        config: SimulationConfig to serialise.

    Returns:
        CSV bytes with 5-column header row.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator='\r\n')
    writer.writerow(['parameter', 'value', 'label', 'valid_range', 'notes'])
    for row in _config_to_rows(config):
        writer.writerow(row)
    return buf.getvalue().encode('utf-8')


def save_config_csv(config: SimulationConfig, path: str) -> None:
    """
    Write SimulationConfig to a CSV file on disk.

    Produces 5-column output: parameter, value, label, valid_range, notes.

    Args:
        config: SimulationConfig to serialise.
        path: Absolute or relative path for the output CSV.
    """
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['parameter', 'value', 'label', 'valid_range', 'notes'])
        for row in _config_to_rows(config):
            writer.writerow(row)


# -- Deserialisation ----------------------------------------------------------

def _infer_type(field_obj: dataclasses.Field):
    """
    Infer Python type from a dataclass field's default value.

    Returns one of: bool, int, float, str.
    Falls back to str for fields with MISSING defaults (factory fields).
    bool is checked before int because bool is a subclass of int.
    """
    default = field_obj.default
    if default is dataclasses.MISSING:
        return str  # Dict fields (factory); handled separately
    if isinstance(default, bool):
        return bool
    if isinstance(default, int):
        return int
    if isinstance(default, float):
        return float
    return str


def load_config_csv(file_obj) -> SimulationConfig:
    """
    Read a SimulationConfig from a CSV file.

    Accepts a file path string or a file-like object (e.g. st.file_uploader).
    Unknown keys in the CSV are silently ignored (forward-compatible).
    Missing keys in the CSV retain SimulationConfig defaults.

    Args:
        file_obj: Path string or file-like object opened in text mode.

    Returns:
        SimulationConfig populated from the CSV.

    Raises:
        ValueError: If the CSV is missing the required 'parameter'/'value' columns.
        Exception: Re-raises CSV parsing errors.
    """
    # Support both file paths and file-like objects
    if isinstance(file_obj, (str,)):
        opener = open(file_obj, newline='', encoding='utf-8')
        close_after = True
    else:
        # st.file_uploader returns bytes-mode; wrap in TextIOWrapper
        try:
            text = file_obj.read()
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            opener = io.StringIO(text)
        except Exception:
            opener = file_obj
        close_after = False

    try:
        reader = csv.DictReader(opener)
        if 'parameter' not in (reader.fieldnames or []):
            raise ValueError(
                "CSV missing 'parameter' column. "
                "Expected header: parameter,value"
            )
        data = {row['parameter']: row['value'] for row in reader}
    finally:
        if close_after:
            opener.close()

    # Start from defaults; only overwrite what is in the CSV.
    config = SimulationConfig()

    # Build a field lookup: name -> Field object
    field_map = {f.name: f for f in dataclasses.fields(config)}

    # -- Process dict fields (asset allocations) ------------------------------
    for dict_field in _DICT_FIELDS:
        current_dict = getattr(config, dict_field).copy()
        changed = False
        for asset in ASSET_CLASSES:
            key = f'{dict_field}.{asset}'
            if key in data and data[key].strip() != '':
                try:
                    current_dict[asset] = float(data[key].strip())
                    changed = True
                except ValueError:
                    pass
        if changed:
            setattr(config, dict_field, current_dict)

    # -- Process scalar fields ------------------------------------------------
    for name, val_str in data.items():
        # Skip dict-field dot-notation keys (already handled above)
        if '.' in name:
            continue

        if name not in field_map:
            continue  # Unknown field -- forward-compatible ignore

        val_str = val_str.strip()

        # Optional[int] fields
        if name in _OPTIONAL_INT_FIELDS:
            if val_str == '':
                setattr(config, name, None)
            else:
                try:
                    setattr(config, name, int(float(val_str)))
                except ValueError:
                    pass
            continue

        if val_str == '':
            continue  # Preserve default for empty values

        python_type = _infer_type(field_map[name])

        try:
            if python_type is bool:
                setattr(config, name, val_str == 'True')
            elif python_type is int:
                setattr(config, name, int(float(val_str)))
            elif python_type is float:
                setattr(config, name, float(val_str))
            else:
                setattr(config, name, val_str)
        except (ValueError, TypeError):
            pass  # Malformed value -- retain default

    # total_portfolio_value is a read-only computed field: always recompute
    # from the constituent balances regardless of what the CSV contained.
    config.total_portfolio_value = (
        config.traditional_balance + config.roth_balance +
        config.taxable_balance
    )

    # Normalize enum fields that may contain display names instead of
    # internal names (e.g., 'Fixed Real' -> 'fixed_real').
    _normalize_enums(config)

    # Auto-detect allocation format: if values sum > 2.0, assume percentage
    # (e.g. 40 instead of 0.40) and convert to decimal.
    _normalise_allocation(config)

    # Validate all constraints; raises ValueError on any violation.
    _validate_config(config)

    return config


# -- Enum normalisation -------------------------------------------------------

# Withdrawal method: display name -> internal name.
_WITHDRAWAL_DISPLAY_TO_INTERNAL = {
    'Fixed Real':               'fixed_real',
    'Fixed Nominal':            'fixed_nominal',
    'Percentage of Portfolio':  'percentage',
    'Guardrails':               'guardrails',
}

_VALID_WITHDRAWAL_METHODS    = {'fixed_real', 'fixed_nominal', 'percentage', 'guardrails'}
_VALID_SIMULATION_METHODS    = {'bootstrap', 'gbm'}
_VALID_INFLATION_METHODS     = {'bootstrap', 'fixed', 'mean_reverting'}
_VALID_FILING_STATUSES       = {'single', 'married_filing_jointly'}


def _normalise_allocation(config: SimulationConfig) -> None:
    """
    Auto-detect allocation format and normalise to decimal (0.0-1.0).

    If sum of allocation weights > 2.0, values are assumed to be in
    percentage format (e.g. 40 = 40%) and are divided by 100.
    Handles both pre_retirement_allocation and retirement_allocation.

    This allows CSV users to enter either 0.40 or 40 for 40%.
    """
    for attr in ('pre_retirement_allocation', 'retirement_allocation'):
        alloc = getattr(config, attr, {})
        total = sum(alloc.values())
        if total > 2.0:
            # Percentage format detected -- convert to decimal
            normalised = {k: v / 100.0 for k, v in alloc.items()}
            setattr(config, attr, normalised)


def _normalize_enums(config: SimulationConfig) -> None:
    """
    Map display-friendly names to internal identifiers.

    Covers withdrawal_method (which the UI stores as display text in
    selectboxes) and leaves already-valid internal names unchanged.
    """
    wm = config.withdrawal_method
    if wm in _WITHDRAWAL_DISPLAY_TO_INTERNAL:
        config.withdrawal_method = _WITHDRAWAL_DISPLAY_TO_INTERNAL[wm]


# -- Validation ---------------------------------------------------------------

def _validate_config(config: SimulationConfig) -> None:
    """
    Validate all SimulationConfig constraints after CSV deserialization.

    Raises ValueError with a descriptive message listing every violation
    found (not just the first).

    Categories checked:
        1. Enum fields (withdrawal_method, simulation_method, etc.)
        2. Non-negative balances and income amounts
        3. Age range bounds (matching Streamlit UI min/max)
        4. Age ordering (start <= end for all income windows)
        5. Allocation sums ~ 1.0 (within 1% tolerance)
        6. Rate bounds (state tax, withdrawal rate, guardrails)
        7. Guardrails floor <= ceiling
        8. SS claiming ages (62-70)
    """
    errors: list[str] = []

    def _check(condition: bool, msg: str) -> None:
        if not condition:
            errors.append(msg)

    # -- 1. Enum fields -------------------------------------------------------
    _check(config.withdrawal_method in _VALID_WITHDRAWAL_METHODS,
           f"withdrawal_method '{config.withdrawal_method}' not in "
           f"{sorted(_VALID_WITHDRAWAL_METHODS)}")

    _check(config.simulation_method in _VALID_SIMULATION_METHODS,
           f"simulation_method '{config.simulation_method}' not in "
           f"{sorted(_VALID_SIMULATION_METHODS)}")

    _check(config.inflation_method in _VALID_INFLATION_METHODS,
           f"inflation_method '{config.inflation_method}' not in "
           f"{sorted(_VALID_INFLATION_METHODS)}")

    _check(config.filing_status in _VALID_FILING_STATUSES,
           f"filing_status '{config.filing_status}' not in "
           f"{sorted(_VALID_FILING_STATUSES)}")

    # -- 2. Non-negative balances and incomes ---------------------------------
    _non_neg_fields = [
        'traditional_balance', 'roth_balance', 'taxable_balance',
        'annual_contribution',
        'annual_withdrawal_real',
        'pension_annual_real', 'rental_income_annual_real',
        'part_time_income_annual',
        'annual_healthcare_cost_real',
        'ss_monthly_benefit_at_fra', 'spouse_ss_monthly_benefit_at_fra',
    ]
    for fname in _non_neg_fields:
        val = getattr(config, fname, 0.0)
        _check(val >= 0, f"{fname} must be >= 0 (got {val})")

    # SERP per-year fields
    for yr in range(2026, 2034):
        key = f'serp_{yr}'
        val = getattr(config, key, 0.0)
        _check(val >= 0, f"{key} must be >= 0 (got {val})")

    # -- 3. Age range bounds --------------------------------------------------
    _check(25 <= config.current_age <= 75,
           f"current_age must be 25-75 (got {config.current_age})")
    _check(45 <= config.retirement_age <= 80,
           f"retirement_age must be 45-80 (got {config.retirement_age})")
    _check(75 <= config.life_expectancy <= 100,
           f"life_expectancy must be 75-100 (got {config.life_expectancy})")

    if config.spouse_age is not None:
        _check(25 <= config.spouse_age <= 75,
               f"spouse_age must be 25-75 (got {config.spouse_age})")

    _check(40 <= config.withdrawal_start_age <= 85,
           f"withdrawal_start_age must be 40-85 (got {config.withdrawal_start_age})")
    _check(62 <= config.ss_claiming_age <= 70,
           f"ss_claiming_age must be 62-70 (got {config.ss_claiming_age})")
    _check(62 <= config.spouse_ss_claiming_age <= 70,
           f"spouse_ss_claiming_age must be 62-70 (got {config.spouse_ss_claiming_age})")

    # -- 4. Age ordering ------------------------------------------------------
    _check(config.retirement_age >= config.current_age,
           f"retirement_age ({config.retirement_age}) must be >= "
           f"current_age ({config.current_age})")
    _check(config.life_expectancy > config.current_age,
           f"life_expectancy ({config.life_expectancy}) must be > "
           f"current_age ({config.current_age})")

    _check(config.pension_start_age <= config.pension_end_age,
           f"pension_start_age ({config.pension_start_age}) must be <= "
           f"pension_end_age ({config.pension_end_age})")
    _check(config.rental_start_age <= config.rental_end_age,
           f"rental_start_age ({config.rental_start_age}) must be <= "
           f"rental_end_age ({config.rental_end_age})")
    _check(config.part_time_income_start_age <= config.part_time_income_end_age,
           f"part_time_income_start_age ({config.part_time_income_start_age}) must be <= "
           f"part_time_income_end_age ({config.part_time_income_end_age})")

    # -- 5. Allocation sums ~ 1.0 (1% tolerance) -----------------------------
    pre_sum = sum(config.pre_retirement_allocation.values())
    _check(abs(pre_sum - 1.0) <= 0.01,
           f"pre_retirement_allocation sums to {pre_sum:.4f}, must be 1.0 +/- 1%")
    ret_sum = sum(config.retirement_allocation.values())
    _check(abs(ret_sum - 1.0) <= 0.01,
           f"retirement_allocation sums to {ret_sum:.4f}, must be 1.0 +/- 1%")

    # -- 6. Rate bounds -------------------------------------------------------
    _check(0.0 <= config.state_tax_rate <= 0.15,
           f"state_tax_rate must be 0.0-0.15 (got {config.state_tax_rate})")
    _check(0.0 <= config.expense_ratio <= 0.05,
           f"expense_ratio must be 0.0-0.05 (got {config.expense_ratio})")
    _check(0.0 <= config.advisory_fee <= 0.03,
           f"advisory_fee must be 0.0-0.03 (got {config.advisory_fee})")
    _check(0.005 <= config.withdrawal_rate <= 0.10,
           f"withdrawal_rate must be 0.005-0.10 (got {config.withdrawal_rate})")

    # -- 7. Guardrails floor <= ceiling ---------------------------------------
    _check(config.withdrawal_floor <= config.withdrawal_ceiling,
           f"withdrawal_floor ({config.withdrawal_floor}) must be <= "
           f"withdrawal_ceiling ({config.withdrawal_ceiling})")

    # -- Raise all collected errors -------------------------------------------
    if errors:
        raise ValueError(
            f"CSV config has {len(errors)} validation error(s):\n"
            + "\n".join(f"  * {e}" for e in errors)
        )


def csv_template_bytes() -> bytes:
    """
    Return CSV bytes for a blank default SimulationConfig template.

    Useful for letting users download a template they can fill in.

    Returns:
        CSV bytes of all fields at their default values.
    """
    return config_to_csv_bytes(SimulationConfig())
