"""
Retirement Monte Carlo Streamlit Application.

Expert-level financial modeling interface for retirement planning. Features
dark theme (black background, white text), sidebar configuration with expanders,
and tabbed results display with comprehensive analytics including portfolio
projections, tax analysis, and risk assessment.

Integrates MonteCarloSimulator with Plotly charts and Excel export.

Version: 1.0.0
"""

import sys
import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime
import os
from pathlib import Path


def _get_layout_mode() -> str:
    """Parse --layout flag from CLI args.  Default: 'sidebar'.

    Usage:  streamlit run app.py -- --layout=tabs
    """
    for arg in sys.argv:
        if arg.startswith("--layout="):
            return arg.split("=", 1)[1]
    return "sidebar"


LAYOUT_MODE = _get_layout_mode()

from config.defaults import SimulationConfig, SimulationResults, ASSET_CLASSES
from models.simulation import MonteCarloSimulator
from utils.charts import (
    portfolio_fan_chart,
    success_probability_chart,
    withdrawal_sustainability_chart,
    income_stack_chart,
    tax_burden_chart,
    asset_allocation_chart,
    inflation_chart,
    portfolio_depletion_histogram,
    withdrawal_schedule_chart,
    portfolio_success_curve_chart,
    portfolio_balance_lines_chart,
)
from utils.helpers import (
    format_currency,
    format_percent,
    validate_allocation,
    normalize_allocation,
    years_to_retirement,
    fmt_k,
    fmt_m,
)
from utils.report_generator import ReportGenerator
from utils.config_io import load_config_csv, save_config_csv, config_to_csv_bytes, csv_template_bytes
from data.loader import DataLoader


# Configure Streamlit page
st.set_page_config(
    page_title="Retirement Monte Carlo",
    layout="wide",
    initial_sidebar_state="collapsed" if LAYOUT_MODE == "tabs" else "expanded",
    menu_items={
        "About": "Expert-level retirement financial modeling using Monte Carlo simulation.",
    }
)

# Inject custom CSS for full black background
st.markdown(
    """
    <style>
        body {
            background-color: #000000;
            color: #ffffff;
        }
        .main {
            background-color: #000000;
            color: #ffffff;
        }
        [data-testid="stSidebar"] {
            background-color: #111111;
            color: #ffffff;
        }
        [data-testid="stHeader"] {
            background-color: #000000;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff;
        }
        .stTabs [data-baseweb="tab-list"] button {
            color: #ffffff;
        }
        .stTabs [aria-selected="true"] {
            color: #ffffff;
        }
        .metric-card {
            background-color: #111111;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #0099ff;
        }

        /* ── Editable input fields: pale blue bg + dark blue text + larger font ── */
        /* Target ALL text/number input elements globally */
        input[type="text"],
        input[type="number"],
        input[type="email"],
        input[type="password"],
        input[type="tel"],
        input[type="url"],
        textarea {
            background-color: #c8dff5 !important;
            color: #0a2540 !important;
            -webkit-text-fill-color: #0a2540 !important;
            font-size: 17px !important;
            font-weight: 500 !important;
            border: 1.5px solid #4A9EFF !important;
            border-radius: 4px !important;
        }
        input:focus, textarea:focus {
            border-color: #6ab4ff !important;
            box-shadow: 0 0 0 3px rgba(74, 158, 255, 0.25) !important;
            outline: none !important;
        }
        /* Input wrapper containers (for border display) */
        [data-baseweb="input"] {
            background-color: #c8dff5 !important;
            border-color: #4A9EFF !important;
        }
        /* +/- stepper buttons on number inputs */
        [data-testid="stNumberInput"] button,
        .stNumberInput button {
            color: #0a2540 !important;
            background-color: #a8c8e8 !important;
            border: 1px solid #4A9EFF !important;
        }

        /* Selectbox / dropdown */
        [data-baseweb="select"] > div {
            background-color: #c8dff5 !important;
            border: 1.5px solid #4A9EFF !important;
            border-radius: 4px !important;
        }
        [data-baseweb="select"] span,
        [data-baseweb="select"] input {
            color: #0a2540 !important;
            -webkit-text-fill-color: #0a2540 !important;
            font-size: 17px !important;
            font-weight: 500 !important;
        }

        /* Slider track */
        .stSlider [data-baseweb="slider"] div[role="slider"] {
            background-color: #4A9EFF !important;
        }

        /* Section dividers — thick bright blue line */
        hr {
            border: none !important;
            height: 4px !important;
            background: #4A9EFF !important;
            margin: 1.2em 0 !important;
            opacity: 1.0 !important;
            border-radius: 2px !important;
        }

        /* Captions — blue-tinted for readability */
        .stCaption, [data-testid="stCaptionContainer"] {
            color: #7ab8ff !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)




# fmt_k and fmt_m are imported from utils.helpers.
# Both are now aliases for fmt_dollar — unified magnitude-aware formatting:
#   >= $1M → "$X.XXM",  >= $1K → "$X.XK",  < $1K → "$X".


def fmt_monthly(v: float) -> str:
    """Format a monthly dollar value showing monthly and annualized amount."""
    return f"${v:,.0f}/month  ·  ${v * 12:,.0f}/year"


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "simulation_results" not in st.session_state:
        st.session_state.simulation_results = None
    if "config_cache" not in st.session_state:
        st.session_state.config_cache = None
    if "sidebar_config" not in st.session_state:
        # Seed with defaults on very first load.
        # Subsequent reruns (including page refresh) seed from this stored config
        # so sidebar inputs survive reruns, tab switches, and browser refreshes.
        st.session_state.sidebar_config = SimulationConfig()
    if "input_csv_name" not in st.session_state:
        st.session_state.input_csv_name = None   # Filename of last loaded CSV
    if "output_csv_path" not in st.session_state:
        st.session_state.output_csv_path = None  # Path of last saved CSV


def _csv_sidebar_panel(container=None) -> None:
    """
    Render the CSV import/export panel at the very top of the sidebar.

    Load from CSV: updates st.session_state.sidebar_config so the next
    call to build_sidebar_config() seeds all widgets from the loaded values.
    Save Config: writes current config to outputs/ and offers a download button.
    Template: allows downloading a blank default config as a starting point.

    Must be called BEFORE build_sidebar_config() so the loaded config is
    in session_state when widgets are rendered.

    Args:
        container: Optional Streamlit container (e.g. a tab) to render into.
                   When None, uses st.sidebar.expander (default sidebar mode).
    """
    with (container if container is not None else st.sidebar.expander("📂 Config File (CSV)", expanded=True)):

        # ── Load ──────────────────────────────────────────────────────────
        uploaded = st.file_uploader(
            "Load config from CSV",
            type=['csv'],
            key='csv_uploader',
            help="Upload a previously saved config CSV to restore all inputs.",
        )
        if uploaded is not None:
            # Use file content hash to detect new/changed uploads
            _file_bytes = uploaded.getvalue()
            _current_hash = hash(_file_bytes)
            _last_hash = st.session_state.get('_last_csv_hash', None)

            if _current_hash != _last_hash:
                try:
                    uploaded.seek(0)  # Reset after reading bytes
                    loaded = load_config_csv(uploaded)

                    # Nuclear option: clear ALL session state except the
                    # keys we're about to set. This guarantees no stale
                    # widget values survive from the prior config.
                    # Increment load counter so all widget labels change,
                    # forcing Streamlit to re-read value= parameters.
                    st.session_state['_csv_load_count'] = (
                        st.session_state.get('_csv_load_count', 0) + 1
                    )

                    _preserve = {'_last_csv_hash', 'sidebar_config',
                                 'input_csv_name', 'output_csv_path',
                                 'simulation_results', 'config_cache',
                                 '_csv_load_count'}
                    _to_delete = [k for k in st.session_state if k not in _preserve]
                    for k in _to_delete:
                        del st.session_state[k]

                    st.session_state.sidebar_config = loaded
                    st.session_state.input_csv_name = uploaded.name
                    st.session_state['_last_csv_hash'] = _current_hash

                    st.success(f"Loaded: {uploaded.name}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Load failed: {exc}")

        # ── Save / Download ───────────────────────────────────────────────
        col1, col2 = st.columns(2)

        current_cfg = st.session_state.sidebar_config
        csv_bytes = config_to_csv_bytes(current_cfg)

        # Auto-versioned filename: YYYY Month DD Retirement Input v#.csv
        _now = datetime.now()
        _date_prefix = _now.strftime('%Y %B %d')
        _base_name = f"{_date_prefix} Retirement Input"
        inp_dir = Path(__file__).parent / 'input'
        inp_dir.mkdir(exist_ok=True)
        # Find next version number by parsing actual version numbers from
        # filenames.  Previous logic used len(existing)+1 which overwrote
        # the latest file when version numbers had gaps (e.g. v1 deleted,
        # v2+v3 exist → len=2 → next=3 → overwrites v3).
        import re as _re
        _existing = sorted(inp_dir.glob(f"{_base_name} v*.csv"))
        _versions = []
        for _f in _existing:
            _m = _re.search(r'v(\d+)\.csv$', _f.name)
            if _m:
                _versions.append(int(_m.group(1)))
        _next_ver = max(_versions) + 1 if _versions else 1
        dl_filename = f"{_base_name} v{_next_ver}.csv"

        with col1:
            if st.download_button(
                label="⬇ Download",
                data=csv_bytes,
                file_name=dl_filename,
                mime='text/csv',
                help="Download current sidebar config as CSV.",
                use_container_width=True,
            ):
                pass  # Download happens client-side; no server action needed

        with col2:
            if st.button(
                "💾 Save to folder",
                help="Save config CSV to the input/ folder in the workspace.",
                use_container_width=True,
            ):
                out_path = inp_dir / dl_filename
                save_config_csv(current_cfg, str(out_path))
                st.session_state.output_csv_path = str(out_path)
                st.success(f"Saved: input/{dl_filename}")

        # ── Template ──────────────────────────────────────────────────────
        st.download_button(
            label="⬇ Blank template",
            data=csv_template_bytes(),
            file_name="retirement_config_template.csv",
            mime='text/csv',
            help="Download a blank default config template to fill in.",
            use_container_width=True,
        )

        # ── Download Historical Data ─────────────────────────────────────
        # Fetches real historical returns from yfinance (8 ETFs), FRED
        # (CPI, 10Y Treasury, 3-month T-Bill) and caches locally.
        # Required for meaningful bootstrap analyses.
        st.markdown("---")
        if st.button(
            "📊 Download Historical Data",
            help=(
                "Download historical asset returns from yfinance and FRED. "
                "Required for bootstrap method. "
                "Data is cached locally under data/cache/."
            ),
            use_container_width=True,
        ):
            with st.spinner("Downloading from yfinance and FRED..."):
                loader = DataLoader()
                results = loader.download_all(force_refresh=True)
            # Display per-source results
            all_ok = True
            for src, info in results.items():
                ok = info.get("success", False)
                msg = info.get("message", "")
                if ok:
                    st.success(f"{src}: {msg}")
                else:
                    st.error(f"{src}: {msg}")
                    all_ok = False
            if all_ok:
                st.success("All data sources downloaded and cached.")
            else:
                st.warning(
                    "Some sources failed. Simulation will use cached or "
                    "fallback data for missing sources."
                )


# Withdrawal method display↔internal name mapping.
# The sidebar selectbox shows display names; config stores internal values.
# portfolio.py, simulation.py, report_generator.py all expect internal values.
_WITHDRAWAL_METHOD_DISPLAY = ['Fixed Real', 'Fixed Nominal', 'Percentage of Portfolio', 'Guardrails']
_WITHDRAWAL_METHOD_INTERNAL = ['fixed_real', 'fixed_nominal', 'percentage', 'guardrails']


def build_sidebar_config(containers=None) -> SimulationConfig:
    """
    Build SimulationConfig from sidebar inputs organized in expanders.

    Seeds from st.session_state.sidebar_config so all entered values are
    retained across reruns, tab switches, and browser page refreshes.
    At the end of the function the resulting config is saved back to
    session_state so the next rerun starts from the last known good state.

    Args:
        containers: Optional dict mapping section names to Streamlit containers
                    (e.g. tab objects).  When None, uses st.sidebar.expander
                    (default sidebar mode).

    Returns:
        Complete SimulationConfig object based on user inputs.
    """
    def _section(name, expanded=False):
        """Return the container for a config section."""
        if containers is not None:
            return containers[name]
        return st.sidebar.expander(name, expanded=expanded)

    # Seed from previously saved config; never reset to defaults mid-session.
    config = st.session_state.sidebar_config

    # Section 1: Personal
    with _section("Personal", expanded=True):
        config.current_age = st.slider(
            "Current Age",
            min_value=25, max_value=75, value=config.current_age,
            help="Your current age in years."
        )

        config.retirement_age = st.slider(
            "Retirement Age",
            min_value=45, max_value=75, value=config.retirement_age,
            help="Target age to retire and begin withdrawals."
        )

        config.life_expectancy = st.slider(
            "Life Expectancy (Planning Horizon)",
            min_value=75, max_value=100, value=config.life_expectancy,
            help="Age you want to plan through (planning horizon)."
        )

        # Age ordering validation
        if config.current_age >= config.retirement_age:
            st.warning("⚠️ Current age must be less than retirement age (already retired scenarios use retirement_age ≤ current_age).")
        if config.retirement_age >= config.life_expectancy:
            st.warning("⚠️ Retirement age should be less than life expectancy for a meaningful simulation horizon.")

        has_spouse = st.checkbox(
            "Has Spouse?",
            value=(config.spouse_age is not None),
            help="Include spouse in retirement plan."
        )

        if has_spouse:
            col1, col2 = st.columns(2)
            with col1:
                config.spouse_age = st.number_input(
                    "Spouse Current Age",
                    min_value=25, max_value=75,
                    value=config.spouse_age if config.spouse_age else config.current_age,
                )
            with col2:
                config.spouse_life_expectancy = st.number_input(
                    "Spouse Life Expectancy",
                    min_value=75, max_value=100,
                    value=config.spouse_life_expectancy,
                )
        else:
            config.spouse_age = None

    # Section 2: Portfolio
    with _section("Portfolio"):
        st.write("**Primary Accounts**")
        col1, col2, col3 = st.columns(3)
        with col1:
            config.traditional_balance = st.number_input(
                "Traditional (401k/IRA)",
                min_value=0.0, value=config.traditional_balance, step=50000.0,
                format="%.0f",
            )
        with col2:
            config.roth_balance = st.number_input(
                "Roth",
                min_value=0.0, value=config.roth_balance, step=50000.0,
                format="%.0f",
            )
        with col3:
            config.taxable_balance = st.number_input(
                "Taxable",
                min_value=0.0, value=config.taxable_balance, step=50000.0,
                format="%.0f",
            )

        st.caption(
            f"Traditional: {fmt_k(config.traditional_balance)}  |  "
            f"Roth: {fmt_k(config.roth_balance)}  |  "
            f"Taxable: {fmt_k(config.taxable_balance)}"
        )
        config.annual_contribution = st.number_input(
            "Annual Contribution (Pre-Retirement)",
            min_value=0.0, value=config.annual_contribution, step=5000.0,
            format="%.0f",
            help="Contribution while employed, in today's dollars.",
        )
        st.caption(fmt_k(config.annual_contribution))

        # ── Computed Household Total (read-only) ──────────────────────────
        # total_portfolio_value = sum of all 3 constituent account balances.
        # Not editable — always derived from parts. Displayed as metric.
        config.total_portfolio_value = (
            config.traditional_balance + config.roth_balance + config.taxable_balance
        )
        st.divider()
        st.metric(
            "Household Portfolio Total",
            fmt_k(config.total_portfolio_value),
            help="Sum of all primary + spouse account balances. Calculated, not editable.",
        )

    # Section 3: Withdrawal Strategy
    with _section("Withdrawal"):
        _wm_idx = (
            _WITHDRAWAL_METHOD_INTERNAL.index(config.withdrawal_method)
            if config.withdrawal_method in _WITHDRAWAL_METHOD_INTERNAL else 0
        )
        _wm_display = st.selectbox(
            "Withdrawal Method",
            options=_WITHDRAWAL_METHOD_DISPLAY,
            index=_wm_idx,
            help="How to determine annual withdrawals.",
        )
        config.withdrawal_method = _WITHDRAWAL_METHOD_INTERNAL[
            _WITHDRAWAL_METHOD_DISPLAY.index(_wm_display)
        ]

        if config.withdrawal_method == 'fixed_real':
            st.write("**Primary**")
            col1, col2 = st.columns(2)
            with col1:
                config.annual_withdrawal_real = st.number_input(
                    "Annual Draw (Today's $)",
                    min_value=0.0, value=config.annual_withdrawal_real, step=10000.0,
                    format="%.0f",
                    help="Primary earner's net portfolio draw in today's dollars. "
                         "Inflation-adjusted at runtime. Set this as the amount needed "
                         "after all other income sources (SS, pension, rental, etc.).",
                )
                st.caption(fmt_k(config.annual_withdrawal_real))
            with col2:
                config.withdrawal_start_age = st.number_input(
                    "Start Age",
                    min_value=40, max_value=85,
                    value=config.withdrawal_start_age,
                    step=1,
                    format="%d",
                    help="Age primary begins drawing from portfolio. "
                         "Independent of retirement age — can start earlier or later.",
                )
                st.caption(f"Primary draws from age {config.withdrawal_start_age}")

        elif config.withdrawal_method == 'percentage':
            config.withdrawal_rate = st.slider(
                "Withdrawal Rate",
                min_value=0.01, max_value=0.10, value=config.withdrawal_rate, step=0.005,
                format="%.2f%%",
                help="Percentage of portfolio to withdraw annually. Household-level.",
            )
        elif config.withdrawal_method == 'guardrails':
            col1, col2 = st.columns(2)
            with col1:
                config.withdrawal_floor = st.slider(
                    "Guardrail Floor (%)",
                    min_value=0.01, max_value=0.05, value=config.withdrawal_floor, step=0.005,
                    format="%.2f%%",
                    help="Minimum withdrawal as % of portfolio.",
                )
            with col2:
                config.withdrawal_ceiling = st.slider(
                    "Guardrail Ceiling (%)",
                    min_value=0.05, max_value=0.15, value=config.withdrawal_ceiling, step=0.005,
                    format="%.2f%%",
                    help="Maximum withdrawal as % of portfolio.",
                )

    # Section 4: Asset Allocation
    with _section("Allocation"):
        _lc = st.session_state.get('_csv_load_count', 0)
        st.write("**Pre-Retirement Allocation**")
        col1, col2 = st.columns(2)
        with col1:
            config.pre_retirement_allocation['us_large_cap'] = st.slider(
                f"US Large Cap #{_lc}", 0.0, 1.0,
                value=config.pre_retirement_allocation['us_large_cap'], step=0.01
            )
            config.pre_retirement_allocation['us_small_cap'] = st.slider(
                f"US Small Cap #{_lc}", 0.0, 1.0,
                value=config.pre_retirement_allocation['us_small_cap'], step=0.01
            )
            config.pre_retirement_allocation['international_dev'] = st.slider(
                f"Intl Developed #{_lc}", 0.0, 1.0,
                value=config.pre_retirement_allocation['international_dev'], step=0.01
            )
            config.pre_retirement_allocation['emerging_markets'] = st.slider(
                f"Emerging Markets #{_lc}", 0.0, 1.0,
                value=config.pre_retirement_allocation['emerging_markets'], step=0.01
            )
        with col2:
            config.pre_retirement_allocation['us_bonds'] = st.slider(
                f"US Bonds #{_lc}", 0.0, 1.0,
                value=config.pre_retirement_allocation['us_bonds'], step=0.01
            )
            config.pre_retirement_allocation['tips'] = st.slider(
                f"TIPS #{_lc}", 0.0, 1.0,
                value=config.pre_retirement_allocation['tips'], step=0.01
            )
            config.pre_retirement_allocation['cash'] = st.slider(
                f"Cash #{_lc}", 0.0, 1.0,
                value=config.pre_retirement_allocation['cash'], step=0.01
            )
            config.pre_retirement_allocation['reits'] = st.slider(
                f"REITs #{_lc}", 0.0, 1.0,
                value=config.pre_retirement_allocation['reits'], step=0.01
            )

        pre_total = sum(config.pre_retirement_allocation.values())
        if abs(pre_total - 1.0) > 0.01:
            st.warning(f"Pre-retirement allocation sums to {pre_total:.1%}, not 100% — will auto-normalize for simulation.")
            config.pre_retirement_allocation = normalize_allocation(
                config.pre_retirement_allocation
            )
            if st.button("Normalize Pre-Retirement Allocation"):
                st.rerun()

        st.write("**Retirement Allocation**")
        col1, col2 = st.columns(2)
        with col1:
            config.retirement_allocation['us_large_cap'] = st.slider(
                f"Ret. US Large Cap #{_lc}", 0.0, 1.0,
                value=config.retirement_allocation['us_large_cap'], step=0.01,
            )
            config.retirement_allocation['us_small_cap'] = st.slider(
                f"Ret. US Small Cap #{_lc}", 0.0, 1.0,
                value=config.retirement_allocation['us_small_cap'], step=0.01,
            )
            config.retirement_allocation['international_dev'] = st.slider(
                f"Ret. Intl Developed #{_lc}", 0.0, 1.0,
                value=config.retirement_allocation['international_dev'], step=0.01,
            )
            config.retirement_allocation['emerging_markets'] = st.slider(
                f"Ret. Emerging Markets #{_lc}", 0.0, 1.0,
                value=config.retirement_allocation['emerging_markets'], step=0.01,
            )
        with col2:
            config.retirement_allocation['us_bonds'] = st.slider(
                f"Ret. US Bonds #{_lc}", 0.0, 1.0,
                value=config.retirement_allocation['us_bonds'], step=0.01,
            )
            config.retirement_allocation['tips'] = st.slider(
                f"Ret. TIPS #{_lc}", 0.0, 1.0,
                value=config.retirement_allocation['tips'], step=0.01,
            )
            config.retirement_allocation['cash'] = st.slider(
                f"Ret. Cash #{_lc}", 0.0, 1.0,
                value=config.retirement_allocation['cash'], step=0.01,
            )
            config.retirement_allocation['reits'] = st.slider(
                f"Ret. REITs #{_lc}", 0.0, 1.0,
                value=config.retirement_allocation['reits'], step=0.01,
            )

        ret_total = sum(config.retirement_allocation.values())
        if abs(ret_total - 1.0) > 0.01:
            st.warning(f"Retirement allocation sums to {ret_total:.1%}, not 100% — will auto-normalize for simulation.")
            config.retirement_allocation = normalize_allocation(
                config.retirement_allocation
            )
            if st.button("Normalize Retirement Allocation"):
                st.rerun()

        config.use_glide_path = st.checkbox(
            "Use Glide Path (gradual transition)",
            value=config.use_glide_path,
            help="Gradually shift allocation from pre-retirement to retirement over time.",
        )

    # Section 5: Tax
    with _section("Tax"):
        _fs_opts = ['single', 'married_filing_jointly']
        config.filing_status = st.selectbox(
            "Filing Status",
            options=_fs_opts,
            index=_fs_opts.index(config.filing_status) if config.filing_status in _fs_opts else 0,
        )

        config.include_state_tax = st.checkbox(
            "Include State Tax",
            value=config.include_state_tax,
        )

        if config.include_state_tax:
            config.state_tax_rate = st.number_input(
                "State Effective Tax Rate",
                min_value=0.0, max_value=0.15, value=config.state_tax_rate, step=0.001,
                format="%.4f",
                help="Enter as decimal: 0.093 = 9.3%. Average state income tax rate.",
            )
            st.caption(f"{config.state_tax_rate * 100:.1f}%")

    # Section 6: Social Security
    with _section("Social Security"):
        config.ss_monthly_benefit_at_fra = st.number_input(
            "Monthly Benefit at Full Retirement Age",
            min_value=0.0, value=config.ss_monthly_benefit_at_fra, step=100.0,
            format="%.0f",
            help="Your primary earner benefit (monthly).",
        )
        st.caption(fmt_monthly(config.ss_monthly_benefit_at_fra))

        _fra_opts = [66, 67]
        config.ss_fra = st.selectbox(
            "Full Retirement Age",
            options=_fra_opts,
            index=_fra_opts.index(config.ss_fra) if config.ss_fra in _fra_opts else 1,
            help="Your full retirement age for Social Security.",
        )

        config.ss_claiming_age = st.slider(
            "Claiming Age",
            min_value=62, max_value=70, value=config.ss_claiming_age,
            help="Age at which you claim Social Security.",
        )

        include_spouse_ss = st.checkbox(
            "Include Spouse Social Security",
            value=config.include_spousal_benefit,
        )

        if include_spouse_ss:
            col1, col2 = st.columns(2)
            with col1:
                config.spouse_ss_monthly_benefit_at_fra = st.number_input(
                    "Spouse Monthly Benefit at FRA",
                    min_value=0.0, value=config.spouse_ss_monthly_benefit_at_fra, step=100.0,
                    format="%.0f",
                )
                st.caption(fmt_monthly(config.spouse_ss_monthly_benefit_at_fra))
            with col2:
                config.spouse_ss_claiming_age = st.slider(
                    "Spouse Claiming Age",
                    min_value=62, max_value=70, value=config.spouse_ss_claiming_age,
                )
            config.include_spousal_benefit = True
        else:
            config.include_spousal_benefit = False

    # Section 7: Other Income
    with _section("Income"):
        # ── Pension ──────────────────────────────────────────────────────────
        st.write("**Pension**")
        config.pension_annual_real = st.number_input(
            "Annual Pension (Real $)",
            min_value=0.0, value=config.pension_annual_real, step=5000.0,
            format="%.0f",
            help="Annual pension in today's dollars. Ordinary income — taxable.",
        )
        st.caption(fmt_k(config.pension_annual_real))
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            config.pension_start_age = st.number_input(
                "Pension Start Age", min_value=45, max_value=85,
                value=config.pension_start_age,
            )
        with col2:
            config.pension_end_age = st.number_input(
                "Pension End Age", min_value=55, max_value=100,
                value=config.pension_end_age,
                help="Set to 95+ for a lifetime pension.",
            )
        with col3:
            config.pension_is_real = st.checkbox(
                "Pension COLA (inflation-adj.)",
                value=config.pension_is_real,
                help="Check if pension has COLA. Unchecked = fixed nominal.",
            )

        st.divider()

        # ── Rental Income ─────────────────────────────────────────────────────
        st.write("**Rental Income** (net Schedule E — after expenses)")
        config.rental_income_annual_real = st.number_input(
            "Annual Net Rental Income (Real $)",
            min_value=0.0, value=config.rental_income_annual_real, step=5000.0,
            format="%.0f",
            help="Net income after all property expenses. Passive ordinary income on Schedule E.",
        )
        st.caption(fmt_k(config.rental_income_annual_real))
        col1, col2 = st.columns(2)
        with col1:
            config.rental_start_age = st.number_input(
                "Rental Start Age", min_value=45, max_value=85,
                value=config.rental_start_age,
            )
        with col2:
            config.rental_end_age = st.number_input(
                "Rental End Age", min_value=55, max_value=100,
                value=config.rental_end_age,
                help="Age property is sold or income stops.",
            )

        st.divider()

        # ── Primary Part-Time Work ────────────────────────────────────────────
        st.write("**Primary Part-Time Work**")
        config.part_time_income_annual = st.number_input(
            "Annual Part-Time Income (Real $)",
            min_value=0.0, value=config.part_time_income_annual, step=5000.0,
            format="%.0f",
            help="Active earned income. FICA (Social Security + Medicare) applies.",
        )
        st.caption(fmt_k(config.part_time_income_annual))
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            config.part_time_income_start_age = st.number_input(
                "Part-Time Start Age", min_value=45, max_value=85,
                value=config.part_time_income_start_age,
            )
        with col2:
            config.part_time_income_end_age = st.number_input(
                "Part-Time End Age", min_value=50, max_value=90,
                value=config.part_time_income_end_age,
            )
        with col3:
            config.part_time_is_real = st.checkbox(
                "Part-time inflation-adjusted",
                value=config.part_time_is_real,
                help="Check if part-time income grows with inflation.",
            )

        st.divider()

        # ── SERP / Nonqualified Deferred Compensation ─────────────────────────
        st.write("**Primary SERP / Nonqualified Deferred Compensation (IRC §409A)**")
        st.caption(
            "Contractual nominal distribution per calendar year. "
            "Ordinary income for federal + state tax. "
            "FICA not applicable — already paid at deferral."
        )
        # Per-year distribution grid (2026–2033).  4 years per row.
        _serp_years = list(range(2026, 2034))
        _serp_row1 = st.columns(4)
        _serp_row2 = st.columns(4)
        _lc = st.session_state.get('_csv_load_count', 0)
        for i, yr in enumerate(_serp_years):
            _col = _serp_row1[i] if i < 4 else _serp_row2[i - 4]
            _age = config.current_age + (yr - config.simulation_start_year)
            with _col:
                val = st.number_input(
                    f"SERP {yr} #{_lc}",
                    min_value=0.0,
                    value=float(getattr(config, f'serp_{yr}')),
                    step=5000.0,
                    format="%.0f",
                )
                st.caption(f"age {_age}")
                setattr(config, f'serp_{yr}', val)
        serp_total = sum(getattr(config, f'serp_{yr}') for yr in _serp_years)
        if serp_total > 0:
            st.caption(f"Total primary SERP (2026–2033): {fmt_k(serp_total)}")

    # ── Section 7b: Spend (v1.8 Feature 1) ──────────────────────────────────
    with _section("Spend"):
        st.caption(
            "Annual discretionary/lifestyle spend. If income < spend, "
            "shortfall is drawn from portfolio. Healthcare is separate below."
        )
        config.spend_annual_real = st.number_input(
            "Annual Spend — Flat Rate (Real $)",
            min_value=0.0, value=config.spend_annual_real, step=5000.0,
            format="%.0f",
            help="Default annual spend in today's dollars. Used when phases are disabled, or as fallback for pre-anchor years with $0.",
        )
        st.caption(fmt_k(config.spend_annual_real))

        config.spend_start_age = st.number_input(
            "Spend Starting Age",
            min_value=config.current_age, max_value=config.life_expectancy,
            value=config.spend_start_age,
            help="Age at which annual spend begins.",
        )

        _surplus_opts = ['ignore', 'reinvest']
        _surplus_labels = {'ignore': 'Ignore surplus', 'reinvest': 'Reinvest surplus to taxable'}
        config.spend_surplus_mode = st.selectbox(
            "If Income > Spend",
            options=_surplus_opts,
            index=_surplus_opts.index(config.spend_surplus_mode) if config.spend_surplus_mode in _surplus_opts else 0,
            format_func=lambda x: _surplus_labels.get(x, x),
            help="What to do when total income exceeds spend.",
        )

        st.divider()
        st.write("**Healthcare Costs**")
        st.caption("Modeled separately with healthcare-specific inflation premium.")

        config.include_healthcare_costs = st.checkbox(
            "Include Healthcare Costs",
            value=config.include_healthcare_costs,
            help="Deducts annual healthcare cost from portfolio when retired.",
        )
        if config.include_healthcare_costs:
            config.annual_healthcare_cost_real = st.number_input(
                "Annual Healthcare Cost (Real $)",
                min_value=0.0, value=config.annual_healthcare_cost_real, step=1000.0,
                format="%.0f",
                help="Annual healthcare cost in today's dollars (Medicare premiums, supplemental, prescriptions).",
            )
            st.caption(fmt_k(config.annual_healthcare_cost_real))

            config.healthcare_is_real = st.checkbox(
                "Healthcare inflation-adjusted",
                value=config.healthcare_is_real,
                help="Check = grows with CPI + healthcare premium. Uncheck = fixed nominal.",
            )

            config.healthcare_inflation_premium = st.number_input(
                "Healthcare Inflation Premium",
                min_value=0.0, max_value=0.10, value=config.healthcare_inflation_premium,
                step=0.005, format="%.3f",
                help="Extra inflation above CPI for healthcare (e.g., 0.020 = 2%).",
            )

    # Section 8: Simulation Settings
    with _section("Simulation"):
        sim_count_options = [1000, 5000, 10000]
        config.n_simulations = st.selectbox(
            "Number of Simulations",
            options=sim_count_options,
            index=sim_count_options.index(config.n_simulations) if config.n_simulations in sim_count_options else 2,
            help="More simulations = better statistics but slower computation.",
        )

        _sm_opts = ['bootstrap', 'gbm']
        config.simulation_method = st.selectbox(
            "Simulation Method",
            options=_sm_opts,
            index=_sm_opts.index(config.simulation_method) if config.simulation_method in _sm_opts else 1,
            help="Bootstrap: historical returns. GBM: geometric Brownian motion.",
        )

        _im_opts = ['bootstrap', 'fixed', 'mean_reverting']
        config.inflation_method = st.selectbox(
            "Inflation Method",
            options=_im_opts,
            index=_im_opts.index(config.inflation_method) if config.inflation_method in _im_opts else 0,
            help="How to generate inflation paths.",
        )

        seed_input = st.number_input(
            "Random Seed (0 = no seed, varies each run)",
            min_value=0, value=0, step=1,
            help="Set to non-zero value for reproducible results.",
        )
        config.random_seed = seed_input if seed_input > 0 else None

    # Persist current config so the next rerun (including page refresh)
    # seeds all sidebar widgets from the last entered values, not defaults.
    st.session_state.sidebar_config = config

    return config


def _run_simulation(config):
    """Execute Monte Carlo simulation and cache results in session state."""
    st.session_state.config_cache = config
    st.session_state.simulation_results = None

    with st.spinner("Running Monte Carlo simulation... (this may take 30-60 seconds)"):
        try:
            simulator = MonteCarloSimulator(config)
            results = simulator.run()
            st.session_state.simulation_results = results
            st.success("Simulation completed successfully!")
        except Exception as e:
            st.error(f"Simulation failed with error: {str(e)}")
            st.stop()


def _file_status_bar():
    """Show currently loaded input config and last saved output."""
    file_cols = st.columns([3, 3, 2])
    with file_cols[0]:
        if st.session_state.input_csv_name:
            st.info(f"📂 Input config: **{st.session_state.input_csv_name}**")
        else:
            st.caption("📂 No config file loaded — using sidebar values.")
    with file_cols[1]:
        if st.session_state.output_csv_path:
            fname = Path(st.session_state.output_csv_path).name
            st.info(f"💾 Last saved: **{fname}**")
        else:
            st.caption("💾 Config not yet saved to folder.")
    with file_cols[2]:
        st.caption(
            "Expert-level retirement planning · Monte Carlo simulation"
        )


# Section names used for both sidebar expanders and tabs-mode layout.
_CONFIG_SECTIONS = [
    "CSV", "Personal", "Portfolio", "Withdrawal", "Allocation",
    "Tax", "Social Security", "Income", "Spend", "Simulation",
]


def main():
    """Main Streamlit application."""
    initialize_session_state()

    if LAYOUT_MODE == "tabs":
        _main_tabs_layout()
    else:
        _main_sidebar_layout()


def _main_sidebar_layout():
    """Default layout: config in sidebar, results in main area."""
    # CSV panel must run BEFORE build_sidebar_config so any loaded config
    # is in session_state when sidebar widgets are rendered.
    _csv_sidebar_panel()

    st.title("Retirement Monte Carlo Financial Planner")
    _file_status_bar()
    st.divider()

    config = build_sidebar_config()

    if st.sidebar.button("🚀 RUN SIMULATION", use_container_width=True):
        _run_simulation(config)

    # Display results if available
    if st.session_state.simulation_results is not None:
        results = st.session_state.simulation_results
        config = st.session_state.config_cache

        # Display success rate prominently
        success_rate = results.success_rate

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Success Rate",
                f"{success_rate*100:.1f}%",
                delta=None,
            )

        with col2:
            _hs = getattr(results, 'household_score', success_rate)
            st.metric(
                "Household Score",
                f"{_hs*100:.0f}/100",
                help="Portfolio survival probability through max(primary, spouse) life expectancy.",
            )

        final_balance_median = np.median(results.portfolio_values[:, -1])
        with col3:
            st.metric(
                "Median Final Balance",
                fmt_m(final_balance_median),
            )

        final_balance_10pct = np.percentile(results.portfolio_values[:, -1], 10)
        with col4:
            st.metric(
                "10th Percentile Final",
                fmt_m(final_balance_10pct),
            )

        median_tax_all_years = np.median(results.taxes)
        with col5:
            st.metric(
                "Median Annual Tax",
                fmt_m(median_tax_all_years),
            )

        # Tabbed results display
        tab1, tab1b, tab1c, tab2, tab2b, tab3, tab3b, tab4, tab5, tab6, tab7, tab7b = st.tabs(
            ["Portfolio Projection", "Portfolio Success", "Portfolio Balances",
             "Income & Withdrawals", "Withdrawal Schedule",
             "Tax Analysis", "Income Breakdown", "Asset Allocation", "Inflation",
             "Risk Analysis", "Portfolio Statistics", "Simulation Summary"]
        )
        _render_result_tabs(results, config,
                            tab1, tab1b, tab1c, tab2, tab2b,
                            tab3, tab3b, tab4, tab5, tab6, tab7, tab7b)

    else:
        # Welcome message before first simulation
        st.info(
            "Configure your retirement parameters in the sidebar and click "
            "**RUN SIMULATION** to start the Monte Carlo analysis. "
            "The simulation will generate thousands of retirement scenarios "
            "to assess success probability and withdrawal sustainability."
        )


def _render_result_tabs(results, config,
                        tab1, tab1b, tab1c, tab2, tab2b,
                        tab3, tab3b, tab4, tab5, tab6, tab7, tab7b):
    """Render all simulation result tabs. Shared by sidebar and tabs layouts."""

    with tab1:
        st.subheader("Portfolio Value Projection")
        fig = portfolio_fan_chart(results, config)
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics table
        st.write("**Portfolio Statistics (End of Plan)**")
        final_vals = results.portfolio_values[:, -1]
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', '10th Percentile', '25th Percentile',
                       '75th Percentile', '90th Percentile', 'Minimum', 'Maximum'],
            'Value': [
                fmt_m(np.mean(final_vals)),
                fmt_m(np.median(final_vals)),
                fmt_m(np.percentile(final_vals, 10)),
                fmt_m(np.percentile(final_vals, 25)),
                fmt_m(np.percentile(final_vals, 75)),
                fmt_m(np.percentile(final_vals, 90)),
                fmt_m(np.min(final_vals)),
                fmt_m(np.max(final_vals)),
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with tab1b:
        st.subheader("Portfolio Success — Year-by-Year Survival")
        fig = portfolio_success_curve_chart(results, config)
        st.plotly_chart(fig, use_container_width=True)

    with tab1c:
        st.subheader("Simulated Portfolio Balances — Percentile Lines")
        _col_log, _col_infl = st.columns(2)
        with _col_log:
            _log_scale = st.checkbox("Logarithmic scale", value=False, key="balance_log")
        with _col_infl:
            _infl_adj = st.checkbox("Inflation adjusted", value=False, key="balance_infl")
        fig = portfolio_balance_lines_chart(results, config, log_scale=_log_scale, inflation_adjusted=_infl_adj)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Annual Withdrawals")
        fig = withdrawal_sustainability_chart(results, config)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Income Sources Over Time")
        fig = income_stack_chart(results, config)
        st.plotly_chart(fig, use_container_width=True)

    with tab2b:
        st.subheader("Annual Household Withdrawal — Primary vs Spouse")
        fig = withdrawal_schedule_chart(results, config)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Tax Burden Analysis")
        fig = tax_burden_chart(results)
        st.plotly_chart(fig, use_container_width=True)

        # Tax statistics table
        st.write("**Annual Tax Statistics (All Years)**")
        tax_stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', '10th Percentile', '90th Percentile'],
            'Annual Tax': [
                fmt_m(np.mean(results.taxes)),
                fmt_m(np.median(results.taxes)),
                fmt_m(np.percentile(results.taxes, 10)),
                fmt_m(np.percentile(results.taxes, 90)),
            ]
        })
        st.dataframe(tax_stats_df, use_container_width=True, hide_index=True)

    with tab3b:
        st.subheader("Income Breakdown by Age")
        st.caption("Deterministic income sources + median-path simulation data (nominal dollars).")

        from models.social_security import SocialSecurityModel
        _ib_ss_model = SocialSecurityModel()

        _ib_rows = []
        for _ib_idx, _ib_age in enumerate(results.ages):
            _ib_age_int = int(_ib_age)
            _ib_cum_inf = float(np.median(
                np.cumprod(1.0 + results.inflation_rates, axis=1)[:, _ib_idx]
            )) if results.inflation_rates.size > 0 else 1.0 + 0.025 * _ib_idx
            _ib_overlays = _ib_ss_model.get_income_overlays(_ib_age_int, config, _ib_cum_inf)

            if hasattr(results, 'gross_income') and results.gross_income.size > 0:
                _ib_gross = float(np.median(results.gross_income[:, _ib_idx]))
            else:
                _ib_gross = 0.0

            _ib_tax = float(np.median(results.taxes[:, _ib_idx]))
            _ib_eff = _ib_tax / _ib_gross if _ib_gross > 0 else 0.0

            _ib_rows.append({
                'Age': _ib_age_int,
                'SERP': fmt_k(_ib_overlays.get('serp_income', 0.0)),
                'Pension': fmt_k(_ib_overlays.get('pension_income', 0.0)),
                'Rental': fmt_k(_ib_overlays.get('rental_income', 0.0)),
                'SS': fmt_k(_ib_overlays.get('ss_income', 0.0)),
                'Gross Inc.': fmt_k(_ib_gross),
                'Tax': fmt_k(_ib_tax),
                'Eff. Rate': format_percent(_ib_eff, decimals=1),
            })

        _ib_df = pd.DataFrame(_ib_rows)
        st.dataframe(_ib_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Asset Allocation Strategy")
        fig = asset_allocation_chart(config)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Pre-Retirement Allocation**")
            pre_ret_df = pd.DataFrame({
                'Asset Class': list(config.pre_retirement_allocation.keys()),
                'Weight': [format_percent(v, decimals=1)
                           for v in config.pre_retirement_allocation.values()],
            })
            st.dataframe(pre_ret_df, use_container_width=True, hide_index=True)

        with col2:
            st.write("**Retirement Allocation**")
            ret_df = pd.DataFrame({
                'Asset Class': list(config.retirement_allocation.keys()),
                'Weight': [format_percent(v, decimals=1)
                           for v in config.retirement_allocation.values()],
            })
            st.dataframe(ret_df, use_container_width=True, hide_index=True)

    with tab5:
        st.subheader("Inflation Projections")
        fig = inflation_chart(results)
        st.plotly_chart(fig, use_container_width=True)

        # Inflation statistics
        st.write("**Inflation Statistics**")
        inflation_mean = np.mean(results.inflation_rates)
        inflation_std = np.std(results.inflation_rates)
        inflation_10pct = np.percentile(results.inflation_rates, 10)
        inflation_90pct = np.percentile(results.inflation_rates, 90)

        inflation_stats = pd.DataFrame({
            'Metric': ['Mean', 'Std Dev', '10th Percentile', '90th Percentile'],
            'Inflation Rate': [
                format_percent(inflation_mean, decimals=2),
                format_percent(inflation_std, decimals=2),
                format_percent(inflation_10pct, decimals=2),
                format_percent(inflation_90pct, decimals=2),
            ]
        })
        st.dataframe(inflation_stats, use_container_width=True, hide_index=True)

    with tab6:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Success Probability**")
            fig = success_probability_chart(results)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("**Portfolio Depletion Analysis**")
            fig = portfolio_depletion_histogram(results, config)
            st.plotly_chart(fig, use_container_width=True)

        # Detailed risk statistics
        st.write("**Risk Metrics**")
        n_failed = np.sum(~results.success_mask)
        n_sims = len(results.success_mask)
        successful_finals = results.portfolio_values[results.success_mask, -1]

        risk_stats = pd.DataFrame({
            'Metric': [
                'Success Rate',
                'Failed Simulations',
                'Success Probability',
                'Median Portfolio (Successful Only)',
                'Median Portfolio (All)',
                'Return to Portfolio Ratio (Median)',
            ],
            'Value': [
                f"{results.success_rate*100:.1f}%",
                f"{n_failed} of {n_sims}",
                format_percent(results.success_rate, decimals=1),
                format_currency(np.median(successful_finals)) if len(successful_finals) > 0 else "N/A",
                format_currency(np.median(results.portfolio_values[:, -1])),
                format_percent(np.median(results.portfolio_returns), decimals=2),
            ]
        })
        st.dataframe(risk_stats, use_container_width=True, hide_index=True)

    with tab7:
        st.subheader("Portfolio Statistics Summary")
        st.caption("Percentile distributions across all Monte Carlo paths.")

        try:
            rich = results.compute_rich_statistics()

            # Build the 8-metric × 5-percentile table
            _pctiles = [10, 25, 50, 75, 90]
            _metric_labels = {
                'twrr_nominal':             'TWRR (Nominal)',
                'twrr_real':                'TWRR (Real)',
                'end_balance_nominal':      'End Balance (Nominal)',
                'end_balance_real':          'End Balance (Real)',
                'max_drawdown':             'Max Drawdown',
                'max_drawdown_ex_cf':       'Max Drawdown (excl. Cash Flows)',
                'safe_withdrawal_rate':     'Safe Withdrawal Rate',
                'perpetual_withdrawal_rate': 'Perpetual Withdrawal Rate',
            }

            rows = []
            for key, label in _metric_labels.items():
                row = {'Metric': label}
                for p in _pctiles:
                    val = rich[key][p]
                    if 'balance' in key:
                        row[f'{p}th'] = fmt_m(val)
                    elif 'drawdown' in key:
                        row[f'{p}th'] = f"{val * 100:.1f}%"
                    else:
                        row[f'{p}th'] = f"{val * 100:.2f}%"
                rows.append(row)

            rich_df = pd.DataFrame(rows)
            st.dataframe(rich_df, use_container_width=True, hide_index=True)

            # Survival footnote
            st.caption(
                f"Survival Rate: {rich['survival_rate']*100:.1f}% | "
                f"Household Score: {rich['household_score']*100:.0f}/100"
            )

        except Exception as e:
            st.warning(f"Could not compute rich statistics: {e}")

    with tab7b:
        st.subheader("Simulation Summary — Common Sense Check")
        st.caption("Median-path summary for quick validation. All values in nominal dollars.")

        _csc_rows = []
        for _csc_idx, _csc_age in enumerate(results.ages):
            _csc_age_int = int(_csc_age)

            _csc_gross = 0.0
            if hasattr(results, 'gross_income') and results.gross_income.size > 0:
                _csc_gross = float(np.median(results.gross_income[:, _csc_idx]))

            _csc_spend = 0.0
            if hasattr(results, 'spend_amounts') and results.spend_amounts.size > 0:
                _csc_spend = float(np.median(results.spend_amounts[:, _csc_idx]))

            _csc_tax = float(np.median(results.taxes[:, _csc_idx]))
            _csc_draw = float(np.median(results.withdrawals[:, _csc_idx]))
            _csc_pv = float(np.median(results.portfolio_values[:, _csc_idx]))

            _csc_rows.append({
                'Age': _csc_age_int,
                'Total Income': fmt_k(_csc_gross),
                'Total Spend': fmt_k(_csc_spend) if _csc_spend > 0 else '—',
                'Total Tax': fmt_k(_csc_tax),
                'Portfolio Draw': fmt_k(_csc_draw) if _csc_draw > 0 else '—',
                'Remaining Portfolio': fmt_m(_csc_pv),
            })

        _csc_df = pd.DataFrame(_csc_rows)
        st.dataframe(_csc_df, use_container_width=True, hide_index=True)

        # Withdrawal breakdown by account type
        if (hasattr(results, 'trad_withdrawals') and results.trad_withdrawals.size > 0):
            st.subheader("Withdrawal Breakdown by Account Type")
            _strategy_label = getattr(config, 'withdrawal_method', 'fixed_real').replace('_', ' ').title()
            st.caption(f"Withdrawal strategy: **{_strategy_label}** — median path, nominal dollars.")
            _wd_rows = []
            _cum_total = 0.0
            for _wd_idx, _wd_age in enumerate(results.ages):
                _wd_trad = float(np.median(results.trad_withdrawals[:, _wd_idx]))
                _wd_roth = float(np.median(results.roth_withdrawals[:, _wd_idx]))
                _wd_taxable = float(np.median(results.taxable_withdrawals[:, _wd_idx]))
                _wd_total = float(np.median(results.withdrawals[:, _wd_idx]))
                _cum_total += _wd_total
                _wd_rows.append({
                    'Age': int(_wd_age),
                    'Traditional': fmt_k(_wd_trad) if _wd_trad > 0 else '—',
                    'Roth': fmt_k(_wd_roth) if _wd_roth > 0 else '—',
                    'Taxable': fmt_k(_wd_taxable) if _wd_taxable > 0 else '—',
                    'Annual Total': fmt_k(_wd_total) if _wd_total > 0 else '—',
                    'Cumulative': fmt_k(_cum_total) if _cum_total > 0 else '—',
                })
            st.dataframe(pd.DataFrame(_wd_rows), use_container_width=True, hide_index=True)

            # Consistency check: per-bucket sum ≈ gross withdrawal
            _wd_check_pass = True
            _wd_check_years = []
            for _wd_idx in range(len(results.ages)):
                _bucket_sum = float(np.median(
                    results.trad_withdrawals[:, _wd_idx] +
                    results.roth_withdrawals[:, _wd_idx] +
                    results.taxable_withdrawals[:, _wd_idx]
                ))
                _gross = float(np.median(results.withdrawals[:, _wd_idx]))
                if _gross > 0 and abs(_bucket_sum - _gross) / _gross > 0.05:
                    _wd_check_pass = False
                    _wd_check_years.append(int(results.ages[_wd_idx]))
            if _wd_check_pass:
                st.success("Per-bucket withdrawals sum to gross withdrawal across all years")
            else:
                st.warning(f"Per-bucket withdrawals differ from gross withdrawal at ages: {_wd_check_years[:5]}")

        # Inline validation checks
        st.markdown("**Validation Checks:**")
        st.info(
            "These checks compare key simulation outputs against your inputs "
            "to help spot configuration errors or unexpected results. "
            "**Starting vs End of Year 1** shows the effect of first-year "
            "investment returns, contributions, and withdrawals on your "
            "portfolio. **Tax vs Income** verifies taxes never exceed total "
            "income. **Portfolio at Life Expectancy** shows whether the "
            "median simulation path sustains your portfolio to the end of "
            "the planning horizon."
        )
        _configured_total = (config.traditional_balance + config.roth_balance +
                             config.taxable_balance)
        _end_yr0_pv = float(np.median(results.portfolio_values[:, 0]))
        st.success(
            f"Starting portfolio: {fmt_m(_configured_total)} | "
            f"End of Year 1: {fmt_m(_end_yr0_pv)} "
            f"({'+'if _end_yr0_pv >= _configured_total else ''}"
            f"{(_end_yr0_pv/_configured_total - 1)*100:.1f}% after returns, "
            f"contributions, and withdrawals)"
        )

        if hasattr(results, 'gross_income') and results.gross_income.size > 0:
            _med_gross = np.median(results.gross_income, axis=0)
            _med_tax = np.median(results.taxes, axis=0)
            _tax_exceeds = np.any(_med_tax > _med_gross * 1.01)  # 1% tolerance
            if not _tax_exceeds:
                st.success("Taxes never exceed gross income across all years")
            else:
                st.warning("Taxes exceed gross income in some years — review Tax Analysis tab")

        _final_pv = float(np.median(results.portfolio_values[:, -1]))
        if _final_pv > 0:
            st.success(f"Median portfolio at life expectancy: {fmt_m(_final_pv)} (positive)")
        else:
            st.warning(f"Median portfolio at life expectancy: {fmt_m(_final_pv)} (depleted)")

    # Export button
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 2])

    with col2:
        if st.button("📄 Generate Report (.docx + .pdf)", use_container_width=True,
                     type="primary"):
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
            with st.spinner("Generating report — building charts and document..."):
                try:
                    generator = ReportGenerator(results, output_dir=output_dir)
                    report_paths = generator.generate()
                    docx_path = report_paths["docx"]
                    pdf_path  = report_paths["pdf"]
                    st.success(f"✓ Report saved to outputs/")
                    st.code(os.path.basename(docx_path))
                    if pdf_path:
                        st.code(os.path.basename(pdf_path))
                    # Offer docx as inline download
                    with open(docx_path, "rb") as f:
                        st.download_button(
                            label="⬇ Download .docx",
                            data=f.read(),
                            file_name=os.path.basename(docx_path),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                        )
                    if pdf_path and os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="⬇ Download .pdf",
                                data=f.read(),
                                file_name=os.path.basename(pdf_path),
                                mime="application/pdf",
                                use_container_width=True,
                            )
                except Exception as e:
                    st.error(f"Report generation failed: {e}")

    with col1:
        if st.button("📊 Export Results to Excel", use_container_width=True):
            # Create Excel workbook with multiple sheets
            output = BytesIO()

            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    'Parameter': [
                        'Current Age', 'Retirement Age', 'Life Expectancy',
                        'Initial Portfolio', 'Annual Withdrawal',
                        'Success Rate', 'Median Final Balance',
                        'Number of Simulations', 'Simulation Date'
                    ],
                    'Value': [
                        str(config.current_age),
                        str(config.retirement_age),
                        str(config.life_expectancy),
                        format_currency(config.total_portfolio_value),
                        format_currency(config.annual_withdrawal_real),
                        format_percent(results.success_rate, decimals=1),
                        format_currency(np.median(results.portfolio_values[:, -1])),
                        str(config.n_simulations),
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                # Portfolio projections
                portfolio_export = pd.DataFrame({
                    'Age': results.ages,
                    'Median_Portfolio': np.percentile(results.portfolio_values, 50, axis=0),
                    '10th_Percentile': np.percentile(results.portfolio_values, 10, axis=0),
                    '25th_Percentile': np.percentile(results.portfolio_values, 25, axis=0),
                    '75th_Percentile': np.percentile(results.portfolio_values, 75, axis=0),
                    '90th_Percentile': np.percentile(results.portfolio_values, 90, axis=0),
                })
                portfolio_export.to_excel(writer, sheet_name='Portfolio', index=False)

                # Withdrawals
                withdrawal_export = pd.DataFrame({
                    'Age': results.ages,
                    'Median_Withdrawal': np.percentile(results.withdrawals, 50, axis=0),
                    '10th_Percentile': np.percentile(results.withdrawals, 10, axis=0),
                    '90th_Percentile': np.percentile(results.withdrawals, 90, axis=0),
                })
                withdrawal_export.to_excel(writer, sheet_name='Withdrawals', index=False)

                # Taxes
                tax_export = pd.DataFrame({
                    'Age': results.ages,
                    'Median_Tax': np.percentile(results.taxes, 50, axis=0),
                    '10th_Percentile': np.percentile(results.taxes, 10, axis=0),
                    '90th_Percentile': np.percentile(results.taxes, 90, axis=0),
                })
                tax_export.to_excel(writer, sheet_name='Taxes', index=False)

            output.seek(0)
            filename = datetime.now().strftime("%Y %B %d Retirement Analysis") + ".xlsx"
            st.download_button(
                label="✓ Download Excel File",
                data=output,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )



def _main_tabs_layout():
    """Alternative layout: config in top tabs, results below."""
    st.title("Retirement Monte Carlo Financial Planner")
    _file_status_bar()
    st.divider()

    # Config tabs across the top
    config_tabs = st.tabs(_CONFIG_SECTIONS)
    containers = dict(zip(_CONFIG_SECTIONS, config_tabs))

    # CSV panel in first tab
    _csv_sidebar_panel(container=containers["CSV"])

    # Build config using tab containers (skip CSV — already handled)
    config = build_sidebar_config(containers=containers)

    # Run button in main area
    st.divider()
    if st.button("🚀 RUN SIMULATION", use_container_width=True, type="primary"):
        _run_simulation(config)

    # Display results if available
    if st.session_state.simulation_results is not None:
        results = st.session_state.simulation_results
        config = st.session_state.config_cache

        # Display success rate prominently
        success_rate = results.success_rate

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Success Rate",
                f"{success_rate*100:.1f}%",
                delta=None,
            )

        with col2:
            _hs = getattr(results, 'household_score', success_rate)
            st.metric(
                "Household Score",
                f"{_hs*100:.0f}/100",
            )

        with col3:
            median_final = np.percentile(results.portfolio_values[:, -1], 50)
            st.metric(
                "Median Final Balance",
                fmt_m(median_final),
            )

        with col4:
            final_balance_10pct = np.percentile(results.portfolio_values[:, -1], 10)
            st.metric(
                "10th Percentile Final",
                fmt_m(final_balance_10pct),
            )

        with col5:
            median_tax_all_years = float(np.median(results.taxes))
            st.metric(
                "Median Annual Tax",
                fmt_m(median_tax_all_years),
            )

        # Result tabs — same as sidebar layout
        tab1, tab1b, tab1c, tab2, tab2b, tab3, tab3b, tab4, tab5, tab6, tab7, tab7b = st.tabs(
            ["Portfolio Projection", "Portfolio Success", "Portfolio Balances",
             "Income & Withdrawals", "Withdrawal Schedule",
             "Tax Analysis", "Income Breakdown", "Asset Allocation", "Inflation",
             "Risk Analysis", "Portfolio Statistics", "Simulation Summary"]
        )

        # Delegate to the shared results renderer
        _render_result_tabs(results, config,
                            tab1, tab1b, tab1c, tab2, tab2b,
                            tab3, tab3b, tab4, tab5, tab6, tab7, tab7b)

    else:
        st.info(
            "Configure your retirement parameters in the tabs above and click "
            "**RUN SIMULATION** to start the Monte Carlo analysis."
        )


if __name__ == "__main__":
    main()
