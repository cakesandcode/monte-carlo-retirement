"""
Monte Carlo Simulation Engine for Retirement Financial Planning.

Orchestrates all sub-models (asset returns, inflation, portfolio mechanics,
tax, Social Security) to run complete Monte Carlo retirement simulations.

Core simulation loop:
  1. Generate return and inflation paths for all simulations
  2. For each simulation and year:
     - Process portfolio state (contributions, withdrawals, returns)
     - Calculate income overlays (SS, pension, other)
     - Compute tax liability
     - Track results
  3. Aggregate results and compute success statistics

Version: 1.0.0

Boundary / Edge Cases:
  - Zero portfolio at start: all paths show immediate depletion; success_rate=0.
  - n_simulations=1: valid but no statistical confidence; convergence check
    will flag as insufficient.
  - life_expectancy = current_age: n_years=0; arrays are empty.
  - All incomes zero + zero withdrawals: portfolio grows with returns only;
    success_rate=100%.
  - Extreme return scenarios (GBM tail): annual returns > 100% or < -90% can
    occur; portfolio values may become astronomical in right tail.
"""

import copy
import numpy as np
from typing import Optional
from config.defaults import SimulationConfig, SimulationResults, ASSET_CLASSES
from models.asset_returns import AssetReturnModel
from models.inflation import InflationModel
from models.portfolio import PortfolioMechanics
from models.tax import TaxCalculator
from models.social_security import SocialSecurityModel


class MonteCarloSimulator:
    """
    Main orchestration engine for retirement simulations.

    Runs Monte Carlo simulations combining stochastic asset returns, inflation,
    portfolio mechanics, and tax calculations. Produces success probabilities
    and detailed year-by-year outcomes.

    Attributes:
        config (SimulationConfig): Simulation configuration.
        asset_model (AssetReturnModel): Asset return generator.
        inflation_model (InflationModel): Inflation rate generator.
        portfolio_mech (PortfolioMechanics): Portfolio mechanics.
        tax_calc (TaxCalculator): Tax calculator.
        ss_model (SocialSecurityModel): Social Security model.
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize simulator with configuration.

        Args:
            config: SimulationConfig specifying all simulation parameters.
        """
        self.config = config
        self.asset_model = AssetReturnModel()
        self.inflation_model = InflationModel()
        self.portfolio_mech = PortfolioMechanics()
        self.tax_calc = TaxCalculator()
        self.ss_model = SocialSecurityModel()

    def run(self) -> SimulationResults:
        """
        Execute full Monte Carlo retirement simulation.

        Steps:
          1. Pre-generate return matrices (n_sims, n_years, n_assets)
          2. Pre-generate inflation matrices (n_sims, n_years)
          3. Outer loop: iterate over n_simulations
          4. Inner loop: iterate over years (current_age to life_expectancy)
          5. For each (sim, year): process annual state transitions
          6. Build SimulationResults with aggregated data

        Returns:
            SimulationResults with portfolio_values, withdrawals, taxes,
            success_rate, and other metrics for all simulations.
        """
        n_years = self.config.life_expectancy - self.config.current_age + 1
        n_sims = self.config.n_simulations

        # Pre-generate all returns and inflation paths
        print("Generating return and inflation paths...")
        return_paths = self.asset_model.get_returns(self.config, n_years)
        inflation_general, inflation_healthcare = (
            self.inflation_model.get_inflation(self.config, n_years)
        )

        # Allocate output arrays
        portfolio_values = np.zeros((n_sims, n_years))
        traditional_values = np.zeros((n_sims, n_years))
        roth_values = np.zeros((n_sims, n_years))
        taxable_values = np.zeros((n_sims, n_years))
        withdrawals = np.zeros((n_sims, n_years))
        trad_withdrawals = np.zeros((n_sims, n_years))
        roth_withdrawals = np.zeros((n_sims, n_years))
        taxable_withdrawals = np.zeros((n_sims, n_years))
        taxes = np.zeros((n_sims, n_years))
        ss_income = np.zeros((n_sims, n_years))
        inflation_rates = np.zeros((n_sims, n_years))
        portfolio_returns = np.zeros((n_sims, n_years))
        success_mask = np.ones(n_sims, dtype=bool)
        # Spend tracking
        spend_amounts = np.zeros((n_sims, n_years))
        spend_shortfalls = np.zeros((n_sims, n_years))

        # Run simulation loop
        print(f"Running {n_sims} simulations over {n_years} years...")
        for sim_idx in range(n_sims):
            if (sim_idx + 1) % max(1, n_sims // 10) == 0:
                print(f"  Simulation {sim_idx + 1}/{n_sims}")

            # Initialize account balances
            trad_bal = self.config.traditional_balance
            roth_bal = self.config.roth_balance
            taxable_bal = self.config.taxable_balance

            cum_inflation = 1.0
            cum_healthcare_inflation = 1.0
            _taxable_cost_basis = taxable_bal

            for year_idx in range(n_years):
                current_age = self.config.current_age + year_idx
                is_retired = current_age >= self.config.retirement_age

                # Store inflation rate
                inflation_rates[sim_idx, year_idx] = inflation_general[sim_idx, year_idx]

                # Year 0 = current year (today's dollars). Inflation compounds starting year 1.
                if year_idx > 0:
                    cum_inflation *= (1.0 + inflation_general[sim_idx, year_idx])
                    cum_healthcare_inflation *= (1.0 + inflation_healthcare[sim_idx, year_idx])

                # Get annual returns for this year
                annual_returns = return_paths[sim_idx, year_idx, :]

                # Process portfolio year (3-bucket)
                portfolio_state = self.portfolio_mech.process_year(
                    age=current_age,
                    traditional=trad_bal,
                    roth=roth_bal,
                    taxable=taxable_bal,
                    annual_returns=annual_returns,
                    inflation_rate=inflation_general[sim_idx, year_idx],
                    cum_inflation=cum_inflation,
                    config=self.config,
                    is_retired=is_retired,
                )

                # Update balances
                trad_bal = portfolio_state['traditional']
                roth_bal = portfolio_state['roth']
                taxable_bal = portfolio_state['taxable']
                total_portfolio = portfolio_state['total_portfolio']
                gross_withdrawal = portfolio_state['gross_withdrawal']
                rmd_amount = portfolio_state['rmd_amount']
                # Actual per-bucket draws
                _actual_trad_withdraw = portfolio_state.get('trad_withdraw', 0.0)
                _actual_roth_withdraw = portfolio_state.get('roth_withdraw', 0.0)
                _actual_taxable_withdraw = portfolio_state.get('taxable_withdraw', 0.0)

                # Get income overlays (SS, pension, rental, part-time)
                income_overlay = self.ss_model.get_income_overlays(
                    current_age,
                    self.config,
                    cum_inflation
                )

                # ── SPEND NETTING ────────────────────────────────────────
                # Compute net = total_income - spend.
                # If net < 0 -> shortfall; draw additional from portfolio.
                # If net >= 0 -> surplus reinvested or ignored per config.
                spend_shortfall = 0.0
                net_after_spend = 0.0
                _spend_start = getattr(self.config, 'spend_start_age', self.config.retirement_age)
                _spend_real = getattr(self.config, 'spend_annual_real', 0.0)
                if _spend_real > 0 and current_age >= _spend_start:
                    annual_spend = _spend_real * cum_inflation
                    total_income = income_overlay.get('total_other_income', 0.0) + rmd_amount
                    net_after_spend = total_income - annual_spend

                    if net_after_spend < 0:
                        # Shortfall — draw additional from portfolio
                        spend_shortfall = abs(net_after_spend)
                        _sf_remaining = spend_shortfall

                        if _sf_remaining > 0 and taxable_bal > 0:
                            _sf_take = min(_sf_remaining, taxable_bal)
                            taxable_bal -= _sf_take
                            _sf_remaining -= _sf_take
                            _actual_taxable_withdraw += _sf_take

                        if _sf_remaining > 0 and trad_bal > 0:
                            _sf_take = min(_sf_remaining, trad_bal)
                            trad_bal -= _sf_take
                            _sf_remaining -= _sf_take
                            _actual_trad_withdraw += _sf_take

                        if _sf_remaining > 0 and roth_bal > 0:
                            _sf_take = min(_sf_remaining, roth_bal)
                            roth_bal -= _sf_take
                            _sf_remaining -= _sf_take
                            _actual_roth_withdraw += _sf_take

                        gross_withdrawal += (spend_shortfall - _sf_remaining)
                        total_portfolio = trad_bal + roth_bal + taxable_bal

                    elif net_after_spend > 0:
                        # Surplus — reinvest RMD excess into taxable
                        _surplus_mode = getattr(self.config, 'spend_surplus_mode', 'ignore')
                        _rmd_excess = max(0.0, rmd_amount - max(0.0, annual_spend - (total_income - rmd_amount)))
                        _other_surplus = net_after_spend - _rmd_excess
                        _reinvest = _rmd_excess
                        if _surplus_mode == 'reinvest':
                            _reinvest += max(0.0, _other_surplus)
                        if _reinvest > 0:
                            taxable_bal += _reinvest
                            _taxable_cost_basis += _reinvest
                            total_portfolio = trad_bal + roth_bal + taxable_bal

                # Calculate healthcare costs and subtract from portfolio
                annual_healthcare = 0.0
                _hc_trad_draw = 0.0
                if self.config.include_healthcare_costs and is_retired:
                    _hc_is_real = getattr(self.config, 'healthcare_is_real', True)
                    if _hc_is_real:
                        annual_healthcare = (
                            self.config.annual_healthcare_cost_real *
                            cum_healthcare_inflation
                        )
                    else:
                        annual_healthcare = self.config.annual_healthcare_cost_real
                    # Ordered draw for healthcare: taxable -> traditional -> roth.
                    # Healthcare is a non-discretionary expense; draw from
                    # tax-inefficient accounts first, preserving Roth tax-free growth.
                    _hc_remaining = min(annual_healthcare, trad_bal + roth_bal + taxable_bal)
                    _hc_drawn = 0.0
                    _hc_trad_draw = 0.0

                    if _hc_remaining > 0 and taxable_bal > 0:
                        _hc_take = min(_hc_remaining, taxable_bal)
                        taxable_bal -= _hc_take
                        _hc_remaining -= _hc_take
                        _hc_drawn += _hc_take

                    if _hc_remaining > 0 and trad_bal > 0:
                        _hc_take = min(_hc_remaining, trad_bal)
                        trad_bal -= _hc_take
                        _hc_remaining -= _hc_take
                        _hc_drawn += _hc_take
                        _hc_trad_draw += _hc_take

                    if _hc_remaining > 0 and roth_bal > 0:
                        _hc_take = min(_hc_remaining, roth_bal)
                        roth_bal -= _hc_take
                        _hc_remaining -= _hc_take
                        _hc_drawn += _hc_take

                    total_portfolio -= _hc_drawn

                # Use actual per-bucket withdrawal amounts from portfolio.py
                trad_withdraw = _actual_trad_withdraw
                roth_withdraw = _actual_roth_withdraw
                taxable_withdraw = _actual_taxable_withdraw
                trad_withdraw_for_tax = trad_withdraw + _hc_trad_draw

                # Estimate embedded capital gains using cost basis tracking.
                if taxable_bal > 0 and taxable_withdraw > 0:
                    _gain_ratio = max(0.0, min(1.0,
                        (taxable_bal - _taxable_cost_basis) / taxable_bal
                    )) if taxable_bal > 0 else 0.0
                    ltcg_amount = taxable_withdraw * _gain_ratio
                    _basis_withdrawn = taxable_withdraw * (1.0 - _gain_ratio)
                    _taxable_cost_basis = max(0.0, _taxable_cost_basis - _basis_withdrawn)
                else:
                    ltcg_amount = 0.0

                # Calculate taxes
                # RMD is already included in gross_withdrawal (from portfolio.py),
                # so subtract it from trad_withdraw to avoid double-counting.
                _trad_withdraw_for_tax = max(0.0, trad_withdraw_for_tax - rmd_amount)

                tax_result = self.tax_calc.calculate_annual_tax(
                    age=current_age,
                    filing_status=self.config.filing_status,
                    state_tax_rate=self.config.state_tax_rate,
                    include_state_tax=self.config.include_state_tax,
                    traditional_withdrawal=_trad_withdraw_for_tax,
                    roth_withdrawal=roth_withdraw,
                    taxable_withdrawal=taxable_withdraw,
                    ss_income=income_overlay['ss_income'],
                    pension_income=income_overlay['pension_income'],
                    rental_income=income_overlay['rental_income'],
                    part_time_income=income_overlay['part_time_income'],
                    rmd_amount=rmd_amount,
                    ltcg_amount=ltcg_amount,
                    serp_income=income_overlay.get('serp_income', 0.0),
                    spouse_age=(self.config.spouse_age + year_idx
                                if self.config.spouse_age is not None else None),
                )

                annual_tax = tax_result['total_tax']
                # Add IRMAA surcharge
                irmaa = self.tax_calc.estimate_irmaa_surcharge(
                    tax_result['agi'],
                    self.config.filing_status
                )
                annual_tax += irmaa

                # ── DEDUCT TAXES FROM PORTFOLIO ──────────────────────────
                # Deduct from Traditional and Taxable only — Roth is tax-free.
                _income_surplus = 0.0
                if _spend_real > 0 and current_age >= _spend_start:
                    _income_surplus = max(0.0, net_after_spend)
                _tax_from_portfolio = max(0.0, annual_tax - _income_surplus)

                _taxable_pool = trad_bal + taxable_bal
                if _taxable_pool > 0 and _tax_from_portfolio > 0:
                    _tax_draw = min(_tax_from_portfolio, _taxable_pool)
                    _tax_frac = _tax_draw / _taxable_pool
                    trad_bal    -= trad_bal * _tax_frac
                    taxable_bal -= taxable_bal * _tax_frac
                    total_portfolio -= _tax_draw
                elif _tax_from_portfolio > 0 and roth_bal > 0:
                    _tax_draw = min(_tax_from_portfolio, roth_bal)
                    roth_bal -= _tax_draw
                    total_portfolio -= _tax_draw

                # Store results
                portfolio_values[sim_idx, year_idx] = total_portfolio
                traditional_values[sim_idx, year_idx] = trad_bal
                roth_values[sim_idx, year_idx] = roth_bal
                taxable_values[sim_idx, year_idx] = taxable_bal
                withdrawals[sim_idx, year_idx] = gross_withdrawal
                trad_withdrawals[sim_idx, year_idx] = trad_withdraw
                roth_withdrawals[sim_idx, year_idx] = roth_withdraw
                taxable_withdrawals[sim_idx, year_idx] = taxable_withdraw
                taxes[sim_idx, year_idx] = annual_tax
                ss_income[sim_idx, year_idx] = income_overlay['ss_income']
                portfolio_returns[sim_idx, year_idx] = portfolio_state['portfolio_return']
                # Spend tracking
                if _spend_real > 0 and current_age >= _spend_start:
                    spend_amounts[sim_idx, year_idx] = _spend_real * cum_inflation
                spend_shortfalls[sim_idx, year_idx] = spend_shortfall

                # Check for portfolio depletion
                if total_portfolio <= 0:
                    success_mask[sim_idx] = False

        # Compute success statistics
        success_rate = float(np.sum(success_mask)) / n_sims

        # Generate year/age arrays
        ages = np.arange(
            self.config.current_age,
            self.config.current_age + n_years
        )
        years = np.arange(0, n_years)

        # Build and return SimulationResults
        results = SimulationResults(
            config=self.config,
            portfolio_values=portfolio_values,
            traditional_values=traditional_values,
            roth_values=roth_values,
            taxable_values=taxable_values,
            withdrawals=withdrawals,
            taxes=taxes,
            ss_income=ss_income,
            inflation_rates=inflation_rates,
            portfolio_returns=portfolio_returns,
            success_mask=success_mask,
            success_rate=success_rate,
            ages=ages,
            years=years,
            spend_amounts=spend_amounts,
            spend_shortfalls=spend_shortfalls,
            trad_withdrawals=trad_withdrawals,
            roth_withdrawals=roth_withdrawals,
            taxable_withdrawals=taxable_withdrawals,
        )

        return results

    def run_deterministic(self) -> dict:
        """
        Run single deterministic simulation using median assumptions.

        Uses median historical returns and mean inflation (no randomness).
        Useful as a baseline for comparison against stochastic simulations.

        Returns:
            Dict with year-by-year results:
              - ages: Array of ages
              - years: Array of years
              - portfolio_values: Year-by-year total portfolio
              - traditional_values: Traditional account balances
              - roth_values: Roth account balances
              - taxable_values: Taxable account balances
              - withdrawals: Annual withdrawals
              - taxes: Annual taxes
        """
        # Use median returns (simple assumption: historical mean)
        median_returns = self.asset_model.historical_mean.copy()
        median_inflation = self.config.inflation_mean

        n_years = self.config.life_expectancy - self.config.current_age + 1

        # Allocate outputs
        ages = []
        portfolio_vals = []
        trad_vals = []
        roth_vals = []
        taxable_vals = []
        withdrawal_vals = []
        tax_vals = []

        # Initialize balances
        trad_bal = self.config.traditional_balance
        roth_bal = self.config.roth_balance
        taxable_bal = self.config.taxable_balance
        cum_inflation = 1.0

        for year_idx in range(n_years):
            current_age = self.config.current_age + year_idx
            is_retired = current_age >= self.config.retirement_age

            ages.append(current_age)
            # Year 0 = current year (today's dollars). Inflation compounds starting year 1.
            if year_idx > 0:
                cum_inflation *= (1.0 + median_inflation)

            # Process portfolio year
            portfolio_state = self.portfolio_mech.process_year(
                age=current_age,
                traditional=trad_bal,
                roth=roth_bal,
                taxable=taxable_bal,
                annual_returns=median_returns,
                inflation_rate=median_inflation,
                cum_inflation=cum_inflation,
                config=self.config,
                is_retired=is_retired
            )

            trad_bal = portfolio_state['traditional']
            roth_bal = portfolio_state['roth']
            taxable_bal = portfolio_state['taxable']
            total_portfolio = portfolio_state['total_portfolio']
            gross_withdrawal = portfolio_state['gross_withdrawal']

            # Get income overlays
            income_overlay = self.ss_model.get_income_overlays(
                current_age,
                self.config,
                cum_inflation
            )

            # Simplified tax (use median assumptions)
            annual_tax = 0.25 * (gross_withdrawal + income_overlay['ss_income'])

            # Store results
            portfolio_vals.append(total_portfolio)
            trad_vals.append(trad_bal)
            roth_vals.append(roth_bal)
            taxable_vals.append(taxable_bal)
            withdrawal_vals.append(gross_withdrawal)
            tax_vals.append(annual_tax)

        return {
            'ages': np.array(ages),
            'years': np.arange(len(ages)),
            'portfolio_values': np.array(portfolio_vals),
            'traditional_values': np.array(trad_vals),
            'roth_values': np.array(roth_vals),
            'taxable_values': np.array(taxable_vals),
            'withdrawals': np.array(withdrawal_vals),
            'taxes': np.array(tax_vals),
        }

    def calculate_safe_withdrawal_rate(self) -> float:
        """
        Estimate safe withdrawal rate for 90% success probability.

        Uses binary search to find the withdrawal amount that results in
        exactly 90% of simulations surviving to life expectancy.

        Returns:
            Safe withdrawal rate (as decimal, e.g., 0.04 for 4%).

        Raises:
            ValueError: If no withdrawal rate between 0% and 10% achieves goal.
        """
        # Binary search bounds
        low_rate = 0.01
        high_rate = 0.10
        target_success = 0.90
        tolerance = 0.005

        for _ in range(20):  # Max iterations
            mid_rate = (low_rate + high_rate) / 2.0

            # Run simulation at this withdrawal rate (deep copy to avoid mutating original)
            test_config = copy.deepcopy(self.config)
            test_config.withdrawal_rate = mid_rate
            test_config.withdrawal_method = 'percentage'

            # Quick test: use fewer simulations
            test_config.n_simulations = min(1000, self.config.n_simulations)

            test_sim = MonteCarloSimulator(test_config)
            test_results = test_sim.run()

            success = test_results.success_rate

            if abs(success - target_success) < tolerance:
                return mid_rate

            if success < target_success:
                # Success too low -> need lower withdrawal rate to survive longer
                high_rate = mid_rate
            else:
                # Success high enough -> can afford higher withdrawal rate
                low_rate = mid_rate

        # Return best guess
        return (low_rate + high_rate) / 2.0

    def calculate_success_at_withdrawal(self, withdrawal_real: float) -> float:
        """
        Calculate success probability for a given real withdrawal amount.

        Runs simulation with annual_withdrawal_real set to the given amount
        and returns the success rate.

        Args:
            withdrawal_real: Real (today's dollars) annual withdrawal amount.

        Returns:
            Success rate (0.0 to 1.0).
        """
        test_config = copy.deepcopy(self.config)
        test_config.withdrawal_method = 'fixed_real'
        test_config.annual_withdrawal_real = withdrawal_real

        simulator = MonteCarloSimulator(test_config)
        results = simulator.run()

        return results.success_rate
