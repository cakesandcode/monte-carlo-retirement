"""
Plotly chart builders for retirement financial modeling.

All charts use dark theme (black background, white text) for professional,
readable output. Functions take SimulationResults and SimulationConfig objects
and return fully configured go.Figure objects ready for display in Streamlit.

Version: 1.0.0
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as px_colors
from config.defaults import SimulationConfig, SimulationResults
from models.asset_returns import AssetReturnModel
from utils.helpers import format_currency, format_percent, calculate_percentiles, fmt_k, fmt_m


# Dark theme configuration (used by all charts)
DARK_TEMPLATE = "plotly_dark"
DARK_BGCOLOR = '#000000'
PLOT_BGCOLOR = '#111111'
FONT_COLOR = 'white'
GRID_COLOR = '#333333'


def _apply_dark_theme(fig: go.Figure) -> go.Figure:
    """
    Apply consistent dark theme to any Plotly figure.

    Args:
        fig: Plotly figure to modify in-place.

    Returns:
        Modified figure with dark theme applied.
    """
    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor=DARK_BGCOLOR,
        plot_bgcolor=PLOT_BGCOLOR,
        font=dict(color=FONT_COLOR, family="Arial, sans-serif", size=11),
        hovermode='x unified',
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, showline=True, linewidth=1, linecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, showline=True, linewidth=1, linecolor=GRID_COLOR)
    return fig


def portfolio_fan_chart(results: SimulationResults, config: SimulationConfig) -> go.Figure:
    """
    Create fan chart showing portfolio value percentile bands over time.

    Displays multiple percentile bands (10th, 25th, 50th, 75th, 90th) with
    shaded regions. The 50th percentile (median) is shown as a white line.
    Percentile bands are rendered as filled areas between successive percentiles.
    A vertical dashed line marks the retirement age.

    Args:
        results: SimulationResults from completed simulation.
        config: Original SimulationConfig.

    Returns:
        Plotly Figure with portfolio value fan chart.
    """
    ages = results.ages
    portfolio_data = np.maximum(results.portfolio_values, 0)  # clamp to 0

    # Calculate percentiles
    percentiles = [10, 25, 50, 75, 90]
    perc_dict = calculate_percentiles(portfolio_data, percentiles)

    # Create figure
    fig = go.Figure()

    # Add shaded region: 10th to 90th percentile (red-ish, lowest confidence)
    fig.add_trace(go.Scatter(
        x=ages, y=perc_dict[90],
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='90th percentile',
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=perc_dict[10],
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(255, 100, 100, 0.2)',
        name='10th-90th percentile',
        hoverinfo='skip',
    ))

    # Add shaded region: 25th to 75th percentile (yellow-ish, medium confidence)
    fig.add_trace(go.Scatter(
        x=ages, y=perc_dict[75],
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=perc_dict[25],
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(255, 200, 100, 0.3)',
        name='25th-75th percentile',
        hoverinfo='skip',
    ))

    # Add median line (50th percentile) - white, prominent
    fig.add_trace(go.Scatter(
        x=ages, y=perc_dict[50],
        mode='lines',
        line=dict(color='white', width=3),
        name='Median (50th percentile)',
        hovertemplate='<b>Age: %{x}</b><br>Median Portfolio: ' +
                      '<b>%{customdata}</b><extra></extra>',
        customdata=[fmt_m(v) for v in perc_dict[50]],
    ))

    # Retirement age vline
    if min(ages) <= config.retirement_age <= max(ages):
        fig.add_vline(
            x=config.retirement_age,
            line_dash='dash',
            line_color='rgba(255, 255, 255, 0.5)',
            annotation_text='Retirement',
            annotation_position='top right',
            annotation_font_size=10,
            annotation_font_color=FONT_COLOR,
        )

    # Update layout
    fig.update_layout(
        title='Portfolio Value — Monte Carlo Percentile Bands',
        xaxis_title='Age',
        yaxis_title='Portfolio Value ($)',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
        ),
    )

    # Format y-axis as currency
    fig.update_yaxes(
        tickformat='$,.0f',
    )

    return _apply_dark_theme(fig)


def success_probability_chart(results: SimulationResults) -> go.Figure:
    """
    Create gauge or indicator chart showing success probability.

    Color-codes the success rate: green if > 85%, yellow if 70-85%, red if < 70%.
    Displays the success percentage prominently, plus median final portfolio value.

    Args:
        results: SimulationResults from completed simulation.

    Returns:
        Plotly Figure with success probability indicator.
    """
    success_rate = results.success_rate * 100
    final_values = results.portfolio_values[:, -1]
    median_final = np.median(final_values)

    # Determine color based on success rate
    if success_rate >= 85:
        gauge_color = 'green'
    elif success_rate >= 70:
        gauge_color = 'yellow'
    else:
        gauge_color = 'red'

    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=success_rate,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': 'Success Probability (90+ Year Plan)'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, 70], 'color': 'rgba(255, 100, 100, 0.3)'},
                {'range': [70, 85], 'color': 'rgba(255, 200, 100, 0.3)'},
                {'range': [85, 100], 'color': 'rgba(100, 200, 100, 0.3)'},
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 2},
                'thickness': 0.75,
                'value': 90,
            }
        },
        number={'suffix': '%'},
        delta={'reference': 80, 'suffix': ' vs 80% target'},
    ))

    # Add annotation for median final balance
    fig.add_annotation(
        text=f"Median Final Balance: {format_currency(median_final, decimals=0)}",
        x=0.5, y=-0.15,
        xref='paper', yref='paper',
        showarrow=False,
        font=dict(size=12, color=FONT_COLOR),
    )

    fig.update_layout(height=500)
    return _apply_dark_theme(fig)


def withdrawal_sustainability_chart(results: SimulationResults, config: SimulationConfig) -> go.Figure:
    """
    Create chart showing annual withdrawals over time with percentile bands.

    Shows median withdrawal vs target withdrawal line. Shaded region between
    25th-75th percentiles indicates variability. All values in today's dollars.

    Args:
        results: SimulationResults from completed simulation.
        config: Original SimulationConfig.

    Returns:
        Plotly Figure with withdrawal sustainability chart.
    """
    ages = results.ages
    withdrawal_data = results.withdrawals  # shape (n_sims, n_years)

    # Calculate percentiles
    percentiles = [25, 50, 75]
    perc_dict = calculate_percentiles(withdrawal_data, percentiles)

    # Create figure
    fig = go.Figure()

    # Add shaded region: 25th to 75th percentile
    fig.add_trace(go.Scatter(
        x=ages, y=perc_dict[75],
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=perc_dict[25],
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(100, 150, 255, 0.2)',
        name='25th-75th percentile',
        hoverinfo='skip',
    ))

    # Add median withdrawal line
    fig.add_trace(go.Scatter(
        x=ages, y=perc_dict[50],
        mode='lines',
        line=dict(color='white', width=2),
        name='Median Withdrawal',
        hovertemplate='<b>Age: %{x}</b><br>Withdrawal: ' +
                      '<b>%{customdata}</b><extra></extra>',
        customdata=[fmt_k(v) for v in perc_dict[50]],
    ))

    # Add target withdrawal line (fixed_real method only).
    if config.withdrawal_method == 'fixed_real':
        target = config.annual_withdrawal_real
        if target > 0:
            fig.add_hline(
                y=target,
                line_dash='dash',
                line_color='rgba(100, 255, 100, 0.7)',
                annotation_text=f'Target: {format_currency(target, decimals=0)}',
                annotation_position='right',
                annotation_font_size=10,
                annotation_font_color='rgba(100, 255, 100, 1)',
            )

    # Retirement age vline
    if min(ages) <= config.retirement_age <= max(ages):
        fig.add_vline(
            x=config.retirement_age,
            line_dash='dash',
            line_color='rgba(255,255,255,0.4)',
            annotation_text='Retirement',
            annotation_position='top right',
            annotation_font_size=9,
            annotation_font_color='rgba(255,255,255,0.6)',
        )

    # Update layout
    fig.update_layout(
        title='Annual Withdrawal Sustainability (Real Dollars)',
        xaxis_title='Age',
        yaxis_title='Annual Withdrawal ($)',
        height=500,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
        ),
    )

    fig.update_yaxes(tickformat='$,.0f')

    return _apply_dark_theme(fig)


def income_stack_chart(results: SimulationResults, config: SimulationConfig) -> go.Figure:
    """
    Create stacked area chart showing all household income sources over time.

    Uses median inflation path for inflation-adjustment.  Zero-value series
    are suppressed to keep the legend clean.  Includes portfolio withdrawal,
    Social Security, pension, SERP, rental, and part-time income.

    Args:
        results: SimulationResults from completed simulation.
        config: Original SimulationConfig.

    Returns:
        Plotly Figure with stacked area income chart (dark theme).
    """
    ages = results.ages
    n = len(ages)

    # Median inflation path for real->nominal scaling
    median_inf = np.percentile(results.inflation_rates, 50, axis=0)
    cum_inf = np.ones(n)
    for i in range(1, n):
        cum_inf[i] = cum_inf[i - 1] * (1.0 + median_inf[i])

    # Median simulated arrays (already nominal in simulation)
    median_withdrawals = np.percentile(results.withdrawals, 50, axis=0)
    median_ss          = np.percentile(results.ss_income,   50, axis=0)

    # Deterministic income sources (derived from config + median inflation)

    def _window(arr_real, start_age, end_age):
        """Return nominal series: active when start_age <= age < end_age."""
        out = np.zeros(n)
        for i, age in enumerate(ages):
            if start_age <= age < end_age:
                out[i] = arr_real * cum_inf[i]
        return out

    # Pension
    pension = _window(
        config.pension_annual_real,
        config.pension_start_age,
        getattr(config, 'pension_end_age', 95) + 1,
    )

    # SERP (IRC 409A -- per-year nominal schedule 2026-2033)
    _start_year = getattr(config, 'simulation_start_year', 2026)
    serp = np.zeros(n)
    for i, age in enumerate(ages):
        _yr = _start_year + (age - config.current_age)
        serp[i] = getattr(config, f'serp_{_yr}', 0.0)

    # Rental income
    rental = _window(
        config.rental_income_annual_real,
        getattr(config, 'rental_start_age', 0),
        getattr(config, 'rental_end_age', 200),
    )

    # Part-time
    part_time = _window(
        config.part_time_income_annual,
        getattr(config, 'part_time_income_start_age', 0),
        config.part_time_income_end_age,
    )

    # Colour palette (one per series)
    all_sources = [
        ('Portfolio Draw',         median_withdrawals,  'rgba(100, 180, 255, 0.85)'),
        ('Social Security',        median_ss,           'rgba(100, 255, 130, 0.85)'),
        ('Pension',                pension,             'rgba(255, 160, 80,  0.85)'),
        ('SERP',                   serp,                'rgba(255, 100, 160, 0.85)'),
        ('Rental Income',          rental,              'rgba(180, 100, 255, 0.85)'),
        ('Part-time',              part_time,           'rgba(255, 240, 80,  0.85)'),
    ]

    fig = go.Figure()

    for name, data, color in all_sources:
        if np.max(data) < 1.0:
            continue  # Suppress zero-value series
        fig.add_trace(go.Scatter(
            x=ages,
            y=data,
            name=name,
            mode='lines',
            line=dict(width=0.5),
            fillcolor=color,
            stackgroup='one',
            hovertemplate=(
                '<b>%{fullData.name}</b><br>'
                'Age: %{x}<br>'
                'Income: %{customdata}<extra></extra>'
            ),
            customdata=[format_currency(v, decimals=0) for v in data],
        ))

    # Retirement age marker
    if min(ages) <= config.retirement_age <= max(ages):
        fig.add_vline(
            x=config.retirement_age,
            line_dash='dash',
            line_color='rgba(255,255,255,0.4)',
            annotation_text='Retirement',
            annotation_position='top right',
            annotation_font_size=9,
            annotation_font_color='rgba(255,255,255,0.7)',
        )

    fig.update_layout(
        title='Household Income by Source — Median Path (Nominal Dollars)',
        xaxis_title='Age',
        yaxis_title='Annual Income ($)',
        height=520,
        hovermode='x unified',
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            font=dict(size=10),
        ),
    )
    fig.update_yaxes(tickformat='$,.0f')

    return _apply_dark_theme(fig)


def tax_burden_chart(results: SimulationResults) -> go.Figure:
    """
    Create chart showing annual tax burden over time (federal + state stacked).

    X-axis: age, Y-axis: annual taxes paid (median path). Includes secondary
    y-axis showing effective tax rate as a line overlay.

    Args:
        results: SimulationResults from completed simulation.

    Returns:
        Plotly Figure with tax burden chart.
    """
    ages = results.ages
    median_taxes = np.percentile(results.taxes, 50, axis=0)

    # Effective tax rate uses gross income (all taxable sources).
    if hasattr(results, 'gross_income') and results.gross_income.size > 0:
        median_gross = np.percentile(results.gross_income, 50, axis=0)
    else:
        median_gross = np.zeros_like(median_taxes)
    effective_tax_rate = np.divide(
        median_taxes,
        median_gross,
        where=(median_gross > 0),
        out=np.zeros_like(median_gross),
    )

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{'secondary_y': True}]])

    # Add tax bar chart
    fig.add_trace(
        go.Bar(
            x=ages, y=median_taxes,
            name='Annual Taxes',
            marker=dict(color='rgba(255, 100, 100, 0.8)'),
            hovertemplate='<b>Age: %{x}</b><br>Taxes: %{customdata}<extra></extra>',
            customdata=[format_currency(v, decimals=0) for v in median_taxes],
        ),
        secondary_y=False,
    )

    # Add effective tax rate line on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=ages, y=effective_tax_rate * 100,
            name='Effective Tax Rate',
            mode='lines+markers',
            line=dict(color='white', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Age: %{x}</b><br>Effective Rate: %{customdata}<extra></extra>',
            customdata=[format_percent(v / 100, decimals=1) for v in (effective_tax_rate * 100)],
        ),
        secondary_y=True,
    )

    # Update axes
    fig.update_yaxes(title_text='Annual Taxes ($)', tickformat='$,.0f', secondary_y=False)
    fig.update_yaxes(title_text='Effective Tax Rate (%)', tickformat='.1f%', secondary_y=True)

    fig.update_layout(
        title='Tax Burden Over Time — Median Path',
        xaxis_title='Age',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
        ),
    )

    return _apply_dark_theme(fig)


def asset_allocation_chart(config: SimulationConfig) -> go.Figure:
    """
    Create pie charts showing asset allocation before and after retirement.

    Uses make_subplots to display two side-by-side pie charts: one for
    pre-retirement allocation and one for retirement allocation.

    Args:
        config: SimulationConfig with allocation data.

    Returns:
        Plotly Figure with two pie charts.
    """
    # Prepare data
    asset_names = list(config.pre_retirement_allocation.keys())
    pre_ret_values = list(config.pre_retirement_allocation.values())
    ret_values = list(config.retirement_allocation.values())

    # Create subplots (1 row, 2 columns of pie charts)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'pie'}, {'type': 'pie'}]],
        subplot_titles=('Pre-Retirement Allocation', 'Retirement Allocation'),
    )

    # Pre-retirement pie
    fig.add_trace(
        go.Pie(
            labels=asset_names,
            values=pre_ret_values,
            name='Pre-Retirement',
            hovertemplate='<b>%{label}</b><br>%{value:.1%}<extra></extra>',
            marker=dict(line=dict(color=DARK_BGCOLOR, width=2)),
        ),
        row=1, col=1,
    )

    # Retirement pie
    fig.add_trace(
        go.Pie(
            labels=asset_names,
            values=ret_values,
            name='Retirement',
            hovertemplate='<b>%{label}</b><br>%{value:.1%}<extra></extra>',
            marker=dict(line=dict(color=DARK_BGCOLOR, width=2)),
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title='Asset Allocation Strategy',
        height=500,
        font=dict(size=10),
    )

    return _apply_dark_theme(fig)


def inflation_chart(results: SimulationResults) -> go.Figure:
    """
    Create chart showing inflation path with percentile bands.

    Shows median inflation with 10th/90th percentile band shaded region.
    X-axis: age, Y-axis: inflation rate (%).

    Args:
        results: SimulationResults from completed simulation.

    Returns:
        Plotly Figure with inflation chart.
    """
    ages = results.ages
    inflation_data = results.inflation_rates * 100  # Convert to percentage

    # Calculate percentiles
    percentiles = [10, 50, 90]
    perc_dict = calculate_percentiles(inflation_data, percentiles)

    fig = go.Figure()

    # Add shaded region: 10th to 90th percentile
    fig.add_trace(go.Scatter(
        x=ages, y=perc_dict[90],
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=perc_dict[10],
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(255, 150, 100, 0.2)',
        name='10th-90th percentile',
        hoverinfo='skip',
    ))

    # Add median line
    fig.add_trace(go.Scatter(
        x=ages, y=perc_dict[50],
        mode='lines',
        line=dict(color='white', width=2),
        name='Median Inflation Rate',
        hovertemplate='<b>Age: %{x}</b><br>Inflation: %{customdata}<extra></extra>',
        customdata=[format_percent(v / 100, decimals=2) for v in perc_dict[50]],
    ))

    fig.update_layout(
        title='Inflation Rate Projections',
        xaxis_title='Age',
        yaxis_title='Inflation Rate (%)',
        height=500,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
        ),
    )

    fig.update_yaxes(tickformat='.1f%')

    return _apply_dark_theme(fig)


def portfolio_depletion_histogram(results: SimulationResults, config: SimulationConfig) -> go.Figure:
    """
    Create histogram showing age of portfolio depletion for failed simulations.

    Shows distribution of ages at which portfolio reached zero (for unsuccessful
    simulations). X-axis: depletion age, Y-axis: count of simulations.

    Args:
        results: SimulationResults from completed simulation.
        config: Original SimulationConfig.

    Returns:
        Plotly Figure with depletion histogram.
    """
    # Find depletion ages for failed simulations
    failed_mask = ~results.success_mask
    n_failed = np.sum(failed_mask)

    if n_failed == 0:
        # No failures - show message
        fig = go.Figure()
        fig.add_annotation(
            text='All simulations succeeded! Portfolio never depleted.',
            xref='paper', yref='paper',
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color=FONT_COLOR),
        )
        fig.update_layout(height=400, title='Portfolio Depletion Age Distribution')
        return _apply_dark_theme(fig)

    # Calculate depletion age for each failed simulation
    depletion_ages = []
    for sim_idx in range(results.portfolio_values.shape[0]):
        if failed_mask[sim_idx]:
            portfolio_path = results.portfolio_values[sim_idx, :]
            depletion_year = np.where(portfolio_path <= 0)[0]
            if len(depletion_year) > 0:
                depletion_age = results.ages[depletion_year[0]]
                depletion_ages.append(depletion_age)

    if len(depletion_ages) == 0:
        depletion_ages = [config.life_expectancy]

    # Create histogram
    fig = go.Figure(data=[
        go.Histogram(
            x=depletion_ages,
            nbinsx=15,
            marker_color='rgba(255, 100, 100, 0.8)',
            name='Failed Simulations',
            hovertemplate='<b>Depletion Age: %{x}</b><br>Count: %{y}<extra></extra>',
        )
    ])

    fig.update_layout(
        title=f'Portfolio Depletion Age Distribution ({n_failed} failed scenarios)',
        xaxis_title='Age at Portfolio Depletion',
        yaxis_title='Number of Simulations',
        height=400,
        showlegend=False,
    )

    fig.update_xaxes(tickformat='d')

    return _apply_dark_theme(fig)


def withdrawal_schedule_chart(results: SimulationResults, config: SimulationConfig) -> go.Figure:
    """
    Create chart showing Annual Household Withdrawal by year.

    Uses median path across all simulations.

    Args:
        results: SimulationResults with withdrawal data.
        config: Original SimulationConfig.

    Returns:
        Plotly Figure with withdrawal schedule chart.
    """
    ages = results.ages

    # Median total withdrawal path
    median_withdrawal = np.percentile(results.withdrawals, 50, axis=0)

    fig = go.Figure()

    # Total withdrawal area
    if np.max(median_withdrawal) > 0:
        fig.add_trace(go.Scatter(
            x=ages,
            y=median_withdrawal,
            name='Household Withdrawal',
            mode='lines',
            line=dict(width=0.5),
            fillcolor='rgba(100, 180, 255, 0.75)',
            fill='tozeroy',
            hovertemplate='<b>Withdrawal</b><br>Age: %{x}<br>Amount: %{customdata}<extra></extra>',
            customdata=[fmt_k(v) for v in median_withdrawal],
        ))

    # Retirement age vline
    if min(ages) <= config.retirement_age <= max(ages):
        fig.add_vline(
            x=config.retirement_age,
            line_dash='dash',
            line_color='rgba(255,255,255,0.4)',
            annotation_text='Retirement',
            annotation_position='top right',
            annotation_font_size=9,
            annotation_font_color='rgba(255,255,255,0.7)',
        )

    fig.update_layout(
        title='Annual Household Withdrawal — Median Path',
        xaxis_title='Age',
        yaxis_title='Annual Withdrawal ($)',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
        ),
    )
    fig.update_yaxes(tickformat='$,.0f')

    return _apply_dark_theme(fig)


# ======================================================================
# v1.8 Feature 4: New report charts
# ======================================================================

def portfolio_success_curve_chart(
    results: SimulationResults,
    config: SimulationConfig = None,
) -> go.Figure:
    """
    Year-by-year portfolio survival probability curve.

    For each year t, computes the fraction of simulation paths where the
    portfolio balance is still > 0. Displayed as a filled area chart.

    Args:
        results: SimulationResults from completed simulation.
        config: Optional SimulationConfig.

    Returns:
        Plotly Figure with filled-area success probability curve.

    Source: Bengen (1994) -- survival probability framework.
    """
    n_sims, n_years = results.portfolio_values.shape

    # Compute survival % at each year
    survival_pct = np.zeros(n_years)
    for t in range(n_years):
        alive = np.sum(results.portfolio_values[:, t] > 0)
        survival_pct[t] = (alive / n_sims) * 100.0

    ages = results.ages if hasattr(results, 'ages') and results.ages is not None else np.arange(config.current_age, config.current_age + n_years)

    fig = go.Figure()

    # Filled area under the survival curve
    fig.add_trace(go.Scatter(
        x=ages,
        y=survival_pct,
        mode='lines',
        line=dict(color='rgba(100, 150, 230, 0.9)', width=2),
        fill='tozeroy',
        fillcolor='rgba(100, 150, 230, 0.35)',
        name='Success',
        hovertemplate='<b>Age %{x}</b><br>Success: %{y:.2f}%<extra></extra>',
    ))

    fig.update_layout(
        title='Portfolio Success',
        xaxis_title='Age',
        yaxis_title='Success',
        height=350,
        yaxis=dict(
            range=[0, 105],
            tickformat='.1f',
            ticksuffix='%',
        ),
        showlegend=False,
    )

    return _apply_dark_theme(fig)


def portfolio_balance_lines_chart(
    results: SimulationResults,
    config: SimulationConfig,
    log_scale: bool = False,
    inflation_adjusted: bool = False,
) -> go.Figure:
    """
    Simulated portfolio balances as discrete percentile lines.

    Displays 5 colored lines (10th, 25th, 50th, 75th, 90th percentile)
    with interactive tooltips showing dollar values at each year.
    Supports optional logarithmic Y-axis and inflation adjustment.

    Args:
        results: SimulationResults from completed simulation.
        config: Original SimulationConfig.
        log_scale: If True, use logarithmic Y-axis.
        inflation_adjusted: If True, deflate portfolio values by cumulative
            inflation to show real (today's dollar) values.

    Returns:
        Plotly Figure with 5 percentile lines and interactive tooltips.
    """
    portfolio_data = results.portfolio_values.copy()  # (n_sims, n_years)
    n_sims, n_years = portfolio_data.shape
    years = np.arange(0, n_years)

    # Inflation adjustment: deflate by cumulative inflation
    if inflation_adjusted:
        cum_infl = np.cumprod(1.0 + results.inflation_rates, axis=1)
        portfolio_data = portfolio_data / np.maximum(cum_infl, 1e-10)

    # Compute percentiles
    percentiles = [10, 25, 50, 75, 90]
    perc_data = {}
    for p in percentiles:
        perc_data[p] = np.percentile(portfolio_data, p, axis=0)

    # Line colors -- distinct, readable on dark background
    colors = {
        10: '#E63946',    # Red (worst case)
        25: '#F4A261',    # Orange
        50: '#E9C46A',    # Yellow/gold (median)
        75: '#2A9D8F',    # Teal
        90: '#457B9D',    # Blue (best case)
    }

    fig = go.Figure()

    for p in percentiles:
        fig.add_trace(go.Scatter(
            x=years,
            y=perc_data[p],
            mode='lines+markers',
            name=f'{p}th Percentile',
            line=dict(color=colors[p], width=2),
            marker=dict(size=4, color=colors[p]),
            hovertemplate=(
                f'<b>{p}th Percentile</b><br>'
                'Year: %{x}<br>'
                'Balance: $%{y:,.0f}<extra></extra>'
            ),
        ))

    _title = 'Simulated Portfolio Balances'
    if inflation_adjusted:
        _title += ' (Inflation Adjusted)'

    fig.update_layout(
        title=_title,
        xaxis_title='Year',
        yaxis_title='Portfolio Balance ($)',
        height=450,
        yaxis=dict(
            tickformat='$,.0f',
            type='log' if log_scale else 'linear',
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5,
        ),
    )

    return _apply_dark_theme(fig)
