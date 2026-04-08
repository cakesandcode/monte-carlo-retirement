# Contributing

Thank you for your interest in contributing to the Monte Carlo Retirement Simulator.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<YOUR_FORK>/monte-carlo-retirement.git`
3. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a feature branch: `git checkout -b feat/your-feature`

## Development Workflow

1. Make your changes on a feature branch — never commit directly to `main`
2. Run the test suite before submitting: `pytest tests/ -v`
3. Ensure the Streamlit app still loads: `streamlit run app.py`
4. Submit a pull request against `main`

## Code Guidelines

- All simulation parameters belong in `SimulationConfig` (`config/defaults.py`)
- Modules must not have circular imports — maintain the DAG structure (see ARCHITECTURE.md)
- New chart functions go in `utils/charts.py` and should apply `_apply_dark_theme()`
- New income sources should be added through `SocialSecurityModel.get_income_overlays()`
- Tax constants (brackets, thresholds) live in `config/defaults.py`

## Testing

- Add tests for new functionality in the `tests/` directory
- Use descriptive test names that explain the scenario being tested
- Run `pytest tests/ -v` to confirm all tests pass before submitting

## Reporting Issues

Open a GitHub issue with:
- A clear description of the bug or feature request
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Python version and OS

## Pull Request Process

1. Update tests if you changed module behavior
2. Ensure `pytest tests/ -v` passes with zero failures
3. Keep PRs focused — one feature or fix per PR
4. Write a clear PR description explaining the "why"
