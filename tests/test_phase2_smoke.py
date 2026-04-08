"""
Phase 2 Tests — Simulation Engine Smoke Tests

Confirms that the stripped core modules work end-to-end:
SimulationConfig instantiates cleanly, a short simulation runs,
and no premium field names remain.
"""
import pytest


PREMIUM_FIELDS = [
    "fragility_",
    "scenario_",
    "spend_phase",
    "spend_pre_year_",
    "roth_conversion_",
    "contribution_traditional_pct",
    "earned_income_is_gross",
    "income_offset_enabled",
]


class TestSimulationConfigInstantiation:
    """SimulationConfig should instantiate with clean defaults."""

    def test_default_config_creates(self):
        from config.defaults import SimulationConfig
        config = SimulationConfig()
        assert config is not None

    def test_default_n_simulations(self):
        from config.defaults import SimulationConfig
        config = SimulationConfig()
        assert config.n_simulations > 0

    def test_default_ages_are_generic(self):
        from config.defaults import SimulationConfig
        config = SimulationConfig()
        assert isinstance(config.current_age, int)
        assert isinstance(config.retirement_age, int)
        assert config.retirement_age > config.current_age


class TestNoPremiumFields:
    """SimulationConfig must not contain any premium field names."""

    def test_no_premium_fields_in_config(self):
        from config.defaults import SimulationConfig
        config = SimulationConfig()
        field_names = [f.name for f in config.__dataclass_fields__.values()]
        for premium_prefix in PREMIUM_FIELDS:
            matches = [f for f in field_names if f.startswith(premium_prefix)]
            assert matches == [], (
                f"Premium field(s) found in SimulationConfig: {matches}"
            )


class TestImportChain:
    """All core modules should import without circular dependency errors."""

    def test_import_config(self):
        from config.defaults import SimulationConfig
        assert SimulationConfig is not None

    def test_import_asset_returns(self):
        from models.asset_returns import AssetReturnModel
        assert AssetReturnModel is not None

    def test_import_inflation(self):
        from models.inflation import InflationModel
        assert InflationModel is not None

    def test_import_portfolio(self):
        from models.portfolio import PortfolioMechanics
        assert PortfolioMechanics is not None

    def test_import_tax(self):
        from models.tax import TaxCalculator
        assert TaxCalculator is not None

    def test_import_social_security(self):
        from models.social_security import SocialSecurityModel
        assert SocialSecurityModel is not None

    def test_import_simulator(self):
        from models.simulation import MonteCarloSimulator
        assert MonteCarloSimulator is not None


class TestSimulationSmokeTest:
    """A 10-simulation run should complete and produce valid results."""

    @pytest.fixture
    def results(self):
        from config.defaults import SimulationConfig
        from models.simulation import MonteCarloSimulator
        config = SimulationConfig()
        config.n_simulations = 10
        sim = MonteCarloSimulator(config)
        return sim.run()

    def test_simulation_completes(self, results):
        assert results is not None

    def test_success_rate_in_range(self, results):
        assert 0.0 <= results.success_rate <= 1.0

    def test_portfolio_paths_shape(self, results):
        assert results.portfolio_values.shape[0] == 10
        assert results.portfolio_values.shape[1] > 0

    def test_portfolio_paths_nonnegative(self, results):
        # Portfolio values should never be negative
        assert (results.portfolio_values >= -0.01).all()


class TestNoPremiumReferencesInSource:
    """Grep-style check that premium terms don't appear in core source files."""

    @pytest.fixture
    def source_files(self):
        import os
        import glob
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        patterns = [
            os.path.join(repo_root, "models", "*.py"),
            os.path.join(repo_root, "config", "*.py"),
            os.path.join(repo_root, "utils", "*.py"),
        ]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        # Also include app.py
        app_path = os.path.join(repo_root, "app.py")
        if os.path.isfile(app_path):
            files.append(app_path)
        return files

    @pytest.mark.parametrize("term", [
        "fragility",
        "spend_phase",
        "NIIT",
        "income_offset",
    ])
    def test_no_premium_term_in_sources(self, source_files, term):
        violations = []
        for filepath in source_files:
            with open(filepath) as f:
                for i, line in enumerate(f, 1):
                    # Skip comments and docstrings that might mention
                    # premium features for documentation purposes
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    if term.lower() in line.lower():
                        violations.append(f"{filepath}:{i}")
        assert violations == [], (
            f"Premium term '{term}' found in source files: {violations}"
        )
