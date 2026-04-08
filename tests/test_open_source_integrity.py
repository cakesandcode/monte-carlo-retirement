"""
Phase 3 Tests — Open Source Integrity Regression Tests

Permanent guard rails for the open-source repo. Ensures no premium
fields leak back in, sample config works, simulation produces valid
results, and reports contain no premium sections.
"""
import os
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Premium field prefixes that must NOT exist in the open-source SimulationConfig
PREMIUM_FIELD_PREFIXES = [
    "fragility_",
    "scenario_",
    "spend_phase",
    "spend_pre_year_",
    "roth_conversion_",
    "contribution_traditional_pct",
    "income_offset_enabled",
]

# Premium section headings that must NOT appear in generated reports
PREMIUM_REPORT_SECTIONS = [
    "Per-Person",
    "Roth Conversion Ladder",
    "Tax Treatment Breakdown",
    "Fragility",
    "Scenario Overlay",
    "Spend Phase",
]


class TestSimulationConfigIntegrity:
    """SimulationConfig must be free of all premium fields."""

    def test_no_premium_fields(self):
        from config.defaults import SimulationConfig
        field_names = list(SimulationConfig.__dataclass_fields__.keys())
        for prefix in PREMIUM_FIELD_PREFIXES:
            matches = [f for f in field_names if f.startswith(prefix)]
            assert matches == [], (
                f"Premium field prefix '{prefix}' found: {matches}"
            )

    def test_no_premium_fields_exact(self):
        """Check specific premium field names that don't follow prefix patterns."""
        from config.defaults import SimulationConfig
        field_names = set(SimulationConfig.__dataclass_fields__.keys())
        exact_premium_fields = {
            "contribution_traditional_pct",
            "earned_income_is_gross",
            "income_offset_enabled",
        }
        leaked = field_names & exact_premium_fields
        assert leaked == set(), f"Premium fields found: {leaked}"


class TestSimulationResultsIntegrity:
    """SimulationResults must not contain per-person tracking arrays."""

    def test_no_per_person_arrays(self):
        from config.defaults import SimulationResults
        field_names = list(SimulationResults.__dataclass_fields__.keys())
        per_person = [f for f in field_names if "primary_" in f or "spouse_" in f]
        assert per_person == [], (
            f"Per-person fields found in SimulationResults: {per_person}"
        )

    def test_no_roth_conversion_tracking(self):
        from config.defaults import SimulationResults
        field_names = list(SimulationResults.__dataclass_fields__.keys())
        roth = [f for f in field_names if "roth_conversion" in f]
        assert roth == [], (
            f"Roth conversion fields found in SimulationResults: {roth}"
        )


class TestSampleConfig:
    """sample_config.csv should load and produce a valid SimulationConfig."""

    @pytest.fixture
    def sample_config_path(self):
        path = os.path.join(REPO_ROOT, "input", "sample_config.csv")
        if not os.path.isfile(path):
            pytest.skip("input/sample_config.csv not yet created")
        return path

    def test_sample_config_loads(self, sample_config_path):
        from utils.config_io import load_config_csv
        with open(sample_config_path) as f:
            config = load_config_csv(f)
        assert config is not None

    def test_sample_config_runs(self, sample_config_path):
        from utils.config_io import load_config_csv
        from models.simulation import MonteCarloSimulator
        with open(sample_config_path) as f:
            config = load_config_csv(f)
        config.n_simulations = 5
        sim = MonteCarloSimulator(config)
        results = sim.run()
        assert 0.0 <= results.success_rate <= 1.0

    def test_sample_config_csv_round_trip(self, sample_config_path):
        """Load -> export -> reload should produce identical config."""
        import io
        from utils.config_io import load_config_csv, config_to_csv_bytes
        with open(sample_config_path) as f:
            config1 = load_config_csv(f)
        csv_bytes = config_to_csv_bytes(config1)
        config2 = load_config_csv(io.BytesIO(csv_bytes))
        # Compare key fields
        assert config1.current_age == config2.current_age
        assert config1.retirement_age == config2.retirement_age
        assert config1.spend_annual_real == config2.spend_annual_real
        assert config1.n_simulations == config2.n_simulations


class TestEndToEndSimulation:
    """A full simulation with default config should succeed."""

    @pytest.fixture
    def results(self):
        from config.defaults import SimulationConfig
        from models.simulation import MonteCarloSimulator
        config = SimulationConfig()
        config.n_simulations = 10
        return MonteCarloSimulator(config).run()

    def test_results_have_expected_arrays(self, results):
        assert hasattr(results, "portfolio_values")
        assert hasattr(results, "success_rate")

    def test_results_success_rate_valid(self, results):
        assert 0.0 <= results.success_rate <= 1.0

    def test_portfolio_values_correct_sim_count(self, results):
        assert results.portfolio_values.shape[0] == 10


class TestReportIntegrity:
    """Generated reports must not contain premium section headings."""

    @pytest.fixture
    def report_docx_path(self, tmp_path):
        """Generate a report to a temp directory."""
        try:
            from config.defaults import SimulationConfig
            from models.simulation import MonteCarloSimulator
            from utils.report_generator import ReportGenerator
        except ImportError:
            pytest.skip("Report generator dependencies not available")

        config = SimulationConfig()
        config.n_simulations = 5
        results = MonteCarloSimulator(config).run()

        gen = ReportGenerator(results, output_dir=str(tmp_path))
        output = gen.generate(filename_base="test_integrity")
        docx_path = output.get("docx")
        if not docx_path or not os.path.isfile(docx_path):
            pytest.skip("DOCX generation not available")
        return docx_path

    def test_report_is_nonempty(self, report_docx_path):
        assert os.path.getsize(report_docx_path) > 0

    def test_report_no_premium_sections(self, report_docx_path):
        """Read DOCX paragraphs and check for premium headings."""
        try:
            from docx import Document
        except ImportError:
            pytest.skip("python-docx not installed")

        doc = Document(report_docx_path)
        headings = [
            p.text for p in doc.paragraphs
            if p.style and "Heading" in (p.style.name or "")
        ]
        for premium_section in PREMIUM_REPORT_SECTIONS:
            matches = [h for h in headings if premium_section.lower() in h.lower()]
            assert matches == [], (
                f"Premium section '{premium_section}' found in report headings: {matches}"
            )
