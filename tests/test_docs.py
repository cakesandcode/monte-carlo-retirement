"""
Phase 4 Tests — Documentation Validation

Confirms all required documentation files exist, are non-empty,
and contain expected sections.
"""
import os
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDocFilesExist:
    """All required documentation files must exist and be non-empty."""

    @pytest.mark.parametrize("filepath", [
        "README.md",
        "ARCHITECTURE.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "docs/PREMIUM_FEATURES.md",
    ])
    def test_doc_file_exists(self, filepath):
        path = os.path.join(REPO_ROOT, filepath)
        assert os.path.isfile(path), f"Documentation file missing: {filepath}"

    @pytest.mark.parametrize("filepath", [
        "README.md",
        "ARCHITECTURE.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "docs/PREMIUM_FEATURES.md",
    ])
    def test_doc_file_not_empty(self, filepath):
        path = os.path.join(REPO_ROOT, filepath)
        if os.path.isfile(path):
            assert os.path.getsize(path) > 100, (
                f"Documentation file is suspiciously small: {filepath}"
            )


class TestReadmeSections:
    """README.md should contain key sections for a good landing page."""

    @pytest.fixture
    def readme_content(self):
        path = os.path.join(REPO_ROOT, "README.md")
        assert os.path.isfile(path), "README.md not found"
        with open(path) as f:
            return f.read()

    @pytest.mark.parametrize("section", [
        "Quick Start",
        "Features",
        "Configuration",
        "Running Tests",
        "Architecture",
        "Premium Features",
        "License",
    ])
    def test_readme_has_section(self, readme_content, section):
        assert section.lower() in readme_content.lower(), (
            f"README.md missing expected section: '{section}'"
        )

    def test_readme_has_mermaid_diagram(self, readme_content):
        assert "```mermaid" in readme_content, (
            "README.md should contain a Mermaid diagram"
        )


class TestArchitectureSections:
    """ARCHITECTURE.md should contain key architecture documentation."""

    @pytest.fixture
    def architecture_content(self):
        path = os.path.join(REPO_ROOT, "ARCHITECTURE.md")
        assert os.path.isfile(path), "ARCHITECTURE.md not found"
        with open(path) as f:
            return f.read()

    @pytest.mark.parametrize("section", [
        "Simulation Flow",
        "Module Responsibilities",
        "Data Flow",
        "Dependency Graph",
        "Key Data Structures",
        "Extension Points",
    ])
    def test_architecture_has_section(self, architecture_content, section):
        assert section.lower() in architecture_content.lower(), (
            f"ARCHITECTURE.md missing expected section: '{section}'"
        )

    def test_architecture_has_mermaid_diagram(self, architecture_content):
        assert "```mermaid" in architecture_content, (
            "ARCHITECTURE.md should contain a Mermaid diagram"
        )

    def test_architecture_documents_all_modules(self, architecture_content):
        """Every core module should be mentioned."""
        modules = [
            "config/defaults.py",
            "models/simulation.py",
            "models/portfolio.py",
            "models/tax.py",
            "models/social_security.py",
            "models/asset_returns.py",
            "models/inflation.py",
            "utils/charts.py",
            "utils/config_io.py",
            "utils/report_generator.py",
            "utils/helpers.py",
            "data/loader.py",
        ]
        for module in modules:
            # Check for the filename portion at minimum
            filename = module.split("/")[-1]
            assert filename in architecture_content, (
                f"ARCHITECTURE.md should document module: {module}"
            )


class TestPremiumFeaturesDoc:
    """docs/PREMIUM_FEATURES.md should describe all premium feature categories."""

    @pytest.fixture
    def premium_content(self):
        path = os.path.join(REPO_ROOT, "docs", "PREMIUM_FEATURES.md")
        assert os.path.isfile(path), "docs/PREMIUM_FEATURES.md not found"
        with open(path) as f:
            return f.read()

    @pytest.mark.parametrize("feature_area", [
        "Spouse",
        "Stress Test",
        "Roth",
        "Tax Optimization",
        "Spending",
        "Reporting",
    ])
    def test_premium_doc_covers_feature(self, premium_content, feature_area):
        assert feature_area.lower() in premium_content.lower(), (
            f"PREMIUM_FEATURES.md should mention: '{feature_area}'"
        )


class TestInternalLinks:
    """Documentation internal links should point to files that exist."""

    @pytest.fixture
    def readme_content(self):
        with open(os.path.join(REPO_ROOT, "README.md")) as f:
            return f.read()

    @pytest.mark.parametrize("linked_file", [
        "ARCHITECTURE.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "docs/PREMIUM_FEATURES.md",
    ])
    def test_readme_linked_file_exists(self, readme_content, linked_file):
        # Only check if the link is actually in README
        if linked_file in readme_content:
            path = os.path.join(REPO_ROOT, linked_file)
            assert os.path.isfile(path), (
                f"README.md links to {linked_file} but file does not exist"
            )
