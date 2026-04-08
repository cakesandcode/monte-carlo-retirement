"""
Tests for open-source release readiness.

Validates that all release artifacts (pyproject.toml, CI, templates,
LICENSE, README) are correctly configured.
"""
import os
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestLicenseAndCopyright:
    """Verify LICENSE has proper attribution."""

    def test_license_has_copyright_holder(self):
        with open(os.path.join(REPO_ROOT, "LICENSE")) as f:
            content = f.read()
        assert "Copyright (c)" in content
        # Should not be blank after "Copyright (c) YEAR"
        assert "cakesandcode" in content, "LICENSE must include copyright holder name"


class TestReadme:
    """Verify README has correct URLs."""

    @pytest.fixture
    def readme(self):
        with open(os.path.join(REPO_ROOT, "README.md")) as f:
            return f.read()

    def test_no_placeholder_username(self, readme):
        assert "YOUR_USERNAME" not in readme, "README still has placeholder YOUR_USERNAME"

    def test_clone_url_correct(self, readme):
        assert "cakesandcode/monte-carlo-retirement" in readme


class TestPyprojectToml:
    """Verify pyproject.toml exists and is well-formed."""

    @pytest.fixture
    def pyproject(self):
        path = os.path.join(REPO_ROOT, "pyproject.toml")
        assert os.path.isfile(path), "pyproject.toml is missing"
        with open(path) as f:
            return f.read()

    def test_has_build_system(self, pyproject):
        assert "[build-system]" in pyproject

    def test_has_project_metadata(self, pyproject):
        assert "[project]" in pyproject
        assert 'name = "monte-carlo-retirement"' in pyproject

    def test_has_version(self, pyproject):
        assert 'version = "1.0.0"' in pyproject

    def test_requires_python_310(self, pyproject):
        assert '>=3.10' in pyproject

    def test_has_dependencies(self, pyproject):
        assert "dependencies" in pyproject
        for pkg in ["numpy", "pandas", "streamlit", "plotly"]:
            assert pkg in pyproject, f"Missing dependency: {pkg}"

    def test_has_dev_dependencies(self, pyproject):
        assert "pytest" in pyproject

    def test_has_project_urls(self, pyproject):
        assert "cakesandcode/monte-carlo-retirement" in pyproject


class TestGitHubActionsCI:
    """Verify CI workflow exists and is configured."""

    @pytest.fixture
    def workflow(self):
        path = os.path.join(REPO_ROOT, ".github", "workflows", "test.yml")
        assert os.path.isfile(path), "CI workflow .github/workflows/test.yml is missing"
        with open(path) as f:
            return f.read()

    def test_runs_on_push_and_pr(self, workflow):
        assert "push:" in workflow
        assert "pull_request:" in workflow

    def test_targets_main_branch(self, workflow):
        assert "main" in workflow

    def test_runs_pytest(self, workflow):
        assert "pytest" in workflow

    def test_multi_python_version(self, workflow):
        assert "3.10" in workflow
        assert "3.12" in workflow


class TestGitHubTemplates:
    """Verify issue and PR templates exist."""

    def test_bug_report_template(self):
        path = os.path.join(REPO_ROOT, ".github", "ISSUE_TEMPLATE", "bug_report.md")
        assert os.path.isfile(path), "Bug report template is missing"

    def test_feature_request_template(self):
        path = os.path.join(REPO_ROOT, ".github", "ISSUE_TEMPLATE", "feature_request.md")
        assert os.path.isfile(path), "Feature request template is missing"

    def test_pr_template(self):
        path = os.path.join(REPO_ROOT, ".github", "pull_request_template.md")
        assert os.path.isfile(path), "PR template is missing"


class TestLegalDisclaimer:
    """Verify legal disclaimers are present."""

    def test_readme_has_disclaimer_section(self):
        with open(os.path.join(REPO_ROOT, "README.md")) as f:
            content = f.read()
        assert "## Disclaimer" in content, "README missing Disclaimer section"

    def test_readme_disclaimer_has_key_phrases(self):
        with open(os.path.join(REPO_ROOT, "README.md")) as f:
            content = f.read()
        for phrase in [
            "educational and informational purposes only",
            "not financial advice",
            "Past performance does not guarantee future results",
            "consult a qualified financial advisor",
        ]:
            assert phrase.lower() in content.lower(), f"README disclaimer missing: {phrase}"

    def test_app_has_disclaimer_function(self):
        with open(os.path.join(REPO_ROOT, "app.py")) as f:
            content = f.read()
        assert "_render_disclaimer" in content, "app.py missing _render_disclaimer function"
        assert "educational and informational purposes only" in content, (
            "app.py disclaimer missing key phrase"
        )
