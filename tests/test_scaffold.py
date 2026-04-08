"""
Phase 1 Tests — Repo Structure Validation

Confirms the new repo has all expected directories, files,
and configuration after initial scaffolding.
"""
import os
import pytest

# Repo root is one level up from tests/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDirectoryStructure:
    """Verify all expected directories exist."""

    @pytest.mark.parametrize("dirname", [
        "config",
        "models",
        "utils",
        "data",
        "docs",
        "tests",
        ".streamlit",
    ])
    def test_directory_exists(self, dirname):
        path = os.path.join(REPO_ROOT, dirname)
        assert os.path.isdir(path), f"Directory {dirname}/ is missing"


class TestGitignore:
    """Verify .gitignore contains required exclusions."""

    @pytest.fixture
    def gitignore_content(self):
        path = os.path.join(REPO_ROOT, ".gitignore")
        assert os.path.isfile(path), ".gitignore is missing"
        with open(path) as f:
            return f.read()

    @pytest.mark.parametrize("pattern", [
        ".venv/",
        "__pycache__/",
        "data/cache/",
        "outputs/",
        "*.pyc",
        ".DS_Store",
        ".pytest_cache/",
    ])
    def test_gitignore_contains_pattern(self, gitignore_content, pattern):
        assert pattern in gitignore_content, (
            f".gitignore missing required pattern: {pattern}"
        )


class TestRequirements:
    """Verify requirements.txt is present and parseable."""

    @pytest.fixture
    def requirements_lines(self):
        path = os.path.join(REPO_ROOT, "requirements.txt")
        assert os.path.isfile(path), "requirements.txt is missing"
        with open(path) as f:
            return [
                line.strip() for line in f
                if line.strip() and not line.startswith("#")
            ]

    def test_requirements_not_empty(self, requirements_lines):
        assert len(requirements_lines) > 0, "requirements.txt is empty"

    @pytest.mark.parametrize("package", [
        "streamlit",
        "plotly",
        "numpy",
        "pandas",
    ])
    def test_core_package_listed(self, requirements_lines, package):
        found = any(package in line.lower() for line in requirements_lines)
        assert found, f"Core package '{package}' not found in requirements.txt"


class TestCopiedFiles:
    """Verify files that should be copied as-is exist and are non-empty."""

    @pytest.mark.parametrize("filepath", [
        "config/__init__.py",
        "config/defaults.py",
        "models/__init__.py",
        "models/inflation.py",
        "utils/__init__.py",
        "data/__init__.py",
        "data/loader.py",
        ".streamlit/config.toml",
    ])
    def test_file_exists(self, filepath):
        path = os.path.join(REPO_ROOT, filepath)
        assert os.path.isfile(path), f"Expected file missing: {filepath}"

    @pytest.mark.parametrize("filepath", [
        "config/defaults.py",
        "models/inflation.py",
        "data/loader.py",
    ])
    def test_file_not_empty(self, filepath):
        path = os.path.join(REPO_ROOT, filepath)
        if os.path.isfile(path):
            assert os.path.getsize(path) > 0, f"File is empty: {filepath}"


class TestLicenseAndDocs:
    """Verify LICENSE file exists."""

    def test_license_exists(self):
        path = os.path.join(REPO_ROOT, "LICENSE")
        assert os.path.isfile(path), "LICENSE file is missing"

    def test_license_is_mit(self):
        path = os.path.join(REPO_ROOT, "LICENSE")
        if os.path.isfile(path):
            with open(path) as f:
                content = f.read()
            assert "MIT License" in content, "LICENSE should be MIT"
