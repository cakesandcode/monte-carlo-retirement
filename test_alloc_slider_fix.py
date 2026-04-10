"""
Tests for allocation slider normalization fix.

Validates:
  1. _ALLOC_SLIDER_SPEC constant is defined correctly in app.py
  2. render_allocation_sliders and render_allocation_status helpers exist
  3. Eager normalize_allocation calls are removed from the allocation section
  4. Pre-simulation guard blocks when allocation deviates >5%
  5. _ALLOC_TOLERANCE in portfolio.py is 0.05
  6. normalize_allocation still works correctly (no changes to helpers.py)

Version: 1.0.0
"""

import pytest
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from utils.helpers import normalize_allocation, validate_allocation
from models.portfolio import _ALLOC_TOLERANCE

APP_PY = os.path.join(os.path.dirname(__file__), 'app.py')


class TestAllocSliderSpec:
    """Verify _ALLOC_SLIDER_SPEC constant in app.py."""

    def test_spec_defined(self):
        with open(APP_PY) as f:
            source = f.read()
        assert '_ALLOC_SLIDER_SPEC' in source, "_ALLOC_SLIDER_SPEC not found in app.py"

    def test_spec_has_8_entries(self):
        with open(APP_PY) as f:
            source = f.read()
        # Count the field name entries in the spec
        fields = ['us_large_cap', 'us_small_cap', 'international_dev',
                   'emerging_markets', 'us_bonds', 'tips', 'cash', 'reits']
        spec_section = source[source.find('_ALLOC_SLIDER_SPEC'):]
        spec_section = spec_section[:spec_section.find(']') + 1]
        for field in fields:
            assert field in spec_section, f"Field '{field}' missing from _ALLOC_SLIDER_SPEC"


class TestHelperFunctions:
    """Verify helper functions exist in app.py."""

    def test_render_allocation_sliders_defined(self):
        with open(APP_PY) as f:
            source = f.read()
        assert 'def render_allocation_sliders(' in source

    def test_render_allocation_status_defined(self):
        with open(APP_PY) as f:
            source = f.read()
        assert 'def render_allocation_status(' in source

    def test_render_allocation_sliders_does_not_normalize(self):
        """The slider renderer must NOT call normalize_allocation."""
        with open(APP_PY) as f:
            source = f.read()
        # Extract the function body
        start = source.find('def render_allocation_sliders(')
        end = source.find('\ndef ', start + 1)
        func_body = source[start:end]
        assert 'normalize_allocation' not in func_body, (
            "render_allocation_sliders should NOT call normalize_allocation"
        )


class TestEagerNormalizationRemoved:
    """Verify that eager normalize_allocation calls are removed from Section 4."""

    def test_no_eager_normalize_in_allocation_section(self):
        """The allocation section must not eagerly mutate config with normalize_allocation."""
        with open(APP_PY) as f:
            source = f.read()
        # Find allocation section (Section 4 to Section 5)
        sec4_start = source.find('Section 4')
        sec5_start = source.find('Section 5')
        if sec4_start < 0 or sec5_start < 0:
            pytest.skip("Could not locate Section 4 / Section 5 markers")
        alloc_section = source[sec4_start:sec5_start]
        # These patterns indicate the old eager normalization
        assert 'config.pre_retirement_allocation = normalize_allocation(' not in alloc_section, (
            "Eager normalize_allocation for pre_retirement_allocation still present"
        )
        assert 'config.retirement_allocation = normalize_allocation(' not in alloc_section, (
            "Eager normalize_allocation for retirement_allocation still present"
        )

    def test_uses_render_helpers_in_allocation_section(self):
        """Section 4 should use the new helper functions."""
        with open(APP_PY) as f:
            source = f.read()
        sec4_start = source.find('Section 4')
        sec5_start = source.find('Section 5')
        alloc_section = source[sec4_start:sec5_start]
        assert 'render_allocation_sliders(' in alloc_section
        assert 'render_allocation_status(' in alloc_section


class TestPreSimulationGuard:
    """Verify pre-simulation guard in _run_simulation."""

    def test_guard_exists(self):
        with open(APP_PY) as f:
            source = f.read()
        run_sim_start = source.find('def _run_simulation(')
        assert run_sim_start > 0, "_run_simulation not found"
        run_sim_body = source[run_sim_start:source.find('\ndef ', run_sim_start + 1)]
        assert 'abs(_total - 1.0) > _ALLOC_TOLERANCE' in run_sim_body, (
            "Pre-simulation guard missing tolerance threshold check"
        )

    def test_guard_checks_both_groups(self):
        with open(APP_PY) as f:
            source = f.read()
        run_sim_start = source.find('def _run_simulation(')
        run_sim_body = source[run_sim_start:source.find('\ndef ', run_sim_start + 1)]
        assert 'pre_retirement_allocation' in run_sim_body
        assert 'retirement_allocation' in run_sim_body

    def test_guard_returns_before_simulation(self):
        """Guard should return (not just warn) to prevent simulation."""
        with open(APP_PY) as f:
            source = f.read()
        run_sim_start = source.find('def _run_simulation(')
        run_sim_body = source[run_sim_start:source.find('\ndef ', run_sim_start + 1)]
        # After the error message, there should be a return statement
        error_idx = run_sim_body.find('Adjust sliders or click Auto-fix')
        assert error_idx > 0
        after_error = run_sim_body[error_idx:error_idx + 100]
        assert 'return' in after_error, (
            "Guard should return to prevent simulation from running"
        )


class TestAllocTolerance:
    """Verify _ALLOC_TOLERANCE in portfolio.py is updated."""

    def test_tolerance_is_005(self):
        assert _ALLOC_TOLERANCE == 0.05, (
            f"_ALLOC_TOLERANCE should be 0.05, got {_ALLOC_TOLERANCE}"
        )


class TestNormalizeAllocationUnchanged:
    """Verify normalize_allocation in helpers.py still works correctly."""

    def test_normalize_balanced(self):
        alloc = {'stocks': 0.60, 'bonds': 0.40}
        result = normalize_allocation(alloc)
        assert abs(sum(result.values()) - 1.0) < 1e-10

    def test_normalize_unbalanced(self):
        alloc = {'a': 0.3, 'b': 0.3, 'c': 0.3}
        result = normalize_allocation(alloc)
        assert abs(sum(result.values()) - 1.0) < 1e-10
        # Each should be ~0.3333
        for v in result.values():
            assert abs(v - 1.0/3.0) < 1e-10

    def test_normalize_zero_sum(self):
        alloc = {'a': 0.0, 'b': 0.0}
        result = normalize_allocation(alloc)
        assert result == {'a': 0.0, 'b': 0.0}


class TestNoSessionStateForAllocation:
    """Verify that allocation sliders do not use session_state directly."""

    def test_no_alloc_session_state(self):
        with open(APP_PY) as f:
            source = f.read()
        sec4_start = source.find('Section 4')
        sec5_start = source.find('Section 5')
        alloc_section = source[sec4_start:sec5_start]
        # Should not have session_state references for allocation values
        # (only _csv_load_count is acceptable)
        lines = alloc_section.split('\n')
        for line in lines:
            if 'session_state' in line and '_csv_load_count' not in line:
                assert False, (
                    f"Allocation section should not use session_state "
                    f"(except _csv_load_count): {line.strip()}"
                )


class TestCleanSliderLabels:
    """Verify that allocation sliders use key= parameter and labels have no #N suffix."""

    def test_render_sliders_has_key_param(self):
        with open(APP_PY) as f:
            source = f.read()
        start = source.find('def render_allocation_sliders(')
        end = source.find('\ndef ', start + 1)
        func_body = source[start:end]
        # The function body should contain key= as part of the slider call
        assert 'key=' in func_body, (
            "render_allocation_sliders should pass key= to st.slider"
        )

    def test_render_sliders_labels_no_hash(self):
        with open(APP_PY) as f:
            source = f.read()
        start = source.find('def render_allocation_sliders(')
        end = source.find('\ndef ', start + 1)
        func_body = source[start:end]
        # Label lines should not contain #{load_count} or #N patterns
        label_lines = [line for line in func_body.split('\n') if 'label =' in line]
        for line in label_lines:
            assert '#{' not in line and '#N' not in line, (
                f"Slider label should not contain # suffix: {line.strip()}"
            )
