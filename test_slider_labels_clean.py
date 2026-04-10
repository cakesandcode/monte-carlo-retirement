"""
Tests for slider label cleanup: no #N suffixes in visible labels.

Source-inspection tests verify that app.py labels are clean and key= params
carry the load_count for Streamlit widget identity.

Exec-based behavioral tests exercise render_allocation_sliders and
render_allocation_status via a MockStreamlit environment.

Version: 1.0.0
"""

import os
import re
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from utils.helpers import normalize_allocation
from models.portfolio import _ALLOC_TOLERANCE

APP_PY = os.path.join(os.path.dirname(__file__), 'app.py')


# ---------------------------------------------------------------------------
# Source-inspection tests
# ---------------------------------------------------------------------------

class TestSourceInspection:
    """Verify labels in app.py source have no #N suffixes."""

    def test_allocation_slider_labels_have_no_hash_suffix(self):
        """render_allocation_sliders label construction must not include #."""
        with open(APP_PY) as f:
            source = f.read()
        start = source.find('def render_allocation_sliders(')
        end = source.find('\ndef ', start + 1)
        func_body = source[start:end]
        label_lines = [l for l in func_body.split('\n') if 'label =' in l]
        assert len(label_lines) > 0, "No label assignment found in render_allocation_sliders"
        for line in label_lines:
            assert '#{' not in line, (
                f"Allocation slider label still contains # suffix: {line.strip()}"
            )

    def test_serp_labels_have_no_hash_suffix(self):
        """SERP number_input labels must not include #N."""
        with open(APP_PY) as f:
            source = f.read()
        serp_start = source.find('SERP / Nonqualified')
        assert serp_start > 0, "SERP section not found in app.py"
        serp_end = source.find('Section 7b', serp_start)
        if serp_end < 0:
            serp_end = serp_start + 2000
        serp_section = source[serp_start:serp_end]
        # Find number_input label strings
        label_matches = re.findall(r'f"SERP\s+\{yr\}[^"]*"', serp_section)
        assert len(label_matches) > 0, "No SERP label patterns found"
        for m in label_matches:
            assert '#{' not in m and '#_lc' not in m, (
                f"SERP label still contains # suffix: {m}"
            )

    def test_allocation_sliders_have_key_param(self):
        """render_allocation_sliders must pass key= to st.slider."""
        with open(APP_PY) as f:
            source = f.read()
        start = source.find('def render_allocation_sliders(')
        end = source.find('\ndef ', start + 1)
        func_body = source[start:end]
        assert 'key=' in func_body, (
            "render_allocation_sliders does not pass key= to st.slider"
        )

    def test_serp_inputs_have_key_param(self):
        """SERP number_inputs must pass key= param."""
        with open(APP_PY) as f:
            source = f.read()
        serp_start = source.find('SERP / Nonqualified')
        serp_end = source.find('Section 7b', serp_start)
        if serp_end < 0:
            serp_end = serp_start + 2000
        serp_section = source[serp_start:serp_end]
        assert 'key=f"serp_{yr}_{_lc}"' in serp_section, (
            "SERP number_input does not pass key= param"
        )


# ---------------------------------------------------------------------------
# MockStreamlit for exec-based behavioral tests
# ---------------------------------------------------------------------------

class MockColumn:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockStreamlit:
    def __init__(self):
        self.slider_calls = []
        self.success_calls = []
        self.warning_calls = []
        self.error_calls = []
        self.button_return = False
        self.rerun_called = False

    def slider(self, label, min_val=0.0, max_val=1.0, value=0.0, step=0.01, key=None):
        self.slider_calls.append({'label': label, 'value': value, 'key': key})
        return value

    def columns(self, n):
        return [MockColumn() for _ in range(n)]

    def success(self, msg):
        self.success_calls.append(msg)

    def warning(self, msg):
        self.warning_calls.append(msg)

    def error(self, msg):
        self.error_calls.append(msg)

    def button(self, label, key=None):
        return self.button_return

    def rerun(self):
        self.rerun_called = True


def _extract_functions():
    """Extract render_allocation_sliders and render_allocation_status via exec."""
    with open(APP_PY) as f:
        source = f.read()

    # Extract _ALLOC_SLIDER_SPEC
    spec_start = source.find('_ALLOC_SLIDER_SPEC = [')
    spec_end = source.find(']', spec_start) + 1
    spec_source = source[spec_start:spec_end]

    # Extract render_allocation_sliders
    sliders_start = source.find('def render_allocation_sliders(')
    sliders_end = source.find('\ndef ', sliders_start + 1)
    sliders_source = source[sliders_start:sliders_end]

    # Extract render_allocation_status
    status_start = source.find('def render_allocation_status(')
    status_end = source.find('\ndef ', status_start + 1)
    status_source = source[status_start:status_end]

    mock_st = MockStreamlit()
    namespace = {
        'st': mock_st,
        'normalize_allocation': normalize_allocation,
        '_ALLOC_TOLERANCE': _ALLOC_TOLERANCE,
    }
    exec(spec_source, namespace)
    exec(sliders_source, namespace)
    exec(status_source, namespace)

    return namespace, mock_st


# ---------------------------------------------------------------------------
# Exec-based behavioral tests
# ---------------------------------------------------------------------------

class TestRenderAllocationSlidersBehavior:
    """Behavioral tests for render_allocation_sliders via exec + MockStreamlit."""

    def _run_sliders(self, prefix="", load_count=3):
        ns, mock_st = _extract_functions()
        alloc = {
            'us_large_cap': 0.30, 'us_small_cap': 0.10,
            'international_dev': 0.10, 'emerging_markets': 0.05,
            'us_bonds': 0.20, 'tips': 0.10,
            'cash': 0.05, 'reits': 0.10,
        }
        result = ns['render_allocation_sliders'](alloc, prefix, load_count)
        return result, mock_st

    def test_returns_dict_with_all_8_fields(self):
        result, _ = self._run_sliders()
        expected = {'us_large_cap', 'us_small_cap', 'international_dev',
                    'emerging_markets', 'us_bonds', 'tips', 'cash', 'reits'}
        assert set(result.keys()) == expected

    def test_slider_label_has_no_hash(self):
        _, mock_st = self._run_sliders(prefix="", load_count=5)
        for call in mock_st.slider_calls:
            assert '#' not in call['label'], (
                f"Slider label contains #: {call['label']}"
            )

    def test_key_param_includes_load_count(self):
        _, mock_st = self._run_sliders(prefix="Ret.", load_count=7)
        for call in mock_st.slider_calls:
            assert call['key'] is not None, "Slider key is None"
            assert '7' in call['key'], (
                f"Slider key does not include load_count: {call['key']}"
            )


class TestRenderAllocationStatusBehavior:
    """Behavioral tests for render_allocation_status via exec + MockStreamlit."""

    def _run_status(self, total, button_return=False):
        ns, mock_st = _extract_functions()
        mock_st.button_return = button_return

        class FakeConfig:
            pre_retirement_allocation = {}

        config = FakeConfig()
        alloc = {'us_large_cap': total, 'us_bonds': 1.0 - total}
        ns['render_allocation_status'](
            total=total,
            label="Pre-Retirement",
            group_key="pre",
            allocation=alloc,
            config_field_name="pre_retirement_allocation",
            config=config,
        )
        return mock_st, config

    def test_status_success_at_100_percent(self):
        mock_st, _ = self._run_status(1.0)
        assert len(mock_st.success_calls) == 1
        assert '100.0%' in mock_st.success_calls[0]

    def test_status_warning_under_tolerance(self):
        mock_st, _ = self._run_status(0.97)  # 3% off, under 5% tolerance
        assert len(mock_st.warning_calls) == 1
        assert len(mock_st.error_calls) == 0

    def test_status_error_over_tolerance(self):
        mock_st, _ = self._run_status(0.90)  # 10% off, over 5% tolerance
        assert len(mock_st.error_calls) == 1
        assert len(mock_st.warning_calls) == 0

    def test_status_shows_over_when_exceeds(self):
        mock_st, _ = self._run_status(1.10)
        msgs = mock_st.warning_calls + mock_st.error_calls
        assert any('over' in m for m in msgs), (
            f"Expected 'over' in status message, got: {msgs}"
        )

    def test_autofix_calls_normalize(self):
        mock_st, config = self._run_status(0.90, button_return=True)
        # When button is clicked, config should be updated with normalized allocation
        assert isinstance(config.pre_retirement_allocation, dict)
        total = sum(config.pre_retirement_allocation.values())
        assert abs(total - 1.0) < 1e-10, (
            f"Auto-fix should normalize to 1.0, got {total}"
        )


class TestHasattrGuard:
    """Verify that render_allocation_status has a hasattr guard before setattr."""

    def test_hasattr_guard_in_source(self):
        with open(APP_PY) as f:
            source = f.read()
        start = source.find('def render_allocation_status(')
        end = source.find('\ndef ', start + 1)
        func_body = source[start:end]
        assert 'hasattr(config, config_field_name)' in func_body, (
            "render_allocation_status is missing hasattr guard"
        )

    def test_hasattr_guard_raises_on_bad_attribute(self):
        """Guard should raise AttributeError for nonexistent config attribute."""
        ns, mock_st = _extract_functions()
        mock_st.button_return = True

        class FakeConfig:
            pass

        config = FakeConfig()
        with pytest.raises(AttributeError, match="SimulationConfig has no attribute"):
            ns['render_allocation_status'](
                total=0.90,
                label="Test",
                group_key="test",
                allocation={'a': 0.5, 'b': 0.4},
                config_field_name="nonexistent_field",
                config=config,
            )
