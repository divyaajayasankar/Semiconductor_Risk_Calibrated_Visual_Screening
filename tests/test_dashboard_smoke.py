import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_utils_import():
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    U = importlib.import_module("dashboard_utils")
    assert "SemiVision" in U.APP_NAME
    assert "<div" in U.kpi("x", 1)


def test_dashboard_runs_all_pages(pipeline):
    """Execute every dashboard page headlessly with Streamlit's AppTest."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    pages = at.sidebar.radio[0].options
    assert len(pages) == 4
    for p in pages:
        at.sidebar.radio[0].set_value(p).run()
        assert not at.exception, (p, at.exception)
