"""Shared fixtures.

The demo is a Streamlit script rather than an importable module, so testing it
means executing the part above ``main()`` against a stubbed streamlit. That is
worth doing: the demo is the only thing most visitors will ever run, and its
failures have all been the kind that local development cannot surface.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "demo" / "app.py"


@pytest.fixture
def demo_namespace(monkeypatch):
    """Everything demo/app.py defines above main(), with streamlit stubbed."""
    import sys
    import types

    stub = types.ModuleType("streamlit")

    def cache_resource(*args, **kwargs):
        # Used both bare (@st.cache_resource) and called (@st.cache_resource(...)).
        return args[0] if args and callable(args[0]) else (lambda f: f)

    stub.cache_resource = cache_resource
    stub.secrets = {}
    stub.session_state = {}
    monkeypatch.setitem(sys.modules, "streamlit", stub)

    namespace = {"__file__": str(APP)}
    source = APP.read_text().split("def main()")[0]
    exec(compile(source, "demo/app.py", "exec"), namespace)
    return namespace
