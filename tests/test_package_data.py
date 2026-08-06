"""The evaluation sets must survive installation.

They live at the repository root so people can read and edit them, and are
force-included in the wheel so they are still there after a pip install.
Without that, every loader raised FileNotFoundError once installed — which is
how the deployed demo failed, and would equally break anyone who pip-installed
the package and tried to reproduce the results.
"""

import zipfile
from pathlib import Path

import pytest

from gdpr_rag.evaluation._paths import evaluation_file

ROOT = Path(__file__).resolve().parents[1]
SETS = ["questions.yaml", "scenarios.yaml", "cases.yaml"]


class TestResolution:
    @pytest.mark.parametrize("name", SETS)
    def test_each_set_resolves_to_an_existing_file(self, name):
        assert evaluation_file(name).exists()

    def test_a_missing_set_still_names_a_sensible_path(self):
        # The error should point at the repository, where a developer looks.
        assert evaluation_file("absent.yaml").parent.name == "evaluation"


class TestWheelPackaging:
    def test_pyproject_force_includes_the_evaluation_directory(self):
        config = (ROOT / "pyproject.toml").read_text()
        assert "force-include" in config
        assert '"evaluation" = "gdpr_rag/_data"' in config

    @pytest.mark.skipif(not list(ROOT.glob("dist/*.whl")), reason="no built wheel to inspect")
    def test_a_built_wheel_carries_the_sets(self):
        wheel = sorted(ROOT.glob("dist/*.whl"))[-1]
        names = zipfile.ZipFile(wheel).namelist()
        for name in SETS:
            assert f"gdpr_rag/_data/{name}" in names


class TestDemoResilience:
    """The demo must not die over decoration.

    Example questions come from the labelled set when it is available. When it
    is not — an install missing its package data, say — the page should still
    work, because examples are a nicety and retrieval is the point.
    """

    def test_examples_fall_back_when_the_set_is_unreadable(self, monkeypatch):
        import sys
        import types

        stub = types.ModuleType("streamlit")

        def cache_resource(*args, **kwargs):
            # Used both bare (@st.cache_resource) and called (@st.cache_resource(...)).
            return args[0] if args and callable(args[0]) else (lambda f: f)

        stub.cache_resource = cache_resource
        stub.secrets = {}
        monkeypatch.setitem(sys.modules, "streamlit", stub)

        namespace = {"__file__": str(ROOT / "demo" / "app.py")}
        source = (ROOT / "demo" / "app.py").read_text().split("def main()")[0]
        exec(compile(source, "demo/app.py", "exec"), namespace)

        def boom():
            raise FileNotFoundError("no evaluation set")

        namespace["load_questions"] = boom
        assert namespace["_examples"] == namespace["_examples"]
        examples = namespace["_examples"]()
        assert examples == namespace["FALLBACK_EXAMPLES"]
        assert all(e.endswith("?") for e in examples)
