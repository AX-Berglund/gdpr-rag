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
