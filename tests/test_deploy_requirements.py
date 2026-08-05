"""The deployment manifests must stay in step with the package's own needs.

A host installing from requirements.txt gets whatever is listed here and
nothing else. When that drifts from pyproject, the app does not crash — it
silently falls back to the lexical baseline and serves noticeably worse results
while looking fine, which is the failure mode worth a test.
"""

import re
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = [ROOT / "requirements.txt", ROOT / "deploy" / "requirements.txt"]


def names(path: Path) -> set[str]:
    found = set()
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            found.add(re.split(r"[<>=\[]", line)[0].strip().lower())
    return found


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: str(p.name))
class TestManifests:
    def test_exists(self, manifest):
        assert manifest.exists()

    def test_covers_the_packages_runtime_dependencies(self, manifest):
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
        required = {
            re.split(r"[<>=\[]", d)[0].strip().lower() for d in pyproject["project"]["dependencies"]
        }
        assert required <= names(manifest), f"missing: {required - names(manifest)}"

    def test_includes_the_local_embedding_stack(self, manifest):
        # Without these the app degrades to the lexical baseline silently.
        assert {"torch", "sentence-transformers"} <= names(manifest)

    def test_pins_the_cpu_torch_index(self, manifest):
        # The default wheel drags in CUDA and blows the disk budget.
        assert "download.pytorch.org/whl/cpu" in manifest.read_text()

    def test_includes_streamlit(self, manifest):
        assert "streamlit" in names(manifest)
