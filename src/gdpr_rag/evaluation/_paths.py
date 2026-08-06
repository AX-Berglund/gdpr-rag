"""Locating the evaluation sets in both layouts.

The YAML lives at the repository root, where it is meant to be read and edited
by people. It is also shipped inside the wheel, because a package whose
evaluation sets vanish on install is a package whose evaluation cannot be
reproduced by anyone who did not clone it.

Both locations are searched. This was a real failure: installed to
site-packages, a path built from ``parents[3]`` pointed into the virtualenv
rather than the repository, and every loader raised FileNotFoundError.
"""

from __future__ import annotations

from pathlib import Path

# Populated in the wheel by the force-include rule in pyproject.toml.
_BUNDLED = Path(__file__).resolve().parent.parent / "_data"

# The repository checkout, when running from source.
_REPO = Path(__file__).resolve().parents[3] / "evaluation"


def evaluation_file(name: str) -> Path:
    """Path to an evaluation set, wherever it is available.

    Returns the repository copy when present so a developer's edits take
    effect without reinstalling; falls back to the bundled copy otherwise.
    """
    for directory in (_REPO, _BUNDLED):
        candidate = directory / name
        if candidate.exists():
            return candidate
    # Neither exists: return the repository path so the error names the place
    # a developer would look first.
    return _REPO / name
