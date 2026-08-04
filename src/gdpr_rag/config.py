"""Local configuration.

Secrets belong in the environment, but typing an API key into every shell is
how keys end up in shell history and screen recordings. A gitignored .env at
the project root is read instead, if one exists.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def load_env(path: Path | None = None) -> bool:
    """Load ``.env`` into the environment. Returns whether a file was found.

    Existing environment variables win, so an explicitly exported key is never
    silently overridden by a stale file.
    """
    env_file = path or ENV_FILE
    if not env_file.exists():
        return False

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    return True
