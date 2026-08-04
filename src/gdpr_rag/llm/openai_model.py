"""OpenAI-backed answer generation.

Optional. Retrieval — the part this project actually measures — needs no
model and no key; this exists so the final answer can be generated when one is
available. The import is deferred so the package stays usable without the
openai extra installed.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIModel:
    """Minimal chat-completion wrapper satisfying the LanguageModel protocol."""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> None:
        self._model = model
        # Zero temperature because an answer about regulation should not vary
        # between identical runs.
        self._temperature = temperature
        self._client = None

    @property
    def name(self) -> str:
        return self._model

    def complete(self, prompt: str) -> str:
        response = self._get_client().chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def _get_client(self):
        if self._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Retrieval works without it; "
                    "only answer generation needs a key."
                )
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "Answer generation needs the 'openai' extra: pip install 'gdpr-rag[openai]'"
                ) from exc
            self._client = OpenAI()
        return self._client
