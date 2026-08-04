"""Removing the answer from an evaluation query.

A referring court's question routinely names the provision it is asking about:
*"Does the award of compensation under Article 82 of the GDPR require…"*. Used
as an evaluation query unchanged, that is not a test of retrieval — it is a
test of whether the retriever can read a citation. Measured on this set, 52% of
cases name at least one of their own answer articles, covering 29% of all
labels.

Redaction is therefore not tidiness, it is the difference between measuring a
system and measuring a leak. Running the evaluation both ways quantifies that
leak rather than assuming it away.
"""

from __future__ import annotations

import re

# "Article 82", "Article 17(1)(a)", "Articles 12 to 22", "Article 5(1)(e) GDPR".
# EUR-Lex wraps editorial insertions in square brackets — "Article [2](h)" —
# so the number may be bracketed. Missing that let one citation through.
_NUMBER = r"\[?\s*\d{1,3}\s*\]?"
_SUBDIVISION = r"(?:\s*\(\s*[a-z0-9]+\s*\))*"
_ARTICLE = re.compile(
    rf"\bArticles?\s+{_NUMBER}{_SUBDIVISION}"
    rf"(?:\s*(?:,|and|to|or)\s*{_NUMBER}{_SUBDIVISION})*",
    re.IGNORECASE,
)

# Recitals explain articles and often name them by subject, which is a softer
# but real hint.
_RECITAL = re.compile(r"\brecitals?\s+\d{1,3}(?:\s*(?:,|and|to)\s*\d{1,3})*", re.IGNORECASE)

PLACEHOLDER = "[provision]"
RECITAL_PLACEHOLDER = "[recital]"


def redact_articles(text: str, placeholder: str = PLACEHOLDER) -> str:
    """Replace every article citation with a placeholder."""
    return _ARTICLE.sub(placeholder, text)


def redact(text: str) -> str:
    """Remove article and recital citations, leaving the substance intact."""
    return _RECITAL.sub(RECITAL_PLACEHOLDER, redact_articles(text))


def leak_score(text: str, labels: set[int]) -> float:
    """Fraction of ``labels`` a query gives away by citing them directly."""
    if not labels:
        return 0.0
    cited = {
        int(n) for match in _ARTICLE.finditer(text) for n in re.findall(r"\d{1,3}", match.group(0))
    }
    return len(labels & cited) / len(labels)
