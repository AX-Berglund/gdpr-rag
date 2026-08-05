"""Extracting evaluation labels from Court of Justice judgments.

Every label in this project so far was written by hand, which makes the
evaluation only as good as the person who wrote it. A judgment carries its own
labels: the Court states which provisions were at issue, and the referring
court's questions are quoted verbatim. Taking both from the document rather
than from an author is the point of this module.

Two signals are used, and which one produced a label is recorded so a reader
can audit it:

  headnote  - the keyword block beneath the case title, where the Court's
              reporters name the provisions at issue. Authoritative.
  frequency - how often an article is discussed in the body. A judgment
              mentions its subject dozens of times and cites neighbouring
              provisions once or twice, so the distribution separates them.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass, field

# "Article 82(1)" / "Article 17" — paragraph parts are dropped, because labels
# in this project are article-level throughout.
_ARTICLE = re.compile(r"Article\s+(\d{1,2})\s*(?:\(\d+\))*")

# The judgment's own keyword block sits between the chamber line and the body,
# wrapped in parentheses and separated by dashes.
_HEADNOTE = re.compile(r"\(\s*Reference for a preliminary ruling(.*?)\)\s*(?:In Case|$)", re.S)

# The Court cites many instruments. "Article 10" in a succession-regulation
# judgment is not Article 10 GDPR, so every article has to be scoped to the
# regulation before it can become a label — an early version of this module
# happily labelled two non-GDPR cases.
GDPR_MARKERS = ("2016/679", "General Data Protection Regulation")

# Inside a headnote the instrument is named, then its provisions follow until
# the next instrument is named.
_INSTRUMENT_SEGMENT = re.compile(r"(Regulation|Directive|Charter|Treaty|Decision)\s*\(?[^–]{0,60}")

# Articles that appear because of how the case reached the Court rather than
# because of what it decided.
PROCEDURAL_ARTICLES = {
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    78,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
}


class NotAGdprCase(ValueError):
    """The judgment does not concern Regulation (EU) 2016/679."""


class NotAJudgment(ValueError):
    """The page holds no judgment text at all.

    EUR-Lex occasionally serves a script shell instead of the document. That is
    a transport failure, and reporting it as "not a GDPR case" would blame the
    Court for a network problem.
    """


@dataclass
class CaseExtract:
    """What could be read out of one judgment."""

    celex: str
    title: str = ""
    headnote_articles: list[int] = field(default_factory=list)
    frequent_articles: list[int] = field(default_factory=list)
    mentions: dict[int, int] = field(default_factory=dict)
    questions: list[str] = field(default_factory=list)

    @property
    def articles(self) -> list[int]:
        """Best-effort labels: the headnote if it parsed, else frequency."""
        chosen = self.headnote_articles or self.frequent_articles
        return sorted(a for a in chosen if a not in PROCEDURAL_ARTICLES)

    @property
    def label_source(self) -> str:
        return "headnote" if self.headnote_articles else "frequency"


def to_text(raw_html: str) -> str:
    """Flatten judgment HTML to plain text."""
    without_tags = re.sub(r"<[^>]+>", " ", raw_html)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def looks_like_a_decision(text: str) -> bool:
    """Whether the page actually contains a judgment or order."""
    upper = text.upper()
    return any(m in upper for m in ("JUDGMENT OF THE COURT", "ORDER OF THE COURT"))


def is_gdpr_case(text: str) -> bool:
    """Whether this judgment concerns the GDPR at all."""
    return any(marker in text for marker in GDPR_MARKERS)


def _gdpr_headnote_segment(headnote: str) -> str:
    """The part of a headnote that follows the naming of the GDPR.

    A headnote lists instruments in turn; only the provisions after
    "Regulation (EU) 2016/679" belong to it.
    """
    start = -1
    for marker in GDPR_MARKERS:
        found = headnote.find(marker)
        if found >= 0:
            start = found
            break
    if start < 0:
        return ""
    rest = headnote[start:]
    # Stop at the next instrument, if the headnote moves on to one.
    following = _INSTRUMENT_SEGMENT.search(rest, len(marker))
    return rest[: following.start()] if following else rest


def extract(celex: str, raw_html: str, *, min_mentions: int = 5) -> CaseExtract:
    """Read labels and referred questions out of one judgment.

    Raises ``NotAGdprCase`` if the judgment is about something else, which is
    the common outcome when a case list is assembled by guessing identifiers.
    """
    text = to_text(raw_html)
    if not looks_like_a_decision(text):
        raise NotAJudgment(f"{celex} returned no decision text ({len(text)} chars)")
    if not is_gdpr_case(text):
        raise NotAGdprCase(f"{celex} does not cite Regulation (EU) 2016/679")
    result = CaseExtract(celex=celex)

    heading = re.search(r"JUDGMENT OF THE COURT[^(]*\(([^)]*)\)\s*(\d+\s+\w+\s+\d{4})", text)
    if heading:
        result.title = f"Judgment of {heading.group(2)} ({heading.group(1).strip()})"

    counted = Counter(int(m.group(1)) for m in _ARTICLE.finditer(text))
    result.mentions = dict(counted.most_common())

    headnote = _HEADNOTE.search(text)
    if headnote:
        # Only the provisions listed after the GDPR is named belong to it. A
        # headnote routinely also cites the Charter and the Treaties.
        segment = _gdpr_headnote_segment(headnote.group(1))
        result.headnote_articles = sorted({int(m.group(1)) for m in _ARTICLE.finditer(segment)})

    # An article discussed many times is what the case is about; one mentioned
    # once is scenery.
    result.frequent_articles = sorted(a for a, n in counted.items() if n >= min_mentions)
    result.questions = _referred_questions(text)
    return result


def _referred_questions(text: str) -> list[str]:
    """The referring court's questions, quoted verbatim in the judgment.

    Using the Court's wording keeps the query side of the evaluation external
    too — otherwise the questions would be the author's paraphrase again.
    """
    block = re.search(
        r"decided to stay the proceedings and to refer the following question[^:]*:(.{0,4000})",
        text,
        re.S,
    )
    if not block:
        return []
    body = block.group(1)
    parts = re.split(r"\(\d+\)\s+|\s\d\.\s+", body)
    questions = []
    for part in parts:
        part = part.strip().strip("‘’'\"")
        if len(part) > 60 and "?" in part:
            questions.append(part[: part.rindex("?") + 1])
    return questions[:5]
