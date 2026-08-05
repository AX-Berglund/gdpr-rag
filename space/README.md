---
title: gdpr-rag
emoji: "§"
colorFrom: indigo
colorTo: gray
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
---

# gdpr-rag

Retrieval over the General Data Protection Regulation with **article-level citations** —
`Article 17(1)(a)`, not just "Article 17".

The chunking strategy is a control in the sidebar, so the central result is reproducible
here rather than only reported. Ask a question with structured chunking, then switch to
fixed-size and ask it again.

## What this demonstrates

Parsing the regulation's own structure — articles, numbered paragraphs, lettered points —
instead of sliding a fixed-size window over it. Every chunk keeps its structural address,
so it is citable by construction.

Whether that also *retrieves* better is measured rather than assumed. On 64 labelled
questions, structured chunking reaches hit@5 of 0.700 against 0.617 for the best
fixed-size configuration — but the ordering reverses under a lexical retriever, which is
the more interesting half of the result.

The full write-up, including three evaluation sets (the hardest labelled by the Court of
Justice rather than by the author) and several results that contradicted their own
assumptions, is in the repository:

**https://github.com/AX-Berglund/gdpr-rag**

## Scope

Retrieval only. Answer generation needs an API key, and retrieval is the part this project
measures.

## Attribution

The bundled corpus is Regulation (EU) 2016/679, © European Union, 1998–2026, obtained from
[EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679). Reuse is
authorised under [Decision 2011/833/EU](https://eur-lex.europa.eu/eli/dec/2011/833/oj).
Only the official text published in the Official Journal is authentic.

Code is MIT licensed.
