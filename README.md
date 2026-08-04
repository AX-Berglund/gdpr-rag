# gdpr-rag

[![CI](https://github.com/AX-Berglund/gdpr-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/AX-Berglund/gdpr-rag/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Retrieval-augmented question answering over the GDPR, where answers cite the articles they
came from — and where the retrieval is **measured** rather than assumed.

Most RAG projects stop at "it returns something plausible." This one asks a narrower
question and answers it with numbers: *does parsing a document's real structure retrieve
better than sliding a fixed-size window over it?*

Runs entirely locally. No API key, no cloud database, no account.

---

## Results

64 labelled questions (60 answerable + 4 refusal probes), scored over the full regulation.
`hit@k` is the share of questions where a correct article appeared in the top *k*.

**Local MiniLM embeddings** (`all-MiniLM-L6-v2`):

| chunking | chunks | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| **structured** | 774 | 0.350 | **0.617** | **0.700** | **0.817** | **0.505** | **0.580** |
| fixed-400 | 522 | 0.300 | 0.567 | 0.650 | 0.783 | 0.467 | 0.544 |
| fixed-800 | 261 | **0.383** | 0.550 | 0.617 | 0.717 | 0.493 | 0.547 |
| fixed-1600 | 131 | 0.300 | 0.517 | 0.600 | 0.733 | 0.433 | 0.505 |

**Lexical baseline** (hashed unigrams, no semantics) — best row shown:

| chunking | hit@1 | hit@5 | hit@10 | nDCG@10 |
|---|---|---|---|---|
| structured | 0.150 | 0.383 | 0.433 | 0.293 |
| fixed-1600 | **0.217** | **0.467** | **0.550** | **0.385** |

Reproduce with `python scripts/run_ablation.py --dense --k 1 3 5 10`.

### What the numbers say

**Structured chunking wins from k=3 upward, on every metric.** At k=5 it retrieves a
correct article for 70% of questions against 65% for the best fixed-size configuration,
and it leads on MRR and nDCG at every depth beyond 1 — so it does not merely find the
right article, it ranks it higher.

**At k=1 the best fixed-size run edges ahead** (0.383 vs 0.350). With 60 questions each
one is worth 1.7 points, so that gap is two questions. It is not a result worth defending
in either direction, and it is reported here because dropping it would be dishonest.

**The whole picture flips on the lexical baseline**, where bigger fixed chunks win
outright. That is not a contradiction — it is the artifact the baseline exists to expose.
Hashed-unigram matching rewards chunks for containing more words, so a 1600-character
window beats a precise one for reasons that have nothing to do with relevance. Any
structural comparison run only against lexical retrieval would reach the wrong conclusion.

**The dense model earns its download**, which is the question the baseline was there to
answer: 0.817 against 0.550 hit@10. That is worth stating because it is not automatic —
on small, jargon-dense corpora a lexical baseline is often competitive.

**Honest limitation.** With n=60, differences of two or three questions are suggestive,
not significant. The consistent direction across k=3, 5 and 10 and across all three
metrics is the real evidence here — not any single cell.

### The advantage that is not in the table

A fixed-size window has no structural address. The best citation it can produce is
"Article 17". Structured chunks carry theirs by construction, so an answer can cite
**Article 17(1)(a)** — the specific condition, not the whole article. For a system whose
purpose is grounding claims in regulation, that is the difference between a citation a
reader can check and one they have to go hunting through.

---

## How it works

```
EUR-Lex HTML ─▶ structure-aware parse ─▶ embed ─▶ SQLite + cosine ─▶ retrieve ─▶ cited answer
                        │                                                            │
                        └── article / paragraph / point preserved ───────────────────┘
```

**Ingest.** The Official Journal HTML marks structure explicitly — an article is
`<div id="art_17">`, its paragraphs are `<div id="017.001">` — so headers are *read*
rather than told apart from the hundreds of "Article 6(1)" cross-references in the prose.
The PDF of the same regulation carries the same text but not the same information: page
furniture bleeds into the body at every page break (30 polluted chunks against 0) and
justified typesetting leaves 514 mid-sentence line breaks. Both sources are parsed and
cross-checked against each other; disagreement about which articles exist means a parser
is wrong.

**Retrieval.** Brute-force exact cosine over roughly a thousand chunks — sub-millisecond,
and it always returns the true top-k. An approximate index would add a tuning surface and,
worse, put index recall into the measurements as a confound.

**Generation.** Citations are checked, not trusted. Asking a model to cite its sources
makes citations *appear*; it does not make them true. Every citation the model emits is
verified against what retrieval actually returned, and invented ones are surfaced rather
than silently kept. Specificity is directional: having retrieved `Article 17(1)(a)`
supports citing `Article 17`, but having retrieved only `Article 15` does not support a
claim about `Article 15(3)`. When retrieval returns nothing, the model is never called at
all — an unanswerable question should not come back as a fluent guess.

---

## Quickstart

```bash
pip install -e ".[local,dev]"
```

**Get the corpus.** EUR-Lex sits behind a bot challenge, so this one step is manual:
open [the regulation on EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679),
save the page as **HTML only** into `data/raw/`, then check it parsed:

```bash
python scripts/validate_corpus.py
# 774 chunks across 99 articles — PASS
```

**Ask it something:**

```bash
gdpr-rag index                                  # build an index
gdpr-rag ask "Can I ask a company to delete my data?"
```

**Run the experiment:**

```bash
python scripts/run_ablation.py --dense --k 1 3 5 10
```

**Or use the demo:**

```bash
pip install -e ".[local,demo]"
streamlit run demo/app.py
```

The chunking strategy is a control in the sidebar, so you can ask one question and
watch structured and fixed-size retrieval disagree — the table above, made touchable.
Retrieval needs no API key. To also generate a cited answer, copy `.env.example` to `.env`
and add an `OPENAI_API_KEY` — `.env` is gitignored. Any citation the model produces that
retrieval did not return is flagged as an error above the evidence.

---

## The evaluation set

`evaluation/questions.yaml` holds 64 hand-labelled questions. Two choices shaped it:

**Labels are article-level, never paragraph-level.** Labelling `Article 17(1)` would score
retrieval as *wrong* for returning Article 17 at a different paragraph, which is not a
failure worth punishing. A validator enforces this so a well-meaning paragraph label
cannot slip in later.

**Four questions the GDPR does not answer.** Two are obvious (Irish tax rates, nginx
config) and two deliberately share vocabulary with it — CCPA opt-out links, ePrivacy
cookie lifetimes. Those are exactly what a RAG system answers confidently and wrongly, and
nothing measures that unless the set contains them. They are excluded from retrieval means,
and the exclusion count is printed with every result: a mean over an unstated subset is how
misleading numbers get published.

---

## Notes

I built the first version of this in late 2024 and it stalled at the point most RAG
projects stall — the pipeline ran, retrieval returned chunks that looked reasonable, and I
had no idea whether it was any good. Rebuilding it, the interesting work turned out not to
be the retrieval at all but deciding what "any good" meant, then finding out I was wrong
about which chunking strategy would win, and wrong in a way that depended on which embedder
I asked.

The lexical baseline was originally a test double, added so the suite could run without
downloading a model. It became the most informative row in the table.

---

## License

MIT
