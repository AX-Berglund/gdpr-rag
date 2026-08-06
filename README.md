# gdpr-rag

[![CI](https://github.com/AX-Berglund/gdpr-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/AX-Berglund/gdpr-rag/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Retrieval-augmented question answering over the GDPR, where answers cite the articles they
came from — and where every design decision is **measured** rather than asserted.

Most RAG projects stop at "it returns something plausible". This one is built around the
question of how you would know. It ships three evaluation sets, and the hardest of them is
labelled by the **Court of Justice** rather than by its author: 70 judgments whose article
labels come from the Court's own headnotes, whose queries are the referring courts'
questions, and whose membership is decided by the EU Publications Office triplestore.

Several results contradicted the assumption that prompted them, and are reported anyway:

- structured chunking beats fixed-size windows — except at k=1, and except under a lexical
  retriever, where the ordering reverses entirely
- HyDE query rewriting takes individuals' questions from 0.250 to 0.917, and lifts a
  hashed-unigram baseline *above* a neural embedding model
- explicitly restating a question's perspective makes retrieval worse overall, despite
  helping the group it targets
- real judgments score 0.268 coverage where hand-written questions score 0.700

The core runs entirely locally — no API key, no cloud database, no account. Query
rewriting and answer generation are opt-in and need a key; retrieval, indexing and the
whole evaluation do not.

**[Try it](https://ax-gdpr-rag.streamlit.app)** — the chunking strategy is a control, so the
central result is reproducible in the browser rather than only reported.

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

### How much does realistic phrasing cost?

The questions above are well-formed English, which flatters retrieval. Real users type
`how do i get a company to delete my data`, not *"Can I ask a company to delete the
personal data it holds about me?"* Every question therefore carries a second, colloquial
phrasing sharing the same labels, so the gap is measured rather than assumed.

| | hit@1 | hit@5 | hit@10 |
|---|---|---|---|
| structured, formal | 0.350 | **0.700** | 0.817 |
| structured, colloquial | 0.250 | **0.600** | 0.750 |
| lexical baseline, formal | 0.150 | **0.383** | 0.433 |
| lexical baseline, colloquial | 0.083 | **0.183** | 0.383 |

`python scripts/run_ablation.py --dense --phrasing formal colloquial`

Casual phrasing costs the dense model about **10 points of hit@5** — six questions out of
sixty. The lexical baseline loses more than half its accuracy, which is what you would
expect from a method that matches words: colloquial English shares far fewer exact terms
with legal drafting than a well-formed question does.

Two things follow. That 10-point gap is the headroom a query-rewriting step would try to
recover, and it is now a measurable target rather than a hunch. And structured chunking
still beats fixed-size under the harder phrasing (0.600 against 0.500), so the main result
holds where it matters most.

### Who is asking matters more than anything else measured here

Every question is also labelled with the perspective it is asked from: `subject` (the
individual whose data it is), `organisation` (controller, processor or DPO), or `neutral`
(definitional questions belonging to no one).

| perspective | n | hit@5, structured + MiniLM |
|---|---|---|
| data subject | 12 | **0.333** |
| organisation | 37 | **0.784** |
| neutral | 11 | 0.818 |

An individual's question retrieves at **less than half** the rate of an organisation's —
a larger effect than chunking strategy or phrasing, the two variables this project set out
to study.

The cause is in how the regulation is drafted. It states obligations on controllers, and
even the rights chapter is phrased that way: Article 17 does not say *you may delete your
data*, it says *"The data subject shall have the right to obtain from the controller the
erasure…"*. An organisation's question already speaks the corpus's language. An
individual's has to be mapped onto somebody else's duty first.

Sometimes perspective changes *which article is correct*, not merely the wording. A
controller asking about breach reporting wants Article 33 (notify the authority); an
individual asking "should they have told me?" wants Article 34 (notify the individual).

Perspective is labelled rather than inferred from pronouns. Pronouns look like a free
signal and break immediately: a sole trader is both parties at once, and a lawyer asks on
behalf of someone else entirely.

*Caveat: n=12 for the subject group, so each question moves that figure by 8 points. The
direction is solid; the precise value is not.*

### The advantage that is not in the table

A fixed-size window has no structural address. The best citation it can produce is
"Article 17". Structured chunks carry theirs by construction, so an answer can cite
**Article 17(1)(a)** — the specific condition, not the whole article. For a system whose
purpose is grounding claims in regulation, that is the difference between a citation a
reader can check and one they have to go hunting through.

---

### Query rewriting closes the gap — and changes which retriever you need

Users write in one register; the regulation is drafted in another. Two rewriting
strategies, each measured against `NullRewriter`, which does nothing:

**hit@5 on colloquial phrasing, structured chunking, by who is asking:**

| retriever | rewrite | subject | organisation | neutral |
|---|---|---|---|---|
| lexical baseline | none | 0.250 | 0.216 | 0.000 |
| lexical baseline | **HyDE** | **0.917** | **0.784** | 0.455 |
| MiniLM | none | 0.417 | 0.676 | 0.545 |
| MiniLM | **HyDE** | 0.750 | **0.784** | 0.545 |
| MiniLM | perspective | 0.583 | 0.757 | 0.091 |

**HyDE nearly closes the individual-versus-organisation gap.** It drafts a hypothetical
passage in the regulation's own obligation-side register and embeds that, so the role-flip
an individual's question needs happens before retrieval rather than not at all. The draft
is frequently wrong on substance; that does not matter, because it is never shown to
anyone and is used only to find real text.

**The surprise: HyDE lifts the lexical baseline above the neural model.** Hashed unigrams
with HyDE reach 0.917 on subject questions against MiniLM's 0.750. The reason is
mechanical — HyDE emits the corpus's exact statutory vocabulary, which is precisely what
word matching needs and what a colloquial question denies it. The retrieval floor becomes
competitive once the query stops being the bottleneck.

That is a real engineering choice rather than a curiosity: HyDE plus free lexical retrieval
needs an API call per query (~1.8 s); MiniLM alone needs a 90 MB local model and no network.

**Explicitly restating the perspective is worse than doing nothing** (0.583 vs 0.600
overall). It helps individuals, costs organisations, and destroys neutral questions
(0.545 → 0.091) by forcing an obligation framing onto definitional ones where it does not
belong. Perspective-aware rewriting has to be conditional; applied blanket it is a net
loss — which is exactly why it is measured rather than assumed.

*Caveat: n=12 for the subject group, so 0.250 → 0.917 is three questions becoming eleven.
The direction is unmistakable; the precision is not.*

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

### Seeing what happened

A score of 0.306 tells you the pipeline failed. It does not tell you *which stage*
failed — and without that, a bad number is a dead end. So every stage records what it
received and produced:

```
$ gdpr-rag ask "can my gym refuse deletion for accounting reasons?" --answer --trace

query: "can my gym refuse deletion for accounting reasons?"  (2676 ms total)
  · retrieve  20 ms
    citations: ['Article 17(1)(e)', 'Article 17(1)(a)', 'Article 18(1)(c)', ...]
    scores: [0.522, 0.508, 0.475, 0.454, 0.453]
  · generate  2656 ms
    prompt_chars: 1801
    completion: No, the excerpts do not provide information that supports...
  · verify_citations  0 ms
    supported: []
    grounded: False
```

That trace diagnoses itself. Retrieval returned Article 17 and Article 18 but never
surfaced the exemption in Article 17(3), which is the entire answer — so the model
correctly declined on the evidence it was given. **The failure is in retrieval, not
generation**, and the trace is what makes that visible rather than a guess.

Tracing is a data structure and a stopwatch, not a framework: no registry, no graph, no
chaining abstraction. Every stage is a plain function taking an optional trace, and the
library behaves identically without one.

## Evaluation

They measure different things, so a combined number would mean nothing.
`python scripts/evaluate.py --dense`:

| set | n | labelled by | what it measures | hit/coverage @5 |
|---|---|---|---|---|
| questions | 64 | hand | one-article lookups | **0.700** hit |
| scenarios | 18 | hand | multi-article narratives | **0.306** coverage |
| **cases** | 70 | **the Court** | real judgments | **0.268** coverage |

**The real-case set is the one that matters most**, because its labels are not
mine. They come from the Court's own headnote, which names the provisions at
issue, and the queries are the referring court's own questions. Both sides are
external. The case list itself is discovered from the EU Publications Office
triplestore — `cdm:case-law_interpretes_resource_legal` — so even the choice of
which cases count is not an authoring decision.

It is also by far the hardest, and that is not a defect. These cases reached the
Court precisely because the law was unclear, so 0.268 coverage is an honest
reading of a genuinely hard task rather than a flattering one. At k=10 the
system surfaces a correct article for 60% of judgments but everything a judgment
needs for only 13%.

### The leak, measured rather than assumed

A referring court's question routinely names the provision it asks about — 52%
of this set gives away at least one of its own answers, 32% of labels overall.
Queries are therefore redacted. Running both ways puts a number on it:

| | coverage @10 |
|---|---|
| redacted queries | 0.323 |
| raw queries | 0.354 |

Only three points, and it moves in both directions across *k*. The reason is
architectural: this pipeline keeps article numbers as chunk **metadata**, not as
embedded text, so citing "Article 82" gives the embedding nothing to match. The
structured chunking decision immunised retrieval against a leak it was never
designed for. Generation is a different matter — it reads the citation directly,
which is why redaction stays on.

### What is not fixed

`gpt-4o-mini` was trained on text that includes these judgments, so a generated
answer may be recalled rather than retrieved. That contamination is unfixable
for older cases and is stated rather than worked around.

### How the hand-written set was built

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
and add an `OPENAI_API_KEY` — `.env` is gitignored.

A public deployment that supplies its own key spends its owner's money on strangers, so
generation is capped: a few answers per visit and a daily ceiling, after which the demo
declines and retrieval carries on unaffected. Set `GEN_PER_SESSION` and `GEN_PER_DAY` to
change the limits. This is a second line of defence — the first is a spending limit set
with the API provider, which is the only thing that holds if the process restarts. Any citation the model produces that
retrieval did not return is flagged as an error above the evidence.

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
