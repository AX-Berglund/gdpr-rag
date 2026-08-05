"""The deployable demo.

Retrieval only. Answer generation needs an API key, and a key in a public
deployment is a key someone else is spending — retrieval is the part this
project measures anyway, so nothing of substance is missing.

The corpus is bundled beside the app rather than downloaded. EUR-Lex sits
behind a bot challenge, so no host can fetch it at build time. EU legislation
is freely reusable under Decision 2011/833/EU with attribution, which the
README gives.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from gdpr_rag.embed import HashingEmbedder, SentenceTransformerEmbedder
from gdpr_rag.ingest import fixed_size_chunks, parse_document
from gdpr_rag.retrieve import Retriever
from gdpr_rag.trace import Trace

CORPUS = Path(__file__).parent / "corpus" / "L_2016119EN.01000101.xml.html"

STRATEGIES = {
    "Structured (article / paragraph / point)": "structured",
    "Fixed-size, 800 characters": "fixed-800",
    "Fixed-size, 1600 characters": "fixed-1600",
}

EXAMPLES = [
    "Can I ask a company to delete the personal data it holds about me?",
    "How long does an organisation have to report a data breach?",
    "Can a decision that significantly affects me be made purely by an algorithm?",
    "What extra protection applies to health data?",
]


@st.cache_resource(show_spinner="Parsing the regulation…")
def load_corpus():
    return parse_document(CORPUS)


@st.cache_resource(show_spinner="Building index…")
def build(strategy: str, dense: bool):
    chunks = load_corpus()
    if strategy.startswith("fixed-"):
        size = int(strategy.split("-")[1])
        chunks = fixed_size_chunks(chunks, size=size, overlap=size // 8)
    embedder = SentenceTransformerEmbedder() if dense else HashingEmbedder(dimensions=2048)
    return Retriever.build(embedder, chunks), len(chunks)


def main() -> None:
    st.set_page_config(page_title="gdpr-rag", page_icon="§", layout="wide")
    st.title("§ gdpr-rag")
    st.caption(
        "Retrieval over the GDPR with article-level citations. "
        "The chunking strategy is a control, so you can reproduce the ablation yourself."
    )

    with st.sidebar:
        st.subheader("Configuration")
        strategy = STRATEGIES[st.radio("Chunking", list(STRATEGIES), index=0)]
        dense = st.toggle(
            "Local MiniLM embeddings",
            value=True,
            help="Off uses a hashed-unigram baseline: instant, and markedly worse.",
        )
        k = st.slider("Results", 1, 10, 5)
        st.divider()
        st.caption(
            "Retrieval only — answer generation needs an API key. "
            "[Source and full results](https://github.com/AX-Berglund/gdpr-rag)"
        )

    retriever, count = build(strategy, dense)
    st.sidebar.metric("Chunks in index", count)

    question = st.text_input("Ask about the GDPR", placeholder=EXAMPLES[0])
    st.caption("Try: " + " · ".join(f"*{e}*" for e in EXAMPLES[1:3]))

    if not question.strip():
        st.info(
            "**What to try.** Ask the first example, then switch Chunking to "
            "*Fixed-size, 1600* and ask it again. Structured chunks cite "
            "`Article 17(1)(a)` — the specific condition. Fixed-size chunks have no "
            "structural address, so the best they can offer is the article number."
        )
        st.stop()

    trace = Trace(question)
    results = retriever.retrieve(question, k=k, trace=trace)

    st.subheader("Retrieved articles")
    for result in results:
        chunk = result.chunk
        title = f" — {chunk.title}" if chunk.title else ""
        with st.container(border=True):
            left, right = st.columns([5, 1])
            left.markdown(f"**{chunk.citation}**{title}")
            right.metric("similarity", f"{result.score:.3f}", label_visibility="collapsed")
            st.write(chunk.text)

    with st.expander("What happened (trace)"):
        st.caption(
            "Every stage records what it received and produced. When an answer is wrong, "
            "this is what tells you which stage was responsible."
        )
        st.code(trace.summary(), language="text")

    if strategy != "structured":
        st.info(
            "Fixed-size chunks carry no structural address, so the best citation available "
            "is the article number — never the specific paragraph or point."
        )


if __name__ == "__main__":
    main()
