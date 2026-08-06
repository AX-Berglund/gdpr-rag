"""Streamlit demo.

Built around the finding rather than around the chatbot: the chunking strategy
is a control, so you can ask one question and watch structured and fixed-size
retrieval disagree. That is the thing the numbers in the README describe, made
touchable.

Answer generation is optional and off unless OPENAI_API_KEY is set — retrieval
is the part this project measures, and it needs no key.

    streamlit run demo/app.py
"""

from __future__ import annotations

import importlib.util
import os
from datetime import date
from pathlib import Path

import streamlit as st

from gdpr_rag.budget import (
    DEFAULT_PER_DAY,
    DEFAULT_PER_SESSION,
    GenerationBudget,
    Usage,
)
from gdpr_rag.config import load_env
from gdpr_rag.embed import HashingEmbedder
from gdpr_rag.evaluation import load_questions
from gdpr_rag.generate import answer_question
from gdpr_rag.ingest import fixed_size_chunks, parse_document
from gdpr_rag.retrieve import Retriever
from gdpr_rag.trace import Trace

ROOT = Path(__file__).resolve().parents[1]

# Locally the corpus is a manual download in data/raw; a deployment carries a
# bundled copy beside deploy/app.py. Searching both means it does not matter
# which entry point a host is pointed at — deploying the wrong one used to
# produce a confident "no corpus found" instead of simply working.
CORPUS_DIRS = (ROOT / "data" / "raw", ROOT / "deploy" / "corpus")

STRATEGIES = {
    "Structured (article / paragraph / point)": "structured",
    "Fixed-size, 800 characters": "fixed-800",
    "Fixed-size, 1600 characters": "fixed-1600",
}


@st.cache_resource(show_spinner="Parsing the regulation...")
def load_corpus(path: str):
    return parse_document(path)


@st.cache_resource(show_spinner="Building index...")
def build_retriever(path: str, strategy: str, embedder_name: str):
    chunks = load_corpus(path)
    if strategy.startswith("fixed-"):
        size = int(strategy.split("-")[1])
        chunks = fixed_size_chunks(chunks, size=size, overlap=size // 8)

    embedder = _embedder(embedder_name)
    return Retriever.build(embedder, chunks), len(chunks)


def _embedder(name: str):
    """Resolve the embedder, degrading rather than crashing.

    A host that installed the package without the 'local' extra should still
    serve a working demo. Falling back to the lexical baseline is a visibly
    worse experience, which is better than a traceback and is announced rather
    than hidden.
    """
    if name == "hashing":
        return HashingEmbedder(dimensions=2048)
    try:
        from gdpr_rag.embed import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
        embedder.encode(["warm-up"])
        return embedder
    except Exception as exc:
        st.warning(
            "The local embedding model is unavailable here, so this is running on the "
            f"hashed-unigram baseline — noticeably worse. ({type(exc).__name__})"
        )
        return HashingEmbedder(dimensions=2048)


# Shown when the labelled set is unavailable. Example text is decoration; a
# demo that dies because it cannot read one is trading a working page for a
# cosmetic detail.
FALLBACK_EXAMPLES = [
    "Can I ask a company to delete the personal data it holds about me?",
    "How long does an organisation have to report a data breach?",
    "Can a decision that significantly affects me be made purely by an algorithm?",
    "What extra protection applies to health data?",
]


def _examples() -> list[str]:
    """Example questions, preferring the labelled set but never requiring it."""
    try:
        found = [q.question for q in load_questions() if not q.unanswerable][:6]
    except Exception:
        return FALLBACK_EXAMPLES
    return found or FALLBACK_EXAMPLES


def _secret_key() -> bool:
    """Make an API key from Streamlit secrets visible to the library.

    Secrets are stored by the host, not in the repository. The key is read
    server-side and never reaches the browser.
    """
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key = ""
    if key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = key
    return bool(os.environ.get("OPENAI_API_KEY"))


def _generation_available() -> bool:
    """Whether the client generation would call is actually installed.

    A key alone is not enough. The openai package lives in an extra, so an
    environment can hold a valid key and no way to spend it — which offered a
    working-looking toggle and then failed at the moment of use.
    """
    return importlib.util.find_spec("openai") is not None


def _budget() -> GenerationBudget:
    """Limits, overridable per deployment without a code change."""
    try:
        per_session = int(st.secrets.get("GEN_PER_SESSION", DEFAULT_PER_SESSION))
        per_day = int(st.secrets.get("GEN_PER_DAY", DEFAULT_PER_DAY))
    except Exception:
        per_session, per_day = DEFAULT_PER_SESSION, DEFAULT_PER_DAY
    return GenerationBudget(per_session=per_session, per_day=per_day)


@st.cache_resource
def _shared_usage() -> dict:
    """Usage shared across visitors, so the daily cap is not per-session."""
    return {"usage": Usage(day=date.today())}


def find_corpus() -> Path | None:
    for directory in CORPUS_DIRS:
        found = next(iter(sorted(directory.glob("*.html"))), None)
        if found is not None:
            return found
    return None


def main() -> None:
    load_env()
    st.set_page_config(page_title="gdpr-rag", page_icon="§", layout="wide")
    st.title("§ gdpr-rag")
    st.caption(
        "Retrieval over the GDPR with article-level citations. "
        "Change the chunking strategy to see the ablation result for yourself."
    )

    corpus = find_corpus()
    if corpus is None:
        st.error(
            "No corpus found. Looked in "
            + " and ".join(f"`{d}`" for d in CORPUS_DIRS)
            + ". Save the EUR-Lex HTML into one of them (see the README) and reload."
        )
        st.stop()

    with st.sidebar:
        st.subheader("Configuration")
        strategy_label = st.radio("Chunking", list(STRATEGIES), index=0)
        strategy = STRATEGIES[strategy_label]
        # Defaults to the dense model: the hashed-unigram baseline retrieves
        # badly enough that a first-time visitor would judge the system by it.
        dense = st.toggle(
            "Local MiniLM embeddings",
            value=True,
            help=(
                "On: semantic matching (downloads ~90 MB once). "
                "Off: a hashed-unigram baseline — instant, and markedly worse. "
                "Turn it off to see why the README reports both."
            ),
        )
        k = st.slider("Results", min_value=1, max_value=10, value=5)

        has_key = _secret_key()
        has_client = _generation_available()
        budget = _budget()
        used = st.session_state.get("generations", 0)
        generate = st.toggle(
            "Generate a cited answer",
            value=False,
            disabled=not (has_key and has_client),
            help="Retrieval works without this, and is the part this project measures.",
        )
        if not has_key:
            st.caption("Set `OPENAI_API_KEY` to enable answer generation.")
        elif not has_client:
            st.caption("A key is set, but the `openai` client is not installed here.")
        elif generate:
            st.caption(
                f"{max(0, budget.per_session - used)} of {budget.per_session} left this visit."
            )

    retriever, chunk_count = build_retriever(str(corpus), strategy, "local" if dense else "hashing")
    st.sidebar.metric("Chunks in index", chunk_count)

    examples = _examples()
    question = st.text_input(
        "Ask about the GDPR", placeholder=examples[0] if examples else "Can I ask for my data?"
    )
    st.caption("Try: " + " · ".join(f"*{e}*" for e in examples[:3]))

    if not question.strip():
        st.stop()

    trace = Trace(question)
    results = retriever.retrieve(question, k=k, trace=trace)

    if generate:
        shared = _shared_usage()
        today = date.today()
        decision = budget.check(shared["usage"], st.session_state.get("generations", 0), today)
        if not decision.allowed:
            st.info(decision.reason)
            generate = False

    if generate:
        from gdpr_rag.llm import OpenAIModel

        # Counted before the call, not after: a failed request still costs a
        # slot, which is what stops a retry loop from spending without bound.
        shared["usage"] = budget.spend(shared["usage"], today)
        st.session_state["generations"] = st.session_state.get("generations", 0) + 1

        answer = None
        try:
            with st.spinner("Generating..."):
                answer = answer_question(question, results, OpenAIModel(), trace=trace)
        except Exception as exc:
            # Generation is the only stage that depends on somebody else's
            # service. Exhausted credit, a rate limit or a revoked key should
            # cost the answer and nothing more — the retrieved articles below
            # are already computed, and they are what this project measures.
            st.error(
                "Generation failed, so only the retrieved articles are shown. "
                f"({type(exc).__name__})"
            )

        if answer is not None:
            st.subheader("Answer")
            st.write(answer.text)
            if answer.unsupported_citations:
                st.error(
                    "Cited articles that retrieval did not return: "
                    + ", ".join(answer.unsupported_citations)
                    + ". These are unsupported by the evidence shown below."
                )
            elif answer.is_grounded:
                st.success("Every citation is backed by a retrieved article.")
            else:
                st.warning("The answer cites nothing, so no claim is grounded in the text.")
        st.divider()

    st.subheader("Retrieved articles")
    if not results:
        st.info("Nothing retrieved.")
        return

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
            "Fixed-size chunks carry no structural address, so the best citation "
            "available is the article number — never the specific paragraph or point. "
            "Switch to structured chunking to compare."
        )


if __name__ == "__main__":
    main()
