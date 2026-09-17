import streamlit as st
import sys
from pathlib import Path
from html import escape
from textwrap import dedent
import markdown

# ============================================================
# PATH SETUP
# ============================================================

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent import SelfGradingAgent

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Self-Grading RAG Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# HTML HELPER
# ============================================================

def render_html(html: str):
    """
    Render raw HTML safely through Streamlit's native HTML renderer.
    This prevents HTML tags from appearing as visible text.
    """
    st.html(dedent(html))


# ============================================================
# CUSTOM CSS
# ============================================================

render_html("""
<style>

/* ============================================================
   GLOBAL APP
   ============================================================ */

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(100, 50, 180, 0.18),
            transparent 35%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(0, 150, 220, 0.12),
            transparent 35%
        ),
        #030712;

    color: #ffffff;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}


/* ============================================================
   HEADER
   ============================================================ */

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 4px;

    background: linear-gradient(
        90deg,
        #a855f7,
        #3b82f6,
        #06b6d4,
        #22c55e
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #b8c7e0;
    font-size: 18px;
    margin-bottom: 8px;
}

.description {
    color: #8ea4c7;
    font-size: 14px;
    margin-bottom: 25px;
}

.badge {
    display: inline-block;

    border: 1px solid #245a8d;
    border-radius: 30px;

    padding: 10px 18px;

    color: #c9e6ff;
    background: rgba(5, 25, 45, 0.7);

    font-size: 13px;
}


/* ============================================================
   CARDS
   ============================================================ */

.card {
    background: linear-gradient(
        145deg,
        rgba(8, 24, 43, 0.95),
        rgba(3, 12, 25, 0.95)
    );

    border: 1px solid #17466f;
    border-radius: 16px;

    padding: 24px;
    margin-top: 18px;

    box-shadow:
        0 0 25px rgba(0, 100, 200, 0.08);
}

.card-title {
    font-size: 21px;
    font-weight: 700;

    color: white;

    margin-bottom: 5px;
}

.card-subtitle {
    font-size: 13px;

    color: #8da6c9;

    margin-bottom: 18px;
}


/* ============================================================
   ANSWER
   ============================================================ */

.answer-box {
    background: rgba(10, 35, 45, 0.65);

    border: 1px solid #1c765e;
    border-radius: 12px;

    padding: 20px;

    color: #e8fff8;

    font-size: 17px;
    line-height: 1.6;

    margin-top: 12px;
}

.answer-card {
    margin-top: 8px;
    padding: 18px;
}

.answer-card .answer-box {
    margin-top: 0;
}

/* ============================================================
   STATUS
   ============================================================ */

.supported {
    display: inline-block;

    padding: 7px 13px;

    border-radius: 20px;

    background: rgba(34, 197, 94, 0.12);

    border: 1px solid #25864b;

    color: #4ade80;

    font-size: 12px;
    font-weight: 700;
}

.unsupported {
    display: inline-block;

    padding: 7px 13px;

    border-radius: 20px;

    background: rgba(239, 68, 68, 0.10);

    border: 1px solid #8b3030;

    color: #f87171;

    font-size: 12px;
    font-weight: 700;
}


/* ============================================================
   METRICS
   ============================================================ */

.metric-card {
    background: #071525;

    border: 1px solid #1b4164;
    border-radius: 13px;

    padding: 18px;

    text-align: center;

    min-height: 105px;
}

.metric-label {
    color: #8da6c9;

    font-size: 13px;

    margin-bottom: 7px;
}

.metric-value {
    color: white;

    font-size: 27px;
    font-weight: 800;
}


/* ============================================================
   EVIDENCE
   ============================================================ */

.evidence {
    background: #071322;

    border: 1px solid #28486a;
    border-radius: 12px;

    padding: 17px;

    margin-top: 12px;
}

.source-name {
    color: #7dd3fc;

    font-weight: 700;
    font-size: 14px;

    margin-bottom: 10px;
}

.source-text {
    color: #c5d4e8;

    font-size: 14px;
    line-height: 1.6;

    white-space: pre-wrap;
}

.no-evidence {
    color: #7186a3;

    font-style: italic;
}


/* ============================================================
   BUTTON
   ============================================================ */

.stButton > button {
    width: 100%;
    height: 43px;

    border: none;
    border-radius: 10px;

    background: linear-gradient(
        90deg,
        #0ea5e9,
        #7c3aed
    );

    color: white;

    font-weight: 700;

    transition: 0.2s;
}

.stButton > button:hover {
    transform: translateY(-1px);

    box-shadow:
        0 0 18px rgba(99, 102, 241, 0.35);
}


/* ============================================================
   INPUT
   ============================================================ */

.stTextInput input {
    background: #06101e !important;

    color: white !important;

    border: 1px solid #315271 !important;
    border-radius: 10px !important;

    height: 43px !important;
}

.stTextInput input::placeholder {
    color: #64748b !important;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;

    color: #7185a5;

    font-size: 12px;

    margin-top: 35px;

    padding-top: 18px;

    border-top: 1px solid #14283e;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .main-title {
        font-size: 32px;
    }

    .subtitle {
        font-size: 16px;
    }

    .description {
        font-size: 13px;
    }

    .badge {
        margin-top: 10px;
    }
}

</style>
""")


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([3.5, 1.5])


with header_left:

    render_html("""
    <div class="main-title">
        🧠 Self-Grading RAG Agent
    </div>

    <div class="subtitle">
        Evidence-Grounded Enterprise Policy Assistant
    </div>

    <div class="description">
        Ask questions from company policy documents and get
        evidence-based answers with automatic self-evaluation.
    </div>
    """)


with header_right:

    render_html("""
    <div style="text-align:right; padding-top:10px;">
        <span class="badge">
            🛡️ RAG • Self-Grading • Reliable Answers
        </span>
    </div>
    """)


# ============================================================
# QUESTION CARD
# ============================================================

render_html("""
<div class="card">

    <div class="card-title">
        💬 Ask a Question
    </div>

    <div class="card-subtitle">
        Enter your question about company policies
    </div>

</div>
""")


# ============================================================
# QUESTION INPUT
# ============================================================

col1, col2 = st.columns([5, 1])


with col1:

    question = st.text_input(
        "Question",
        placeholder=(
            "e.g. How many paid annual leave days do "
            "full-time employees receive?"
        ),
        label_visibility="collapsed",
    )


with col2:

    ask = st.button(
        "🚀 Ask Agent",
        use_container_width=True,
    )


# ============================================================
# LOAD AGENT
# ============================================================

@st.cache_resource
def load_agent():

    return SelfGradingAgent()


# ============================================================
# INITIAL STATE
# ============================================================

sources = []


# ============================================================
# RUN AGENT
# ============================================================

if ask:

    if not question.strip():

        st.warning("Please enter a question.")
        st.stop()


    with st.spinner(
        "Retrieving evidence and evaluating the answer..."
    ):

        try:

            agent = load_agent()

            result = agent.run(
                question.strip()
            )

        except Exception as e:

            st.error(
                f"Application error: {e}"
            )

            st.stop()


    # ========================================================
    # RESULT VALUES
    # ========================================================

    def extract_source_info(item):
        """
        Extract filename and evidence text from
        different source formats.
        """

        if isinstance(item, str):

            return (
                "Retrieved source",
                item.strip()
            )


        if not isinstance(item, dict):

            return (
                "Retrieved source",
                str(item).strip()
            )


        metadata = item.get(
            "metadata",
            {}
        )


        if not isinstance(metadata, dict):

            metadata = {}


        # ----------------------------------------------------
        # SOURCE NAME
        # ----------------------------------------------------

        name = (
            item.get("source")
            or item.get("source_name")
            or item.get("filename")
            or item.get("file_name")
            or metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("filename")
            or "Unknown source"
        )


        # ----------------------------------------------------
        # SOURCE CONTENT
        # ----------------------------------------------------

        content = (
            item.get("text")
            or item.get("content")
            or item.get("page_content")
            or item.get("chunk")
            or item.get("evidence")
            or metadata.get("text")
            or metadata.get("content")
            or metadata.get("page_content")
            or ""
        )


        # ----------------------------------------------------
        # NESTED CONTENT
        # ----------------------------------------------------

        if isinstance(content, dict):

            content = (
                content.get("page_content")
                or content.get("content")
                or content.get("text")
                or ""
            )


        return (
            str(name).strip(),
            str(content).strip()
        )


    # ========================================================
    # ANSWER
    # ========================================================

    answer = result.get(
        "answer",
        "No answer available."
    )


    # ========================================================
    # STATUS
    # ========================================================

    status = result.get(
        "status",
        "UNKNOWN"
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = result.get(
        "confidence",
        "Unknown"
    )


    # ========================================================
    # OVERALL SCORE
    # ========================================================

    try:

        overall = float(
            result.get(
                "final_score",
                0.0
            )
        )

    except (TypeError, ValueError):

        overall = 0.0


    # ========================================================
    # SOURCES
    # ========================================================

    sources = result.get(
        "sources",
        []
    )


    if not isinstance(sources, list):

        sources = list(sources) if sources else []


    # ========================================================
    # GROUP SOURCES
    # ========================================================

    grouped_sources = {}


    for item in sources:

        source_name, source_text = (
            extract_source_info(item)
        )


        if not source_name:

            source_name = "Unknown source"


        if source_name not in grouped_sources:

            grouped_sources[source_name] = []


        if (
            source_text
            and source_text
            not in grouped_sources[source_name]
        ):

            grouped_sources[source_name].append(
                source_text
            )


    # Unique document count
    unique_source_count = len(
        grouped_sources
    )


# ============================================================
# ANSWER CARD + METRICS + RETRIEVED EVIDENCE
# ============================================================

    answer_col, status_col = st.columns([4, 1])

    with answer_col:
        render_html("""
        <div class="card-title">
            📄 Answer
        </div>
        """)

    with status_col:
        if status == "SUCCESS":
            render_html("""
            <div style="text-align:right;">
                <span class="supported">
                    ✓ SUPPORTED ANSWER
                </span>
            </div>
            """)
        else:
            safe_status = escape(str(status))
            render_html(f"""
            <div style="text-align:right;">
                <span class="unsupported">
                    {safe_status}
                </span>
            </div>
            """)

    # ------------------------------------------------------------
    # ANSWER CONTENT
    # ------------------------------------------------------------

    answer_html = markdown.markdown(
        str(answer),
        extensions=["extra", "nl2br"]
    )

    render_html(f"""
    <div class="card answer-card">
    <div class="answer-box">
        {answer_html}
    </div>
 </div>""")

    # ------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------

    m1, m2, m3 = st.columns(3)

    with m1:
        render_html(f"""
        <div class="metric-card">
            <div class="metric-label">
                ⭐ Overall Score
            </div>
            <div class="metric-value">
                {overall:.2f}
            </div>
        </div>
        """)

    with m2:
        safe_confidence = escape(str(confidence))

        render_html(f"""
        <div class="metric-card">
            <div class="metric-label">
                📊 Confidence
            </div>
            <div class="metric-value" style="font-size:20px; padding-top:5px;">
                {safe_confidence}
            </div>
        </div>
        """)

    with m3:
        render_html(f"""
        <div class="metric-card">
            <div class="metric-label">
                🔎 Sources
            </div>
            <div class="metric-value">
                {unique_source_count}
            </div>
        </div>
        """)

    # ------------------------------------------------------------
    # RETRIEVED EVIDENCE HEADER
    # ------------------------------------------------------------

    render_html(f"""
    <div class="card">
        <div class="card-title">
            📚 Retrieved Evidence
        </div>
        <div class="card-subtitle">
            Evidence used to generate and evaluate the answer
            • {unique_source_count} relevant source(s)
        </div>
    </div>
    """)

    # ------------------------------------------------------------
    # DISPLAY GROUPED SOURCES
    # ------------------------------------------------------------

    if grouped_sources:

        for source_name, chunks in grouped_sources.items():

            safe_source_name = escape(str(source_name))

            if chunks:
                evidence_html = "<br><br>".join(
                    escape(str(chunk)).replace("\n", "<br>")
                    for chunk in chunks
                )
            else:
                evidence_html = (
                    '<span class="no-evidence">'
                    "Evidence text was not returned by the retrieval layer."
                    "</span>"
                )

            render_html(f"""
            <div class="evidence">
                <div class="source-name">
                    📄 {safe_source_name}
                </div>
                <div class="source-text">
                    {evidence_html}
                </div>
            </div>
            """)

    else:
        st.info("No relevant evidence was retrieved.")


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="footer">
    Self-Grading RAG Agent
    &nbsp;|&nbsp;
    Built by Shaik Khaja Mainuddin

    <br><br>

    “Better Answers. Safer Decisions.”
</div>
""")
