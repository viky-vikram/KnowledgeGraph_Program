"""Streamlit interface for Tamil Nadu Agriculture Schemes RAG."""

from __future__ import annotations

import base64
from html import escape
import logging
from pathlib import Path
import subprocess
import sys

import streamlit as st

from config import load_config, validate_config
from rag_pipeline import (
    answer_question_stream,
    build_neo4j_graph,
    index_requires_rebuild,
    load_index_metadata,
    load_scheme_data,
)
from scraper import scrape_all_schemes, schemes_to_dataframe


def launch_with_streamlit() -> None:
    """Allow `python app.py` to launch the Streamlit app in a browser."""

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--server.headless=false",
        ],
        check=False,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Tamil Nadu Agriculture Schemes Assistant",
    page_icon="TN",
    layout="wide",
    initial_sidebar_state="auto",
)


def asset_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{path.suffix.lstrip('.').lower()};base64,{encoded}"


def inject_styles() -> None:
    hero_image = asset_data_uri(ROOT_DIR / "assets" / "agriculture-hero.png")
    st.markdown(
        f"""
        <style>
        :root {{
            --leaf: #176b45;
            --leaf-dark: #0e4f34;
            --field: #2f8b57;
            --gold: #b7791f;
            --sun: #f4c765;
            --ink: #10231f;
            --muted: #52665f;
            --panel: #ffffff;
            --mist: #f4faf6;
            --line: #cfe0d5;
            --soft-blue: #e8f2ff;
            --soft-yellow: #fff7d6;
        }}
        html, body, [class*="css"] {{
            color: var(--ink);
        }}
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(47, 139, 87, 0.14), transparent 32rem),
                radial-gradient(circle at bottom right, rgba(244, 199, 101, 0.15), transparent 30rem),
                linear-gradient(180deg, #f8fbf8 0%, #ffffff 44%, #f5faf6 100%);
            color: var(--ink);
        }}
        .block-container {{
            max-width: 1280px;
            padding-top: 2.3rem;
            padding-bottom: 3rem;
        }}
        .hero {{
            position: relative;
            overflow: hidden;
            min-height: 21rem;
            display: flex;
            align-items: flex-end;
            border: 1px solid rgba(207, 224, 213, .85);
            padding: 2rem;
            background:
                linear-gradient(90deg, rgba(6, 37, 25, .88) 0%, rgba(6, 37, 25, .72) 42%, rgba(6, 37, 25, .20) 100%),
                url("{hero_image}") center/cover no-repeat,
                linear-gradient(135deg, #1d6b43 0%, #e1f1df 100%);
            border-radius: 8px;
            box-shadow: 0 24px 55px rgba(16, 35, 31, 0.16);
            margin-bottom: 1.15rem;
        }}
        .hero-content {{
            width: min(47rem, 100%);
        }}
        .hero-kicker {{
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            padding: .38rem .65rem;
            border-radius: 999px;
            background: rgba(255,255,255,.16);
            border: 1px solid rgba(255,255,255,.28);
            color: #f8fff9;
            font-size: .8rem;
            font-weight: 800;
            letter-spacing: .02rem;
            margin-bottom: .85rem;
        }}
        .hero h1 {{
            margin: 0 0 .35rem 0;
            font-size: clamp(2rem, 5vw, 4.2rem);
            line-height: 1;
            color: #ffffff;
            letter-spacing: 0;
            text-shadow: 0 4px 24px rgba(0,0,0,.28);
        }}
        .hero p {{
            margin: 0;
            font-size: 1.02rem;
            color: rgba(255,255,255,.92);
            max-width: 44rem;
        }}
        .hero-actions {{
            display: flex;
            flex-wrap: wrap;
            gap: .55rem;
            margin-top: 1.05rem;
        }}
        .hero-chip {{
            padding: .48rem .7rem;
            border-radius: 8px;
            background: rgba(255,255,255,.92);
            color: var(--leaf-dark);
            font-size: .82rem;
            font-weight: 800;
            box-shadow: 0 8px 22px rgba(0,0,0,.12);
        }}
        .assist-panel {{
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            background: rgba(255,255,255,.78);
            box-shadow: 0 12px 34px rgba(16, 35, 31, .08);
            margin: .85rem 0 1rem 0;
        }}
        div[data-testid="stMetric"] {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: .9rem 1rem;
            box-shadow: 0 8px 22px rgba(16, 35, 31, 0.04);
        }}
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: var(--ink) !important;
        }}
        section[data-testid="stSidebar"] {{
            background: #eef7f0;
            border-right: 1px solid var(--line);
        }}
        section[data-testid="stSidebar"] * {{
            color: var(--ink) !important;
        }}
        section[data-testid="stSidebar"] .stCaptionContainer,
        section[data-testid="stSidebar"] small {{
            color: var(--muted) !important;
        }}
        .side-card {{
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: .7rem .8rem;
            margin: .4rem 0 .55rem 0;
            box-shadow: 0 8px 20px rgba(16, 35, 31, 0.04);
        }}
        .side-label {{
            display: block;
            color: var(--muted);
            font-size: .77rem;
            font-weight: 700;
            letter-spacing: .02rem;
            text-transform: uppercase;
            margin-bottom: .22rem;
        }}
        .side-value {{
            display: block;
            color: var(--ink);
            font-size: .95rem;
            font-weight: 700;
            overflow-wrap: anywhere;
        }}
        .status-row {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin: 1rem 0 1rem 0;
        }}
        .status-card {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            min-height: 6rem;
            box-shadow: 0 8px 22px rgba(16, 35, 31, 0.04);
        }}
        .status-card span {{
            display: block;
            color: var(--muted);
            font-size: .82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .02rem;
        }}
        .status-card strong {{
            display: block;
            color: var(--ink);
            font-size: clamp(1.1rem, 2vw, 1.75rem);
            line-height: 1.2;
            margin-top: .45rem;
            overflow-wrap: anywhere;
        }}
        .sidebar-divider {{
            height: 1px;
            background: var(--line);
            margin: .85rem 0 1rem 0;
        }}
        .compact-note {{
            color: var(--muted);
            font-size: .84rem;
            line-height: 1.35;
            margin: .35rem 0 .7rem 0;
        }}
        .example-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .65rem;
            margin: .45rem 0 1rem 0;
        }}
        .stButton > button, .stDownloadButton > button, a[data-testid="stLinkButton"] {{
            border-radius: 8px;
            min-height: 2.75rem;
            font-weight: 700;
            border: 1px solid #a8c9b2 !important;
            background: #ffffff !important;
            color: var(--leaf-dark) !important;
            box-shadow: 0 4px 10px rgba(16, 35, 31, 0.05);
        }}
        .stButton > button:hover, .stDownloadButton > button:hover,
        a[data-testid="stLinkButton"]:hover {{
            border-color: var(--leaf) !important;
            color: #ffffff !important;
            background: var(--leaf) !important;
        }}
        .stButton > button:disabled,
        .stButton > button:disabled:hover {{
            background: #eef4f0 !important;
            border-color: #c8dbd0 !important;
            color: #52665f !important;
            opacity: 1 !important;
            box-shadow: none !important;
        }}
        .stButton > button:disabled *,
        .stButton > button:disabled:hover * {{
            color: #52665f !important;
        }}
        .stButton > button[kind="primary"], button[kind="primary"] {{
            background: var(--leaf) !important;
            border-color: var(--leaf) !important;
            color: #ffffff !important;
        }}
        .stButton > button[kind="primary"] *,
        button[kind="primary"] * {{
            color: #ffffff !important;
        }}
        .stButton > button:hover *,
        .stDownloadButton > button:hover *,
        a[data-testid="stLinkButton"]:hover * {{
            color: #ffffff !important;
        }}
        .stAlert {{
            border-radius: 8px;
            border: 1px solid rgba(16, 35, 31, .08);
        }}
        .stAlert p, .stAlert div {{
            color: var(--ink) !important;
        }}
        div[data-testid="stForm"] {{
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 10px 24px rgba(16, 35, 31, 0.07);
            max-width: 1280px;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            padding: .55rem .7rem .65rem .7rem;
        }}
        div[data-testid="stForm"] input {{
            min-height: 2.9rem;
            color: var(--ink) !important;
            border-color: transparent !important;
            background: #f4faf6 !important;
        }}
        div[data-testid="stForm"] input::placeholder {{
            color: #6a7f77 !important;
            opacity: 1 !important;
        }}
        div[data-testid="stChatMessage"] {{
            width: 100%;
            max-width: 1280px;
            margin-left: auto;
            margin-right: auto;
        }}
        div[data-testid="stChatMessage"] > div {{
            max-width: 100%;
        }}
        div[data-testid="stChatMessageContent"] {{
            width: 100%;
            max-width: 100%;
            overflow-wrap: anywhere;
        }}
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
            max-width: 100%;
        }}
        div[data-testid="stChatMessageContent"] p,
        div[data-testid="stChatMessageContent"] li {{
            overflow-wrap: anywhere;
        }}
        div[data-baseweb="tab-list"] button {{
            color: var(--muted) !important;
            font-weight: 700;
        }}
        div[data-baseweb="tab-list"] button[aria-selected="true"] {{
            color: var(--leaf-dark) !important;
        }}
        @media (max-width: 900px) {{
            .status-row, .example-grid {{
                grid-template-columns: 1fr;
            }}
            .hero {{
                min-height: 24rem;
                padding: 1.35rem;
                background:
                    linear-gradient(180deg, rgba(6, 37, 25, .86) 0%, rgba(6, 37, 25, .66) 100%),
                    url("{hero_image}") center/cover no-repeat,
                    linear-gradient(135deg, #1d6b43 0%, #e1f1df 100%);
            }}
            .hero h1 {{
                font-size: 1.55rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults = {
        "messages": [],
        "last_refresh_result": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_data(show_spinner=False)
def cached_scheme_data(json_mtime: float | None) -> list[dict]:
    _ = json_mtime
    return load_scheme_data(load_config())


def get_mtime(path: Path) -> float | None:
    return path.stat().st_mtime if path.exists() else None


def _safe_error(config, exc: Exception, fallback: str) -> str:
    """Return a user-safe error message, hiding internals in production."""

    if config.is_production:
        return f"{fallback} Please check the server logs for details."
    return f"{fallback} ({exc})"


def admin_unlocked(config) -> bool:
    """Return True when administrative actions are permitted this session.

    The password is read from configuration (environment), never hard-coded.
    When ADMIN_ACTIONS_ENABLED is false, admin actions are always blocked.
    When enabled without a configured password (local development), actions are
    allowed; production validation requires a password in that case.
    """

    if not config.admin_actions_enabled:
        return False
    if not config.admin_password_configured:
        return True
    if st.session_state.get("admin_ok"):
        return True
    entered = st.text_input(
        "Administrator password",
        type="password",
        help="Required to run scraping, index rebuild, or data-replacing actions.",
    )
    if entered:
        if entered == config.admin_password:
            st.session_state.admin_ok = True
            return True
        st.error("Incorrect administrator password.")
    return False


def render_sidebar(config, schemes: list[dict], metadata: dict) -> int:
    with st.sidebar:
        chat_model = escape(config.openai_chat_model)
        embedding_model = escape(config.openai_embedding_model)
        tracing_status = escape(config.tracing_status)
        st.subheader("Source & Models")
        st.link_button("Open Tamil Nadu schemes page", config.source_url, use_container_width=True)
        st.markdown(
            f"""
            <div class="side-card">
                <span class="side-label">Chat model</span>
                <span class="side-value">{chat_model}</span>
            </div>
            <div class="side-card">
                <span class="side-label">Embedding model</span>
                <span class="side-value">{embedding_model}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Actions")
        is_admin = admin_unlocked(config)
        if not config.admin_actions_enabled:
            st.caption("Administrative controls are disabled (ADMIN_ACTIONS_ENABLED=false).")
        elif not is_admin:
            st.caption("Enter the administrator password to enable data controls.")

        confirm_refresh = st.checkbox(
            "Confirm refresh can replace local data",
            help="Existing data is only replaced after a successful non-empty scrape.",
            disabled=not is_admin,
        )
        if st.button(
            "Refresh Website Data",
            type="primary",
            use_container_width=True,
            disabled=not is_admin,
        ):
            if schemes and not confirm_refresh:
                st.warning("Confirm refresh before replacing the existing local dataset.")
            else:
                with st.spinner("Scraping the Tamil Nadu Government schemes page..."):
                    st.session_state.last_refresh_result = scrape_all_schemes(config)
                st.cache_data.clear()
                result = st.session_state.last_refresh_result
                if result["success"]:
                    st.success(f"Scraped {result['scheme_count']} schemes.")
                else:
                    st.error("Refresh failed. Existing data was preserved when available.")
                st.rerun()

        if st.button("Rebuild Knowledge Graph", use_container_width=True, disabled=not is_admin):
            with st.spinner("Creating embeddings and rebuilding the Neo4j Knowledge Graph..."):
                try:
                    new_metadata = build_neo4j_graph(schemes, config)
                    st.success(f"Indexed {new_metadata['chunk_count']} chunks into Neo4j.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Knowledge Graph rebuild failed.")
                    st.error(_safe_error(config, exc, "Knowledge Graph rebuild failed."))

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.subheader("Dataset Status")
        last_scraped = escape(schemes[0].get("scraped_at", "No data") if schemes else "No data")
        st.markdown(
            f"""
            <div class="side-card">
                <span class="side-label">Schemes scraped</span>
                <span class="side-value">{len(schemes)}</span>
            </div>
            <div class="side-card">
                <span class="side-label">Chunks indexed</span>
                <span class="side-value">{metadata.get("chunk_count", 0)}</span>
            </div>
            <p class="compact-note">Last scrape: {last_scraped}</p>
            <p class="compact-note">LangSmith tracing: {tracing_status}</p>
            """,
            unsafe_allow_html=True,
        )

        if index_requires_rebuild(schemes, config):
            st.warning("Knowledge Graph needs rebuild.")
        elif metadata:
            st.success("Neo4j Knowledge Graph ready.")
        else:
            st.info("No Knowledge Graph yet.")

        k = st.slider(
            "Retriever results",
            min_value=1,
            max_value=10,
            value=max(1, min(config.retriever_k, 10)),
            help="Number of scheme chunks retrieved for each answer.",
        )

    return k


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <div class="hero-kicker">Grounded farmer-scheme guidance</div>
                <h1>Tamil Nadu Agriculture Schemes Assistant</h1>
                <p>
                    Ask in English or Tamil and get source-grounded answers from
                    scraped Tamil Nadu Government agriculture scheme records.
                </p>
                <div class="hero-actions">
                    <span class="hero-chip">54 schemes indexed</span>
                    <span class="hero-chip">Tamil query support</span>
                    <span class="hero-chip">Streaming answers</span>
                    <span class="hero-chip">Official sources</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_example_questions(index_ready: bool) -> None:
    examples = [
        "What training schemes are available for farmers?",
        "விவசாயிகளுக்கான பயிற்சி திட்டங்கள் என்ன?",
        "Are there any schemes related to seed production?",
        "Which schemes provide subsidies?",
        "How can farmers apply for these schemes?",
    ]
    st.caption("Example questions")
    columns = st.columns(len(examples))
    for column, question in zip(columns, examples):
        if column.button(question, use_container_width=True, disabled=not index_ready):
            st.session_state.pending_question = question


def render_status_summary(config, schemes: list[dict], metadata: dict) -> None:
    tracing = escape(config.tracing_status)
    st.markdown(
        f"""
        <div class="status-row">
            <div class="status-card">
                <span>Schemes</span>
                <strong>{len(schemes)}</strong>
            </div>
            <div class="status-card">
                <span>Indexed chunks</span>
                <strong>{metadata.get("chunk_count", 0)}</strong>
            </div>
            <div class="status-card">
                <span>Tracing</span>
                <strong>{tracing}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_messages() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))


def render_chat_input(config, k: int, index_ready: bool) -> None:
    pending = st.session_state.pop("pending_question", None) if index_ready else None
    with st.form("chat_composer", clear_on_submit=True):
        text_column, button_column = st.columns([10, 1.15])
        with text_column:
            typed_prompt = st.text_input(
                "Ask about Tamil Nadu agriculture schemes",
                placeholder=(
                    "Ask in English or Tamil about eligibility, subsidies, training, "
                    "documents, or application steps..."
                ),
                disabled=not index_ready,
                max_chars=config.max_input_chars,
                label_visibility="collapsed",
            )
        with button_column:
            submitted = st.form_submit_button(
                "Ask",
                type="primary",
                disabled=not index_ready,
                use_container_width=True,
            )
    prompt = pending or (typed_prompt if submitted else None)
    if not prompt:
        return

    # Render the user's message immediately, then stream the assistant answer.
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.status("Finding matching scheme records...", expanded=True) as status:
            result = answer_question_stream(prompt, k=k, config=config)
            status.update(label="Generating a grounded answer...", state="running")
        stream = result.get("stream")
        if stream is None:
            answer_text = result.get("answer", "")
            st.markdown(answer_text)
        else:
            # st.write_stream renders tokens incrementally and returns the full text.
            answer_text = st.write_stream(stream)
        status.update(label="Answer ready", state="complete", expanded=False)
        sources = result.get("sources", [])
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "sources": sources}
    )
    # Bound chat history to avoid unbounded session growth.
    max_messages = max(2, config.max_history_messages)
    if len(st.session_state.messages) > max_messages:
        st.session_state.messages = st.session_state.messages[-max_messages:]
    st.rerun()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Retrieved sources", expanded=False):
        seen = set()
        for source in sources:
            key = (source.get("scheme_name"), source.get("source_url"))
            if key in seen:
                continue
            seen.add(key)
            st.markdown(f"**{source.get('scheme_name', 'Unknown scheme')}**")
            if source.get("source_url"):
                st.markdown(f"[Official source]({source['source_url']})")
            st.caption(source.get("retrieved_text", "")[:900])
            st.divider()


def render_data_browser(config, schemes: list[dict]) -> None:
    st.subheader("Browse Scraped Schemes")
    if not schemes:
        st.info("No local scheme data yet. Use Refresh Website Data in the sidebar.")
        return

    frame = schemes_to_dataframe(schemes)
    visible_columns = [
        "scheme_name",
        "category",
        "description",
        "benefits",
        "eligibility",
        "scheme_detail_url",
    ]
    st.dataframe(
        frame[[column for column in visible_columns if column in frame.columns]],
        use_container_width=True,
        hide_index=True,
    )

    if config.schemes_csv_path.exists():
        st.download_button(
            "Download scraped CSV",
            data=config.schemes_csv_path.read_bytes(),
            file_name="tn_agriculture_schemes.csv",
            mime="text/csv",
            use_container_width=False,
        )

    with st.expander("Full extracted records", expanded=False):
        st.dataframe(frame, use_container_width=True, hide_index=True)


def render_validation_messages(errors: list[str], schemes: list[dict]) -> None:
    if errors:
        for error in errors:
            st.warning(error)
    if not schemes:
        st.info(
            "Start by adding your keys in `.env`, then use Refresh Website Data. "
            "After data exists, rebuild the Knowledge Graph."
        )


def main() -> None:
    inject_styles()
    initialize_state()
    config = load_config()
    schemes = cached_scheme_data(get_mtime(config.schemes_json_path))
    metadata = load_index_metadata(config)
    errors = validate_config(config, require_openai=False)

    render_header()

    k = render_sidebar(config, schemes, metadata)
    render_validation_messages(errors, schemes)

    index_ready = bool(schemes) and not index_requires_rebuild(schemes, config)

    render_chat_messages()

    if not index_ready:
        st.warning("The chat box is disabled until a valid Neo4j Knowledge Graph is available.")
    render_chat_input(config, k, index_ready=index_ready)
    render_example_questions(index_ready)


if __name__ == "__main__":
    if hasattr(st, "runtime") and st.runtime.exists():
        main()
    else:
        launch_with_streamlit()
