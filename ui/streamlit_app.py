"""Streamlit UI for Agentic Wealth Intelligence System."""

import logging
import os
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

# Structured logging to stdout so output is visible in Render log viewer
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("wealth_ui")

# Suppress noisy third-party warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*tf\\..*is deprecated.*")
warnings.filterwarnings("ignore", message=".*torch\\.classes.*")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file (fallback to .env.example)
from dotenv import load_dotenv
env_file = project_root / ".env"
if not env_file.exists():
    env_file = project_root / ".env.example"
load_dotenv(env_file)

from backend.src.common.tracing import configure_tracing

configure_tracing()

import streamlit as st
import streamlit.components.v1 as st_components

from backend.src.agents.asset_agent import AssetAgent
from backend.src.agents.orchestrator import WealthOrchestrator
from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.asset_management.price_updater import PriceUpdater
from backend.src.asset_management.visualization import VisualizationEngine
from backend.src.common.llm_client import LLMClient, LLMProvider, get_available_providers
from backend.src.profiling import ProfilingAgent, ConversationState
from backend.src.multi_agent.nodes import conv_state_to_investment_profile
from backend.src.recommendation.rag import RAGInitializer
from backend.src.recommendation.collectors import DataCollectionManager

# Register PlaidToken on Base.metadata before any DB engine is created
try:
    import backend.src.plaid.token_store as _plaid_token_store_module  # noqa: F401
    _PLAID_AVAILABLE = True
except ImportError:
    _PLAID_AVAILABLE = False

# Import custom styles
from styles import (
    MAIN_CSS,
    ICONS,
    get_icon,
    icon_html,
    render_header,
    render_metric_card,
    render_section_title,
    render_status_badge,
    render_guidance_callout,
    render_welcome_banner,
)


# Page config
st.set_page_config(
    page_title="Wealth Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown(MAIN_CSS, unsafe_allow_html=True)



@st.cache_resource
def get_ui_log_repository():
    """Get a cached LogRepository for the Streamlit UI."""
    from backend.src.logging_db import LogRepository, get_engine, get_session_factory

    engine = get_engine()
    factory = get_session_factory(engine)
    return LogRepository(factory)


@st.cache_resource
def load_data():
    """Load and cache portfolio data.

    Also registers a PlaidPortfolioSource when PLAID_CLIENT_ID and
    PLAID_SECRET are present in the environment.
    """
    data_path = project_root / "data" / "synthetic" / "synthetic_portfolios.json"
    aggregator = PortfolioAggregator(data_path)
    aggregator.load_data()

    # Attach live Plaid source if credentials are available
    if _PLAID_AVAILABLE and os.environ.get("PLAID_CLIENT_ID") and os.environ.get("PLAID_SECRET"):
        try:
            plaid_source = _build_plaid_source()
            aggregator.register_plaid_source(plaid_source)
        except Exception as _exc:
            logger.warning("Could not register Plaid source in UI: %s", _exc)

    return aggregator


def _build_plaid_source():
    """Create a PlaidPortfolioSource wired to the local SQLite DB."""
    from backend.src.logging_db import get_engine, get_session_factory
    from backend.src.plaid.client import PlaidClient
    from backend.src.plaid.adapter import PlaidPortfolioAdapter
    from backend.src.plaid.source import PlaidPortfolioSource
    from backend.src.plaid.token_store import PlaidTokenRepository

    engine = get_engine()
    factory = get_session_factory(engine)
    repo = PlaidTokenRepository(factory)
    env = os.environ.get("PLAID_ENV", "sandbox")
    client = PlaidClient(
        client_id=os.environ["PLAID_CLIENT_ID"],
        secret=os.environ["PLAID_SECRET"],
        env=env,
    )
    return PlaidPortfolioSource(
        plaid_client=client,
        adapter=PlaidPortfolioAdapter(),
        token_repo=repo,
        current_env=env,
    )


# RAG state — events reset on each process restart
_rag_ready = threading.Event()
_rag_thread_started = threading.Event()
_rag_index_result: dict = {}

# Reindex state
_reindex_event = threading.Event()
_reindex_result: dict = {}

def _run_rag_init_background(rag: "RAGInitializer") -> None:
    global _rag_index_result
    logger.info("RAG indexing started in background thread")
    try:
        _rag_index_result = rag.initialize_with_sample_documents()
        logger.info("RAG indexing complete: %s", _rag_index_result)
    except Exception as e:
        logger.exception("RAG indexing failed")
        _rag_index_result = {"action": "failed", "error": str(e)}
    finally:
        _rag_ready.set()


def _run_reindex_background(rag: "RAGInitializer") -> None:
    global _reindex_result
    try:
        _reindex_result = rag.initialize_with_sample_documents(force_reindex=True)
        logger.info("Force reindex complete: %s", _reindex_result)
    except Exception as e:
        logger.exception("Force reindex failed")
        _reindex_result = {"action": "failed", "error": str(e)}
    finally:
        _reindex_event.set()


@st.cache_resource
def initialize_rag_system():
    """Create RAG objects at startup — no heavy work, no background thread yet."""
    logger.info("Initializing RAG system objects")
    persist_dir = str(project_root / "data" / "chroma")
    rag = RAGInitializer(persist_directory=persist_dir)
    collection_manager = DataCollectionManager(rag_initializer=rag)
    return rag, collection_manager


def ensure_rag_indexed() -> dict:
    """Return immediately if pre-built ChromaDB exists; otherwise start background thread."""
    global _rag_index_result

    # Fast path: pre-built index on disk — mark ready instantly, no thread needed
    chroma_db = project_root / "data" / "chroma" / "chroma.sqlite3"
    if chroma_db.exists() and not _rag_thread_started.is_set():
        _rag_thread_started.set()
        _rag_index_result = {"action": "skipped", "already_initialized": True,
                             "documents_indexed": 0, "chunks_created": 0}
        _rag_ready.set()
        logger.info("RAG index already exists, skipping background thread")

    # Slow path: no pre-built index found — start background thread
    elif not _rag_thread_started.is_set():
        _rag_thread_started.set()
        rag, _ = initialize_rag_system()
        logger.info("Starting RAG background thread (no pre-built index found)")
        t = threading.Thread(target=_run_rag_init_background, args=(rag,), daemon=True)
        t.start()

    if not _rag_ready.is_set():
        st.info("Initializing knowledge base... this may take up to 30 seconds on first load.")
        time.sleep(2)
        st.rerun()
    return _rag_index_result


def create_agent(aggregator: PortfolioAggregator, provider: LLMProvider) -> AssetAgent:
    """Create agent with specified LLM provider."""
    llm_client = LLMClient(provider=provider)
    return AssetAgent(aggregator, llm_client=llm_client)


def render_sidebar(aggregator: PortfolioAggregator):
    """Render sidebar with user selection and info."""
    st.sidebar.markdown(
        f'{render_section_title("Select User", "user")}',
        unsafe_allow_html=True
    )

    user_ids = aggregator.get_user_ids()

    # Collect live Plaid user_ids to correctly label custom-named Plaid users
    _live_ids: set[str] = set()
    if aggregator._plaid_source is not None:
        try:
            _live_ids = set(aggregator._plaid_source.get_user_ids())
        except Exception:
            pass

    def _format_user_id(uid: str) -> str:
        if uid in _live_ids or uid.startswith("plaid_"):
            return f"[Live] {uid}"
        return uid

    selected_user = st.sidebar.selectbox(
        "Choose a user portfolio:",
        user_ids,
        format_func=_format_user_id,
    )

    # Show user info
    if selected_user:
        portfolio = aggregator.get_portfolio(selected_user)
        if portfolio:
            st.sidebar.markdown("---")
            st.sidebar.markdown(
                f'{render_section_title("User Info", "user")}',
                unsafe_allow_html=True
            )
            st.sidebar.write(f"**Name:** {portfolio.user.name}")
            st.sidebar.write(f"**Age:** {portfolio.user.age}")
            st.sidebar.write(f"**Occupation:** {portfolio.user.occupation}")
            st.sidebar.write(f"**Annual Income:** ${portfolio.user.annual_income:,.0f}")

            if portfolio.summary:
                st.sidebar.markdown("---")
                st.sidebar.markdown(
                    f'{render_section_title("Quick Stats", "dollar-sign")}',
                    unsafe_allow_html=True
                )
                st.sidebar.metric(
                    "Net Worth",
                    f"${portfolio.summary.total_net_worth:,.0f}",
                )
                st.sidebar.metric(
                    "Liquidity Ratio",
                    f"{portfolio.summary.liquidity_ratio:.1%}",
                )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'{render_section_title("Example Queries", "book-open")}',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        '<p style="font-size: 0.95rem; color: #475569; font-weight: 500;">Try these example questions:</p>',
        unsafe_allow_html=True,
    )

    examples = [
        ("What's my net worth?", "Total assets minus liabilities"),
        ("Show my asset allocation", "Breakdown by asset type"),
        ("Show my asset allocation excluding real estate", "Focus on financial assets"),
        ("What's my liquidity ratio?", "How much cash is available"),
        ("Show my top holdings", "Largest positions by value"),
        ("What are my gains and losses?", "Unrealized profit & loss"),
    ]

    for ex, desc in examples:
        if st.sidebar.button(ex, key=f"ex_{ex}", help=desc):
            st.session_state.selected_query = ex

    # API keys — enter any one to use the app
    st.sidebar.markdown("---")
    with st.sidebar.expander("API Keys", expanded=not bool(
        os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    )):
        st.caption("Enter at least one API key. Keys are not stored beyond this session.")

        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            key="sidebar_anthropic_key",
            help="Enables Claude. Get a key at console.anthropic.com",
        )
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key

        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            key="sidebar_openai_key",
            help="Enables GPT-4o. Get a key at platform.openai.com",
        )
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            key="sidebar_gemini_key",
            help="Enables Gemini 2.0 Flash. Get a key at aistudio.google.com",
        )
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key

        if anthropic_key or openai_key or gemini_key:
            st.success("Key(s) active for this session.")

    # Admin login section
    st.sidebar.markdown("---")
    if not st.session_state.get("is_admin"):
        with st.sidebar.expander("Admin Login"):
            admin_pwd = st.text_input(
                "Admin Password",
                type="password",
                key="sidebar_admin_password",
            )
            if st.button("Login as Admin", key="sidebar_admin_login"):
                expected = os.environ.get("ADMIN_PASSWORD", "admin")
                if admin_pwd == expected:
                    st.session_state["is_admin"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
    else:
        st.sidebar.markdown(
            '<p style="font-size: 0.85rem; color: #16a34a;">Admin session active</p>',
            unsafe_allow_html=True,
        )
        if st.sidebar.button("Logout Admin", key="sidebar_admin_logout"):
            st.session_state["is_admin"] = False
            st.rerun()

    return selected_user


def render_dashboard(aggregator: PortfolioAggregator, user_id: str):
    """Render the main dashboard view."""
    portfolio = aggregator.get_portfolio(user_id)
    if not portfolio or not portfolio.summary:
        st.warning("No portfolio data available.")
        return

    summary = portfolio.summary
    viz = VisualizationEngine()

    # Metrics row with styled cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            render_metric_card(
                "Net Worth",
                f"${summary.total_net_worth:,.0f}",
                icon="dollar-sign"
            ),
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            render_metric_card(
                "Total Assets",
                f"${summary.total_assets:,.0f}",
                icon="briefcase"
            ),
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            render_metric_card(
                "Liabilities",
                f"${summary.total_liabilities:,.0f}",
                icon="credit-card"
            ),
            unsafe_allow_html=True
        )

    with col4:
        savings_rate = summary.monthly_savings_rate or 0
        st.markdown(
            render_metric_card(
                "Savings Rate",
                f"{savings_rate:.1%}",
                icon="percent"
            ),
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f'{render_section_title("Asset Allocation", "pie-chart")}',
            unsafe_allow_html=True
        )
        fig = viz.create_asset_allocation_pie(summary.allocation_by_asset_type)
        st.plotly_chart(fig, width='stretch', key="dash_allocation_pie")

    with col2:
        st.markdown(
            f'{render_section_title("Net Worth Breakdown", "activity")}',
            unsafe_allow_html=True
        )
        fig = viz.create_net_worth_breakdown(portfolio)
        st.plotly_chart(fig, width='stretch', key="dash_net_worth")

    # Holdings section
    st.markdown("---")
    st.markdown(
        f'{render_section_title("Top Holdings", "bar-chart-2")}',
        unsafe_allow_html=True
    )

    top_holdings = aggregator.get_top_holdings(user_id, 10)
    fig = viz.create_holdings_bar(top_holdings, by="value", title="Top 10 Holdings by Value")
    st.plotly_chart(fig, width='stretch', key="dash_top_holdings")


def render_chat_interface(aggregator: PortfolioAggregator, user_id: str, available_providers: list):
    """Render the chat interface for querying."""
    st.markdown(
        f'{render_section_title("Ask About Your Portfolio", "message-circle")}',
        unsafe_allow_html=True
    )
    st.markdown(
        render_guidance_callout(
            "Type a question in natural language below. The AI can analyze your holdings, "
            "calculate metrics, and generate charts from your portfolio data.",
            "info",
        ),
        unsafe_allow_html=True,
    )

    # Check for LLM providers
    if not available_providers:
        st.warning("No LLM providers configured. Please enter an API key in the sidebar (Anthropic, OpenAI, or Gemini).")
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Check for selected query from sidebar buttons
    default_query = ""
    if "selected_query" in st.session_state and st.session_state.selected_query:
        default_query = st.session_state.selected_query

    # === QUERY INPUT AT THE TOP ===
    with st.form(key="query_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            query = st.text_input(
                "Ask a question",
                value=default_query,
                placeholder="e.g., What's my net worth?",
                label_visibility="collapsed",
            )
        with col2:
            provider_names = [p["name"] for p in available_providers]
            selected_idx = st.selectbox(
                "Model",
                range(len(provider_names)),
                format_func=lambda i: provider_names[i],
                key="llm_provider",
                label_visibility="collapsed",
            )
        with col3:
            submit = st.form_submit_button("Send", type="primary")

    selected_provider = available_providers[selected_idx]["provider"]
    selected_model = available_providers[selected_idx]["model"]

    # Create agent with selected provider
    try:
        agent = create_agent(aggregator, selected_provider)
    except Exception as e:
        st.error(f"Failed to initialize {provider_names[selected_idx]}: {e}")
        return

    # Placeholder for analyzing message (appears below form)
    analyzing_placeholder = st.empty()

    # Process query if submitted
    if submit and query and query.strip():
        # Clear selected query if it was used
        if "selected_query" in st.session_state:
            st.session_state.selected_query = ""

        # Show currently analyzing query prominently
        analyzing_placeholder.info(f"**Analyzing:** {query}")

        # Get response
        logger.info("Portfolio query: user=%s model=%s query=%r", user_id, selected_model, query[:80])
        with st.spinner(f"Processing with {selected_model}..."):
            try:
                response = agent.process(user_id, query)
                logger.info("Portfolio query complete")
            except Exception as e:
                analyzing_placeholder.empty()
                logger.exception("agent.process failed")
                st.error(f"Error processing query: {e}")
                st.session_state["last_ui_error"] = str(e)
                return

        # Clear the analyzing message
        analyzing_placeholder.empty()

        # Store messages (as a pair for display)
        st.session_state.messages.append({
            "query": query,
            "answer": response.answer,
            "visualization": response.visualization,
        })

    st.markdown("---")

    # === CHAT HISTORY - NEWEST FIRST ===
    if st.session_state.messages:
        # Display in reverse order (newest at top)
        for i, message in enumerate(reversed(st.session_state.messages)):
            # Show "Latest" badge for the most recent
            if i == 0:
                st.markdown(
                    f'{render_status_badge("Latest Result", "success")}',
                    unsafe_allow_html=True
                )

            st.markdown(f"**You:** {message['query']}")
            st.markdown(f"**Assistant:** {message['answer']}")
            if message.get("visualization"):
                st.plotly_chart(message["visualization"], width='stretch', key=f"chat_viz_{i}")
            st.markdown("---")
    else:
        st.markdown(
            '<div style="text-align: center; padding: 28px 20px; background: #f8fafc; '
            'border-radius: 12px; border: 2px dashed #cbd5e1;">'
            '<div style="font-size: 36px; margin-bottom: 10px;">&#128172;</div>'
            '<div style="font-size: 1.1rem; font-weight: 600; color: #334155; margin-bottom: 6px;">'
            'No conversations yet</div>'
            '<div style="font-size: 0.95rem; color: #64748b;">'
            'Type a question in the box above, or click an example query from the sidebar.</div>'
            '</div>',
            unsafe_allow_html=True,
        )


def render_holdings_view(aggregator: PortfolioAggregator, user_id: str):
    """Render detailed holdings view."""
    portfolio = aggregator.get_portfolio(user_id)
    if not portfolio:
        st.warning("No portfolio data available.")
        return

    viz = VisualizationEngine()

    st.markdown(
        f'{render_section_title("Holdings by Sector", "pie-chart")}',
        unsafe_allow_html=True
    )

    # Sector bar chart
    fig = viz.create_holdings_by_sector(portfolio.holdings)
    st.plotly_chart(fig, width='stretch', key="portfolio_sector")

    st.markdown("---")

    # Gains/Losses
    st.markdown(
        f'{render_section_title("Unrealized Gains & Losses", "trending-up")}',
        unsafe_allow_html=True
    )
    fig = viz.create_gain_loss_chart(portfolio.holdings)
    st.plotly_chart(fig, width='stretch', key="portfolio_gain_loss")

    st.markdown("---")

    # Holdings table
    st.markdown(
        f'{render_section_title("All Holdings", "file-text")}',
        unsafe_allow_html=True
    )

    holdings_data = []
    for h in portfolio.holdings:
        holdings_data.append(
            {
                "Symbol": h.symbol,
                "Name": h.name,
                "Type": h.asset_type.value.title(),
                "Sector": h.sector or "N/A",
                "Quantity": f"{h.quantity:,.4f}",
                "Avg Cost": f"${h.average_cost:,.2f}",
                "Current Price": f"${h.current_price:,.2f}",
                "Market Value": f"${h.market_value:,.2f}",
                "Gain/Loss": f"${h.unrealized_gain_loss:+,.2f}",
                "Gain/Loss %": f"{h.unrealized_gain_loss_percent:+.2f}%",
            }
        )

    st.dataframe(holdings_data, width='stretch')


def _yf_symbol(symbol: str, asset_type) -> str:
    """Return the Yahoo Finance ticker for a given symbol and asset type."""
    from backend.src.common.models import AssetType
    if asset_type == AssetType.CRYPTO:
        return f"{symbol.upper()}-USD"
    return symbol.upper()


def render_live_data_section(aggregator: PortfolioAggregator, user_id: str):
    """Render live market quotes using yfinance (no API key required)."""
    import yfinance as yf

    st.markdown(
        f'{render_section_title("Live Market Data", "activity")}',
        unsafe_allow_html=True
    )

    portfolio = aggregator.get_portfolio(user_id)
    if not portfolio or not portfolio.holdings:
        st.info("No holdings found for live quotes.")
        return

    # Top 5 holdings by market value
    top_holdings = sorted(portfolio.holdings, key=lambda h: h.market_value, reverse=True)[:5]
    top_symbols = [h.symbol for h in top_holdings]
    st.markdown(f"**Your Top Holdings:** {', '.join(top_symbols)}")

    if st.button("Refresh Live Quotes", key="refresh_quotes_btn", type="primary"):
        with st.spinner("Fetching live quotes..."):
            quotes: dict = {}
            for holding in top_holdings:
                sym = holding.symbol
                yf_sym = _yf_symbol(sym, holding.asset_type)
                try:
                    hist = yf.Ticker(yf_sym).history(period="2d")
                    if hist.empty or len(hist) < 1:
                        quotes[sym] = {"error": "No data"}
                        continue
                    price = float(hist["Close"].iloc[-1])
                    change_pct = None
                    if len(hist) >= 2:
                        prev = float(hist["Close"].iloc[-2])
                        if prev > 0:
                            change_pct = (price - prev) / prev * 100
                    quotes[sym] = {"price": price, "change_percent": change_pct}
                except Exception as e:
                    quotes[sym] = {"error": str(e)}

            st.session_state.live_quotes = quotes
            st.session_state.live_quotes_time = datetime.now()

    if "live_quotes" in st.session_state and st.session_state.live_quotes:
        quotes = st.session_state.live_quotes
        quote_time = st.session_state.get("live_quotes_time", datetime.now())
        st.caption(f"Last updated: {quote_time.strftime('%Y-%m-%d %H:%M:%S')}")

        cols = st.columns(min(len(quotes), 5))
        for i, (symbol, quote) in enumerate(quotes.items()):
            with cols[i % 5]:
                if "error" in quote:
                    st.metric(symbol, "N/A", help=quote["error"])
                else:
                    price = quote.get("price")
                    change_pct = quote.get("change_percent")
                    delta = f"{change_pct:+.2f}%" if change_pct is not None else None
                    st.metric(
                        symbol,
                        f"${price:,.2f}" if price is not None else "N/A",
                        delta=delta,
                        delta_color="normal" if delta else "off",
                    )


def render_knowledge_search(rag: RAGInitializer, rag_init_result: dict, collection_manager: DataCollectionManager, aggregator: PortfolioAggregator = None, user_id: str = None):
    """Render the RAG-based knowledge search interface."""
    rag_init_result = ensure_rag_indexed()

    st.markdown(
        f'{render_section_title("Financial Knowledge Search", "book-open")}',
        unsafe_allow_html=True
    )
    st.markdown(
        render_guidance_callout(
            "<strong>Search the financial knowledge base.</strong> "
            "Find ETF fact sheets, FOMC meeting summaries, and investment guides "
            "powered by semantic search. Type a question or topic below.",
            "info",
        ),
        unsafe_allow_html=True,
    )

    # --- Price Data Section ---
    if aggregator:
        st.markdown("---")
        st.markdown(
            f'{render_section_title("Price Data", "activity")}',
            unsafe_allow_html=True,
        )

        updater = PriceUpdater(aggregator)

        col_status, col_btn = st.columns([3, 1])
        with col_btn:
            do_update = st.button("Update Prices", key="update_prices_btn", type="primary")

        if do_update:
            progress_bar = st.progress(0, text="Starting...")

            def _progress(current: int, total: int, symbol: str) -> None:
                pct = current / total if total else 1.0
                label = "Saving..." if symbol == "Saving..." else f"Fetching {symbol} ({current}/{total})"
                progress_bar.progress(min(pct, 1.0), text=label)

            try:
                result = updater.update_all_prices(progress_callback=_progress)
                progress_bar.empty()
                n_ok = len(result["updated"])
                n_fail = len(result["failed"])
                parts = []
                if n_ok:
                    parts.append(f"Updated {n_ok} symbols")
                if n_fail:
                    parts.append(f"Failed: {', '.join(result['failed'])}")
                if n_ok:
                    st.success(". ".join(parts) + ".")
                elif n_fail:
                    st.warning(". ".join(parts) + ".")
            except Exception as exc:
                progress_bar.empty()
                st.error(f"Update failed: {exc}")

        # Staleness indicator — rendered after any update so it reflects current state
        with col_status:
            oldest = updater.get_oldest_update()
            if oldest:
                age_str = oldest.strftime("%Y-%m-%d %H:%M")
                if updater.is_stale():
                    st.warning(f"Prices are stale (oldest: {age_str})")
                else:
                    st.caption(f"Last price update: {age_str}")
            else:
                st.caption("No price data yet.")

        # Price history viewer
        symbols = updater.get_unique_symbols()
        if symbols:
            with st.expander("Price History", expanded=False):
                selected_sym = st.selectbox(
                    "Symbol",
                    symbols,
                    key="price_history_symbol",
                )
                history = updater.get_price_history(selected_sym)
                if not history:
                    st.info("No history yet. Click 'Update Prices' to start tracking.")
                else:
                    import pandas as pd
                    df = pd.DataFrame(history)[["date", "price"]].rename(
                        columns={"date": "Date", "price": "Close Price"}
                    )
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.sort_values("Date")
                    st.line_chart(df.set_index("Date")["Close Price"])
                    st.dataframe(
                        df.sort_values("Date", ascending=False).reset_index(drop=True),
                        width='stretch',
                        hide_index=True,
                    )

        st.markdown("---")

    # Live Data Section (if aggregator and user_id provided)
    if aggregator and user_id:
        render_live_data_section(aggregator, user_id)
        st.markdown("---")

    # Show RAG system status
    with st.expander("Knowledge Base Status", expanded=False):
        stats = rag.get_stats()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents Indexed", stats["unique_documents"])
        with col2:
            st.metric("Total Chunks", stats["total_chunks"])
        with col3:
            status = "Ready" if stats["total_chunks"] > 0 else "Empty"
            st.metric("Status", status)

        if rag_init_result["action"] == "initialized":
            st.success(f"Indexed {rag_init_result['documents_indexed']} documents on startup.")
        elif rag_init_result["action"] == "skipped":
            st.info("Using existing indexed documents.")

        # Reindex button — runs in background thread to avoid blocking WebSocket
        reindexing = st.session_state.get("reindexing", False)
        if reindexing:
            if not _reindex_event.is_set():
                st.info("Reindexing documents... please wait.")
                time.sleep(2)
                st.rerun()
            else:
                r = _reindex_result
                if r.get("action") == "failed":
                    st.error(f"Reindex failed: {r.get('error')}")
                else:
                    st.success(f"Reindexed {r['documents_indexed']} documents ({r['chunks_created']} chunks)")
                st.session_state["reindexing"] = False
                _reindex_event.clear()

        if not reindexing and st.button("Reindex Sample Documents", key="reindex_btn"):
            _reindex_event.clear()
            threading.Thread(target=_run_reindex_background, args=(rag,), daemon=True).start()
            st.session_state["reindexing"] = True
            st.rerun()

    # Data Collection Section
    with st.expander("Collect Real Financial Data", expanded=False):
        st.markdown("Collect real-time data from financial APIs and add to the knowledge base.")

        # Show collector status
        collector_status = collection_manager.get_collector_status()
        st.markdown("**API Status:**")

        status_cols = st.columns(3)
        for i, (name, status) in enumerate(collector_status.items()):
            with status_cols[i % 3]:
                if status["configured"]:
                    st.markdown(
                        f'{render_status_badge(status["description"].split(" from ")[0], "success")}',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'{render_status_badge(f"{name} - Not configured", "warning")}',
                        unsafe_allow_html=True
                    )

        configured = collection_manager.get_configured_collectors()
        if not configured:
            st.warning(
                "No data collectors configured. Add API keys to your `.env` file:\n"
                "- `ALPHA_VANTAGE_API_KEY` - Financial news & ETF data\n"
                "- `SEC_USER_AGENT` - SEC filings (format: 'Name email@domain.com')\n"
                "- `NEWS_API_KEY` - Business news"
            )
        else:
            st.markdown("---")
            st.markdown("**Collect Data:**")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Collect News", key="collect_news_btn",
                           disabled="alpha_vantage" not in configured and "news_api" not in configured):
                    with st.spinner("Collecting financial news..."):
                        result = collection_manager.collect_news()
                        if result.is_success:
                            st.success(f"Collected {result.total_documents} articles, indexed {result.total_indexed} chunks")
                        else:
                            st.error(f"Collection failed: {result.errors}")

            with col2:
                if st.button("Collect ETF Data", key="collect_etf_btn"):
                    with st.spinner("Collecting ETF data..."):
                        result = collection_manager.collect_etf_data()
                        if result.is_success:
                            st.success(f"Collected {result.total_documents} ETF profiles, indexed {result.total_indexed} chunks")
                        else:
                            st.error(f"Collection failed: {result.errors}")

            with col3:
                if st.button("Collect SEC Filings", key="collect_sec_btn",
                           disabled="sec_edgar" not in configured):
                    with st.spinner("Collecting SEC filings..."):
                        result = collection_manager.collect_sec_filings()
                        if result.is_success:
                            st.success(f"Collected {result.total_documents} filings, indexed {result.total_indexed} chunks")
                        else:
                            st.error(f"Collection failed: {result.errors}")

            # Collect all button
            st.markdown("---")
            if st.button("Collect All Available Data", key="collect_all_btn", type="primary"):
                with st.spinner("Collecting data from all sources..."):
                    result = collection_manager.collect_all()
                    if result.is_success:
                        st.success(
                            f"Collection complete!\n"
                            f"- Documents collected: {result.total_documents}\n"
                            f"- Chunks indexed: {result.total_indexed}"
                        )
                    else:
                        st.warning(f"Partial collection: {result.total_documents} documents. Errors: {result.errors}")

            # Cleanup expired
            st.markdown("---")
            col_clean, col_info = st.columns([1, 2])
            with col_clean:
                if st.button("Cleanup Expired Documents", key="cleanup_btn"):
                    removed = collection_manager.cleanup_expired()
                    st.info(f"Removed {removed} expired document chunks")
            with col_info:
                st.markdown(
                    """
**Document expiration policy** — each document type has a fixed retention window
counted from its **publish date**:

| Document Type | Retention | Auto-extend on access |
|---|---|---|
| News Article | 7 days | No |
| Market Analysis | 30 days | +7 days per access |
| FOMC Minutes | 90 days | +30 days per access |
| Research Report | 90 days | +30 days per access |
| ETF Factsheet | 365 days | No |
| Investment Guide | 365 days | No |

Expired documents are excluded from search results automatically.
Click **Cleanup Expired Documents** to physically remove them from the index.
                    """
                )

    st.markdown("---")

    # Search interface
    if "rag_search_history" not in st.session_state:
        st.session_state.rag_search_history = []

    # Example queries
    st.markdown("**Example queries:**")
    example_cols = st.columns(4)
    example_queries = [
        "What is VOO expense ratio?",
        "Federal Reserve interest rates",
        "How to diversify portfolio?",
        "Retirement planning strategies",
    ]
    for col, query in zip(example_cols, example_queries):
        if col.button(query, key=f"rag_ex_{query}"):
            st.session_state.rag_selected_query = query

    # Check for selected example query
    default_query = st.session_state.get("rag_selected_query", "")

    # Search form
    with st.form(key="rag_search_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            query = st.text_input(
                "Search",
                value=default_query,
                placeholder="Ask about ETFs, market conditions, investment strategies...",
                label_visibility="collapsed",
            )
        with col2:
            top_k = st.selectbox("Results", [3, 5, 10], index=1, label_visibility="collapsed")
        with col3:
            search_btn = st.form_submit_button("Search", type="primary")

    # Determine if we should search (form submit OR example query clicked)
    search_query = None
    if search_btn and query and query.strip():
        search_query = query.strip()
    elif default_query:
        search_query = default_query.strip()

    # Clear selected query after use
    if "rag_selected_query" in st.session_state:
        st.session_state.rag_selected_query = ""

    # Perform search
    if search_query:
        with st.spinner("Searching knowledge base..."):
            results = rag.search(search_query, top_k=top_k)

        # Store in history
        st.session_state.rag_search_history.insert(0, results)
        # Keep only last 5 searches
        st.session_state.rag_search_history = st.session_state.rag_search_history[:5]

    # Display results
    if st.session_state.rag_search_history:
        for i, search_result in enumerate(st.session_state.rag_search_history):
            if i == 0:
                st.markdown(
                    f'{render_section_title("Search Results", "search")}',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"### Previous Search: {search_result['query']}")

            st.markdown(f"**Query:** {search_result['query']}")
            st.markdown(f"**Found:** {search_result['num_results']} relevant documents")

            for j, result in enumerate(search_result["results"]):
                with st.expander(
                    f"**{j+1}.** [{result['document_type']}] {result['source']}"
                    + (f" ({result['ticker']})" if result['ticker'] else "")
                    + f" — Score: {result['score']:.2f}",
                    expanded=(i == 0 and j == 0),  # Expand first result of latest search
                ):
                    st.markdown(result["content"])

            st.markdown("---")
    else:
        st.info("Enter a query above to search the financial knowledge base.")


def _render_existing_profile(profile) -> None:
    """Display an existing investment profile."""
    from backend.src.common.models import InvestmentProfile

    completeness = profile.profile_completeness
    st.progress(completeness, text=f"Profile {completeness:.0%} complete")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Risk & Horizon**")
        st.write(f"- Risk Tolerance: **{profile.risk_tolerance.value.title() if profile.risk_tolerance else 'Not set'}**")
        st.write(f"- Investment Horizon: **{profile.investment_horizon.value.title() if profile.investment_horizon else 'Not set'}**")
        st.write(f"- Experience Level: **{profile.investment_experience.value.title() if profile.investment_experience else 'Not set'}**")

        st.markdown("**Financial Info**")
        st.write(f"- Monthly Income: **{f'${profile.monthly_income:,.0f}' if profile.monthly_income else 'Not set'}**")
        st.write(f"- Monthly Expenses: **{f'${profile.monthly_expenses:,.0f}' if profile.monthly_expenses else 'Not set'}**")
        if profile.monthly_savings_target:
            st.write(f"- Savings Target: **${profile.monthly_savings_target:,.0f}**")

    with col2:
        st.markdown("**Preferences**")
        if profile.preferred_sectors:
            st.write(f"- Preferred Sectors: **{', '.join(profile.preferred_sectors)}**")
        if profile.excluded_sectors:
            st.write(f"- Excluded Sectors: **{', '.join(profile.excluded_sectors)}**")
        if profile.esg_preference is not None:
            st.write(f"- ESG Preference: **{'Yes' if profile.esg_preference else 'No'}**")

        st.markdown("**Constraints**")
        st.write(f"- Liquidity Needs: **{profile.liquidity_needs.value.title() if profile.liquidity_needs else 'Not set'}**")

        if profile.goals:
            st.markdown("**Goals**")
            for goal in profile.goals:
                target = f" (${goal.target_amount:,.0f})" if goal.target_amount else ""
                st.write(f"- {goal.goal_type.value.replace('_', ' ').title()}{target}")

    st.caption(f"Last updated: {profile.last_updated.strftime('%Y-%m-%d %H:%M')}")


def render_profiling_interface(aggregator: PortfolioAggregator, user_id: str, available_providers: list):
    """Render the investment profiling chat interface."""
    st.markdown(
        f'{render_section_title("Investment Profile Builder", "target")}',
        unsafe_allow_html=True
    )
    st.markdown(
        render_guidance_callout(
            "<strong>Build your personalized investment profile.</strong> "
            "Answer a few questions about your goals, risk tolerance, and preferences. "
            "The AI advisor will guide you through a natural conversation (about 3-5 minutes).",
            "tip",
        ),
        unsafe_allow_html=True,
    )

    # Show existing stored profile if available
    portfolio = aggregator.get_portfolio(user_id)
    if portfolio and portfolio.investment_profile:
        with st.expander("Current Stored Profile", expanded=True):
            _render_existing_profile(portfolio.investment_profile)

    # Check for LLM providers
    if not available_providers:
        st.warning("No LLM providers configured. Please enter an API key in the sidebar (Anthropic, OpenAI, or Gemini).")
        return

    st.markdown("---")

    # Initialize profiling session state
    if "profiling_state" not in st.session_state:
        st.session_state.profiling_state = None
    if "profiling_agent" not in st.session_state:
        st.session_state.profiling_agent = None
    if "profiling_initialized" not in st.session_state:
        st.session_state.profiling_initialized = False
    if "last_processed_input" not in st.session_state:
        st.session_state.last_processed_input = None
    if "profiling_user_id" not in st.session_state:
        st.session_state.profiling_user_id = None

    # Reset profiling state when the active user changes
    if st.session_state.profiling_user_id != user_id:
        st.session_state.profiling_state = None
        st.session_state.profiling_agent = None
        st.session_state.profiling_initialized = False
        st.session_state.last_processed_input = None
        st.session_state.profiling_user_id = user_id

    # Provider selection
    col1, col2 = st.columns([3, 1])
    with col2:
        provider_names = [p["name"] for p in available_providers]
        selected_idx = st.selectbox(
            "AI Model",
            range(len(provider_names)),
            format_func=lambda i: provider_names[i],
            key="profiling_llm_provider",
        )

    selected_provider = available_providers[selected_idx]["provider"]

    # Start / Resume buttons
    has_stored_profile = portfolio and portfolio.investment_profile
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if has_stored_profile:
            continue_clicked = st.button("Continue Profile", type="primary", key="continue_profiling_btn")
        else:
            continue_clicked = False
    with col2:
        start_new_clicked = st.button("Start New Profile", key="start_new_profiling_btn")

    def _init_profiling(resume: bool):
        """Initialize the profiling agent, optionally resuming from stored profile."""
        with st.spinner("Initializing your profile advisor..."):
            try:
                llm_client = LLMClient(provider=selected_provider)
                agent = ProfilingAgent(llm_client=llm_client)
                current_portfolio = aggregator.get_portfolio(user_id)
                user_name = current_portfolio.user.name if current_portfolio else "there"

                if resume and current_portfolio and current_portfolio.investment_profile:
                    state = agent.resume_conversation(user_id, user_name, current_portfolio.investment_profile)
                else:
                    state = agent.start_conversation(user_id, user_name)

                st.session_state.profiling_agent = agent
                st.session_state.profiling_state = state
                st.session_state.profiling_initialized = True
            except Exception as e:
                import traceback
                st.error(f"Failed to start profiling: {e}")
                st.code(traceback.format_exc())

    # Handle button clicks
    if continue_clicked or start_new_clicked:
        # Reset any existing session before starting
        st.session_state.profiling_state = None
        st.session_state.profiling_agent = None
        st.session_state.profiling_initialized = False
        _init_profiling(resume=continue_clicked)

    # Check if profiling is in progress
    if st.session_state.profiling_state is None:
        if has_stored_profile:
            st.info("Click 'Continue Profile' to resume where you left off, or 'Start New Profile' to start fresh.")
        else:
            st.info("Click 'Start New Profile' to begin the investment profiling conversation.")
        return

    state: ConversationState = st.session_state.profiling_state
    agent: ProfilingAgent = st.session_state.profiling_agent

    # Progress bar
    st.progress(state.completion_percentage / 100, text=f"Profile {state.completion_percentage:.0f}% complete")

    st.markdown("---")

    # Display conversation history
    st.markdown("#### Conversation")

    # Create a container for messages
    chat_container = st.container()

    with chat_container:
        for msg in state.messages:
            if msg.role == "assistant":
                st.markdown(f"**Advisor:** {msg.content}")
            else:
                st.markdown(f"**You:** {msg.content}")

    # Placeholder: filled with the user's pending message while advisor is thinking
    profiling_pending_container = st.empty()

    # Check if profile is complete
    if state.is_complete:
        st.success("Profile complete! Your investment profile has been built.")

        # Show profile summary
        st.markdown("#### Your Investment Profile")
        summary = agent.get_profile_summary(state)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Filled Information:**")
            for slot_name, slot_info in summary["filled_slots"].items():
                st.write(f"- {slot_name}: **{slot_info['value']}**")

        with col2:
            st.metric("Completion", f"{summary['completion_percentage']:.0f}%")
            st.metric("Conversation Turns", summary["conversation_turns"])

        # Export button
        if st.button("Export Profile"):
            profile_data = agent.export_profile(state)
            st.json(profile_data)

        return

    # Input form for user response (always at the bottom)
    with st.form(key="profiling_form", clear_on_submit=True):
        col_in, col_send = st.columns([5, 1])
        with col_in:
            user_input = st.text_input(
                "Your response",
                placeholder="Type your answer here...",
                label_visibility="collapsed",
            )
        with col_send:
            submit = st.form_submit_button("Send", type="primary")

    # Auto-focus the input box after each response
    if st.session_state.get("profiling_focus_input"):
        st.session_state.profiling_focus_input = False
        st_components.html(
            "<script>setTimeout(function(){"
            "var inputs=window.parent.document.querySelectorAll('input[type=\"text\"]');"
            "if(inputs.length>0)inputs[inputs.length-1].focus();"
            "},150);</script>",
            height=0,
        )

    # Process form submission with deduplication to prevent rerun loop
    if submit and user_input and user_input.strip():
        current_input = user_input.strip()
        if st.session_state.last_processed_input == current_input:
            st.session_state.last_processed_input = None
        else:
            profiling_pending_container.markdown(f"**You:** {current_input}")
            with st.spinner("Advisor is thinking..."):
                try:
                    updated_state = agent.process_response(state, current_input)
                    st.session_state.profiling_state = updated_state
                    st.session_state.last_processed_input = current_input
                    st.session_state.profiling_focus_input = True
                    if updated_state.is_complete:
                        new_profile = conv_state_to_investment_profile(user_id, updated_state)
                        aggregator.update_investment_profile(user_id, new_profile)
                        aggregator.save_portfolios()
                except Exception as e:
                    st.error(f"Error processing response: {e}")
            st.rerun()


def render_recommendations_tab(
    aggregator: PortfolioAggregator,
    user_id: str,
    available_providers: list,
    rag_system: RAGInitializer,
    collection_manager: DataCollectionManager,
):
    """Render the personalized recommendations tab."""
    from backend.src.recommendation.engine import (
        RecommendationEngine,
        RecommendationRequest,
        RecommendationCategory,
    )

    rag_init_result = ensure_rag_indexed()

    st.markdown(
        f'{render_section_title("Personalized Recommendations", "target")}',
        unsafe_allow_html=True,
    )
    st.markdown(
        render_guidance_callout(
            "<strong>Get AI-powered investment recommendations</strong> based on your portfolio, "
            "investment profile, market data, and financial knowledge base. "
            "For best results, complete your Investment Profile first.",
            "tip",
        ),
        unsafe_allow_html=True,
    )

    # Provider selection and query input
    with st.form(key="recommendation_form", clear_on_submit=False):
        st.markdown("**What would you like advice on?**")
        col_q, col_btn = st.columns([4, 1])
        with col_q:
            query = st.text_input(
                "What would you like advice on?",
                placeholder="e.g., How should I diversify? or leave blank for general recommendations",
                label_visibility="collapsed",
            )
        with col_btn:
            submit = st.form_submit_button("Generate Recommendations", type="primary")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            provider_names = [p["name"] for p in available_providers]
            selected_idx = st.selectbox(
                "AI Model",
                range(len(provider_names)),
                format_func=lambda i: provider_names[i],
                key="rec_llm_provider",
            )
        with col2:
            max_recs = st.selectbox("Max Recommendations", [3, 5, 7], index=1)
        with col3:
            include_live = st.checkbox("Include Live Data", value=False)

    pending_rec_container = st.empty()

    if submit:
        query_text = query.strip() if query.strip() else "General recommendations"
        pending_rec_container.markdown(f"**Your request:** {query_text}")
        selected_provider = available_providers[selected_idx]["provider"]
        with st.spinner("Analyzing your portfolio and generating recommendations..."):
            try:
                llm_client = LLMClient(provider=selected_provider)

                # Optionally create hybrid data provider
                hybrid_provider = None
                if include_live:
                    try:
                        from backend.src.mcp.integration.hybrid_data_provider import (
                            HybridDataProvider,
                        )
                        hybrid_provider = HybridDataProvider(
                            rag_initializer=rag_system,
                            portfolio_aggregator=aggregator,
                        )
                    except Exception as e:
                        logging.warning(f"Live data provider unavailable: {e}")

                engine = RecommendationEngine(
                    portfolio_aggregator=aggregator,
                    rag_initializer=rag_system,
                    llm_client=llm_client,
                    hybrid_data_provider=hybrid_provider,
                )

                request = RecommendationRequest(
                    user_id=user_id,
                    query=query.strip() if query else "",
                    max_recommendations=max_recs,
                    include_live_data=include_live,
                )

                response = engine.generate_recommendations(request)
                st.session_state.recommendation_response = response

            except Exception as e:
                st.error(f"Failed to generate recommendations: {e}")

    # Display results
    if "recommendation_response" not in st.session_state:
        st.info(
            "Configure your options above and click **Generate Recommendations** to get started. "
            "For best results, complete your Investment Profile first."
        )
        return

    response = st.session_state.recommendation_response

    if not response.success:
        st.error(f"Failed to generate recommendations: {response.error}")
        return

    st.markdown("---")

    # Summary
    st.markdown(f"**{response.summary}**")
    st.caption(
        f"Data sources: {', '.join(response.data_sources_used)} | "
        f"Generated in {response.processing_time_ms:.0f}ms"
    )

    # Warnings
    for warning in response.warnings:
        if "financial advice" in warning.lower():
            st.caption(f"*{warning}*")
        else:
            st.warning(warning)

    st.markdown("---")

    # Recommendation cards
    if not response.has_recommendations:
        st.info("No recommendations were generated. Try a different query or check your portfolio data.")
        return

    for i, rec in enumerate(response.recommendations):
        priority_label = rec.priority.value.upper()
        category_label = rec.category.value.replace("_", " ").title()

        with st.expander(
            f"{priority_label} | {category_label} | {rec.title}",
            expanded=(i == 0),
        ):
            st.markdown(f"**{rec.summary}**")
            st.markdown(rec.detailed_rationale)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk", rec.risk_level.value.title())
            with col2:
                st.metric("Expected Return", rec.expected_return_range or "N/A")
            with col3:
                st.metric("Confidence", rec.confidence.value.title())

            if rec.tickers:
                st.markdown(f"**Tickers:** {', '.join(rec.tickers)}")
            if rec.suggested_allocation_pct is not None:
                st.markdown(
                    f"**Suggested Allocation:** {rec.suggested_allocation_pct:.1f}%"
                )
            if rec.time_horizon:
                st.markdown(f"**Time Horizon:** {rec.time_horizon}")

            st.markdown(f"**Action:** {rec.suggested_action}")

            st.caption(
                f"Score: {rec.composite_score:.2f} "
                f"(Relevance: {rec.relevance_score:.2f}, "
                f"Risk Alignment: {rec.risk_alignment_score:.2f}, "
                f"Diversification: {rec.diversification_score:.2f})"
            )


def render_orchestrator_chat(
    aggregator: PortfolioAggregator,
    user_id: str,
    available_providers: list,
    rag_system: RAGInitializer,
):
    """Render the AI Advisor tab with orchestrator-powered chat."""
    st.markdown(
        f'{render_section_title("AI Advisor", "message-circle")}',
        unsafe_allow_html=True,
    )
    st.markdown(
        render_guidance_callout(
            "<strong>Chat with your AI wealth advisor.</strong> Ask about your portfolio, "
            "build your investment profile, or get personalized recommendations &mdash; all in one place.",
            "info",
        ),
        unsafe_allow_html=True,
    )

    if not available_providers:
        st.warning("No LLM providers configured. Please enter an API key in the sidebar (Anthropic, OpenAI, or Gemini).")
        return

    # Initialize orchestrator session state
    if "orchestrator_state" not in st.session_state:
        st.session_state.orchestrator_state = None
    if "orchestrator_instance" not in st.session_state:
        st.session_state.orchestrator_instance = None
    if "orchestrator_provider_idx" not in st.session_state:
        st.session_state.orchestrator_provider_idx = 0
    if "orchestrator_last_input" not in st.session_state:
        st.session_state.orchestrator_last_input = None
    if "orchestrator_session_id" not in st.session_state:
        st.session_state.orchestrator_session_id = None

    log_repo = get_ui_log_repository()

    # Provider selection
    col1, col2 = st.columns([4, 1])
    with col2:
        provider_names = [p["name"] for p in available_providers]
        selected_idx = st.selectbox(
            "AI Model",
            range(len(provider_names)),
            format_func=lambda i: provider_names[i],
            key="orchestrator_llm_provider",
        )

    selected_provider = available_providers[selected_idx]["provider"]

    # Initialize or reinitialize orchestrator when provider changes
    if (
        st.session_state.orchestrator_instance is None
        or st.session_state.orchestrator_provider_idx != selected_idx
    ):
        try:
            llm_client = LLMClient(provider=selected_provider)
            orchestrator = WealthOrchestrator(aggregator, llm_client, rag_system)
            st.session_state.orchestrator_instance = orchestrator
            st.session_state.orchestrator_state = orchestrator.create_initial_state(user_id)
            st.session_state.orchestrator_provider_idx = selected_idx
        except Exception as e:
            st.error(f"Failed to initialize AI Advisor: {e}")
            return

    orchestrator: WealthOrchestrator = st.session_state.orchestrator_instance
    orch_state = st.session_state.orchestrator_state

    # Update user_id if switched
    if orch_state.get("user_id") != user_id:
        orch_state["user_id"] = user_id

    # Process log panel (only shown after a recommendation request)
    process_log = st.session_state.get("last_process_log", [])
    if process_log:
        with st.expander("Agent process trace", expanded=False):
            _AGENT_ICONS = {
                "supervisor": "🧭",
                "portfolio_fetch": "📊",
                "profiling_fetch": "👤",
                "recommend": "💡",
            }
            for entry in process_log:
                icon = _AGENT_ICONS.get(entry.get("agent", ""), "•")
                ts = entry.get("ts", "")
                if ts:
                    ts = ts[11:19]  # HH:MM:SS only
                agent = entry.get("agent", "?")
                note = entry.get("note", "")
                st.markdown(f"`{ts}` {icon} **{agent}** — {note}")

    # Hint box — shown only when conversation is empty
    if not orch_state.get("messages"):
        st.info(
            "Start chatting! Try:\n"
            '- "What\'s my net worth?"\n'
            '- "Help me build my investment profile"\n'
            '- "What should I invest in?"\n'
            '- "Hello!"'
        )

    # Input form at the top
    with st.form(key="orchestrator_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input(
                "Message",
                placeholder="Ask your AI wealth advisor anything...",
                label_visibility="collapsed",
            )
        with col2:
            submit = st.form_submit_button("Send", type="primary")

    # Live data toggle (outside form so it persists across submits)
    use_live_data = st.toggle(
        "Use live market data",
        value=st.session_state.get("advisor_use_live_data", True),
        help="Fetch real-time quotes and news when generating recommendations. "
             "Disable for faster responses or when market data is unavailable.",
        key="advisor_use_live_data",
    )

    st.markdown("---")

    # Display conversation history (newest first — reverse in user+assistant pairs)
    messages = orch_state.get("messages", [])
    if messages:
        # Build display list: group into [user, assistant] pairs, reverse pair order
        display_messages = []
        i = len(messages) - 1
        while i >= 0:
            if i > 0 and messages[i-1].get("role") == "user" and messages[i].get("role") == "assistant":
                display_messages.append(messages[i-1])
                display_messages.append(messages[i])
                i -= 2
            else:
                display_messages.append(messages[i])
                i -= 1
        for msg_i, msg in enumerate(display_messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                st.markdown(f"**You:** {content}")
            elif role == "assistant":
                rec_resp = msg.get("recommendation_response")
                if rec_resp and rec_resp.success and rec_resp.has_recommendations:
                    st.markdown(f"**Advisor:** {rec_resp.summary}")
                    st.caption(
                        f"Data sources: {', '.join(rec_resp.data_sources_used)} | "
                        f"Generated in {rec_resp.processing_time_ms:.0f}ms"
                    )
                    for warning in rec_resp.warnings:
                        if "financial advice" in warning.lower():
                            st.caption(f"*{warning}*")
                        else:
                            st.warning(warning)
                    st.markdown("---")
                    for i, rec in enumerate(rec_resp.recommendations):
                        priority_label = rec.priority.value.upper()
                        category_label = rec.category.value.replace("_", " ").title()
                        with st.expander(
                            f"{priority_label} | {category_label} | {rec.title}",
                            expanded=(i == 0),
                        ):
                            st.markdown(f"**{rec.summary}**")
                            st.markdown(rec.detailed_rationale)
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Risk", rec.risk_level.value.title())
                            with col2:
                                st.metric("Expected Return", rec.expected_return_range or "N/A")
                            with col3:
                                st.metric("Confidence", rec.confidence.value.title())
                            if rec.tickers:
                                st.markdown(f"**Tickers:** {', '.join(rec.tickers)}")
                            if rec.suggested_allocation_pct is not None:
                                st.markdown(f"**Suggested Allocation:** {rec.suggested_allocation_pct:.1f}%")
                            if rec.time_horizon:
                                st.markdown(f"**Time Horizon:** {rec.time_horizon}")
                            st.markdown(f"**Action:** {rec.suggested_action}")
                            st.caption(
                                f"Score: {rec.composite_score:.2f} "
                                f"(Relevance: {rec.relevance_score:.2f}, "
                                f"Risk Alignment: {rec.risk_alignment_score:.2f}, "
                                f"Diversification: {rec.diversification_score:.2f})"
                            )
                else:
                    st.markdown(f"**Advisor:** {content}")
                if msg.get("visualization"):
                    st.plotly_chart(msg["visualization"], width='stretch', key=f"advisor_viz_{msg_i}")
    # Placeholder: filled with the user's pending message while advisor is thinking
    pending_msg_container = st.empty()

    # Auto-focus the input box after each response
    if st.session_state.get("advisor_focus_input"):
        st.session_state.advisor_focus_input = False
        st_components.html(
            "<script>setTimeout(function(){"
            "var inputs=window.parent.document.querySelectorAll('input[type=\"text\"]');"
            "if(inputs.length>0)inputs[inputs.length-1].focus();"
            "},150);</script>",
            height=0,
        )

    if submit and user_input and user_input.strip():
        current_input = user_input.strip()
        if st.session_state.orchestrator_last_input == current_input:
            st.session_state.orchestrator_last_input = None
        else:
            pending_msg_container.markdown(f"**You:** {current_input}")
            logger.info("Advisor query: user=%s query=%r", user_id, current_input[:80])
            with st.spinner("Advisor is thinking..."):
                try:
                    updated_state = orchestrator.process_message(
                        current_input, orch_state, include_live_data=use_live_data
                    )
                    st.session_state.last_process_log = updated_state.get("process_log", [])
                    st.session_state.orchestrator_state = updated_state
                    st.session_state.orchestrator_last_input = current_input
                    st.session_state.advisor_focus_input = True

                    # Log interaction (non-critical)
                    try:
                        if not st.session_state.orchestrator_session_id:
                            st.session_state.orchestrator_session_id = (
                                log_repo.create_session(user_id)
                            )
                        sid = st.session_state.orchestrator_session_id
                        log_repo.log_interaction(
                            session_id=sid,
                            user_message=current_input,
                            response_content=updated_state.get("response", ""),
                            intent=updated_state.get("intent", ""),
                            module_source=updated_state.get("module_source", ""),
                        )
                        log_repo.save_session_state(sid, updated_state)
                    except Exception:
                        pass  # logging failure must not break UI
                except Exception as e:
                    logger.exception("orchestrator.process_message failed")
                    st.error(f"Error: {e}")
                    st.session_state["last_ui_error"] = str(e)
            st.rerun()

    # Reset button
    if messages:
        if st.button("Clear Conversation", key="clear_orchestrator"):
            st.session_state.orchestrator_state = orchestrator.create_initial_state(user_id)
            st.session_state.orchestrator_last_input = None
            st.session_state.orchestrator_session_id = None
            st.rerun()


def render_user_management_tab(aggregator: PortfolioAggregator):
    """Render the admin-only User Management tab."""
    if not st.session_state.get("is_admin"):
        st.warning("This tab is restricted to administrators. Please log in via the sidebar.")
        return

    # Show results stored during this session
    if "plaid_connect_success" in st.session_state:
        st.success(st.session_state.pop("plaid_connect_success"))
    if "plaid_connect_error" in st.session_state:
        st.error(st.session_state.pop("plaid_connect_error"))

    st.markdown("### User Management")
    st.caption("Add, view, and remove users. Changes take effect immediately across all tabs.")

    # ------------------------------------------------------------------
    # Current user list
    # ------------------------------------------------------------------
    st.markdown("#### Current Users")

    user_ids = aggregator.get_user_ids()
    # Build a set of Plaid user_ids for source detection (handles custom IDs too)
    plaid_user_ids: set[str] = set()
    if _PLAID_AVAILABLE:
        try:
            from backend.src.plaid.token_store import PlaidTokenRepository
            from backend.src.logging_db import get_engine, get_session_factory
            _plaid_repo = PlaidTokenRepository(get_session_factory(get_engine()))
            plaid_user_ids = {r.user_id for r in _plaid_repo.get_all_active()}
        except Exception:
            pass

    if not user_ids:
        st.info("No users loaded.")
    else:
        rows = []
        for uid in user_ids:
            source = "Plaid (Live)" if uid in plaid_user_ids or uid.startswith("plaid_") else "Synthetic"
            portfolio = aggregator.get_portfolio(uid)
            name = portfolio.user.name if portfolio else uid
            net_worth = (
                f"${portfolio.summary.total_net_worth:,.0f}"
                if portfolio and portfolio.summary
                else "N/A"
            )
            rows.append({"User ID": uid, "Name": name, "Source": source, "Net Worth": net_worth})

        import pandas as pd
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Add new user
    # ------------------------------------------------------------------
    st.markdown("#### Add New User")

    new_user_id = st.text_input(
        "User ID (leave blank to auto-generate)",
        placeholder="e.g. user_006",
        key="mgmt_new_user_id",
    )

    col_plaid, col_synthetic = st.columns(2)

    # ---- Synthetic generation ----------------------------------------
    with col_synthetic:
        st.markdown("**Generate Synthetic Data**")
        with st.form("synthetic_form"):
            age_input = st.number_input("Age", min_value=18, max_value=99, value=35, step=1)
            income_input = st.number_input(
                "Annual Income ($)", min_value=0, max_value=10_000_000, value=80_000, step=1_000
            )
            risk_input = st.selectbox(
                "Risk Tolerance",
                ["conservative", "moderate", "aggressive"],
                index=1,
            )
            occupation_input = st.text_input("Occupation (optional)", placeholder="Software Engineer")
            submitted_synthetic = st.form_submit_button("Generate Synthetic User")

        if submitted_synthetic:
            from decimal import Decimal
            from backend.src.data_generation.generator import SyntheticDataGenerator
            import random

            target_uid = new_user_id.strip() or None
            if target_uid is None:
                existing = set(aggregator.get_user_ids())
                i = 1
                while True:
                    candidate = f"user_{i:03d}"
                    if candidate not in existing:
                        target_uid = candidate
                        break
                    i += 1

            if aggregator.get_portfolio(target_uid) is not None:
                st.error(f"User '{target_uid}' already exists.")
            else:
                gen = SyntheticDataGenerator(seed=random.randint(0, 999_999))
                portfolio = gen.generate_user_portfolio(user_id=target_uid)
                portfolio.user.age = int(age_input)
                portfolio.user.annual_income = Decimal(str(income_input))
                if occupation_input:
                    portfolio.user.occupation = occupation_input
                if portfolio.investment_profile:
                    from backend.src.common.models import RiskTolerance
                    try:
                        portfolio.investment_profile.risk_tolerance = RiskTolerance(risk_input)
                    except ValueError:
                        pass

                aggregator._portfolios[target_uid] = portfolio
                try:
                    aggregator.save_portfolios()
                    st.success(
                        f"Created synthetic user **{target_uid}** ({portfolio.user.name}). "
                        "Select them from the sidebar."
                    )
                    # Clear cached load_data so the updated JSON is reloaded on next fresh session
                    load_data.clear()
                except Exception as exc:
                    aggregator._portfolios.pop(target_uid, None)
                    st.error(f"Failed to save: {exc}")

    # ---- Plaid connection --------------------------------------------
    with col_plaid:
        st.markdown("**Connect Real Account via Plaid**")
        if not _PLAID_AVAILABLE:
            st.info("plaid-python is not installed. Add `plaid-python>=28.0.0` to requirements.txt.")
        elif not (os.environ.get("PLAID_CLIENT_ID") and os.environ.get("PLAID_SECRET")):
            st.info("Set PLAID_CLIENT_ID and PLAID_SECRET in .env to enable Plaid connection.")
        else:
            plaid_env = os.environ.get("PLAID_ENV", "sandbox")
            st.caption(f"Environment: `{plaid_env}`")
            if plaid_env == "sandbox":
                st.caption("Sandbox credentials: user `user_good` / password `pass_good`")

            plaid_age = st.number_input("Age", min_value=18, max_value=99, value=35, step=1, key="mgmt_plaid_age")
            plaid_income = st.number_input("Annual Income ($)", min_value=0, max_value=10_000_000, value=80_000, step=1_000, key="mgmt_plaid_income")
            plaid_occupation = st.text_input("Occupation", placeholder="Software Engineer", key="mgmt_plaid_occupation")

            if st.button("Connect Account via Plaid", key="mgmt_plaid_btn"):
                try:
                    from backend.src.plaid.client import PlaidClient
                    client = PlaidClient(
                        client_id=os.environ["PLAID_CLIENT_ID"],
                        secret=os.environ["PLAID_SECRET"],
                        env=plaid_env,
                    )
                    link_token = client.create_link_token(
                        user_id="streamlit_admin",
                        products=["transactions"],
                    )
                    st.session_state["mgmt_plaid_link_token"] = link_token
                    # Store profile values so the component can pass them to /exchange
                    st.session_state["mgmt_plaid_profile"] = {
                        "age": int(plaid_age),
                        "annual_income": float(plaid_income),
                        "occupation": plaid_occupation.strip() or "Unknown",
                        "custom_uid": new_user_id.strip(),
                    }
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to create link token: {exc}")

            if st.session_state.get("mgmt_plaid_link_token"):
                profile = st.session_state.get("mgmt_plaid_profile", {})
                _render_plaid_link_component(
                    link_token=st.session_state["mgmt_plaid_link_token"],
                    age=profile.get("age", 35),
                    annual_income=profile.get("annual_income", 0.0),
                    occupation=profile.get("occupation", "Unknown"),
                    custom_uid=profile.get("custom_uid", ""),
                )

    # ------------------------------------------------------------------
    # Remove user
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### Remove User")
    remove_uid = st.selectbox(
        "Select user to remove",
        [""] + user_ids,
        format_func=lambda x: "(select)" if x == "" else x,
        key="mgmt_remove_uid",
    )
    if remove_uid:
        is_plaid = remove_uid.startswith("plaid_")
        action_label = "Disconnect (Plaid)" if is_plaid else "Delete Synthetic User"
        if st.button(action_label, key="mgmt_remove_btn", type="primary"):
            if is_plaid:
                # Deactivate in DB via token store
                try:
                    from backend.src.logging_db import get_engine, get_session_factory
                    from backend.src.plaid.token_store import PlaidTokenRepository
                    engine = get_engine()
                    repo = PlaidTokenRepository(get_session_factory(engine))
                    repo.deactivate(remove_uid)
                    if aggregator._plaid_source:
                        aggregator._plaid_source.invalidate_cache(remove_uid)
                    st.success(f"Plaid connection for {remove_uid} deactivated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error: {exc}")
            else:
                aggregator._portfolios.pop(remove_uid, None)
                try:
                    aggregator.save_portfolios()
                    load_data.clear()
                    st.success(f"User {remove_uid} deleted.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to persist deletion: {exc}")


@st.cache_resource
def _start_plaid_file_server() -> int:
    """Start a background HTTP server that serves plaid_link.html and handles token exchange.

    Routes:
      GET /plaid_link.html  — serves the Plaid Link popup page
      GET /exchange         — exchanges public_token, saves to DB, clears data cache

    Returns the port number. Runs once per process (cached by st.cache_resource).
    """
    import http.server
    import socketserver
    import socket
    import urllib.parse

    plaid_html_path = Path(__file__).parent / "static" / "plaid_link.html"

    class _PlaidHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)

            if parsed.path == "/plaid_link.html":
                # Serve the Plaid Link popup page
                try:
                    content = plaid_html_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(content)
                except Exception as exc:
                    self.send_error(500, str(exc))

            elif parsed.path == "/exchange":
                # Exchange public_token for access_token and save to DB.
                # Runs entirely in this background thread — no Streamlit session needed.
                public_token = qs.get("token", [""])[0]
                try:
                    age = int(qs.get("age", ["35"])[0])
                except (ValueError, TypeError):
                    age = 35
                try:
                    annual_income = float(qs.get("income", ["0"])[0])
                except (ValueError, TypeError):
                    annual_income = 0.0
                occupation = qs.get("occ", ["Unknown"])[0] or "Unknown"
                custom_uid = qs.get("custom_uid", [""])[0].strip() or None

                try:
                    if not public_token:
                        raise ValueError("Missing public_token")
                    from backend.src.plaid.client import PlaidClient
                    from backend.src.plaid.token_store import PlaidTokenRepository
                    from backend.src.logging_db import get_engine, get_session_factory

                    env = os.environ.get("PLAID_ENV", "sandbox")
                    client = PlaidClient(
                        client_id=os.environ["PLAID_CLIENT_ID"],
                        secret=os.environ["PLAID_SECRET"],
                        env=env,
                    )
                    access_token, item_id = client.exchange_public_token(public_token)
                    institution_name = client.get_institution_name(access_token)
                    engine = get_engine()
                    repo = PlaidTokenRepository(get_session_factory(engine))
                    user_id = repo.save(
                        item_id=item_id,
                        access_token=access_token,
                        institution_name=institution_name,
                        products=["transactions"],
                        env=env,
                        age=age,
                        annual_income=annual_income,
                        occupation=occupation,
                        custom_user_id=custom_uid,
                    )
                    # Clear the Streamlit data cache so next load picks up new user
                    load_data.clear()
                    logger.info("Plaid exchange success: user_id=%s inst=%s", user_id, institution_name)

                    body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Connected</title></head>
<body style="font-family:sans-serif;text-align:center;padding:40px;background:#f8fafc">
  <p style="color:#16a34a;font-size:18px;margin-bottom:8px">Connected!</p>
  <p style="color:#475569;font-size:13px">{user_id} ({institution_name})</p>
  <p style="color:#94a3b8;font-size:12px">Closing window...</p>
  <script>setTimeout(function(){{window.close();}}, 1200);</script>
</body></html>""".encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                except Exception as exc:
                    logger.error("Plaid exchange error: %s", exc)
                    err_body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Error</title></head>
<body style="font-family:sans-serif;text-align:center;padding:40px;background:#f8fafc">
  <p style="color:#dc2626;font-size:16px">Exchange failed</p>
  <p style="color:#475569;font-size:12px">{str(exc)[:200]}</p>
  <script>setTimeout(function(){{window.close();}}, 4000);</script>
</body></html>""".encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(err_body)))
                    self.end_headers()
                    self.wfile.write(err_body)

            else:
                self.send_error(404)

        def log_message(self, *args):  # suppress access logs
            pass

    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    server = socketserver.TCPServer(("", port), _PlaidHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return port


def _render_plaid_link_component(
    link_token: str,
    age: int = 35,
    annual_income: float = 0.0,
    occupation: str = "Unknown",
    custom_uid: str = "",
):
    """Render a button that opens Plaid Link in a popup window.

    Opens ui/static/plaid_link.html served by a background Python HTTP server
    (correct Content-Type, proper HTTP origin so Plaid's flink.js postMessage works).
    On success the popup navigates to /exchange which saves the token and closes.
    """
    port = _start_plaid_file_server()
    html = f"""
<style>body {{ margin: 0; font-family: sans-serif; }}</style>
<div style="padding: 8px 0;">
    <button id="plaid-open-btn" style="
        background: #2563eb; color: white; border: none;
        border-radius: 6px; padding: 8px 18px; font-size: 14px; cursor: pointer;
    ">Open Plaid Link</button>
    <div id="plaid-status" style="margin-top: 8px; font-size: 13px; color: #475569; min-height: 20px;">
        Ready — click the button to connect your account.
    </div>
</div>
<script>
(function() {{
    var statusEl = document.getElementById('plaid-status');
    var btn = document.getElementById('plaid-open-btn');
    var exchangeDone = false;

    btn.addEventListener('click', function() {{
        btn.disabled = true;
        exchangeDone = false;
        statusEl.style.color = '#475569';
        statusEl.textContent = 'Opening Plaid...';

        // Build the plaid_link.html URL served by the background Python HTTP server.
        // The popup completes Plaid Link and then navigates itself to /exchange,
        // which exchanges the token and saves it to DB — no Streamlit URL manipulation needed.
        var plaidPageUrl = 'http://' + window.top.location.hostname
            + ':{port}/plaid_link.html'
            + '?token={link_token}'
            + '&age={age}'
            + '&income={annual_income}'
            + '&occ=' + encodeURIComponent('{occupation}')
            + '&custom_uid=' + encodeURIComponent('{custom_uid}')
            + '&exchange_base=http://' + window.top.location.hostname + ':{port}';

        var popup = window.open(plaidPageUrl, 'PlaidLink', 'width=500,height=700,left=200,top=80');
        if (!popup || popup.closed) {{
            statusEl.style.color = '#dc2626';
            statusEl.textContent = 'Popup blocked — allow popups for this site and try again.';
            btn.disabled = false;
            return;
        }}

        statusEl.textContent = 'Plaid window open — complete the flow there.';

        // Poll until popup closes.
        // If exchange succeeded, the /exchange page shows a success message then
        // closes after ~1.2s. We then reload Streamlit to pick up the new user.
        var timer = setInterval(function() {{
            if (popup.closed) {{
                clearInterval(timer);
                if (exchangeDone) {{
                    statusEl.style.color = '#16a34a';
                    statusEl.textContent = 'Account connected! Reloading...';
                    setTimeout(function() {{ window.top.location.reload(); }}, 400);
                }} else {{
                    btn.disabled = false;
                    statusEl.style.color = '#64748b';
                    statusEl.textContent = 'Window closed. Click the button to try again.';
                }}
            }}
        }}, 500);
    }});

    // Listen for exchange-complete signal from the popup
    window.addEventListener('message', function(event) {{
        if (event.data && event.data.type === 'plaid_exchange_done') {{
            exchangeDone = true;
        }}
    }});
}})();
</script>
"""
    st_components.html(html, height=100)




def main():
    """Main application entry point."""
    # Initialize session state
    if "selected_query" not in st.session_state:
        st.session_state.selected_query = ""

    # Render fancy header
    st.markdown(render_header(), unsafe_allow_html=True)

    # Welcome banner (dismissible via session state)
    if "dismiss_welcome" not in st.session_state:
        st.session_state.dismiss_welcome = False

    if not st.session_state.dismiss_welcome:
        st.markdown(render_welcome_banner(), unsafe_allow_html=True)
        col_spacer, col_dismiss = st.columns([6, 1])
        with col_dismiss:
            if st.button("Dismiss", key="dismiss_welcome_btn", type="secondary"):
                st.session_state.dismiss_welcome = True
                st.rerun()

    # Load data
    try:
        aggregator = load_data()
    except FileNotFoundError:
        st.error(
            "Synthetic data not found. Please run the data generator first:\n\n"
            "```python\n"
            "from backend.src.data_generation import generate_sample_data\n"
            "generate_sample_data()\n"
            "```"
        )
        return

    # Create RAG objects and start background indexing (non-blocking)
    rag_system, collection_manager = initialize_rag_system()

    # Sidebar (must run before get_available_providers so user-entered keys are in os.environ)
    selected_user = render_sidebar(aggregator)

    # Get available LLM providers (picks up any keys entered in sidebar)
    available_providers = get_available_providers()

    if not selected_user:
        st.info("Please select a user from the sidebar to begin.")
        return

    # Main content tabs (tab8 = User Management, admin only)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "Dashboard",
            "AI Advisor",
            "Ask About Portfolio",
            "Investment Profile",
            "Recommendations",
            "Knowledge Search",
            "Holdings",
            "User Management",
        ]
    )

    with tab1:
        render_dashboard(aggregator, selected_user)

    with tab2:
        if available_providers:
            render_orchestrator_chat(aggregator, selected_user, available_providers, rag_system)
        else:
            st.warning(
                "No LLM providers available.\n\n"
                "Please enter an API key in the **API Keys** section in the sidebar (Anthropic, OpenAI, or Gemini)."
            )

    with tab3:
        if available_providers:
            render_chat_interface(aggregator, selected_user, available_providers)
        else:
            st.warning(
                "No LLM providers available.\n\n"
                "Please enter an API key in the **API Keys** section in the sidebar (Anthropic, OpenAI, or Gemini)."
            )

    with tab4:
        if available_providers:
            render_profiling_interface(aggregator, selected_user, available_providers)
        else:
            st.warning(
                "No LLM providers available.\n\n"
                "Please enter an API key in the **API Keys** section in the sidebar (Anthropic, OpenAI, or Gemini)."
            )

    with tab5:
        if available_providers:
            render_recommendations_tab(
                aggregator, selected_user, available_providers,
                rag_system, collection_manager,
            )
        else:
            st.warning(
                "No LLM providers available.\n\n"
                "Please enter an API key in the **API Keys** section in the sidebar (Anthropic, OpenAI, or Gemini)."
            )

    with tab6:
        render_knowledge_search(rag_system, None, collection_manager, aggregator, selected_user)

    with tab7:
        render_holdings_view(aggregator, selected_user)

    with tab8:
        render_user_management_tab(aggregator)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #64748b; font-size: 0.95rem; padding: 0.5rem 0;'>"
        "Wealth Intelligence System | Portfolio Analysis, Profiling & Knowledge Search"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
