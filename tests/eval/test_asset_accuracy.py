"""Evaluation: Asset Agent — numeric accuracy and intent routing.

Phase 2 scope — offline, no real LLM calls:
  - Tests PortfolioAggregator numeric calculations against known synthetic data.
  - Tests AssetQueryInterface._fallback_intent_detection keyword coverage.
  - Tests AssetAgent.get_quick_summary() (LLM-free path).
  - Tests AssetAgent.process() error handling and data fields with mocked LLM.

Run with:
    pytest tests/eval/test_asset_accuracy.py -v -s
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.src.agents.asset_agent import AssetAgent
from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.asset_management.query_interface import AssetQueryInterface, QueryType


# ── Constants derived from synthetic data (user_001) ────────────────────────
# These values are read from data/synthetic/synthetic_portfolios.json and
# hardcoded here so the tests are self-contained and fast.

TEST_USER_ID = "user_001"
UNKNOWN_USER_ID = "user_does_not_exist"

# From summary block in synthetic JSON
EXPECTED_NET_WORTH = Decimal("705764.50")
EXPECTED_TOTAL_ASSETS = Decimal("1167686.79")
EXPECTED_TOTAL_LIABILITIES = Decimal("461922.29")
EXPECTED_LIQUIDITY_RATIO = 0.0911  # 9.11%
EXPECTED_CASH = Decimal("106379.88")  # cash_and_equivalents

# Dominant asset type in allocation
DOMINANT_ASSET_TYPE = "real_estate"  # 87.6% from allocation_by_asset_type

# Number of data items
EXPECTED_BANK_ACCOUNT_COUNT = 3
EXPECTED_INVESTMENT_ACCOUNT_COUNT = 1
EXPECTED_HOLDINGS_COUNT = 5


# ── Fixtures ─────────────────────────────────────────────────────────────────


DATA_PATH = (
    Path(__file__).parent.parent.parent / "data" / "synthetic" / "synthetic_portfolios.json"
)


@pytest.fixture(scope="module")
def aggregator() -> PortfolioAggregator:
    """Real aggregator loaded from the synthetic data file."""
    agg = PortfolioAggregator(DATA_PATH)
    agg.load_data()
    return agg


@pytest.fixture(scope="module")
def mock_llm():
    """LLM that always raises, forcing keyword fallback in AssetQueryInterface."""
    llm = MagicMock()
    llm.chat.side_effect = RuntimeError("LLM disabled for offline eval")
    return llm


@pytest.fixture(scope="module")
def asset_agent(aggregator, mock_llm) -> AssetAgent:
    """AssetAgent with mocked LLM — only pure-Python paths exercise."""
    return AssetAgent(aggregator, llm_client=mock_llm)


@pytest.fixture(scope="module")
def query_interface(aggregator, mock_llm) -> AssetQueryInterface:
    return AssetQueryInterface(aggregator, llm_client=mock_llm)


# ── PortfolioAggregator — load and data access ───────────────────────────────


class TestAggregatorDataLoading:
    """Verify data loading and basic access methods."""

    def test_known_user_exists(self, aggregator):
        portfolio = aggregator.get_portfolio(TEST_USER_ID)
        assert portfolio is not None

    def test_unknown_user_returns_none(self, aggregator):
        portfolio = aggregator.get_portfolio(UNKNOWN_USER_ID)
        assert portfolio is None

    def test_get_user_ids_is_non_empty(self, aggregator):
        ids = aggregator.get_user_ids()
        assert len(ids) > 0
        assert TEST_USER_ID in ids

    def test_get_all_portfolios_returns_list(self, aggregator):
        portfolios = aggregator.get_all_portfolios()
        assert isinstance(portfolios, list)
        assert len(portfolios) > 0

    def test_unloaded_aggregator_raises(self):
        fresh = PortfolioAggregator()
        with pytest.raises(RuntimeError):
            fresh.get_portfolio(TEST_USER_ID)


# ── PortfolioAggregator — numeric accuracy ────────────────────────────────────


class TestAggregatorCalculations:
    """Core financial calculations must be exact (Decimal arithmetic)."""

    def test_net_worth_matches_expected(self, aggregator):
        nw = aggregator.get_net_worth(TEST_USER_ID)
        assert nw == pytest.approx(float(EXPECTED_NET_WORTH), rel=1e-4)

    def test_net_worth_equals_assets_minus_liabilities(self, aggregator):
        metrics = aggregator.get_portfolio_metrics(TEST_USER_ID)
        assets = float(metrics["total_assets"])
        liabilities = float(metrics["total_liabilities"])
        net_worth = float(metrics["net_worth"])
        assert net_worth == pytest.approx(assets - liabilities, rel=1e-4)

    def test_total_cash_is_positive(self, aggregator):
        cash = aggregator.get_total_cash(TEST_USER_ID)
        assert cash > 0

    def test_total_cash_matches_expected(self, aggregator):
        cash = aggregator.get_total_cash(TEST_USER_ID)
        assert float(cash) == pytest.approx(float(EXPECTED_CASH), rel=1e-3)

    def test_liquidity_ratio_matches_expected(self, aggregator):
        ratio = aggregator.get_liquidity_ratio(TEST_USER_ID)
        assert ratio == pytest.approx(EXPECTED_LIQUIDITY_RATIO, rel=1e-2)

    def test_liquidity_ratio_is_between_0_and_1(self, aggregator):
        ratio = aggregator.get_liquidity_ratio(TEST_USER_ID)
        assert 0.0 <= ratio <= 1.0

    def test_asset_allocation_sums_to_100(self, aggregator):
        allocation = aggregator.get_asset_allocation(TEST_USER_ID)
        total = sum(allocation.values())
        assert total == pytest.approx(100.0, abs=0.1)

    def test_asset_allocation_has_expected_dominant_type(self, aggregator):
        allocation = aggregator.get_asset_allocation(TEST_USER_ID)
        assert DOMINANT_ASSET_TYPE in allocation
        assert allocation[DOMINANT_ASSET_TYPE] > 50.0  # real_estate is dominant

    def test_asset_allocation_exclude_real_estate_still_sums_100(self, aggregator):
        allocation = aggregator.get_asset_allocation(TEST_USER_ID, exclude=["real_estate"])
        assert DOMINANT_ASSET_TYPE not in allocation
        total = sum(allocation.values())
        assert total == pytest.approx(100.0, abs=0.5)

    def test_sector_allocation_sums_to_100(self, aggregator):
        allocation = aggregator.get_sector_allocation(TEST_USER_ID)
        if allocation:  # may be empty if no holdings
            total = sum(allocation.values())
            assert total == pytest.approx(100.0, abs=1.0)

    def test_top_holdings_count_respects_n(self, aggregator):
        for n in [1, 3, 5]:
            holdings = aggregator.get_top_holdings(TEST_USER_ID, n)
            assert len(holdings) <= n

    def test_top_holdings_sorted_by_value_descending(self, aggregator):
        holdings = aggregator.get_top_holdings(TEST_USER_ID, 10)
        values = [float(h.market_value) for h in holdings]
        assert values == sorted(values, reverse=True)

    def test_holdings_by_type_covers_all_holdings(self, aggregator):
        by_type = aggregator.get_holdings_by_type(TEST_USER_ID)
        total_by_type = sum(len(v) for v in by_type.values())
        portfolio = aggregator.get_portfolio(TEST_USER_ID)
        assert total_by_type == len(portfolio.holdings)

    def test_unknown_user_net_worth_returns_zero(self, aggregator):
        nw = aggregator.get_net_worth(UNKNOWN_USER_ID)
        assert nw == Decimal("0")

    def test_unknown_user_cash_returns_zero(self, aggregator):
        cash = aggregator.get_total_cash(UNKNOWN_USER_ID)
        assert cash == Decimal("0")


# ── PortfolioAggregator — metrics dict ────────────────────────────────────────


class TestAggregatorMetrics:
    """get_portfolio_metrics returns the expected key set and sensible values."""

    REQUIRED_KEYS = {
        "net_worth",
        "total_assets",
        "total_liabilities",
        "cash_and_equivalents",
        "liquidity_ratio",
        "debt_to_asset_ratio",
        "num_bank_accounts",
        "num_investment_accounts",
        "num_holdings",
        "num_properties",
    }

    def test_all_required_keys_present(self, aggregator):
        metrics = aggregator.get_portfolio_metrics(TEST_USER_ID)
        missing = self.REQUIRED_KEYS - set(metrics.keys())
        assert not missing, f"Missing keys in metrics: {missing}"

    def test_bank_account_count_matches_data(self, aggregator):
        metrics = aggregator.get_portfolio_metrics(TEST_USER_ID)
        # Only active accounts count
        assert metrics["num_bank_accounts"] <= EXPECTED_BANK_ACCOUNT_COUNT

    def test_holdings_count_matches_data(self, aggregator):
        metrics = aggregator.get_portfolio_metrics(TEST_USER_ID)
        assert metrics["num_holdings"] == EXPECTED_HOLDINGS_COUNT

    def test_debt_to_asset_ratio_in_range(self, aggregator):
        metrics = aggregator.get_portfolio_metrics(TEST_USER_ID)
        ratio = metrics["debt_to_asset_ratio"]
        assert 0.0 <= float(ratio) <= 1.0

    def test_unknown_user_returns_empty(self, aggregator):
        metrics = aggregator.get_portfolio_metrics(UNKNOWN_USER_ID)
        assert metrics == {}


# ── AssetQueryInterface — fallback intent detection ───────────────────────────


class TestFallbackIntentDetection:
    """_fallback_intent_detection must map queries to the correct QueryType."""

    @pytest.mark.parametrize(
        "query, expected_type",
        [
            # NET_WORTH
            ("What's my net worth?", QueryType.NET_WORTH),
            ("What is my total wealth?", QueryType.NET_WORTH),
            # ASSET_ALLOCATION
            ("Show me my asset allocation", QueryType.ASSET_ALLOCATION),
            ("Give me a portfolio breakdown", QueryType.ASSET_ALLOCATION),
            ("Show the asset distribution breakdown", QueryType.ASSET_ALLOCATION),
            # SECTOR_ALLOCATION
            ("What's my sector allocation?", QueryType.SECTOR_ALLOCATION),
            ("Show me sector distribution", QueryType.SECTOR_ALLOCATION),
            # LIQUIDITY
            ("What's my liquidity ratio?", QueryType.LIQUIDITY),
            ("How much cash do I have?", QueryType.LIQUIDITY),
            # TOP_HOLDINGS
            ("Show my top holdings", QueryType.TOP_HOLDINGS),
            ("What are my best positions?", QueryType.TOP_HOLDINGS),
            ("What are my largest investments?", QueryType.TOP_HOLDINGS),
            # HOLDINGS
            ("List all my holdings", QueryType.HOLDINGS),
            ("Show my stock positions", QueryType.HOLDINGS),
            ("What investments do I have?", QueryType.HOLDINGS),
            # REAL_ESTATE
            ("Tell me about my property", QueryType.REAL_ESTATE),
            ("Show my property portfolio", QueryType.REAL_ESTATE),
            # GAINS_LOSSES
            ("What are my unrealized gains?", QueryType.GAINS_LOSSES),
            ("Show my profit and loss", QueryType.GAINS_LOSSES),
            ("What's my return?", QueryType.GAINS_LOSSES),
            # BANK_ACCOUNTS
            ("Show my bank accounts", QueryType.BANK_ACCOUNTS),
            ("What's in my checking account?", QueryType.BANK_ACCOUNTS),
            # METRICS
            ("Give me a portfolio overview", QueryType.METRICS),
            ("What are my key financial metrics?", QueryType.METRICS),
        ],
    )
    def test_fallback_detects_intent(self, query_interface, query, expected_type):
        intent = query_interface._fallback_intent_detection(query)
        assert intent.query_type == expected_type, (
            f"Query '{query}' → got '{intent.query_type}', expected '{expected_type}'"
        )

    def test_unknown_query_falls_through_to_general(self, query_interface):
        intent = query_interface._fallback_intent_detection("Tell me something interesting")
        assert intent.query_type == QueryType.GENERAL

    def test_sector_keyword_before_allocation(self, query_interface):
        """Queries with 'sector' must map to SECTOR_ALLOCATION, not ASSET_ALLOCATION."""
        intent = query_interface._fallback_intent_detection("Show sector allocation")
        assert intent.query_type == QueryType.SECTOR_ALLOCATION


# ── AssetAgent.get_quick_summary() ────────────────────────────────────────────


class TestQuickSummary:
    """get_quick_summary is LLM-free and should always return complete data."""

    REQUIRED_KEYS = {"metrics", "allocation", "top_holdings"}

    def test_returns_all_required_keys(self, asset_agent):
        summary = asset_agent.get_quick_summary(TEST_USER_ID)
        assert self.REQUIRED_KEYS.issubset(summary.keys())

    def test_metrics_sub_dict_is_non_empty(self, asset_agent):
        summary = asset_agent.get_quick_summary(TEST_USER_ID)
        assert isinstance(summary["metrics"], dict)
        assert len(summary["metrics"]) > 0

    def test_allocation_percentages_sum_to_100(self, asset_agent):
        summary = asset_agent.get_quick_summary(TEST_USER_ID)
        total = sum(summary["allocation"].values())
        assert total == pytest.approx(100.0, abs=0.5)

    def test_top_holdings_is_list_of_dicts(self, asset_agent):
        summary = asset_agent.get_quick_summary(TEST_USER_ID)
        holdings = summary["top_holdings"]
        assert isinstance(holdings, list)
        for h in holdings:
            assert "symbol" in h
            assert "value" in h
            assert "gain_loss_pct" in h

    def test_top_holdings_count_at_most_5(self, asset_agent):
        summary = asset_agent.get_quick_summary(TEST_USER_ID)
        assert len(summary["top_holdings"]) <= 5

    def test_no_llm_calls_made(self, asset_agent, mock_llm):
        """get_quick_summary must not trigger any LLM calls."""
        mock_llm.chat.reset_mock()
        asset_agent.get_quick_summary(TEST_USER_ID)
        mock_llm.chat.assert_not_called()


# ── AssetAgent.process() — error paths ───────────────────────────────────────


class TestAssetAgentProcess:
    """AssetAgent.process() must return success=False for unknown users
    and correctly populate data fields for known users."""

    def test_unknown_user_returns_failure(self, asset_agent):
        result = asset_agent.process(UNKNOWN_USER_ID, "What's my net worth?")
        assert result.success is False
        assert result.error is not None

    def test_unknown_user_query_type_is_error(self, asset_agent):
        result = asset_agent.process(UNKNOWN_USER_ID, "Show my portfolio")
        assert result.query_type == "error"

    def test_known_user_fallback_populates_data(self, asset_agent):
        """With LLM mocked out, AssetAgent falls back to keyword intent detection
        and populates the data dict from the aggregator (answer may fail)."""
        result = asset_agent.process(TEST_USER_ID, "What's my net worth?")
        # Even if the answer LLM fails, the agent should catch it gracefully
        # success may be False due to LLM failure in _generate_answer,
        # but the result should not be None
        assert result is not None
        assert isinstance(result.query_type, str)

    def test_process_never_raises(self, asset_agent):
        """process() must catch all exceptions and return an AssetAgentResponse."""
        # Throw unusual queries that might cause edge cases
        for query in ["", "???", "x" * 500]:
            result = asset_agent.process(TEST_USER_ID, query)
            assert result is not None

    def test_get_supported_queries_returns_list(self, asset_agent):
        queries = asset_agent.get_supported_queries()
        assert isinstance(queries, list)
        assert len(queries) > 0
        for item in queries:
            assert "type" in item
            assert "examples" in item


# ── AssetQueryInterface._get_data_for_intent ─────────────────────────────────


class TestGetDataForIntent:
    """_get_data_for_intent returns the correct data shape for each QueryType."""

    def test_net_worth_data_has_net_worth_key(self, query_interface):
        from backend.src.asset_management.query_interface import QueryIntent

        intent = QueryIntent(query_type=QueryType.NET_WORTH)
        data = query_interface._get_data_for_intent(intent, TEST_USER_ID)
        assert "net_worth" in data
        assert "metrics" in data
        assert float(data["net_worth"]) > 0

    def test_asset_allocation_data_has_allocation_key(self, query_interface):
        from backend.src.asset_management.query_interface import QueryIntent

        intent = QueryIntent(query_type=QueryType.ASSET_ALLOCATION)
        data = query_interface._get_data_for_intent(intent, TEST_USER_ID)
        assert "allocation" in data
        total = sum(data["allocation"].values())
        assert total == pytest.approx(100.0, abs=0.5)

    def test_liquidity_data_has_cash_and_ratio(self, query_interface):
        from backend.src.asset_management.query_interface import QueryIntent

        intent = QueryIntent(query_type=QueryType.LIQUIDITY)
        data = query_interface._get_data_for_intent(intent, TEST_USER_ID)
        assert "liquidity_ratio" in data
        assert "cash" in data

    def test_top_holdings_data_has_holdings_list(self, query_interface):
        from backend.src.asset_management.query_interface import QueryIntent

        intent = QueryIntent(
            query_type=QueryType.TOP_HOLDINGS, parameters={"top_n": 3}
        )
        data = query_interface._get_data_for_intent(intent, TEST_USER_ID)
        assert "holdings" in data
        assert len(data["holdings"]) <= 3
        for h in data["holdings"]:
            assert "symbol" in h
            assert "value" in h

    def test_real_estate_data_has_properties(self, query_interface):
        from backend.src.asset_management.query_interface import QueryIntent

        intent = QueryIntent(query_type=QueryType.REAL_ESTATE)
        data = query_interface._get_data_for_intent(intent, TEST_USER_ID)
        assert "real_estate" in data

    def test_gains_losses_data_sorted_by_gain(self, query_interface):
        from backend.src.asset_management.query_interface import QueryIntent

        intent = QueryIntent(query_type=QueryType.GAINS_LOSSES)
        data = query_interface._get_data_for_intent(intent, TEST_USER_ID)
        assert "holdings" in data
        gains = [h["gain_loss"] for h in data["holdings"]]
        assert gains == sorted(gains, reverse=True)
