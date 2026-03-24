"""Dynamic visualization engine for financial data."""

from decimal import Decimal
from typing import Any

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend.src.common.models import Holding, UserPortfolio


class VisualizationEngine:
    """Generate dynamic visualizations for financial data."""

    def __init__(self, theme: str = "plotly_white"):
        """Initialize visualization engine.

        Args:
            theme: Plotly theme to use.
        """
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set2

    def create_asset_allocation_pie(
        self,
        allocation: dict[str, float],
        title: str = "Asset Allocation",
    ) -> go.Figure:
        """Create pie chart for asset allocation.

        Args:
            allocation: Dict of asset type to percentage.
            title: Chart title.

        Returns:
            Plotly figure.
        """
        # Filter out zero allocations
        filtered = {k: v for k, v in allocation.items() if v > 0.5}

        labels = [k.replace("_", " ").title() for k in filtered.keys()]
        values = list(filtered.values())

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    textinfo="label+percent",
                    textposition="outside",
                    textfont=dict(size=11),
                    marker=dict(colors=self.color_palette),
                    domain=dict(x=[0.15, 0.85], y=[0.1, 0.9]),
                )
            ]
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            template=self.theme,
            showlegend=False,
            height=500,
            margin=dict(t=60, b=60, l=100, r=100),
        )

        return fig

    def create_sector_allocation_pie(
        self,
        allocation: dict[str, float],
        title: str = "Sector Allocation",
    ) -> go.Figure:
        """Create pie chart for sector allocation."""
        return self.create_asset_allocation_pie(allocation, title)

    def create_holdings_bar(
        self,
        holdings: list[Holding],
        by: str = "value",
        title: str = "Holdings Overview",
    ) -> go.Figure:
        """Create bar chart for holdings.

        Args:
            holdings: List of holdings.
            by: 'value' for market value, 'gain' for unrealized gain/loss.
            title: Chart title.

        Returns:
            Plotly figure.
        """
        if not holdings:
            return self._empty_figure("No holdings data")

        sorted_holdings = sorted(holdings, key=lambda h: h.market_value, reverse=True)[:10]

        symbols = [h.symbol for h in sorted_holdings]

        if by == "gain":
            values = [float(h.unrealized_gain_loss) for h in sorted_holdings]
            colors = ["green" if v >= 0 else "red" for v in values]
            y_title = "Unrealized Gain/Loss ($)"
        else:
            values = [float(h.market_value) for h in sorted_holdings]
            colors = self.color_palette[0]
            y_title = "Market Value ($)"

        fig = go.Figure(
            data=[
                go.Bar(
                    x=symbols,
                    y=values,
                    marker_color=colors,
                    text=[f"${v:,.0f}" for v in values],
                    textposition="outside",
                )
            ]
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_title="Symbol",
            yaxis_title=y_title,
            template=self.theme,
            height=400,
        )

        return fig

    def create_net_worth_breakdown(self, portfolio: UserPortfolio) -> go.Figure:
        """Create waterfall chart showing net worth breakdown.

        Args:
            portfolio: User portfolio data.

        Returns:
            Plotly figure.
        """
        if not portfolio.summary:
            return self._empty_figure("No summary data")

        summary = portfolio.summary

        categories = []
        values = []
        measures = []

        # Assets
        categories.append("Cash & Equivalents")
        values.append(float(summary.cash_and_equivalents))
        measures.append("relative")

        categories.append("Stocks")
        values.append(float(summary.stocks))
        measures.append("relative")

        categories.append("ETFs")
        values.append(float(summary.etfs))
        measures.append("relative")

        categories.append("Bonds")
        values.append(float(summary.bonds))
        measures.append("relative")

        if summary.mutual_funds > 0:
            categories.append("Mutual Funds")
            values.append(float(summary.mutual_funds))
            measures.append("relative")

        if summary.crypto > 0:
            categories.append("Crypto")
            values.append(float(summary.crypto))
            measures.append("relative")

        if summary.real_estate_equity > 0:
            categories.append("Real Estate Equity")
            values.append(float(summary.real_estate_equity))
            measures.append("relative")

        # Liabilities
        if summary.total_liabilities > 0:
            categories.append("Liabilities")
            values.append(-float(summary.total_liabilities))
            measures.append("relative")

        # Net Worth Total
        categories.append("Net Worth")
        values.append(float(summary.total_net_worth))
        measures.append("total")

        fig = go.Figure(
            go.Waterfall(
                name="",
                orientation="v",
                measure=measures,
                x=categories,
                y=values,
                textposition="outside",
                text=[f"${v:,.0f}" if v >= 0 else f"-${abs(v):,.0f}" for v in values],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#2ecc71"}},
                decreasing={"marker": {"color": "#e74c3c"}},
                totals={"marker": {"color": "#3498db"}},
            )
        )

        fig.update_layout(
            title=dict(text="Net Worth Breakdown", x=0.5, xanchor="center"),
            yaxis_title="Value ($)",
            template=self.theme,
            height=450,
            showlegend=False,
        )

        return fig

    def create_account_summary_table(self, portfolio: UserPortfolio) -> go.Figure:
        """Create table showing account summary.

        Args:
            portfolio: User portfolio data.

        Returns:
            Plotly figure with table.
        """
        accounts = []
        types = []
        institutions = []
        balances = []

        for acc in portfolio.bank_accounts:
            accounts.append(acc.account_id[:12] + "...")
            types.append(acc.account_type.value.title())
            institutions.append(acc.bank_name)
            balances.append(f"${acc.balance:,.2f}")

        for acc in portfolio.investment_accounts:
            accounts.append(acc.account_id[:12] + "...")
            types.append(acc.account_type.value.upper())
            institutions.append(acc.institution)
            balances.append(f"${acc.total_value:,.2f}")

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=["Account ID", "Type", "Institution", "Balance"],
                        fill_color="#3498db",
                        font=dict(color="white", size=12),
                        align="left",
                    ),
                    cells=dict(
                        values=[accounts, types, institutions, balances],
                        fill_color=[["#f8f9fa", "#ffffff"] * (len(accounts) // 2 + 1)],
                        align="left",
                        height=30,
                    ),
                )
            ]
        )

        fig.update_layout(
            title=dict(text="Account Summary", x=0.5, xanchor="center"),
            height=300,
        )

        return fig

    def create_holdings_by_sector(
        self,
        holdings: list[Holding],
        title: str = "Holdings by Sector",
    ) -> go.Figure:
        """Create stacked bar chart showing holdings grouped by sector.

        Args:
            holdings: List of holdings.
            title: Chart title.

        Returns:
            Plotly figure.
        """
        if not holdings:
            return self._empty_figure("No holdings data")

        # Group holdings by sector
        sector_data: dict[str, dict] = {}
        for h in holdings:
            sector = h.sector or "Other"
            if sector not in sector_data:
                sector_data[sector] = {"value": 0, "holdings": []}
            sector_data[sector]["value"] += float(h.market_value)
            sector_data[sector]["holdings"].append(h)

        # Sort by value
        sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1]["value"], reverse=True)

        sectors = []
        values = []
        hover_texts = []

        for sector, data in sorted_sectors:
            sectors.append(sector)
            values.append(data["value"])
            # Create hover text with holdings breakdown
            holdings_list = sorted(data["holdings"], key=lambda x: x.market_value, reverse=True)
            holdings_str = "<br>".join(
                [f"  {h.symbol}: ${float(h.market_value):,.0f}" for h in holdings_list[:5]]
            )
            if len(holdings_list) > 5:
                holdings_str += f"<br>  ...and {len(holdings_list) - 5} more"
            hover_texts.append(f"<b>{sector}</b><br>Total: ${data['value']:,.0f}<br>{holdings_str}")

        fig = go.Figure(
            data=[
                go.Bar(
                    x=values,
                    y=sectors,
                    orientation="h",
                    marker_color=self.color_palette[:len(sectors)],
                    text=[f"${v:,.0f}" for v in values],
                    textposition="auto",
                    hovertext=hover_texts,
                    hoverinfo="text",
                )
            ]
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_title="Market Value ($)",
            yaxis_title="Sector",
            template=self.theme,
            height=400,
            showlegend=False,
        )

        return fig

    def create_gain_loss_chart(
        self,
        holdings: list[Holding],
        title: str = "Unrealized Gains/Losses",
    ) -> go.Figure:
        """Create horizontal bar chart for gains/losses.

        Args:
            holdings: List of holdings.
            title: Chart title.

        Returns:
            Plotly figure.
        """
        if not holdings:
            return self._empty_figure("No holdings data")

        # Sort by gain/loss
        sorted_holdings = sorted(
            holdings, key=lambda h: h.unrealized_gain_loss, reverse=True
        )

        symbols = [h.symbol for h in sorted_holdings]
        gains = [float(h.unrealized_gain_loss) for h in sorted_holdings]
        colors = ["#2ecc71" if g >= 0 else "#e74c3c" for g in gains]

        fig = go.Figure(
            data=[
                go.Bar(
                    y=symbols,
                    x=gains,
                    orientation="h",
                    marker_color=colors,
                    text=[f"${g:+,.0f}" for g in gains],
                    textposition="outside",
                )
            ]
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_title="Unrealized Gain/Loss ($)",
            yaxis_title="Symbol",
            template=self.theme,
            height=max(300, len(holdings) * 30),
        )

        return fig

    def create_metrics_dashboard(
        self,
        metrics: dict[str, Any],
        title: str = "Portfolio Metrics",
    ) -> go.Figure:
        """Create indicator dashboard for key metrics.

        Args:
            metrics: Dict of metric name to value.
            title: Dashboard title.

        Returns:
            Plotly figure.
        """
        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [{"type": "indicator"}] * 3,
                [{"type": "indicator"}] * 3,
            ],
            vertical_spacing=0.3,
        )

        # Net Worth
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=float(metrics.get("net_worth", 0)),
                title={"text": "Net Worth"},
                number={"prefix": "$", "valueformat": ",.0f"},
            ),
            row=1,
            col=1,
        )

        # Liquidity Ratio
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.get("liquidity_ratio", 0) * 100,
                title={"text": "Liquidity Ratio"},
                number={"suffix": "%", "valueformat": ".1f"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#3498db"},
                    "steps": [
                        {"range": [0, 20], "color": "#e74c3c"},
                        {"range": [20, 50], "color": "#f39c12"},
                        {"range": [50, 100], "color": "#2ecc71"},
                    ],
                },
            ),
            row=1,
            col=2,
        )

        # Savings Rate
        savings_rate = metrics.get("monthly_savings_rate", 0) or 0
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=savings_rate * 100,
                title={"text": "Savings Rate"},
                number={"suffix": "%", "valueformat": ".1f"},
                gauge={
                    "axis": {"range": [0, 50]},
                    "bar": {"color": "#2ecc71"},
                },
            ),
            row=1,
            col=3,
        )

        # Total Assets
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=float(metrics.get("total_assets", 0)),
                title={"text": "Total Assets"},
                number={"prefix": "$", "valueformat": ",.0f"},
            ),
            row=2,
            col=1,
        )

        # Unrealized Gain
        gain = float(metrics.get("total_unrealized_gain", 0))
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=gain,
                title={"text": "Unrealized Gain/Loss"},
                number={"prefix": "$", "valueformat": "+,.0f"},
                delta={
                    "reference": 0,
                    "valueformat": "+,.0f",
                    "relative": False,
                },
            ),
            row=2,
            col=2,
        )

        # Debt Ratio
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=metrics.get("debt_to_asset_ratio", 0) * 100,
                title={"text": "Debt-to-Asset Ratio"},
                number={"suffix": "%", "valueformat": ".1f"},
            ),
            row=2,
            col=3,
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            template=self.theme,
            height=450,
        )

        return fig

    def _empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            template=self.theme,
            height=300,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig
