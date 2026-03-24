"""Synthetic financial document generator for RAG system testing."""

import uuid
from datetime import datetime, timedelta

from ..recommendation.rag.schemas import Document, DocumentMetadata, DocumentType


# Sample ETF data for generating fact sheets
ETF_DATA = [
    {
        "ticker": "VOO",
        "name": "Vanguard S&P 500 ETF",
        "source": "Vanguard",
        "expense_ratio": "0.03%",
        "aum": "$350 billion",
        "sector": "Broad Market",
        "holdings": "503 stocks",
        "description": "Tracks the S&P 500 Index, providing exposure to 500 of the largest U.S. companies.",
    },
    {
        "ticker": "VTI",
        "name": "Vanguard Total Stock Market ETF",
        "source": "Vanguard",
        "expense_ratio": "0.03%",
        "aum": "$300 billion",
        "sector": "Broad Market",
        "holdings": "3,945 stocks",
        "description": "Provides broad exposure to the entire U.S. equity market, including small, mid, and large-cap stocks.",
    },
    {
        "ticker": "QQQ",
        "name": "Invesco QQQ Trust",
        "source": "Invesco",
        "expense_ratio": "0.20%",
        "aum": "$200 billion",
        "sector": "Technology",
        "holdings": "100 stocks",
        "description": "Tracks the Nasdaq-100 Index, focusing on the 100 largest non-financial companies listed on the Nasdaq.",
    },
    {
        "ticker": "VGT",
        "name": "Vanguard Information Technology ETF",
        "source": "Vanguard",
        "expense_ratio": "0.10%",
        "aum": "$55 billion",
        "sector": "Technology",
        "holdings": "316 stocks",
        "description": "Provides exposure to companies in the information technology sector.",
    },
    {
        "ticker": "BND",
        "name": "Vanguard Total Bond Market ETF",
        "source": "Vanguard",
        "expense_ratio": "0.03%",
        "aum": "$90 billion",
        "sector": "Fixed Income",
        "holdings": "10,000+ bonds",
        "description": "Broad exposure to U.S. investment-grade bonds including government, corporate, and mortgage-backed securities.",
    },
    {
        "ticker": "VXUS",
        "name": "Vanguard Total International Stock ETF",
        "source": "Vanguard",
        "expense_ratio": "0.07%",
        "aum": "$60 billion",
        "sector": "International",
        "holdings": "7,800+ stocks",
        "description": "Exposure to stocks from developed and emerging markets outside the United States.",
    },
    {
        "ticker": "VNQ",
        "name": "Vanguard Real Estate ETF",
        "source": "Vanguard",
        "expense_ratio": "0.12%",
        "aum": "$35 billion",
        "sector": "Real Estate",
        "holdings": "160 stocks",
        "description": "Invests in real estate investment trusts (REITs) that purchase office buildings, hotels, and other properties.",
    },
    {
        "ticker": "SCHD",
        "name": "Schwab U.S. Dividend Equity ETF",
        "source": "Charles Schwab",
        "expense_ratio": "0.06%",
        "aum": "$50 billion",
        "sector": "Dividend",
        "holdings": "100 stocks",
        "description": "Focuses on high-quality U.S. companies with consistent dividend payments.",
    },
]


def generate_etf_factsheet(etf_data: dict, doc_id: str | None = None) -> Document:
    """Generate a synthetic ETF fact sheet document.

    Args:
        etf_data: Dictionary with ETF information.
        doc_id: Optional document ID, auto-generated if not provided.

    Returns:
        Document instance with ETF fact sheet content.
    """
    doc_id = doc_id or f"etf-{etf_data['ticker'].lower()}-{uuid.uuid4().hex[:8]}"

    content = f"""# {etf_data['name']} ({etf_data['ticker']})

## Fund Overview
{etf_data['description']}

## Key Statistics
- **Ticker Symbol**: {etf_data['ticker']}
- **Expense Ratio**: {etf_data['expense_ratio']}
- **Assets Under Management**: {etf_data['aum']}
- **Number of Holdings**: {etf_data['holdings']}
- **Sector Focus**: {etf_data['sector']}

## Investment Objective
The fund seeks to track the performance of its benchmark index, providing investors with diversified exposure to the {etf_data['sector'].lower()} market segment.

## Risk Considerations
- Market risk: The value of the fund may decline due to general market conditions.
- Concentration risk: The fund may be concentrated in a particular sector or industry.
- Liquidity risk: Some securities may be difficult to sell at favorable prices.

## Who Should Invest
This fund is suitable for investors seeking:
- Long-term capital appreciation
- Diversified exposure to {etf_data['sector'].lower()} securities
- Low-cost index investing approach

## Performance History
Past performance does not guarantee future results. The fund has historically tracked its benchmark index closely, with minimal tracking error.

## How to Invest
Shares can be purchased through most brokerage accounts. The fund trades on major exchanges under the ticker symbol {etf_data['ticker']}.
"""

    metadata = DocumentMetadata(
        source=etf_data["source"],
        document_type=DocumentType.ETF_FACTSHEET,
        publish_date=datetime.now() - timedelta(days=30),
        expiry_date=datetime.now() + timedelta(days=365),
        tags=["etf", etf_data["sector"].lower(), "investing"],
        author=f"{etf_data['source']} Research",
        ticker=etf_data["ticker"],
        sector=etf_data["sector"],
    )

    return Document(
        doc_id=doc_id,
        title=f"{etf_data['name']} ({etf_data['ticker']}) - Fact Sheet",
        content=content,
        metadata=metadata,
    )


# Sample FOMC data
FOMC_SUMMARIES = [
    {
        "date": "January 2024",
        "decision": "rates unchanged at 5.25-5.50%",
        "outlook": "cautiously optimistic",
        "key_points": [
            "Inflation has eased but remains above the 2% target",
            "Labor market remains strong with solid job gains",
            "Economic activity expanded at a solid pace",
            "Committee remains data-dependent on future rate decisions",
        ],
    },
    {
        "date": "March 2024",
        "decision": "rates unchanged at 5.25-5.50%",
        "outlook": "balanced",
        "key_points": [
            "Inflation has made progress toward 2% target but remains elevated",
            "Unexpected inflation persistence could delay rate cuts",
            "Consumer spending remains resilient",
            "Housing market showing signs of recovery",
        ],
    },
    {
        "date": "May 2024",
        "decision": "rates unchanged at 5.25-5.50%",
        "outlook": "patient",
        "key_points": [
            "Need greater confidence inflation moving sustainably toward 2%",
            "Economic growth moderated slightly from Q4",
            "Financial conditions have tightened somewhat",
            "Prepared to adjust stance as economic outlook evolves",
        ],
    },
]


def generate_fomc_summary(fomc_data: dict, doc_id: str | None = None) -> Document:
    """Generate a synthetic FOMC meeting summary document.

    Args:
        fomc_data: Dictionary with FOMC meeting information.
        doc_id: Optional document ID, auto-generated if not provided.

    Returns:
        Document instance with FOMC summary content.
    """
    doc_id = doc_id or f"fomc-{fomc_data['date'].lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"

    key_points_text = "\n".join(f"- {point}" for point in fomc_data["key_points"])

    content = f"""# Federal Reserve FOMC Meeting Summary - {fomc_data['date']}

## Policy Decision
The Federal Open Market Committee decided to keep the federal funds rate {fomc_data['decision']}.

## Economic Outlook
The Committee's outlook is {fomc_data['outlook']} based on recent economic data.

## Key Takeaways
{key_points_text}

## Market Implications
- **Equities**: The decision suggests continued monetary policy support for risk assets in the near term.
- **Bonds**: Treasury yields may remain stable with the current rate outlook.
- **Dollar**: USD may see modest movements based on rate expectations.

## What This Means for Investors
The Federal Reserve's patient approach indicates:
1. Interest rates likely to remain elevated in the near term
2. Opportunity for higher yields in savings and fixed income
3. Continued importance of diversification across asset classes

## Looking Ahead
The Committee will continue to monitor incoming economic data and adjust policy as appropriate. Key indicators to watch include:
- Monthly inflation reports (CPI, PCE)
- Employment situation
- Consumer spending trends
- Housing market data

The next FOMC meeting will provide further guidance on the policy path.
"""

    # Parse date for metadata
    month_year = fomc_data["date"].split()
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    publish_date = datetime(int(month_year[1]), month_map[month_year[0]], 15)

    metadata = DocumentMetadata(
        source="Federal Reserve",
        document_type=DocumentType.FOMC_MINUTES,
        publish_date=publish_date,
        expiry_date=publish_date + timedelta(days=90),
        tags=["fomc", "monetary policy", "interest rates", "federal reserve"],
        author="Federal Open Market Committee",
    )

    return Document(
        doc_id=doc_id,
        title=f"FOMC Meeting Summary - {fomc_data['date']}",
        content=content,
        metadata=metadata,
    )


# Investment guide topics
INVESTMENT_GUIDES = [
    {
        "topic": "Asset Allocation",
        "subtitle": "Building a Diversified Portfolio",
        "content_sections": [
            ("What is Asset Allocation?", "Asset allocation is the process of dividing investments among different asset categories, such as stocks, bonds, and cash. This strategy helps manage risk and optimize returns based on your investment goals, time horizon, and risk tolerance."),
            ("The Importance of Diversification", "Diversification reduces portfolio volatility by spreading investments across various asset classes that don't move in perfect correlation. When one asset class underperforms, others may compensate, helping to smooth overall returns."),
            ("Strategic vs Tactical Allocation", "Strategic allocation sets long-term targets based on your investment profile. Tactical allocation makes short-term adjustments to capitalize on market opportunities while staying within risk parameters."),
            ("Common Allocation Models", "Conservative portfolios typically hold 20-40% stocks, 50-70% bonds. Moderate portfolios balance around 50-60% stocks, 30-40% bonds. Aggressive portfolios may hold 80-90% stocks with minimal bonds."),
            ("Rebalancing Your Portfolio", "Regular rebalancing ensures your portfolio stays aligned with target allocations. Consider rebalancing when allocations drift more than 5% from targets, or at least annually."),
        ],
    },
    {
        "topic": "Retirement Planning",
        "subtitle": "Strategies for Long-Term Wealth Building",
        "content_sections": [
            ("Starting Early", "The power of compound interest makes starting early crucial. Even small contributions in your 20s and 30s can grow significantly over decades. Time in the market often matters more than timing the market."),
            ("Tax-Advantaged Accounts", "Maximize contributions to 401(k)s, IRAs, and Roth accounts. These vehicles provide tax benefits that accelerate wealth building. For 2024, 401(k) limits are $23,000 ($30,500 if 50+)."),
            ("Employer Matching", "Always contribute enough to capture the full employer match in your 401(k). This is essentially free money with an immediate 100% return on your contribution."),
            ("Target-Date Funds", "These funds automatically adjust asset allocation as you approach retirement, becoming more conservative over time. They offer a simple, hands-off approach to retirement investing."),
            ("Social Security Planning", "Delaying Social Security benefits until age 70 can increase monthly payments by up to 77% compared to claiming at 62. Consider this in your overall retirement income strategy."),
        ],
    },
    {
        "topic": "Risk Management",
        "subtitle": "Protecting Your Investment Portfolio",
        "content_sections": [
            ("Understanding Risk Types", "Market risk affects all investments. Specific risk relates to individual securities. Inflation risk erodes purchasing power. Interest rate risk impacts bond prices. Currency risk affects international investments."),
            ("Risk Tolerance Assessment", "Your risk tolerance depends on your time horizon, financial goals, income stability, and emotional comfort with volatility. Be honest about how you'd react to significant portfolio declines."),
            ("Hedging Strategies", "Options and inverse ETFs can protect against downside risk. Dollar-cost averaging reduces timing risk. Holding cash provides a buffer during market downturns."),
            ("Emergency Fund Importance", "Maintain 3-6 months of expenses in liquid savings. This prevents the need to sell investments at unfavorable times during personal financial emergencies."),
            ("Portfolio Stress Testing", "Evaluate how your portfolio would perform in various scenarios: 2008-style crash, rising interest rates, high inflation. Adjust allocations if potential losses exceed your tolerance."),
        ],
    },
]


def generate_investment_guide(guide_data: dict, doc_id: str | None = None) -> Document:
    """Generate a synthetic investment guide document.

    Args:
        guide_data: Dictionary with guide topic and sections.
        doc_id: Optional document ID, auto-generated if not provided.

    Returns:
        Document instance with investment guide content.
    """
    doc_id = doc_id or f"guide-{guide_data['topic'].lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"

    sections_text = ""
    for section_title, section_content in guide_data["content_sections"]:
        sections_text += f"\n## {section_title}\n{section_content}\n"

    content = f"""# {guide_data['topic']}: {guide_data['subtitle']}

{sections_text}

## Key Takeaways
- Understand your personal financial situation and goals
- Diversification is essential for managing risk
- Regular review and rebalancing keeps your strategy on track
- Consider consulting with a financial advisor for personalized guidance

## Disclaimer
This guide is for educational purposes only and should not be considered personalized investment advice. Past performance does not guarantee future results. Consult with a qualified financial professional before making investment decisions.
"""

    metadata = DocumentMetadata(
        source="Wealth Intelligence Research",
        document_type=DocumentType.INVESTMENT_GUIDE,
        publish_date=datetime.now() - timedelta(days=60),
        expiry_date=datetime.now() + timedelta(days=365 * 2),
        tags=["guide", guide_data["topic"].lower(), "education", "investing"],
        author="Wealth Intelligence Team",
    )

    return Document(
        doc_id=doc_id,
        title=f"{guide_data['topic']}: {guide_data['subtitle']}",
        content=content,
        metadata=metadata,
    )


def generate_all_sample_documents() -> list[Document]:
    """Generate all sample financial documents for the RAG system.

    Returns:
        List of Document instances.
    """
    documents = []

    # Generate ETF fact sheets
    for etf in ETF_DATA:
        documents.append(generate_etf_factsheet(etf))

    # Generate FOMC summaries
    for fomc in FOMC_SUMMARIES:
        documents.append(generate_fomc_summary(fomc))

    # Generate investment guides
    for guide in INVESTMENT_GUIDES:
        documents.append(generate_investment_guide(guide))

    return documents


def get_sample_documents_by_type(doc_type: DocumentType) -> list[Document]:
    """Get sample documents filtered by type.

    Args:
        doc_type: Type of documents to generate.

    Returns:
        List of Document instances of the specified type.
    """
    all_docs = generate_all_sample_documents()
    return [doc for doc in all_docs if doc.metadata.document_type == doc_type]
