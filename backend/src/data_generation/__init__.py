"""Synthetic data generation for the Agentic Wealth Intelligence system."""

from backend.src.data_generation.generator import (
    SyntheticDataGenerator,
    generate_sample_data,
)
from backend.src.data_generation.document_generator import (
    generate_all_sample_documents,
    generate_etf_factsheet,
    generate_fomc_summary,
    generate_investment_guide,
    get_sample_documents_by_type,
    ETF_DATA,
    FOMC_SUMMARIES,
    INVESTMENT_GUIDES,
)

__all__ = [
    "SyntheticDataGenerator",
    "generate_sample_data",
    "generate_all_sample_documents",
    "generate_etf_factsheet",
    "generate_fomc_summary",
    "generate_investment_guide",
    "get_sample_documents_by_type",
    "ETF_DATA",
    "FOMC_SUMMARIES",
    "INVESTMENT_GUIDES",
]
