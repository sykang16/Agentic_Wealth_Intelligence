"""UI configuration settings."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic" / "synthetic_portfolios.json"

# UI Settings
PAGE_TITLE = "Wealth Intelligence"
PAGE_ICON = "💰"
LAYOUT = "wide"

# Theme
THEME = {
    "primaryColor": "#1f77b4",
    "backgroundColor": "#ffffff",
    "secondaryBackgroundColor": "#f8f9fa",
    "textColor": "#262730",
}
