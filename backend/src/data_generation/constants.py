"""Constants for synthetic data generation."""

# Major US Banks
BANKS = [
    "Chase",
    "Bank of America",
    "Wells Fargo",
    "Citibank",
    "US Bank",
    "PNC Bank",
    "Capital One",
    "TD Bank",
    "Truist",
    "Ally Bank",
]

# Investment Institutions
INSTITUTIONS = [
    "Fidelity",
    "Vanguard",
    "Charles Schwab",
    "TD Ameritrade",
    "E*TRADE",
    "Merrill Lynch",
    "Morgan Stanley",
    "Interactive Brokers",
    "Robinhood",
    "Wealthfront",
]

# Popular Stocks with sectors
STOCKS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology", "price_range": (150, 250)},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology", "price_range": (350, 450)},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology", "price_range": (140, 180)},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical", "price_range": (170, 220)},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology", "price_range": (800, 1200)},
    {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technology", "price_range": (450, 600)},
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Cyclical", "price_range": (200, 350)},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financial Services", "price_range": (180, 220)},
    {"symbol": "V", "name": "Visa Inc.", "sector": "Financial Services", "price_range": (270, 320)},
    {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "price_range": (150, 180)},
    {"symbol": "UNH", "name": "UnitedHealth Group", "sector": "Healthcare", "price_range": (500, 600)},
    {"symbol": "PG", "name": "Procter & Gamble Co.", "sector": "Consumer Defensive", "price_range": (160, 180)},
    {"symbol": "HD", "name": "The Home Depot Inc.", "sector": "Consumer Cyclical", "price_range": (350, 420)},
    {"symbol": "MA", "name": "Mastercard Inc.", "sector": "Financial Services", "price_range": (450, 520)},
    {"symbol": "XOM", "name": "Exxon Mobil Corporation", "sector": "Energy", "price_range": (100, 130)},
    {"symbol": "CVX", "name": "Chevron Corporation", "sector": "Energy", "price_range": (140, 170)},
    {"symbol": "PFE", "name": "Pfizer Inc.", "sector": "Healthcare", "price_range": (25, 40)},
    {"symbol": "KO", "name": "The Coca-Cola Company", "sector": "Consumer Defensive", "price_range": (55, 70)},
    {"symbol": "PEP", "name": "PepsiCo Inc.", "sector": "Consumer Defensive", "price_range": (165, 190)},
    {"symbol": "DIS", "name": "The Walt Disney Company", "sector": "Communication Services", "price_range": (90, 130)},
]

# Popular ETFs
ETFS = [
    {"symbol": "VOO", "name": "Vanguard S&P 500 ETF", "sector": "Broad Market", "price_range": (450, 520)},
    {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF", "sector": "Broad Market", "price_range": (250, 290)},
    {"symbol": "QQQ", "name": "Invesco QQQ Trust", "sector": "Technology", "price_range": (450, 530)},
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "sector": "Broad Market", "price_range": (480, 550)},
    {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "sector": "Small Cap", "price_range": (190, 230)},
    {"symbol": "VGT", "name": "Vanguard Information Technology ETF", "sector": "Technology", "price_range": (500, 580)},
    {"symbol": "VHT", "name": "Vanguard Health Care ETF", "sector": "Healthcare", "price_range": (250, 290)},
    {"symbol": "VNQ", "name": "Vanguard Real Estate ETF", "sector": "Real Estate", "price_range": (80, 100)},
    {"symbol": "BND", "name": "Vanguard Total Bond Market ETF", "sector": "Bonds", "price_range": (70, 80)},
    {"symbol": "SCHD", "name": "Schwab US Dividend Equity ETF", "sector": "Dividend", "price_range": (75, 85)},
]

# Bonds / Bond ETFs
BONDS = [
    {"symbol": "AGG", "name": "iShares Core US Aggregate Bond ETF", "sector": "Bonds", "price_range": (95, 105)},
    {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "sector": "Treasury", "price_range": (85, 105)},
    {"symbol": "LQD", "name": "iShares iBoxx $ Investment Grade Corp Bond ETF", "sector": "Corporate Bonds", "price_range": (105, 115)},
    {"symbol": "HYG", "name": "iShares iBoxx $ High Yield Corp Bond ETF", "sector": "High Yield Bonds", "price_range": (75, 82)},
    {"symbol": "VCIT", "name": "Vanguard Intermediate-Term Corp Bond ETF", "sector": "Corporate Bonds", "price_range": (78, 85)},
]

# Cryptocurrencies
CRYPTO = [
    {"symbol": "BTC", "name": "Bitcoin", "sector": "Cryptocurrency", "price_range": (60000, 100000)},
    {"symbol": "ETH", "name": "Ethereum", "sector": "Cryptocurrency", "price_range": (3000, 4500)},
    {"symbol": "SOL", "name": "Solana", "sector": "Cryptocurrency", "price_range": (150, 250)},
    {"symbol": "ADA", "name": "Cardano", "sector": "Cryptocurrency", "price_range": (0.4, 0.8)},
    {"symbol": "DOT", "name": "Polkadot", "sector": "Cryptocurrency", "price_range": (6, 12)},
]

# Mutual Funds
MUTUAL_FUNDS = [
    {"symbol": "FXAIX", "name": "Fidelity 500 Index Fund", "sector": "Broad Market", "price_range": (180, 210)},
    {"symbol": "VFIAX", "name": "Vanguard 500 Index Fund Admiral", "sector": "Broad Market", "price_range": (450, 520)},
    {"symbol": "VTSAX", "name": "Vanguard Total Stock Market Index Admiral", "sector": "Broad Market", "price_range": (120, 140)},
    {"symbol": "FSKAX", "name": "Fidelity Total Market Index Fund", "sector": "Broad Market", "price_range": (140, 160)},
    {"symbol": "VBTLX", "name": "Vanguard Total Bond Market Index Admiral", "sector": "Bonds", "price_range": (9, 11)},
]

# Market Sectors
SECTORS = [
    "Technology",
    "Healthcare",
    "Financial Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Industrials",
    "Communication Services",
    "Utilities",
    "Real Estate",
    "Materials",
]

# US States for addresses
US_STATES = [
    "California", "Texas", "Florida", "New York", "Pennsylvania",
    "Illinois", "Ohio", "Georgia", "North Carolina", "Michigan",
    "New Jersey", "Virginia", "Washington", "Arizona", "Massachusetts",
    "Tennessee", "Indiana", "Missouri", "Maryland", "Wisconsin",
    "Colorado", "Minnesota", "South Carolina", "Alabama", "Louisiana",
]

# Cities for addresses (major metros)
CITIES = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"),
    ("Houston", "TX"), ("Phoenix", "AZ"), ("Philadelphia", "PA"),
    ("San Antonio", "TX"), ("San Diego", "CA"), ("Dallas", "TX"),
    ("San Jose", "CA"), ("Austin", "TX"), ("Jacksonville", "FL"),
    ("Fort Worth", "TX"), ("Columbus", "OH"), ("Charlotte", "NC"),
    ("San Francisco", "CA"), ("Indianapolis", "IN"), ("Seattle", "WA"),
    ("Denver", "CO"), ("Boston", "MA"), ("Nashville", "TN"),
    ("Atlanta", "GA"), ("Portland", "OR"), ("Miami", "FL"),
]

# Occupations with typical income ranges
OCCUPATIONS = [
    {"title": "Software Engineer", "income_range": (80000, 200000)},
    {"title": "Data Scientist", "income_range": (90000, 180000)},
    {"title": "Product Manager", "income_range": (100000, 200000)},
    {"title": "Financial Analyst", "income_range": (60000, 120000)},
    {"title": "Marketing Manager", "income_range": (70000, 150000)},
    {"title": "Physician", "income_range": (200000, 400000)},
    {"title": "Attorney", "income_range": (80000, 250000)},
    {"title": "Accountant", "income_range": (50000, 100000)},
    {"title": "Teacher", "income_range": (40000, 80000)},
    {"title": "Nurse", "income_range": (60000, 100000)},
    {"title": "Sales Manager", "income_range": (70000, 150000)},
    {"title": "Consultant", "income_range": (80000, 200000)},
    {"title": "Entrepreneur", "income_range": (50000, 500000)},
    {"title": "Real Estate Agent", "income_range": (40000, 150000)},
    {"title": "Retired", "income_range": (30000, 100000)},
]

# Savings rate ranges by age group (percentage of income)
SAVINGS_RATES = {
    "20-30": (0.05, 0.20),
    "30-40": (0.10, 0.25),
    "40-50": (0.15, 0.30),
    "50-60": (0.15, 0.35),
    "60+": (0.10, 0.25),
}

# Interest rates for bank accounts
INTEREST_RATES = {
    "checking": (0.0, 0.01),
    "savings": (0.03, 0.05),
    "money_market": (0.04, 0.055),
}
