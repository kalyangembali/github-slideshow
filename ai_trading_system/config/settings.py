import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

os.makedirs(CACHE_DIR, exist_ok=True)

# --- STOCK UNIVERSE ---
# The 15 core stocks across 5 sectors, plus 3 ETFs
TRADING_STOCKS = [
    # BANKING
    "HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS",
    # IT
    "TCS.NS", "INFY.NS", "WIPRO.NS",
    # OIL & ENERGY
    "RELIANCE.NS", "ONGC.NS",
    # PHARMA
    "SUNPHARMA.NS", "DRREDDY.NS",
    # FMCG + AUTO
    "HINDUNILVR.NS", "MARUTI.NS",
    # ETFs
    "NIFTYBEES.NS", "BANKBEES.NS", "JUNIORBEES.NS"
]

MARKET_CONTEXT_SYMBOLS = [
    "^NSEI",        # NIFTY50 index
    "^NSEBANK",     # BANKNIFTY index
    "^INDIAVIX",    # India VIX (volatility)
    "^GSPC",        # S&P500 (global context)
    "^IXIC",        # NASDAQ (tech context)
    "GC=F",         # Gold futures
    "CL=F",         # Crude Oil futures
    "DX-Y.NYB"      # US Dollar Index
]

ALL_SYMBOLS = TRADING_STOCKS + MARKET_CONTEXT_SYMBOLS

# --- HYPERPARAMETERS & RULES ---
SWING_ATR_MULTIPLIER = 1.5
CAPITAL = 500000
MAX_RISK_PER_TRADE = 0.02
MAX_CAPITAL_PER_STOCK = 0.15

# Training Data range parameters (for YFinance limits)
START_DATE_DAILY = "2018-01-01"
# For dry run/testing, limit the scope so yfinance download is fast
TESTING_START_DATE = "2023-01-01"
