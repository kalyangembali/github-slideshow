import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

os.makedirs(CACHE_DIR, exist_ok=True)

# --- STOCK UNIVERSE ---
# Expanded to full NIFTY 50 universe (50 stocks) + 3 ETFs as requested
TRADING_STOCKS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS",
    "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "LTIM.NS", "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS",
    "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
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
