import pandas as pd
import numpy as np

def calculate_india_context(data_dict: dict) -> pd.DataFrame:
    """Calculates NIFTY, BANKNIFTY, and IndiaVIX features."""

    context_df = None

    # NIFTY 50
    if "^NSEI" in data_dict and not data_dict["^NSEI"].empty:
        nifty = data_dict["^NSEI"][['Close']].copy()
        nifty['nifty_return_1d'] = nifty['Close'].pct_change(1)
        ema20 = nifty['Close'].ewm(span=20, adjust=False).mean()
        nifty['nifty_trend'] = np.where(nifty['Close'] > ema20, 1, -1)
        nifty = nifty.drop(columns=['Close'])
        context_df = nifty

    # BANKNIFTY
    if "^NSEBANK" in data_dict and not data_dict["^NSEBANK"].empty:
        bank = data_dict["^NSEBANK"][['Close']].copy()
        bank['banknifty_return_1d'] = bank['Close'].pct_change(1)
        bank = bank.drop(columns=['Close'])
        if context_df is None:
            context_df = bank
        else:
            context_df = context_df.join(bank, how='outer')

    # India VIX
    if "^INDIAVIX" in data_dict and not data_dict["^INDIAVIX"].empty:
        vix = data_dict["^INDIAVIX"][['Close']].copy()
        vix['india_vix_level'] = vix['Close']
        vix['india_vix_change'] = vix['Close'].pct_change(1)
        vix = vix.drop(columns=['Close'])
        if context_df is None:
            context_df = vix
        else:
            context_df = context_df.join(vix, how='outer')

    if context_df is not None:
        context_df = context_df.ffill().bfill()

    return context_df
