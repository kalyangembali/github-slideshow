import pandas as pd
import numpy as np

def calculate_global_context(data_dict: dict) -> pd.DataFrame:
    """Calculates S&P500, NASDAQ, Gold, Crude, and DXY context features."""

    # Need basic global indicators
    global_symbols = {
        "^GSPC": "sp500",
        "^IXIC": "nasdaq",
        "GC=F": "gold",
        "CL=F": "crude",
        "DX-Y.NYB": "dxy"
    }

    context_df = None

    for symbol, prefix in global_symbols.items():
        if symbol in data_dict and not data_dict[symbol].empty:
            df = data_dict[symbol][['Close']].copy()
            df = df.rename(columns={'Close': f'{prefix}_close'})
            df[f'{prefix}_return_1d'] = df[f'{prefix}_close'].pct_change(1)

            if prefix in ['sp500', 'nasdaq']:
                # Trend score: 1 if above 20MA else -1
                ma20 = df[f'{prefix}_close'].rolling(20).mean()
                df[f'{prefix}_trend'] = np.where(df[f'{prefix}_close'] > ma20, 1, -1)

            df = df.drop(columns=[f'{prefix}_close'])

            if context_df is None:
                context_df = df
            else:
                context_df = context_df.join(df, how='outer')

    if context_df is not None:
        context_df = context_df.ffill().bfill()

    return context_df
