import pandas as pd
import numpy as np

def calculate_sector_context(data_dict: dict) -> pd.DataFrame:
    """Calculates sector ETF strength and ranker."""

    sector_etfs = {
        "NIFTYBEES.NS": "index",
        "BANKBEES.NS": "banking",
        "JUNIORBEES.NS": "midcap"
    }

    context_df = None

    for symbol, sector_name in sector_etfs.items():
        if symbol in data_dict and not data_dict[symbol].empty:
            df = data_dict[symbol][['Close']].copy()
            df = df.rename(columns={'Close': f'{sector_name}_close'})

            # Momentum/Return over 5 days
            df[f'{sector_name}_return_1d'] = df[f'{sector_name}_close'].pct_change(1)
            df[f'{sector_name}_momentum_5d'] = df[f'{sector_name}_close'].pct_change(5)

            df = df.drop(columns=[f'{sector_name}_close'])

            if context_df is None:
                context_df = df
            else:
                context_df = context_df.join(df, how='outer')

    if context_df is not None:
        context_df = context_df.ffill().bfill()

    return context_df
