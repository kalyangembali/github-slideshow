import pandas as pd
import numpy as np
from loguru import logger

def calculate_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for Market Microstructure features (Order Book Dynamics).
    This script is designed for Live Intraday Trading (Phase 5+) where
    tick-by-tick or 1-minute order book data is available via a broker API
    (e.g., Zerodha Kite Connect).

    If the required columns ('Bid', 'Ask', 'Total_Buy_Qty', 'Total_Sell_Qty')
    are missing (because we are using yfinance), it creates empty/zeroed features
    to maintain pipeline compatibility.
    """
    df = df.copy()

    # Check if we have Level 2 data
    has_l2_data = all(col in df.columns for col in ['Bid', 'Ask', 'Total_Buy_Qty', 'Total_Sell_Qty'])

    if has_l2_data:
        # 1. Bid-Ask Spread
        # Wide spreads indicate low liquidity -> Higher slippage risk
        df['bid_ask_spread'] = df['Ask'] - df['Bid']

        # Relative Spread (Spread as a % of price)
        df['relative_spread_bps'] = (df['bid_ask_spread'] / df['Close']) * 10000

        # 2. Order Imbalance
        # (Total Buy Orders - Total Sell Orders) / Total Orders
        # Positive = Buying Pressure, Negative = Selling Pressure
        total_orders = df['Total_Buy_Qty'] + df['Total_Sell_Qty']
        # Avoid division by zero
        df['order_imbalance'] = np.where(total_orders == 0, 0,
                                        (df['Total_Buy_Qty'] - df['Total_Sell_Qty']) / total_orders)

        # Imbalance Momentum (Is buying pressure increasing?)
        df['imbalance_momentum'] = df['order_imbalance'].diff(1)

        logger.info("Computed Microstructure Features from Level 2 data.")
    else:
        # Fallback for yfinance / historical data
        # We fill with 0 so the model pipeline (XGBoost/RF) doesn't break
        # if it expects exactly 55 features.
        df['bid_ask_spread'] = 0.0
        df['relative_spread_bps'] = 0.0
        df['order_imbalance'] = 0.0
        df['imbalance_momentum'] = 0.0

        # A simple proxy for buying pressure using volume if we only have OHLCV
        if 'Close' in df.columns and 'Open' in df.columns and 'Volume' in df.columns:
            # If close > open, assume volume was mostly buying
            buying_volume = np.where(df['Close'] > df['Open'], df['Volume'],
                                     np.where(df['Close'] < df['Open'], 0, df['Volume'] * 0.5))
            # Avoid division by zero
            df['buying_pressure'] = np.where(df['Volume'] == 0, 0, buying_volume / df['Volume'])

    return df
