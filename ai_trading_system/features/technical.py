import pandas as pd
import ta
import numpy as np

def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RSI, MACD, ATR, EMA, Bollinger Bands using the `ta` library."""

    # We must have close, high, low, volume to compute indicators
    required_cols = ['Close', 'High', 'Low', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return df

    df = df.copy()

    # Fill missing values if any exist from yfinance
    df = df.ffill().bfill()

    # --- PRICE & RETURNS ---
    df['returns_1'] = df['Close'].pct_change(1)
    df['returns_5'] = df['Close'].pct_change(5)
    df['returns_10'] = df['Close'].pct_change(10)

    # Handle division by zero
    high_low_range = df['High'] - df['Low']
    df['high_low_range'] = high_low_range
    df['close_position_in_range'] = np.where(high_low_range == 0, 0.5, (df['Close'] - df['Low']) / high_low_range)

    df['price_vs_high20'] = df['Close'] / df['High'].rolling(20).max()
    df['price_vs_low20'] = df['Close'] / df['Low'].rolling(20).min()

    # --- MOMENTUM ---
    # RSI
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['RSI_9'] = ta.momentum.RSIIndicator(close=df['Close'], window=9).rsi()

    # MACD
    macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_line'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_histogram'] = macd.macd_diff()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['Stochastic_K'] = stoch.stoch()
    df['Stochastic_D'] = stoch.stoch_signal()

    # CCI & ROC
    df['CCI_20'] = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20).cci()
    df['ROC_10'] = ta.momentum.ROCIndicator(close=df['Close'], window=10).roc()

    # EMA Alignment
    df['EMA_9'] = ta.trend.EMAIndicator(close=df['Close'], window=9).ema_indicator()
    df['EMA_21'] = ta.trend.EMAIndicator(close=df['Close'], window=21).ema_indicator()
    df['EMA_alignment_score'] = np.where(df['EMA_9'] > df['EMA_21'], 1, -1)

    # --- VOLATILITY ---
    # ATR
    atr_14 = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR_14'] = atr_14.average_true_range()

    # To handle 0 ATR safely
    atr_rolling_mean = df['ATR_14'].rolling(20).mean()
    df['ATR_ratio'] = np.where(atr_rolling_mean == 0, 1.0, df['ATR_14'] / atr_rolling_mean)

    # --- VOLUME ---
    # Avoid zero division
    vol_mean_20 = df['Volume'].rolling(20).mean()
    df['volume_ratio_20'] = np.where(vol_mean_20 == 0, 1.0, df['Volume'] / vol_mean_20)

    vol_mean_5 = df['Volume'].rolling(5).mean()
    df['volume_ratio_5'] = np.where(vol_mean_5 == 0, 1.0, df['Volume'] / vol_mean_5)

    # OBV
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['OBV_trend'] = np.where(df['OBV'] > df['OBV'].rolling(10).mean(), 1, -1)

    # VWAP (Approximation for daily bars: Typical Price * Volume)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP_approx'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['VWAP_distance_pct'] = (df['Close'] - df['VWAP_approx']) / df['VWAP_approx']

    df['unusual_volume_flag'] = np.where(df['volume_ratio_20'] > 2.0, 1, 0)

    # Clean up NaNs created by rolling windows
    df = df.dropna()

    return df

if __name__ == "__main__":
    from config import settings
    from data.fetcher import YFinanceFetcher

    fetcher = YFinanceFetcher()
    test_df = fetcher.fetch_symbol("RELIANCE.NS", force_refresh=False)

    if not test_df.empty:
        features_df = calculate_technical_features(test_df)
        print(f"Features created successfully. Columns: {len(features_df.columns)}")
        print(features_df[['Close', 'RSI_14', 'MACD_line', 'ATR_14', 'VWAP_distance_pct']].tail())
