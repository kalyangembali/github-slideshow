import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from loguru import logger
import os
import joblib

from config import settings

class MarketRegimeDetector:
    """
    Phase 7+: Unsupervised Market Regime Detection.
    Classifies the current market conditions (Trending vs. Choppy) based on the first
    30 minutes of NIFTY 50 trading data or a rolling window.
    """
    def __init__(self, model_dir=os.path.join(settings.BASE_DIR, "models", "saved")):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        # 2 Clusters: High Volatility Trend (Regime 1), Low Volatility Range (Regime 0)
        self.model = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.is_trained = False

    def _extract_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts volatility and trend features suitable for clustering."""
        df = df.copy()

        # We need sufficient data
        if len(df) < 14:
            return pd.DataFrame()

        # 1. Volatility (ATR normalized)
        df['high_low_range'] = df['High'] - df['Low']
        df['atr_proxy'] = df['high_low_range'].rolling(14).mean()
        df['volatility_normalized'] = df['atr_proxy'] / df['Close']

        # 2. Trend Strength (ADX proxy using absolute returns vs path)
        df['returns_abs_sum'] = df['Close'].pct_change(1).abs().rolling(14).sum()
        df['net_return'] = df['Close'].pct_change(14).abs()

        # Efficiency Ratio (Net Return / Sum of Absolute Returns)
        # High ER = Trending, Low ER = Choppy
        df['efficiency_ratio'] = np.where(df['returns_abs_sum'] == 0, 0,
                                          df['net_return'] / df['returns_abs_sum'])

        df = df.dropna(subset=['volatility_normalized', 'efficiency_ratio'])
        return df[['volatility_normalized', 'efficiency_ratio']]

    def train(self, historical_nifty_df: pd.DataFrame):
        """Train the K-Means clustering model on historical NIFTY data."""
        logger.info("Training Market Regime Detector...")
        features = self._extract_regime_features(historical_nifty_df)

        if features.empty:
            logger.error("Not enough data to train Regime Detector.")
            return False

        self.model.fit(features)
        self.is_trained = True
        logger.info("Regime Detector trained.")
        self.save()
        return True

    def detect_regime(self, current_nifty_df: pd.DataFrame) -> int:
        """
        Detects the current regime based on recent NIFTY data.
        Returns:
            1: Trending / High Volatility
            0: Choppy / Range Bound
            -1: Error / Not enough data
        """
        if not self.is_trained:
            logger.warning("Regime Detector is not trained.")
            return -1

        features = self._extract_regime_features(current_nifty_df)

        if features.empty:
            return -1

        # Predict the regime for the latest candle
        latest_features = features.iloc[-1:]
        regime = self.model.predict(latest_features)[0]
        return regime

    def save(self, name="regime_model.pkl"):
        path = os.path.join(self.model_dir, name)
        joblib.dump(self.model, path)
        logger.info(f"Regime Model saved to {path}")

    def load(self, name="regime_model.pkl"):
        path = os.path.join(self.model_dir, name)
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_trained = True
            logger.info(f"Regime Model loaded from {path}")
            return True
        return False
