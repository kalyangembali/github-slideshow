import pandas as pd
import numpy as np
from loguru import logger
import warnings
from typing import Dict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from config import settings
from data.fetcher import YFinanceFetcher
from features.technical import calculate_technical_features
from features.microstructure import calculate_microstructure_features
from labels.triple_barrier import apply_triple_barrier
from models.xgboost_model import XGBoostModel

def setup_logger():
    """Setup structured logging via loguru."""
    logger.add("ai_trading_system_intraday.log", rotation="10 MB")
    logger.info("Starting AI Trading System INTRADAY Training Pipeline...")

def fetch_intraday_data() -> Dict[str, pd.DataFrame]:
    """Fetch 5m OHLCV data."""
    logger.info("=== PHASE 1: INTRADAY DATA FETCHING (5m) ===")
    fetcher = YFinanceFetcher(start_date=settings.TESTING_START_DATE)

    # We will test the loop on the first 5 trading stocks + Market Context
    test_symbols = settings.TRADING_STOCKS[:5]
    raw_data = fetcher.fetch_all(symbols=test_symbols, interval="5m", force_refresh=True)

    logger.info(f"Fetched {len(raw_data)} symbols.")
    return raw_data

def generate_intraday_features(raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Generate technical and microstructure features."""
    logger.info("=== PHASE 2: INTRADAY FEATURE ENGINEERING ===")
    features_dict = {}

    for symbol, df in raw_data.items():
        if len(df) < 50:
            logger.warning(f"Skipping {symbol}: Not enough data ({len(df)} rows).")
            continue

        logger.info(f"Generating features for {symbol}...")
        df = calculate_technical_features(df)
        df = calculate_microstructure_features(df)
        features_dict[symbol] = df

    return features_dict

def train_intraday_model(features_dict: Dict[str, pd.DataFrame]):
    """Train XGBoost using the Triple Barrier Labeler (Phase 5+)."""
    logger.info("=== PHASE 4: INTRADAY MODEL TRAINING (XGBoost) ===")

    combined_X = pd.DataFrame()
    combined_y = pd.Series(dtype='int')

    model = XGBoostModel()

    for symbol, df in features_dict.items():
        if "^" in symbol:
            continue

        logger.info(f"Applying Triple Barrier Labeling for {symbol}...")
        # 1. Apply Triple Barrier Labeler instead of default swing labeler
        df = apply_triple_barrier(df, tp_mult=2.0, sl_mult=1.0, t_barrier=10)

        # 2. Prepare Data specific to Triple Barrier
        df = df.dropna(subset=['target_triple_barrier'])

        # Define features explicitly (excluding non-features and target)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'target_triple_barrier', 'VWAP_approx']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]
        y = df['target_triple_barrier']

        if not X.empty:
            combined_X = pd.concat([combined_X, X])
            combined_y = pd.concat([combined_y, y])

    logger.info(f"Combined Intraday Training Data: {len(combined_X)} samples.")

    if len(combined_X) == 0:
        logger.error("No data available for training.")
        return None

    # Store feature names inside model for live prediction compatibility
    model.feature_cols = list(combined_X.columns)

    # Simple temporal split for dry-run
    split_idx = int(len(combined_X) * 0.8)
    X_train = combined_X.iloc[:split_idx]
    X_test = combined_X.iloc[split_idx:]
    y_train = combined_y.iloc[:split_idx]
    y_test = combined_y.iloc[split_idx:]

    model.train(X_train, y_train)
    accuracy = model.evaluate(X_test, y_test)

    logger.info(f"XGBoost Intraday Accuracy on Validation Set: {accuracy:.2f}")

    model.save_model("xgboost_baseline.pkl")
    return model

def main():
    setup_logger()
    raw_data = fetch_intraday_data()
    features_data = generate_intraday_features(raw_data)
    model = train_intraday_model(features_data)

    if model:
        logger.success("✅ Intraday XGBoost Model trained and saved successfully!")
    else:
        logger.error("❌ Pipeline failed.")

if __name__ == "__main__":
    main()
