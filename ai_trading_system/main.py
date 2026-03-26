import os
import pandas as pd
from loguru import logger
import warnings
from typing import Dict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from config import settings
from data.fetcher import YFinanceFetcher
from features.technical import calculate_technical_features
from models.random_forest import RandomForestModel

def setup_logger():
    """Setup structured logging via loguru."""
    logger.add("ai_trading_system.log", rotation="10 MB")
    logger.info("Starting AI Trading System Pipeline...")

def fetch_data() -> Dict[str, pd.DataFrame]:
    """Phase 1: Fetch raw OHLCV data."""
    logger.info("=== PHASE 1: DATA FETCHING ===")
    fetcher = YFinanceFetcher(start_date=settings.TESTING_START_DATE)

    test_symbols = settings.TRADING_STOCKS + ["^NSEI", "^INDIAVIX"]
    raw_data = fetcher.fetch_all(symbols=test_symbols, interval="1d", force_refresh=False)

    logger.info(f"Fetched {len(raw_data)} symbols.")
    return raw_data

def generate_features(raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Phase 2: Generate technical indicators."""
    logger.info("=== PHASE 2: FEATURE ENGINEERING ===")
    features_dict = {}

    for symbol, df in raw_data.items():
        if len(df) < 50:  # Need enough rows for rolling windows
            logger.warning(f"Skipping {symbol}: Not enough data ({len(df)} rows).")
            continue

        logger.info(f"Generating features for {symbol}...")
        feat_df = calculate_technical_features(df)
        features_dict[symbol] = feat_df
        logger.info(f"Generated {len(feat_df.columns)} features for {symbol}.")

    return features_dict

def train_baseline_model(features_dict: Dict[str, pd.DataFrame]):
    """Phase 4: Train Baseline Model (Random Forest)."""
    logger.info("=== PHASE 4: MODEL TRAINING ===")

    # Combine data from all trading stocks for training
    combined_X = pd.DataFrame()
    combined_y = pd.Series(dtype='int')

    rf = RandomForestModel()

    for symbol, df in features_dict.items():
        if "^" in symbol:  # Skip indices/context for stock-level prediction target creation
            continue

        logger.info(f"Preparing targets for {symbol}...")
        X, y = rf.prepare_data(df)

        if not X.empty:
            combined_X = pd.concat([combined_X, X])
            combined_y = pd.concat([combined_y, y])

    logger.info(f"Combined Training Data: {len(combined_X)} total samples.")

    if len(combined_X) == 0:
        logger.error("No data available for training.")
        return None

    # Split into train/test using simple time split for the combined dataset
    # (Note: proper walk-forward validation is meant for Phase 7 backtester, simple split here for MVP)
    split_idx = int(len(combined_X) * 0.8)
    X_train = combined_X.iloc[:split_idx]
    X_test = combined_X.iloc[split_idx:]
    y_train = combined_y.iloc[:split_idx]
    y_test = combined_y.iloc[split_idx:]

    # Train
    rf.train(X_train, y_train)

    # Evaluate
    accuracy = rf.evaluate(X_test, y_test)
    logger.info(f"Baseline Accuracy on Validation Set: {accuracy:.2f}")

    # Save
    rf.save_model()
    return rf

def main():
    setup_logger()

    # 1. Fetch
    raw_data = fetch_data()

    # 2. Features
    features_data = generate_features(raw_data)

    # 3. Model
    model = train_baseline_model(features_data)

    if model:
        logger.success("✅ AI Trading System pipeline completed successfully!")
    else:
        logger.error("❌ Pipeline failed during model training.")

if __name__ == "__main__":
    main()
