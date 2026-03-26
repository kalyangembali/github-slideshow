import time
import pandas as pd
from loguru import logger
from datetime import datetime
import pytz

from config import settings
from data.fetcher import YFinanceFetcher
from features.technical import calculate_technical_features
from features.microstructure import calculate_microstructure_features
from models.xgboost_model import XGBoostModel

class LiveIntradayPredictor:
    """
    Phase 7: Live Engine
    Connects to live market data (yfinance 5m candles), runs the 55 feature pipeline,
    and queries the trained XGBoost model for immediate intraday signals.
    """
    def __init__(self, symbols=settings.TRADING_STOCKS):
        self.symbols = symbols
        self.fetcher = YFinanceFetcher()
        self.model = XGBoostModel()
        self.is_model_loaded = self.model.load_model("xgboost_baseline.pkl")

        if not self.is_model_loaded:
            logger.warning("No trained XGBoost model found. Predictions will be disabled.")

    def fetch_latest_candle(self, symbol) -> pd.DataFrame:
        """Fetches the latest 5-minute candle data."""
        # yfinance allows 5m data for the last 60 days
        try:
            df = self.fetcher.fetch_symbol(symbol, interval="5m", force_refresh=True)
            if df.empty:
                return pd.DataFrame()
            return df
        except Exception as e:
            logger.error(f"Error fetching live 5m data for {symbol}: {e}")
            return pd.DataFrame()

    def run_prediction_cycle(self):
        """Runs one complete cycle across all tracked symbols."""
        logger.info(f"--- Running Prediction Cycle at {datetime.now(pytz.timezone('Asia/Kolkata'))} IST ---")

        predictions = []

        for symbol in self.symbols:
            # 1. Fetch live 5m data
            df = self.fetch_latest_candle(symbol)
            if df.empty or len(df) < 20: # Need history for rolling windows (EMA/RSI)
                continue

            # 2. Build Features
            df = calculate_technical_features(df)
            df = calculate_microstructure_features(df)

            # The last row is the current unclosed or just-closed 5m candle
            latest_data = df.iloc[-1:]

            # Ensure we have the same columns the model was trained on
            if self.is_model_loaded and self.model.feature_cols:
                # Fill missing columns with 0 if our live stream lacks them
                for col in self.model.feature_cols:
                    if col not in latest_data.columns:
                        latest_data[col] = 0.0

                # Extract only the features the model knows
                X_live = latest_data[self.model.feature_cols]

                # 3. Predict
                preds = self.model.model.predict(X_live)
                probs = self.model.model.predict_proba(X_live)

                # 0 = SELL, 1 = HOLD, 2 = BUY
                signal = preds[0]
                confidence = probs[0][signal] * 100

                signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}

                predictions.append({
                    "Symbol": symbol,
                    "Time": latest_data.index[0],
                    "Price": latest_data['Close'].values[0],
                    "Signal": signal_map.get(signal, "UNKNOWN"),
                    "Confidence": f"{confidence:.1f}%",
                    "RSI": latest_data['RSI_14'].values[0]
                })

        # Print Dashboard to Terminal
        if predictions:
            pred_df = pd.DataFrame(predictions)
            print("\n--- LIVE INTRADAY DASHBOARD ---")
            print(pred_df.to_string(index=False))
            print("-------------------------------\n")

        return predictions

def start_live_loop(interval_seconds=300):
    """Starts the infinite loop for intraday trading (runs every 5 minutes)."""
    predictor = LiveIntradayPredictor()

    logger.info("Starting Live Intraday Loop...")
    try:
        while True:
            predictor.run_prediction_cycle()
            logger.info(f"Sleeping for {interval_seconds} seconds before next cycle...")
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("Live loop stopped manually.")

if __name__ == "__main__":
    # For testing, we run just one cycle instead of the infinite loop
    predictor = LiveIntradayPredictor()
    predictor.run_prediction_cycle()
