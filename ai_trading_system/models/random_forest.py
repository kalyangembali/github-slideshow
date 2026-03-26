import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from loguru import logger
import joblib
import os

from config import settings

class RandomForestModel:
    def __init__(self, model_dir=os.path.join(settings.BASE_DIR, "models", "saved")):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
        self.feature_cols = None

    def prepare_data(self, df: pd.DataFrame):
        """Prepare features and target labels."""
        # --- Create Target Label (Swing Label Strategy) ---
        # Volatility-Adjusted Forward Return (ATR-based)
        k = settings.SWING_ATR_MULTIPLIER
        df['forward_return_5d'] = df['Close'].shift(-5) / df['Close'] - 1.0

        # We need ATR_14 to exist (from features/technical.py)
        if 'ATR_14' not in df.columns:
            logger.error("ATR_14 not found in features. Cannot generate ATR-based labels.")
            return pd.DataFrame(), pd.Series()

        # ATR threshold percentage
        df['atr_pct'] = df['ATR_14'] / df['Close']
        df['threshold'] = k * df['atr_pct']

        conditions = [
            (df['forward_return_5d'] > df['threshold']),
            (df['forward_return_5d'] < -df['threshold'])
        ]
        choices = [1, -1] # 1: BUY, -1: SELL, 0: HOLD

        df['target'] = np.select(conditions, choices, default=0)

        # Drop NaNs created by shift(-5) for future prediction
        df = df.dropna()

        # Isolate features vs target
        # exclude non-feature columns
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'forward_return_5d', 'threshold', 'atr_pct', 'target']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[self.feature_cols]
        y = df['target']

        return X, y

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the Random Forest model."""
        logger.info(f"Training Random Forest on {len(X_train)} samples with {len(self.feature_cols)} features...")
        self.model.fit(X_train, y_train)
        logger.info("Training complete.")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate the model's performance."""
        logger.info("Evaluating Random Forest...")
        preds = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        logger.info(f"Accuracy: {accuracy:.4f}")

        report = classification_report(y_test, preds, zero_division=0)
        print("\nClassification Report:\n", report)
        return accuracy

    def predict(self, df: pd.DataFrame):
        """Predict actions for unseen data."""
        X = df[self.feature_cols]
        probs = self.model.predict_proba(X)
        preds = self.model.predict(X)
        return preds, probs

    def save_model(self, name="rf_baseline.pkl"):
        path = os.path.join(self.model_dir, name)
        joblib.dump({'model': self.model, 'features': self.feature_cols}, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, name="rf_baseline.pkl"):
        path = os.path.join(self.model_dir, name)
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data['model']
            self.feature_cols = data['features']
            logger.info(f"Model loaded from {path}")
            return True
        else:
            logger.error(f"Model not found at {path}")
            return False

if __name__ == "__main__":
    from data.fetcher import YFinanceFetcher
    from features.technical import calculate_technical_features

    # 1. Fetch
    fetcher = YFinanceFetcher()
    df = fetcher.fetch_symbol("RELIANCE.NS", force_refresh=False)

    # 2. Features
    df_feat = calculate_technical_features(df)

    # 3. Model Pipeline
    rf = RandomForestModel()
    X, y = rf.prepare_data(df_feat)

    if not X.empty:
        # Time-series split (walk-forward simulation for testing)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        rf.train(X_train, y_train)
        rf.evaluate(X_test, y_test)
        rf.save_model()
