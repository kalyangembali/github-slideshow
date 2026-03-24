import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from loguru import logger
import joblib
import os

from config import settings

class XGBoostModel:
    def __init__(self, model_dir=os.path.join(settings.BASE_DIR, "models", "saved")):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        # We attempt to use CUDA/GPU if available, else CPU.
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            tree_method='hist', # 'gpu_hist' is deprecated in modern xgboost in favor of device='cuda'
            device='cuda',      # use device=cuda for GPU on RTX 3060
        )
        self.feature_cols = None

    def prepare_data(self, df: pd.DataFrame):
        """Prepare features and target labels."""
        # Using same ATR target logic as RF model
        k = settings.SWING_ATR_MULTIPLIER
        df['forward_return_5d'] = df['Close'].shift(-5) / df['Close'] - 1.0

        if 'ATR_14' not in df.columns:
            logger.error("ATR_14 not found in features. Cannot generate ATR-based labels.")
            return pd.DataFrame(), pd.Series()

        df['atr_pct'] = df['ATR_14'] / df['Close']
        df['threshold'] = k * df['atr_pct']

        conditions = [
            (df['forward_return_5d'] > df['threshold']),
            (df['forward_return_5d'] < -df['threshold'])
        ]

        # XGBoost requires target classes to start from 0 and be positive integers for multi-class classification
        # so instead of -1, 0, 1 -> 0(Sell), 1(Hold), 2(Buy)
        choices = [2, 0] # 2: BUY, 0: SELL

        df['target'] = np.select(conditions, choices, default=1) # 1: HOLD

        df = df.dropna()

        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'forward_return_5d', 'threshold', 'atr_pct', 'target']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[self.feature_cols]
        y = df['target']

        return X, y

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the XGBoost model."""
        logger.info(f"Training XGBoost on {len(X_train)} samples with {len(self.feature_cols)} features...")
        try:
             self.model.fit(X_train, y_train)
        except xgb.core.XGBoostError as e:
             # Fallback to CPU if CUDA is not available on current machine
             logger.warning(f"Failed to train on GPU, falling back to CPU: {e}")
             self.model.set_params(device='cpu')
             self.model.fit(X_train, y_train)

        logger.info("Training complete.")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate the model's performance."""
        logger.info("Evaluating XGBoost...")
        preds = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        logger.info(f"Accuracy: {accuracy:.4f}")

        report = classification_report(y_test, preds, zero_division=0)
        print("\nClassification Report:\n", report)
        return accuracy

    def save_model(self, name="xgboost_baseline.pkl"):
        path = os.path.join(self.model_dir, name)
        joblib.dump({'model': self.model, 'features': self.feature_cols}, path)
        logger.info(f"Model saved to {path}")
