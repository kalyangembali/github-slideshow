import pandas as pd
import numpy as np
from loguru import logger

def apply_triple_barrier(df: pd.DataFrame,
                         tp_mult=2.0,
                         sl_mult=1.0,
                         t_barrier=10,
                         price_col='Close',
                         high_col='High',
                         low_col='Low',
                         atr_col='ATR_14') -> pd.DataFrame:
    """
    Applies the Triple Barrier Method (Advances in Financial Machine Learning)
    to generate intraday labels based on the path of the price.

    Args:
        df: DataFrame containing price and ATR columns.
        tp_mult: Take Profit multiplier of ATR.
        sl_mult: Stop Loss multiplier of ATR.
        t_barrier: Vertical barrier (time expiration) in number of bars.
        price_col: Entry price column.
        high_col: High price column for TP check.
        low_col: Low price column for SL check.
        atr_col: Average True Range column.

    Returns:
        DataFrame with an added 'target' column:
        2: Upper barrier hit first (BUY)
        0: Lower barrier hit first (SELL)
        1: Vertical barrier hit first (HOLD/Time Expired)
    """
    df = df.copy()

    if atr_col not in df.columns:
        logger.error(f"Missing {atr_col} column. Cannot apply Triple Barrier.")
        return df

    targets = []

    # Iterate through the rows to calculate future path
    # (Note: For large datasets, this can be vectorized or optimized with numba)
    for i in range(len(df) - t_barrier):
        entry_price = df.iloc[i][price_col]
        atr = df.iloc[i][atr_col]

        # Calculate barrier levels
        upper_barrier = entry_price + (tp_mult * atr)
        lower_barrier = entry_price - (sl_mult * atr)

        # Look ahead exactly t_barrier steps
        window = df.iloc[i+1 : i+1+t_barrier]

        hit_upper = False
        hit_lower = False

        for j in range(len(window)):
            # Check high and low of each future candle
            if window.iloc[j][high_col] >= upper_barrier:
                hit_upper = True
                break # First touch wins
            if window.iloc[j][low_col] <= lower_barrier:
                hit_lower = True
                break # First touch wins

        # Assign label based on which barrier was hit first
        if hit_upper:
            targets.append(2) # BUY
        elif hit_lower:
            targets.append(0) # SELL
        else:
            targets.append(1) # HOLD (Vertical/Time barrier expired)

    # For the last 't_barrier' rows, we don't have enough future data to label securely.
    # We pad with NaN or a default 'HOLD' (1)
    targets.extend([np.nan] * t_barrier)

    df['target_triple_barrier'] = targets
    return df

if __name__ == "__main__":
    # Simple test to ensure it runs
    np.random.seed(42)
    test_df = pd.DataFrame({
        'Close': np.random.uniform(100, 110, 100),
        'High': np.random.uniform(105, 115, 100),
        'Low': np.random.uniform(95, 105, 100),
        'ATR_14': np.random.uniform(1, 3, 100)
    })

    labeled_df = apply_triple_barrier(test_df)
    print(labeled_df['target_triple_barrier'].value_counts(dropna=False))
