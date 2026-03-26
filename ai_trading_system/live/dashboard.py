import streamlit as st
import pandas as pd
import time
from datetime import datetime
import pytz
import sys
import os

# Automatically add the ai_trading_system directory to the path so Streamlit can find imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.predictor import LiveIntradayPredictor

st.set_page_config(
    page_title="AI Trading System - Live Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("🤖 AI Trading System - Live Intraday Dashboard")
st.markdown("""
This dashboard fetches the latest 5-minute candle data, generates all 55 technical and microstructure features,
and queries the XGBoost model to predict the next price movement based on the Triple Barrier labeler.
""")

# Initialize predictor
@st.cache_resource
def get_predictor():
    return LiveIntradayPredictor()

predictor = get_predictor()

if not predictor.is_model_loaded:
    st.error("⚠️ Trained XGBoost model (`xgboost_baseline.pkl`) not found in `models/saved/`. Predictions will be disabled. Run the training script first.")

# Create placeholders for live updating
status_placeholder = st.empty()
table_placeholder = st.empty()
last_update_placeholder = st.empty()

# Auto-refresh logic using a button (Streamlit doesn't natively while-loop well without custom components)
if st.button("Refresh Live Data Now"):
    with st.spinner("Fetching latest 5m candles and running AI predictions..."):
        try:
            predictions = predictor.run_prediction_cycle()

            if predictions:
                df = pd.DataFrame(predictions)

                # Apply styling to signals
                def highlight_signal(val):
                    color = 'green' if val == 'BUY' else 'red' if val == 'SELL' else 'orange'
                    return f'color: {color}; font-weight: bold;'

                styled_df = df.style.map(highlight_signal, subset=['Signal'])

                table_placeholder.dataframe(styled_df, use_container_width=True, hide_index=True)

                ist_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %I:%M:%S %p")
                last_update_placeholder.success(f"✅ Last updated at: {ist_time} IST")
            else:
                table_placeholder.warning("No predictions available (Market might be closed or API limits reached).")

        except Exception as e:
            st.error(f"Error during prediction cycle: {str(e)}")
else:
    st.info("Click the 'Refresh Live Data Now' button to run the AI prediction cycle.")

st.markdown("---")
st.caption("Developed for RTX 3060 fully local execution | Phase 7 AI System")
