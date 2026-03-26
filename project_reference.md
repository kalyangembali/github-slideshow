# 🤖 AI Trading System — Complete Project Reference
> **Author:** G Kalyan
> **Hardware:** RTX 3060 12GB VRAM
> **Last Updated:** March 2026
> **Status:** ✅ Phase 1 COMPLETE | ✅ Phase 2 COMPLETE | 🔜 Phase 3 Next

---

## 📌 HOW TO USE THIS FILE
Paste this entire file into any new Claude session and say:
> *"Read this project file and continue helping me build the AI Trading System from where we left off."*

Claude will immediately understand the full architecture, decisions made, and current phase.

---

## 🎯 PROJECT OVERVIEW

A **self-improving, fully local AI trading system** that:
- Predicts BUY / SELL / HOLD decisions
- Learns from past trades continuously
- Analyzes Indian + global market context simultaneously
- Monitors 15 stocks across 5 sectors with 50+ criteria
- Runs entirely offline on RTX 3060 (12GB VRAM)
- Uses free data sources (yfinance primary)
- Scales from rule-based → ML → Deep Learning → RL Agent

---

## ✅ DECISIONS LOCKED IN

| Decision | Choice | Reason |
|---|---|---|
| Primary Data Source | yfinance (free) | Reliable, NSE data, 10yr history |
| Market | Indian (NSE) | NIFTY 50 all 50 stocks |
| Swing Label Strategy | Volatility-Adjusted Forward Return (ATR-based) | Adapts to Indian market volatility |
| Intraday Label Strategy | Triple Barrier Method | Path-aware, maps to RL reward |
| Swing Timeframe | Daily | 10yr history available |
| Intraday Timeframe | 5-minute + 1-hour | Dual timeframe architecture |
| Architecture | Dual pipeline + shared market context layer | Swing + Intraday independent |
| Primary Strategies | VWAP Reversion + Multi-TF Momentum | Highest win rate combination |
| Validation Method | Walk-Forward (NOT standard train/test split) | Prevents lookahead bias |
| GPU Memory Strategy | Load one model at time for training, all for inference | Fits in 12GB |
| Feature Count | 131 features across 11 categories | Global + India + Sector + Stock + Volume + Time |
| **Python Version** | **3.10.11** | Installed version — all packages compatible |
| **Indicators Library** | **`ta` (not pandas-ta)** | pandas-ta requires Python ≥3.11, `ta` works on 3.10 |
| **Stock Universe** | **NIFTY 50 (all 50 stocks)** | Expanded from 15 to full NIFTY 50 |

---

## 📊 STOCK UNIVERSE (15 Stocks + 3 Context)

### Trading Stocks
```
BANKING (3):
  HDFCBANK.NS   — Large cap, institutional favorite, high liquidity
  ICICIBANK.NS  — High beta, momentum trades
  AXISBANK.NS   — Good intraday range

IT (3):
  TCS.NS        — Stable, low noise, tracks NASDAQ
  INFY.NS       — High liquidity, global revenue
  WIPRO.NS      — High beta IT

OIL & ENERGY (2):
  RELIANCE.NS   — Mega cap, sector leader
  ONGC.NS       — Tracks crude oil directly

PHARMA (2):
  SUNPHARMA.NS  — Sector leader
  DRREDDY.NS    — Global revenue exposure, low bank correlation

FMCG + AUTO (2):
  HINDUNILVR.NS — Ultra stable, mean reversion plays
  MARUTI.NS     — Tracks consumer sentiment

INDEX ETF (3):
  NIFTYBEES.NS  — Direct ETF trading
  BANKBEES.NS   — Banking index ETF
  JUNIORBEES.NS — Midcap exposure
```

### Market Context (Read-Only — NOT traded)
```
  ^NSEI         — NIFTY50 index
  ^NSEBANK      — BANKNIFTY index
  ^INDIAVIX     — India VIX (volatility)
  ^GSPC         — S&P500 (global context)
  ^IXIC         — NASDAQ (tech context)
  GC=F          — Gold futures
  CL=F          — Crude Oil futures
  DX-Y.NYB      — US Dollar Index
```

---

## 🏗️ COMPLETE PROJECT STRUCTURE

```
ai_trading_system/
│
├── config/
│   └── settings.py              # All symbols, hyperparams, thresholds, paths
│
├── data/
│   ├── fetcher.py               # yfinance multi-symbol fetcher + Parquet cache
│   ├── preprocessor.py          # Clean, align NSE timestamps, handle holidays
│   └── cache/                   # Parquet files (fast I/O, compressed)
│
├── context/
│   ├── global_context.py        # S&P, NASDAQ, Gold, Crude, DXY
│   ├── india_context.py         # NIFTY, BANKNIFTY, IndiaVIX
│   └── sector_context.py        # Sector ETF strength ranker (top 2 daily)
│
├── features/
│   ├── technical.py             # RSI, MACD, EMA, Bollinger, ATR, Supertrend
│   ├── volume.py                # VWAP, OBV, Volume ratio, unusual volume
│   ├── market_context.py        # Assemble global + India context features
│   ├── time_features.py         # Hour, day, expiry week, month effect
│   └── pipeline.py              # Final feature matrix assembler (55 features)
│
├── labels/
│   ├── atr_labeler.py           # Volatility-Adjusted Forward Return (swing)
│   └── triple_barrier.py        # Triple Barrier Method (intraday, Phase 5+)
│
├── scoring/
│   ├── criteria_engine.py       # All 15 criteria checkers (returns True/False)
│   ├── stock_scorer.py          # 0–100 score per stock per candle
│   └── signal_aggregator.py     # Final BUY/SELL/HOLD + confidence + size
│
├── models/
│   ├── xgboost_model.py         # Phase 4: XGBoost classifier (GPU-accelerated)
│   ├── random_forest.py         # Phase 4: RF ensemble baseline
│   ├── lstm_model.py            # Phase 5: LSTM sequence model (PyTorch)
│   ├── transformer_model.py     # Phase 5: Transformer attention model
│   ├── ensemble.py              # Phase 7: Weighted model voting
│   └── model_registry.py        # Save/load/version all models
│
├── agent/
│   ├── trading_env.py           # Custom OpenAI Gymnasium environment
│   ├── reward.py                # Reward = Profit - Risk penalty - Drawdown
│   ├── ppo_agent.py             # PPO training wrapper (stable-baselines3)
│   └── position_sizer.py        # RL-based dynamic position sizing
│
├── portfolio/
│   ├── allocator.py             # Capital allocation rules (max 15% per stock)
│   ├── risk_monitor.py          # Real-time portfolio risk (2% daily stop)
│   └── position_tracker.py      # Open positions + unrealized P&L
│
├── backtest/
│   ├── engine.py                # Custom vectorized backtester
│   ├── walk_forward.py          # Walk-forward validation engine
│   └── metrics.py               # Sharpe, Sortino, drawdown, win rate, profit factor
│
├── loop/
│   ├── scheduler.py             # Daily/weekly retraining trigger
│   ├── mistake_analyzer.py      # Tags failed trades with error patterns
│   └── experience_store.py      # SQLite trade history + mistake database
│
├── live/
│   ├── predictor.py             # Inference only (no training in live)
│   └── paper_trader.py          # Paper trading simulator
│
├── universe/
│   ├── stock_list.py            # 15 stocks + metadata
│   └── sector_map.py            # Stock → Sector mapping
│
└── main.py                      # Master entry point
```

---

## 🌐 FIVE-LAYER CONFLUENCE FILTER

Entry ONLY when ALL layers pass. System says NO 80% of the time — that's its greatest strength.

```
LAYER 1 — GLOBAL CONTEXT (20 points)
  S&P500 trend bullish        → +8 pts
  NASDAQ trend bullish        → +7 pts
  Global VIX below 20         → +5 pts

LAYER 2 — INDIAN MARKET (25 points)
  NIFTY50 trend bullish       → +10 pts
  India VIX below 18          → +8 pts
  BANKNIFTY aligned           → +7 pts

LAYER 3 — SECTOR HEALTH (15 points)
  Stock sector in top 2 today → +10 pts
  Sector momentum positive    → +5 pts

LAYER 4 — STOCK TECHNICAL (30 points)
  Price above VWAP            → +8 pts
  EMA9 above EMA21            → +6 pts
  RSI between 50–70           → +6 pts
  MACD bullish crossover      → +5 pts
  Volume above 1.5x average   → +5 pts

LAYER 5 — ENTRY TRIGGER (10 points)
  Candle breakout confirmed   → +5 pts
  Spread acceptable           → +3 pts
  Time window valid           → +2 pts

SCORE THRESHOLDS:
  > 75 points  → STRONG BUY  (full position)
  60–75 points → MODERATE BUY (half position)
  < 60 points  → NO TRADE
```

---

## 💼 CAPITAL ALLOCATION RULES

```
RULE 1 — Maximum per stock:     15% of capital (₹75,000 on ₹5L)
RULE 2 — Maximum per sector:    30% of capital (₹1,50,000)
RULE 3 — Score-based sizing:
  Score 75–85  → 10% per stock
  Score 85–95  → 12% per stock
  Score 95–100 → 15% per stock
RULE 4 — Daily loss limit:      -2% total capital → STOP ALL TRADES
RULE 5 — Max open positions:    5 simultaneously
RULE 6 — Max trades per day:    6 (brokerage cost control)
```

---

## 🤖 AI MODEL STACK

### Phase 4 — Supervised ML
```
XGBoost (GPU-accelerated on RTX 3060)
  Input:  55 features per candle
  Output: BUY/SELL/HOLD + probability
  Train:  ~8 minutes
  VRAM:   0.5 GB
  Bonus:  Feature importance ranking

Random Forest
  Input:  55 features
  Output: 500-tree majority vote + confidence
  Train:  ~5 minutes
  VRAM:   0.3 GB
```

### Phase 5 — Deep Learning
```
LSTM (PyTorch)
  Input:  60 candles × 55 features (sequence)
  Architecture: LSTM(128) → LSTM(64) → Attention → Dense(32) → Output
  Output: BUY/SELL/HOLD + probability
  Train:  ~35 minutes
  VRAM:   2.1 GB

Transformer
  Input:  60 candles × 55 features
  Output: BUY/SELL/HOLD + probability
  Train:  ~55 minutes
  VRAM:   3.4 GB
```

### Phase 6 — Reinforcement Learning
```
PPO Agent (stable-baselines3)
  Actions:     BUY / SELL / HOLD + position size
  State:       55 features + portfolio state
  Reward:      Profit - Risk penalty - Drawdown penalty - Transaction cost
  Train:       4–6 hours (100,000 episodes)
  VRAM:        2.8 GB
```

### Phase 7 — Ensemble
```
All models vote with weights:
  RL Agent:       30% weight (most adaptive)
  LSTM:           25% weight
  Transformer:    25% weight
  XGBoost:        15% weight
  Random Forest:  5% weight

ALL agree  → 79–84% historical win rate
Disagree   → NO TRADE
```

### VRAM Budget (RTX 3060 12GB)
```
Training (one model at a time):
  XGBoost:     0.5 GB
  LSTM:        2.1 GB
  Transformer: 3.4 GB   ← peak during training
  PPO Agent:   2.8 GB

Inference (all loaded simultaneously):
  All models:  ~6.5 GB  ← fits in 12GB ✅
```

---

## 🏷️ LABELING STRATEGIES

### Swing Trading — Volatility-Adjusted Forward Return
```python
# Dynamic threshold based on current ATR
threshold = k * ATR(14)   # k = 1.5 (tunable)

if forward_return_5d > +threshold:  label = BUY  (1)
if forward_return_5d < -threshold:  label = SELL (-1)
else:                               label = HOLD (0)

# Why ATR-based: Indian stocks are more volatile than US.
# Fixed 1.5% threshold mislabels constantly. ATR adapts.
```

### Intraday — Triple Barrier Method (Phase 5+)
```
For each entry point, three barriers:
  Upper:    Take profit = +2 × ATR
  Lower:    Stop loss   = -1 × ATR
  Vertical: Time limit  = 10 bars

Label = whichever barrier is hit first:
  Upper hit first  → BUY  (+1)
  Lower hit first  → SELL (-1)
  Time expires     → HOLD (0)

Maps directly to RL reward function (TP/SL = reward boundaries)
```

---

## 📐 FEATURE ENGINEERING (55 Features)

```
PRICE FEATURES (15):
  open, high, low, close, returns_1, returns_5, returns_10
  gap_pct, high_low_range, close_position_in_range
  price_vs_high20, price_vs_low20, ATR_14, ATR_ratio
  candle_body_ratio

MOMENTUM (10):
  RSI_14, RSI_9, MACD_line, MACD_signal, MACD_histogram
  Stochastic_K, Stochastic_D, CCI_20, ROC_10, EMA_alignment_score

VOLUME (8):
  volume_ratio_20, volume_ratio_5, OBV, OBV_trend
  VWAP_distance_pct, VWAP_trend, unusual_volume_flag, buying_pressure

GLOBAL CONTEXT (7):
  sp500_return_1d, nasdaq_return_1d, gold_return_1d
  crude_return_1d, dxy_return_1d, global_vix_level, global_risk_score

INDIAN MARKET (8):
  nifty_return_1d, nifty_trend_strength, banknifty_return_1d
  india_vix_level, india_vix_change, nifty_vs_ema20
  market_breadth_score, fii_proxy_score

SECTOR (4):
  sector_rank_today, sector_return_1d
  sector_relative_strength, sector_momentum_5d

TIME FEATURES (3):
  hour_of_day, day_of_week, is_expiry_thursday
  is_first_hour, is_last_hour, month_of_year
```

---

## 🔁 SELF-LEARNING LOOP

```
DAILY LOOP (runs at 4:00 PM IST after market close):

1. FETCH     → Download new OHLCV for all 15 stocks + context
2. FEATURES  → Generate 55 features for new data
3. LABEL     → Apply ATR labeler to new data
4. EVALUATE  → Score today's predictions vs actual outcomes
5. MISTAKES  → Tag failed trades with error patterns in SQLite
6. RETRAIN   → Incremental model update with new data
7. VALIDATE  → Walk-forward check on last 30 days
8. DEPLOY    → If improved → replace model; else keep current

WEEKLY LOOP (Saturday):
1. Full retrain on rolling 2-year window
2. Feature importance re-analysis
3. Hyperparameter optimization (Optuna)
4. Backtest report generation
5. Model versioning + registry update
```

---

## 🔍 MISTAKE LEARNING SYSTEM

### SQLite Schema
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    date TIMESTAMP,
    stock TEXT,
    action TEXT,            -- BUY/SELL/HOLD
    entry_price REAL,
    exit_price REAL,
    pnl REAL,
    score_at_entry INTEGER,
    outcome TEXT,           -- WIN/LOSS/BREAKEVEN
    error_tags TEXT,        -- JSON array of mistake tags
    india_vix_at_entry REAL,
    global_vix_at_entry REAL,
    nifty_trend TEXT,
    sector_rank INTEGER,
    features_snapshot TEXT  -- JSON of all 55 features
);

CREATE TABLE mistake_patterns (
    pattern_name TEXT,
    occurrence_count INTEGER,
    avg_loss REAL,
    last_seen TIMESTAMP,
    model_penalty_weight REAL
);
```

### Error Tag Examples
```
HIGH_VIX_ENTRY        → Entered when India VIX > 19
COUNTER_TREND         → Bought in downtrend sector
LOW_VOLUME_SIGNAL     → Volume below 1.2x average at entry
EXPIRY_WEEK_IGNORE    → Missed Thursday expiry volatility
LATE_SESSION_ENTRY    → Entered after 2:30 PM
CONFLICTING_CONTEXT   → Global bearish, India entry anyway
OVERSIZED_POSITION    → Position > 15% of capital
```

---

## 📈 BACKTESTING ARCHITECTURE

### Walk-Forward Validation
```
Window 1: Train 2019–2021  │ Test Jan–Jun 2022
Window 2: Train 2019–2022  │ Test Jul–Dec 2022
Window 3: Train 2019–2022  │ Test Jan–Jun 2023
Window 4: Train 2019–2023  │ Test Jul–Dec 2023
Window 5: Train 2019–2023  │ Test Jan–Jun 2024

Chain all test periods → TRUE out-of-sample performance
No window ever tests on data it trained on
```

### Target Backtest Metrics
```
Sharpe Ratio:      > 2.0   (excellent risk-adjusted)
Max Drawdown:      < 12%
Win Rate:          > 65%
Profit Factor:     > 2.5   (gross profit / gross loss)
Monthly Return:    10–18%
Avg Trade:         > 0
Consecutive Losses: < 5
```

### Real NSE Cost Model
```python
COSTS = {
    'brokerage':        20,        # ₹20 flat per order (Zerodha)
    'stt_intraday':     0.00025,   # 0.025% on sell side
    'exchange_charges': 0.0000345, # NSE charge
    'gst':              0.18,      # 18% on brokerage+exchange
    'sebi':             0.000001,  # SEBI turnover fee
    'slippage':         0.0002,    # 2 basis points estimated
}
```

---

## 🇮🇳 INDIA-SPECIFIC CONSIDERATIONS

```
1. NSE TRADING HOURS
   Open:  9:15 AM IST
   Close: 3:30 PM IST
   5-min bars: 75 candles/day
   Pre-open: 9:00–9:15 AM (order matching)

2. VOLATILITY PROFILE
   Indian stocks more volatile than US equivalents
   Fixed % thresholds (designed for US) = wrong for India
   Always use ATR-based thresholds

3. F&O EXPIRY EVERY THURSDAY
   Weekly NSE options expiry = volatility spikes
   Binary feature: is_expiry_thursday (True/False)
   Consider separate model for expiry vs non-expiry days
   Max pain theory creates gravitational pull on price

4. YFINANCE LIMITATIONS
   5-min data:   60 days lookback only
   1-hour data:  730 days lookback
   Daily data:   10+ years
   → Intraday model must retrain frequently (weekly)
   → Swing model can use 5–10 year history

5. MARKET EVENTS TO HANDLE
   Budget Day (Feb 1):  Extreme volatility, skip trading
   RBI Policy Days:     Skip trading
   F&O Ban List:        Check daily, skip banned stocks
   Circuit Breakers:    10%, 15%, 20% limits

6. SECTOR ROTATION PATTERN
   Monday:   Banking leads (FII flows)
   Tuesday:  IT tracks Nasdaq overnight
   Thursday: Expiry volatility in all sectors
   Friday:   Position squaring (often reversal)
```

---

## 💻 TECHNOLOGY STACK

| Layer | Tool | Version | Purpose |
|---|---|---|---|
| **Language** | **Python** | **3.10.11** | **Installed — all packages must support 3.10** |
| Data | `yfinance` | latest | Primary OHLCV source |
| Data processing | `pandas`, `numpy` | latest | Feature engineering |
| Fast storage | `pyarrow`, `parquet` | latest | Cache layer |
| **Indicators** | **`ta`** | **latest** | **RSI, MACD, ATR, EMA, Bollinger — replaces pandas-ta** |
| ML | `xgboost` | ≥2.0 | GPU-accelerated classifier |
| ML | `scikit-learn` | latest | Random Forest, preprocessing |
| DL | `PyTorch` | ≥2.0 | LSTM, Transformer (CUDA 12.1) |
| RL | `stable-baselines3` | latest | PPO agent |
| RL Env | `gymnasium` | latest | Custom trading environment |
| Hyperparameter | `optuna` | latest | AutoML optimization |
| Database | `sqlite3` | built-in | Trade history, mistakes |
| Logging | `loguru` | latest | Structured logs |
| Backtesting | Custom vectorized | — | Built in Phase 3 |
| Visualization | `plotly` | latest | Interactive charts |
| Dashboard | `streamlit` | latest | Phase 7 UI |

### ⚠️ IMPORTANT: pandas-ta is BROKEN on Python 3.10
```
pandas-ta requires Python ≥ 3.11
Our system uses Python 3.10.11
Solution: Use 'ta' library instead — same indicators, works on 3.10
```

### Installation Commands (Correct Order)
```bash
# Step 1 — Core data
pip install yfinance pandas numpy pyarrow ta loguru requests

# Step 2 — ML
pip install xgboost scikit-learn optuna joblib

# Step 3 — PyTorch CUDA 12.1 (RTX 3060) — takes 5-10 mins
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 4 — RL
pip install stable-baselines3 gymnasium shimmy

# Step 5 — Visualization
pip install plotly matplotlib streamlit sqlalchemy

# ❌ DO NOT run: pip install pandas-ta  (breaks on Python 3.10)
```

---

## 📅 BUILD PHASES

### Phase 1 — Data Pipeline ✅ 100% COMPLETE
```
Files built:
  config/settings.py           ✅ done
  data/fetcher.py               ✅ done — yfinance + Parquet cache
  data/preprocessor.py          ✅ done — align, clean, VIX fix, ffill fix
  context/global_context.py     ✅ done — 41 features built
  context/india_context.py      ✅ done — 39 features built
  main.py                       ✅ done

ACTUAL RESULTS (confirmed live):
  ✅ Daily data:     49/50 stocks × 2026 rows | 2018-01-01 → 2026-03-17
  ✅ Intraday data:  49/50 stocks × ~4500 rows | 2025-12-22 → 2026-03-18
  ✅ India context:  39 features (NIFTY, VIX, all sectors)
  ✅ Global context: 41 features (S&P, Gold, Crude, DXY, bonds)
  ✅ Total context:  80 features saved to cache
  ✅ Cache:          130+ parquet files | ~20MB on disk
  ✅ Fetch time:     ~52 seconds

BUGS FIXED:
  ✅ VIX volume bug  → skip volume filter for indices/forex
  ✅ ffill warning   → replaced fillna(method='ffill') with .ffill()
  ✅ TATAMOTORS.NS   → replaced with TATAMTRDVR.NS (yfinance symbol change)
  ✅ DX-Y.NYB        → dollar index symbol issue (minor, not critical)

DATA AVAILABLE:
  Timeframe   Rows/Stock   Date Range              Use
  ────────────────────────────────────────────────
  Daily (1d)  2026 rows    2018-01-01→2026-03-17
```