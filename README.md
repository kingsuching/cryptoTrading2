# CryptoTrading2

An AI-powered cryptocurrency trading system that combines multiple ML/DL forecasting models with sentiment analysis to generate buy/sell signals.

## Overview

1. A user inputs a lump-sum investment amount (`LUMP_SUM` in `files/CONSTANTS.py`).
2. Each model independently forecasts the next 7-day price for every tracked coin.
3. Predictions are weighted by each model's inverse RMSE percentile rank.
4. Sentiment analysis (DistilRoBERTa fine-tuned on financial news) is layered on top.
5. Combined signals drive BUY / SELL / HOLD decisions with configurable risk parameters.

---

## Project Structure

```
cryptoTrading2/
├── START_HERE.ipynb          # Entry point: fetch market data + run sentiment pipeline
├── data/                     # Raw OHLCV + indicator CSVs  ({COIN}_df.csv)
├── files/
│   ├── CONSTANTS.py          # All project-wide constants
│   ├── functions.py          # All helper & pipeline functions
│   ├── training_columns.txt  # 23 feature column names used for model training
│   ├── queries.txt           # News search queries for sentiment
│   └── API_KEYS.py           # API keys (not committed)
├── implementations/          # Model class definitions
│   ├── gbm_model.py          # GradientBoostingRegressor wrapper
│   ├── arima_model.py        # ARIMA statsmodels wrapper
│   ├── lstm_model.py         # PyTorch multi-layer LSTM
│   ├── svm_model.py          # SVR wrapper
│   ├── tft_model.py          # Temporal Fusion Transformer (PyTorch)
│   └── transformer_model.py  # Simple Transformer with attention (PyTorch)
├── notebooks/                # Training notebooks (one per model)
│   ├── GBM_training.ipynb
│   ├── ARIMA_training.ipynb
│   ├── KNN_training.ipynb
│   ├── LSTM_training.ipynb
│   ├── SVM_training.ipynb
│   ├── PROPHET_training.ipynb
│   ├── TFT_training.ipynb
│   └── Transformer_training.ipynb
├── models/                   # Saved model artifacts  (models/{COIN}/)
├── predictions/              # Future price CSVs + standardized RMSE  (predictions/{COIN}/)
├── metrics/                  # Per-model RMSE text files  (metrics/{COIN}/)
├── newspapers/               # Scraped news articles  ({COIN}_newspapers.csv)
└── plots/                    # Bollinger band & analysis plots
```

---

## Models

| Notebook | Model | Type | Library |
|---|---|---|---|
| `GBM_training.ipynb` | Gradient Boosting Machine | Tabular | scikit-learn |
| `ARIMA_training.ipynb` | ARIMA | Univariate time-series | statsmodels |
| `KNN_training.ipynb` | K-Nearest Neighbours | Tabular | scikit-learn |
| `LSTM_training.ipynb` | LSTM | Sequence (deep learning) | PyTorch |
| `SVM_training.ipynb` | Support Vector Machine | Tabular | scikit-learn |
| `PROPHET_training.ipynb` | Prophet | Time-series with regressors | Meta Prophet |
| `TFT_training.ipynb` | Temporal Fusion Transformer | Sequence (deep learning) | PyTorch |
| `Transformer_training.ipynb` | Transformer + Attention | Sequence (deep learning) | PyTorch |

---

## Features Used for Training

The 23 explanatory variables (defined in `EXPLANATORY_VARIABLES` / `training_columns.txt`):

| Category | Columns |
|---|---|
| Price | `open`, `high`, `low`, `volume` |
| Fear & Greed | `value` |
| Simple MA | `SMA_7`, `SMA_20`, `SMA_50` |
| Exponential MA | `EMA_12`, `EMA_26`, `EMA_20`, `EMA_50` |
| Momentum | `RSI`, `MACD`, `MACD_Signal`, `MACD_Hist` |
| Bollinger Bands | `BB_Upper`, `BB_Middle`, `BB_Lower`, `BB_STD` |
| Volume | `Volume_MA_7`, `OBV` |
| Sentiment | `avg_sentiment` |

**Response variable:** `close` (daily closing price)

---

## Quickstart

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn statsmodels prophet torch matplotlib tqdm \
            transformers beautifulsoup4 requests serpapi
```

### 2. Set API keys

Populate `files/API_KEYS.py`:

```python
CMC_KEY     = "your_coinmarketcap_key"
SERPAPI_KEY = "your_serpapi_key"
```

### 3. Fetch data & run sentiment pipeline

Open and run `START_HERE.ipynb`. This:
- Pulls OHLCV data from Coinbase Exchange
- Fetches Fear & Greed Index from CoinMarketCap
- Computes all technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, OBV)
- Scrapes news articles and scores sentiment with DistilRoBERTa
- Saves a complete `data/{COIN}_df.csv`

### 4. Train models

Run any notebook in `notebooks/`. Each notebook follows the same steps:

1. Load & prepare data
2. Train / validation split (80/20 chronological)
3. Define hyperparameter grid
4. Tune hyperparameters + normalization method
5. Train best model
6. Save model to `models/{COIN}/`
7. Predict on validation set
8. Forecast next 7 days
9. Save predictions to `predictions/{COIN}/` and RMSE to `metrics/{COIN}/`

---

## Configuration

All tunable constants live in `files/CONSTANTS.py`:

```python
COIN        = 'BTC'          # Active coin
LUMP_SUM    = 10000          # Investment amount (USD)
TEST_DAYS   = 7              # Forecast horizon (do not change)
TRAIN_PCT   = 0.8            # Train/val split ratio
COINS       = ['BTC', 'ETH', 'LTC', 'XRP', 'BCH', 'SOL']

# Signal thresholds
BUY_THRESHOLD         = 0.02   # 2% expected gain
STRONG_BUY_THRESHOLD  = 0.05   # 5% expected gain
SELL_THRESHOLD        = -0.015 # 1.5% expected loss
STRONG_SELL_THRESHOLD = -0.03  # 3% expected loss
HIGH_CONFIDENCE       = 0.80   # 80% model agreement required
```

---

## Output Format

Each model produces two files under `predictions/{COIN}/`:

| File | Description |
|---|---|
| `{model}_future_predictions.csv` | 7-row CSV with columns `date`, `predicted_price` |
| `{model}_standardized_rmse.txt` | Standardized RMSE (RMSE / std of actuals) |

Predictions are combined into a **7 × N matrix** (rows = dates, columns = models) for ensemble weighting.
