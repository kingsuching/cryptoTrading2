# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AI-powered cryptocurrency trading system. Multiple ML/DL models independently forecast 7-day prices for each coin, weighted by inverse RMSE for ensemble predictions, combined with news sentiment analysis to generate BUY/SELL/HOLD signals.

## Running the System

**All notebooks and scripts must be run from the project root** (`cryptoTrading2/`). The `base_dir()` utility in `functions.py` resolves paths by walking up from `os.getcwd()` to find the directory named `REPO = 'cryptoTrading2'`.

### Full pipeline (programmatic)
```python
from files.functions import coinbase_market_analysis_gradient, newspaper_sentiment_pipeline, train_all_models, predict_matrix
# 1. Fetch OHLCV + indicators + sentiment
coinbase_market_analysis_gradient('BTC', dataPath='data', plotPath='plots')
newspaper_sentiment_pipeline('BTC', newspaper_path='newspapers/BTC_newspapers.csv', queries_path='files/queries.txt')
# 2. Train all 8 models
train_all_models('BTC')
# 3. Generate prediction matrix
matrix = predict_matrix('BTC')  # shape: (8 models × 7 days)
```

### Using notebooks
Run `START_HERE.ipynb` for data fetch, then any notebook in `notebooks/` to train individual models.

**Notebook cell 1 pattern** (required boilerplate):
```python
_root = next((p for p in [Path(os.getcwd()), *Path(os.getcwd()).parents] if p.name == 'cryptoTrading2'), None)
if _root:
    os.chdir(str(_root))
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
```

## Architecture

### Data Flow
1. `coinbase_market_analysis_gradient(coin, ...)` → fetches OHLCV from Coinbase Exchange API, merges Fear & Greed Index (CoinMarketCap), computes all 23 technical indicators → saves `data/{COIN}_df.csv`
2. `newspaper_sentiment_pipeline(coin, ...)` → SerpAPI news scrape → DistilRoBERTa sentiment scoring → merges `avg_sentiment` into `data/{COIN}_df.csv`
3. `dataSetup(raw_df, ...)` → aggregates to daily, computes `avg_sentiment` from `score` column → returns indexed DataFrame ready for training
4. Training notebooks → `models/{COIN}/{COIN}_{model}_model.pkl` or `.pt`
5. `predict_matrix(coin)` → loads all saved artifacts → returns 8×7 prediction matrix

### Key Abstractions in `files/functions.py` (~2200 lines)

**Shared pipeline helpers** (used inside all `run_*_pipeline` functions):
- `_prepare_tabular_dataset(coin, response, cols_path)` — loads data, calls `dataSetup`, splits X/y
- `_time_series_train_val_split(X, y)` — chronological 80/20 split
- `_scaler_factory(method)` — returns `StandardScaler | MinMaxScaler | RobustScaler`
- `_tabular_future_forecast(model, scaler, X, daily_data, cols, response, n)` — iterative n-step forecast for tabular models
- `_seq_future_forecast_torch(model, scaler, data_full, cols, response, n, seq_len, device)` — iterative forecast for PyTorch sequence models
- `_torch_train(model, X_tr, y_tr, X_va, y_va, ...)` — training loop with early stopping
- `_save_model_artifact(obj, coin, filename)` — saves to `models/{COIN}/`
- `_save_metrics(std_rmse, coin, model_tag)` — saves to `metrics/{COIN}/` and `predictions/{COIN}/`
- `_standardized_rmse(y_true, y_pred)` → `(rmse, rmse / std(actuals))`

**Top-level pipeline functions** (one per model):
`run_gbm_pipeline`, `run_svm_pipeline`, `run_knn_pipeline`, `run_arima_pipeline`, `run_lstm_pipeline`, `run_tft_pipeline`, `run_transformer_pipeline`, `run_prophet_pipeline`

Each pipeline function: loads data → tunes hyperparameters + scaler → retrains on full train set → computes validation RMSE → saves artifact + metrics + 7-day predictions.

### Model Implementations (`implementations/`)
Thin wrappers over libraries with `.fit(X, y)` / `.predict(X)` interface:
- `GBMModel` — `sklearn.GradientBoostingRegressor`
- `SVMModel` (implied) — `sklearn.SVR`
- `ARIMAModel` — `statsmodels`
- `LSTMModel` — PyTorch multi-layer LSTM
- `TemporalFusionTransformer` / `CryptoTransformer` — PyTorch attention-based models

PyTorch models use `_get_device()` (auto-selects CUDA/MPS/CPU).

### Configuration (`files/CONSTANTS.py`)
Change `COIN` here to switch the active coin for all notebooks. `TEST_DAYS = 7` is fixed. `LUMP_SUM` is the investment amount for portfolio simulation.

### API Keys (`files/API_KEYS.py`, not committed)
Must define:
```python
CMC_KEY     = "..."   # CoinMarketCap Pro API
SERPAPI_KEY = "..."   # SerpAPI (Google News)
```

## Output Artifacts

| Path | Description |
|---|---|
| `data/{COIN}_df.csv` | Full feature dataset |
| `models/{COIN}/{COIN}_{model}_model.pkl` | Scikit-learn artifacts |
| `models/{COIN}/{COIN}_{model}_model.pt` | PyTorch artifacts |
| `predictions/{COIN}/{model}_future_predictions.csv` | 7-row CSV: `date`, `predicted_price` |
| `metrics/{COIN}/{model}_rmse.txt` | Standardized RMSE value |

## Install Dependencies

```bash
pip install pandas numpy scikit-learn statsmodels prophet torch matplotlib tqdm \
            transformers beautifulsoup4 requests serpapi
```