import os

COIN = 'BTC'
RESPONSE = 'close'
RESPONSE_VARIABLE = 'close'   # response variable (price)
EXPLANATORY_VARIABLES = [
    'open', 'high', 'low', 'volume',
    'value',
    'SMA_7', 'SMA_20', 'SMA_50',
    'EMA_12', 'EMA_26', 'EMA_20', 'EMA_50',
    'RSI',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_STD',
    'Volume_MA_7', 'OBV',
    'avg_sentiment',
]
EMPTY_STRING = '-'
LIMIT = 365
TRAINING_COLUMNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_columns.txt')
TRAIN_PCT = 0.8
MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
PATH = f'../newspapers/BTC_newspapers.csv'
SLEEP = 3
FILL = -99999999.0
TEST_DAYS = 7 # do not change
COINS = ['BTC', 'ETH', 'LTC', 'XRP', 'BCH', 'SOL']

BUY_THRESHOLD = 0.02        # 2% expected gain
STRONG_BUY_THRESHOLD = 0.05 # 5% for strong buy
SELL_THRESHOLD = -0.015     # -1.5% expected loss
STRONG_SELL_THRESHOLD = -0.03  # -3% for strong sell
HIGH_CONFIDENCE = 0.80      # 80% model agreement
MIN_CONFIDENCE = 0.50       # 50% minimum for action

# =============================================================================
# USER INPUT: Enter your investment amount here
# =============================================================================
LUMP_SUM = 10000  # <-- CHANGE THIS to your investment amount in USD
# =============================================================================

# Risk parameters
MAX_POSITION_PCT = 0.10   # Max 10% per trade
STOP_LOSS_PCT = 0.05      # 5% stop loss
TAKE_PROFIT_PCT = 0.10    # 10% take profit

SIGNAL_MULTIPLIERS = {
    "STRONG BUY": 1.0,
    "BUY": 0.7,
    "HOLD": 0.0,
    "SELL": 0.7,
    "STRONG SELL": 1.0
}

MODEL_FILES = {
        'knn': ('knn_future_predictions.csv', 'predicted_price'),
        'rf': ('rf_future_predictions.csv', 'predicted_price'),
        'prophet': ('prophet_future_predictions.csv', 'predicted_price'),
        'tft': ('tft_future_predictions.csv', 'predicted_price'),
        'xgboost': ('xgb_future_predictions.csv', 'predicted_price'),
        'lstm': ('lstm_future_predictions.csv', 'predicted_price'),
        'lightgbm': ('lightgbm_future_predictions.csv', 'predicted_price'),
        'elasticnet': ('elasticnet_future_predictions.csv', 'predicted_price'),
        'svm': ('svm_future_predictions.csv', 'predicted_price'),
        'arima': ('arima_future_predictions.csv', 'predicted_price'),
        'transformer': ('transformer_future_predictions.csv', 'predicted_price'),
        'gbm': ('gbm_future_predictions.csv', 'predicted_price'),
    }