import os
from pathlib import Path

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
LIMIT = 730
TRAINING_COLUMNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_columns.txt')
TRAIN_PCT = 0.8
MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
PATH = f'../newspapers/BTC_newspapers.csv'
SLEEP = 3
FILL = -99999999.0

# =============================================================================
# Phase 2 — News-ingestion agent: credibility gate, query budget, rate limits
# =============================================================================
# Primary credibility gate for the news agent. An article is accepted only if
# its SerpAPI `source.name` matches CREDIBLE_SOURCES (case-insensitive) OR its
# link domain matches CREDIBLE_DOMAINS. Seeded from outlet names already present
# in the existing newspapers/*.csv corpus (curated to reputable crypto/finance
# desks; promotional exchange blogs and price-aggregator pages were excluded).
CREDIBLE_SOURCES = [
    # Crypto-native news desks
    'CoinDesk', 'Cointelegraph', 'Decrypt', 'The Block', 'Brave New Coin',
    'CryptoSlate', 'crypto.news', 'Cryptonews', 'Cryptonews.net', 'BeInCrypto',
    'CoinGape', 'Coinspeaker', 'CryptoPotato', 'AMBCrypto', 'DailyCoin',
    'Cryptopolitan', 'U.Today', 'CCN.com', '99Bitcoins', 'Bitcoin.com News',
    'Bitcoin Magazine', 'DL News', 'dlnews.com', 'Blockworks', 'Coinpedia',
    'The Defiant', 'CoinCodex', 'BanklessTimes', 'Coinfomania', 'blockchain.news',
    # Mainstream finance / general press
    'Reuters', 'Bloomberg', 'Bloomberg.com', 'CNBC', 'Forbes', 'BBC',
    'Al Jazeera', 'Financial Times', 'The Wall Street Journal', 'Yahoo Finance',
    'Investopedia', 'The Economic Times', 'Investing.com', 'FXStreet',
    'Invezz', 'Capital.com', 'InvestorPlace', 'thestreet.com',
    'markets.businessinsider.com', 'Business Insider',
]
# Secondary gate: accept if the article's link domain matches any of these,
# even when source.name is missing/odd. Matched as a regex against the host.
CREDIBLE_DOMAINS = [
    r'coindesk\.com', r'cointelegraph\.com', r'decrypt\.co', r'theblock\.co',
    r'bravenewcoin\.com', r'cryptoslate\.com', r'crypto\.news', r'cryptonews\.com',
    r'beincrypto\.com', r'coingape\.com', r'coinspeaker\.com', r'cryptopotato\.com',
    r'ambcrypto\.com', r'dailycoin\.com', r'cryptopolitan\.com', r'u\.today',
    r'ccn\.com', r'99bitcoins\.com', r'news\.bitcoin\.com', r'bitcoinmagazine\.com',
    r'dlnews\.com', r'blockworks\.co', r'coinpedia\.org', r'thedefiant\.io',
    r'reuters\.com', r'bloomberg\.com', r'cnbc\.com', r'forbes\.com', r'bbc\.com',
    r'aljazeera\.com', r'ft\.com', r'wsj\.com', r'finance\.yahoo\.com',
    r'investopedia\.com', r'economictimes\.indiatimes\.com', r'investing\.com',
    r'fxstreet\.com', r'invezz\.com', r'capital\.com', r'investorplace\.com',
    r'thestreet\.com', r'businessinsider\.com',
]
# Per-coin, per-run article ingestion budget (after dedup + credibility filter).
MAX_ARTICLES_PER_COIN = 50
# Soft guard on total SerpAPI calls in a single agent run, to protect the key's
# daily quota. The agent stops issuing new searches once this is reached.
SERPAPI_DAILY_LIMIT = 250
TEST_DAYS = 7 # do not change
COINS = [
    'BTC', 'AVAX', 'ETH', 'LTC', 'SOL', 'ICP', 'DOGE', 'USDT',
    'XRP', 'ADA', 'DOT', 'LINK', 'BCH', 'ATOM', 'UNI', 'NEAR',
    'XLM', 'MATIC', 'FIL', 'APT',
]

BUY_THRESHOLD = 0.02        # 2% expected gain
STRONG_BUY_THRESHOLD = 0.05 # 5% for strong buy
SELL_THRESHOLD = -0.015     # -1.5% expected loss
STRONG_SELL_THRESHOLD = -0.03  # -3% for strong sell
HIGH_CONFIDENCE = 0.80      # 80% model agreement
MIN_CONFIDENCE = 0.50       # 50% minimum for action
REPO = 'cryptoTrading2'

# =============================================================================
# USER INPUT: Enter your investment amount here
# =============================================================================
LUMP_SUM = 100  # <-- CHANGE THIS to your investment amount in USD
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

NOTEBOOKS = 'notebooks'