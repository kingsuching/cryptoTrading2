import os
import re
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.nn.functional import softmax
from tqdm import tqdm
from files import CONSTANTS
from files.API_KEYS import *
from files.CONSTANTS import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from pathlib import Path
from typing import Tuple, Dict
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def preprocess(str):
    # Remove punctuation, whitespace, and special characters, standardize case
    pattern = r'[^\w\s]|[\n\r\t]'
    preprocessed = re.sub(pattern, '', str).upper()
    return preprocessed


def prepare_coin_data(coin: str) -> pd.DataFrame:
    """
    Load and preprocess coin data by adding missing columns required for training.

    Args:
        coin: Coin symbol (e.g., 'BTC', 'ETH')

    Returns:
        Preprocessed DataFrame ready for dataSetup()
    """
    # Load data
    data_path = fullDataPath(coin)
    df = pd.read_csv(data_path)

    # Add SMA_7 if missing (calculate from close)
    if 'SMA_7' not in df.columns and 'close' in df.columns:
        df['SMA_7'] = df['close'].rolling(window=7, min_periods=1).mean()

    # Add Volume_MA_7 if missing (calculate from volume)
    if 'Volume_MA_7' not in df.columns and 'volume' in df.columns:
        df['Volume_MA_7'] = df['volume'].rolling(window=7, min_periods=1).mean()

    # Add value (fear/greed index) if missing - default to neutral (50)
    if 'value' not in df.columns:
        df['value'] = 50.0

    # Add avg_sentiment if missing - default to neutral (0)
    if 'avg_sentiment' not in df.columns:
        df['avg_sentiment'] = 0.0

    # Add score column if missing (needed for dataSetup aggregation)
    if 'score' not in df.columns:
        df['score'] = 0.0

    return df


def article_metadata(query):
    from serpapi import GoogleSearch
    params = {
        "api_key": SERPAPI_KEY,
        "engine": "google_news",
        "hl": "en",
        "gl": "us",
        "q": query
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    article_metadata = results['news_results']
    return article_metadata


def get_newspapers(query, existing_links):
    articles = article_metadata(query)
    article_df = pd.DataFrame(articles)
    if 'stories' in article_df.columns:
        header = pd.json_normalize(article_df['stories'][0])
        intersection = np.intersect1d(article_df.columns, header.columns)
        article_df = pd.concat([header[intersection], article_df[intersection]])
        article_df = article_df.dropna(subset=['date'])

    article_df = article_df[~article_df['link'].isin(existing_links)]
    texts = []
    for index, row in tqdm(article_df.iterrows(), total=len(article_df)):
        text = ""
        try:
            time.sleep(SLEEP)
            response = requests.get(row['link'], timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for paragraph in soup.find_all('p'):
                text += paragraph.get_text() + '\n'
            if text.strip() == '':
                text = row['title']
        except:
            text = row['title']

        texts.append(text)

    article_df['text'] = texts
    return article_df


def sentimentAnalysis(newspapers_df, NEGATIVE, NEUTRAL, POSITIVE):
    """STEP 2"""
    newspapers_df['things'] = newspapers_df['text'].astype(str).apply(get_sentiment)
    newspapers_df['sentiment'] = newspapers_df['things'].apply(lambda x: x[0])
    newspapers_df['score'] = newspapers_df['things'].apply(
        lambda x: sentiment_score(x[1], NEGATIVE, NEUTRAL, POSITIVE, MULTIPLIER=100))
    newspapers_df.drop('things', axis=1, inplace=True)
    return newspapers_df


def sentiment_score(x, NEGATIVE=-1, NEUTRAL=0, POSITIVE=1, MULTIPLIER=100):
    return MULTIPLIER * (NEGATIVE * x[0] + NEUTRAL * x[1] + POSITIVE * x[2])


def get_sentiment(text):
    """HELPER FUNCTION"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    tokens = tokenizer(preprocess(text), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)

    probabilities = softmax(outputs.logits, dim=-1)  # Activation function to get predicted class probabilities
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    label_mapping = model.config.id2label
    predicted_label = label_mapping[predicted_class]  # map it back to a readable label
    return predicted_label, probabilities.tolist()[0]  # return predicted label and associated probabilities


def newspapers_from_queries(coin, queries_path):
    """STEP 1"""
    queries = None
    with open(queries_path) as queries:
        query = queries.read().splitlines()
        query = [f'{coin} {q}' for q in query]
        queries = query

    newspapers = pd.DataFrame()
    links = []
    if os.path.exists(PATH):
        newspapers = pd.read_csv(PATH)
        links = newspapers['link'].tolist()
    for q in tqdm(queries):
        stuff = get_newspapers(q, links)
        newspapers = pd.concat([newspapers, stuff], ignore_index=True)

    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    newspapers.to_csv(PATH, index=True)
    return newspapers


def newspaper_sentiment_pipeline(coin, newspaper_path=None, queries_path='queries.txt', NEGATIVE=-1, NEUTRAL=0,
                                 POSITIVE=1):
    # Step 1: get newspapers from queries
    nfq = None
    try:
        nfq = pd.read_csv(newspaper_path)
    except:
        nfq = newspapers_from_queries(coin, queries_path)

    # Step 2: sentiment analysis
    nfq = sentimentAnalysis(nfq, NEGATIVE, NEUTRAL, POSITIVE)

    # Step 3: Load in the newspaper data (with sentiment) and preprocess it
    coin_newspapers = pd.read_csv(f'../newspapers/{coin}_newspapers.csv')
    coin_newspapers['date'] = pd.to_datetime(coin_newspapers['date'], format="%m/%d/%Y, %I:%M %p, %z UTC")
    coin_newspapers['date'] = coin_newspapers['date'].dt.date
    coin_newspapers['score'] = nfq['score']
    coin_newspapers['sentiment'] = nfq['sentiment']

    # Step 4: Merge the newspaper data with the full/market data
    df = pd.read_csv(fullDataPath(coin))
    unnamed = df.columns.str.contains(r'^Unnamed: \d+(\.\d+)?$')
    df = df.loc[:, ~unnamed]
    df['time'] = pd.to_datetime(df['time'])
    coin_newspapers['date'] = pd.to_datetime(coin_newspapers['date'])
    merged_df = pd.merge(df, coin_newspapers, left_on='time', right_on='date', how='left')
    myFillNa(merged_df)
    merged_df.to_csv(fullDataPath(coin), index=True)

    return merged_df


def fullDataPath(coin):
    candidates = [
        f'../data/{coin}_df.csv',
        f'data/{coin}_df.csv',
        os.path.join('files', 'data', f'{coin}_df.csv'),
        os.path.join(BASE_DIR, 'data', f'{coin}_df.csv'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[-1]


def get_fgi_data(df):
    url = f'https://pro-api.coinmarketcap.com/v3/fear-and-greed/historical?CMC_PRO_API_KEY={CMC_KEY}&limit={min(len(df), 500)}'
    json = requests.get(url)
    if json.status_code == 200:
        data = json.json()
        fgi_df = pd.DataFrame(data['data'])
        fgi_df['timestamp'] = pd.to_datetime(fgi_df['timestamp'], unit='s')
        return fgi_df
    else:
        print("Couldn\'t get FGI data with status code = " + str(json.status_code))
        return None


def get_global_market_data():
    """
    Fetch overall market data such as total market capitalization, trading volume, and Bitcoin dominance.
    Purpose: Helps understand the overall health and activity of the cryptocurrency market.
    """
    url = f"https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
    headers = {
        'X-CMC_PRO_API_KEY': CMC_KEY
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return {
            'total_market_cap': data['data']['quote']['USD']['total_market_cap'],
            'total_volume_24h': data['data']['quote']['USD']['total_volume_24h'],
            'btc_dominance': data['data']['btc_dominance'],
            'active_cryptocurrencies': data['data']['active_cryptocurrencies']
        }
    else:
        print(f"Failed to fetch global market data: {response.status_code}")
        return None


def get_tokenomics(coin):
    """
    Fetch tokenomics data such as circulating supply and total supply.
    Purpose: Helps understand the supply dynamics of a coin, which can influence its price.
    """
    url = f"https://pro-api.coinmarketcap.com/v1/cryptocurrency/info"
    headers = {
        'X-CMC_PRO_API_KEY': CMC_KEY
    }
    params = {
        'symbol': coin
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        supply_data = data['data'][coin]
        return supply_data
    else:
        print(f"Failed to fetch tokenomics for {coin}: {response.status_code}")
        return None


def myFillNa(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(0, inplace=True)
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            df[col].fillna(CONSTANTS.EMPTY_STRING, inplace=True)
        else:
            df[col].fillna(np.nan, inplace=True)


def cv_metrics(model, data, yCol='gradient', v=5, trainingColsPath='files/training_columns.txt'):
    trainingCols = open(trainingColsPath, 'r').readlines()
    trainingCols = [i.strip() for i in trainingCols]
    assert yCol not in trainingCols, f'{yCol} should not be in trainingCols but was found in it'
    myFillNa(data)
    X = pd.get_dummies(data[trainingCols])
    y = data[yCol]
    cv_scores = -cross_val_score(model, X, y, cv=v, scoring='neg_root_mean_squared_error')
    cv_scores = pd.Series(cv_scores)
    cv_scores.index += 1
    cv_scores.plot.bar()
    print(f'CV RMSE: {cv_scores.mean()}')
    return cv_scores


def setup(coin, targetCol='gradient', closeCol='close'):
    data = pd.read_csv(fullDataPath(coin))
    trainingCols = open(TRAINING_COLUMNS, 'r').readlines()
    trainingCols = [i.strip() for i in trainingCols]
    setDiff = np.setdiff1d(trainingCols, data.columns)
    assert np.isin(trainingCols, data.columns).all(), f'{", ".join(setDiff)} not in data'
    X = data[trainingCols]
    data[targetCol] = data[closeCol].diff().fillna(0.0)
    data['TextType'] = data['link'].apply(lambda x: 'tweet' if x == CONSTANTS.EMPTY_STRING else 'newspaper')
    y = data[targetCol]
    return data, X, y


def prices(product_id, period=30, granularity=86400, start=None, end=None):
    """
    Fetch historical candlestick data for a cryptocurrency pair from now to the specified number of days in the past.

    :param product_id: The product ID for the crypto pair (e.g., 'BTC-USD').
    :param period: Number of days of historical data to fetch.
    :param granularity: Desired time slice in seconds (60, 300, 900, 3600, 21600, 86400).
    :return: DataFrame containing historical data.
    """
    if not product_id.endswith('-USD'):
        product_id += '-USD'
    product_id = product_id.upper()
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    if start is None and end is None:  # get data from specified number of days ago if date bounds are not specified.
        end = datetime.now()
        start = end - timedelta(days=period)
    coin = product_id.split('-')[0]
    all_data = []

    while start < end:
        end_slice = min(start + timedelta(seconds=granularity * 300), end)
        params = {
            'start': start.isoformat(),
            'end': end_slice.isoformat(),
            'granularity': granularity
        }

        try:
            response = requests.get(url, params=params)
        except Exception as e:
            print('Fetch failed with exception: ', e)
            return None, coin

        if response.status_code == 200:
            data = response.json()
            all_data.extend(data)
        else:
            print("Failed to fetch data:", response.text)
            break

        start = end_slice

    if all_data:
        columns = ['time', 'low', 'high', 'open', 'close', 'volume']
        data = pd.DataFrame(all_data, columns=columns)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        data['change'] = data['close'] - data['open']
        data['pct_change'] = (data['change'] / data['open']) * 100
        return data, coin
    return None, coin


def sequence(column, index):
    if index >= len(column):
        raise ValueError(f'Index = {index} ≥ Length = {len(column)}')

    sequence_data = column.values[:index].tolist()
    remaining_values = len(column) - index - 1
    next_values_count = min(TEST_DAYS, remaining_values)
    if next_values_count == 0:
        # Use the last available value (at index) repeated 7 times
        last_value = column.values[index] if index < len(column) else column[index]
        return sequence_data, [last_value] * TEST_DAYS
    next_values = column.values[index:(index + next_values_count)].tolist()

    # Repeat the last available value to fill up to 7
    if len(next_values) < TEST_DAYS:
        last_value = next_values[-1]
        while len(next_values) < TEST_DAYS:
            next_values.append(last_value)

    return sequence_data, next_values


def padding(sequence, target_length=5):
    """Pad or truncate sequence to target_length"""
    if len(sequence) > target_length:
        return sequence[:target_length]  # Truncate
    elif len(sequence) < target_length:
        return sequence + [CONSTANTS.FILL] * (target_length - len(sequence))  # Do padding
    return sequence  # Already correct length


def compute_rsi(series, window=14):
    delta = series.diff()
    # Separate positive and negative gains
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Simple moving average over 'window' periods for gains/losses
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def normalize_sequences(df_column, scaler):
    normalized_sequences = []
    for seq in df_column:
        seq_array = np.array(seq).reshape(-1, 1)
        normalized_seq = scaler.fit_transform(seq_array).flatten()
        normalized_sequences.append(normalized_seq.tolist())
    return normalized_sequences


def dataSetup(data, trainingColPath='training_columns.txt', response='close',  number=None):
    """
    Sets up the data for training
    :param data: full dataset
    :param trainingColPath: path to specified columns to be used in training
    :param response: response variable
    :return: dataset where each OU is a day combined with sentiment for all news found that day about the coin
    """

    data['time'] = pd.to_datetime(data['time'], errors='coerce')
    data = data.dropna(subset=['time'])
    data['date'] = data['time'].dt.date

    # Group by date and aggregate values (one row per day)
    with open(trainingColPath, 'r') as file:
        trainingCols = [i.strip() for i in file.readlines()]
    d = {}
    trainingCols.append(response)

    # set up the aggregation dictionary kwargs
    for col in trainingCols:
        if col == 'avg_sentiment':
            d[col] = ('score', 'mean')
        elif col == 'tweet_count':
            d[col] = ('score', 'count')
        else:
            d[col] = (col, 'last')

    # Perform the aggregation
    daily_data = (
        data
        .groupby('date')
        .agg(**d)
        .reset_index()
    )

    # Clean up dataset
    daily_data['time'] = pd.to_datetime(daily_data['date'])
    daily_data = daily_data.drop('date', axis=1)
    daily_data = daily_data.sort_values('time')
    daily_data.set_index('time', inplace=True)
    daily_data['gradient'] = daily_data['close'].diff().fillna(0.0)  # proper gradient
    if number:
        daily_data = daily_data.iloc[-number:, :]
    return daily_data


def predict_sequence(model, price, starter, scaler, sequence_length=30, total_length=365):
    """Generate a sequence of future price predictions autoregressively.

    Args:
        model: trained model with predict method expecting DataFrame of sequences
        price: last known price to seed the sequence
        starter: recent historical price series (pd.Series)
        scaler: fitted scaler used to inverse transform predictions
        sequence_length: number of future steps to predict
        total_length: total length for padded input sequence
    Returns:
        list[float]: predicted future prices length = sequence_length
    """

    # Start with the initial price
    current_sequence = starter.values.tolist()
    current_sequence.append(price)
    current_sequence = current_sequence[-total_length:]  # Keep only the last 'total_length' prices

    # Generate predictions one by one using the rolling window with existing predictions
    for i in range(sequence_length):
        input_data = padding(current_sequence, target_length=total_length)
        input_df = pd.DataFrame({'sequences': [input_data]})
        next_prediction = model.predict(input_df)[0][0]
        next_prediction = scaler.inverse_transform([[next_prediction]])[0][0]
        current_sequence.append(next_prediction)
        current_sequence = current_sequence[-total_length:]

    return current_sequence[1:(sequence_length + 1)]


def transformerDataSetup(daily_data, col='close'):
    seqs = []
    nexts = []
    for i in range(len(daily_data)):
        try:
            seq, next = sequence(daily_data[col], i)
            seqs.append(seq)
            nexts.append(next)
        except Exception as e:
            print(f'Error at {i} | Exception:', e)

    seqs = [padding(seq, len(daily_data)) for seq in seqs]
    daily_data['sequence'] = seqs
    daily_data['next'] = nexts
    return daily_data


def transformerXTrainYTrain(daily_data, testSize):
    train_data = daily_data.iloc[:testSize]
    train_data_idx = train_data.index
    test_data = daily_data.iloc[testSize:]
    test_data_idx = test_data.index
    X_train = train_data[['sequence']]
    y_train = train_data['next']
    X_test = test_data[['sequence']]
    y_test = test_data['next']
    X_train.index = train_data_idx
    X_test.index = test_data_idx
    y_train.index = train_data_idx
    y_test.index = test_data_idx
    return X_train, X_test, y_train, y_test, None, None


def normalize(X_train, X_test, y_train, y_test):
    # Create scalers for both sequences and targets
    xTrainIdx = X_train.index
    xTestIdx = X_test.index
    yTrainIdx = y_train.index
    yTestIdx = y_test.index

    sequence_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit the scalers on the training data
    y_train_arrays = np.array(y_train.tolist())  # Shape: (n_samples, 7)
    y_test_arrays = np.array(y_test.tolist())  # Shape: (n_samples, 7)
    y_train_flat = y_train_arrays.flatten().reshape(-1, 1)
    target_scaler.fit(y_train_flat)

    # Transform the target data
    y_train_scaled = target_scaler.transform(y_train_arrays.reshape(-1, 1)).reshape(y_train_arrays.shape)
    y_test_scaled = target_scaler.transform(y_test_arrays.reshape(-1, 1)).reshape(y_test_arrays.shape)

    # Apply normalization to sequence data
    X_train_norm = pd.DataFrame({
        'sequences': normalize_sequences(X_train.iloc[:, 0], sequence_scaler)
    })
    X_test_norm = pd.DataFrame({
        'sequences': normalize_sequences(X_test.iloc[:, 0], sequence_scaler)
    })

    y_train_norm = pd.Series([row.tolist() for row in y_train_scaled])
    y_test_norm = pd.Series([row.tolist() for row in y_test_scaled])
    X_train_norm.index = xTrainIdx
    X_test_norm.index = xTestIdx
    y_train_norm.index = yTrainIdx
    y_test_norm.index = yTestIdx
    return X_train_norm, X_test_norm, y_train_norm, y_test_norm, sequence_scaler, target_scaler


def normalize_regular(X_train, X_test, y_train, y_test):
    sequence_scaler = StandardScaler()
    target_scaler = StandardScaler()

    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
    X_train_scaled = sequence_scaler.fit_transform(X_train)
    X_test_scaled = sequence_scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    y_train_scaled = pd.Series(y_train_scaled)
    y_test_scaled = pd.Series(y_test_scaled)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, sequence_scaler, target_scaler


def predictNextNDaysTransformer(model, daily_data, sequence_scaler, RESPONSE='close', n=7, total_length=365):
    """
    predict the next n (default 7) close prices using the provided transformer
    :param model:
    :param daily_data:
    :param sequence_scaler:
    :param RESPONSE:
    :param n:
    :param total_length:
    :return:
    """
    price = daily_data[RESPONSE].iloc[-1]
    starter = daily_data[RESPONSE].iloc[-30:]
    predictions = predict_sequence(model, price, starter, sequence_scaler, sequence_length=n, total_length=total_length)
    new_days = pd.date_range(start=daily_data.index[-1] + pd.Timedelta(days=1), periods=len(predictions), freq='D')
    predictions_df = pd.DataFrame(predictions, index=new_days, columns=[RESPONSE])
    predictions_df.plot.line()
    plt.show()
    return predictions_df


def trainingCols(path=TRAINING_COLUMNS):
    with open(path, 'r') as file:
        trainingCols = [i.strip() for i in file.readlines()]
    return trainingCols


def myLoss(y_true, y_pred):
    """
    Custom loss function for training
    :param y_true: true values
    :param y_pred: predicted values
    :return: loss value
    """
    return torch.mean((y_true - y_pred) ** 2) - torch.std(y_pred)


def normalize_col(df, column_name):
    """Normalize a specified column in a DataFrame using StandardScaler."""
    scaler = StandardScaler()
    df[column_name] = scaler.fit_transform(df[[column_name]])
    return df


def preprocess_data(data, scaler=StandardScaler(), closeScaler=StandardScaler(), response='close'):
    mergedIndex = data.index
    mergedCols = data.columns

    mergedWithoutClose = data.drop(columns=[response])
    mwocCols = mergedWithoutClose.columns
    mwocIndex = mergedWithoutClose.index
    mergedWithoutClose = scaler.fit_transform(mergedWithoutClose)
    mergedWithoutClose = pd.DataFrame(mergedWithoutClose, index=mergedIndex, columns=mwocCols)

    data[response] = closeScaler.fit_transform(data[response].values.reshape(-1, 1))
    data = pd.concat([mergedWithoutClose, data[response]], axis=1)

    # Simple train-test split for Random Forest (no sequences needed)
    # Remove the last TEST_DAYS for final testing
    test_size_days = TEST_DAYS
    train_data = data.iloc[:-test_size_days]
    test_data = data.iloc[-test_size_days:]

    # Prepare features and target - no need to duplicate training columns
    X_train = train_data.drop(columns=[response])  # This already contains all training columns
    y_train = train_data[response].values  # Single values, not sequences
    X_test = test_data.drop(columns=[response])  # This already contains all training columns
    y_test = test_data[response].values

    # Use the features directly (no need to concat training_cols again)
    X_train_norm = X_train
    X_test_norm = X_test

    return X_train_norm, X_test_norm, y_train, y_test, scaler, closeScaler


def create_sequences(data, sequence_length=30, prediction_horizon=1):
    """Create sequences for time series prediction.
    Expects `data` as a 2D numpy array where the last column is the target (close).
    Returns X shaped (n_samples, sequence_length, n_features) and y shaped (n_samples, prediction_horizon).
    """
    X, y = [], []
    for i in range(sequence_length, len(data) - prediction_horizon + 1):
        X.append(data[i - sequence_length:i])
        y.append(data[i:i + prediction_horizon, -1])
    return np.array(X), np.array(y)


def normalize_data(data, method='standard', target_col='close'):
    """Normalize data using different methods and return scaled array and scalers.
    `data` should be a pandas DataFrame.
    """
    data_copy = data.copy()

    # Separate target column
    target_data = data_copy[target_col].values.reshape(-1, 1)
    feature_data = data_copy.drop(columns=[target_col]).values

    if method == 'standard':
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
    elif method == 'minmax':
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
    elif method == 'robust':
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
    elif method == 'log':
        # Log transform (add small constant to avoid log(0))
        feature_data = np.log(feature_data + 1e-8)
        target_data = np.log(target_data + 1e-8)
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
    else:
        raise ValueError("Method must be 'standard', 'minmax', 'robust', or 'log'")

    # Fit and transform
    feature_data_scaled = feature_scaler.fit_transform(feature_data)
    target_data_scaled = target_scaler.fit_transform(target_data)

    # Combine back
    scaled_data = np.column_stack([feature_data_scaled, target_data_scaled.flatten()])

    return scaled_data, feature_scaler, target_scaler


# ===================== KNN PIPELINE HELPERS =====================

# New helper scaler classes/factories to enable tuning normalization
class IdentityScaler:
    """A passthrough scaler used when no scaling is desired."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


def _scaler_factory(method: str):
    method = method.lower()
    if method == 'standard':
        return StandardScaler()
    if method == 'minmax':
        return MinMaxScaler()
    if method == 'robust':
        return RobustScaler()
    if method == 'none':
        return IdentityScaler()
    raise ValueError(f"Unknown scaler method: {method}")


# Modified to accept a scaling method

def _scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, method: str = 'standard'):
    scaler = _scaler_factory(method)
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
    return scaler, X_train_scaled, X_val_scaled


def _knn_param_grid(X_train_len: int):
    # Filter neighbor counts so they are <= training size
    base_neighbors = [3, 5, 7, 9, 11, 15]
    valid_neighbors = [k for k in base_neighbors if k <= max(1, X_train_len)]
    if len(valid_neighbors) == 0:
        valid_neighbors = [1]
    return {
        'n_neighbors': valid_neighbors,
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }


def _knn_grid_search(X_train, y_train, param_grid: dict, n_splits: int = 3):
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(X_train) // 4)))
    best_score = np.inf
    best_params = None
    results = []
    for params in ParameterGrid(param_grid):
        fold_rmses = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
            if params['n_neighbors'] > len(X_tr):
                continue  # skip invalid config
            model = KNeighborsRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            rmse = mean_squared_error(y_va, preds, squared=False)
            fold_rmses.append(rmse)
        if len(fold_rmses) == 0:
            continue
        avg_rmse = float(np.mean(fold_rmses))
        results.append({**params, 'rmse': avg_rmse})
        if avg_rmse < best_score:
            best_score = avg_rmse
            best_params = params
    return best_params, results


def _knn_full_grid_search(X_train: pd.DataFrame, y_train: pd.Series, scaler_methods=None, base_param_grid=None,
                          n_splits: int = 3):
    """Grid search across scaler methods and KNN hyperparameters with time-series CV.
    Returns (best_params_with_scaler, results_list).
    best_params_with_scaler includes key 'scaler' for the chosen normalization method.
    """
    if scaler_methods is None:
        scaler_methods = ['standard', 'minmax', 'robust']
    if base_param_grid is None:
        base_param_grid = _knn_param_grid(len(X_train))

    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(X_train) // 4)))
    best_score = np.inf
    best_combo = None
    results = []

    for scaler_method in scaler_methods:
        for params in ParameterGrid(base_param_grid):
            fold_rmses = []
            valid_config = True
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
                if params['n_neighbors'] > len(X_tr):
                    valid_config = False
                    break
                scaler = _scaler_factory(scaler_method)
                X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
                X_va_scaled = pd.DataFrame(scaler.transform(X_va), index=X_va.index, columns=X_va.columns)
                model = KNeighborsRegressor(**params)
                model.fit(X_tr_scaled, y_tr)
                preds = model.predict(X_va_scaled)
                rmse = mean_squared_error(y_va, preds, squared=False)
                fold_rmses.append(rmse)
            if not valid_config or len(fold_rmses) == 0:
                continue
            avg_rmse = float(np.mean(fold_rmses))
            record = {**params, 'scaler': scaler_method, 'rmse': avg_rmse}
            results.append(record)
            if avg_rmse < best_score:
                best_score = avg_rmse
                best_combo = record.copy()
    return best_combo, results


def _standardized_rmse(y_true: pd.Series, y_pred: np.ndarray):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    denom = y_true.std()
    if denom == 0 or np.isnan(denom):
        return rmse, rmse
    return rmse, rmse / denom


def _save_validation_predictions(df: pd.DataFrame, coin: str, model_name: str):
    path = os.path.join(base_dir('predictions'), coin)
    path += f'{model_name}_validation_predictions.csv'
    df.to_csv(path, index=True)
    return path


def _save_future_predictions(df: pd.DataFrame, coin: str, model_name: str):
    path = os.path.join(base_dir('predictions'), coin)
    path += f'{model_name}_future_predictions.csv'
    df.to_csv(path, index=True)
    return path


def _save_standardized_rmse(value: float, coin: str, model_name: str):
    path = os.path.join(base_dir('predictions'), coin)
    path += f'{model_name}_standardized_rmse.txt'
    with open(path, 'w') as f:
        f.write(f"{value:.6f}\n")
    return path


def _save_knn_model(model: KNeighborsRegressor, scaler: StandardScaler, coin: str):
    model_path = os.path.join(base_dir('models'), coin)
    model_path += f'{coin}_knn_model.pkl'
    scaler_path = os.path.join(base_dir('models'), coin)
    scaler_path += f'{coin}_scaler.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    return model_path, scaler_path


def _knn_future_forecast(model: KNeighborsRegressor, scaler: StandardScaler, X_full: pd.DataFrame,
                         daily_data: pd.DataFrame, training_cols: list, response: str = RESPONSE, n: int = TEST_DAYS):
    # Use last available feature row, iteratively update price-related columns with predicted price
    last_features = X_full.iloc[-1].copy()
    predictions = []
    current_features = last_features.copy()
    for _ in range(n):
        scaled_vec = scaler.transform(pd.DataFrame([current_features], columns=training_cols))
        pred = float(model.predict(scaled_vec)[0])
        predictions.append(pred)
        # Update price-related fields if they are part of the features
        for price_col in ['close', 'open', 'high', 'low']:
            if price_col in current_features.index:
                current_features[price_col] = pred
        # Moving averages / bands left unchanged (simple baseline)
    future_index = pd.date_range(start=daily_data.index[-1] + pd.Timedelta(days=1), periods=n, freq='D')
    future_df = pd.DataFrame(predictions, index=future_index, columns=['predicted_price'])
    return future_df


def run_knn_pipeline(coin: str = COIN, response: str = RESPONSE, training_cols_path: str = TRAINING_COLUMNS):
    """End-to-end KNN pipeline: data prep, split, tune, train, validate, forecast next 7 days, save outputs."""
    model_tag = 'knn'
    daily_data, X, y, training_cols = _prepare_knn_dataset(coin, response=response,
                                                           training_cols_path=training_cols_path)
    X_train, X_val, y_train, y_val = _time_series_train_val_split(X, y)
    scaler, X_train_scaled, X_val_scaled = _scale_features(X_train, X_val)

    # Hyperparameter grid & tuning
    grid = _knn_param_grid(len(X_train_scaled))
    best_params, tuning_results = _knn_grid_search(X_train_scaled, y_train, grid)
    if best_params is None:
        # Fallback to default
        best_params = {'n_neighbors': min(5, len(X_train_scaled)), 'weights': 'distance', 'p': 2}

    # Train best model
    best_model = KNeighborsRegressor(**best_params)
    best_model.fit(X_train_scaled, y_train)

    # Validation predictions
    val_preds = best_model.predict(X_val_scaled)
    val_df = pd.DataFrame({'predicted_price': val_preds}, index=X_val_scaled.index)

    # Metrics
    rmse, std_rmse = _standardized_rmse(y_val, val_preds)

    # Save artifacts
    _save_validation_predictions(val_df, coin, model_tag)
    _save_standardized_rmse(std_rmse, coin, model_tag)
    _save_knn_model(best_model, scaler, coin)

    # Future predictions (use full feature set scaled)
    X_full_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    future_df = _knn_future_forecast(best_model, scaler, X, daily_data, training_cols, response=response, n=TEST_DAYS)
    _save_future_predictions(future_df, coin, model_tag)

    return {
        'best_params': best_params,
        'rmse': rmse,
        'standardized_rmse': std_rmse,
        'validation_predictions': val_df,
        'future_predictions': future_df,
        'tuning_results': tuning_results
    }


# Reinstate missing KNN dataset preparation helpers (were removed in prior edit)

def _prepare_knn_dataset(coin: str, response: str = RESPONSE, training_cols_path: str = TRAINING_COLUMNS):
    raw_path = fullDataPath(coin)
    data = pd.read_csv(raw_path)
    daily_data = dataSetup(data, trainingColPath=training_cols_path, response=response)
    training_cols = trainingCols(training_cols_path)
    missing = [c for c in training_cols if c not in daily_data.columns]
    assert len(missing) == 0, f"Missing training columns: {missing}"
    X = daily_data[training_cols].copy()
    y = daily_data[response].copy()
    return daily_data, X, y, training_cols


def _time_series_train_val_split(X: pd.DataFrame, y: pd.Series, train_pct: float = TRAIN_PCT):
    n = len(X)
    split_idx = max(1, int(n * train_pct))
    if split_idx >= n:
        split_idx = n - 1
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_val, y_train, y_val


# ===================== END KNN PIPELINE HELPERS =====================
# ===================== PROPHET PIPELINE HELPERS =====================
try:
    from prophet import Prophet
except ImportError:  # allow file to load without prophet installed
    Prophet = None


class ProphetWrapper:
    """Light wrapper to store Prophet model and associated scalers/metadata."""

    def __init__(self, model: Prophet, target_scaler, feature_scaler, features, scaler_method: str, params: dict):
        self.model = model
        self.target_scaler = target_scaler
        self.feature_scaler = feature_scaler
        self.features = features
        self.scaler_method = scaler_method
        self.params = params

    def predict(self, df_future: pd.DataFrame):
        return self.model.predict(df_future)


def _prepare_prophet_dataset(coin: str, response: str = RESPONSE, training_cols_path: str = TRAINING_COLUMNS):
    raw_path = fullDataPath(coin)
    data = pd.read_csv(raw_path)
    daily_data = dataSetup(data, trainingColPath=training_cols_path, response=response)
    training_cols = trainingCols(training_cols_path)
    missing = [c for c in training_cols if c not in daily_data.columns]
    assert len(missing) == 0, f"Missing training columns: {missing}"
    return daily_data, training_cols


def _prophet_scale(daily_data: pd.DataFrame, features: list, target: str, method: str):
    """Scale features & target; returns scaled df, feature_scaler, target_scaler.
    method in {'standard','minmax','robust','log','none'}.
    """
    df = daily_data.copy()
    feature_scaler = _scaler_factory(
        method if method != 'log' else 'standard') if method != 'none' else IdentityScaler()
    target_scaler = _scaler_factory(method if method != 'log' else 'standard') if method != 'none' else IdentityScaler()

    feat_df = df[features].copy()
    tgt_series = df[target].values.reshape(-1, 1)

    if method == 'log':
        # log transform before standard scaling
        feat_df = np.log(feat_df.replace(0, np.nan).fillna(method='bfill').fillna(method='ffill').abs() + 1e-8)
        tgt_series = np.log(
            df[target].replace(0, np.nan).fillna(method='bfill').fillna(method='ffill').abs().values.reshape(-1,
                                                                                                             1) + 1e-8)

    feat_scaled = feature_scaler.fit_transform(feat_df.values)
    tgt_scaled = target_scaler.fit_transform(tgt_series)

    df_scaled = pd.DataFrame(feat_scaled, index=df.index, columns=features)
    df_scaled[target] = tgt_scaled.flatten()
    return df_scaled, feature_scaler, target_scaler


def _prophet_param_grid():
    return list(ParameterGrid({
        'seasonality_mode': ['additive', 'multiplicative'],
        'changepoint_prior_scale': [0.01, 0.1, 0.5],
        'seasonality_prior_scale': [1.0, 5.0, 10.0],
        'weekly_fourier_order': [3, 5],
        'yearly_seasonality': [True, False]
    }))


def _prophet_train_single(train_df: pd.DataFrame, features: list, params: dict):
    if Prophet is None:
        raise ImportError("prophet library not installed")
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,  # we'll add custom weekly via fourier
        yearly_seasonality=params['yearly_seasonality'],
        seasonality_mode=params['seasonality_mode'],
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale']
    )
    # Add weekly custom seasonality
    model.add_seasonality(name='weekly', period=7, fourier_order=params['weekly_fourier_order'])
    # Add regressors
    for f in features:
        model.add_regressor(f)
    model.fit(train_df)
    return model


def _prophet_time_split(df_scaled: pd.DataFrame, train_pct: float):
    n = len(df_scaled)
    split_idx = max(1, int(n * train_pct))
    if split_idx >= n:
        split_idx = n - 1
    train_df = df_scaled.iloc[:split_idx]
    val_df = df_scaled.iloc[split_idx:]
    return train_df, val_df


def _prophet_build_train_frame(df_scaled: pd.DataFrame, target: str):
    dfp = df_scaled.copy()
    dfp = dfp.reset_index().rename(columns={'time': 'ds', target: 'y'})
    return dfp


def _prophet_make_future(val_df_scaled: pd.DataFrame, full_scaled: pd.DataFrame, features: list, target: str):
    future = val_df_scaled.reset_index().rename(columns={'time': 'ds', target: 'y'})[['ds'] + features]
    return future


def _prophet_inverse_target(values: np.ndarray, target_scaler, method: str):
    if method == 'log':
        inv = target_scaler.inverse_transform(values.reshape(-1, 1))
        return np.exp(inv).flatten()
    if isinstance(target_scaler, IdentityScaler):
        return values
    return target_scaler.inverse_transform(values.reshape(-1, 1)).flatten()


def _prophet_tune(daily_data: pd.DataFrame, features: list, target: str, train_pct: float, scaler_methods=None):
    if scaler_methods is None:
        scaler_methods = ['standard', 'minmax', 'robust']
    param_grid = _prophet_param_grid()
    best_score = np.inf
    best_combo = None
    results = []
    for sm in scaler_methods:
        df_scaled, feature_scaler, target_scaler = _prophet_scale(daily_data, features, target, sm)
        train_scaled, val_scaled = _prophet_time_split(df_scaled, train_pct)
        train_df = _prophet_build_train_frame(train_scaled, target)
        val_future = _prophet_make_future(val_scaled, df_scaled, features, target)
        y_val_orig = daily_data.loc[val_scaled.index, target].values  # original scale
        for params in param_grid:
            try:
                model = _prophet_train_single(train_df, features, params)
                # Add regressors values for prediction
                future = val_future.copy()
                preds_df = model.predict(future)
                yhat_scaled = preds_df['yhat'].values
                yhat = _prophet_inverse_target(yhat_scaled, target_scaler, sm)
                rmse = mean_squared_error(y_val_orig, yhat, squared=False)
                denom = y_val_orig.std()
                std_rmse = rmse if denom == 0 or np.isnan(denom) else rmse / denom
                rec = {**params, 'scaler': sm, 'rmse': rmse, 'standardized_rmse': std_rmse}
                results.append(rec)
                if std_rmse < best_score:
                    best_score = std_rmse
                    best_combo = {
                        'params': params,
                        'scaler_method': sm,
                        'rmse': rmse,
                        'standardized_rmse': std_rmse,
                        'feature_scaler': feature_scaler,
                        'target_scaler': target_scaler
                    }
            except Exception as e:
                # Skip failed configuration
                continue
    return best_combo, results


def _prophet_future_forecast(wrapper: ProphetWrapper, daily_data: pd.DataFrame, features: list, target: str,
                             n: int = TEST_DAYS):
    last_date = daily_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n, freq='D')
    # Build future regressors DataFrame using last known values (naive)
    last_row = daily_data.iloc[-1]
    reg_mat = np.tile(last_row[features].values, (n, 1))
    # scale regressors according to scaler method
    feature_scaler = wrapper.feature_scaler
    if isinstance(feature_scaler, IdentityScaler):
        reg_scaled = reg_mat
    else:
        reg_scaled = feature_scaler.transform(reg_mat)
    future_df = pd.DataFrame(reg_scaled, columns=features)
    future_df['ds'] = future_dates
    # Predict
    preds = wrapper.model.predict(future_df)
    yhat_scaled = preds['yhat'].values
    if wrapper.scaler_method == 'log':
        inv_target = np.exp(wrapper.target_scaler.inverse_transform(yhat_scaled.reshape(-1, 1))).flatten()
    elif isinstance(wrapper.target_scaler, IdentityScaler):
        inv_target = yhat_scaled
    else:
        inv_target = wrapper.target_scaler.inverse_transform(yhat_scaled.reshape(-1, 1)).flatten()
    return pd.DataFrame({'predicted_price': inv_target}, index=future_dates)


def _save_prophet_model(wrapper: ProphetWrapper, coin: str):
    import pickle
    model_dir = f'models/{coin}'
    os.makedirs(model_dir, exist_ok=True)
    path = f'{model_dir}/{coin}_prophet_model.pkl'
    with open(path, 'wb') as f:
        pickle.dump(wrapper, f)
    return path


def run_prophet_pipeline(coin: str = COIN, response: str = RESPONSE, training_cols_path: str = TRAINING_COLUMNS,
                         train_pct: float = TRAIN_PCT, scaler_methods=None, forecast_days: int = TEST_DAYS):
    """End-to-end Prophet pipeline with scaler + hyperparameter tuning. Produces one future prediction file and standardized RMSE file."""
    daily_data, features = _prepare_prophet_dataset(coin, response=response, training_cols_path=training_cols_path)
    best_combo, tuning_results = _prophet_tune(daily_data, features, response, train_pct, scaler_methods=scaler_methods)
    if best_combo is None:
        raise RuntimeError('Prophet tuning failed to produce a valid model.')
    # Retrain on full dataset using best scaler & params
    df_scaled_full, feature_scaler, target_scaler = _prophet_scale(daily_data, features, response,
                                                                   best_combo['scaler_method'])
    full_train_df = _prophet_build_train_frame(df_scaled_full, response)
    final_model = _prophet_train_single(full_train_df, features, best_combo['params'])
    wrapper = ProphetWrapper(final_model, target_scaler, feature_scaler, features, best_combo['scaler_method'],
                             best_combo['params'])
    future_df = _prophet_future_forecast(wrapper, daily_data, features, response, n=forecast_days)
    correct_path = os.path.join(base_dir('predictions'), coin)
    # Save artifacts
    os.makedirs(f'{correct_path}/{coin}', exist_ok=True)
    future_path = f'{correct_path}/{coin}/prophet_future_predictions.csv'
    future_df.to_csv(future_path, index=True)
    std_rmse_path = f'{correct_path}/{coin}/prophet_standardized_rmse.txt'
    with open(std_rmse_path, 'w') as f:
        f.write(f"{best_combo['standardized_rmse']:.6f}\n")
    model_path = _save_prophet_model(wrapper, coin)
    # Save tuning results
    tuning_path = f'{correct_path}/{coin}/prophet_tuning_results.csv'
    pd.DataFrame(tuning_results).sort_values('standardized_rmse').to_csv(tuning_path, index=False)
    return {
        'best_params': best_combo['params'],
        'scaler_method': best_combo['scaler_method'],
        'rmse': best_combo['rmse'],
        'standardized_rmse': best_combo['standardized_rmse'],
        'future_predictions': future_df,
        'model_path': model_path,
        'future_path': future_path,
        'std_rmse_path': std_rmse_path,
        'tuning_path': tuning_path
    }


def train_validation_split(series, test_size):
    """
    Splits a pandas Series into train and validation sets.
    Args:
        series (pd.Series): The time series data.
        test_size (int): Number of samples for validation set.
    Returns:
        train (pd.Series): Training set.
        val (pd.Series): Validation set.
    """
    train = series.iloc[:-test_size]
    val = series.iloc[-test_size:]
    return train, val


def standardize_series(series: pd.Series, method: str):
    """
    Apply selected transformation and return a pandas Series with the original index preserved.
    Returns (transformed_series (pd.Series), params (dict)).
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    method = method.lower()
    if method == 'none':
        return series.astype(float).copy(), {'method': 'none'}
    if method == 'log':
        shift = 0.0
        if (series <= 0).any():
            shift = float(abs(series.min()) + 1)
        transformed = np.log(series + shift)
        return pd.Series(transformed, index=series.index, dtype=float), {'method': 'log', 'shift': shift}
    if method == 'minmax':
        min_v = float(series.min())
        max_v = float(series.max())
        rng = max_v - min_v if max_v != min_v else 1.0
        transformed = (series - min_v) / rng
        return pd.Series(transformed, index=series.index, dtype=float), {'method': 'minmax', 'min': min_v, 'range': rng}
    if method == 'zscore':
        mean_v = float(series.mean())
        std_v = float(series.std()) if series.std() != 0 else 1.0
        transformed = (series - mean_v) / std_v
        return pd.Series(transformed, index=series.index, dtype=float), {'method': 'zscore', 'mean': mean_v,
                                                                         'std': std_v}
    raise ValueError(f"Unknown standardization method: {method}")


def inverse_transform(values, method: str, params: Dict):
    """
    Inverse transform values (array-like or pd.Series) using params produced by standardize_series.
    Preserves index when possible and returns a pd.Series of floats.
    """
    # Accept pd.Series or numpy array
    if isinstance(values, pd.Series):
        idx = values.index
        vals = values.values
    else:
        try:
            vals = np.asarray(values)
            idx = None
        except Exception:
            vals = np.array(values)
            idx = None

    method_used = params.get('method', method).lower() if isinstance(params, dict) else method.lower()

    if method_used == 'none':
        res = pd.Series(vals.astype(float))
    elif method_used == 'log':
        shift = params.get('shift', 0.0)
        res = pd.Series(np.exp(vals) - shift)
    elif method_used == 'minmax':
        rng = params.get('range', 1.0)
        min_v = params.get('min', 0.0)
        res = pd.Series(vals * rng + min_v)
    elif method_used == 'zscore':
        std_v = params.get('std', 1.0)
        mean_v = params.get('mean', 0.0)
        res = pd.Series(vals * std_v + mean_v)
    else:
        raise ValueError(f"Unknown inverse method: {method_used}")

    # Preserve index if we have it
    if idx is not None:
        res.index = idx
    return res.astype(float)


def evaluate_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Standardized RMSE = RMSE / std(actual).
    """
    actual = actual.astype(float)
    predicted = predicted.astype(float)
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    denom = actual.std() or 1.0
    return rmse


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_predictions(df: pd.DataFrame, filepath: str):
    ensure_dir(Path(filepath).parent)
    df.to_csv(filepath, index=True)


def save_rmse(value: float, filepath: str):
    ensure_dir(Path(filepath).parent)
    with open(filepath, 'w') as f:
        f.write(f"{value:.6f}")


def generate_future_index(last_index, periods: int = 7):
    """
    Generate next 'periods' daily timestamps after last_index.
    Assumes DatetimeIndex or convertible.
    """
    if not isinstance(last_index, pd.Timestamp):
        last_index = pd.to_datetime(last_index)
    return pd.date_range(start=last_index + pd.Timedelta(days=1), periods=periods, freq='D')

def coinbase_market_analysis_gradient(c, dataPath, plotPath):
    start = datetime.strptime('01/01/2008', '%m/%d/%Y')
    end = datetime.now()
    df, coin = prices(c, start=start, end=end)
    df.sort_values('time', ascending=False, inplace=True)

    fgi = get_fgi_data(df)
    df = pd.merge(df, fgi, left_on='time', right_on='timestamp', how='left')
    df['timestamp'].fillna(pd.to_datetime('1970-01-01'), inplace=True)
    val = df[df['value'].isna() == False]['value'].values[0]  # Last known FGI value
    fgiClassification = df[df['value_classification'].isna() == False]['value_classification'].values[
        0]  # Last known FGI classification
    df['value'].fillna(val, inplace=True)
    df['value_classification'].fillna(fgiClassification, inplace=True)

    # Short-term SMA (20 periods) and long-term SMA (50 periods)
    df['SMA_20'] = df['close'].rolling(window=20).mean().fillna(method='bfill')
    df['SMA_50'] = df['close'].rolling(window=50).mean().fillna(method='bfill')
    df[f'SMA_{TEST_DAYS}'] = df['close'].rolling(window=TEST_DAYS).mean().fillna(method='bfill')

    # Exponential Moving Averages (optional, can be used in place of or along with SMAs)
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean().fillna(method='bfill')
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean().fillna(method='bfill')
    df['RSI'] = compute_rsi(df['close'], window=TEST_DAYS)

    # Calculate the 12-period and 26-period EMA of the closing price
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    df['BB_Middle'] = df['SMA_20']  # Middle band is the 20-period SMA
    df['BB_STD'] = df['close'].rolling(window=20).std().fillna(method='bfill')
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_STD']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_STD']
    df[f'Volume_MA_{TEST_DAYS}'] = df['volume'].rolling(window=TEST_DAYS).mean()

    # Daily Return and On-Balance Volume (OBV)
    df['daily_return'] = df['close'].pct_change()
    df['direction'] = np.where(df['daily_return'] > 0, 1, -1)
    df.loc[df['daily_return'].isna(), 'direction'] = 0
    df['OBV'] = (df['volume'] * df['direction']).fillna(0).cumsum()  # cumulative on balance volume
    df.drop(columns=['daily_return', 'direction'], inplace=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot the price and Bollinger Bands
    ax.plot(df['time'], df['close'], label='Close Price', color='blue')
    ax.plot(df['time'], df['BB_Upper'], label='Upper Band', color='green')
    ax.plot(df['time'], df['BB_Middle'], label='Middle Band', color='orange')
    ax.plot(df['time'], df['BB_Lower'], label='Lower Band', color='red')
    ax.fill_between(df['time'], df['BB_Upper'], df['BB_Middle'], color='green', alpha=0.3)
    ax.fill_between(df['time'], df['BB_Middle'], df['BB_Lower'], color='red', alpha=0.3)

    ax.set_xlabel('Year')
    ax.set_ylabel('Price')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.set_title(f'{c} Bollinger Bands with Shaded Regions')
    ax.legend()
    ax.grid(True)

    fig.autofmt_xdate()
    if not os.path.exists(plotPath):
        os.makedirs(plotPath)
    plt.savefig(f'{plotPath}/{c}_Bollinger_Bands.png')
    try:
        df.set_index(df['time'], inplace=True, drop=True)
        df.drop('time', axis=1, inplace=True)
    except:
        pass
    df = df.sort_values(by=['time'], ascending=True)
    df.to_csv(f'{dataPath}/{c}_df.csv', index=True)

    print(f'Saved {c} Data')


# =============================================================================
# SHARED TABULAR PIPELINE HELPERS  (GBM, SVM)
# =============================================================================

def _prepare_tabular_dataset(
    coin: str, response: str = RESPONSE, training_cols_path: str = TRAINING_COLUMNS
):
    """Load data and return (daily_data, X, y, cols) for tabular models."""
    raw_path = fullDataPath(coin)
    data = pd.read_csv(raw_path)
    daily_data = dataSetup(data, trainingColPath=training_cols_path, response=response)
    cols = trainingCols(training_cols_path)
    missing = [c for c in cols if c not in daily_data.columns]
    assert len(missing) == 0, f"Missing training columns: {missing}"
    X = daily_data[cols].copy()
    y = daily_data[response].copy()
    return daily_data, X, y, cols


def _tabular_future_forecast(model, scaler, X_full, daily_data, cols, response=RESPONSE, n=TEST_DAYS):
    """Iterative future forecast for tabular models (GBM, SVM, KNN-like)."""
    last_features = X_full.iloc[-1].copy()
    predictions = []
    for _ in range(n):
        scaled = scaler.transform(pd.DataFrame([last_features], columns=cols))
        pred = float(model.predict(scaled)[0])
        predictions.append(pred)
        for pc in ['close', 'open', 'high', 'low']:
            if pc in last_features.index:
                last_features[pc] = pred
    future_index = generate_future_index(daily_data.index[-1], n)
    return pd.DataFrame(predictions, index=future_index, columns=['predicted_price'])


def _save_model_artifact(obj, coin: str, filename: str):
    """Pickle-save a model artifact to models/{coin}/."""
    model_dir = os.path.join(base_dir('models'), coin)
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, filename)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return path

def _save_metrics(value: float, coin: str, model_name: str):
    """Save standardized RMSE to both predictions/ and metrics/ folders."""
    _save_standardized_rmse(value, coin, model_name)
    metrics_dir = os.path.join(base_dir('metrics'), coin)
    os.makedirs(metrics_dir, exist_ok=True)
    path = f'{metrics_dir}/{model_name}_rmse.txt'
    with open(path, 'w') as f:
        f.write(f"{value:.6f}\n")
    return path


# =============================================================================
# GBM PIPELINE HELPERS
# =============================================================================

def _gbm_param_grid() -> list:
    return list(ParameterGrid({
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
    }))


def _gbm_tune(X_train, y_train, param_grid: list, scaler_methods=None, n_splits: int = 3):
    """Grid search over GBM hyperparameters + scaler methods using time-series CV."""
    from implementations.gbm_model import GBMModel
    if scaler_methods is None:
        scaler_methods = ['standard', 'minmax', 'robust']
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(X_train) // 4)))
    best_score, best_combo, results = np.inf, None, []
    for sm in scaler_methods:
        for params in param_grid:
            fold_rmses = []
            for tr_idx, va_idx in tscv.split(X_train):
                X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
                sc = _scaler_factory(sm)
                X_tr_s = sc.fit_transform(X_tr)
                X_va_s = sc.transform(X_va)
                m = GBMModel(**params)
                m.fit(X_tr_s, y_tr.values)
                fold_rmses.append(mean_squared_error(y_va, m.predict(X_va_s), squared=False))
            if not fold_rmses:
                continue
            avg = float(np.mean(fold_rmses))
            rec = {**params, 'scaler': sm, 'rmse': avg}
            results.append(rec)
            if avg < best_score:
                best_score = avg
                best_combo = rec.copy()
    return best_combo, results


def run_gbm_pipeline(
    coin: str = COIN, response: str = RESPONSE, training_cols_path: str = TRAINING_COLUMNS
) -> dict:
    """End-to-end GBM pipeline: load → split → tune → train → validate → forecast → save."""
    from implementations.gbm_model import GBMModel
    model_tag = 'gbm'
    daily_data, X, y, cols = _prepare_tabular_dataset(coin, response, training_cols_path)
    X_train, X_val, y_train, y_val = _time_series_train_val_split(X, y)
    best_combo, tuning_results = _gbm_tune(X_train, y_train, _gbm_param_grid())
    if best_combo is None:
        best_combo = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3,
                      'subsample': 1.0, 'scaler': 'standard', 'rmse': np.nan}
    scaler = _scaler_factory(best_combo['scaler'])
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=cols)
    X_val_s = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=cols)
    model_params = {k: v for k, v in best_combo.items() if k not in ('scaler', 'rmse')}
    model = GBMModel(**model_params)
    model.fit(X_train_s, y_train.values)
    val_preds = model.predict(X_val_s)
    val_df = pd.DataFrame({'predicted_price': val_preds}, index=X_val_s.index)
    rmse, std_rmse = _standardized_rmse(y_val, val_preds)
    _save_validation_predictions(val_df, coin, model_tag)
    _save_metrics(std_rmse, coin, model_tag)
    _save_model_artifact({'model': model, 'scaler': scaler}, coin, f'{coin}_gbm_model.pkl')
    X_full_s = pd.DataFrame(scaler.transform(X), index=X.index, columns=cols)
    future_df = _tabular_future_forecast(model, scaler, X, daily_data, cols, response)
    _save_future_predictions(future_df, coin, model_tag)
    return {
        'best_params': model_params, 'scaler': best_combo['scaler'],
        'rmse': rmse, 'std_rmse': std_rmse,
        'val_preds': val_df, 'future_preds': future_df,
        'tuning_results': pd.DataFrame(tuning_results),
    }


# =============================================================================
# ARIMA PIPELINE HELPERS
# =============================================================================

def _arima_grid_search(train_series: pd.Series, p_range=(0, 1, 2), d_range=(0, 1), q_range=(0, 1, 2)):
    """Find best (p, d, q) order by minimum AIC on the training series."""
    from implementations.arima_model import ARIMAModel
    best_aic, best_order = np.inf, (1, 1, 1)
    results = []
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    m = ARIMAModel(p, d, q)
                    m.fit(train_series)
                    rec = {'p': p, 'd': d, 'q': q, 'aic': m.aic, 'bic': m.bic}
                    results.append(rec)
                    if m.aic < best_aic:
                        best_aic = m.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    return best_order, pd.DataFrame(results).sort_values('aic') if results else pd.DataFrame()


def run_arima_pipeline(
    coin: str = COIN, response: str = RESPONSE, training_cols_path: str = TRAINING_COLUMNS
) -> dict:
    """End-to-end ARIMA pipeline (univariate on response/price series)."""
    from implementations.arima_model import ARIMAModel
    model_tag = 'arima'
    raw_path = fullDataPath(coin)
    data = pd.read_csv(raw_path)
    daily_data = dataSetup(data, trainingColPath=training_cols_path, response=response)
    price_series = daily_data[response].copy()
    n = len(price_series)
    split_idx = max(1, int(n * TRAIN_PCT))
    train_series = price_series.iloc[:split_idx]
    val_series = price_series.iloc[split_idx:]
    best_order, tuning_df = _arima_grid_search(train_series)
    # Retrain on full history for validation + forecast
    model = ARIMAModel(*best_order)
    model.fit(train_series)
    # In-sample val: rolling one-step predictions
    val_preds_list = []
    history = train_series.tolist()
    for actual in val_series:
        from statsmodels.tsa.arima.model import ARIMA as _ARIMA
        tmp = _ARIMA(history, order=best_order).fit()
        val_preds_list.append(float(tmp.forecast(steps=1)[0]))
        history.append(actual)
    val_preds = np.array(val_preds_list)
    val_df = pd.DataFrame({'predicted_price': val_preds}, index=val_series.index)
    rmse, std_rmse = _standardized_rmse(val_series, val_preds)
    _save_validation_predictions(val_df, coin, model_tag)
    _save_metrics(std_rmse, coin, model_tag)
    # Retrain on ALL data then forecast future
    final_model = ARIMAModel(*best_order)
    final_model.fit(price_series)
    forecast_vals = final_model.forecast(steps=TEST_DAYS)
    future_index = generate_future_index(daily_data.index[-1], TEST_DAYS)
    future_df = pd.DataFrame({'predicted_price': forecast_vals}, index=future_index)
    _save_future_predictions(future_df, coin, model_tag)
    _save_model_artifact(final_model, coin, f'{coin}_arima_model.pkl')
    return {
        'best_order': best_order, 'rmse': rmse, 'std_rmse': std_rmse,
        'val_preds': val_df, 'future_preds': future_df, 'tuning_df': tuning_df,
    }


# =============================================================================
# SHARED PYTORCH TRAINING UTILITY
# =============================================================================

def _get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _torch_train(model, X_train_t, y_train_t, X_val_t, y_val_t,
                  lr: float = 1e-3, epochs: int = 50, batch_size: int = 32,
                  patience: int = 10, device: str = 'cpu'):
    """Generic PyTorch training loop with early stopping. Returns trained model and best val loss."""
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    model = model.to(device)
    X_train_t, y_train_t = X_train_t.to(device), y_train_t.to(device)
    X_val_t, y_val_t = X_val_t.to(device), y_val_t.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=False)
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            preds = model(Xb).squeeze(-1)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t).squeeze(-1), y_val_t).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model = model.to('cpu')
    return model, best_val_loss


def _seq_future_forecast_torch(model, scaled_data: np.ndarray, seq_len: int,
                                target_scaler, daily_data: pd.DataFrame,
                                n: int = TEST_DAYS) -> pd.DataFrame:
    """Iterative future forecast for sequence-based PyTorch models."""
    import torch
    model.eval()
    last_seq = scaled_data[-seq_len:].copy()    # (seq_len, n_features)
    predictions = []
    for _ in range(n):
        x = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)  # (1, seq, feats)
        with torch.no_grad():
            pred_scaled = float(model(x).squeeze().item())
        pred = float(target_scaler.inverse_transform([[pred_scaled]])[0][0])
        predictions.append(pred)
        new_row = last_seq[-1].copy()
        new_row[-1] = pred_scaled          # update target (last column = close)
        last_seq = np.vstack([last_seq[1:], new_row])
    future_index = generate_future_index(daily_data.index[-1], n)
    return pd.DataFrame(predictions, index=future_index, columns=['predicted_price'])


# =============================================================================
# LSTM PIPELINE HELPERS
# =============================================================================

def _lstm_param_grid() -> list:
    return list(ParameterGrid({
        'hidden_size': [32, 64],
        'num_layers': [1, 2],
        'dropout': [0.1, 0.2],
        'seq_len': [14, 30],
    }))


def run_lstm_pipeline(
    coin: str = COIN, response: str = RESPONSE, training_cols_path: str = TRAINING_COLUMNS,
    lr: float = 1e-3, epochs: int = 50, batch_size: int = 32, patience: int = 10,
    scaler_methods: list = None,
) -> dict:
    """End-to-end LSTM pipeline."""
    import torch
    from implementations.lstm_model import LSTMModel
    if scaler_methods is None:
        scaler_methods = ['minmax', 'standard']
    model_tag = 'lstm'
    device = _get_device()
    daily_data, X_df, y_s, cols = _prepare_tabular_dataset(coin, response, training_cols_path)
    data_full = daily_data[cols + [response]].copy()
    param_grid = _lstm_param_grid()
    best_score, best_combo = np.inf, None
    results = []
    for sm in scaler_methods:
        scaled_data, feat_sc, tgt_sc = normalize_data(data_full, method=sm, target_col=response)
        for params in param_grid:
            seq_len = params['seq_len']
            X_seq, y_seq = create_sequences(scaled_data, sequence_length=seq_len, prediction_horizon=1)
            if len(X_seq) < 10:
                continue
            n_train = max(1, int(len(X_seq) * TRAIN_PCT))
            X_tr = torch.tensor(X_seq[:n_train], dtype=torch.float32)
            y_tr = torch.tensor(y_seq[:n_train].flatten(), dtype=torch.float32)
            X_va = torch.tensor(X_seq[n_train:], dtype=torch.float32)
            y_va = torch.tensor(y_seq[n_train:].flatten(), dtype=torch.float32)
            hp = {k: v for k, v in params.items() if k != 'seq_len'}
            m = LSTMModel(input_size=X_tr.shape[2], **hp)
            m, val_loss = _torch_train(m, X_tr, y_tr, X_va, y_va,
                                       lr=lr, epochs=epochs, batch_size=batch_size,
                                       patience=patience, device=device)
            rec = {**params, 'scaler': sm, 'val_loss': val_loss}
            results.append(rec)
            if val_loss < best_score:
                best_score = val_loss
                best_combo = rec.copy()
    if best_combo is None:
        best_combo = {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1, 'seq_len': 30, 'scaler': 'minmax'}
    # Retrain with best config on full data
    sm = best_combo['scaler']
    seq_len = best_combo['seq_len']
    scaled_data, feat_sc, tgt_sc = normalize_data(data_full, method=sm, target_col=response)
    X_seq, y_seq = create_sequences(scaled_data, sequence_length=seq_len, prediction_horizon=1)
    n_train = max(1, int(len(X_seq) * TRAIN_PCT))
    X_tr = torch.tensor(X_seq[:n_train], dtype=torch.float32)
    y_tr = torch.tensor(y_seq[:n_train].flatten(), dtype=torch.float32)
    X_va = torch.tensor(X_seq[n_train:], dtype=torch.float32)
    y_va_np = y_seq[n_train:].flatten()
    y_va_t = torch.tensor(y_va_np, dtype=torch.float32)
    hp = {k: v for k, v in best_combo.items() if k not in ('seq_len', 'scaler', 'val_loss')}
    best_model = LSTMModel(input_size=X_tr.shape[2], **hp)
    best_model, _ = _torch_train(best_model, X_tr, y_tr, X_va, y_va_t,
                                  lr=lr, epochs=epochs, batch_size=batch_size,
                                  patience=patience, device=device)
    best_model.eval()
    # Validation predictions (inverse transform)
    with torch.no_grad():
        val_preds_scaled = best_model(X_va).squeeze(-1).numpy()
    val_preds = tgt_sc.inverse_transform(val_preds_scaled.reshape(-1, 1)).flatten()
    val_true = tgt_sc.inverse_transform(y_va_np.reshape(-1, 1)).flatten()
    val_idx = daily_data.index[n_train + seq_len: n_train + seq_len + len(val_preds)]
    val_df = pd.DataFrame({'predicted_price': val_preds}, index=val_idx[:len(val_preds)])
    rmse, std_rmse = float(np.sqrt(np.mean((val_true - val_preds) ** 2))), 0.0
    if val_true.std() > 0:
        std_rmse = rmse / val_true.std()
    _save_validation_predictions(val_df, coin, model_tag)
    _save_metrics(std_rmse, coin, model_tag)
    torch.save(best_model.state_dict(), f'models/{coin}/{coin}_lstm_model.pt')
    _save_model_artifact({'feat_scaler': feat_sc, 'tgt_scaler': tgt_sc, 'params': best_combo}, coin, f'{coin}_lstm_meta.pkl')
    future_df = _seq_future_forecast_torch(best_model, scaled_data, seq_len, tgt_sc, daily_data)
    _save_future_predictions(future_df, coin, model_tag)
    return {
        'best_params': best_combo, 'rmse': rmse, 'std_rmse': std_rmse,
        'val_preds': val_df, 'future_preds': future_df, 'tuning_results': results,
    }


# =============================================================================
# SVM PIPELINE HELPERS
# =============================================================================

def _svm_param_grid() -> list:
    return list(ParameterGrid({
        'kernel': ['rbf', 'linear'],
        'C': [0.1, 1.0, 10.0],
        'epsilon': [0.01, 0.1],
        'gamma': ['scale'],
    }))


def _svm_tune(X_train, y_train, param_grid: list, scaler_methods=None, n_splits: int = 3):
    """Grid search over SVM hyperparameters + scaler methods with time-series CV."""
    from implementations.svm_model import SVMModel
    if scaler_methods is None:
        scaler_methods = ['standard', 'minmax']
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(X_train) // 4)))
    best_score, best_combo, results = np.inf, None, []
    for sm in scaler_methods:
        for params in param_grid:
            fold_rmses = []
            for tr_idx, va_idx in tscv.split(X_train):
                X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
                sc = _scaler_factory(sm)
                X_tr_s = sc.fit_transform(X_tr)
                X_va_s = sc.transform(X_va)
                m = SVMModel(**params)
                m.fit(X_tr_s, y_tr.values)
                fold_rmses.append(mean_squared_error(y_va, m.predict(X_va_s), squared=False))
            if not fold_rmses:
                continue
            avg = float(np.mean(fold_rmses))
            rec = {**params, 'scaler': sm, 'rmse': avg}
            results.append(rec)
            if avg < best_score:
                best_score = avg
                best_combo = rec.copy()
    return best_combo, results


def run_svm_pipeline(
    coin: str = COIN, response: str = RESPONSE, training_cols_path: str = TRAINING_COLUMNS
) -> dict:
    """End-to-end SVM pipeline: load → split → tune → train → validate → forecast → save."""
    from implementations.svm_model import SVMModel
    model_tag = 'svm'
    daily_data, X, y, cols = _prepare_tabular_dataset(coin, response, training_cols_path)
    X_train, X_val, y_train, y_val = _time_series_train_val_split(X, y)
    best_combo, tuning_results = _svm_tune(X_train, y_train, _svm_param_grid())
    if best_combo is None:
        best_combo = {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1, 'gamma': 'scale',
                      'scaler': 'standard', 'rmse': np.nan}
    scaler = _scaler_factory(best_combo['scaler'])
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=cols)
    X_val_s = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=cols)
    model_params = {k: v for k, v in best_combo.items() if k not in ('scaler', 'rmse')}
    model = SVMModel(**model_params)
    model.fit(X_train_s.values, y_train.values)
    val_preds = model.predict(X_val_s.values)
    val_df = pd.DataFrame({'predicted_price': val_preds}, index=X_val_s.index)
    rmse, std_rmse = _standardized_rmse(y_val, val_preds)
    _save_validation_predictions(val_df, coin, model_tag)
    _save_metrics(std_rmse, coin, model_tag)
    _save_model_artifact({'model': model, 'scaler': scaler}, coin, f'{coin}_svm_model.pkl')
    future_df = _tabular_future_forecast(model, scaler, X, daily_data, cols, response)
    _save_future_predictions(future_df, coin, model_tag)
    return {
        'best_params': model_params, 'scaler': best_combo['scaler'],
        'rmse': rmse, 'std_rmse': std_rmse,
        'val_preds': val_df, 'future_preds': future_df,
        'tuning_results': pd.DataFrame(tuning_results),
    }


# =============================================================================
# TFT PIPELINE HELPERS
# =============================================================================

def _tft_param_grid() -> list:
    return list(ParameterGrid({
        'hidden_size': [32, 64],
        'num_heads': [2, 4],
        'num_lstm_layers': [1, 2],
        'dropout': [0.1],
    }))


def run_tft_pipeline(
    coin: str = COIN, response: str = RESPONSE, training_cols_path: str = TRAINING_COLUMNS,
    lr: float = 1e-3, epochs: int = 50, batch_size: int = 32, patience: int = 10,
    seq_len: int = 30, scaler_method: str = 'minmax',
) -> dict:
    """End-to-end TFT pipeline."""
    import torch
    from implementations.tft_model import TemporalFusionTransformer
    model_tag = 'tft'
    device = _get_device()
    daily_data, X_df, y_s, cols = _prepare_tabular_dataset(coin, response, training_cols_path)
    data_full = daily_data[cols + [response]].copy()
    scaled_data, feat_sc, tgt_sc = normalize_data(data_full, method=scaler_method, target_col=response)
    X_seq, y_seq = create_sequences(scaled_data, sequence_length=seq_len, prediction_horizon=1)
    n_train = max(1, int(len(X_seq) * TRAIN_PCT))
    X_tr = torch.tensor(X_seq[:n_train], dtype=torch.float32)
    y_tr = torch.tensor(y_seq[:n_train].flatten(), dtype=torch.float32)
    X_va = torch.tensor(X_seq[n_train:], dtype=torch.float32)
    y_va_np = y_seq[n_train:].flatten()
    y_va_t = torch.tensor(y_va_np, dtype=torch.float32)
    # Tune hyperparameters
    best_score, best_combo = np.inf, None
    tuning_results = []
    num_features = X_tr.shape[2]
    for params in _tft_param_grid():
        m = TemporalFusionTransformer(num_features=num_features, **params)
        m, val_loss = _torch_train(m, X_tr, y_tr, X_va, y_va_t,
                                   lr=lr, epochs=epochs, batch_size=batch_size,
                                   patience=patience, device=device)
        rec = {**params, 'val_loss': val_loss}
        tuning_results.append(rec)
        if val_loss < best_score:
            best_score = val_loss
            best_combo = rec.copy()
    if best_combo is None:
        best_combo = {'hidden_size': 64, 'num_heads': 4, 'num_lstm_layers': 2, 'dropout': 0.1}
    # Final model with best params
    best_hp = {k: v for k, v in best_combo.items() if k != 'val_loss'}
    best_model = TemporalFusionTransformer(num_features=num_features, **best_hp)
    best_model, _ = _torch_train(best_model, X_tr, y_tr, X_va, y_va_t,
                                  lr=lr, epochs=epochs, batch_size=batch_size,
                                  patience=patience, device=device)
    best_model.eval()
    with torch.no_grad():
        val_preds_scaled = best_model(X_va).squeeze(-1).numpy()
    val_preds = tgt_sc.inverse_transform(val_preds_scaled.reshape(-1, 1)).flatten()
    val_true = tgt_sc.inverse_transform(y_va_np.reshape(-1, 1)).flatten()
    val_idx = daily_data.index[n_train + seq_len: n_train + seq_len + len(val_preds)]
    val_df = pd.DataFrame({'predicted_price': val_preds}, index=val_idx[:len(val_preds)])
    rmse = float(np.sqrt(np.mean((val_true - val_preds) ** 2)))
    std_rmse = rmse / val_true.std() if val_true.std() > 0 else rmse
    _save_validation_predictions(val_df, coin, model_tag)
    _save_metrics(std_rmse, coin, model_tag)
    os.makedirs(f'models/{coin}', exist_ok=True)
    torch.save(best_model.state_dict(), f'models/{coin}/{coin}_tft_model.pt')
    _save_model_artifact({'feat_scaler': feat_sc, 'tgt_scaler': tgt_sc, 'params': best_combo,
                          'seq_len': seq_len, 'scaler_method': scaler_method},
                         coin, f'{coin}_tft_meta.pkl')
    future_df = _seq_future_forecast_torch(best_model, scaled_data, seq_len, tgt_sc, daily_data)
    _save_future_predictions(future_df, coin, model_tag)
    return {
        'best_params': best_hp, 'rmse': rmse, 'std_rmse': std_rmse,
        'val_preds': val_df, 'future_preds': future_df, 'tuning_results': tuning_results,
    }


# =============================================================================
# TRANSFORMER PIPELINE HELPERS
# =============================================================================

def _transformer_param_grid() -> list:
    return list(ParameterGrid({
        'd_model': [32, 64],
        'nhead': [2, 4],
        'num_encoder_layers': [1, 2],
        'dim_feedforward': [128, 256],
        'dropout': [0.1],
    }))


def run_transformer_pipeline(
    coin: str = COIN, response: str = RESPONSE, training_cols_path: str = TRAINING_COLUMNS,
    lr: float = 1e-3, epochs: int = 50, batch_size: int = 32, patience: int = 10,
    seq_len: int = 30, scaler_method: str = 'minmax',
) -> dict:
    """End-to-end simple Transformer pipeline."""
    import torch
    from implementations.transformer_model import CryptoTransformer
    model_tag = 'transformer'
    device = _get_device()
    daily_data, X_df, y_s, cols = _prepare_tabular_dataset(coin, response, training_cols_path)
    data_full = daily_data[cols + [response]].copy()
    scaled_data, feat_sc, tgt_sc = normalize_data(data_full, method=scaler_method, target_col=response)
    X_seq, y_seq = create_sequences(scaled_data, sequence_length=seq_len, prediction_horizon=1)
    n_train = max(1, int(len(X_seq) * TRAIN_PCT))
    X_tr = torch.tensor(X_seq[:n_train], dtype=torch.float32)
    y_tr = torch.tensor(y_seq[:n_train].flatten(), dtype=torch.float32)
    X_va = torch.tensor(X_seq[n_train:], dtype=torch.float32)
    y_va_np = y_seq[n_train:].flatten()
    y_va_t = torch.tensor(y_va_np, dtype=torch.float32)
    best_score, best_combo = np.inf, None
    tuning_results = []
    num_features = X_tr.shape[2]
    for params in _transformer_param_grid():
        # nhead must divide d_model
        if params['d_model'] % params['nhead'] != 0:
            continue
        m = CryptoTransformer(num_features=num_features, **params)
        m, val_loss = _torch_train(m, X_tr, y_tr, X_va, y_va_t,
                                   lr=lr, epochs=epochs, batch_size=batch_size,
                                   patience=patience, device=device)
        rec = {**params, 'val_loss': val_loss}
        tuning_results.append(rec)
        if val_loss < best_score:
            best_score = val_loss
            best_combo = rec.copy()
    if best_combo is None:
        best_combo = {'d_model': 64, 'nhead': 4, 'num_encoder_layers': 2,
                      'dim_feedforward': 256, 'dropout': 0.1}
    best_hp = {k: v for k, v in best_combo.items() if k != 'val_loss'}
    best_model = CryptoTransformer(num_features=num_features, **best_hp)
    best_model, _ = _torch_train(best_model, X_tr, y_tr, X_va, y_va_t,
                                  lr=lr, epochs=epochs, batch_size=batch_size,
                                  patience=patience, device=device)
    best_model.eval()
    with torch.no_grad():
        val_preds_scaled = best_model(X_va).squeeze(-1).numpy()
    val_preds = tgt_sc.inverse_transform(val_preds_scaled.reshape(-1, 1)).flatten()
    val_true = tgt_sc.inverse_transform(y_va_np.reshape(-1, 1)).flatten()
    val_idx = daily_data.index[n_train + seq_len: n_train + seq_len + len(val_preds)]
    val_df = pd.DataFrame({'predicted_price': val_preds}, index=val_idx[:len(val_preds)])
    rmse = float(np.sqrt(np.mean((val_true - val_preds) ** 2)))
    std_rmse = rmse / val_true.std() if val_true.std() > 0 else rmse
    _save_validation_predictions(val_df, coin, model_tag)
    _save_metrics(std_rmse, coin, model_tag)
    os.makedirs(f'models/{coin}', exist_ok=True)
    torch.save(best_model.state_dict(), f'models/{coin}/{coin}_transformer_model.pt')
    _save_model_artifact({'feat_scaler': feat_sc, 'tgt_scaler': tgt_sc, 'params': best_combo,
                          'seq_len': seq_len, 'scaler_method': scaler_method},
                         coin, f'{coin}_transformer_meta.pkl')
    future_df = _seq_future_forecast_torch(best_model, scaled_data, seq_len, tgt_sc, daily_data)
    _save_future_predictions(future_df, coin, model_tag)
    return {
        'best_params': best_hp, 'rmse': rmse, 'std_rmse': std_rmse,
        'val_preds': val_df, 'future_preds': future_df, 'tuning_results': tuning_results,
    }

# =============================================================================
# PREDICTION MATRIX — registry-based design
#
# Each loader has signature:
#   (coin, models_dir, data_full, X, cols, daily_data) -> (np.ndarray, DatetimeIndex)
#
# To add a new model:
#   1. Write a loader function or use one of the factories below.
#   2. Append (display_name, loader) to _MODEL_REGISTRY.
# =============================================================================

def _pm_tabular(tag: str):
    """Factory for tabular models whose artifact is {'model': ..., 'scaler': ...}.

    Covers GBM, SVM, and any future sklearn-style model saved the same way.
    Artifact file expected: models/{coin}/{coin}_{tag}_model.pkl
    """
    def _load(coin, models_dir, data_full, X, cols, daily_data):
        artifact = pickle.load(open(os.path.join(models_dir, f'{coin}_{tag}_model.pkl'), 'rb'))
        df = _tabular_future_forecast(
            artifact['model'], artifact['scaler'],
            X, daily_data, cols, RESPONSE_VARIABLE, n=TEST_DAYS,
        )
        return df['predicted_price'].values, df.index
    return _load


def _pm_knn(coin, models_dir, data_full, X, cols, daily_data):
    """Loader for KNN.

    _save_knn_model uses string concat instead of os.path.join, producing a
    doubled coin prefix in the filename (e.g. BTCBTC_knn_model.pkl) inside
    models/{coin}/.
    """
    knn_model  = pickle.load(open(os.path.join(models_dir, f'{coin}{coin}_knn_model.pkl'), 'rb'))
    knn_scaler = pickle.load(open(os.path.join(models_dir, f'{coin}{coin}_scaler.pkl'), 'rb'))
    df = _knn_future_forecast(
        knn_model, knn_scaler, X, daily_data, cols,
        response=RESPONSE_VARIABLE, n=TEST_DAYS,
    )
    return df['predicted_price'].values, df.index


def _pm_arima(coin, models_dir, data_full, X, cols, daily_data):
    """Loader for ARIMA (ARIMAModel pkl, forecasts via .forecast())."""
    arima = pickle.load(open(os.path.join(models_dir, f'{coin}_arima_model.pkl'), 'rb'))
    vals  = np.asarray(arima.forecast(steps=TEST_DAYS))
    idx   = generate_future_index(daily_data.index[-1], TEST_DAYS)
    return vals, idx


def _pm_torch_seq(tag: str, impl_module: str, impl_class: str,
                  size_kwarg: str = 'num_features'):
    """Factory for PyTorch sequence models (.pt state dict + _meta.pkl).

    Parameters
    ----------
    tag         : file-name tag, e.g. 'lstm', 'tft', 'transformer'
    impl_module : dotted import path, e.g. 'implementations.lstm_model'
    impl_class  : class name, e.g. 'LSTMModel'
    size_kwarg  : constructor kwarg for input width
                  ('input_size' for LSTM, 'num_features' for TFT/Transformer)

    Artifact files expected:
      models/{coin}/{coin}_{tag}_model.pt
      models/{coin}/{coin}_{tag}_meta.pkl
    """
    def _load(coin, models_dir, data_full, X, cols, daily_data):
        import importlib
        model_cls = getattr(importlib.import_module(impl_module), impl_class)

        meta     = pickle.load(open(os.path.join(models_dir, f'{coin}_{tag}_meta.pkl'), 'rb'))
        scaled, _, _ = normalize_data(
            data_full, method=meta['scaler_method'], target_col=RESPONSE_VARIABLE
        )
        hp       = {k: v for k, v in meta['params'].items()
                    if k not in ('seq_len', 'scaler', 'val_loss')}
        model    = model_cls(**{size_kwarg: data_full.shape[1]}, **hp)
        model.load_state_dict(
            torch.load(os.path.join(models_dir, f'{coin}_{tag}_model.pt'), map_location='cpu')
        )
        model.eval()
        df = _seq_future_forecast_torch(
            model, scaled, meta['seq_len'], meta['tgt_scaler'], daily_data, n=TEST_DAYS
        )
        return df['predicted_price'].values, df.index
    return _load


def _pm_prophet(coin, models_dir, data_full, X, cols, daily_data):
    """Loader for Prophet (ProphetWrapper pkl)."""
    wrapper = pickle.load(open(os.path.join(models_dir, f'{coin}_prophet_model.pkl'), 'rb'))
    df = _prophet_future_forecast(wrapper, daily_data, cols, RESPONSE_VARIABLE, n=TEST_DAYS)
    return df['predicted_price'].values, df.index


# ── Registry ──────────────────────────────────────────────────────────────────
# Each entry: (display_name, loader_fn)
# Adding a new model = one new line here; no other changes needed.
_MODEL_REGISTRY: list = [
    ('GBM',         _pm_tabular('gbm')),
    ('SVM',         _pm_tabular('svm')),
    ('KNN',         _pm_knn),
    ('ARIMA',       _pm_arima),
    ('LSTM',        _pm_torch_seq('lstm',
                                  'implementations.lstm_model', 'LSTMModel',
                                  size_kwarg='input_size')),
    ('TFT',         _pm_torch_seq('tft',
                                  'implementations.tft_model', 'TemporalFusionTransformer')),
    ('Transformer', _pm_torch_seq('transformer',
                                  'implementations.transformer_model', 'CryptoTransformer')),
    ('Prophet',     _pm_prophet),
]


def predict_matrix(coin: str) -> pd.DataFrame:
    """Load every registered model for *coin* and return an m×n DataFrame.

    Iterates over ``_MODEL_REGISTRY``; models whose artifact files are missing
    are skipped with a warning so the function stays usable when only a subset
    of models has been trained for a given coin.

    Parameters
    ----------
    coin : str
        Coin symbol matching the saved model directory, e.g. ``'BTC'``.

    Returns
    -------
    pd.DataFrame
        Shape (m, TEST_DAYS) — model names as index, date strings as columns,
        predicted USD prices as values.
    """
    models_dir = os.path.join(base_dir(), 'models', coin)
    data       = pd.read_csv(fullDataPath(coin))
    daily_data = dataSetup(data, trainingColPath=TRAINING_COLUMNS,
                           response=RESPONSE_VARIABLE, number=LIMIT)
    cols       = trainingCols(TRAINING_COLUMNS)
    X          = daily_data[cols].copy()
    data_full  = daily_data[cols + [RESPONSE_VARIABLE]].copy()

    preds        = {}
    future_index = None

    for name, loader in _MODEL_REGISTRY:
        try:
            vals, idx = loader(coin, models_dir, data_full, X, cols, daily_data)
            preds[name] = vals
            if future_index is None:
                future_index = idx
        except Exception as e:
            print(f'  [predict_matrix] {name} skipped — {e}')

    matrix = pd.DataFrame(preds, index=future_index).T
    matrix.columns = [str(d.date()) for d in matrix.columns]
    matrix.index.name = 'Model'
    return matrix


def base_dir(folders = [], create=False):
    things = os.getcwd().split('/')
    ct2Index = things.index(REPO)
    if isinstance(folders, str):
        folders = [folders]
    final = things[:(ct2Index+1)] + folders
    final = '/'.join(final)

    if create and not os.path.exists(final):
        os.makedirs(final, exist_ok=True)
        return final
    elif not create and os.path.exists(final):
        return final
    elif create and os.path.exists(final):
        return final
    else:
        raise ValueError('Folder already exists')


def train_all_models(coin: str) -> dict:
    """Train all 8 models for *coin* end-to-end and return their standardised RMSEs.

    Calls the individual pipeline functions in sequence.  If one model fails it is
    skipped and the rest continue.  Artifacts are saved to ``models/{coin}/`` and
    metrics to ``metrics/{coin}/``.

    Parameters
    ----------
    coin : str
        Coin symbol, e.g. ``'BTC'``, ``'ETH'``.

    Returns
    -------
    dict
        ``{model_name: std_rmse}`` for each model.  Value is ``None`` when a model
        failed.
    """
    pipelines = [
        ('GBM',         lambda: run_gbm_pipeline(coin)),
        ('SVM',         lambda: run_svm_pipeline(coin)),
        ('KNN',         lambda: run_knn_pipeline(coin)),
        ('ARIMA',       lambda: run_arima_pipeline(coin)),
        ('LSTM',        lambda: run_lstm_pipeline(coin)),
        ('TFT',         lambda: run_tft_pipeline(coin)),
        ('Transformer', lambda: run_transformer_pipeline(coin)),
        ('Prophet',     lambda: run_prophet_pipeline(coin)),
    ]

    results = {}
    width = 60
    print(f"\n{'=' * width}")
    print(f"  Training all models for {coin}")
    print(f"{'=' * width}\n")

    for name, run in pipelines:
        print(f"[{name}] starting...")
        try:
            out = run()
            std_rmse = out.get('std_rmse', float('nan'))
            results[name] = std_rmse
            print(f"[{name}] done  —  std_rmse = {std_rmse:.4f}\n")
        except Exception as exc:
            print(f"[{name}] FAILED: {exc}\n")
            results[name] = None

    print(f"\n{'=' * width}")
    print(f"  Results for {coin}")
    print(f"{'=' * width}")
    for name, val in sorted(results.items(),
                            key=lambda kv: (kv[1] is None, kv[1] or 0)):
        label = f"{val:.4f}" if val is not None else "FAILED"
        print(f"  {name:<15}  {label}")

    return results