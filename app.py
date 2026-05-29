import os
import re
import sys
import json
import select
import time
import subprocess
from itertools import product as _iproduct, combinations as _icombinations
from pathlib import Path
from files.CONSTANTS import *
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from files.CONSTANTS import COINS, RESPONSE_VARIABLE, TRAINING_COLUMNS, LIMIT
from files.functions import fullDataPath, dataSetup, base_dir

# ── Resolve project root (works locally and in Colab / cloud deploy) ──────────
_root = next(
    (p for p in [Path(__file__).parent, *Path(__file__).parent.parents]
     if p.name == REPO),
    Path(__file__).parent,
)
os.chdir(str(_root))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

def _read_proc_output(proc: subprocess.Popen) -> str:
    """Drain all lines currently available on proc.stdout without blocking."""
    if proc.stdout is None:
        return ''
    lines = []
    while True:
        ready, _, _ = select.select([proc.stdout], [], [], 0)
        if not ready:
            break
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line)
    return ''.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CryptoTrading2 — Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_predictions(coin: str) -> dict:
    """Load all available model future-prediction CSVs for *coin*.

    Files live at  predictions/{coin}{model}_future_predictions.csv
    Returns  {MODEL_TAG: pd.Series(date_str -> predicted_price)}
    """
    pred_base = base_dir('predictions')
    preds = {}
    for fname in os.listdir(pred_base):
        if fname.startswith(coin) and fname.endswith('_future_predictions.csv'):
            tag = fname[len(coin):].replace('_future_predictions.csv', '').upper()
            fpath = os.path.join(pred_base, fname)
            try:
                df = pd.read_csv(fpath, index_col=0)
                if 'predicted_price' in df.columns:
                    preds[tag] = df['predicted_price'].astype(float)
            except Exception:
                pass
    return preds


def _load_rmse(coin: str) -> dict:
    """Return {MODEL_TAG: rmse_float} from metrics/{coin}/."""
    metrics_dir = os.path.join(base_dir('metrics'), coin)
    rmses = {}
    if os.path.exists(metrics_dir):
        for fname in os.listdir(metrics_dir):
            if fname.endswith('_rmse.txt'):
                tag = fname.replace('_rmse.txt', '').upper()
                try:
                    with open(os.path.join(metrics_dir, fname)) as fh:
                        rmses[tag] = float(fh.read().strip())
                except Exception:
                    pass
    return rmses


def _current_price(coin: str) -> float | None:
    """Return the most recent daily close price directly from the CSV."""
    try:
        raw = pd.read_csv(fullDataPath(coin))
        raw['time'] = pd.to_datetime(raw['time'], errors='coerce')
        raw = raw.dropna(subset=['time'])
        # Group by date and take the last close for each day (handles intraday rows)
        daily = raw.groupby(raw['time'].dt.date)[RESPONSE_VARIABLE].last()
        return float(daily.iloc[-1])
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _available_coins() -> list[str]:
    """All coins that have a data CSV AND at least one future-prediction file."""
    data_dir  = os.path.join(str(_root), 'data')
    pred_base = base_dir('predictions') or os.path.join(str(_root), 'predictions')
    data_coins = set()
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith('_df.csv'):
                data_coins.add(f.replace('_df.csv', ''))
    pred_coins = set()
    if os.path.exists(pred_base):
        for f in os.listdir(pred_base):
            if '_future_predictions.csv' in f:
                for c in data_coins:
                    if f.startswith(c):
                        pred_coins.add(c)
                        break
    return sorted(pred_coins)


@st.cache_data(show_spinner=False)
def _build_ensemble(coins: tuple) -> tuple:
    """Return (ensemble_df, current_prices) for the given coin list."""
    ensemble_map = {}
    curr_prices  = {}

    for coin in coins:
        model_preds = _load_predictions(coin)
        model_rmses = _load_rmse(coin)
        if not model_preds:
            continue

        matrix = pd.DataFrame(model_preds).T        # rows = models, cols = dates
        rmse_s = pd.Series(model_rmses).reindex(matrix.index)
        weights = (1 / rmse_s).fillna(0)
        w_sum   = weights.sum()
        weights = weights / w_sum if w_sum > 0 else pd.Series(1 / len(matrix), index=matrix.index)

        ensemble_map[coin] = matrix.astype(float).multiply(weights.values, axis=0).sum()

        cp = _current_price(coin)
        if cp is not None:
            curr_prices[coin] = cp

    if not ensemble_map:
        return pd.DataFrame(), {}

    ensemble_df = pd.DataFrame(ensemble_map)
    return ensemble_df, curr_prices


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio optimisation
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(arr: np.ndarray) -> np.ndarray:
    e = np.exp(arr - arr.max())
    return e / e.sum()


def _optimal_portfolio(cum_ret_df: pd.DataFrame, principal: float):
    """Search every non-empty subset of coins × every exit-day combination.

    Returns (best_coins, opt_days_dict, alloc_w_array, best_portfolio_value).
    Complexity: (1 + n_days)^N − 1 evaluations  (e.g. 8^6 ≈ 262k for 6 coins).
    """
    all_coins = list(cum_ret_df.columns)
    n_days    = len(cum_ret_df)
    arr       = cum_ret_df.values.astype(float)        # (n_days, n_coins)
    cidx      = {c: i for i, c in enumerate(all_coins)}

    best_V     = -np.inf
    best_coins = all_coins[:1]
    best_days  = {all_coins[0]: 0}
    best_w     = np.array([1.0])

    for size in range(1, len(all_coins) + 1):
        for subset in _icombinations(all_coins, size):
            sub = arr[:, [cidx[c] for c in subset]]   # (n_days, size)
            for combo in _iproduct(range(n_days), repeat=size):
                r = np.array([sub[combo[i], i] for i in range(size)])
                w = _softmax(r)
                V = principal * (1.0 + float(np.dot(w, r)))
                if V > best_V:
                    best_V     = V
                    best_coins = list(subset)
                    best_days  = {subset[i]: combo[i] for i in range(size)}
                    best_w     = w

    return best_coins, best_days, best_w, best_V


def run_optimization(principal: float, coins: list[str]) -> dict | None:
    """Build ensemble → search all coin-subsets × exit-days for max portfolio value."""
    ensemble_df, curr_prices = _build_ensemble(tuple(coins))
    # Only keep coins for which we have both predictions and a current price
    candidates = [c for c in coins if c in ensemble_df.columns and c in curr_prices]
    if not candidates:
        return None

    ensemble_df = ensemble_df[candidates]
    n_days      = len(ensemble_df)

    # ── Cumulative returns from current price ─────────────────────────────────
    cum_ret = pd.DataFrame(index=ensemble_df.index, columns=candidates, dtype=float)
    for coin in candidates:
        cum_ret[coin] = (ensemble_df[coin] - curr_prices[coin]) / curr_prices[coin]

    # ── Optimal subset + exit days ────────────────────────────────────────────
    best_coins, opt_days, alloc_w, _ = _optimal_portfolio(cum_ret, principal)

    # Restrict everything to the winning subset
    valid   = best_coins
    alloc_w = alloc_w   # softmax weights, already computed

    opt_returns = {c: float(cum_ret[c].iloc[opt_days[c]]) for c in valid}

    # ── Day-by-day simulation ─────────────────────────────────────────────────
    cash       = principal * float(1 - alloc_w.sum())
    holdings   = {c: principal * float(w) for c, w in zip(valid, alloc_w)}
    buy_price  = dict(curr_prices)
    # Units held per coin (= USD invested / purchase price)
    buy_units  = {c: holdings[c] / buy_price[c] for c in valid}
    prev_value = principal

    schedule_rows = []

    # "Today" row: BUY
    today_row = {'Date': 'Today', 'Portfolio ($)': principal,
                 'Daily Δ (%)': '—', 'vs Principal (%)': '—', 'Cash ($)': round(cash, 2)}
    for cidx2, coin in enumerate(valid):
        is_buy = alloc_w[cidx2] > 0
        units  = buy_units[coin]
        today_row[f'{coin} Price ($)'] = f"${curr_prices[coin]:,.2f}"
        today_row[f'{coin} Action']    = 'BUY 🟢' if is_buy else 'SKIP ⚪'
        today_row[f'{coin} Units']     = f"+{units:.6f}" if is_buy else '—'
        today_row[f'{coin} Txn ($)']   = round(-holdings[coin], 2) if is_buy else 0.0
        today_row[f'{coin} Value ($)'] = round(holdings[coin], 2)
    schedule_rows.append(today_row)

    for i, date in enumerate(ensemble_df.index):
        # Capture units before liquidation so SELL row shows correct qty
        units_before = {c: buy_units[c] if holdings[c] > 0 else 0.0 for c in valid}

        # Liquidate on optimal exit day
        for coin in valid:
            if i == opt_days[coin] and holdings[coin] > 0:
                pred        = float(ensemble_df[coin].iloc[i])
                ret         = (pred - buy_price[coin]) / buy_price[coin]
                cash       += holdings[coin] * (1 + ret)
                holdings[coin] = 0.0

        coin_values = {}
        for coin in valid:
            if holdings[coin] > 0:
                pred              = float(ensemble_df[coin].iloc[i])
                bp                = buy_price[coin]
                coin_values[coin] = holdings[coin] * (1 + (pred - bp) / bp)
            else:
                coin_values[coin] = 0.0

        pv           = cash + sum(coin_values.values())
        daily_ret    = (pv - prev_value) / prev_value if prev_value else 0.0
        vs_principal = (pv - principal) / principal * 100 if principal else 0.0
        prev_value   = pv

        row = {'Date': str(date), 'Portfolio ($)': round(pv, 2),
               'Daily Δ (%)': f"{daily_ret * 100:+.2f}%",
               'vs Principal (%)': f"{vs_principal:+.2f}%",
               'Cash ($)': round(cash, 2)}

        for cidx3, coin in enumerate(valid):
            w3      = alloc_w[cidx3]
            pred_px = float(ensemble_df[coin].iloc[i])
            u_before = units_before[coin]   # units held before any liquidation this day

            if w3 <= 0:
                action   = 'SKIP ⚪'
                units_str = '—'
                txn       = 0.0
            elif i == opt_days[coin]:
                action    = 'SELL 🔴'
                units_str = f"-{u_before:.6f}"
                txn       = round(u_before * pred_px, 2)   # cash received
            elif i < opt_days[coin]:
                action    = 'HOLD 🟡'
                units_str = f"{buy_units[coin]:.6f}"
                txn       = 0.0
            else:
                action    = 'CASH 💵'
                units_str = '—'
                txn       = 0.0

            coin_val = round(buy_units[coin] * pred_px, 2) if holdings[coin] > 0 else 0.0
            row[f'{coin} Price ($)'] = f"${pred_px:,.2f}"
            row[f'{coin} Action']    = action
            row[f'{coin} Units']     = units_str
            row[f'{coin} Txn ($)']   = txn
            row[f'{coin} Value ($)'] = coin_val
        schedule_rows.append(row)

    schedule_df = pd.DataFrame(schedule_rows).set_index('Date')

    # Reorder columns: per-coin (Price → Action → Units → Txn$ → Value$), then summary cols
    _coin_cols = []
    for _c in valid:
        for _sfx in [' Price ($)', ' Action', ' Units', ' Txn ($)', ' Value ($)']:
            _col = f'{_c}{_sfx}'
            if _col in schedule_df.columns:
                _coin_cols.append(_col)
    _other_cols = ['Cash ($)', 'Portfolio ($)', 'Daily Δ (%)', 'vs Principal (%)']
    schedule_df = schedule_df[[c for c in _coin_cols + _other_cols if c in schedule_df.columns]]

    # ── Allocation summary ────────────────────────────────────────────────────
    alloc_rows = []
    for coin, w in zip(valid, alloc_w):
        exit_i = opt_days[coin]
        alloc_rows.append({
            'Coin':              coin,
            'Allocation (%)':    round(w * 100, 1),
            'Amount ($)':        round(principal * w, 2),
            'Current Price ($)': round(curr_prices[coin], 2),
            'Expected Return (%)': round(opt_returns[coin] * 100, 2),
            'Exit Day':          exit_i + 1,
            'Exit Date':         str(ensemble_df.index[exit_i]),
        })
    alloc_df = pd.DataFrame(alloc_rows)

    # ── Portfolio value & returns series (incl. "Today") ─────────────────────
    dates_all  = ['Today'] + [r['Date'] for r in schedule_rows[1:]]
    pv_values  = [float(principal)] + [float(r['Portfolio ($)']) for r in schedule_rows[1:]]
    pv_series  = pd.Series(pv_values, index=dates_all, name='Portfolio Value ($)')

    # Day-over-day portfolio return (%) — Today is 0 by definition
    ret_values = [0.0]
    for i in range(1, len(pv_values)):
        prev = pv_values[i - 1]
        ret_values.append((pv_values[i] - prev) / prev * 100 if prev else 0.0)
    ret_series = pd.Series(ret_values, index=dates_all, name='Daily Return (%)')

    # Per-coin predicted-price return vs previous day (for the returns chart)
    # Day 0: vs current price; Day k: vs predicted price on day k-1
    coin_returns = {}
    for coin in valid:
        prices_with_current = [float(curr_prices[coin])] + [
            float(ensemble_df[coin].iloc[i]) for i in range(len(ensemble_df))
        ]
        rets = [0.0]  # "Today" = 0
        for i in range(1, len(prices_with_current)):
            prev = prices_with_current[i - 1]
            rets.append((prices_with_current[i] - prev) / prev * 100 if prev else 0.0)
        coin_returns[coin] = rets  # length = 1 + n_days

    return {
        'schedule':      schedule_df,
        'allocation':    alloc_df,
        'pv_series':     pv_series,
        'ret_series':    ret_series,
        'coin_returns':  coin_returns,
        'dates_all':     dates_all,
        'final_value':   prev_value,
        'principal':     principal,
        'ensemble_df':   ensemble_df,
        'curr_prices':   curr_prices,
        'valid_coins':   valid,
        'alloc_w':       alloc_w,
        'opt_days':      opt_days,
        'opt_returns':   opt_returns,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Retrain helpers
# ─────────────────────────────────────────────────────────────────────────────

def _spawn_train(coins: list[str], number: int) -> subprocess.Popen:
    """Fetch fresh data then run train_all_models for every coin in a background subprocess."""
    script = f"""
import sys, os
sys.path.insert(0, {repr(str(_root))})
os.chdir({repr(str(_root))})
from files.functions import coinbase_market_analysis_gradient, train_all_models
coins  = {coins!r}
number = {number}
for coin in coins:
    print(f'[{{coin}}] Fetching fresh market data...', flush=True)
    try:
        coinbase_market_analysis_gradient(coin, dataPath='data', plotPath='plots')
        print(f'[{{coin}}] Data ready.', flush=True)
    except Exception as e:
        print(f'[{{coin}}] Data fetch failed: {{e}}', flush=True)
    print(f'[{{coin}}] Training all models (last {{number}} days)...', flush=True)
    results = train_all_models(coin, number=number)
    ok  = [k for k, v in results.items() if v is not None]
    bad = [k for k, v in results.items() if v is None]
    print(f'[{{coin}}] Done — passed: {{ok}}, failed: {{bad}}', flush=True)
print('All coins finished.', flush=True)
"""
    return subprocess.Popen(
        [sys.executable, '-c', script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=str(_root),
    )


def _spawn_notebook(coins: list[str]) -> subprocess.Popen:
    """Replace myCoins in START_HERE.ipynb and execute the full pipeline notebook."""
    import json as _json
    nb_path = str(_root / 'START_HERE.ipynb')
    with open(nb_path) as fh:
        nb = _json.load(fh)
    coins_repr = repr(coins)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
            if src.strip().startswith('myCoins'):
                cell['source'] = f'myCoins = {coins_repr}'
                break
    with open(nb_path, 'w') as fh:
        _json.dump(nb, fh, indent=1)
    return subprocess.Popen(
        [sys.executable, '-m', 'jupyter', 'nbconvert',
         '--to', 'notebook', '--execute', '--inplace',
         '--ExecutePreprocessor.timeout=3600', nb_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=str(_root),
    )


def _add_coin_to_constants(coin: str) -> bool:
    """Append *coin* to the COINS list in CONSTANTS.py. Returns True if added, False if already present."""
    constants_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files', 'CONSTANTS.py')
    with open(constants_path) as fh:
        content = fh.read()
    match = re.search(r"COINS\s*=\s*\[([^\]]*)\]", content)
    if not match:
        return False
    existing = [c.strip().strip("'\"") for c in match.group(1).split(',') if c.strip()]
    if coin.upper() in existing:
        return False
    existing.append(coin.upper())
    new_line = "COINS = [" + ', '.join(f"'{c}'" for c in existing) + "]"
    new_content = content[:match.start()] + new_line + content[match.end():]
    with open(constants_path, 'w') as fh:
        fh.write(new_content)
    return True


def _spawn_retrain(coin: str) -> subprocess.Popen:
    """Launch all pipeline functions for *coin* in a background subprocess."""
    script = f"""
import sys, os
sys.path.insert(0, {repr(str(_root))})
os.chdir({repr(str(_root))})
from files.functions import (
    run_gbm_pipeline, run_svm_pipeline, run_knn_pipeline,
    run_arima_pipeline, run_lstm_pipeline, run_tft_pipeline,
    run_transformer_pipeline, run_prophet_pipeline,
)
coin = {repr(coin.upper())}
pipelines = [
    run_gbm_pipeline, run_svm_pipeline, run_knn_pipeline,
    run_arima_pipeline, run_lstm_pipeline, run_tft_pipeline,
    run_transformer_pipeline, run_prophet_pipeline,
]
for fn in pipelines:
    print(f'Starting {{fn.__name__}} for {{coin}} ...', flush=True)
    try:
        fn(coin=coin)
        print(f'  ✓ {{fn.__name__}} done', flush=True)
    except Exception as exc:
        print(f'  ✗ {{fn.__name__}} failed: {{exc}}', flush=True)
print('All pipelines finished.', flush=True)
"""
    return subprocess.Popen(
        [sys.executable, '-c', script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("📈 CryptoTrading2 — Portfolio Optimizer")
st.caption("RMSE-weighted ensemble of GBM · SVM · KNN · ARIMA · LSTM · TFT · Transformer · Prophet, "
           "with optimal 7-day exit strategy.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🏋️ Train Models")

    train_coins_raw = st.text_input(
        "Coins to train (comma-separated)",
        value=', '.join(COINS),
        help="Fetches the latest market data, then retrains all models for each coin.",
        key="train_coins_input",
    )
    train_days = st.number_input(
        "Training days (most recent)",
        min_value=30, max_value=3650, value=365, step=30,
        help="How many days of data to use, counting back from today. Default 365.",
        key="train_days_input",
    )
    train_btn = st.button("🚀 Fetch Data & Train", type="primary", use_container_width=True)

    if train_btn:
        parsed_train = [c.strip().upper() for c in train_coins_raw.split(',') if c.strip()]
        if not parsed_train:
            st.warning("Enter at least one coin code.")
        else:
            st.session_state['train_proc']  = _spawn_train(parsed_train, int(train_days))
            st.session_state['train_coins'] = parsed_train
            st.session_state['auto_optimize'] = False
            st.info(f"Training started for **{', '.join(parsed_train)}** "
                    f"using last **{int(train_days)}** days — this may take several minutes.")

    # Training status
    if 'train_proc' in st.session_state:
        tp: subprocess.Popen = st.session_state['train_proc']
        new_chunk = _read_proc_output(tp)
        if new_chunk:
            st.session_state['train_output'] = st.session_state.get('train_output', '') + new_chunk
        train_out_so_far = st.session_state.get('train_output', '')
        if tp.poll() is None:
            st.warning("⏳ Training in progress…")
            st.code(train_out_so_far or "(starting…)", language=None)
            time.sleep(2)
            st.rerun()
        else:
            remaining, _ = tp.communicate()
            full_out = train_out_so_far + (remaining or '')
            if tp.returncode == 0:
                st.success("✅ Training complete — portfolio will refresh.")
                _available_coins.clear()
                _build_ensemble.clear()
                st.session_state['auto_optimize'] = True
            else:
                st.error("Training finished with errors — check log.")
            with st.expander("Training log"):
                st.code(full_out or "(no output)", language=None)
            del st.session_state['train_proc']
            st.session_state.pop('train_output', None)

    st.divider()
    st.header("⚙️ Parameters")

    lump_sum = st.number_input(
        "Lump Sum (USD $)",
        min_value=1.0,
        value=100.0,
        step=100.0,
        format="%.2f",
        help="Total capital to invest across all selected coins.",
    )

    available = _available_coins()
    # Include any newly trained coins that session state may have added
    _extra = [c for c in st.session_state.get('extra_coins', []) if c in available]
    _default = list(dict.fromkeys(available + _extra))   # preserve order, no dupes
    selected_coins = st.multiselect(
        "Candidate coins (optimizer auto-selects best subset)",
        options=_default,
        default=_default,
        help="All coins with trained predictions. The optimizer automatically picks the subset and exit days that maximise portfolio value.",
    )

    # ── Add a brand-new coin inline ───────────────────────────────────────────
    _nc_col, _nb_col = st.columns([3, 1])
    new_portfolio_coin = _nc_col.text_input(
        "Add new coin",
        placeholder="e.g. DOGE",
        label_visibility="collapsed",
        key="new_portfolio_coin",
    ).strip().upper()
    add_coin_btn = _nb_col.button("➕ Add", use_container_width=True)

    if add_coin_btn:
        if not new_portfolio_coin:
            st.warning("Enter a coin code first.")
        elif new_portfolio_coin in available:
            st.info(f"**{new_portfolio_coin}** already has trained models — select it above.")
        else:
            _add_coin_to_constants(new_portfolio_coin)
            st.session_state['train_proc']  = _spawn_train([new_portfolio_coin], int(train_days))
            st.session_state['train_coins'] = [new_portfolio_coin]
            st.session_state['auto_optimize'] = False
            extras = st.session_state.get('extra_coins', [])
            if new_portfolio_coin not in extras:
                extras.append(new_portfolio_coin)
            st.session_state['extra_coins'] = extras
            st.info(f"Fetching data & training **{new_portfolio_coin}** — "
                    "it will appear in the list when done.")

    run_btn = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

    if run_btn:
        if selected_coins:
            st.session_state['nb_proc'] = _spawn_notebook(selected_coins)
            st.info(f"Starting full pipeline for **{', '.join(selected_coins)}** — "
                    "this may take several minutes.")

    st.divider()
    st.caption("Models are weighted by **1 / RMSE** — "
               "better-calibrated models carry more weight in the ensemble.")

    st.divider()
    st.header("🔁 Retrain Models")
    new_coin_input = st.text_input(
        "Coin code",
        placeholder="e.g. DOGE",
        help="Enter a coin ticker to add to COINS and retrain all models.",
    ).strip().upper()
    retrain_btn = st.button("➕ Add Coin & Retrain", use_container_width=True)

    if retrain_btn:
        if not new_coin_input:
            st.warning("Enter a coin code first.")
        else:
            added = _add_coin_to_constants(new_coin_input)
            if added:
                st.success(f"**{new_coin_input}** added to COINS.")
            else:
                st.info(f"**{new_coin_input}** already in COINS.")
            st.session_state['retrain_proc'] = _spawn_retrain(new_coin_input)
            st.session_state['retrain_coin'] = new_coin_input
            st.info(f"Retraining all models for **{new_coin_input}** in background…")

    # Show live log if retraining is running
    if 'retrain_proc' in st.session_state:
        proc: subprocess.Popen = st.session_state['retrain_proc']
        coin_name = st.session_state.get('retrain_coin', '')
        if proc.poll() is None:
            st.warning(f"⏳ Training **{coin_name}** in progress…")
        else:
            output, _ = proc.communicate()
            if proc.returncode == 0:
                st.success(f"✅ **{coin_name}** training complete!")
            else:
                st.error(f"Training **{coin_name}** finished with errors.")
            with st.expander("Training log"):
                st.code(output or "(no output)")
            del st.session_state['retrain_proc']

# ── Main area ─────────────────────────────────────────────────────────────────
auto_optimize = st.session_state.pop('auto_optimize', False)

if 'nb_proc' in st.session_state:
    _np = st.session_state['nb_proc']
    _new_chunk = _read_proc_output(_np)
    if _new_chunk:
        st.session_state['nb_output'] = st.session_state.get('nb_output', '') + _new_chunk
    _nb_out_so_far = st.session_state.get('nb_output', '')
    if _np.poll() is None:
        st.info("⏳ Full notebook pipeline is running…")
        st.code(_nb_out_so_far or "(starting…)", language=None)
        time.sleep(2)
        st.rerun()
        st.stop()
    else:
        _remaining, _ = _np.communicate()
        _full_nb_out = _nb_out_so_far + (_remaining or '')
        if _np.returncode == 0:
            st.success("✅ Notebook pipeline complete — running portfolio optimization.")
            _available_coins.clear()
            _build_ensemble.clear()
            auto_optimize = True
        else:
            st.error("Notebook pipeline finished with errors.")
            with st.expander("Notebook log"):
                st.code(_full_nb_out or "(no output)", language=None)
            st.stop()
        del st.session_state['nb_proc']
        st.session_state.pop('nb_output', None)

if not run_btn and not auto_optimize:
    st.info("👈 Set your lump sum, select coins, and click **Run Optimization**.")
    st.stop()

if not selected_coins:
    st.warning("Please select at least one coin.")
    st.stop()

with st.spinner("Computing ensemble forecasts and optimizing portfolio…"):
    result = run_optimization(lump_sum, selected_coins)

if result is None:
    st.error("No prediction data found for the selected coins. "
             "Run the training notebooks first to generate `predictions/` files.")
    st.stop()

final_value    = result['final_value']
total_ret_pct  = (final_value / lump_sum - 1) * 100
net_gain       = final_value - lump_sum

# ── Key metrics ───────────────────────────────────────────────────────────────
st.subheader("📊 Forecast Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Principal",        f"${lump_sum:,.2f}")
c2.metric("Final Portfolio",  f"${final_value:,.2f}", delta=f"{total_ret_pct:+.2f}%")
c3.metric("Net Gain / Loss",  f"${net_gain:+,.2f}",  delta=f"{total_ret_pct:+.2f}%")
_best = result['valid_coins']
c4.metric("Coins Selected",   f"{len(_best)} of {len(selected_coins)} candidates",
          help=f"Auto-selected: {', '.join(_best)}")

# ── Daily Action Plan ────────────────────────────────────────────────────────
st.subheader("📋 Daily Trading Instructions")
st.caption(f"Optimal portfolio: **{', '.join(result['valid_coins'])}** — "
           f"auto-selected from {len(selected_coins)} candidates to maximise portfolio value.")

_sched      = result['schedule']
_valid      = result['valid_coins']
_curr_px    = result['curr_prices']
_buy_units  = {c: result['alloc_w'][i] * lump_sum / _curr_px[c]
               for i, c in enumerate(_valid)}
_exit_days  = result['opt_days']

_ACTION_ICON = {'BUY 🟢': '🟢', 'SELL 🔴': '🔴', 'HOLD 🟡': '🟡', 'CASH 💵': '💵', 'SKIP ⚪': '⚪'}

for _day_idx, (_date, _row) in enumerate(_sched.iterrows()):
    # Determine which coins are transacting today
    _selling = [c for c in _valid if f'{c} Action' in _row.index and 'SELL' in str(_row[f'{c} Action'])]
    _buying  = [c for c in _valid if f'{c} Action' in _row.index and 'BUY'  in str(_row[f'{c} Action'])]
    if _selling:
        _day_label = f"**{'TODAY' if _day_idx == 0 else f'Day {_day_idx}'} — {_date}** &nbsp; 🔴 SELL: {', '.join(_selling)}"
    elif _buying:
        _day_label = f"**{'TODAY' if _day_idx == 0 else f'Day {_day_idx}'} — {_date}** &nbsp; 🟢 BUY: {', '.join(_buying)}"
    else:
        _day_label = f"**{'TODAY' if _day_idx == 0 else f'Day {_day_idx}'} — {_date}** &nbsp; 🟡 HOLD ALL"

    with st.expander(_day_label, expanded=(_day_idx == 0 or bool(_selling))):
        _cols = st.columns([1, 1, 1, 1, 1])
        _cols[0].markdown("**Coin**")
        _cols[1].markdown("**Action**")
        _cols[2].markdown("**Units**")
        _cols[3].markdown("**@ Price**")
        _cols[4].markdown("**Amount / Value**")
        st.divider()

        for _c in _valid:
            _action  = str(_row.get(f'{_c} Action', '—'))
            _units_s = str(_row.get(f'{_c} Units', '—'))
            _price_s = str(_row.get(f'{_c} Price ($)', '—'))
            _txn     = _row.get(f'{_c} Txn ($)', 0.0)
            _val     = _row.get(f'{_c} Value ($)', 0.0)
            _cols2   = st.columns([1, 1, 1, 1, 1])
            _cols2[0].write(f"**{_c}**")
            _cols2[1].write(_action)
            _cols2[2].write(_units_s)
            _cols2[3].write(_price_s)
            # Amount: show transaction if BUY/SELL, else current value
            if 'SELL' in _action and _txn:
                _buy_spent = _buy_units.get(_c, 0) * _curr_px.get(_c, 0)
                _gain      = float(_txn) - _buy_spent
                _cols2[4].markdown(
                    f"<span style='color:#155724;font-weight:bold'>"
                    f"+${float(_txn):,.2f} &nbsp;({_gain:+,.2f})</span>",
                    unsafe_allow_html=True,
                )
            elif 'BUY' in _action and _txn:
                _cols2[4].markdown(
                    f"<span style='color:#721c24;font-weight:bold'>"
                    f"−${abs(float(_txn)):,.2f}</span>",
                    unsafe_allow_html=True,
                )
            else:
                _cols2[4].write(f"${float(_val):,.2f}" if _val else "—")

        st.divider()
        _pv   = _row.get('Portfolio ($)', 0.0)
        _cash = _row.get('Cash ($)', 0.0)
        _d1, _d2, _d3 = st.columns(3)
        _d1.metric("Portfolio Value", f"${float(_pv):,.2f}")
        _d2.metric("Cash",            f"${float(_cash):,.2f}")
        _d3.metric("vs Principal",    _row.get('vs Principal (%)', '—'))

st.divider()

# ── Charts (collapsible) ─────────────────────────────────────────────────────
pv        = result['pv_series']
rets      = result['ret_series']
dates_all = result['dates_all']
pv_vals   = pv.values.tolist()

with st.expander("💹 Portfolio Value & Returns Chart", expanded=True):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.06,
        subplot_titles=('Portfolio Value (USD)', 'Day-over-Day Return (%)'),
    )
    fig.add_trace(go.Scatter(
        x=dates_all, y=[max(v, lump_sum) for v in pv_vals],
        fill=None, mode='lines', line=dict(width=0), showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates_all, y=[lump_sum] * len(dates_all),
        fill='tonexty', mode='lines', line=dict(width=0),
        fillcolor='rgba(76,175,80,0.18)', name='Gain', showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates_all, y=[min(v, lump_sum) for v in pv_vals],
        fill=None, mode='lines', line=dict(width=0), showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates_all, y=[lump_sum] * len(dates_all),
        fill='tonexty', mode='lines', line=dict(width=0),
        fillcolor='rgba(244,67,54,0.18)', name='Loss', showlegend=True,
    ), row=1, col=1)
    fig.add_hline(y=lump_sum, line_dash='dash', line_color='gray',
                  annotation_text=f'Principal ${lump_sum:,.0f}',
                  annotation_position='bottom right', row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates_all, y=pv_vals,
        mode='lines+markers+text', line=dict(color='#2196F3', width=2.5),
        marker=dict(size=8),
        text=[f'${v:,.0f}' for v in pv_vals], textposition='top center',
        textfont=dict(size=11), name='Portfolio Value',
    ), row=1, col=1)
    coin_colors = ['#FF9800', '#9C27B0', '#F44336', '#00BCD4', '#795548', '#E91E63']
    for cidx, coin in enumerate(result['valid_coins']):
        exit_idx  = result['opt_days'][coin] + 1
        exit_date = dates_all[exit_idx]
        color     = coin_colors[cidx % len(coin_colors)]
        fig.add_vline(x=exit_date, line_dash='dot', line_color=color, line_width=1.5, row='all', col=1)
        fig.add_trace(go.Scatter(
            x=[exit_date], y=[pv_vals[exit_idx]], mode='markers+text',
            marker=dict(symbol='star', size=18, color=color, line=dict(color='white', width=1)),
            text=[f'SELL {coin}<br>{result["opt_returns"][coin]*100:+.2f}%'],
            textposition='bottom center', textfont=dict(size=10, color=color),
            name=f'Sell {coin}', showlegend=True,
        ), row=1, col=1)
    ret_vals = rets.values.tolist()
    fig.add_trace(go.Bar(
        x=dates_all, y=ret_vals,
        marker_color=['#4CAF50' if r >= 0 else '#F44336' for r in ret_vals],
        text=[f'{r:+.2f}%' for r in ret_vals], textposition='outside',
        textfont=dict(size=10), name='Daily Return', showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_color='gray', line_width=1, row=2, col=1)
    fig.update_yaxes(tickprefix='$', tickformat=',.0f', row=1, col=1)
    fig.update_yaxes(ticksuffix='%', row=2, col=1)
    fig.update_layout(height=580, margin=dict(t=40, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      plot_bgcolor='white')
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    st.plotly_chart(fig, use_container_width=True)

with st.expander("📈 Predicted Price Returns by Coin"):
    fig2 = go.Figure()
    line_colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336', '#00BCD4']
    for idx, coin in enumerate(result['valid_coins']):
        rets_coin  = result['coin_returns'][coin]
        lc         = line_colors[idx % len(line_colors)]
        exit_idx   = result['opt_days'][coin] + 1
        fig2.add_trace(go.Scatter(
            x=dates_all, y=rets_coin, mode='lines+markers+text', name=coin,
            line=dict(color=lc, width=2), marker=dict(size=7, color=lc),
            text=[f'{r:+.2f}%' for r in rets_coin], textposition='top center',
            textfont=dict(size=9),
        ))
        fig2.add_trace(go.Scatter(
            x=[dates_all[exit_idx]], y=[rets_coin[exit_idx]], mode='markers+text',
            marker=dict(symbol='star', size=18, color=lc, line=dict(color='white', width=1)),
            text=[f'Exit<br>{result["opt_returns"][coin]*100:+.2f}%'],
            textposition='bottom center', textfont=dict(size=9, color=lc), showlegend=False,
        ))
        fig2.add_vline(x=dates_all[exit_idx], line_dash='dot', line_color=lc, line_width=1.5)
    fig2.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1)
    fig2.update_layout(yaxis_title='Return vs Previous Day (%)', yaxis_ticksuffix='%',
                       height=360, margin=dict(t=20, b=20), plot_bgcolor='white',
                       legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig2.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig2.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    st.plotly_chart(fig2, use_container_width=True)

# ── Allocation table ──────────────────────────────────────────────────────────
with st.expander("🎯 Capital Allocation"):
    alloc_df  = result['allocation']
    invested  = alloc_df[alloc_df['Allocation (%)'] > 0]
    cash_pct  = 100.0 - invested['Allocation (%)'].sum()
    if not invested.empty:
        st.dataframe(
            invested.style
                .format({'Allocation (%)': '{:.1f}%', 'Amount ($)': '${:,.2f}',
                         'Current Price ($)': '${:,.2f}', 'Expected Return (%)': '{:+.2f}%'})
                .background_gradient(subset=['Expected Return (%)'], cmap='RdYlGn'),
            use_container_width=True, hide_index=True,
        )
    if cash_pct > 0.5:
        st.info(f"💵 **{cash_pct:.1f}%** (${lump_sum*cash_pct/100:,.2f}) stays in cash.")

# ── Detailed schedule table ───────────────────────────────────────────────────
with st.expander("📅 Full Day-by-Day Schedule Table"):
    sched       = result['schedule'].copy()
    action_cols = [c for c in sched.columns if c.endswith(' Action')]
    value_cols  = [c for c in sched.columns if c.endswith(' Value ($)')]
    txn_cols    = [c for c in sched.columns if c.endswith(' Txn ($)')]

    def _color_action(val):
        if 'BUY'  in str(val): return 'background-color:#d4edda;color:#155724;font-weight:bold'
        if 'SELL' in str(val): return 'background-color:#f8d7da;color:#721c24;font-weight:bold'
        if 'HOLD' in str(val): return 'background-color:#fff3cd;color:#856404'
        return ''

    def _color_txn(val):
        try:
            v = float(val)
            if v > 0: return 'color:#155724;font-weight:bold'
            if v < 0: return 'color:#721c24;font-weight:bold'
        except Exception:
            pass
        return ''

    def _color_delta(val):
        try:
            v = float(str(val).replace('%','').replace('+','').replace('—','0'))
            if v > 0: return 'color:#155724'
            if v < 0: return 'color:#721c24'
        except Exception:
            pass
        return ''

    def _color_vp(val):
        try:
            v = float(str(val).replace('%','').replace('+','').replace('—','0'))
            if v > 0: return 'background-color:#d4edda;color:#155724;font-weight:bold'
            if v < 0: return 'background-color:#f8d7da;color:#721c24;font-weight:bold'
        except Exception:
            pass
        return ''

    styled_sched = (
        sched.style
            .applymap(_color_action, subset=action_cols)
            .applymap(_color_txn,    subset=txn_cols)
            .applymap(_color_delta,  subset=['Daily Δ (%)'])
            .applymap(_color_vp,     subset=['vs Principal (%)'])
            .format({'Portfolio ($)': '${:,.2f}', 'Cash ($)': '${:,.2f}',
                     **{c: '${:,.2f}'  for c in value_cols},
                     **{c: '${:+,.2f}' for c in txn_cols}})
    )
    st.dataframe(styled_sched, use_container_width=True)
    st.caption("Txn ($): negative = cash spent, positive = cash received. Units = coin quantity.")

# ── Ensemble forecast ─────────────────────────────────────────────────────────
with st.expander("🔮 Model Ensemble Forecast Prices"):
    ens = result['ensemble_df'].copy()
    ens.index.name = 'Date'
    st.dataframe(ens.style.format({c: '${:,.2f}' for c in ens.columns})
                           .background_gradient(cmap='RdYlGn', axis=0),
                 use_container_width=True)
    st.caption("RMSE-weighted average prediction across all trained models.")
