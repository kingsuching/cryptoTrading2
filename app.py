import os
import re
import sys
import subprocess
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
    try:
        raw   = pd.read_csv(fullDataPath(coin))
        daily = dataSetup(raw, trainingColPath=TRAINING_COLUMNS,
                          response=RESPONSE_VARIABLE, number=LIMIT)
        return float(daily[RESPONSE_VARIABLE].iloc[-1])
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _available_coins() -> list[str]:
    pred_base = base_dir('predictions')
    return [c for c in COINS
            if any(f.startswith(c) and '_future_predictions.csv' in f
                   for f in os.listdir(pred_base))]


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

def run_optimization(principal: float, coins: list[str]) -> dict | None:
    """Run the RMSE-weighted ensemble + optimal-exit portfolio algorithm.

    Returns a dict with all outputs, or None if data is missing.
    """
    ensemble_df, curr_prices = _build_ensemble(tuple(coins))
    valid = [c for c in coins if c in ensemble_df.columns and c in curr_prices]
    if not valid:
        return None

    ensemble_df = ensemble_df[valid]
    n_days      = len(ensemble_df)

    # ── Cumulative returns from current price ─────────────────────────────────
    cum_ret = pd.DataFrame(index=ensemble_df.index, columns=valid, dtype=float)
    for coin in valid:
        cp            = curr_prices[coin]
        cum_ret[coin] = (ensemble_df[coin] - cp) / cp

    opt_days    = {c: int(cum_ret[c].values.argmax())    for c in valid}
    opt_returns = {c: float(cum_ret[c].iloc[opt_days[c]]) for c in valid}

    # ── Capital allocation ────────────────────────────────────────────────────
    # Prefer positive-return coins; if none exist, allocate to the least-negative.
    ret_arr  = np.array([opt_returns[c] for c in valid])
    pos      = np.maximum(ret_arr, 0.0)
    total    = pos.sum()
    if total > 0:
        alloc_w = pos / total
    else:
        # All returns are ≤ 0 — shift so best coin = max weight, worst = 0
        shifted = ret_arr - ret_arr.min()          # all ≥ 0, best coin is largest
        s       = shifted.sum()
        alloc_w = shifted / s if s > 0 else np.ones(len(valid)) / len(valid)

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
        today_row[f'{coin} Price ($)']  = f"${curr_prices[coin]:,.2f}"
        today_row[f'{coin} Value ($)']  = round(holdings[coin], 2)
        today_row[f'{coin} Action']     = 'BUY 🟢' if alloc_w[cidx2] > 0 else 'SKIP ⚪'
    schedule_rows.append(today_row)

    for i, date in enumerate(ensemble_df.index):
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
                pred            = float(ensemble_df[coin].iloc[i])
                bp              = buy_price[coin]
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
            w3  = alloc_w[cidx3]
            pred_px = float(ensemble_df[coin].iloc[i])
            if w3 <= 0:
                action = 'SKIP ⚪'
            elif i < opt_days[coin]:
                action = 'HOLD 🟡'
            elif i == opt_days[coin]:
                action = 'SELL 🔴'
            else:
                action = 'CASH 💵'
            # Value = units * predicted close price (0 once sold)
            coin_val = round(buy_units[coin] * pred_px, 2) if holdings[coin] > 0 else 0.0
            row[f'{coin} Price ($)'] = f"${pred_px:,.2f}"
            row[f'{coin} Value ($)'] = coin_val
            row[f'{coin} Action']    = action
        schedule_rows.append(row)

    schedule_df = pd.DataFrame(schedule_rows).set_index('Date')

    # Reorder columns: per-coin (Price → Value → Action), then Cash, then portfolio metrics
    _coin_cols = []
    for _c in valid:
        for _sfx in [' Price ($)', ' Value ($)', ' Action']:
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
    st.header("⚙️ Parameters")

    lump_sum = st.number_input(
        "Lump Sum (USD $)",
        min_value=1.0,
        value=10_000.0,
        step=100.0,
        format="%.2f",
        help="Total capital to invest across all selected coins.",
    )

    available = _available_coins()
    selected_coins = st.multiselect(
        "Coins to include",
        options=available,
        default=available,
        help="Only coins with trained model predictions are shown.",
    )

    run_btn = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

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
if not run_btn:
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
st.subheader("📊 7-Day Forecast Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Principal",           f"${lump_sum:,.2f}")
c2.metric("Final Portfolio",     f"${final_value:,.2f}",   delta=f"{total_ret_pct:+.2f}%")
c3.metric("Net Gain / Loss",     f"${net_gain:+,.2f}",     delta=f"{total_ret_pct:+.2f}%")
c4.metric("Coins Invested",      f"{int((result['alloc_w'] > 0).sum())} / {len(selected_coins)}")

# ── Portfolio value + daily returns (two-panel Plotly chart) ─────────────────
st.subheader("💹 Portfolio Value & Daily Returns")

pv   = result['pv_series']
rets = result['ret_series']
dates_all = result['dates_all']

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.06,
    subplot_titles=('Portfolio Value (USD)', 'Day-over-Day Return (%)'),
)

# ── Top panel: portfolio value ────────────────────────────────────────────────
pv_vals = pv.values.tolist()

# Gain fill (green above principal)
fig.add_trace(go.Scatter(
    x=dates_all, y=[max(v, lump_sum) for v in pv_vals],
    fill=None, mode='lines', line=dict(width=0), showlegend=False,
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=dates_all, y=[lump_sum] * len(dates_all),
    fill='tonexty', mode='lines', line=dict(width=0),
    fillcolor='rgba(76,175,80,0.18)', name='Gain', showlegend=True,
), row=1, col=1)

# Loss fill (red below principal)
fig.add_trace(go.Scatter(
    x=dates_all, y=[min(v, lump_sum) for v in pv_vals],
    fill=None, mode='lines', line=dict(width=0), showlegend=False,
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=dates_all, y=[lump_sum] * len(dates_all),
    fill='tonexty', mode='lines', line=dict(width=0),
    fillcolor='rgba(244,67,54,0.18)', name='Loss', showlegend=True,
), row=1, col=1)

# Principal line
fig.add_hline(y=lump_sum, line_dash='dash', line_color='gray',
              annotation_text=f'Principal ${lump_sum:,.0f}',
              annotation_position='bottom right', row=1, col=1)

# Portfolio value line
fig.add_trace(go.Scatter(
    x=dates_all,
    y=pv_vals,
    mode='lines+markers+text',
    line=dict(color='#2196F3', width=2.5),
    marker=dict(size=8),
    text=[f'${v:,.0f}' for v in pv_vals],
    textposition='top center',
    textfont=dict(size=11),
    name='Portfolio Value',
), row=1, col=1)

# ── Optimal exit markers on portfolio value chart ─────────────────────────────
coin_colors = ['#FF9800', '#9C27B0', '#F44336', '#00BCD4', '#795548', '#E91E63']
for cidx, coin in enumerate(result['valid_coins']):
    w = result['alloc_w'][cidx]
    if w <= 0:
        continue
    exit_idx   = result['opt_days'][coin] + 1          # +1 because dates_all[0]='Today'
    exit_date  = dates_all[exit_idx]
    exit_pv    = pv_vals[exit_idx]
    exit_ret   = result['opt_returns'][coin] * 100
    color      = coin_colors[cidx % len(coin_colors)]

    # Vertical dashed line through both panels
    fig.add_vline(x=exit_date, line_dash='dot', line_color=color,
                  line_width=1.5, row='all', col=1)

    # Star marker at portfolio value on exit day
    fig.add_trace(go.Scatter(
        x=[exit_date],
        y=[exit_pv],
        mode='markers+text',
        marker=dict(symbol='star', size=18, color=color,
                    line=dict(color='white', width=1)),
        text=[f'SELL {coin}<br>{exit_ret:+.2f}%'],
        textposition='bottom center',
        textfont=dict(size=10, color=color),
        name=f'Sell {coin}',
        showlegend=True,
    ), row=1, col=1)

# ── Bottom panel: day-over-day returns ────────────────────────────────────────
ret_vals   = rets.values.tolist()
bar_colors_ret = ['#4CAF50' if r >= 0 else '#F44336' for r in ret_vals]

fig.add_trace(go.Bar(
    x=dates_all,
    y=ret_vals,
    marker_color=bar_colors_ret,
    text=[f'{r:+.2f}%' for r in ret_vals],
    textposition='outside',
    textfont=dict(size=10),
    name='Daily Return',
    showlegend=False,
), row=2, col=1)
fig.add_hline(y=0, line_color='gray', line_width=1, row=2, col=1)

fig.update_yaxes(tickprefix='$', tickformat=',.0f', row=1, col=1)
fig.update_yaxes(ticksuffix='%', row=2, col=1)
fig.update_layout(
    height=580,
    margin=dict(t=40, b=20),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    plot_bgcolor='white',
)
fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')

st.plotly_chart(fig, use_container_width=True)

# ── Per-coin predicted-price returns chart ────────────────────────────────────
st.subheader("📈 Predicted Price Returns by Coin (Day-over-Day)")

fig2 = go.Figure()
line_colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336', '#00BCD4']
for idx, coin in enumerate(result['valid_coins']):
    rets_coin  = result['coin_returns'][coin]
    line_color = line_colors[idx % len(line_colors)]
    exit_idx   = result['opt_days'][coin] + 1   # +1 for 'Today' offset
    exit_ret   = result['opt_returns'][coin] * 100

    # Regular line + markers
    fig2.add_trace(go.Scatter(
        x=dates_all,
        y=rets_coin,
        mode='lines+markers+text',
        name=coin,
        line=dict(color=line_color, width=2),
        marker=dict(size=7, color=line_color),
        text=[f'{r:+.2f}%' for r in rets_coin],
        textposition='top center',
        textfont=dict(size=9),
    ))

    # Star at optimal exit day
    fig2.add_trace(go.Scatter(
        x=[dates_all[exit_idx]],
        y=[rets_coin[exit_idx]],
        mode='markers+text',
        marker=dict(symbol='star', size=18, color=line_color,
                    line=dict(color='white', width=1)),
        text=[f'Max exit<br>{exit_ret:+.2f}% total'],
        textposition='bottom center',
        textfont=dict(size=9, color=line_color),
        showlegend=False,
    ))

    # Vertical dashed line at exit day
    fig2.add_vline(x=dates_all[exit_idx], line_dash='dot',
                   line_color=line_color, line_width=1.5)

fig2.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1)
fig2.update_layout(
    yaxis_title='Return vs Previous Day (%)',
    yaxis_ticksuffix='%',
    height=360,
    margin=dict(t=20, b=20),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    plot_bgcolor='white',
)
fig2.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
fig2.update_yaxes(showgrid=True, gridcolor='#f0f0f0')

st.plotly_chart(fig2, use_container_width=True)

# ── Allocation table ──────────────────────────────────────────────────────────
st.subheader("🎯 Capital Allocation")
alloc_df    = result['allocation']
invested    = alloc_df[alloc_df['Allocation (%)'] > 0]
cash_pct    = 100.0 - invested['Allocation (%)'].sum()
cash_amount = lump_sum * cash_pct / 100

if not invested.empty:
    st.dataframe(
        invested.style
            .format({'Allocation (%)': '{:.1f}%', 'Amount ($)': '${:,.2f}',
                     'Current Price ($)': '${:,.2f}', 'Expected Return (%)': '{:+.2f}%'})
            .background_gradient(subset=['Expected Return (%)'], cmap='RdYlGn'),
        use_container_width=True,
        hide_index=True,
    )

if cash_pct > 0.01:
    st.info(f"💵 **{cash_pct:.1f}%** (${cash_amount:,.2f}) stays in cash.")

# ── Day-by-day schedule ───────────────────────────────────────────────────────
st.subheader("📅 Day-by-Day Trading Schedule")

sched = result['schedule'].copy()
action_cols = [c for c in sched.columns if 'Action' in c]
price_cols  = [c for c in sched.columns if 'Price' in c]
value_cols  = [c for c in sched.columns if 'Value ($)' in c]

def _color_action(val):
    if 'BUY'  in str(val): return 'background-color: #d4edda; color: #155724'
    if 'SELL' in str(val): return 'background-color: #f8d7da; color: #721c24'
    if 'HOLD' in str(val): return 'background-color: #fff3cd; color: #856404'
    return ''

def _color_delta(val):
    try:
        v = float(str(val).replace('%', '').replace('+', '').replace('—', '0'))
        if v > 0:  return 'color: #155724'
        if v < 0:  return 'color: #721c24'
    except Exception:
        pass
    return ''

def _color_vs_principal(val):
    try:
        v = float(str(val).replace('%', '').replace('+', '').replace('—', '0'))
        if v > 0:  return 'background-color: #d4edda; color: #155724; font-weight: bold'
        if v < 0:  return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
    except Exception:
        pass
    return ''

_value_fmt = {c: '${:,.2f}' for c in value_cols}
styled_sched = (
    sched.style
        .applymap(_color_action,       subset=action_cols)
        .applymap(_color_delta,        subset=['Daily Δ (%)'])
        .applymap(_color_vs_principal, subset=['vs Principal (%)'])
        .format({'Portfolio ($)': '${:,.2f}', 'Cash ($)': '${:,.2f}', **_value_fmt})
)
st.dataframe(styled_sched, use_container_width=True)

# ── Ensemble forecast table ───────────────────────────────────────────────────
with st.expander("🔮 Model Ensemble Forecast Prices"):
    ens = result['ensemble_df'].copy()
    ens.index.name = 'Date'
    fmt = {c: '${:,.2f}' for c in ens.columns}
    st.dataframe(ens.style.format(fmt).background_gradient(cmap='RdYlGn', axis=0),
                 use_container_width=True)
    st.caption("Each column is the RMSE-weighted average prediction across all trained models.")
