# ============================================================
# run_weekly.py — AI Stock Screener (Indian Markets)
# Day 10: Weekly Automation
#
# Run every weekend:
#   python run_weekly.py
#
# Estimated time: ~25 minutes
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import os
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ── PATHS ────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, 'data')
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
UNIVERSE_DIR  = os.path.join(DATA_DIR, 'universe')
FUND_DIR      = os.path.join(DATA_DIR, 'fundamentals')
PRICES_DIR    = os.path.join(DATA_DIR, 'prices')
SCORES_DIR    = os.path.join(DATA_DIR, 'scores')
REPORTS_TECH  = os.path.join(DATA_DIR, 'reports', 'technical')
REPORTS_ML    = os.path.join(DATA_DIR, 'reports', 'ml')
TEMP_DIR      = os.path.join(DATA_DIR, 'temp')

for d in [UNIVERSE_DIR, FUND_DIR, PRICES_DIR, SCORES_DIR,
          REPORTS_TECH, REPORTS_ML, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print("  AI Stock Screener — Weekly Run")
print(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)

# ── STEP 1: LOAD STOCKS ──────────────────────────────────────
print("\n[1/9] Loading stocks...")

quality_df = pd.read_csv(os.path.join(UNIVERSE_DIR, 'quality_passed.csv'))
prefilt_df = pd.read_csv(os.path.join(UNIVERSE_DIR, 'prefilt_passed.csv'))

ticker_map = {}
for _, row in prefilt_df.iterrows():
    sym = row['Symbol']
    ticker_map[sym] = f"{sym}.NS" if row['Exchange'] == 'NSE' else f"{sym}.BO"

symbols = quality_df['Symbol'].tolist()
print(f"     Stocks : {len(symbols)}")

# ── STEP 2: INCREMENTAL PRICE UPDATE ─────────────────────────
print(f"\n[2/9] Updating price data (incremental)...")
print(f"      Started : {datetime.now().strftime('%H:%M:%S')}")

PRICE_FILE = os.path.join(PRICES_DIR, 'price_data_full.pkl')

with open(PRICE_FILE, 'rb') as f:
    price_data = pickle.load(f)

updated = 0
skipped = 0
failed  = []

for i, symbol in enumerate(symbols):
    ticker = ticker_map.get(symbol, f"{symbol}.NS")
    try:
        existing_df = price_data.get(symbol)
        if existing_df is not None and len(existing_df) > 0:
            last_date = existing_df.index[-1]
            if hasattr(last_date, 'tzinfo') and last_date.tzinfo is not None:
                last_date = last_date.tz_localize(None)
            start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            start_date = '2010-01-01'

        today    = datetime.now().strftime('%Y-%m-%d')
        tomorrow = (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        # Skip if start_date is in the future OR is a weekend (market closed)
        start_dt    = pd.Timestamp(start_date)
        is_future   = start_date > today
        is_weekend  = start_dt.weekday() >= 5  # Sat=5, Sun=6

        if is_future or is_weekend:
            skipped += 1
            continue

        new_df = yf.download(ticker, start=start_date, end=tomorrow,
                             interval='1d', progress=False, auto_adjust=True)

        if isinstance(new_df.columns, pd.MultiIndex):
            new_df.columns = [col[0] for col in new_df.columns]
        new_df = new_df.loc[:, ~new_df.columns.duplicated()]

        if new_df is not None and len(new_df) > 0:
            if existing_df is not None:
                if hasattr(existing_df.index, 'tz') and existing_df.index.tz is not None:
                    existing_df.index = existing_df.index.tz_localize(None)
                if hasattr(new_df.index, 'tz') and new_df.index.tz is not None:
                    new_df.index = new_df.index.tz_localize(None)
                combined = pd.concat([existing_df, new_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
                price_data[symbol] = combined
            else:
                price_data[symbol] = new_df
            updated += 1
        else:
            skipped += 1

    except Exception as e:
        failed.append((symbol, str(e)[:60]))

    if (i + 1) % 50 == 0 or (i + 1) == len(symbols):
        pct = (i + 1) / len(symbols) * 100
        print(f"      [{i+1:4d}/{len(symbols)}] {pct:5.1f}% | "
              f"Updated: {updated} | Skipped: {skipped} | "
              f"Failed: {len(failed)} | "
              f"Time: {datetime.now().strftime('%H:%M:%S')}")
    time.sleep(0.5)

with open(PRICE_FILE, 'wb') as f:
    pickle.dump(price_data, f)

print(f"      Done — Updated: {updated} | Skipped: {skipped} | Failed: {len(failed)}")

# ── STEP 3: COMPUTE INDICATORS ────────────────────────────────
print(f"\n[3/9] Computing technical indicators...")

def compute_indicators(df):
    df = df.copy()
    df['EMA20']  = df['Close'].ewm(span=20,  adjust=False).mean()
    df['EMA50']  = df['Close'].ewm(span=50,  adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    delta    = df['Close'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12           = df['Close'].ewm(span=12, adjust=False).mean()
    ema26           = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line       = ema12 - ema26
    signal_line     = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD']      = macd_line
    df['Signal']    = signal_line
    df['MACD Hist'] = macd_line - signal_line
    high     = df['High']
    low      = df['Low']
    close    = df['Close']
    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr14      = tr.ewm(com=13,     adjust=False).mean()
    plus_di14  = 100 * plus_dm.ewm(com=13,  adjust=False).mean() / atr14.replace(0, np.nan)
    minus_di14 = 100 * minus_dm.ewm(com=13, adjust=False).mean() / atr14.replace(0, np.nan)
    dx         = 100 * (plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14).replace(0, np.nan)
    df['ADX']      = dx.ewm(com=13, adjust=False).mean()
    df['DI_Plus']  = plus_di14
    df['DI_Minus'] = minus_di14
    df['ATR']      = atr14
    df['Vol_MA20']  = df['Volume'].rolling(20).mean()
    df['Vol Ratio'] = df['Volume'] / df['Vol_MA20'].replace(0, np.nan)
    return df

indicator_data = {}
for symbol, df in price_data.items():
    try:
        indicator_data[symbol] = compute_indicators(df)
    except:
        pass

INDICATOR_FILE = os.path.join(PRICES_DIR, 'indicator_data_full.pkl')
with open(INDICATOR_FILE, 'wb') as f:
    pickle.dump(indicator_data, f)

print(f"     Done — {len(indicator_data)} stocks")

# ── STEP 4: BUILD ML FEATURES ─────────────────────────────────
print(f"\n[4/9] Building ML features...")

FEATURE_COLS = [
    'return_1d', 'return_5d', 'return_20d', 'return_60d',
    'dist_52w_high', 'dist_52w_low', 'atr_pct',
    'volatility_20d', 'volatility_60d',
    'vol_ratio_5d', 'vol_ratio_20d', 'obv_slope_10d', 'vol_spike',
    'rsi', 'rsi_slope_5d', 'rsi_oversold', 'rsi_overbought',
    'macd_hist', 'macd_slope_3d', 'macd_slope_5d', 'macd_cross',
    'adx', 'adx_slope', 'di_spread',
    'price_vs_ema20', 'price_vs_ema50', 'price_vs_ema200',
    'ema20_vs_ema50', 'ema50_vs_ema200',
]

def build_features(symbol, df):
    df = df.copy()
    df['return_1d']   = df['Close'].pct_change(1)
    df['return_5d']   = df['Close'].pct_change(5)
    df['return_20d']  = df['Close'].pct_change(20)
    df['return_60d']  = df['Close'].pct_change(60)
    df['52w_high']    = df['Close'].rolling(252).max()
    df['52w_low']     = df['Close'].rolling(252).min()
    df['dist_52w_high'] = (df['Close'] - df['52w_high']) / df['52w_high']
    df['dist_52w_low']  = (df['Close'] - df['52w_low'])  / df['52w_low']
    df['atr_pct']       = df['ATR'] / df['Close']
    df['volatility_20d']= df['return_1d'].rolling(20).std()
    df['volatility_60d']= df['return_1d'].rolling(60).std()
    vol_ma5             = df['Volume'].rolling(5).mean()
    vol_ma20            = df['Volume'].rolling(20).mean()
    df['vol_ratio_5d']  = df['Volume'] / vol_ma5.replace(0, np.nan)
    df['vol_ratio_20d'] = df['Volume'] / vol_ma20.replace(0, np.nan)
    df['vol_spike']     = (df['vol_ratio_5d'] > 2.0).astype(int)
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['obv_slope_10d'] = obv.diff(10) / (df['Volume'].rolling(10).mean().replace(0, np.nan))
    df['rsi']           = df['RSI']
    df['rsi_slope_5d']  = df['RSI'].diff(5)
    df['rsi_oversold']  = (df['RSI'] < 30).astype(int)
    df['rsi_overbought']= (df['RSI'] > 70).astype(int)
    df['macd_hist']     = df['MACD Hist']
    df['macd_slope_3d'] = df['MACD Hist'].diff(3)
    df['macd_slope_5d'] = df['MACD Hist'].diff(5)
    df['macd_cross']    = ((df['MACD Hist'] > 0) & (df['MACD Hist'].shift(1) <= 0)).astype(int)
    df['adx']           = df['ADX']
    df['adx_slope']     = df['ADX'].diff(5)
    df['di_spread']     = df['DI_Plus'] - df['DI_Minus']
    df['price_vs_ema20']  = (df['Close'] - df['EMA20'])  / df['EMA20']
    df['price_vs_ema50']  = (df['Close'] - df['EMA50'])  / df['EMA50']
    df['price_vs_ema200'] = (df['Close'] - df['EMA200']) / df['EMA200']
    df['ema20_vs_ema50']  = (df['EMA20'] - df['EMA50'])  / df['EMA50']
    df['ema50_vs_ema200'] = (df['EMA50'] - df['EMA200']) / df['EMA200']
    return df

all_features = {}
for symbol, df in indicator_data.items():
    try:
        all_features[symbol] = build_features(symbol, df)
    except:
        pass

print(f"     Done — {len(all_features)} stocks")

# ── STEP 5: LOAD ML MODELS ────────────────────────────────────
print(f"\n[5/9] Loading ML models...")

def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

bottom_models         = load_pkl(os.path.join(MODELS_DIR, 'bottom_models.pkl'))
bottom_encoders       = load_pkl(os.path.join(MODELS_DIR, 'bottom_encoders.pkl'))
top_models            = load_pkl(os.path.join(MODELS_DIR, 'top_models.pkl'))
top_encoders          = load_pkl(os.path.join(MODELS_DIR, 'top_encoders.pkl'))
trend_models          = load_pkl(os.path.join(MODELS_DIR, 'trend_models.pkl'))
trend_encoders        = load_pkl(os.path.join(MODELS_DIR, 'trend_encoders.pkl'))
trend_label_encoders  = load_pkl(os.path.join(MODELS_DIR, 'trend_label_encoders.pkl'))
forecast_models       = load_pkl(os.path.join(MODELS_DIR, 'forecast_models.pkl'))
forecast_encoders     = load_pkl(os.path.join(MODELS_DIR, 'forecast_encoders.pkl'))
bullish_cont_models   = load_pkl(os.path.join(MODELS_DIR, 'bullish_cont_models.pkl'))
bullish_cont_encoders = load_pkl(os.path.join(MODELS_DIR, 'bullish_cont_encoders.pkl'))
bearish_cont_models   = load_pkl(os.path.join(MODELS_DIR, 'bearish_cont_models.pkl'))
bearish_cont_encoders = load_pkl(os.path.join(MODELS_DIR, 'bearish_cont_encoders.pkl'))
symbol_group          = load_pkl(os.path.join(MODELS_DIR, 'symbol_group.pkl'))
group_stocks          = load_pkl(os.path.join(MODELS_DIR, 'group_stocks.pkl'))

print(f"     Done — models loaded for {len(group_stocks)} groups")

# ── STEP 6: RUN ML INFERENCE ──────────────────────────────────
print(f"\n[6/9] Running ML inference...")

def get_latest_features(symbol):
    df = all_features[symbol].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS)
    if len(df) == 0:
        return None
    return df.iloc[[-1]]

def get_symbol_enc(symbol, encoder):
    try:
        return encoder.transform([symbol])[0]
    except:
        return 0

def assign_ml_label(best_setup, forecast_25d):
    if best_setup == 'Momentum':
        return 'Bullish Continual' if forecast_25d >= 0 else 'Bearish Continual'
    elif best_setup == 'Reversal':
        return 'Reversal' if forecast_25d >= 0 else 'Bearish'
    elif best_setup == 'Watching':
        return 'Bullish' if forecast_25d >= 0 else 'No Signal'
    else:
        return 'No Signal'

def assign_ml_confidence(ml_label, result):
    if ml_label == 'Bullish Continual':
        return result.get('Bullish_Cont_Prob')
    elif ml_label == 'Bearish Continual':
        return result.get('Bearish_Cont_Prob')
    elif ml_label in ('Reversal', 'Bullish'):
        return result.get('Bottom_Rev_Prob')
    elif ml_label == 'Bearish':
        return result.get('Top_Rev_Prob')
    return None

def run_inference(symbol, best_setup):
    result = {
        'Symbol': symbol, 'Group': None,
        'Bottom_Rev_Prob': None, 'Top_Rev_Prob': None,
        'Bottom_Rev_Flag': False, 'Top_Rev_Flag': False,
        'Bullish_Cont_Prob': None, 'Bearish_Cont_Prob': None,
        'Forecast_25d_Pct': None, 'Forecast_45d_Pct': None,
        'Forecast_180d_Pct': None, 'Forecast_25d_Price': None,
        'Forecast_45d_Price': None, 'Forecast_180d_Price': None,
        'ML_Prediction': 'No Signal', 'ML_Confidence': None,
    }
    group = symbol_group.get(symbol)
    if group is None:
        return result
    result['Group'] = group
    latest = get_latest_features(symbol)
    if latest is None:
        return result
    feat_cols = FEATURE_COLS + ['symbol_enc']

    if group in bottom_models:
        sym_enc              = get_symbol_enc(symbol, bottom_encoders[group])
        latest['symbol_enc'] = sym_enc
        X                    = latest[feat_cols]
        bottom_prob = bottom_models[group].predict_proba(X)[0][1]
        result['Bottom_Rev_Prob'] = round(float(bottom_prob) * 100, 1)
        result['Bottom_Rev_Flag'] = bottom_prob >= 0.60
        top_prob = top_models[group].predict_proba(X)[0][1]
        result['Top_Rev_Prob'] = round(float(top_prob) * 100, 1)
        result['Top_Rev_Flag'] = top_prob >= 0.60

    if group in bullish_cont_models:
        sym_enc              = get_symbol_enc(symbol, bullish_cont_encoders[group])
        latest['symbol_enc'] = sym_enc
        X                    = latest[feat_cols]
        bc_prob = bullish_cont_models[group].predict_proba(X)[0][1]
        result['Bullish_Cont_Prob'] = round(float(bc_prob) * 100, 1)

    if group in bearish_cont_models:
        sym_enc              = get_symbol_enc(symbol, bearish_cont_encoders[group])
        latest['symbol_enc'] = sym_enc
        X                    = latest[feat_cols]
        dc_prob = bearish_cont_models[group].predict_proba(X)[0][1]
        result['Bearish_Cont_Prob'] = round(float(dc_prob) * 100, 1)

    if group in forecast_models:
        sym_enc              = get_symbol_enc(symbol, forecast_encoders[group]['25d'])
        latest['symbol_enc'] = sym_enc
        X                    = latest[feat_cols]
        current_price        = all_features[symbol]['Close'].iloc[-1]
        for horizon in ['25d', '45d', '180d']:
            if horizon not in forecast_models[group]:
                continue
            pred_return = forecast_models[group][horizon].predict(X)[0]
            pred_price  = current_price * (1 + pred_return)
            result[f'Forecast_{horizon}_Pct']  = round(float(pred_return) * 100, 1)
            result[f'Forecast_{horizon}_Price'] = round(float(pred_price), 2)

    forecast_25d            = result['Forecast_25d_Pct'] or 0
    ml_label                = assign_ml_label(best_setup, forecast_25d)
    result['ML_Prediction'] = ml_label
    result['ML_Confidence'] = assign_ml_confidence(ml_label, result)
    return result

# Run inference using last known Best Setup
try:
    old_scores = pd.read_csv(os.path.join(SCORES_DIR, 'technical_report_full.csv'))
    setup_map  = dict(zip(old_scores['Symbol'], old_scores['Best Setup']))
except:
    setup_map  = {}

all_ml_scores = []
for symbol in symbols:
    try:
        best_setup = setup_map.get(symbol, 'Watching')
        all_ml_scores.append(run_inference(symbol, best_setup))
    except:
        pass

print(f"     Done — {len(all_ml_scores)} stocks")

# ── SIMPLIFIED SCORING FUNCTIONS (no VP dependency) ──────────
def score_momentum_simple(df):
    scores = {}
    latest = df.iloc[-1]
    close  = latest['Close']
    ema20  = latest['EMA20']
    ema50  = latest['EMA50']
    ema200 = latest['EMA200']
    if close > ema20 > ema50 > ema200:  scores['EMA'] = 30
    elif close > ema50 > ema200:        scores['EMA'] = 20
    elif close > ema200:                scores['EMA'] = 10
    else:                               scores['EMA'] = 0
    rsi = latest['RSI']
    if 50 <= rsi <= 70:    scores['RSI'] = 20
    elif 45 <= rsi < 50:   scores['RSI'] = 12
    elif 70 < rsi <= 80:   scores['RSI'] = 8
    else:                  scores['RSI'] = 0
    macd_col = 'MACD_Hist' if 'MACD_Hist' in df.columns else 'MACD Hist'
    hist     = latest[macd_col]
    scores['MACD'] = 20 if hist > 0 else (10 if hist > -0.5 else 0)
    adx      = latest['ADX']
    di_plus  = latest['DI_Plus']
    di_minus = latest['DI_Minus']
    if adx > 25 and di_plus > di_minus:   scores['ADX'] = 15
    elif adx > 20 and di_plus > di_minus:  scores['ADX'] = 10
    elif adx > 25:                         scores['ADX'] = 5
    else:                                  scores['ADX'] = 0
    scores['VP'] = 5  # neutral without VP
    return sum(scores.values()), scores

def score_reversal_simple(df, lookback=20):
    scores   = {}
    macd_col = 'MACD_Hist' if 'MACD_Hist' in df.columns else 'MACD Hist'
    vol_col  = 'Vol_Ratio'  if 'Vol_Ratio'  in df.columns else 'Vol Ratio'
    if len(df) < lookback + 10:
        return 0, {}
    latest_rsi = df['RSI'].iloc[-1]
    if latest_rsi < 35:    scores['RSI_Level'] = 20
    elif latest_rsi < 45:  scores['RSI_Level'] = 10
    else:                  scores['RSI_Level'] = 0
    rsi_5d    = df['RSI'].tail(5).min()
    rsi_20d   = df['RSI'].tail(20).min()
    price_5d  = df['Close'].tail(5).min()
    price_20d = df['Close'].tail(20).min()
    scores['RSI_Divergence'] = 15 if (price_5d <= price_20d and rsi_5d > rsi_20d) else 0
    latest_hist = df[macd_col].iloc[-1]
    prev_hist   = df[macd_col].iloc[-5]
    if latest_hist > 0 and prev_hist < 0:
        scores['MACD_Cross']  = 20
        scores['MACD_Rising'] = 0
    elif latest_hist > prev_hist and latest_hist < 0:
        scores['MACD_Cross']  = 0
        scores['MACD_Rising'] = 15
    else:
        scores['MACD_Cross']  = 0
        scores['MACD_Rising'] = 0
    macd_5d  = df[macd_col].tail(5).min()
    macd_20d = df[macd_col].tail(20).min()
    scores['MACD_Divergence'] = 15 if (price_5d <= price_20d and macd_5d > macd_20d) else 0
    adx_latest = df['ADX'].iloc[-1]
    adx_10d    = df['ADX'].iloc[-10]
    if adx_latest < adx_10d and adx_latest > 20:  scores['ADX'] = 15
    elif adx_latest < adx_10d:                    scores['ADX'] = 8
    else:                                          scores['ADX'] = 0
    vol_ratio = df[vol_col].iloc[-1]
    if vol_ratio < 0.6:    scores['Volume'] = 15
    elif vol_ratio < 0.8:  scores['Volume'] = 8
    else:                  scores['Volume'] = 0
    return min(sum(scores.values()), 100), scores

# ── STEP 7: FULL TECHNICAL ANALYSIS ──────────────────────────
print(f"\n[7/9] Running technical analysis...")
print(f"      Started : {datetime.now().strftime('%H:%M:%S')}")

TECH_CHECKPOINT = os.path.join(TEMP_DIR, 'tech_checkpoint.pkl')

fund_full  = pd.read_csv(os.path.join(FUND_DIR, 'fundamental_scores_full.csv'))
prefilt_df = pd.read_csv(os.path.join(UNIVERSE_DIR, 'prefilt_passed.csv'))
fund_full  = fund_full.merge(
    prefilt_df[['Symbol', 'Market_Cap_Cr']], on='Symbol', how='left'
)

def classify_mcap(mcap_cr):
    if mcap_cr >= 20000:  return 'Large Cap'
    elif mcap_cr >= 5000: return 'Mini Large Cap'
    elif mcap_cr >= 1000: return 'Mid Cap'
    else:                 return 'Small Cap'

if os.path.exists(TECH_CHECKPOINT):
    with open(TECH_CHECKPOINT, 'rb') as f:
        ckpt = pickle.load(f)
    tech_reports = ckpt['reports']
    done_syms    = ckpt['done']
    print(f"      Resuming from checkpoint — {len(done_syms)} already done")
else:
    tech_reports = []
    done_syms    = set()

remaining = [s for s in symbols if s in indicator_data and s not in done_syms]

for i, symbol in enumerate(remaining):
    try:
        df       = indicator_data[symbol]
        latest   = df.iloc[-1]
        fund_row = fund_full[fund_full['Symbol'] == symbol]

        fund_score = float(fund_row['Final Score'].values[0]) \
                     if len(fund_row) > 0 and 'Final Score' in fund_row.columns else 50.0
        sector     = fund_row['Sector'].values[0] if len(fund_row) > 0 else 'Unknown'
        mcap_cr    = float(fund_row['Market_Cap_Cr'].values[0]) \
                     if len(fund_row) > 0 and 'Market_Cap_Cr' in fund_row.columns else 0

        macd_col = 'MACD_Hist' if 'MACD_Hist' in df.columns else 'MACD Hist'
        close    = latest['Close']
        rsi      = round(latest['RSI'],    2)
        adx      = round(latest['ADX'],    2)
        ema20    = round(latest['EMA20'],  2)
        ema50    = round(latest['EMA50'],  2)
        ema200   = round(latest['EMA200'], 2)
        di_plus  = round(latest['DI_Plus'],  2)
        di_minus = round(latest['DI_Minus'], 2)
        hist     = round(latest[macd_col], 4)
        vol_5d_avg  = df['Volume'].iloc[-5:].mean()
        vol_20d_avg = df['Volume'].iloc[-20:].mean()
        vol_r       = round(vol_5d_avg / vol_20d_avg if vol_20d_avg > 0 else 1.0, 2)
        cap_category = classify_mcap(mcap_cr)

        # Score momentum and reversal (no VP needed — use None)
        mom_score, _ = score_momentum_simple(df)
        rev_score, _ = score_reversal_simple(df)

        # EMA alignment check — Momentum requires Price > EMA50 at minimum
        latest_close = df.iloc[-1]['Close']
        latest_ema50 = df.iloc[-1]['EMA50']
        price_above_ema50 = latest_close > latest_ema50

        if mom_score >= rev_score and mom_score >= 50 and price_above_ema50:
            best_setup = 'Momentum'
            tech_score = mom_score
        elif rev_score >= mom_score and rev_score >= 50:
            best_setup = 'Reversal'
            tech_score = rev_score
        else:
            best_setup = 'Watching'
            tech_score = max(mom_score, rev_score)

        if fund_score >= 60 and tech_score >= 65 and best_setup != 'Watching':
            tier = f'TIER 1 — BUY NOW ({best_setup})'
        elif fund_score >= 60 and tech_score >= 40 and best_setup != 'Watching':
            tier = 'TIER 2 — WATCHLIST'
        else:
            tier = 'TIER 3 — WAITING'

        tech_reports.append({
            'Symbol': symbol, 'Sector': sector,
            'Fund Score': fund_score, 'Market Cap Cr': round(mcap_cr, 2),
            'Cap Category': cap_category, 'Current Price': round(close, 2),
            'RSI': rsi, 'ADX': adx, 'MACD Hist': hist,
            'Vol Ratio': vol_r, 'EMA20': ema20, 'EMA50': ema50, 'EMA200': ema200,
            'DI_Plus': di_plus, 'DI_Minus': di_minus,
            'Momentum Score': mom_score, 'Reversal Score': rev_score,
            'Best Setup': best_setup, 'Tech Score': tech_score, 'Tier': tier,
        })
        done_syms.add(symbol)

    except Exception:
        pass

    if (i + 1) % 50 == 0 or (i + 1) == len(remaining):
        pct = (i + 1) / len(remaining) * 100
        print(f"      [{i+1:4d}/{len(remaining)}] {pct:5.1f}% | "
              f"Done: {len(tech_reports)} | "
              f"Time: {datetime.now().strftime('%H:%M:%S')}")
        with open(TECH_CHECKPOINT, 'wb') as f:
            pickle.dump({'reports': tech_reports, 'done': done_syms}, f)

# Merge ML scores with correct Best Setup
tech_df      = pd.DataFrame(tech_reports)
setup_map_new = dict(zip(tech_df['Symbol'], tech_df['Best Setup']))
all_ml_scores = []
for symbol in symbols:
    try:
        best_setup = setup_map_new.get(symbol, 'Watching')
        all_ml_scores.append(run_inference(symbol, best_setup))
    except:
        pass
ml_scores_df = pd.DataFrame(all_ml_scores)

ml_merge_cols = [
    'Symbol', 'ML_Prediction', 'ML_Confidence',
    'Forecast_25d_Pct', 'Forecast_45d_Pct', 'Forecast_180d_Pct',
    'Forecast_25d_Price', 'Forecast_45d_Price', 'Forecast_180d_Price',
    'Bottom_Rev_Prob', 'Top_Rev_Prob', 'Bottom_Rev_Flag', 'Top_Rev_Flag',
    'Bullish_Cont_Prob', 'Bearish_Cont_Prob',
]
tech_final = tech_df.merge(ml_scores_df[ml_merge_cols], on='Symbol', how='left')

# Rule-based Bullish Continual (ML_Confidence=0) → rename to 'Tech Bullish'
# These were assigned by price/EMA rules, not scored by ML model
mask = (
    (tech_final['ML_Prediction'] == 'Bullish Continual') &
    (tech_final['ML_Confidence'].fillna(0) == 0)
)
tech_final.loc[mask, 'ML_Prediction'] = 'Tech Bullish'
print(f"     Tech Bullish (rule-based) : {mask.sum()} stocks")
print(f"     Bullish Continual (ML)    : "
      f"{(tech_final['ML_Prediction']=='Bullish Continual').sum()} stocks")

tech_final.to_csv(os.path.join(SCORES_DIR, 'technical_report_full.csv'), index=False)
ml_scores_df.to_csv(os.path.join(SCORES_DIR, 'ml_scores_full.csv'), index=False)

if os.path.exists(TECH_CHECKPOINT):
    os.remove(TECH_CHECKPOINT)

print(f"     Done — {len(tech_final)} stocks analysed")
print(f"     Tier distribution:")
for tier, cnt in tech_final['Tier'].value_counts().items():
    print(f"       {tier}: {cnt}")

# ── STEP 8: WEEKLY REPORT DATA PREP ──────────────────────────
print(f"\n[8/9] Preparing weekly report data...")

tech_df   = pd.read_csv(os.path.join(SCORES_DIR, 'technical_report_full.csv'))
fund_full = pd.read_csv(os.path.join(FUND_DIR, 'fundamental_scores_full.csv'))
fund_full = fund_full.merge(
    prefilt_df[['Symbol', 'Market_Cap_Cr']], on='Symbol', how='left'
)

def classify_mcap_score(mcap_cr):
    if mcap_cr >= 20000:  return 'Large Cap'
    elif mcap_cr >= 5000: return 'Mini Large Cap'
    elif mcap_cr >= 1000: return 'Mid Cap'
    else:                 return 'Small Cap'

fund_full['Cap Category'] = fund_full['Market_Cap_Cr'].apply(
    lambda x: classify_mcap_score(x or 0)
)

fund_full['Sector Score'] = 0.0
for sector in fund_full['Sector'].unique():
    mask      = fund_full['Sector'] == sector
    max_score = fund_full[mask]['Final Score'].max()
    if max_score > 0:
        fund_full.loc[mask, 'Sector Score'] = (
            fund_full[mask]['Final Score'] / max_score * 10
        ).round(1)

fund_full['Cap Score'] = 0.0
for sector in fund_full['Sector'].unique():
    for cap in ['Large Cap', 'Mini Large Cap', 'Mid Cap', 'Small Cap']:
        mask   = (fund_full['Sector'] == sector) & (fund_full['Cap Category'] == cap)
        subset = fund_full[mask]
        if len(subset) == 0:
            continue
        max_score = subset['Final Score'].max()
        if max_score > 0:
            fund_full.loc[mask, 'Cap Score'] = (
                subset['Final Score'] / max_score * 10
            ).round(1)

for col in ['Sector Score', 'Cap Score', 'Sector Score_x', 'Cap Score_x',
            'Sector Score_y', 'Cap Score_y']:
    if col in tech_df.columns:
        tech_df = tech_df.drop(columns=[col])

tech_df = tech_df.merge(
    fund_full[['Symbol', 'Sector Score', 'Cap Score',
               'Final Score', 'Market_Cap_Cr']],
    on='Symbol', how='left'
)
# Rename for consistency with aggregation function
if 'Final Score' in tech_df.columns:
    tech_df['Fund Score'] = tech_df['Final Score']
if 'Market_Cap_Cr' in tech_df.columns:
    tech_df['Market Cap Cr'] = tech_df['Market Cap Cr'].fillna(
        tech_df['Market_Cap_Cr'])

# Rank jumpers
SCORES_FILE  = os.path.join(SCORES_DIR, 'last_week_scores.csv')
rank_jumpers = set()
current_scores = tech_df[['Symbol', 'Sector Score', 'Cap Score']].copy()
if os.path.exists(SCORES_FILE):
    last_scores   = pd.read_csv(SCORES_FILE)
    merged_scores = current_scores.merge(
        last_scores[['Symbol', 'Sector Score', 'Cap Score']],
        on='Symbol', how='left', suffixes=('_now', '_prev')
    )
    for _, row in merged_scores.iterrows():
        prev_sec = row.get('Sector Score_prev', 0) or 0
        prev_cap = row.get('Cap Score_prev',    0) or 0
        curr_sec = row.get('Sector Score_now',  0) or 0
        curr_cap = row.get('Cap Score_now',     0) or 0
        sec_jump = (curr_sec - prev_sec) / prev_sec * 100 if prev_sec > 0 else 0
        cap_jump = (curr_cap - prev_cap) / prev_cap * 100 if prev_cap > 0 else 0
        if sec_jump >= 10 or cap_jump >= 10:
            rank_jumpers.add(row['Symbol'])
current_scores.to_csv(SCORES_FILE, index=False)
print(f"      Rank jumpers: {len(rank_jumpers)}")

# Vol 5D ratio
vol_5d_data = {}
for symbol, df in price_data.items():
    try:
        if len(df) < 20:
            vol_5d_data[symbol] = 1.0
            continue
        vol_5d_avg      = df['Volume'].iloc[-5:].mean()
        vol_prior15_avg = df['Volume'].iloc[-20:-5].mean()
        vol_5d_data[symbol] = round(
            vol_5d_avg / vol_prior15_avg if vol_prior15_avg > 0 else 1.0, 2)
    except:
        vol_5d_data[symbol] = 1.0
tech_df['Vol 5D Ratio'] = tech_df['Symbol'].map(vol_5d_data).fillna(1.0)


# Volume inference
def get_volume_inference(row):
    vol   = float(row.get('Vol 5D Ratio', 1.0))
    setup = str(row['Best Setup'])
    adx   = float(row['ADX'])
    if   vol >= 3.0: vol_label = 'Extremely High'
    elif vol >= 2.0: vol_label = 'Very High'
    elif vol >= 1.5: vol_label = 'High'
    elif vol >= 1.0: vol_label = 'Normal'
    elif vol >= 0.7: vol_label = 'Low'
    else:            vol_label = 'Very Low'
    if setup == 'Momentum':
        if   vol >= 2.0: inf = 'Strong participation → trend likely to continue'
        elif vol >= 1.5: inf = 'Good volume support → momentum confirmed'
        elif vol >= 1.0: inf = 'Average volume → monitor for increase'
        elif vol >= 0.7: inf = 'Weak volume → momentum may fade, wait'
        else:            inf = 'Very low volume → no conviction, caution'
    elif setup == 'Reversal':
        if   vol <= 0.5: inf = 'Sellers exhausted → reversal likely'
        elif vol <= 0.7: inf = 'Low volume on down move → selling pressure easing'
        elif vol <= 1.0: inf = 'Neutral volume → wait for low-volume base to form'
        elif vol <= 1.5: inf = 'Elevated volume on reversal → watch direction'
        else:            inf = 'High volume → could be capitulation or distribution'
    else:
        if   adx > 40 and vol >= 2.0: inf = 'Above avg volume but no setup → watch closely'
        elif adx > 40:                inf = 'Strong downtrend → no entry signal'
        elif vol <= 0.5:              inf = 'Very low volume → no interest, wait'
        elif vol >= 1.5 and adx < 35: inf = 'Above avg volume but no setup → watch'
        elif vol >= 1.0:              inf = 'Normal activity → no setup forming yet'
        else:                         inf = 'Low activity → no setup forming yet'
    return vol_label, inf

tech_df['Vol Label']     = ''
tech_df['Vol Inference'] = ''
for idx, row in tech_df.iterrows():
    label, inf = get_volume_inference(row)
    tech_df.at[idx, 'Vol Label']     = label
    tech_df.at[idx, 'Vol Inference'] = inf

# Sector trend
# Primary sector index map — best available ETF/index per sector
# Sectors with no dedicated ETF fall through to stock aggregation
# Only sectors with a DEDICATED, ACCURATE index or ETF
# Removed wrong mappings:
#   Power/Utilities → ^CNXENERGY is Oil & Gas dominated (ONGC/Reliance heavy)
#   Construction Materials → ^CNXCMDT is Commodities index, NOT cement stocks
#   Telecom/Media/Textiles/Forest → MID150BEES is meaningless proxy
# These sectors use stock aggregation which is always accurate (stocks ARE the index)
SECTOR_INDEX_MAP = {
    'Information Technology'            : '^CNXIT',
    'Healthcare'                        : '^CNXPHARMA',
    'Financial Services'                : 'FINIETF.NS',
    'Capital Goods'                     : '^CNXINFRA',
    'Consumer Durables'                 : '^CNXCONSUM',
    'Oil, Gas & Consumable Fuels'       : '^CNXENERGY',
    'Automobile and Auto Components'    : '^CNXAUTO',
    'Banking'                           : '^NSEBANK',
    'Fast Moving Consumer Goods'        : '^CNXFMCG',
    'Metals & Mining'                   : '^CNXMETAL',
    'Realty'                            : '^CNXREALTY',
    'Services'                          : '^CNXSERVICE',
    'Construction'                      : '^CNXINFRA',   # infra proxy — acceptable
    'Pharmaceuticals'                   : 'PHARMABEES.NS',
    'Chemicals'                         : 'MOGSEC.NS',
    'Consumer Services'                 : 'CONSUMBEES.NS',
    'Diversified'                       : 'JUNIORBEES.NS',
    # All other sectors → stock aggregation (no dedicated index)
}

print(f"      Fetching sector index data...")
SECTOR_INDEX_FILE = os.path.join(SCORES_DIR, 'sector_index_data.pkl')

if os.path.exists(SECTOR_INDEX_FILE):
    # Incremental — load existing and append last 7 days only
    print(f"      Existing sector index found — incremental update (7 days)...")
    with open(SECTOR_INDEX_FILE, 'rb') as f:
        ticker_data = pickle.load(f)
    for ticker in set(SECTOR_INDEX_MAP.values()):
        try:
            new_df = yf.download(ticker, period='7d', interval='1d',
                                 progress=False, auto_adjust=True)
            if isinstance(new_df.columns, pd.MultiIndex):
                new_df.columns = [col[0] for col in new_df.columns]
            if len(new_df) == 0:
                continue
            if ticker in ticker_data:
                combined = pd.concat([ticker_data[ticker], new_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
                cutoff = combined.index[-1] - pd.DateOffset(years=5)
                ticker_data[ticker] = combined[combined.index >= cutoff]
            else:
                ticker_data[ticker] = new_df
        except:
            pass
    print(f"      Incremental update done — {len(ticker_data)} tickers")
else:
    # First run — download full 5Y history
    print(f"      First run — downloading full 5Y history (one-time, ~3 mins)...")
    ticker_data = {}
    for ticker in set(SECTOR_INDEX_MAP.values()):
        try:
            df = yf.download(ticker, period='5y', interval='1d',
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            if len(df) > 50:
                ticker_data[ticker] = df
        except:
            pass
    print(f"      Full 5Y download done — {len(ticker_data)} tickers")

# Save updated sector index data for run_sector_analysis.py
with open(SECTOR_INDEX_FILE, 'wb') as f:
    pickle.dump(ticker_data, f)
print(f"      Sector index saved: {SECTOR_INDEX_FILE}")

# Assign index data only for sectors in SECTOR_INDEX_MAP
# All other sectors use stock aggregation (get_sector_trend_from_stocks)
sector_price = {}
for sector, ticker in SECTOR_INDEX_MAP.items():
    if ticker in ticker_data:
        sector_price[sector] = ticker_data[ticker]

def calculate_sector_indicators(df):
    data = df.copy()
    data['EMA20']  = data['Close'].ewm(span=20,  adjust=False).mean()
    data['EMA50']  = data['Close'].ewm(span=50,  adjust=False).mean()
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
    delta    = data['Close'].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    rs       = gain.ewm(span=14, adjust=False).mean() / (
                loss.ewm(span=14, adjust=False).mean() + 1e-9)
    data['RSI']       = 100 - (100 / (1 + rs))
    ema12             = data['Close'].ewm(span=12, adjust=False).mean()
    ema26             = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD']      = ema12 - ema26
    data['Signal']    = data['MACD'].ewm(span=9,  adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['Signal']
    high     = data['High']
    low      = data['Low']
    close    = data['Close']
    tr       = pd.concat([high - low, (high - close.shift()).abs(),
                          (low - close.shift()).abs()], axis=1).max(axis=1)
    dm_plus  = high.diff()
    dm_minus = -low.diff()
    dm_plus  = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    atr      = tr.ewm(span=14, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(span=14, adjust=False).mean() / (atr + 1e-9)
    di_minus = 100 * dm_minus.ewm(span=14, adjust=False).mean() / (atr + 1e-9)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    data['ADX'] = dx.ewm(span=14, adjust=False).mean()
    return data

# Sectors with real unique index — use index data
# All sectors with a dedicated index or ETF — try index first, fall to stocks if download fails
# Sectors that use index data (must match SECTOR_INDEX_MAP keys exactly)
REAL_INDEX_SECTORS = {
    'Automobile and Auto Components',  # ^CNXAUTO
    'Banking',                         # ^NSEBANK
    'Capital Goods',                   # ^CNXINFRA
    'Consumer Durables',               # ^CNXCONSUM
    'Fast Moving Consumer Goods',      # ^CNXFMCG
    'Financial Services',              # FINIETF.NS
    'Healthcare',                      # ^CNXPHARMA
    'Information Technology',          # ^CNXIT
    'Metals & Mining',                 # ^CNXMETAL
    'Oil, Gas & Consumable Fuels',     # ^CNXENERGY
    'Realty',                          # ^CNXREALTY
    'Services',                        # ^CNXSERVICE
    'Construction',                    # ^CNXINFRA (acceptable proxy)
    'Pharmaceuticals',                 # PHARMABEES.NS
    'Chemicals',                       # MOGSEC.NS
    'Consumer Services',               # CONSUMBEES.NS
    'Diversified',                     # JUNIORBEES.NS
    # Removed: Power, Utilities, Construction Materials → stock aggregation
    # Removed: Telecom, Media, Textiles, Forest → stock aggregation
}

def derive_trend_label(rsi, adx, above_ema200, above_ema50,
                       ema_aligned, ema_bearish, macd_bull, macd_bear):
    """Shared trend label logic used by both index and stock aggregation."""
    bull_score = sum([above_ema200, above_ema50, ema_aligned, macd_bull, rsi > 55])
    bear_score = sum([not above_ema200, not above_ema50,
                      ema_bearish, macd_bear, rsi < 45])
    if   bull_score >= 5 and adx > 25:  label = 'Strong Uptrend ↑↑'
    elif bull_score >= 4 and adx > 20:  label = 'Uptrend ↑'
    elif bull_score >= 3 and adx <= 20: label = 'Weak Uptrend →↑'
    elif bull_score == bear_score:      label = 'Sideways →'
    elif bear_score >= 5 and adx > 25:  label = 'Strong Downtrend ↓↓'
    elif bear_score >= 4 and adx > 20:  label = 'Downtrend ↓'
    elif bear_score >= 3 and adx <= 20: label = 'Weak Downtrend →↓'
    else:                               label = 'Sideways →'
    return label

def get_sector_trend_from_index(sector):
    """Use real sector index data — for 13 sectors with unique index."""
    if sector not in sector_price:
        return None, None
    df     = calculate_sector_indicators(sector_price[sector])
    latest = df.iloc[-1]
    close    = latest['Close']
    rsi      = round(latest['RSI'],   1)
    adx      = round(latest['ADX'],   1)
    ema20    = latest['EMA20']
    ema50    = latest['EMA50']
    ema200   = latest['EMA200']
    macd     = latest['MACD']
    signal   = latest['Signal']
    macd_h   = latest['MACD_Hist']
    above_ema200 = close > ema200
    above_ema50  = close > ema50
    ema_aligned  = ema20 > ema50 > ema200
    ema_bearish  = ema20 < ema50 < ema200
    macd_bull    = macd > signal and macd_h > 0
    macd_bear    = macd < signal and macd_h < 0
    label    = derive_trend_label(rsi, adx, above_ema200, above_ema50,
                                  ema_aligned, ema_bearish, macd_bull, macd_bear)
    ema_str  = '↑↑↑' if ema_aligned else '↓↓↓' if ema_bearish else 'mixed'
    macd_str = '▲' if macd_bull else '▼' if macd_bear else '~'
    detail   = (f"RSI {rsi} | ADX {adx} | MACD {macd_str} | "
                f"EMA {ema_str} | {'Above' if above_ema200 else 'Below'} EMA200"
                f" [idx]")
    return label, detail

# Sector name aliases — handle variants between fund_df and tech_df
SECTOR_ALIASES = {
    'Media Entertainment & Publication' : 'Media, Entertainment & Publication',
    'Media, Entertainment & Publication': 'Media Entertainment & Publication',
}

def get_sector_trend_from_stocks(sector, df):
    """Aggregate from top 25 stocks by MCap — for 13 sectors without unique index."""
    sector_stocks = df[df['Sector'] == sector].copy()
    # Try alias if no stocks found
    if len(sector_stocks) == 0:
        alias = SECTOR_ALIASES.get(sector)
        if alias:
            sector_stocks = df[df['Sector'] == alias].copy()
    if len(sector_stocks) == 0:
        return 'No data', 'Not in quality universe'

    # Resolve MCap column name
    mcap_col = 'Market Cap Cr' if 'Market Cap Cr' in df.columns else \
               'Market_Cap_Cr' if 'Market_Cap_Cr' in df.columns else None

    # Adaptive selection: Sector Score OR Cap Score >= 4, top 25 by MCap
    quality = sector_stocks[
        (sector_stocks['Sector Score'].fillna(0) >= 4) |
        (sector_stocks['Cap Score'].fillna(0)    >= 4)
    ].copy()
    if len(quality) < 5:
        # Relax — use all stocks in sector regardless of rank
        quality = sector_stocks.copy()
    if len(quality) == 0:
        return 'No data', '—'

    if mcap_col and len(quality) > 0:
        quality = quality.nlargest(25, mcap_col)
    else:
        quality = quality.head(25)
    n       = len(quality)

    # Aggregate indicators
    rsi         = round(quality['RSI'].median(), 1)
    adx         = round(quality['ADX'].median(), 1)
    macd_bull_pct  = (quality['MACD Hist'] > 0).mean()
    above_ema200   = (quality['Current Price'] > quality['EMA200']).mean()
    above_ema50    = (quality['Current Price'] > quality['EMA50']).mean()
    ema_aligned_pct= ((quality['EMA20'] > quality['EMA50']) &
                      (quality['EMA50'] > quality['EMA200'])).mean()
    ema_bearish_pct= ((quality['EMA20'] < quality['EMA50']) &
                      (quality['EMA50'] < quality['EMA200'])).mean()

    # Convert to booleans using 50% threshold
    ab_ema200   = above_ema200   >= 0.5
    ab_ema50    = above_ema50    >= 0.5
    ema_aligned = ema_aligned_pct >= 0.5
    ema_bearish = ema_bearish_pct >= 0.5
    macd_bull   = macd_bull_pct  >= 0.5
    macd_bear   = macd_bull_pct  <  0.5

    label    = derive_trend_label(rsi, adx, ab_ema200, ab_ema50,
                                  ema_aligned, ema_bearish, macd_bull, macd_bear)
    ema_str  = '↑↑↑' if ema_aligned else '↓↓↓' if ema_bearish else 'mixed'
    macd_str = '▲'   if macd_bull   else '▼'
    data_note = f'limited:{n}' if n < 10 else f'{n}stks'
    detail   = (f"RSI {rsi} | ADX {adx} | MACD {macd_str} | "
                f"EMA {ema_str} | {'Above' if ab_ema200 else 'Below'} EMA200"
                f" [{data_note}]")
    return label, detail

def get_sector_trend(sector, df=None):
    """Route to index or stock aggregation based on sector."""
    if sector in REAL_INDEX_SECTORS:
        label, detail = get_sector_trend_from_index(sector)
        if label is not None:
            return label, detail
        # Index failed — fall through to stock aggregation
    # Stock aggregation for all other sectors (or index fallback)
    if df is not None:
        return get_sector_trend_from_stocks(sector, df)
    return 'No data', '—' 

tech_df['Sector Trend']  = ''
tech_df['Sector Detail'] = ''
for idx, row in tech_df.iterrows():
    s_label, s_detail = get_sector_trend(str(row['Sector']), tech_df)
    tech_df.at[idx, 'Sector Trend']  = s_label
    tech_df.at[idx, 'Sector Detail'] = s_detail


today_str = datetime.now().strftime('%Y-%m-%d')

tech_df.to_csv(os.path.join(SCORES_DIR, 'technical_report_full.csv'), index=False)
print(f"     Done — {len(tech_df)} rows, {len(tech_df.columns)} cols")

# ── STEP 9: GENERATE REPORTS ──────────────────────────────────
print(f"\n[9/9] Generating reports...")

today_file = datetime.now().strftime('%Y%m%d')
CAP_ORDER  = ['Large Cap', 'Mini Large Cap', 'Mid Cap', 'Small Cap']

def get_reason(row):
    setup  = str(row['Best Setup'])
    rsi    = float(row['RSI'])
    adx    = float(row['ADX'])
    macd   = float(row['MACD Hist'])
    price  = float(row['Current Price'])
    ema20  = float(row['EMA20'])
    ema50  = float(row['EMA50'])
    ema200 = float(row['EMA200'])
    above_ema200 = price > ema200
    ema_aligned  = ema20 > ema50 > ema200
    if setup == 'Momentum':
        if ema_aligned and rsi > 55:      return f'Strong uptrend, EMA aligned, RSI {rsi:.0f}'
        elif above_ema200 and rsi > 50:   return f'Momentum building, above EMA200, RSI {rsi:.0f}'
        elif above_ema200 and rsi <= 50:  return f'Consolidating above EMA200, RSI {rsi:.0f}'
        elif not above_ema200 and rsi>60: return f'Short bounce, below EMA200, RSI {rsi:.0f}'
        else:                             return f'Weak momentum, below EMA200, RSI {rsi:.0f}'
    elif setup == 'Reversal':
        if rsi < 30:   return f'Deeply oversold RSI {rsi:.0f}, reversal imminent'
        elif rsi < 35: return f'Oversold RSI {rsi:.0f}, watch for turn'
        elif macd > 0: return f'RSI {rsi:.0f}, MACD turning up'
        else:          return f'Divergence forming, RSI {rsi:.0f}'
    else:
        if adx > 40:   return f'Strong downtrend ADX {adx:.0f}, wait'
        elif adx > 25: return f'Downtrend ADX {adx:.0f}, no setup yet'
        elif rsi < 30: return f'Deeply oversold RSI {rsi:.0f}, watch for turn'
        elif rsi < 40: return f'Oversold RSI {rsi:.0f}, watch for turn'
        else:          return f'RSI {rsi:.0f}, setup not confirmed'

def get_breakout_vol_inference(week_vol, break_vol):
    if   break_vol >= 2.0: return f"Week Vol:{week_vol:.2f}x | Break Vol:{break_vol:.2f}x ⚡⚡ → Strong breakout volume building"
    elif break_vol >= 1.5: return f"Week Vol:{week_vol:.2f}x | Break Vol:{break_vol:.2f}x ⚡  → Volume building toward breakout"
    elif break_vol >= 1.0: return f"Week Vol:{week_vol:.2f}x | Break Vol:{break_vol:.2f}x ~  → Normal volume, no breakout pressure"
    elif break_vol >= 0.7: return f"Week Vol:{week_vol:.2f}x | Break Vol:{break_vol:.2f}x    → Quiet, below consolidation avg"
    else:                  return f"Week Vol:{week_vol:.2f}x | Break Vol:{break_vol:.2f}x    → Very quiet, no breakout interest"

def passes_rank_filter(row):
    return (float(row.get('Cap Score',    0) or 0) >= 7 or
            float(row.get('Sector Score', 0) or 0) >= 7)

def cap_order(x):
    return CAP_ORDER.index(x) if x in CAP_ORDER else 99

def mcap_str(mcap_cr):
    try:
        v = float(mcap_cr)
        if v >= 100000: return f"Rs{v/100000:.1f}L Cr"
        return f"Rs{v:,.0f}Cr"
    except:
        return "Rs—"

def tier_abbr(tier):
    return (str(tier)
            .replace('TIER 1 — BUY NOW (Reversal)', 'T1-Rev')
            .replace('TIER 1 — BUY NOW (Momentum)', 'T1-Mom')
            .replace('TIER 1 — BREAKOUT IMMINENT',  'T1-Brkout')
            .replace('TIER 2 — WATCHLIST',           'T2')
            .replace('TIER 2 — BASE BUILDING',       'T2B')
            .replace('TIER 3 — WAITING',             'T3'))

# Data slices
tier1_mom = tech_df[
    tech_df['Tier'].str.startswith('TIER 1') &
    (tech_df['Best Setup'] == 'Momentum')
].copy()
tier1_mom['_cap_order'] = tier1_mom['Cap Category'].apply(cap_order)
tier1_mom = tier1_mom.sort_values(
    ['_cap_order', 'Fund Score'], ascending=[True, False]
).reset_index(drop=True)

tier2a = tech_df[tech_df['Tier'] == 'TIER 2 — WATCHLIST'].copy()
tier2a = tier2a[tier2a.apply(passes_rank_filter, axis=1) |
                tier2a['Symbol'].isin(rank_jumpers)
               ].sort_values('Fund Score', ascending=False).head(25).reset_index(drop=True)

tier3_mom = tech_df[
    (tech_df['Tier'] == 'TIER 3 — WAITING') &
    (tech_df['Best Setup'] == 'Momentum')
].copy()
tier3_mom['_cap_order'] = tier3_mom['Cap Category'].apply(cap_order)
tier3_mom = tier3_mom[tier3_mom.apply(passes_rank_filter, axis=1) |
                      tier3_mom['Symbol'].isin(rank_jumpers)
                     ].sort_values(
    ['_cap_order', 'Fund Score'], ascending=[True, False]
).reset_index(drop=True)

all_reversal    = tech_df[tech_df['Best Setup'] == 'Reversal'].copy()
reversal_by_cap = {}
for cap in CAP_ORDER:
    subset = all_reversal[all_reversal['Cap Category'] == cap].copy()
    reversal_by_cap[cap] = subset.sort_values('Reversal Score', ascending=False).head(10)

def generate_tech_report(is_short):
    lines  = []
    serial = 1

    def p(line=''):
        lines.append(str(line))

    mode_label  = 'SHORT' if is_short else 'LONG'
    today_label = datetime.now().strftime('%d %B %Y')

    p(f"{'='*78}")
    p(f"  AI STOCK SCREENER — WEEKLY REPORT ({mode_label})")
    p(f"  {today_label}  |  Universe: {len(tech_df)} stocks")
    p(f"{'='*78}")
    p(f"  SecRank  = rank vs same sector  |  CapRank = rank vs same sector + cap")
    p(f"  Week Vol = this week avg vs prior 15-day avg")
    p(f"  ↑ = jumped 10%+ in rank vs last week")

    # Sector summary — both short and long
    p(f"\n{'─'*78}")
    p(f"  SECTOR TREND SUMMARY")
    p(f"{'─'*78}")
    p(f"  {'Sector':42} {'Trend':28} Detail")
    p(f"  {'─'*76}")
    for sector in sorted(SECTOR_INDEX_MAP.keys()):
        label, detail = get_sector_trend(sector, tech_df)
        p(f"  {sector:42} {label:28} {detail}")

    # Tier 1 Momentum
    p(f"\n{'─'*78}")
    p(f"  TIER 1 — MOMENTUM  ({len(tier1_mom)} stocks)")
    p(f"  Fund >=60 + Tech >=65 + Momentum setup confirmed")
    p(f"{'─'*78}")
    current_cap = None
    for _, row in tier1_mom.iterrows():
        cap = row['Cap Category']
        if cap != current_cap:
            current_cap = cap
            cap_short   = {'Large Cap':'L','Mini Large Cap':'ML',
                           'Mid Cap':'M','Small Cap':'S'}.get(cap,'?')
            p(f"\n  [{cap_short}] {cap}")
            p(f"  {'─'*68}")
        consol_line = ''
        jump_tag = ' ↑' if row['Symbol'] in rank_jumpers else ''
        p(f"""
  #{serial}  {row['Symbol']}{jump_tag}  ({row['Sector']})  MCap {mcap_str(row['Market Cap Cr'])}  Rs{row['Current Price']:.2f}
    Setup   : {row['Best Setup']:10}  Tech {row['Tech Score']:3.0f}/100
    Scores  : Fund {row['Fund Score']}/100  Sector Rank {row['Sector Score']}/10  Cap Rank {row['Cap Score']}/10{consol_line}
    Tech    : RSI {row['RSI']:.0f}  ADX {row['ADX']:.0f}  MACD {row['MACD Hist']:+.2f}  → {get_reason(row)}
    Volume  : Week Vol:{row['Vol 5D Ratio']:.2f}x ({row['Vol Label']}) → {row['Vol Inference']}
    Sector  : {row['Sector Trend']} | {row['Sector Detail']}""")
        serial += 1

    # Tier 2A
    p(f"\n{'─'*78}")
    p(f"  TIER 2A — WATCHLIST  (Top 25 | Cap/Sec Rank >=7 or ↑)")
    p(f"  Setup forming — wait for confirmation before entering")
    p(f"{'─'*78}")
    for _, row in tier2a.iterrows():
        consol_tag = ''
        jump_tag   = ' ↑' if row['Symbol'] in rank_jumpers else ''
        p(f"\n  #{serial}  {row['Symbol']}{jump_tag}  "
          f"Rs{row['Current Price']:.2f}  "
          f"MCap {mcap_str(row['Market Cap Cr'])}  "
          f"Fund:{row['Fund Score']:4.1f}  "
          f"Sec:{row['Sector Score']:4.1f}  "
          f"Cap:{row['Cap Score']:4.1f}{consol_tag}")
        p(f"      Technical : [{row['Best Setup']}|{row['Tech Score']:.0f}]  "
          f"RSI:{row['RSI']:3.0f}  ADX:{row['ADX']:3.0f}  "
          f"MACD:{round(row['MACD Hist']):+d}  → {get_reason(row)}")
        p(f"      Volume    : Week Vol:{row['Vol 5D Ratio']:.2f}x "
          f"({row['Vol Label']}) → {row['Vol Inference']}")
        p(f"      Sector    : {row['Sector Trend']} | {row['Sector Detail']}")
        serial += 1


    # Tier 3 Momentum (long only)
    if not is_short:
        p(f"\n{'─'*78}")
        p(f"  TIER 3 — MOMENTUM WAITING  ({len(tier3_mom)} shown | Cap/Sec Rank >=7 or ↑)")
        p(f"  Good businesses with momentum — setup not yet confirmed")
        p(f"{'─'*78}")
        current_cap = None
        for _, row in tier3_mom.iterrows():
            cap = row['Cap Category']
            if cap != current_cap:
                current_cap = cap
                cap_short   = {'Large Cap':'L','Mini Large Cap':'ML',
                               'Mid Cap':'M','Small Cap':'S'}.get(cap,'?')
                p(f"\n  [{cap_short}] {cap}")
                p(f"  {'─'*68}")
            jump_tag = ' ↑' if row['Symbol'] in rank_jumpers else ''
            vol_line = (f"Week Vol:{row['Vol 5D Ratio']:.2f}x "
                        f"({row['Vol Label']}) → {row['Vol Inference']}")
            p(f"\n  #{serial}  {row['Symbol']}{jump_tag}  "
              f"Rs{row['Current Price']:.2f}  "
              f"MCap {mcap_str(row['Market Cap Cr'])}  "
              f"Fund:{row['Fund Score']:4.1f}  "
              f"Sec:{row['Sector Score']:4.1f}  "
              f"Cap:{row['Cap Score']:4.1f}")
            p(f"      Technical : RSI:{row['RSI']:3.0f}  ADX:{row['ADX']:3.0f}  "
              f"MACD:{round(row['MACD Hist']):+d}  → {get_reason(row)}")
            p(f"      Volume    : {vol_line}")
            p(f"      Sector    : {row['Sector Trend']} | {row['Sector Detail']}")
            serial += 1



    # Reversal candidates
    p(f"\n{'─'*78}")
    p(f"  BULLISH REVERSAL CANDIDATES  (Top 10 per cap category)")
    p(f"  Oversold + divergence forming — wait for confirmation")
    p(f"{'─'*78}")
    for cap in CAP_ORDER:
        subset = reversal_by_cap.get(cap, pd.DataFrame())
        if len(subset) == 0:
            continue
        cap_short = {'Large Cap':'L','Mini Large Cap':'ML',
                     'Mid Cap':'M','Small Cap':'S'}.get(cap,'?')
        p(f"\n  [{cap_short}] {cap}")
        p(f"  {'─'*68}")
        for _, row in subset.iterrows():
            jump_tag = ' ↑' if row['Symbol'] in rank_jumpers else ''
            p(f"\n  #{serial}  {row['Symbol']}{jump_tag}  "
              f"Rs{row['Current Price']:.2f}  "
              f"MCap {mcap_str(row['Market Cap Cr'])}  "
              f"[{tier_abbr(row['Tier'])}]  "
              f"Fund:{row['Fund Score']:4.1f}  "
              f"Rev:{row['Reversal Score']:.0f}  "
              f"Sec:{row['Sector Score']:4.1f}  "
              f"Cap:{row['Cap Score']:4.1f}")
            p(f"      Tech    : RSI:{row['RSI']:3.0f}  ADX:{row['ADX']:3.0f}  "
              f"MACD:{round(row['MACD Hist']):+d}  → {get_reason(row)}")
            p(f"      Volume  : Week Vol:{row['Vol 5D Ratio']:.2f}x "
              f"({row['Vol Label']}) → {row['Vol Inference']}")
            p(f"      Sector  : {row['Sector Trend']} | {row['Sector Detail']}")
            serial += 1

    # Rank jumpers
    if rank_jumpers:
        p(f"\n{'─'*78}")
        p(f"  ↑ RANK JUMPERS THIS WEEK  ({len(rank_jumpers)} stocks)")
        p(f"{'─'*78}")
        jumper_df = tech_df[tech_df['Symbol'].isin(rank_jumpers)].sort_values(
            'Fund Score', ascending=False)
        for _, row in jumper_df.iterrows():
            p(f"  {row['Symbol']:12}  Rs{row['Current Price']:.2f}  "
              f"MCap {mcap_str(row['Market Cap Cr'])}  "
              f"Fund:{row['Fund Score']:4.1f}  "
              f"Sec:{row['Sector Score']:4.1f}  "
              f"Cap:{row['Cap Score']:4.1f}  "
              f"{tier_abbr(row['Tier'])}  {row['Sector']}")

    p(f"\n{'─'*78}")
    p(f"  L=Large Cap  ML=Mini Large Cap  M=Mid Cap  S=Small Cap")
    p(f"  SecRank  = rank vs same sector | CapRank = rank vs same sector + cap")
    p(f"  Week Vol = this week avg vs prior 15-day avg")
    p(f"  ↑ = jumped 10%+ in Cap or Sector rank vs last week")
    p(f"{'─'*78}")
    return lines

# ML report generator
ml_scores_df  = pd.read_csv(os.path.join(SCORES_DIR, 'ml_scores_full.csv'))
tech_for_ml   = pd.read_csv(os.path.join(SCORES_DIR, 'technical_report_full.csv'))

CONF_THRESHOLD = {
    'Bullish Continual': 40.0,
    'Tech Bullish'     :  0.0,  # no confidence threshold — rule-based
    'Bullish'          : 50.0,
    'Reversal'         : 50.0,
}

buy_labels = ['Bullish Continual', 'Tech Bullish', 'Bullish', 'Reversal']

filtered = tech_for_ml[
    tech_for_ml['ML_Prediction'].isin(buy_labels) &
    (
        tech_for_ml['ML_Confidence'].notna() |
        (tech_for_ml['ML_Prediction'] == 'Tech Bullish')
    )
].copy()
filtered = filtered[
    filtered.apply(
        lambda r: r['ML_Confidence'] >= CONF_THRESHOLD.get(r['ML_Prediction'], 50),
        axis=1
    )
].copy()
filtered = filtered.sort_values('ML_Confidence', ascending=False).reset_index(drop=True)

def get_tech_trend(row):
    try:
        price  = float(row['Current Price'])
        ema20  = float(row['EMA20'])
        ema50  = float(row['EMA50'])
        ema200 = float(row['EMA200'])
        adx    = float(row['ADX'])
        rsi    = float(row['RSI'])
        macd   = float(row['MACD Hist'])
        vol    = float(row.get('Vol 5D Ratio', row.get('Vol Ratio', 1.0)))
        full_bull_align = price > ema20 > ema50 > ema200
        partial_bull    = price > ema50 > ema200
        full_bear_align = price < ema20 and ema20 < ema50 and ema50 < ema200
        partial_bear    = price < ema200 and ema20 < ema50
        strong_trend    = adx > 25
        mild_trend      = adx > 20
        bull_momentum   = rsi > 55 and macd > 0
        bear_momentum   = rsi < 45 and macd < 0
        high_vol        = vol > 1.0
        if full_bull_align and strong_trend and bull_momentum and high_vol:
            return 'Str-Up'
        elif partial_bull and (mild_trend or rsi > 50 or macd > 0):
            return 'Up'
        elif full_bear_align and strong_trend and bear_momentum and high_vol:
            return 'Str-Dn'
        elif partial_bear and (mild_trend or macd < 0):
            return 'Down'
        else:
            return 'Side'
    except:
        return '?'

def build_ml_report(df, fmt='long'):
    now   = datetime.now().strftime('%Y-%m-%d %H:%M')
    lines = []
    sep   = "=" * 74

    def p(line=''):
        lines.append(str(line))

    p(sep)
    p(f"  AI STOCK SCREENER — ML REPORT ({fmt.upper()})")
    p(f"  Generated : {now}")
    p(f"  Universe  : 752 stocks | Buy signals: {len(df)}")
    p(sep)

    for label in buy_labels:
        section = df[df['ML_Prediction'] == label].copy()
        if len(section) == 0:
            continue
        p(f"\n{'─' * 74}")
        p(f"  {label.upper()}  ({len(section)} stocks)")
        if label in ('Bullish Continual', 'Tech Bullish'):
            p(f"  Already in uptrend — ML confirms continuation  [threshold: >=40%]")
        elif label == 'Bullish':
            p(f"  No clear setup yet — ML sees upside potential  [threshold: >=50%]")
        elif label == 'Reversal':
            p(f"  In downtrend — ML sees high bounce probability [threshold: >=50%]")
        p(f"{'─' * 74}")

        if fmt == 'short':
            p(f"  {'#':>3}  {'Symbol':<14} {'Cap':>12}  {'SecRnk':>6}  "
              f"{'CapRnk':>6}  {'Conf':>5}  {'25d%':>5}  {'45d%':>5}  "
              f"{'Trend':<6}  {'Tier'}")
            p(f"  {'─'*3}  {'─'*14} {'─'*12}  {'─'*6}  "
              f"{'─'*6}  {'─'*5}  {'─'*5}  {'─'*5}  "
              f"{'─'*6}  {'─'*10}")
            for i, (_, row) in enumerate(section.iterrows(), 1):
                tech_trend = get_tech_trend(row)
                tier_s     = tier_abbr(str(row.get('Tier', '')))
                p(f"  {i:>3}  {row['Symbol']:<14} "
                  f"{mcap_str(row['Market Cap Cr']):>12}  "
                  f"{row.get('Sector Score', 0):>6.1f}  "
                  f"{row.get('Cap Score', 0):>6.1f}  "
                  f"{row['ML_Confidence']:>5.1f}  "
                  f"{row['Forecast_25d_Pct']:>+5.1f}  "
                  f"{row['Forecast_45d_Pct']:>+5.1f}  "
                  f"{tech_trend:<6}  "
                  f"{tier_s}")
        else:
            for i, (_, row) in enumerate(section.iterrows(), 1):
                tech_trend = get_tech_trend(row)
                p(f"\n  {i}. {row['Symbol']}  |  {row['Sector']}  |  "
                  f"{row['Cap Category']}  |  {mcap_str(row['Market Cap Cr'])}")
                p(f"     Confidence : {row['ML_Confidence']:.1f}%  |  "
                  f"Trend: {tech_trend}  |  "
                  f"Tier: {tier_abbr(str(row.get('Tier', '')))}  |  "
                  f"Fund: {row.get('Fund Score', '—')}")
                p(f"     Forecast   : 25d={row['Forecast_25d_Pct']:+.1f}%  "
                  f"45d={row['Forecast_45d_Pct']:+.1f}%  "
                  f"180d={row['Forecast_180d_Pct']:+.1f}%")
                p(f"     Prices     : Now=Rs{row['Current Price']:.0f}  "
                  f"25d=Rs{row['Forecast_25d_Price']:.0f}  "
                  f"45d=Rs{row['Forecast_45d_Price']:.0f}  "
                  f"180d=Rs{row['Forecast_180d_Price']:.0f}")
                p(f"     Scores     : SecRnk={row.get('Sector Score', 0):.1f}  "
                  f"CapRnk={row.get('Cap Score', 0):.1f}  "
                  f"FundScore={row.get('Fund Score', '—')}")
                p(f"     ML Probs   : BullCont={row['Bullish_Cont_Prob']:.1f}%  "
                  f"BotRev={row['Bottom_Rev_Prob']:.1f}%  "
                  f"TopRev={row['Top_Rev_Prob']:.1f}%")

    p(f"\n{'=' * 74}")
    p(f"  END OF REPORT")
    p(f"{'=' * 74}\n")
    return lines

# Save all 4 reports
tech_short = generate_tech_report(is_short=True)
tech_long  = generate_tech_report(is_short=False)
ml_short   = build_ml_report(filtered, fmt='short')
ml_long    = build_ml_report(filtered, fmt='long')

with open(os.path.join(REPORTS_TECH, f'weekly_report_short_{today_file}.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(tech_short))
with open(os.path.join(REPORTS_TECH, f'weekly_report_long_{today_file}.txt'),  'w', encoding='utf-8') as f:
    f.write('\n'.join(tech_long))
with open(os.path.join(REPORTS_ML,   f'ml_report_short_{today_file}.txt'),     'w', encoding='utf-8') as f:
    f.write('\n'.join(ml_short))
with open(os.path.join(REPORTS_ML,   f'ml_report_long_{today_file}.txt'),      'w', encoding='utf-8') as f:
    f.write('\n'.join(ml_long))

print(f"     Reports saved:")
print(f"       {REPORTS_TECH}/weekly_report_short_{today_file}.txt")
print(f"       {REPORTS_TECH}/weekly_report_long_{today_file}.txt")
print(f"       {REPORTS_ML}/ml_report_short_{today_file}.txt")
print(f"       {REPORTS_ML}/ml_report_long_{today_file}.txt")

print()
print("=" * 60)
print("  Weekly run complete!")
print(f"  Finished : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)