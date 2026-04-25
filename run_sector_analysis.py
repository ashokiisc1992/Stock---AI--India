# ============================================================
# run_sector_analysis.py — AI Stock Screener (Indian Markets)
# Day 14: Sector Analysis
#
# Features:
#   1. Long Term Stage Analysis  (sector index EMA stages)
#   2. Short Term Sector Trends  (weekly trend drill-down)
#   3. Fund Score Improvements   (quarterly comparison)
#
# Run after run_weekly.py
# ============================================================

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime

# ── PATHS ─────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, 'data')
SCORES_DIR   = os.path.join(DATA_DIR, 'scores')
FUND_DIR     = os.path.join(DATA_DIR, 'fundamentals')
UNIVERSE_DIR = os.path.join(DATA_DIR, 'universe')
PRICES_DIR   = os.path.join(DATA_DIR, 'prices')
REPORT_DIR   = os.path.join(DATA_DIR, 'reports', 'sector_analysis')
os.makedirs(REPORT_DIR, exist_ok=True)

TECH_FILE         = os.path.join(SCORES_DIR,   'technical_report_full.csv')
FUND_FILE         = os.path.join(FUND_DIR,     'fundamental_scores_full.csv')
FUND_PREV_FILE    = os.path.join(FUND_DIR,     'fundamental_scores_prev.csv')
PREFILT_FILE      = os.path.join(UNIVERSE_DIR, 'prefilt_passed.csv')
SECTOR_INDEX_FILE = os.path.join(SCORES_DIR,   'sector_index_data.pkl')
PRICE_FILE        = os.path.join(PRICES_DIR,   'price_data_full.pkl')

# ── LOAD DATA ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  AI STOCK SCREENER — SECTOR ANALYSIS")
print(f"  {datetime.now().strftime('%d %B %Y  %H:%M')}")
print(f"{'='*60}")
print(f"\nLoading data...")

for fpath, label in [
    (TECH_FILE,         'technical_report_full.csv'),
    (FUND_FILE,         'fundamental_scores_full.csv'),
    (SECTOR_INDEX_FILE, 'sector_index_data.pkl'),
]:
    if not os.path.exists(fpath):
        print(f"  ERROR: {label} not found. Run run_weekly.py first.")
        exit(1)

tech_df    = pd.read_csv(TECH_FILE)
fund_df    = pd.read_csv(FUND_FILE)
prefilt_df = pd.read_csv(PREFILT_FILE)

if 'Market_Cap_Cr' not in fund_df.columns:
    fund_df = fund_df.merge(
        prefilt_df[['Symbol', 'Market_Cap_Cr']], on='Symbol', how='left')

with open(SECTOR_INDEX_FILE, 'rb') as f:
    sector_index_data = pickle.load(f)

try:
    with open(PRICE_FILE, 'rb') as f:
        price_data = pickle.load(f)
except:
    price_data = {}

print(f"  Tech report       : {len(tech_df)} stocks")
print(f"  Fund scores       : {len(fund_df)} stocks")
print(f"  Sector index data : {len(sector_index_data)} tickers")

# ── CONSTANTS ──────────────────────────────────────────────────
CAP_ORDER = ['Large Cap', 'Mini Large Cap', 'Mid Cap', 'Small Cap']
MIN_RANK  = 6.0
MIN_CONF  = 40.0

# ── COMPUTE SCORES ─────────────────────────────────────────────
def classify_mcap(v):
    try:
        v = float(v)
        if v >= 20000: return 'Large Cap'
        elif v >= 5000: return 'Mini Large Cap'
        elif v >= 1000: return 'Mid Cap'
        else:           return 'Small Cap'
    except:
        return 'Small Cap'

fund_df['Cap Category'] = fund_df['Market_Cap_Cr'].apply(classify_mcap)

fund_df['Sector Score'] = 0.0
for sector in fund_df['Sector'].unique():
    mask      = fund_df['Sector'] == sector
    max_score = fund_df[mask]['Final Score'].max()
    if max_score > 0:
        fund_df.loc[mask, 'Sector Score'] = (
            fund_df[mask]['Final Score'] / max_score * 10).round(1)

fund_df['Cap Score'] = 0.0
for sector in fund_df['Sector'].unique():
    for cap in CAP_ORDER:
        mask   = (fund_df['Sector'] == sector) & (fund_df['Cap Category'] == cap)
        subset = fund_df[mask]
        if len(subset) == 0: continue
        max_score = subset['Final Score'].max()
        if max_score > 0:
            fund_df.loc[mask, 'Cap Score'] = (
                subset['Final Score'] / max_score * 10).round(1)

# Build work_df
work_df = tech_df.copy()
for col in ['Sector Score', 'Cap Score', 'Final Score', 'Market_Cap_Cr',
            'Sector Score_x', 'Cap Score_x', 'Sector Score_y', 'Cap Score_y']:
    if col in work_df.columns:
        work_df = work_df.drop(columns=[col])

work_df = work_df.merge(
    fund_df[['Symbol', 'Sector Score', 'Cap Score',
             'Final Score', 'Market_Cap_Cr', 'Cap Category']],
    on='Symbol', how='left')

ALL_SECTORS = sorted(work_df['Sector'].dropna().unique().tolist())
print(f"  Sectors           : {len(ALL_SECTORS)}")
print(f"  Work_df           : {work_df.shape[0]} stocks, {work_df.shape[1]} cols")

# ── HELPERS ────────────────────────────────────────────────────
SECTOR_SHORT = {
    'Information Technology'            : 'IT',
    'Financial Services'                : 'Financial',
    'Chemicals'                         : 'Chemicals',
    'Healthcare'                        : 'Healthcare',
    'Consumer Services'                 : 'Con Services',
    'Automobile and Auto Components'    : 'Auto',
    'Fast Moving Consumer Goods'        : 'FMCG',
    'Capital Goods'                     : 'Capital Goods',
    'Metals & Mining'                   : 'Metals',
    'Pharmaceuticals'                   : 'Pharma',
    'Realty'                            : 'Realty',
    'Construction'                      : 'Construction',
    'Construction Materials'            : 'Cement/Const',
    'Power'                             : 'Power',
    'Oil, Gas & Consumable Fuels'       : 'Oil & Gas',
    'Banking'                           : 'Banking',
    'Services'                          : 'Services',
    'Consumer Durables'                 : 'Con Durables',
    'Textiles'                          : 'Textiles',
    'Telecommunication'                 : 'Telecom',
    'Media, Entertainment & Publication': 'Media',
    'Agriculture'                       : 'Agriculture',
    'Defence'                           : 'Defence',
    'Diversified'                       : 'Diversified',
    'Utilities'                         : 'Utilities',
    'Forest Materials'                  : 'Forest Mat',
}

TREND_SHORT = {
    'Strong Uptrend ↑↑'  : 'Str Up ↑↑',
    'Uptrend ↑'          : 'Uptrend ↑ ',
    'Weak Uptrend →↑'    : 'Wk Up →↑ ',
    'Sideways →'         : 'Sideways →',
    'Weak Downtrend →↓'  : 'Wk Dn →↓ ',
    'Downtrend ↓'        : 'Downtrend ↓',
    'Strong Downtrend ↓↓': 'Str Dn ↓↓',
}

def short_sector(s): return SECTOR_SHORT.get(str(s), str(s)[:14])
def short_trend(t):  return TREND_SHORT.get(str(t), str(t)[:12])

def mcap_str(v):
    try:
        v = float(v)
        if v >= 100000: return f"Rs{v/100000:.1f}L Cr"
        return f"Rs{v:,.0f}Cr"
    except:
        return "—"

def get_vmax(symbol):
    try:
        df = price_data.get(symbol)
        if df is None or len(df) < 20: return 0.0
        return round(df['Volume'].iloc[-5:].max() /
                     df['Volume'].iloc[-20:].mean(), 2)
    except:
        return 0.0

def pct_above_ema50(row):
    try:
        price = float(row.get('Current Price', 0) or 0)
        ema50 = float(row.get('EMA50', 0) or 0)
        return round((price - ema50) / ema50 * 100, 1) if ema50 > 0 else 0.0
    except:
        return 0.0

def print_stock_table(df, top_n=10):
    print(f"\n  {'#':<3}  {'Symbol':<12} {'Sector':<14} {'Sec Trend':<12} "
          f"{'ML Prediction':<20} {'Conf':>5}  "
          f"{'Setup':<9} {'Tech':>4}  {'EMA50%':>6}  "
          f"{'SecRnk':>6}  {'CapRnk':>6}  "
          f"{'Price':>8}  {'MCap':>12}  {'V5D':>5}  {'VMax':>5}")
    print(f"  {'─'*148}")
    serial = 1
    for cap in CAP_ORDER:
        cap_df = df[df['Cap Category'] == cap].head(top_n)
        if len(cap_df) == 0: continue
        cap_short = {'Large Cap':'L','Mini Large Cap':'ML',
                     'Mid Cap':'M','Small Cap':'S'}.get(cap,'?')
        print(f"\n  [{cap_short}] {cap}")
        for _, row in cap_df.iterrows():
            conf  = float(row.get('ML_Confidence', 0) or 0)
            price = float(row.get('Current Price', 0) or 0)
            mcap  = float(row.get('Market_Cap_Cr',
                          row.get('Market Cap Cr', 0)) or 0)
            v5d   = float(row.get('Vol 5D Ratio',
                          row.get('Vol Ratio', 0)) or 0)
            vmax  = get_vmax(row['Symbol'])
            pct   = pct_above_ema50(row)
            sec_t = str(row.get('Sector Trend', ''))
            conf_disp = f"{conf:>5.1f}" if conf > 0 else f"{'—':>5}"
            print(f"  {serial:<3}  "
                  f"{row['Symbol']:<12} "
                  f"{short_sector(row.get('Sector','')):<14} "
                  f"{short_trend(sec_t):<12} "
                  f"{str(row.get('ML_Prediction','')):<20} "
                  f"{conf_disp}  "
                  f"{str(row.get('Best Setup','')):<9} "
                  f"{float(row.get('Tech Score', 0) or 0):>4.0f}  "
                  f"{pct:>+6.1f}%  "
                  f"{float(row.get('Sector Score', 0) or 0):>6.1f}  "
                  f"{float(row.get('Cap Score', 0) or 0):>6.1f}  "
                  f"{price:>8.2f}  "
                  f"{mcap_str(mcap):>12}  "
                  f"{v5d:>4.2f}x  "
                  f"{vmax:>4.2f}x")
            serial += 1
    print(f"\n  {'─'*148}")

# ── SECTOR INDEX MAP ───────────────────────────────────────────
# Only sectors with dedicated accurate index
# Removed wrong mappings:
#   Power/Utilities       → ^CNXENERGY is Oil & Gas dominated
#   Construction Materials → ^CNXCMDT is Commodities, not cement
#   Telecom/Media/Textiles/Forest → MID150BEES is meaningless proxy
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
    'Construction'                      : '^CNXINFRA',
    'Pharmaceuticals'                   : 'PHARMABEES.NS',
    'Chemicals'                         : 'MOGSEC.NS',
    'Consumer Services'                 : 'CONSUMBEES.NS',
    'Diversified'                       : 'JUNIORBEES.NS',
}

# ── STAGE COMPUTATION FUNCTIONS ────────────────────────────────
def compute_index_indicators(df):
    d = df.copy()
    d['EMA20']  = d['Close'].ewm(span=20,  adjust=False).mean()
    d['EMA50']  = d['Close'].ewm(span=50,  adjust=False).mean()
    d['EMA200'] = d['Close'].ewm(span=200, adjust=False).mean()
    delta = d['Close'].diff()
    gain  = delta.where(delta > 0, 0)
    loss  = -delta.where(delta < 0, 0)
    rs    = gain.ewm(span=14, adjust=False).mean() / (
            loss.ewm(span=14, adjust=False).mean() + 1e-9)
    d['RSI'] = 100 - (100 / (1 + rs))
    ema12       = d['Close'].ewm(span=12, adjust=False).mean()
    ema26       = d['Close'].ewm(span=26, adjust=False).mean()
    d['MACD']   = ema12 - ema26
    d['Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['MACD_H'] = d['MACD'] - d['Signal']
    high = d['High']; low = d['Low']; close = d['Close']
    tr   = pd.concat([high - low,
                      (high - close.shift()).abs(),
                      (low  - close.shift()).abs()], axis=1).max(axis=1)
    dmp  = high.diff().clip(lower=0)
    dmm  = (-low.diff()).clip(lower=0)
    dmp  = dmp.where(dmp > dmm, 0)
    dmm  = dmm.where(dmm > dmp, 0)
    atr  = tr.ewm(span=14, adjust=False).mean()
    di_p = 100 * dmp.ewm(span=14, adjust=False).mean() / (atr + 1e-9)
    di_m = 100 * dmm.ewm(span=14, adjust=False).mean() / (atr + 1e-9)
    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    d['ADX']      = dx.ewm(span=14, adjust=False).mean()
    d['DI_Plus']  = di_p
    d['DI_Minus'] = di_m
    return d

def score_momentum_index(df):
    latest = df.iloc[-1]
    scores = {}
    close  = latest['Close']
    ema20  = latest['EMA20']
    ema50  = latest['EMA50']
    ema200 = latest['EMA200']
    if close > ema20 > ema50 > ema200:  scores['EMA'] = 30
    elif close > ema50 > ema200:         scores['EMA'] = 20
    elif close > ema200:                 scores['EMA'] = 10
    else:                                scores['EMA'] = 0
    rsi = latest['RSI']
    if 50 <= rsi <= 70:    scores['RSI'] = 20
    elif 45 <= rsi < 50:   scores['RSI'] = 12
    elif 70 < rsi <= 80:   scores['RSI'] = 10
    elif rsi > 80:         scores['RSI'] = 5
    else:                  scores['RSI'] = 0
    hist = latest['MACD_H']
    if hist > 0 and latest['MACD'] > latest['Signal']: scores['MACD'] = 25
    elif hist > 0:                                       scores['MACD'] = 15
    elif hist > -0.5:                                    scores['MACD'] = 8
    else:                                                scores['MACD'] = 0
    adx = latest['ADX']
    if adx > 25 and latest['DI_Plus'] > latest['DI_Minus']: scores['ADX'] = 25
    elif adx > 20 and latest['DI_Plus'] > latest['DI_Minus']: scores['ADX'] = 15
    elif adx > 25:                                             scores['ADX'] = 8
    else:                                                      scores['ADX'] = 0
    return min(sum(scores.values()), 100)

def score_momentum_stocks(sector):
    stocks = work_df[work_df['Sector'] == sector]
    if len(stocks) == 0:
        return 0
    pct_full_align   = (
        (stocks['Current Price'] > stocks['EMA50']) &
        (stocks['EMA50']         > stocks['EMA200'])
    ).mean()
    pct_above_ema50  = (stocks['Current Price'] > stocks['EMA50']).mean()
    pct_above_ema200 = (stocks['Current Price'] > stocks['EMA200']).mean()
    avg_rsi = stocks['RSI'].median()
    pct_macd_bull = (stocks['MACD Hist'] > 0).mean() \
                    if 'MACD Hist' in stocks.columns else 0.5
    score  = 0
    score += pct_full_align   * 30
    score += pct_above_ema50  * 10
    score += pct_above_ema200 * 10
    if 50 <= avg_rsi <= 70:   score += 20
    elif 45 <= avg_rsi < 50:  score += 12
    elif avg_rsi > 70:        score += 8
    score += pct_macd_bull * 25
    return min(int(score), 100)

def get_cross_info(df):
    ema50  = df['EMA50']
    ema200 = df['EMA200']
    above  = ema50 > ema200
    current_state = above.iloc[-1]
    days = 0
    for val in reversed(above.values):
        if val == current_state:
            days += 1
        else:
            break
    gap_now  = float(ema50.iloc[-1]  - ema200.iloc[-1])
    gap_prev = float(ema50.iloc[-10] - ema200.iloc[-10]) if len(df) > 10 else gap_now
    widening = abs(gap_now) > abs(gap_prev)
    return current_state, days, gap_now, widening

def get_stock_cross_info(sector):
    stocks = work_df[work_df['Sector'] == sector]
    if len(stocks) == 0:
        return False, 0, 0, False
    pct          = (stocks['EMA50'] > stocks['EMA200']).mean()
    is_above     = pct >= 0.5
    avg_gap_now  = (stocks['EMA50'] - stocks['EMA200']).mean()
    widening     = avg_gap_now > 0 if is_above else avg_gap_now < 0
    return is_above, int(round(pct * 100)), float(pct - 0.5), widening

def get_stock_confirmation(sector):
    stocks = work_df[work_df['Sector'] == sector]
    if len(stocks) == 0:
        return {'pct_above_ema200': 0, 'pct_recovery': 0,
                'pct_momentum': 0, 'pct_bull_cont': 0,
                'avg_rsi': 50, 'n': 0}
    n = len(stocks)
    pct_ema200 = (stocks['Current Price'] > stocks['EMA200']).mean() * 100
    pct_rec = (
        (stocks['Current Price'] > stocks['EMA50']) &
        (stocks['Current Price'] < stocks['EMA200'])
    ).mean() * 100
    pct_mom = (
        (stocks['Current Price'] > stocks['EMA50']) &
        (stocks['EMA50']         > stocks['EMA200'])
    ).mean() * 100
    pct_bc = (
        (stocks['Current Price'] > stocks['EMA50']) &
        (stocks['EMA50']         > stocks['EMA200']) &
        (stocks['ML_Prediction'] == 'Bullish Continual') &
        (stocks['ML_Confidence'].fillna(0) >= 40)
    ).mean() * 100
    avg_rsi = stocks['RSI'].median()
    return {
        'pct_above_ema200': round(pct_ema200, 1),
        'pct_recovery'    : round(pct_rec,    1),
        'pct_momentum'    : round(pct_mom,    1),
        'pct_bull_cont'   : round(pct_bc,     1),
        'avg_rsi'         : round(avg_rsi,    1),
        'n'               : n,
    }

def assign_stage(is_above, days, widening, mom_score, conf, from_stocks=False):
    pct_ema200      = conf['pct_above_ema200']
    avg_rsi         = conf['avg_rsi']
    pct_above_ema50 = conf['pct_momentum'] + conf['pct_recovery']
    strong_stock_bull = (pct_ema200 >= 55 and pct_above_ema50 >= 70)

    # Stage 4 — Euphoria (index sectors only)
    if (not from_stocks and is_above and days > 20 and widening and
            mom_score >= 80 and pct_ema200 >= 75 and avg_rsi >= 68):
        return 4, 'Euphoria 🔥'

    # Stock aggregation sectors — stage from stock EMA alignment only
    if from_stocks:
        if pct_ema200 >= 50 and pct_above_ema50 >= 60:
            return 3, 'Confirmed Bull ✅'
        if pct_ema200 >= 30 or pct_above_ema50 >= 35:
            return 2, 'Early Entry ⚡'
        return 1, 'Accumulation ⏳'

    # Index sectors — use cross duration + stock confirmation
    if is_above and days > 20 and widening and pct_ema200 >= 50:
        if pct_ema200 < 25:
            return 2, 'Early Entry ⚡'
        return 3, 'Confirmed Bull ✅'
    if is_above and days > 10 and pct_ema200 >= 45:
        if pct_ema200 < 25:
            return 2, 'Early Entry ⚡'
        return 3, 'Confirmed Bull ✅'
    if strong_stock_bull:
        return 3, 'Confirmed Bull ✅'
    if is_above and days <= 20:
        return 2, 'Early Entry ⚡'
    if not is_above and days <= 20:
        return 2, 'Early Entry ⚡'
    if not is_above and not widening and pct_ema200 >= 30:
        return 2, 'Early Entry ⚡'
    if not is_above and days > 20 and pct_ema200 < 30:
        return 1, 'Accumulation ⏳'
    if not is_above:
        return 1, 'Accumulation ⏳'
    return 3, 'Confirmed Bull ✅'

# ── COMPUTE SECTOR STAGES ──────────────────────────────────────
sector_stages = {}
for sector in ALL_SECTORS:
    ticker = SECTOR_INDEX_MAP.get(sector)
    conf   = get_stock_confirmation(sector)
    if ticker and ticker in sector_index_data:
        df_raw = sector_index_data[ticker]
        try:
            df_ind    = compute_index_indicators(df_raw)
            latest    = df_ind.iloc[-1]
            is_above, days, gap, widening = get_cross_info(df_ind)
            mom_score = score_momentum_index(df_ind)
            stage_n, stage_label = assign_stage(
                is_above, days, widening, mom_score, conf, from_stocks=False)
            rsi       = round(latest['RSI'], 1)
            adx       = round(latest['ADX'], 1)
            cross_str = (f"{'Bull' if is_above else 'Bear'} {days}d "
                         f"({'widen' if widening else 'narrow'})")
        except:
            is_above, days, gap, widening = get_stock_cross_info(sector)
            mom_score = score_momentum_stocks(sector)
            stage_n, stage_label = assign_stage(
                is_above, days, widening, mom_score, conf, from_stocks=True)
            rsi       = conf['avg_rsi']
            adx       = 0.0
            cross_str = (f"{'Bull' if is_above else 'Bear'} "
                         f"({'widen' if widening else 'narrow'})")
    else:
        is_above, days, gap, widening = get_stock_cross_info(sector)
        mom_score = score_momentum_stocks(sector)
        stage_n, stage_label = assign_stage(
            is_above, days, widening, mom_score, conf, from_stocks=True)
        rsi       = conf['avg_rsi']
        adx       = 0.0
        cross_str = (f"{'Bull' if is_above else 'Bear'} "
                     f"({'widen' if widening else 'narrow'})")
    sector_stages[sector] = {
        'stage_n'    : stage_n,
        'stage_label': stage_label,
        'mom_score'  : mom_score,
        'rsi'        : rsi,
        'adx'        : adx,
        'cross'      : cross_str,
        'conf'       : conf,
    }

print(f"\n  Stage computation done.")

# ── SIGNAL DEFINITIONS ─────────────────────────────────────────
SIGNAL_LABELS = {
    '1': 'Bullish Continual  (Price>EMA50>EMA200, ML or Tech confirmed)',
    '2': 'Bullish            (Price>EMA50>EMA200, Conf>=40%)',
    '3': 'Reversal           (ML or Setup=Reversal, Conf>=40%)',
    '4': 'Watching / Basing  (Price<EMA50, Setup=Watching, ADX<30, RSI>35)',
    '5': 'Bearish Continual  (avoid / monitor for exit)',
    '6': 'All signals        (every ranked stock — no ML filter)',
}

SORT_COLS = {
    '1': ('ML_Confidence',  False),
    '2': ('ML_Confidence',  False),
    '3': ('Reversal Score', False),
    '4': ('Sector Score',   False),
    '5': ('ML_Confidence',  False),
    '6': ('ML_Confidence',  False),
}

def get_stage_stocks(df, signal_type):
    if signal_type == '1':
        return df[
            (df['ML_Prediction'].isin(['Bullish Continual', 'Tech Bullish'])) &
            (df['Current Price'] > df['EMA50']) &
            (df['EMA50']         > df['EMA200'])
        ].copy()
    elif signal_type == '2':
        return df[
            (df['ML_Prediction'] == 'Bullish') &
            (df['ML_Confidence'].fillna(0) >= MIN_CONF) &
            (df['Current Price'] > df['EMA50']) &
            (df['EMA50']         > df['EMA200'])
        ].copy()
    elif signal_type == '3':
        return df[
            (
                (df['ML_Prediction'] == 'Reversal') |
                (df['Best Setup']    == 'Reversal')
            ) &
            (df['ML_Confidence'].fillna(0) >= MIN_CONF)
        ].copy()
    elif signal_type == '4':
        return df[
            (df['Current Price'] < df['EMA50']) &
            (df['Best Setup']    == 'Watching') &
            (df['ADX'].fillna(99) < 30) &
            (df['RSI'].fillna(0)  > 35)
        ].copy()
    elif signal_type == '5':
        return df[df['ML_Prediction'] == 'Bearish Continual'].copy()
    else:
        return df.copy()

# ── FEATURE 1: LONG TERM STAGE ANALYSIS ───────────────────────
def print_stage_dashboard():
    print(f"\n{'='*130}")
    print(f"  SECTOR LONG TERM STAGE DASHBOARD  —  {datetime.now().strftime('%d %B %Y')}")
    print(f"{'='*130}")
    print(f"""
  STAGE DEFINITIONS
  ─────────────────────────────────────────────────────────────────────────────────────
  Stage 4 Euphoria      : Index RSI very high + >75% stocks above EMA200 + MomScr>=80
                          → Overheated. Reduce new entries. Book partial profits.
  Stage 3 Confirmed Bull: EMA50 > EMA200 sustained + >50% stocks above EMA200
                          → Best risk/reward for fresh entries on dips.
  Stage 2 Early Entry ⚡ : EMA50 crossing EMA200 (within 20 days either side)
                          → High potential. Confirm with stock signals before entry.
  Stage 1 Accumulation  : EMA50 firmly below EMA200 + <30% stocks above EMA200
                          → Wait. No clear signal yet.

  PRICE vs EMA PHASES — four stages of a stock/sector cycle
  ─────────────────────────────────────────────────────────────────────────────────────
  Phase 1: Price < EMA50 < EMA200  → Setup=Watching  — Downtrend / basing
  Phase 2: Price > EMA50 < EMA200  → Setup=Momentum  — Partial recovery
                                      Above short term avg (EMA50), below long term (EMA200)
                                      %Rec column counts these stocks
  Phase 3: Price > EMA50 > EMA200  → Setup=Momentum  — Full bull confirmed
                                      Both short + long term EMAs aligned bullish
                                      %Mom column counts these stocks
  Phase 4: EMA20>EMA50>EMA200      → Setup=Momentum  — Strongest. All EMAs aligned.
                                      ML=Bullish Continual scores highest here
                                      %BC counts Phase 4 stocks (ML conf >= 40%)

  SIGNAL TYPES
  ─────────────────────────────────────────────────────────────────────────────────────
  Bullish Continual : ML Model 4 ran AND gave confidence 40-100% + Price>EMA50>EMA200
                      Strongest signal — both technically AND ML confirmed.
  Tech Bullish      : Price>EMA50>EMA200 by rules only — ML model did NOT run
                      Same uptrend condition, no ML backing. Slightly lower confidence.
  Bullish           : ML predicts upward move Conf>=40%, Price>EMA50>EMA200
  Reversal          : ML or Setup=Reversal, Conf>=40%. Price may be below EMA50.
  Watching/Basing   : Price<EMA50, ADX<30 (trend fading), RSI>35 (not oversold)
  Bearish Continual : ML predicts continued downtrend. Avoid for new entries.

  NOTE ON RANK FILTER (SecRnk>=6 OR CapRnk>=6):
  For Stage 1 sectors — some stocks may be in individual uptrend but fundamentally
  weaker (low rank). If signal 1 returns 0 results, try signal 6 (All) to find them.
  Example: IT sector has BBOX, SATTRIX in uptrend — both below rank threshold.
  ─────────────────────────────────────────────────────────────────────────────────────
""")
    print(f"  {'Sector':<44} {'Stage':<22} {'MomScr':>6}  {'Cross':<22}  "
          f"{'RSI':>5}  {'ADX':>5}  {'%EMA200':>7}  {'%Rec':>5}  {'%Mom':>5}  {'%BC':>4}")
    print(f"  {'─'*133}")
    for stage in [4, 3, 2, 1]:
        stage_headers = {
            4: '── STAGE 4  EUPHORIA ──────────────────────────',
            3: '── STAGE 3  CONFIRMED BULL ────────────────────',
            2: '── STAGE 2  EARLY ENTRY ⚡ ─────────────────────',
            1: '── STAGE 1  ACCUMULATION ──────────────────────',
        }
        first = True
        for sector, info in sorted(
            sector_stages.items(),
            key=lambda x: (-x[1]['stage_n'], -x[1]['mom_score'])
        ):
            if info['stage_n'] != stage: continue
            if first:
                print(f"\n  {stage_headers[stage]}")
                first = False
            c = info['conf']
            print(f"  {sector:<44} {info['stage_label']:<22} "
                  f"{info['mom_score']:>6}  "
                  f"{info['cross']:<22}  "
                  f"{info['rsi']:>5.1f}  "
                  f"{info['adx']:>5.1f}  "
                  f"{c['pct_above_ema200']:>6.1f}%  "
                  f"{c['pct_recovery']:>5.1f}  "
                  f"{c['pct_momentum']:>5.1f}  "
                  f"{c['pct_bull_cont']:>4.1f}")
    print(f"\n  {'─'*133}")
    print(f"  MomScr=Momentum Score 0-100 | Cross=EMA50 vs EMA200 | Bull/Bear from index or stocks")
    print(f"  %EMA200=Price>EMA200 | %Rec=partial(>EMA50,<EMA200) | %Mom=full(>EMA50>EMA200) | %BC=ML confirmed")

def run_long_term_stage_analysis():
    print_stage_dashboard()

    print(f"\n{'─'*65}")
    stage_input = input(
        "\n  Which stage? (1/2/3/4/all) [default all]: "
    ).strip().lower() or 'all'

    if stage_input == 'all':
        selected_stages = [4, 3, 2, 1]
    elif stage_input in ['1','2','3','4']:
        selected_stages = [int(stage_input)]
    else:
        print("  Invalid — showing all.")
        selected_stages = [4, 3, 2, 1]

    available_sectors = {
        s: info for s, info in sector_stages.items()
        if info['stage_n'] in selected_stages
    }

    print(f"\n{'─'*65}")
    print(f"  Sectors in selected stage(s):")
    sector_list = sorted(available_sectors.keys())
    for i, s in enumerate(sector_list, 1):
        info     = available_sectors[s]
        c        = info['conf']
        all_n    = len(work_df[work_df['Sector'] == s])
        ranked_n = len(work_df[
            (work_df['Sector'] == s) &
            (
                (work_df['Sector Score'].fillna(0) >= MIN_RANK) |
                (work_df['Cap Score'].fillna(0)    >= MIN_RANK)
            )
        ])
        print(f"  {i:>3}. {s:<44} [Stage {info['stage_n']}]  "
              f"Total:{all_n:>4}  Ranked:{ranked_n:>4}  "
              f"Cross:{info['cross']:<22}  "
              f"%EMA200:{c['pct_above_ema200']:>5.1f}%  "
              f"%Rec:{c['pct_recovery']:>5.1f}%  "
              f"%Mom:{c['pct_momentum']:>5.1f}%  "
              f"%BC:{c['pct_bull_cont']:>4.1f}%")
    print(f"    0. All sectors")

    sec_input = input("\n  Choose sector (number or 0 for all): ").strip()
    try:
        sec_num = int(sec_input)
        if sec_num == 0:
            selected_sectors = sector_list
        elif 1 <= sec_num <= len(sector_list):
            selected_sectors = [sector_list[sec_num - 1]]
        else:
            selected_sectors = sector_list
    except:
        selected_sectors = sector_list

    print(f"\n{'─'*65}")
    print(f"  What stocks to show?")
    for k, v in SIGNAL_LABELS.items():
        print(f"  {k}. {v}")

    # Remind about rank filter for Stage 1
    if len(selected_stages) == 1 and selected_stages[0] == 1:
        print(f"\n  ℹ️  Stage 1 — if signal 1 returns 0, try signal 6 (All)")

    signal_type = input(
        "\n  Signal type (1-6) [default 6]: "
    ).strip() or '6'
    if signal_type not in SIGNAL_LABELS:
        signal_type = '6'

    try:
        top_n = int(input(
            "\n  Stocks per cap category [default 10]: "
        ).strip() or 10)
    except:
        top_n = 10

    base = work_df[
        work_df['Sector'].isin(selected_sectors) &
        (
            (work_df['Sector Score'].fillna(0) >= MIN_RANK) |
            (work_df['Cap Score'].fillna(0)    >= MIN_RANK)
        )
    ].copy()

    filtered = get_stage_stocks(base, signal_type)
    sort_col, sort_asc = SORT_COLS.get(signal_type, ('ML_Confidence', False))
    filtered['_cap_order'] = filtered['Cap Category'].map(
        {c: i for i, c in enumerate(CAP_ORDER)}).fillna(99)
    if sort_col in filtered.columns:
        filtered = filtered.sort_values(
            ['_cap_order', sort_col], ascending=[True, sort_asc])

    stage_label  = (f"Stage {selected_stages[0]}"
                    if len(selected_stages) == 1 else "All Stages")
    sec_label    = (selected_sectors[0]
                    if len(selected_sectors) == 1
                    else f"{len(selected_sectors)} sectors")
    total_all    = len(work_df[work_df['Sector'].isin(selected_sectors)])
    total_ranked = len(base)

    print(f"\n{'='*65}")
    print(f"  {stage_label}  |  {sec_label}")
    print(f"  Signal  : {SIGNAL_LABELS[signal_type]}")
    print(f"  Total:{total_all}  Ranked:{total_ranked}  "
          f"Below rank:{total_all-total_ranked}  "
          f"Match signal:{len(filtered)}")
    print(f"{'='*65}")

    if len(filtered) == 0:
        print(f"\n  No stocks match this filter.")
        if signal_type == '1':
            print(f"  → Try signal 4 (Watching/Basing) for Stage 1/2 sectors")
            print(f"  → Try signal 6 (All) to see all ranked stocks")
        else:
            print(f"  → Try signal 6 (All) to see all ranked stocks")
        return

    print_stock_table(filtered, top_n=top_n)
    print(f"  SecRnk/CapRnk=Relative rank 0-10  |  "
          f"V5D=5day avg vol  |  VMax=peak day vol")

# ── FEATURE 2: SHORT TERM SECTOR TRENDS ───────────────────────
TREND_ORDER = {
    'Strong Uptrend ↑↑'  : 1,
    'Uptrend ↑'          : 2,
    'Weak Uptrend →↑'    : 3,
    'Sideways →'         : 4,
    'Weak Downtrend →↓'  : 5,
    'Downtrend ↓'        : 6,
    'Strong Downtrend ↓↓': 7,
    'No data'            : 8,
}

def run_short_term_sector_trends():
    sector_trend_map = {}
    for sector in ALL_SECTORS:
        s_stocks = work_df[work_df['Sector'] == sector]
        if len(s_stocks) == 0:
            sector_trend_map[sector] = 'No data'
            continue
        trend_counts = s_stocks['Sector Trend'].value_counts()
        sector_trend_map[sector] = (
            trend_counts.index[0] if len(trend_counts) > 0 else 'No data')

    sorted_sectors = sorted(
        ALL_SECTORS,
        key=lambda s: TREND_ORDER.get(sector_trend_map.get(s, 'No data'), 8)
    )

    print(f"\n{'='*92}")
    print(f"  SHORT TERM SECTOR TRENDS")
    print(f"  Source: Weekly technical report  |  {datetime.now().strftime('%d %B %Y')}")
    print(f"{'='*92}")
    print(f"""
  HOW TO READ
  ───────────────────────────────────────────────────────────────────────────────────────
  Trend   : Weekly sector trend from index or stock aggregation (run_weekly.py)
  Total   : All stocks in sector within 752 quality universe
  Ranked  : Stocks with SecRnk>=6 OR CapRnk>=6 shown in detail view
            Remaining stocks are lower quality within the sector (still tracked)

            NOTE ON RANK FILTER:
            For strong bull sectors (Stage 3) — rank filter is appropriate.
            For weak/accumulation sectors (Stage 1) — some bullish stocks may be
            fundamentally weaker (low SecRnk) but still in individual uptrend.
            If you see %Mom or %BC in dashboard but 0 results in signal 1 —
            use signal 6 (All) to find them regardless of rank.

  %Mom    : % of ALL sector stocks with Price > EMA50 > EMA200 (full bull alignment)
  %BC     : % of ALL sector stocks with ML = Bullish Continual conf >= 40%
            Both computed from all stocks, not just ranked ones
  ───────────────────────────────────────────────────────────────────────────────────────
""")

    print(f"  {'#':>3}  {'Sector':<44}  {'Trend':<12}  "
          f"{'Total':>6}  {'Ranked':>7}  {'%Mom':>6}  {'%BC':>5}")
    print(f"  {'─'*95}")

    for i, sector in enumerate(sorted_sectors, 1):
        trend   = sector_trend_map[sector]
        stocks  = work_df[work_df['Sector'] == sector]
        n       = len(stocks)
        ranked  = len(stocks[
            (stocks['Sector Score'].fillna(0) >= MIN_RANK) |
            (stocks['Cap Score'].fillna(0)    >= MIN_RANK)
        ])
        pct_mom = (
            (stocks['Current Price'] > stocks['EMA50']) &
            (stocks['EMA50']         > stocks['EMA200'])
        ).mean() * 100 if n > 0 else 0
        pct_bc  = (
            (stocks['Current Price'] > stocks['EMA50']) &
            (stocks['EMA50']         > stocks['EMA200']) &
            (stocks['ML_Prediction'] == 'Bullish Continual') &
            (stocks['ML_Confidence'].fillna(0) >= 40)
        ).mean() * 100 if n > 0 else 0
        print(f"  {i:>3}. {sector:<44}  {short_trend(trend):<12}  "
              f"{n:>6}  {ranked:>7}  {pct_mom:>5.1f}%  {pct_bc:>4.1f}%")

    print(f"  {'─'*95}")
    print(f"  Ranked=stocks with SecRnk>=6 OR CapRnk>=6 shown in detail view")

    print(f"\n  Enter sector number for detail, or 0 to exit: ", end='')

    while True:
        sec_input = input("").strip()
        try:
            sec_num = int(sec_input)
        except:
            print(f"  Invalid. Try again: ", end='')
            continue

        if sec_num == 0:
            print(f"  Exiting sector trends.")
            return

        if not (1 <= sec_num <= len(sorted_sectors)):
            print(f"  Invalid. Enter 1-{len(sorted_sectors)} or 0: ", end='')
            continue

        selected_sector = sorted_sectors[sec_num - 1]
        selected_trend  = sector_trend_map[selected_sector]
        all_stocks      = work_df[work_df['Sector'] == selected_sector]
        ranked_stocks   = all_stocks[
            (all_stocks['Sector Score'].fillna(0) >= MIN_RANK) |
            (all_stocks['Cap Score'].fillna(0)    >= MIN_RANK)
        ]
        unranked = len(all_stocks) - len(ranked_stocks)

        print(f"\n{'─'*65}")
        print(f"  Sector  : {selected_sector}")
        print(f"  Trend   : {selected_trend}")
        print(f"  Total   : {len(all_stocks)} stocks  "
              f"| Ranked (>=6): {len(ranked_stocks)}  "
              f"| Below rank: {unranked}")
        print(f"\n  What stocks to show?")
        for k, v in SIGNAL_LABELS.items():
            print(f"  {k}. {v}")

        stage_n = sector_stages.get(selected_sector, {}).get('stage_n', 2)
        if stage_n == 1:
            print(f"\n  ℹ️  Stage 1 sector — if signal 1 returns 0, try signal 6 (All)")

        signal_type = input(
            "\n  Signal type (1-6) [default 6]: "
        ).strip() or '6'
        if signal_type not in SIGNAL_LABELS:
            signal_type = '6'

        try:
            top_n = int(input(
                "\n  Stocks per cap category [default 10]: "
            ).strip() or 10)
        except:
            top_n = 10

        base = work_df[
            (work_df['Sector'] == selected_sector) &
            (
                (work_df['Sector Score'].fillna(0) >= MIN_RANK) |
                (work_df['Cap Score'].fillna(0)    >= MIN_RANK)
            )
        ].copy()

        filtered = get_stage_stocks(base, signal_type)
        sort_col, sort_asc = SORT_COLS.get(signal_type, ('ML_Confidence', False))
        filtered['_cap_order'] = filtered['Cap Category'].map(
            {c: i for i, c in enumerate(CAP_ORDER)}).fillna(99)
        if sort_col in filtered.columns:
            filtered = filtered.sort_values(
                ['_cap_order', sort_col], ascending=[True, sort_asc])

        print(f"\n{'='*65}")
        print(f"  {selected_sector}  |  {selected_trend}")
        print(f"  Signal  : {SIGNAL_LABELS[signal_type]}")
        print(f"  Total:{len(all_stocks)}  Ranked:{len(ranked_stocks)}  "
              f"Below rank:{unranked}  Match signal:{len(filtered)}")
        print(f"{'='*65}")

        if len(filtered) == 0:
            print(f"\n  No ranked stocks match this signal filter.")
            print(f"  → Try signal 6 (All) to see all ranked stocks")
        else:
            print_stock_table(filtered, top_n=top_n)
            print(f"  SecRnk/CapRnk=Relative rank 0-10  |  "
                  f"V5D=5day avg vol  |  VMax=peak day vol")

        print(f"\n  Enter another sector number, or 0 to exit: ", end='')

# ── FEATURE 3: FUND SCORE IMPROVEMENTS ────────────────────────
def run_fund_score_improvements():
    print(f"\n{'='*65}")
    print(f"  FUND SCORE IMPROVEMENTS")
    print(f"  Compares current vs previous quarter fundamental scores")
    print(f"{'='*65}")

    if not os.path.exists(FUND_PREV_FILE):
        print(f"\n  No previous score file found at:")
        print(f"  {FUND_PREV_FILE}")
        print(f"\n  This file is created automatically by run_quarterly.py")
        print(f"  Run at least two quarterly cycles to use this feature.")
        return

    prev_df = pd.read_csv(FUND_PREV_FILE)
    curr_df = fund_df.copy()

    print(f"  Current quarter : {len(curr_df)} stocks")
    print(f"  Previous quarter: {len(prev_df)} stocks")

    merged = curr_df[['Symbol', 'Sector', 'Final Score',
                       'Historical Score', 'Peer Score',
                       'Quality Score']].merge(
        prev_df[['Symbol', 'Final Score',
                 'Historical Score', 'Peer Score',
                 'Quality Score']].rename(columns={
            'Final Score'      : 'Prev Final',
            'Historical Score' : 'Prev Historical',
            'Peer Score'       : 'Prev Peer',
            'Quality Score'    : 'Prev Quality',
        }),
        on='Symbol', how='inner'
    )

    merged['Delta']      = merged['Final Score']      - merged['Prev Final']
    merged['Delta_Hist'] = merged['Historical Score'] - merged['Prev Historical']
    merged['Delta_Peer'] = merged['Peer Score']       - merged['Prev Peer']
    merged['Delta_Qual'] = (merged['Quality Score'].astype(float) -
                            merged['Prev Quality'].astype(float))
    merged = merged.merge(
        fund_df[['Symbol', 'Cap Category', 'Sector Score',
                 'Cap Score', 'Market_Cap_Cr']],
        on='Symbol', how='left')

    new_symbols     = set(curr_df['Symbol']) - set(prev_df['Symbol'])
    dropped_symbols = set(prev_df['Symbol']) - set(curr_df['Symbol'])

    crossed_60  = merged[(merged['Final Score'] >= 60) &
                          (merged['Prev Final']  <  60) &
                          (merged['Delta']       >   0)
                         ].sort_values('Delta', ascending=False)
    improved_10 = merged[(merged['Delta'] >= 10) &
                          (~merged['Symbol'].isin(crossed_60['Symbol']))
                         ].sort_values('Delta', ascending=False)
    improved_5  = merged[(merged['Delta'] >= 5) &
                          (merged['Delta'] <  10) &
                          (~merged['Symbol'].isin(crossed_60['Symbol']))
                         ].sort_values('Delta', ascending=False)
    declined_10 = merged[merged['Delta'] <= -10].sort_values('Delta')
    dropped_60  = merged[(merged['Final Score'] <  60) &
                          (merged['Prev Final']  >= 60) &
                          (merged['Delta']       <   0)
                         ].sort_values('Delta')

    def print_fund_table(df):
        if len(df) == 0:
            print(f"  None")
            return
        print(f"  {'Symbol':<12} {'Sector':<26} {'Cap':<14} "
              f"{'Prev':>5} {'Now':>5} {'Δ':>5}  "
              f"{'ΔHist':>6} {'ΔPeer':>6} {'ΔQual':>6}  "
              f"{'SecRnk':>6} {'CapRnk':>6}  {'MCap':>11}")
        print(f"  {'─'*120}")
        for _, row in df.iterrows():
            print(f"  {row['Symbol']:<12} "
                  f"{str(row['Sector'])[:26]:<26} "
                  f"{str(row.get('Cap Category',''))[:14]:<14} "
                  f"{row['Prev Final']:>5.1f} "
                  f"{row['Final Score']:>5.1f} "
                  f"{row['Delta']:>+5.1f}  "
                  f"{row['Delta_Hist']:>+6.1f} "
                  f"{row['Delta_Peer']:>+6.1f} "
                  f"{row['Delta_Qual']:>+6.1f}  "
                  f"{float(row.get('Sector Score', 0)):>6.1f} "
                  f"{float(row.get('Cap Score', 0)):>6.1f}  "
                  f"{mcap_str(row.get('Market_Cap_Cr', 0)):>11}")

    print(f"\n  Stocks compared : {len(merged)} (in both quarters)")
    print(f"  New this quarter: {len(new_symbols)}")
    print(f"  Dropped         : {len(dropped_symbols)}")

    print(f"\n{'─'*65}")
    print(f"  🚀 CROSSED 60 — Entered Tier 1 territory ({len(crossed_60)} stocks)")
    print()
    print_fund_table(crossed_60)

    print(f"\n{'─'*65}")
    print(f"  ⬆️  IMPROVED >= 10 pts ({len(improved_10)} stocks)")
    print()
    print_fund_table(improved_10)

    print(f"\n{'─'*65}")
    print(f"  ↑  IMPROVED 5-9 pts ({len(improved_5)} stocks)")
    print()
    print_fund_table(improved_5)

    print(f"\n{'─'*65}")
    print(f"  ⚠️  DROPPED BELOW 60 ({len(dropped_60)} stocks)")
    print()
    print_fund_table(dropped_60)

    print(f"\n{'─'*65}")
    print(f"  ⬇️  DECLINED >= 10 pts ({len(declined_10)} stocks)")
    print()
    print_fund_table(declined_10)

    if new_symbols:
        print(f"\n{'─'*65}")
        print(f"  🆕 NEW STOCKS this quarter ({len(new_symbols)} stocks)")
        print()
        new_df = curr_df[curr_df['Symbol'].isin(new_symbols)].merge(
            fund_df[['Symbol', 'Cap Category', 'Sector Score',
                     'Cap Score', 'Market_Cap_Cr']],
            on='Symbol', how='left'
        ).sort_values('Final Score', ascending=False)
        print(f"  {'Symbol':<12} {'Sector':<26} {'Cap':<14} "
              f"{'Score':>5}  {'SecRnk':>6} {'CapRnk':>6}  {'MCap':>11}")
        print(f"  {'─'*85}")
        for _, row in new_df.iterrows():
            print(f"  {row['Symbol']:<12} "
                  f"{str(row.get('Sector',''))[:26]:<26} "
                  f"{str(row.get('Cap Category',''))[:14]:<14} "
                  f"{row['Final Score']:>5.1f}  "
                  f"{float(row.get('Sector Score', 0)):>6.1f} "
                  f"{float(row.get('Cap Score', 0)):>6.1f}  "
                  f"{mcap_str(row.get('Market_Cap_Cr', 0)):>11}")

    print(f"\n{'─'*65}")
    print(f"  SUMMARY")
    print(f"  {'─'*40}")
    print(f"  Crossed 60        : {len(crossed_60):>4}")
    print(f"  Improved >= 10    : {len(improved_10):>4}")
    print(f"  Improved 5-9      : {len(improved_5):>4}")
    print(f"  No change (<5pt)  : {len(merged[merged['Delta'].abs() < 5]):>4}")
    print(f"  Declined 5-9      : {len(merged[(merged['Delta'] <= -5) & (merged['Delta'] > -10)]):>4}")
    print(f"  Declined >= 10    : {len(declined_10):>4}")
    print(f"  Dropped below 60  : {len(dropped_60):>4}")
    print(f"  New stocks        : {len(new_symbols):>4}")
    print(f"  Stocks left 752   : {len(dropped_symbols):>4}")

# ── MAIN MENU ──────────────────────────────────────────────────
def run_sector_analysis():
    while True:
        print(f"\n{'='*65}")
        print(f"  AI STOCK SCREENER — SECTOR ANALYSIS")
        print(f"  {datetime.now().strftime('%d %B %Y')}")
        print(f"{'='*65}")
        print(f"""
  1. Long Term Stage Analysis
     → Sector index EMA stages (1=Accumulation to 4=Euphoria)
     → Stock confirmation via %EMA200, %Rec, %Mom, %BC
     → Drill into any stage/sector to see qualifying stocks

  2. Short Term Sector Trends
     → Weekly trend for all 22 sectors (from run_weekly.py)
     → Sorted by trend strength — Strong Uptrend to Strong Downtrend
     → Drill into any sector to see stocks by signal type

  3. Fund Score Improvements
     → Compare current vs previous quarter fundamental scores
     → Flags: Crossed 60, Improved 10+, Improved 5-9, Declined
     → Useful after every quarterly run
""")
        choice = input("  Enter choice (1/2/3/q to quit): ").strip().lower()

        if choice == '1':
            run_long_term_stage_analysis()
        elif choice == '2':
            run_short_term_sector_trends()
        elif choice == '3':
            run_fund_score_improvements()
        elif choice in ('q', 'quit', 'exit', '0'):
            print(f"\n  Exiting Sector Analysis.")
            break
        else:
            print(f"  Invalid choice. Enter 1, 2, 3 or q.")

# ── RUN ────────────────────────────────────────────────────────
run_sector_analysis()

print(f"\n{'='*60}")
print(f"  Sector Analysis complete!")
print(f"  {datetime.now().strftime('%d %B %Y  %H:%M')}")
print(f"{'='*60}")
