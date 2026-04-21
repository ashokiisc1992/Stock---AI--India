# ============================================================
# run_consolidation.py — AI Stock Screener (Indian Markets)
# Volume Profile + Consolidation Detection
#
# Run separately when needed (monthly or as required):
#   python run_consolidation.py
#
# Reads from existing price_data_full.pkl and
# technical_report_full.csv — no yfinance calls needed.
# Run run_weekly.py first to update prices.
#
# Menu:
#   1. Breakout Confirmed     (broke above range — last 3 weeks)
#   2. Near Breakout Watch    (within 10% of resistance)
#   3. Both
#
# Takes ~15 minutes for 752 stocks.
# Output saved to data/reports/consolidation/
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
REPORT_DIR   = os.path.join(DATA_DIR, 'reports', 'consolidation')
TEMP_DIR     = os.path.join(DATA_DIR, 'temp')

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR,   exist_ok=True)

today_str  = datetime.now().strftime('%Y-%m-%d')
today_file = datetime.now().strftime('%Y%m%d')

CAP_ORDER = ['Large Cap', 'Mini Large Cap', 'Mid Cap', 'Small Cap']

# ── LOAD DATA ─────────────────────────────────────────────────
print("=" * 65)
print("  AI Stock Screener — Consolidation Analysis")
print(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 65)
print("\nLoading data...")

PRICE_FILE   = os.path.join(PRICES_DIR, 'price_data_full.pkl')
INDIC_FILE   = os.path.join(PRICES_DIR, 'indicator_data_full.pkl')
TECH_FILE    = os.path.join(SCORES_DIR, 'technical_report_full.csv')
FUND_FILE    = os.path.join(FUND_DIR,   'fundamental_scores_full.csv')
PREFILT      = os.path.join(UNIVERSE_DIR,'prefilt_passed.csv')
BKOUT_FILE   = os.path.join(SCORES_DIR, 'breakout_tracker.csv')
CHECKPOINT   = os.path.join(TEMP_DIR,   'consol_checkpoint.pkl')

for fpath, label in [
    (INDIC_FILE, 'indicator_data_full.pkl'),
    (TECH_FILE,  'technical_report_full.csv'),
]:
    if not os.path.exists(fpath):
        print(f"\n  ERROR: {label} not found. Run run_weekly.py first.")
        exit(1)

# Use indicator_data (has OHLCV + Vol Ratio needed by consolidation detection)
with open(INDIC_FILE, 'rb') as f:
    price_data = pickle.load(f)
print(f"  Indicator data: {len(price_data)} stocks")

tech_df    = pd.read_csv(TECH_FILE)
fund_df    = pd.read_csv(FUND_FILE)
prefilt_df = pd.read_csv(PREFILT)

if 'Market_Cap_Cr' not in fund_df.columns:
    fund_df = fund_df.merge(
        prefilt_df[['Symbol', 'Market_Cap_Cr']], on='Symbol', how='left')

# Compute Sector Score and Cap Score for consolidation display
def classify_mcap(mcap_cr):
    try:
        v = float(mcap_cr)
        if v >= 20000:  return 'Large Cap'
        elif v >= 5000: return 'Mini Large Cap'
        elif v >= 1000: return 'Mid Cap'
        else:           return 'Small Cap'
    except:
        return 'Small Cap'

def compute_scores(df):
    df = df.copy()
    df['Cap Category'] = df['Market_Cap_Cr'].apply(classify_mcap)
    df['Sector Score'] = 0.0
    for sector in df['Sector'].unique():
        mask      = df['Sector'] == sector
        max_score = df[mask]['Final Score'].max() if 'Final Score' in df.columns else 0
        if max_score > 0:
            df.loc[mask, 'Sector Score'] = (
                df[mask]['Final Score'] / max_score * 10).round(1)
    df['Cap Score'] = 0.0
    for sector in df['Sector'].unique():
        for cap in ['Large Cap', 'Mini Large Cap', 'Mid Cap', 'Small Cap']:
            mask   = (df['Sector'] == sector) & (df['Cap Category'] == cap)
            subset = df[mask]
            if len(subset) == 0: continue
            max_score = subset['Final Score'].max() if 'Final Score' in df.columns else 0
            if max_score > 0:
                df.loc[mask, 'Cap Score'] = (
                    subset['Final Score'] / max_score * 10).round(1)
    return df

fund_df = compute_scores(fund_df)
# Merge scores into tech_df
tech_df = tech_df.merge(
    fund_df[['Symbol', 'Sector Score', 'Cap Score']],
    on='Symbol', how='left')

symbols = tech_df['Symbol'].tolist()
print(f"  Tech report   : {len(tech_df)} stocks")
print(f"  Universe      : {len(symbols)} symbols")

# ── HELPER FUNCTIONS ──────────────────────────────────────────
# Short sector names
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
    'Strong Uptrend ↑↑' : 'Str Up ↑↑',
    'Uptrend ↑'         : 'Uptrend ↑ ',
    'Weak Uptrend →↑'   : 'Wk Up →↑ ',
    'Sideways →'        : 'Sideways →',
    'Weak Downtrend →↓' : 'Wk Dn →↓ ',
    'Downtrend ↓'       : 'Downtrend ↓',
    'Strong Downtrend ↓↓':'Str Dn ↓↓',
}

def short_sector(s):
    return SECTOR_SHORT.get(str(s), str(s)[:14])

def short_trend(t):
    return TREND_SHORT.get(str(t), str(t)[:12])

def mcap_str(mcap_cr):
    try:
        v = float(mcap_cr)
        if v >= 100000: return f"Rs{v/100000:.1f}L Cr"
        return f"Rs{v:,.0f}Cr"
    except:
        return "Rs—"

# ── VOLUME PROFILE ─────────────────────────────────────────────
def calculate_volume_profile(df, lookback_days=252, bins=30):
    recent    = df.tail(lookback_days).copy()
    price_min = recent['Low'].min()
    price_max = recent['High'].max()
    if price_max <= price_min:
        return None
    price_bins      = np.linspace(price_min, price_max, bins + 1)
    volume_at_price = np.zeros(bins)
    for _, row in recent.iterrows():
        candle_low   = row['Low']
        candle_high  = row['High']
        candle_vol   = row['Volume']
        candle_range = candle_high - candle_low
        if candle_range == 0:
            continue
        for i in range(bins):
            bin_low      = price_bins[i]
            bin_high     = price_bins[i + 1]
            overlap_low  = max(candle_low,  bin_low)
            overlap_high = min(candle_high, bin_high)
            if overlap_high > overlap_low:
                overlap_pct          = (overlap_high - overlap_low) / candle_range
                volume_at_price[i]  += candle_vol * overlap_pct
    poc_idx   = np.argmax(volume_at_price)
    poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
    total_vol  = volume_at_price.sum()
    target_vol = total_vol * 0.70
    lower_idx  = poc_idx
    upper_idx  = poc_idx
    va_vol     = volume_at_price[poc_idx]
    while va_vol < target_vol:
        can_up   = upper_idx + 1 < bins
        can_down = lower_idx - 1 >= 0
        exp_up   = volume_at_price[upper_idx + 1] if can_up   else 0
        exp_down = volume_at_price[lower_idx - 1] if can_down else 0
        if not can_up and not can_down:
            break
        if exp_up >= exp_down and can_up:
            upper_idx += 1
            va_vol    += exp_up
        elif can_down:
            lower_idx -= 1
            va_vol    += exp_down
        else:
            break
    val           = (price_bins[lower_idx] + price_bins[lower_idx + 1]) / 2
    vah           = (price_bins[upper_idx] + price_bins[upper_idx + 1]) / 2
    max_vol       = volume_at_price.max()
    edge_vol      = (volume_at_price[:3].mean() + volume_at_price[-3:].mean()) / 2
    is_bell       = max_vol > edge_vol * 2.5
    current_price = df['Close'].iloc[-1]
    if current_price > vah * 1.02:   breakout_status = 'BREAKOUT_UP'
    elif current_price < val * 0.98: breakout_status = 'BREAKOUT_DOWN'
    elif current_price > vah * 0.98: breakout_status = 'AT_RESISTANCE'
    else:                            breakout_status = 'INSIDE'
    return {
        'poc_price': round(poc_price, 2), 'val': round(val, 2),
        'vah': round(vah, 2), 'is_bell': is_bell,
        'breakout_status': breakout_status,
        'current_price': round(current_price, 2),
    }

# ── CONSOLIDATION DETECTION ────────────────────────────────────
def detect_consolidation_dynamic(df, min_days=300, max_range_pct=50,
                                  max_total_drift=10, inside_pct_required=0.82):
    close = df['Close']
    n     = len(df)
    if n < min_days + 5:
        return {'found': False, 'is_valid': False, 'consol_days': 0}
    best_result = None
    for lookback in range(min_days, min(800, n - 5)):
        start_idx  = n - lookback
        segment    = df.iloc[start_idx:n]
        range_high = segment['High'].quantile(0.95)
        range_low  = segment['Low'].quantile(0.05)
        range_pct  = (range_high - range_low) / range_low * 100
        rng        = range_high - range_low
        if range_pct > max_range_pct or rng == 0:
            continue
        buffer      = 0.03
        bars_inside = (
            (segment['High'] <= range_high * (1 + buffer)) &
            (segment['Low']  >= range_low  * (1 - buffer))
        ).sum()
        pct_inside = bars_inside / len(segment)
        if pct_inside < inside_pct_required:
            continue
        x        = np.arange(len(segment))
        hi_drift = abs(np.polyfit(x, segment['High'].values,  1)[0] / segment['High'].mean()  * 100) * lookback
        lo_drift = abs(np.polyfit(x, segment['Low'].values,   1)[0] / segment['Low'].mean()   * 100) * lookback
        cl_drift = abs(np.polyfit(x, segment['Close'].values, 1)[0] / segment['Close'].mean() * 100) * lookback
        if hi_drift > max_total_drift or lo_drift > max_total_drift or cl_drift > max_total_drift:
            continue
        price_start  = segment['Close'].iloc[0]
        price_end    = segment['Close'].iloc[-1]
        price_change = (price_end - price_start) / price_start * 100
        if abs(price_change) > 25:
            continue
        end_vs_range = (price_end - range_low) / rng * 100
        if end_vs_range < -10:
            continue
        close_arr    = segment['Close'].values
        peak_idx     = np.argmax(close_arr)
        peak_pos     = peak_idx / lookback * 100
        peak_price   = close_arr.max()
        peak_vs_rng  = (peak_price  - range_low) / rng * 100
        start_vs_rng = (price_start - range_low) / rng * 100
        if (peak_pos > 20 and peak_pos < 80 and start_vs_rng < 35
                and end_vs_range < 35 and peak_vs_rng > 90):
            continue
        pre_start   = max(0, start_idx - 120)
        pre_segment = df.iloc[pre_start:start_idx]
        pre_close   = pre_segment['Close'].values
        if len(pre_close) > 5:
            pre_x    = np.arange(len(pre_close))
            pre_norm = np.polyfit(pre_x, pre_close, 1)[0] / pre_close.mean() * 100
            if pre_norm < (-0.15 if lookback < 365 else -0.60):
                continue
            pre_high = pre_segment['High'].max()
            drop_pct = (range_high - pre_high) / pre_high * 100
            if drop_pct < (-20 if lookback < 365 else -60):
                continue
        if lookback > (best_result['consol_days'] if best_result else 0):
            best_result = {
                'consol_days': lookback,
                'range_high' : round(range_high, 2),
                'range_low'  : round(range_low,  2),
                'range_pct'  : round(range_pct,  2),
                'pct_inside' : round(pct_inside,  3),
                'start_idx'  : start_idx,
            }
    if best_result is None:
        return {'found': False, 'is_valid': False, 'consol_days': 0}
    consol_days   = best_result['consol_days']
    range_high    = best_result['range_high']
    range_low     = best_result['range_low']
    current_price = close.iloc[-1]
    consol_df     = df.iloc[best_result['start_idx']:n]
    res_touches   = int((consol_df['High'] >= range_high * 0.98).sum())
    sup_touches   = int((consol_df['Low']  <= range_low  * 1.02).sum())
    pct_above     = (current_price - range_high) / range_high * 100
    vol_col       = 'Vol_Ratio' if 'Vol_Ratio' in df.columns else 'Vol Ratio'
    vol_ratio     = round(df[vol_col].tail(5).max(), 2)
    vp            = calculate_volume_profile(consol_df, lookback_days=consol_days, bins=20)
    has_bell      = vp['is_bell'] if vp else False
    if consol_days < 365:   duration_label = f'{consol_days}d (under 1 year)'
    elif consol_days < 500: duration_label = f'{consol_days}d (1-1.5 years)'
    elif consol_days < 730: duration_label = f'{consol_days}d (1.5-2 years)'
    else:                   duration_label = f'{consol_days}d (2+ years)'
    is_valid = res_touches >= 2 and sup_touches >= 2
    return {
        'found': True, 'is_valid': is_valid,
        'consol_days': consol_days, 'duration_label': duration_label,
        'range_high': range_high, 'range_low': range_low,
        'range_pct': best_result['range_pct'],
        'pct_above': round(pct_above, 2),
        'is_breaking_out': current_price >= range_high * 0.98,
        'breakout_volume': vol_ratio, 'has_bell': has_bell,
        'current_price': round(current_price, 2),
        'resistance_touches': res_touches, 'support_touches': sup_touches,
    }

def get_consolidation_info(df):
    result = detect_consolidation_dynamic(df)
    if not result.get('found') or not result.get('is_valid'):
        return {
            'consolidating': False, 'consol_days': 0,
            'duration_label': 'No consolidation',
            'range_high': None, 'range_low': None, 'range_pct': None,
            'pct_to_breakout': None, 'is_breaking_out': False,
            'breakout_volume': None, 'has_bell': False,
            'resistance_touches': 0, 'support_touches': 0,
            'current_price': df['Close'].iloc[-1],
        }
    return {
        'consolidating': True,
        'consol_days': result['consol_days'],
        'duration_label': result['duration_label'],
        'range_high': result['range_high'], 'range_low': result['range_low'],
        'range_pct': result['range_pct'],
        'pct_to_breakout': result['pct_above'],
        'is_breaking_out': result['is_breaking_out'],
        'breakout_volume': result['breakout_volume'],
        'has_bell': result['has_bell'],
        'resistance_touches': result['resistance_touches'],
        'support_touches': result['support_touches'],
        'current_price': result['current_price'],
    }

def get_breakout_vol_inference(week_vol, break_vol):
    if   break_vol >= 2.0: return f"Week Vol:{week_vol:.2f}x | Break Vol:{break_vol:.2f}x ⚡⚡ → Strong breakout volume building"
    elif break_vol >= 1.5: return f"Week Vol:{week_vol:.2f}x | Break Vol:{break_vol:.2f}x ⚡  → Volume building toward breakout"
    elif break_vol >= 1.0: return f"Week Vol:{week_vol:.2f}x | Break Vol:{break_vol:.2f}x ~  → Normal volume, no breakout pressure"
    elif break_vol >= 0.7: return f"Week Vol:{week_vol:.2f}x | Break Vol:{break_vol:.2f}x    → Quiet, below consolidation avg"
    else:                  return f"Week Vol:{week_vol:.2f}x | Break Vol:{break_vol:.2f}x    → Very quiet, no breakout interest"

# ── MAIN COMPUTATION ───────────────────────────────────────────
print("\nComputing consolidation for all stocks (~15 mins)...")
print(f"  Started : {datetime.now().strftime('%H:%M:%S')}")

# Load checkpoint if exists
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT, 'rb') as f:
        ckpt = pickle.load(f)
    consol_results = ckpt['results']
    done_syms      = ckpt['done']
    print(f"  Resuming from checkpoint — {len(done_syms)} already done")
else:
    consol_results = {}
    done_syms      = set()

remaining = [s for s in symbols if s in price_data and s not in done_syms]

for i, symbol in enumerate(remaining):
    try:
        df         = price_data[symbol]
        tech_row   = tech_df[tech_df['Symbol'] == symbol]
        fund_row   = fund_df[fund_df['Symbol'] == symbol]

        if len(tech_row) == 0:
            continue

        t          = tech_row.iloc[0]
        mcap_cr    = float(fund_row['Market_Cap_Cr'].values[0]) \
                     if len(fund_row) > 0 and 'Market_Cap_Cr' in fund_row.columns else 0
        cap_cat    = classify_mcap(mcap_cr)
        sec_score  = float(fund_row['Sector Score'].values[0]) \
                     if len(fund_row) > 0 and 'Sector Score' in fund_row.columns else 0.0
        cap_score  = float(fund_row['Cap Score'].values[0]) \
                     if len(fund_row) > 0 and 'Cap Score' in fund_row.columns else 0.0

        consol_info = get_consolidation_info(df)

        if consol_info['consolidating']:
            consol_days = consol_info['consol_days']
            # Compute breakout vol: this week avg vs consolidation period avg
            week_vol   = df['Volume'].iloc[-5:].mean()
            consol_avg = df['Volume'].iloc[-consol_days:-5].mean() \
                         if len(df) >= consol_days + 5 else week_vol
            break_vol  = round(week_vol / consol_avg if consol_avg > 0 else 1.0, 2)

            # Vol 5D Ratio from tech_df
            vol_5d = float(t.get('Vol 5D Ratio', t.get('Vol Ratio', 1.0)) or 1.0)

            consol_results[symbol] = {
                'Symbol'           : symbol,
                'Sector'           : str(t.get('Sector', '')),
                'Cap Category'     : cap_cat,
                'Current Price'    : float(t.get('Current Price', 0)),
                'Market Cap Cr'    : round(mcap_cr, 2),
                'Fund Score'       : float(t.get('Fund Score', 0) or 0),
                'Sector Score'     : sec_score,
                'Cap Score'        : cap_score,
                'RSI'              : float(t.get('RSI', 0)),
                'ADX'              : float(t.get('ADX', 0)),
                'Vol 5D Ratio'     : vol_5d,
                'Breakout Vol'     : break_vol,
                'In Consolidation' : True,
                'Consol Days'      : consol_info['consol_days'],
                'Consol Label'     : consol_info['duration_label'],
                'Pct to Breakout'  : consol_info['pct_to_breakout'] or 0,
                'Range High'       : consol_info['range_high'],
                'Range Low'        : consol_info['range_low'],
                'Sector Trend'     : str(t.get('Sector Trend', '—')),
            }
        done_syms.add(symbol)

    except Exception:
        pass

    if (i + 1) % 50 == 0 or (i + 1) == len(remaining):
        pct = (i + 1) / len(remaining) * 100
        print(f"  [{i+1:4d}/{len(remaining)}] {pct:5.1f}% | "
              f"Consolidating: {len(consol_results)} | "
              f"Time: {datetime.now().strftime('%H:%M:%S')}")
        with open(CHECKPOINT, 'wb') as f:
            pickle.dump({'results': consol_results, 'done': done_syms}, f)

print(f"\n  Done — {len(consol_results)} stocks in consolidation")

# Build consolidation dataframe
if len(consol_results) == 0:
    print("  No consolidation found. Exiting.")
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)
    exit(0)

consol_df = pd.DataFrame(list(consol_results.values()))

# ── BREAKOUT TRACKER ──────────────────────────────────────────
if os.path.exists(BKOUT_FILE):
    bk_df = pd.read_csv(BKOUT_FILE)
    bk_df = bk_df[bk_df['Weeks_Count'] <= 3].copy()
else:
    bk_df = pd.DataFrame(columns=[
        'Symbol', 'First_Breakout_Date', 'Weeks_Count',
        'Last_Pct', 'Cap_Category', 'Sector', 'Fund_Score'
    ])

current_breakouts = consol_df[consol_df['Pct to Breakout'] > 0].copy()
updated_rows = []

for _, row in current_breakouts.iterrows():
    sym      = row['Symbol']
    existing = bk_df[bk_df['Symbol'] == sym]
    if len(existing) > 0:
        weeks = int(existing.iloc[0]['Weeks_Count']) + 1
        if weeks <= 3:
            updated_rows.append({
                'Symbol'             : sym,
                'First_Breakout_Date': existing.iloc[0]['First_Breakout_Date'],
                'Weeks_Count'        : weeks,
                'Last_Pct'           : round(row['Pct to Breakout'], 2),
                'Cap_Category'       : row['Cap Category'],
                'Sector'             : row['Sector'],
                'Fund_Score'         : row['Fund Score'],
            })
    else:
        updated_rows.append({
            'Symbol'             : sym,
            'First_Breakout_Date': today_str,
            'Weeks_Count'        : 1,
            'Last_Pct'           : round(row['Pct to Breakout'], 2),
            'Cap_Category'       : row['Cap Category'],
            'Sector'             : row['Sector'],
            'Fund_Score'         : row['Fund Score'],
        })

for _, row in bk_df.iterrows():
    sym           = row['Symbol']
    already_added = any(r['Symbol'] == sym for r in updated_rows)
    if not already_added and int(row['Weeks_Count']) < 3:
        c_row = consol_df[consol_df['Symbol'] == sym]
        if len(c_row) > 0:
            pct = float(c_row.iloc[0]['Pct to Breakout'])
            if pct > -5:
                updated_rows.append({
                    'Symbol'             : sym,
                    'First_Breakout_Date': row['First_Breakout_Date'],
                    'Weeks_Count'        : int(row['Weeks_Count']),
                    'Last_Pct'           : round(pct, 2),
                    'Cap_Category'       : row['Cap_Category'],
                    'Sector'             : row['Sector'],
                    'Fund_Score'         : row['Fund_Score'],
                })

new_bk_df = pd.DataFrame(updated_rows) if updated_rows else pd.DataFrame(
    columns=['Symbol','First_Breakout_Date','Weeks_Count',
             'Last_Pct','Cap_Category','Sector','Fund_Score'])
new_bk_df.to_csv(BKOUT_FILE, index=False)

# Merge tracker into consol_df
if len(new_bk_df) > 0:
    consol_df = consol_df.merge(
        new_bk_df[['Symbol', 'Weeks_Count', 'First_Breakout_Date']],
        on='Symbol', how='left')
else:
    consol_df['Weeks_Count']         = None
    consol_df['First_Breakout_Date'] = None

# Clean checkpoint
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)

# ── REPORT GENERATION ─────────────────────────────────────────
def cap_order_key(x):
    return CAP_ORDER.index(x) if x in CAP_ORDER else 99

def generate_report():
    lines = []
    def p(line=''): lines.append(str(line))

    p(f"{'─'*86}")
    p(f"  BREAKOUT CONFIRMED  ({len(breakout_confirmed)} stocks)")
    p(f"  Stocks that broke above consolidation range")
    p(f"  Shown for 3 weeks from breakout date")
    p(f"{'─'*86}")
    p(f"  {'Symbol':12} "
      f"{'Sector':<14} {'Trend':<11} {'Price':>9}  {'MCap':>11}  {'Above%':>7}  {'Weeks':>6}  {'WkVol':>6}  {'RSI':>5}  {'SecRnk':>6}  {'CapRnk':>6}  Breakout Date")
    p(f"  {'─'*100}")

    current_cap = None
    for _, row in breakout_confirmed.iterrows():
        cap = row['Cap Category']
        if cap != current_cap:
            current_cap = cap
            cap_short   = {'Large Cap':'L','Mini Large Cap':'ML',
                           'Mid Cap':'M','Small Cap':'S'}.get(cap,'?')
            p(f"\n  [{cap_short}] {cap}")
        weeks    = int(row['Weeks_Count']) if pd.notna(row.get('Weeks_Count')) else 1
        bk_date  = row['First_Breakout_Date'] \
                   if pd.notna(row.get('First_Breakout_Date')) else today_str
        wvol     = float(row['Vol 5D Ratio'])
        wvol_tag = '⚡' if wvol >= 1.5 else ' '
        p(f"  {row['Symbol']:12} "
          f"{short_sector(row.get('Sector','')):<14}  "
          f"{short_trend(row.get('Sector Trend','—')):<11}  "
          f"Rs{row['Current Price']:>8.2f}  "
          f"{mcap_str(row['Market Cap Cr']):>11}  "
          f"{row['Pct to Breakout']:>+6.1f}%  "
          f"Wk{weeks}/3  "
          f"{wvol:.2f}x{wvol_tag}  "
          f"RSI{row['RSI']:>3.0f}  "
          f"{float(row.get('Sector Score',0)):>6.1f}  "
          f"{float(row.get('Cap Score',0)):>6.1f}  "
          f"[{bk_date}]")

    p(f"\n{'─'*86}")
    p(f"  NEAR BREAKOUT BASE WATCH  ({len(consol_near)} stocks within 10% of breakout)")
    p(f"  Break Vol > 1.5x = volume building — high priority watch")
    p(f"{'─'*86}")
    p(f"  {'Symbol':12} "
      f"{'Sector':<14} {'Trend':<11} {'Price':>9}  {'MCap':>11}  {'Days':>5}  {'ToBreak':>7}  {'WkVol':>6}  {'BrkVol':>7}  {'SecRnk':>6}  {'CapRnk':>6}  Status")
    p(f"  {'─'*100}")

    for _, row in consol_near.iterrows():
        pct      = row['Pct to Breakout']
        bvol     = float(row['Breakout Vol'])
        wvol     = float(row['Vol 5D Ratio'])
        status   = 'NEAR BREAKOUT ⚡' if pct > -5 else 'Approaching'
        bvol_tag = '⚡' if bvol >= 1.5 else '~' if bvol >= 0.8 else ' '
        p(f"  {row['Symbol']:12} "
          f"{short_sector(row.get('Sector','')):<14}  "
          f"{short_trend(row.get('Sector Trend','—')):<11}  "
          f"Rs{row['Current Price']:>8.2f}  "
          f"{mcap_str(row['Market Cap Cr']):>11}  "
          f"{int(row['Consol Days']):>5}d  "
          f"{pct:>+6.1f}%  "
          f"{wvol:.2f}x  "
          f"{bvol:.2f}x{bvol_tag}  "
          f"{float(row.get('Sector Score',0)):>6.1f}  "
          f"{float(row.get('Cap Score',0)):>6.1f}  "
          f"[{status}]")

    p(f"\n{'─'*86}")
    p(f"  WkVol = this week avg / prior 15d avg")
    p(f"  BrkVol = this week avg / consolidation period avg")
    p(f"  ⚡ = volume above 1.5x threshold")
    p(f"{'─'*86}")
    return lines

# ── DATA SLICES ───────────────────────────────────────────────
breakout_confirmed = consol_df[
    consol_df['Weeks_Count'].notna() &
    (consol_df['Pct to Breakout'] > 0)
].copy()
if len(breakout_confirmed) > 0:
    breakout_confirmed['_cap_order'] = breakout_confirmed['Cap Category'].apply(cap_order_key)
    breakout_confirmed = breakout_confirmed.sort_values(
        ['_cap_order', 'Pct to Breakout'], ascending=[True, False])

consol_near = consol_df[
    (consol_df['Pct to Breakout'] >= -10) &
    (consol_df['Pct to Breakout'] <= 0)
].sort_values('Pct to Breakout', ascending=False).reset_index(drop=True)

# ── MENU ──────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  CONSOLIDATION ANALYSIS")
print(f"  Breakout Confirmed : {len(breakout_confirmed)} stocks")
print(f"  Near Breakout      : {len(consol_near)} stocks")
print(f"{'='*65}")
print(f"\n  1. Breakout Confirmed")
print(f"  2. Near Breakout Watch")
print(f"  3. Both")
print()

choice = input("  Enter choice (1/2/3) [default 3]: ").strip() or '3'

lines = []
def p(line=''): lines.append(str(line))

now_label = datetime.now().strftime('%d %B %Y %H:%M')
p(f"{'='*86}")
p(f"  AI STOCK SCREENER — CONSOLIDATION ANALYSIS")
p(f"  {now_label}  |  Universe: {len(symbols)} stocks")
p(f"  In consolidation: {len(consol_df)} | Breakout confirmed: {len(breakout_confirmed)} | Near breakout: {len(consol_near)}")
p(f"{'='*86}")

if choice in ('1', '3'):
    p(f"\n{'─'*86}")
    p(f"  BREAKOUT CONFIRMED  ({len(breakout_confirmed)} stocks)")
    p(f"  Stocks that broke above consolidation range — shown for 3 weeks")
    p(f"{'─'*86}")
    if len(breakout_confirmed) == 0:
        p(f"  None currently.")
    else:
        p(f"  {'Symbol':12} "
          f"{'Sector':<14} {'Trend':<11} {'Price':>9}  {'MCap':>11}  {'Above%':>7}  {'Weeks':>6}  {'WkVol':>6}  {'RSI':>5}  {'SecRnk':>6}  {'CapRnk':>6}  Breakout Date")
        p(f"  {'─'*100}")
        current_cap = None
        for _, row in breakout_confirmed.iterrows():
            cap = row['Cap Category']
            if cap != current_cap:
                current_cap = cap
                cap_short   = {'Large Cap':'L','Mini Large Cap':'ML',
                               'Mid Cap':'M','Small Cap':'S'}.get(cap,'?')
                p(f"\n  [{cap_short}] {cap}")
            weeks    = int(row['Weeks_Count']) if pd.notna(row.get('Weeks_Count')) else 1
            bk_date  = row['First_Breakout_Date'] \
                       if pd.notna(row.get('First_Breakout_Date')) else today_str
            wvol     = float(row['Vol 5D Ratio'])
            wvol_tag = '⚡' if wvol >= 1.5 else ' '
            p(f"  {row['Symbol']:12} "
              f"{short_sector(row.get('Sector','')):<14}  "
              f"{short_trend(row.get('Sector Trend','—')):<11}  "
              f"Rs{row['Current Price']:>8.2f}  "
              f"{mcap_str(row['Market Cap Cr']):>11}  "
              f"{row['Pct to Breakout']:>+6.1f}%  "
              f"Wk{weeks}/3  "
              f"{wvol:.2f}x{wvol_tag}  "
              f"RSI{row['RSI']:>3.0f}  "
              f"{float(row.get('Sector Score',0)):>6.1f}  "
              f"{float(row.get('Cap Score',0)):>6.1f}  "
              f"[{bk_date}]")

if choice in ('2', '3'):
    p(f"\n{'─'*86}")
    p(f"  NEAR BREAKOUT BASE WATCH  ({len(consol_near)} stocks within 10% of breakout)")
    p(f"  Break Vol > 1.5x = volume building — high priority watch")
    p(f"{'─'*86}")
    p(f"  {'Symbol':12} "
      f"{'Sector':<14} {'Trend':<11} {'Price':>9}  {'MCap':>11}  {'Days':>5}  {'ToBreak':>7}  {'WkVol':>6}  {'BrkVol':>7}  {'SecRnk':>6}  {'CapRnk':>6}  Status")
    p(f"  {'─'*100}")
    if len(consol_near) == 0:
        p(f"  None currently.")
    else:
        for _, row in consol_near.iterrows():
            pct      = row['Pct to Breakout']
            bvol     = float(row['Breakout Vol'])
            wvol     = float(row['Vol 5D Ratio'])
            status   = 'NEAR BREAKOUT ⚡' if pct > -5 else 'Approaching'
            bvol_tag = '⚡' if bvol >= 1.5 else '~' if bvol >= 0.8 else ' '
            p(f"  {row['Symbol']:12} "
              f"{short_sector(row.get('Sector','')):<14}  "
              f"{short_trend(row.get('Sector Trend','—')):<11}  "
              f"Rs{row['Current Price']:>8.2f}  "
              f"{mcap_str(row['Market Cap Cr']):>11}  "
              f"{int(row['Consol Days']):>5}d  "
              f"{pct:>+6.1f}%  "
              f"{wvol:.2f}x  "
              f"{bvol:.2f}x{bvol_tag}  "
              f"{float(row.get('Sector Score',0)):>6.1f}  "
              f"{float(row.get('Cap Score',0)):>6.1f}  "
              f"[{status}]")

p(f"\n{'─'*86}")
p(f"  WkVol  = this week avg / prior 15d avg")
p(f"  BrkVol = this week avg / consolidation period avg  |  ⚡ >= 1.5x")
p(f"{'─'*86}")

# Print to screen
for line in lines:
    print(line)

# Save report
rpath = os.path.join(REPORT_DIR, f'consolidation_{today_file}.txt')
with open(rpath, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"\n  Report saved: {rpath}")
print(f"\n{'='*65}")
print(f"  Done! — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*65}")
