# ============================================================
# run_portfolio.py — AI Stock Screener (Indian Markets)
# Day 13: Portfolio Manager
#
# Run manually:
#   python run_portfolio.py
#
# Menu:
#   1. Long Term Portfolio Review
#   2. Swing Trading Portfolio Review
#   3. Both Reviews
#   4. Create / Update Model Portfolio
#   5. Review Model Portfolios (P&L tracking)
#   6. All Reviews (LT + Swing + Model)
#
# Portfolio files:
#   data/portfolio/long_term_portfolio.csv
#   data/portfolio/swing_portfolio.csv
#   data/portfolio/model_portfolios/<name>.csv
#
# Reports saved to:
#   data/reports/portfolio/
# ============================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime
from difflib import get_close_matches

# ── PATHS ─────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, 'data')
UNIVERSE_DIR   = os.path.join(DATA_DIR, 'universe')
FUND_DIR       = os.path.join(DATA_DIR, 'fundamentals')
SCORES_DIR     = os.path.join(DATA_DIR, 'scores')
PORTFOLIO_DIR  = os.path.join(DATA_DIR, 'portfolio')
MODEL_PORT_DIR = os.path.join(PORTFOLIO_DIR, 'model_portfolios')
REPORTS_DIR    = os.path.join(DATA_DIR, 'reports', 'portfolio')

for d in [PORTFOLIO_DIR, MODEL_PORT_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

LT_FILE    = os.path.join(PORTFOLIO_DIR, 'long_term_portfolio.csv')
SWING_FILE = os.path.join(PORTFOLIO_DIR, 'swing_portfolio.csv')

today_str  = datetime.now().strftime('%Y-%m-%d')
today_file = datetime.now().strftime('%Y%m%d')

# ── CONSTANTS ─────────────────────────────────────────────────
SWING_CAPITAL  = 150000
DRAWDOWN_LIMIT = 0.08
MAX_HOLD_DAYS  = 30
MIN_CARRY_CONF = 65.0
MIN_CONF       = 40.0
MIN_FUND_SWING = 40.0
TOP_N          = 10
MAX_REVERSAL   = 2
SWING_TOP_N    = 10
CAP_ORDER      = ['Large Cap', 'Mini Large Cap', 'Mid Cap', 'Small Cap']

LT_COLS = [
    'Symbol', 'Entry_Price', 'Entry_Date', 'Quantity', 'Cap_Category',
    'Sector_Rank_At_Entry', 'Cap_Rank_At_Entry',
    'Sector_Rank_Change', 'Cap_Rank_Change',
    'Accumulation_Mode', 'Notes'
]

SWING_COLS = [
    'Symbol', 'Entry_Price', 'Entry_Date', 'Quantity', 'Cap_Category',
    'Sector_Rank_At_Entry', 'Cap_Rank_At_Entry',
    'Sector_Rank_Change', 'Cap_Rank_Change',
    'Cycle_Start_Date', 'Notes'
]

MODEL_COLS = [
    'Symbol', 'Entry_Price', 'Entry_Date', 'Quantity', 'Cap_Category',
    'Sector_Rank_At_Entry', 'Cap_Rank_At_Entry',
    'Sector_Rank_Change', 'Cap_Rank_Change', 'Notes'
]

# ── LOAD DATA ──────────────────────────────────────────────────
print("=" * 60)
print("  AI Stock Screener — Portfolio Manager")
print(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)
print("\nLoading data...")

tech_df    = pd.read_csv(os.path.join(SCORES_DIR,   'technical_report_full.csv'))
fund_df    = pd.read_csv(os.path.join(FUND_DIR,     'fundamental_scores_full.csv'))
prefilt_df = pd.read_csv(os.path.join(UNIVERSE_DIR, 'prefilt_passed.csv'))
quality_df = pd.read_csv(os.path.join(UNIVERSE_DIR, 'quality_passed.csv'))

if 'Market_Cap_Cr' not in fund_df.columns:
    fund_df = fund_df.merge(
        prefilt_df[['Symbol', 'Market_Cap_Cr']], on='Symbol', how='left'
    )

valid_symbols = set(quality_df['Symbol'].tolist())

# Compute Sector Score + Cap Score if missing
if 'Sector Score' not in tech_df.columns or tech_df['Sector Score'].isna().all():
    fund_df['Cap Category'] = pd.cut(
        fund_df['Market_Cap_Cr'].fillna(0),
        bins=[0, 1000, 5000, 20000, float('inf')],
        labels=['Small Cap', 'Mid Cap', 'Mini Large Cap', 'Large Cap']
    ).astype(str)
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
            mask   = ((fund_df['Sector'] == sector) &
                      (fund_df['Cap Category'] == cap))
            subset = fund_df[mask]
            if len(subset) == 0:
                continue
            max_score = subset['Final Score'].max()
            if max_score > 0:
                fund_df.loc[mask, 'Cap Score'] = (
                    subset['Final Score'] / max_score * 10).round(1)
    for col in ['Sector Score', 'Cap Score', 'Sector Score_x', 'Cap Score_x',
                'Sector Score_y', 'Cap Score_y']:
        if col in tech_df.columns:
            tech_df = tech_df.drop(columns=[col])
    tech_df = tech_df.merge(
        fund_df[['Symbol', 'Final Score', 'Sector Score',
                 'Cap Score', 'Cap Category']],
        on='Symbol', how='left'
    )

print(f"     Tech report  : {len(tech_df)} stocks")
print(f"     Valid symbols: {len(valid_symbols)}")

# ── PRICE FILTER SETUP ─────────────────────────────────────────
print("\n" + "─" * 60)
print("  PRICE FILTER FOR RECOMMENDATIONS")
print("─" * 60)
print("  1. Standard     (Price > EMA50 > EMA200)")
print("  2. Early Entry  (Price > EMA50 > EMA200, within X% of EMA50)")
print("  3. Both         (show all, mark [HIGH] if > X% above EMA50)")
print()
price_filter_choice = input("  Enter choice (1/2/3) [default 3]: ").strip()
if price_filter_choice not in ['1', '2', '3']:
    price_filter_choice = '3'

EARLY_ENTRY_PCT = 10.0  # default
if price_filter_choice in ['2', '3']:
    try:
        pct_input = input(
            f"  Max % above EMA50 for Early Entry [default {EARLY_ENTRY_PCT}]: "
        ).strip()
        if pct_input:
            EARLY_ENTRY_PCT = float(pct_input)
    except:
        EARLY_ENTRY_PCT = 10.0
    print(f"  Early entry threshold: {EARLY_ENTRY_PCT}% above EMA50")

print(f"\n  Filter mode    : "
      f"{'Standard' if price_filter_choice=='1' else 'Early Entry' if price_filter_choice=='2' else 'Both (Standard + [HIGH] markers)'}")
if price_filter_choice in ['2', '3']:
    print(f"  Early entry cap: {EARLY_ENTRY_PCT}% above EMA50")
print("─" * 60)

# ── HELPER FUNCTIONS ───────────────────────────────────────────
SECTOR_SHORT = {
    'Information Technology'            : 'IT',
    'Financial Services'                : 'Financial',
    'Chemicals'                         : 'Chemicals',
    'Healthcare'                        : 'Healthcare',
    'Consumer Durables'                 : 'Con Durables',
    'Consumer Services'                 : 'Con Services',
    'Fast Moving Consumer Goods'        : 'FMCG',
    'Capital Goods'                     : 'Capital Goods',
    'Automobile and Auto Components'    : 'Auto',
    'Construction'                      : 'Construction',
    'Construction Materials'            : 'Const Matrl',
    'Textiles'                          : 'Textiles',
    'Services'                          : 'Services',
    'Metals & Mining'                   : 'Metals',
    'Oil, Gas & Consumable Fuels'       : 'Oil & Gas',
    'Power'                             : 'Power',
    'Realty'                            : 'Realty',
    'Utilities'                         : 'Utilities',
    'Telecommunication'                 : 'Telecom',
    'Media, Entertainment & Publication': 'Media',
    'Media Entertainment & Publication' : 'Media',
    'Forest Materials'                  : 'Forest Matrl',
    'Diversified'                       : 'Diversified',
    'Pharmaceuticals'                   : 'Pharma',
    'Banking'                           : 'Banking',
    'Cement'                            : 'Cement',
    'Defence'                           : 'Defence',
    'Agriculture'                       : 'Agriculture',
}

SECTOR_TREND_SHORT = {
    'Strong Uptrend ↑↑'  : 'Str Up ↑↑',
    'Uptrend ↑'          : 'Uptrend ↑',
    'Weak Uptrend →↑'    : 'Wk Up →↑',
    'Sideways →'         : 'Sideways →',
    'Weak Downtrend →↓'  : 'Wk Dn →↓',
    'Downtrend ↓'        : 'Downtrend ↓',
    'Strong Downtrend ↓↓': 'Str Dn ↓↓',
    'No data'            : 'No data',
}

def mcap_str(mcap_cr):
    if mcap_cr is None or pd.isna(mcap_cr):
        return '—'
    if mcap_cr >= 100000:
        return f'Rs{mcap_cr/100000:.1f}L Cr'
    elif mcap_cr >= 1000:
        return f'Rs{mcap_cr:,.0f}Cr'
    else:
        return f'Rs{mcap_cr:.0f}Cr'

def tier_abbr(tier):
    tier = str(tier)
    if 'BUY NOW (Momentum)' in tier: return 'T1-Mom'
    if 'BUY NOW (Reversal)' in tier: return 'T1-Rev'
    if 'BREAKOUT IMMINENT'  in tier: return 'T1-Brk'
    if 'WATCHLIST'          in tier: return 'T2-Wtch'
    if 'BASE BUILDING'      in tier: return 'T2-Base'
    return 'T3'

def short_sector(name):
    return SECTOR_SHORT.get(str(name), str(name)[:14])

def short_trend(trend):
    t = str(trend)
    for k, v in SECTOR_TREND_SHORT.items():
        if k in t:
            return v
    return t[:12]

def short_ml(pred):
    return 'Bull Cont' if pred == 'Bullish Continual' else str(pred)[:12]

def is_sector_bullish(row):
    return 'Uptrend' in str(row.get('Sector Trend', ''))

def is_sector_strong_bull(row):
    return 'Strong Uptrend' in str(row.get('Sector Trend', ''))

def passes_ema_filter(row):
    """Standard EMA filter: Price > EMA50 > EMA200"""
    try:
        price  = float(row.get('Current Price', 0) or 0)
        ema50  = float(row.get('EMA50',  0) or 0)
        ema200 = float(row.get('EMA200', 0) or 0)
        return price > ema50 > ema200
    except:
        return False

def is_early_entry(row, max_pct=None):
    """
    True if price is within max_pct% above EMA50.
    Used to flag stocks that haven't run too far yet.
    """
    if max_pct is None:
        max_pct = EARLY_ENTRY_PCT
    try:
        price = float(row.get('Current Price', 0) or 0)
        ema50 = float(row.get('EMA50',  0) or 0)
        if ema50 <= 0:
            return False
        pct_above = (price - ema50) / ema50 * 100
        return pct_above <= max_pct
    except:
        return False

def pct_above_ema50(row):
    """Return % price is above EMA50."""
    try:
        price = float(row.get('Current Price', 0) or 0)
        ema50 = float(row.get('EMA50',  0) or 0)
        if ema50 <= 0:
            return 0
        return round((price - ema50) / ema50 * 100, 1)
    except:
        return 0

def passes_price_filter(row):
    """
    Apply price filter based on user's choice at startup.
    Mode 1: Standard only
    Mode 2: Early entry only (within X% of EMA50)
    Mode 3: Both (standard passes, early entry is just a marker)
    """
    ema_ok = passes_ema_filter(row)
    if not ema_ok:
        return False
    if price_filter_choice == '1':
        return True  # standard — all EMA-confirmed stocks
    elif price_filter_choice == '2':
        return is_early_entry(row)  # only within X% of EMA50
    else:
        return True  # mode 3 — show all, mark [HIGH] separately

def get_high_tag(row):
    """
    Returns '[HIGH]' if stock has run > EARLY_ENTRY_PCT above EMA50.
    Only used in mode 3.
    """
    if price_filter_choice == '3' and not is_early_entry(row):
        return '[HIGH]'
    return '      '

def get_current_ranks(symbol):
    row = tech_df[tech_df['Symbol'] == symbol]
    if len(row) == 0:
        return None, None
    r = row.iloc[0]
    return (float(r.get('Sector Score', 0) or 0),
            float(r.get('Cap Score',    0) or 0))

def update_rank_changes(df):
    for idx, row in df.iterrows():
        sym               = str(row['Symbol'])
        curr_sec, curr_cap = get_current_ranks(sym)
        entry_sec         = float(row.get('Sector_Rank_At_Entry', 0) or 0)
        entry_cap         = float(row.get('Cap_Rank_At_Entry',    0) or 0)
        if curr_sec is not None:
            df.at[idx, 'Sector_Rank_Change'] = round(curr_sec - entry_sec, 1)
            df.at[idx, 'Cap_Rank_Change']    = round(curr_cap - entry_cap, 1)
    return df

def load_portfolio(filepath, cols):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        for c in cols:
            if c not in df.columns:
                df[c] = None
        return df
    return pd.DataFrame(columns=cols)

def save_portfolio(df, filepath):
    df.to_csv(filepath, index=False)
    print(f"  ✅ Saved: {filepath}")

def save_report(lines, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  ✅ Report saved: {filepath}")

def validate_symbol(user_input):
    s = user_input.strip().upper()
    if s in valid_symbols:
        return s
    matches = get_close_matches(s, list(valid_symbols), n=3, cutoff=0.6)
    if not matches:
        print(f"  ❌ '{s}' not found. No close matches.")
        return None
    print(f"  ⚠️  '{s}' not found. Did you mean:")
    for i, m in enumerate(matches, 1):
        row = tech_df[tech_df['Symbol'] == m]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"    {i}. {m:<14} "
                  f"{str(r.get('Sector','?')):<28} "
                  f"Rs{r.get('Current Price',0):.2f}  "
                  f"MCap {mcap_str(r.get('Market Cap Cr',0))}")
        else:
            print(f"    {i}. {m}")
    print(f"    0. None / skip")
    choice = input("  Enter number: ").strip()
    if choice in [str(i) for i in range(1, len(matches)+1)]:
        return matches[int(choice)-1]
    return None

def get_priority_tier(row):
    ml_pred    = str(row.get('ML_Prediction', ''))
    best_setup = str(row.get('Best Setup', ''))
    fund_score = float(row.get('Fund Score', 0) or 0)
    ema_ok     = passes_price_filter(row)
    conf_ok    = float(row.get('ML_Confidence', 0) or 0) >= MIN_CONF

    if (ml_pred == 'Bullish Continual' and best_setup == 'Momentum'
            and ema_ok and conf_ok and is_sector_bullish(row)):
        return 1
    if (ml_pred == 'Bullish Continual' and best_setup == 'Momentum'
            and ema_ok and conf_ok):
        return 2
    if (ml_pred == 'Reversal' and best_setup == 'Reversal'
            and fund_score >= 65):
        return 3
    return None

def get_priority_tier_swing(row):
    ml_pred    = str(row.get('ML_Prediction', ''))
    best_setup = str(row.get('Best Setup', ''))
    fund_score = float(row.get('Fund Score', 0) or 0)
    ema_ok     = passes_price_filter(row)
    conf_ok    = float(row.get('ML_Confidence', 0) or 0) >= MIN_CONF

    if fund_score < MIN_FUND_SWING:    return None
    if not ema_ok:                     return None
    if not conf_ok:                    return None
    if ml_pred != 'Bullish Continual': return None
    if best_setup != 'Momentum':       return None
    if is_sector_bullish(row):         return 1
    return 2

def compute_priority_score(row):
    sec_rank   = float(row.get('Sector Score',   0) or 0)
    cap_rank   = float(row.get('Cap Score',      0) or 0)
    ml_conf    = float(row.get('ML_Confidence', 0) or 0)
    base       = sec_rank * 0.5 + cap_rank * 0.5
    if is_sector_strong_bull(row): bonus = 2.0
    elif is_sector_bullish(row):   bonus = 1.0
    else:                          bonus = 0.0
    conf_boost = min(ml_conf / 200, 0.5)
    return round(base + bonus + conf_boost, 3)

def input_stock(is_swing=False, is_model=False):
    sym_input = input("\n  Symbol (or 'done'): ").strip()
    if sym_input.lower() == 'done':
        return 'done'

    symbol = validate_symbol(sym_input)
    if symbol is None:
        print("  Skipping.")
        return None

    info = tech_df[tech_df['Symbol'] == symbol]
    if len(info) > 0:
        r = info.iloc[0]
        print(f"  ✅ {symbol} — {r.get('Sector','?')} | "
              f"Rs{r.get('Current Price',0):.2f} | "
              f"MCap {mcap_str(r.get('Market Cap Cr',0))} | "
              f"ML: {r.get('ML_Prediction','')} | "
              f"Setup: {r.get('Best Setup','')}")

    try:
        entry_price = float(input(f"  Entry price (Rs): ").strip())
    except:
        print("  Invalid price. Skipping.")
        return None

    try:
        quantity = int(input(f"  Quantity (shares): ").strip())
    except:
        print("  Invalid quantity. Skipping.")
        return None

    entry_date = input(
        f"  Entry date (YYYY-MM-DD) [Enter for today]: ").strip()
    if not entry_date:
        entry_date = today_str

    accum_mode  = 'No'
    cycle_start = None

    if not is_swing and not is_model:
        accum = input(
            f"  Accumulation mode? (y/n) "
            f"[slow avg down — exit suppressed]: ").strip().lower()
        accum_mode = 'Yes' if accum == 'y' else 'No'

    if is_swing:
        cycle_start = input(
            f"  Cycle start date (YYYY-MM-DD) "
            f"[Enter for today]: ").strip()
        if not cycle_start:
            cycle_start = today_str

    notes = input(f"  Notes (optional): ").strip()

    curr_sec, curr_cap = get_current_ranks(symbol)
    cap_cat = str(info.iloc[0].get('Cap Category', '')) \
              if len(info) > 0 else ''

    record = {
        'Symbol'              : symbol,
        'Entry_Price'         : entry_price,
        'Entry_Date'          : entry_date,
        'Quantity'            : quantity,
        'Cap_Category'        : cap_cat,
        'Sector_Rank_At_Entry': curr_sec or 0,
        'Cap_Rank_At_Entry'   : curr_cap or 0,
        'Sector_Rank_Change'  : 0.0,
        'Cap_Rank_Change'     : 0.0,
        'Notes'               : notes,
    }

    if not is_swing and not is_model:
        record['Accumulation_Mode'] = accum_mode
    if is_swing:
        record['Cycle_Start_Date'] = cycle_start

    invested = entry_price * quantity
    print(f"  ✅ Added {symbol} — {quantity} shares @ "
          f"Rs{entry_price:.2f} = Rs{invested:,.0f}  "
          f"SecRnk={curr_sec:.1f}  CapRnk={curr_cap:.1f}")
    return record

def manage_portfolio(filepath, cols, portfolio_type, is_swing=False):
    df = load_portfolio(filepath, cols)

    if len(df) == 0:
        print(f"\n  No {portfolio_type} portfolio found.")
        choice = input(f"  Create new portfolio? (y/n): ").strip().lower()
        if choice != 'y':
            return df
    else:
        df = update_rank_changes(df)
        print(f"\n  {portfolio_type} portfolio — {len(df)} stocks")
        print(f"\n  Current holdings:")
        for i, row in df.iterrows():
            sec_chg = float(row.get('Sector_Rank_Change', 0) or 0)
            cap_chg = float(row.get('Cap_Rank_Change',    0) or 0)
            accum   = ' [ACCUM]' \
                      if str(row.get('Accumulation_Mode','')) == 'Yes' \
                      else ''
            print(f"    {i+1:>2}. {str(row['Symbol']):<14} "
                  f"Rs{float(row['Entry_Price']):.2f} × "
                  f"{int(row['Quantity'])}  "
                  f"SecRnk:{float(row.get('Sector_Rank_At_Entry',0)):.1f}"
                  f"({sec_chg:+.1f})  "
                  f"CapRnk:{float(row.get('Cap_Rank_At_Entry',0)):.1f}"
                  f"({cap_chg:+.1f}){accum}")

        print(f"\n  Options:")
        print(f"  1. Add new stocks")
        print(f"  2. Remove stocks")
        if not is_swing:
            print(f"  3. Toggle accumulation mode")
        print(f"  4. No changes — proceed to review")
        sub = input("  Choice: ").strip()

        if sub == '2':
            to_remove = input(
                "  Row numbers to remove (comma separated): ").strip()
            try:
                indices = [int(x.strip())-1
                           for x in to_remove.split(',')]
                df      = df.drop(df.index[indices]).reset_index(drop=True)
                df      = update_rank_changes(df)
                save_portfolio(df, filepath)
            except:
                print("  Invalid input. No changes.")
            return df
        elif sub == '3' and not is_swing:
            sym_upd = input("  Symbol to toggle: ").strip().upper()
            mask    = df['Symbol'] == sym_upd
            if mask.any():
                curr = df.loc[mask, 'Accumulation_Mode'].values[0]
                new  = 'No' if curr == 'Yes' else 'Yes'
                df.loc[mask, 'Accumulation_Mode'] = new
                save_portfolio(df, filepath)
                print(f"  {sym_upd} Accumulation_Mode → {new}")
            else:
                print(f"  {sym_upd} not found.")
            return df
        elif sub == '4':
            return df

    # Add new stocks
    print(f"\n  Enter stocks to add. Type 'done' when finished.")
    new_records = []
    while True:
        record = input_stock(is_swing=is_swing)
        if record == 'done':
            break
        if record is not None:
            new_records.append(record)

    if new_records:
        new_df = pd.DataFrame(new_records)
        df     = pd.concat([df, new_df], ignore_index=True)
        df     = df.drop_duplicates(subset=['Symbol'], keep='last')
        df     = update_rank_changes(df)
        save_portfolio(df, filepath)

    return df

# ── RECOMMENDATION BUILDERS ────────────────────────────────────
def build_lt_recommendations():
    tech_df['Priority_Tier']  = tech_df.apply(
        lambda r: get_priority_tier(r), axis=1)
    tech_df['Priority_Score'] = tech_df.apply(
        compute_priority_score, axis=1)
    tech_df['Effective_Conf'] = tech_df.apply(
        lambda r: float(r.get('Bottom_Rev_Prob', 0) or 0)
                  if str(r.get('ML_Prediction','')) == 'Reversal'
                  else float(r.get('ML_Confidence', 0) or 0),
        axis=1)

    filter_label = {
        '1': 'Standard (Price > EMA50 > EMA200)',
        '2': f'Early Entry (within {EARLY_ENTRY_PCT}% of EMA50)',
        '3': f'Both — [HIGH] = >{EARLY_ENTRY_PCT}% above EMA50',
    }.get(price_filter_choice, 'Standard')

    lt_symbols = set()
    lt_lines   = []
    def p(line=''): lt_lines.append(str(line))

    p("=" * 112)
    p("  LONG TERM PORTFOLIO — STOCK RECOMMENDATIONS")
    p(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
      f"Universe: {len(tech_df)} stocks  |  "
      f"ML Confidence filter: >={MIN_CONF}%")
    p(f"  Price filter  : {filter_label}")
    p()
    p("  ★★★ Tier 1 : Sector bullish + ML Bull Cont + Tech Momentum "
      "+ Price > EMA50 > EMA200")
    p("  ★★☆ Tier 2 : ML Bull Cont + Tech Momentum + Price > EMA50 > EMA200 "
      "(sector not bullish)")
    p("  ★☆☆ Tier 3 : ML Reversal + Tech Reversal + Fund Score >= 65 "
      "(contrarian, max 2 per cap)")
    if price_filter_choice == '3':
        p(f"  [HIGH] = price already >{EARLY_ENTRY_PCT}% above EMA50 — "
          f"consider waiting for pullback")
    p("=" * 112)

    for cap in CAP_ORDER:
        cap_lbl = {'Large Cap':'L','Mini Large Cap':'ML',
                   'Mid Cap':'M','Small Cap':'S'}[cap]

        t1_cap = tech_df[
            (tech_df['Priority_Tier']  == 1) &
            (tech_df['Cap Category']   == cap) &
            (tech_df['Effective_Conf'] >= MIN_CONF)
        ].sort_values('Priority_Score', ascending=False)

        t2_cap = tech_df[
            (tech_df['Priority_Tier']  == 2) &
            (tech_df['Cap Category']   == cap) &
            (tech_df['Effective_Conf'] >= MIN_CONF)
        ].sort_values('Priority_Score', ascending=False)

        t3_cap = tech_df[
            (tech_df['Priority_Tier']  == 3) &
            (tech_df['Cap Category']   == cap) &
            (tech_df['Effective_Conf'] >= MIN_CONF)
        ].sort_values('Priority_Score', ascending=False).head(MAX_REVERSAL)

        bc_slots  = TOP_N - len(t3_cap)
        combined  = pd.concat([t1_cap, t2_cap]).head(bc_slots)
        all_picks = pd.concat([combined, t3_cap])

        if len(all_picks) == 0:
            p(f"\n  [{cap_lbl}] {cap}  —  No qualifying stocks")
            continue

        lt_symbols.update(all_picks['Symbol'].tolist())

        # Count early entry vs high in this cap
        early_count = sum(1 for _, r in all_picks.iterrows()
                         if is_early_entry(r))
        high_count  = len(all_picks) - early_count

        p(f"\n{'─'*112}")
        p(f"  [{cap_lbl}] {cap}  —  "
          f"{len(t1_cap)} Tier1  |  {len(t2_cap)} Tier2  |  "
          f"{len(t3_cap)} Tier3  "
          f"(showing {min(len(all_picks), TOP_N)}"
          + (f"  |  Early entry: {early_count}  [HIGH]: {high_count}"
             if price_filter_choice == '3' else '')
          + ")")
        p(f"{'─'*112}")
        p(f"  {'#':<3}  {'★':<3}  {'Symbol':<12} {'Flag':<6} "
          f"{'Sector':<14} {'Sec Trend':<12} "
          f"{'ML Prediction':<20} {'Conf':>5}  "
          f"{'Setup':<9} {'Tech':>4}  {'EMA50%':>6}  "
          f"{'SecRnk':>6}  {'CapRnk':>6}  "
          f"{'Price':>8}  {'MCap':>12}")
        p(f"  {'─'*3}  {'─'*3}  {'─'*12} {'─'*6} "
          f"{'─'*14} {'─'*12} "
          f"{'─'*20} {'─'*5}  "
          f"{'─'*9} {'─'*4}  {'─'*6}  "
          f"{'─'*6}  {'─'*6}  "
          f"{'─'*8}  {'─'*12}")

        for serial, (_, row) in enumerate(all_picks.iterrows(), 1):
            tier_n   = int(row.get('Priority_Tier', 0) or 0)
            stars    = {1:'★★★', 2:'★★☆', 3:'★☆☆'}.get(tier_n, '   ')
            sec_bull = '✅' if is_sector_bullish(row) else '  '
            conf     = float(row.get('Effective_Conf', 0) or 0)
            price    = float(row.get('Current Price',  0) or 0)
            mcap     = float(row.get('Market Cap Cr',  0) or 0)
            pct_ema  = pct_above_ema50(row)
            flag     = get_high_tag(row)

            p(f"  {serial:<3}  {stars}  {row['Symbol']:<12} {flag:<6} "
              f"{short_sector(row.get('Sector','?')):<14} "
              f"{sec_bull}{short_trend(row.get('Sector Trend','')):<11} "
              f"{str(row.get('ML_Prediction','')):<20} "
              f"{conf:>5.1f}  "
              f"{str(row.get('Best Setup','')):<9} "
              f"{float(row.get('Tech Score',0) or 0):>4.0f}  "
              f"{pct_ema:>+6.1f}%  "
              f"{float(row.get('Sector Score',0) or 0):>6.1f}  "
              f"{float(row.get('Cap Score',0) or 0):>6.1f}  "
              f"{price:>8.2f}  "
              f"{mcap_str(mcap):>12}")

    p(f"\n{'─'*112}")
    p(f"  ★★★ Sector+ML+Tech all bullish  |  "
      f"★★☆ ML+Tech bullish, sector not  |  ★☆☆ Reversal")
    p(f"  ✅ Sector uptrend  |  EMA50% = % price is above EMA50")
    if price_filter_choice == '3':
        p(f"  [HIGH] = >{EARLY_ENTRY_PCT}% above EMA50 — "
          f"already run up, consider waiting for pullback")
    p(f"{'─'*112}")

    return lt_lines, lt_symbols

def build_swing_recommendations(lt_symbols):
    tech_df['Swing_Priority_Tier']  = tech_df.apply(
        lambda r: get_priority_tier_swing(r), axis=1)
    tech_df['Swing_Priority_Score'] = tech_df.apply(
        compute_priority_score, axis=1)

    filter_label = {
        '1': 'Standard (Price > EMA50 > EMA200)',
        '2': f'Early Entry (within {EARLY_ENTRY_PCT}% of EMA50)',
        '3': f'Both — [HIGH] = >{EARLY_ENTRY_PCT}% above EMA50',
    }.get(price_filter_choice, 'Standard')

    swing_lines = []
    def ps(line=''): swing_lines.append(str(line))

    ps("=" * 110)
    ps("  SWING TRADING PORTFOLIO — STOCK RECOMMENDATIONS")
    ps(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
       f"Universe: {len(tech_df)} stocks  |  "
       f"ML Confidence filter: >={MIN_CONF}%")
    ps(f"  Price filter  : {filter_label}")
    ps()
    ps("  ★★★ Tier 1 : Sector bullish + ML Bull Cont + Tech Momentum "
       "+ Price > EMA50 > EMA200")
    ps("  ★★☆ Tier 2 : ML Bull Cont + Tech Momentum + Price > EMA50 > EMA200 "
       "(sector not bullish)")
    ps(f"  Bull Cont only | No reversals | Fund Score >= {MIN_FUND_SWING} "
       f"| 30-day cycle")
    ps(f"  [LT] = also in Long Term reco")
    if price_filter_choice == '3':
        ps(f"  [HIGH] = price already >{EARLY_ENTRY_PCT}% above EMA50")
    ps("=" * 110)

    for cap in CAP_ORDER:
        cap_lbl = {'Large Cap':'L','Mini Large Cap':'ML',
                   'Mid Cap':'M','Small Cap':'S'}[cap]

        t1_cap = tech_df[
            (tech_df['Swing_Priority_Tier'] == 1) &
            (tech_df['Cap Category']        == cap)
        ].sort_values('Swing_Priority_Score', ascending=False)

        t2_cap = tech_df[
            (tech_df['Swing_Priority_Tier'] == 2) &
            (tech_df['Cap Category']        == cap)
        ].sort_values('Swing_Priority_Score', ascending=False)

        all_picks = pd.concat([t1_cap, t2_cap]).head(SWING_TOP_N)

        if len(all_picks) == 0:
            ps(f"\n  [{cap_lbl}] {cap}  —  No qualifying stocks")
            continue

        lt_overlap  = sum(1 for s in all_picks['Symbol']
                          if s in lt_symbols)
        early_count = sum(1 for _, r in all_picks.iterrows()
                          if is_early_entry(r))
        high_count  = len(all_picks) - early_count

        ps(f"\n{'─'*110}")
        ps(f"  [{cap_lbl}] {cap}  —  "
           f"{len(t1_cap)} Tier1  |  {len(t2_cap)} Tier2  "
           f"(showing {min(len(all_picks), SWING_TOP_N)})  "
           f"[LT overlap: {lt_overlap}]"
           + (f"  |  Early entry: {early_count}  [HIGH]: {high_count}"
              if price_filter_choice == '3' else ''))
        ps(f"{'─'*110}")
        ps(f"  {'#':<3}  {'★':<3}  {'Symbol':<12} {'LT':<4} {'Flag':<6} "
           f"{'Sector':<14} {'Sec Trend':<12} "
           f"{'ML':<10} {'Conf':>5}  "
           f"{'Tech':>4}  {'EMA50%':>6}  "
           f"{'SecRnk':>6}  {'CapRnk':>6}  "
           f"{'Price':>8}  {'MCap':>12}  {'25d%':>5}  {'45d%':>5}")
        ps(f"  {'─'*3}  {'─'*3}  {'─'*12} {'─'*4} {'─'*6} "
           f"{'─'*14} {'─'*12} "
           f"{'─'*10} {'─'*5}  "
           f"{'─'*4}  {'─'*6}  "
           f"{'─'*6}  {'─'*6}  "
           f"{'─'*8}  {'─'*12}  {'─'*5}  {'─'*5}")

        for serial, (_, row) in enumerate(all_picks.iterrows(), 1):
            tier_n   = int(row.get('Swing_Priority_Tier', 0) or 0)
            stars    = {1:'★★★', 2:'★★☆'}.get(tier_n, '   ')
            sec_bull = '✅' if is_sector_bullish(row) else '  '
            conf     = float(row.get('ML_Confidence', 0) or 0)
            price    = float(row.get('Current Price', 0) or 0)
            mcap     = float(row.get('Market Cap Cr', 0) or 0)
            f25      = float(row.get('Forecast_25d_Pct', 0) or 0)
            f45      = float(row.get('Forecast_45d_Pct', 0) or 0)
            lt_tag   = '[LT]' if row['Symbol'] in lt_symbols else '    '
            flag     = get_high_tag(row)
            pct_ema  = pct_above_ema50(row)

            ps(f"  {serial:<3}  {stars}  {row['Symbol']:<12} "
               f"{lt_tag:<4} {flag:<6} "
               f"{short_sector(row.get('Sector','?')):<14} "
               f"{sec_bull}{short_trend(row.get('Sector Trend','')):<11} "
               f"{short_ml(row.get('ML_Prediction','')):<10} "
               f"{conf:>5.1f}  "
               f"{float(row.get('Tech Score',0) or 0):>4.0f}  "
               f"{pct_ema:>+6.1f}%  "
               f"{float(row.get('Sector Score',0) or 0):>6.1f}  "
               f"{float(row.get('Cap Score',0) or 0):>6.1f}  "
               f"{price:>8.2f}  "
               f"{mcap_str(mcap):>12}  "
               f"{f25:>+5.1f}  "
               f"{f45:>+5.1f}")

    ps(f"\n{'─'*110}")
    ps(f"  ★★★ Sector+ML+Tech all bullish  |  "
       f"★★☆ ML+Tech bullish, sector not  |  ✅ Sector uptrend")
    ps(f"  [LT] = also in LT reco  |  "
       f"EMA50% = % price above EMA50  |  No reversals in swing")
    if price_filter_choice == '3':
        ps(f"  [HIGH] = >{EARLY_ENTRY_PCT}% above EMA50 — "
           f"already run up, consider waiting for pullback")
    ps(f"{'─'*110}")

    return swing_lines

# ── LONG TERM REVIEW ───────────────────────────────────────────
def run_lt_review(lt_df):
    if len(lt_df) == 0:
        return []

    lines = []
    def p(line=''): lines.append(str(line))

    p("=" * 100)
    p("  LONG TERM PORTFOLIO — HOLDINGS REVIEW")
    p(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    p(f"  Holdings  : {len(lt_df)} stocks")
    p()
    p("  Exit : Both ranks dropped significantly OR ML bearish + forecast negative")
    p("  Add  : Ranks improved + ML strong + Tech strong + Sector bullish")
    p("  [A]  : Accumulation mode — price exits suppressed")
    p("=" * 100)

    total_invested = 0
    total_curr_val = 0

    for _, holding in lt_df.iterrows():
        symbol     = str(holding['Symbol'])
        entry_px   = float(holding.get('Entry_Price',  0) or 0)
        qty        = int(holding.get('Quantity',        0) or 0)
        entry_date = str(holding.get('Entry_Date',     '') or '')
        entry_sec  = float(holding.get('Sector_Rank_At_Entry', 0) or 0)
        entry_cap  = float(holding.get('Cap_Rank_At_Entry',    0) or 0)
        sec_chg    = float(holding.get('Sector_Rank_Change',   0) or 0)
        cap_chg    = float(holding.get('Cap_Rank_Change',      0) or 0)
        accum_mode = str(holding.get('Accumulation_Mode', 'No') or 'No')
        notes      = str(holding.get('Notes', '') or '')
        invested   = entry_px * qty
        total_invested += invested

        info = tech_df[tech_df['Symbol'] == symbol]
        if len(info) == 0:
            p(f"\n  {'─'*96}")
            p(f"  {symbol}  ⚠️  Not found in current universe")
            p(f"  Entry   : Rs{entry_px:.2f} × {qty} = Rs{invested:,.0f}")
            p(f"  ► ACTION: 🔴 REVIEW MANUALLY")
            continue

        r          = info.iloc[0]
        curr_px    = float(r.get('Current Price',    0) or 0)
        ml_pred    = str(r.get('ML_Prediction',      ''))
        ml_conf    = float(r.get('ML_Confidence',    0) or 0)
        best_setup = str(r.get('Best Setup',         ''))
        tech_score = float(r.get('Tech Score',       0) or 0)
        tier       = str(r.get('Tier',               ''))
        rsi        = float(r.get('RSI',              0) or 0)
        adx        = float(r.get('ADX',              0) or 0)
        macd       = float(r.get('MACD Hist',        0) or 0)
        curr_sec   = float(r.get('Sector Score',     0) or 0)
        curr_cap   = float(r.get('Cap Score',        0) or 0)
        sec_trend  = str(r.get('Sector Trend',       ''))
        forecast25 = float(r.get('Forecast_25d_Pct', 0) or 0)
        forecast45 = float(r.get('Forecast_45d_Pct', 0) or 0)
        bull_prob  = float(r.get('Bullish_Cont_Prob',0) or 0)
        top_prob   = float(r.get('Top_Rev_Prob',     0) or 0)
        ema200     = float(r.get('EMA200',           0) or 0)
        pct_ema    = pct_above_ema50(r)

        curr_val   = curr_px * qty
        total_curr_val += curr_val
        pl_abs     = curr_val - invested
        pl_pct     = (pl_abs / invested * 100) if invested > 0 else 0

        try:
            held_days = (datetime.now() -
                         datetime.strptime(entry_date, '%Y-%m-%d')).days
        except:
            held_days = 0

        exit_reasons = []
        add_reasons  = []
        warn_reasons = []

        if sec_chg <= -1.5 and cap_chg <= -1.5:
            exit_reasons.append(
                f"Both ranks dropped — "
                f"SecRnk:{entry_sec:.1f}→{curr_sec:.1f}({sec_chg:+.1f})  "
                f"CapRnk:{entry_cap:.1f}→{curr_cap:.1f}({cap_chg:+.1f})")
        if ml_pred in ('Bearish Continual', 'Bearish') and ml_conf >= 50:
            exit_reasons.append(
                f"ML bearish: {ml_pred} ({ml_conf:.1f}%)")
        if forecast25 < -3 and forecast45 < -3:
            exit_reasons.append(
                f"Forecast negative: 25d={forecast25:+.1f}%  "
                f"45d={forecast45:+.1f}%")
        if top_prob >= 65 and ml_conf < 40:
            warn_reasons.append(
                f"Top reversal risk: {top_prob:.1f}% — watch closely")
        if curr_px < ema200 and accum_mode != 'Yes':
            warn_reasons.append(
                f"Price below EMA200 (Rs{ema200:.2f}) — trend weakening")
        if pct_ema > EARLY_ENTRY_PCT:
            warn_reasons.append(
                f"Price {pct_ema:+.1f}% above EMA50 — "
                f"already run up, watch for pullback")
        if sec_chg >= 1.0 and cap_chg >= 1.0:
            add_reasons.append(
                f"Both ranks improved ({sec_chg:+.1f} / {cap_chg:+.1f})")
        if ml_pred == 'Bullish Continual' and ml_conf >= 65:
            add_reasons.append(
                f"Strong ML: {ml_pred} ({ml_conf:.1f}%)")
        if tech_score >= 70 and best_setup == 'Momentum':
            add_reasons.append(
                f"Strong tech: {best_setup} score={tech_score:.0f}")
        if is_sector_bullish(r) and best_setup == 'Momentum':
            add_reasons.append(f"Sector + stock both bullish ✅")

        if accum_mode == 'Yes':
            exit_reasons = [e for e in exit_reasons
                            if 'Forecast' not in e]
            warn_reasons.append(
                f"Accumulation mode ON — price dips expected")

        if exit_reasons:            action = "🔴 EXIT"
        elif len(add_reasons) >= 2: action = "🟢 ADD"
        elif add_reasons:           action = "🟡 HOLD  (mild add signal)"
        else:                       action = "🟡 HOLD"

        accum_tag = " [A]" if accum_mode == 'Yes' else ""
        p(f"\n  {'─'*96}")
        p(f"  {symbol}{accum_tag}  |  "
          f"{short_sector(r.get('Sector','?'))}  |  "
          f"{r.get('Cap Category','?')}  |  "
          f"MCap {mcap_str(r.get('Market Cap Cr',0))}  |  "
          f"Held: {held_days}d  |  "
          f"EMA50: {pct_ema:+.1f}%"
          + (' [HIGH]' if pct_ema > EARLY_ENTRY_PCT else ''))
        p(f"  Entry   : Rs{entry_px:.2f} × {qty} on {entry_date} "
          f"= Rs{invested:,.0f}")
        p(f"  Current : Rs{curr_px:.2f}  |  "
          f"Value: Rs{curr_val:,.0f}  |  "
          f"P&L: Rs{pl_abs:+,.0f} ({pl_pct:+.1f}%)")
        p(f"  Ranks   : SecRnk {entry_sec:.1f}→{curr_sec:.1f} "
          f"({sec_chg:+.1f})  |  "
          f"CapRnk {entry_cap:.1f}→{curr_cap:.1f} ({cap_chg:+.1f})")
        p(f"  ML      : {ml_pred}  Conf:{ml_conf:.1f}%  "
          f"BullProb:{bull_prob:.1f}%  TopRevRisk:{top_prob:.1f}%")
        p(f"  Forecast: 25d={forecast25:+.1f}%  45d={forecast45:+.1f}%")
        p(f"  Tech    : Score={tech_score:.0f}  Setup={best_setup}  "
          f"Tier={tier_abbr(tier)}  "
          f"RSI={rsi:.0f}  ADX={adx:.0f}  MACD={macd:+.2f}")
        p(f"  Sector  : {short_trend(sec_trend)}")
        if notes:
            p(f"  Notes   : {notes}")
        p(f"\n  ► ACTION : {action}")
        for r_ in exit_reasons:
            p(f"    ✗ Exit : {r_}")
        for r_ in add_reasons:
            p(f"    ✓ Add  : {r_}")
        for r_ in warn_reasons:
            p(f"    ⚠ Warn : {r_}")

    portfolio_pl     = total_curr_val - total_invested
    portfolio_pl_pct = (portfolio_pl / total_invested * 100) \
                        if total_invested > 0 else 0
    p(f"\n{'─'*100}")
    p(f"  PORTFOLIO SUMMARY")
    p(f"  Total Invested : Rs{total_invested:,.0f}")
    p(f"  Current Value  : Rs{total_curr_val:,.0f}")
    p(f"  Total P&L      : Rs{portfolio_pl:+,.0f} ({portfolio_pl_pct:+.1f}%)")
    p(f"{'─'*100}")
    p(f"  [A] Accumulation mode — price dips tolerated")
    p(f"  [HIGH] = price >{EARLY_ENTRY_PCT}% above EMA50 — watch for pullback")
    p(f"  Exit based on rank deterioration + ML signal, not price alone")
    p(f"{'─'*100}")
    return lines

# ── SWING REVIEW ───────────────────────────────────────────────
def compute_swing_allocation(swing_df):
    weights = []
    symbols = []
    for _, row in swing_df.iterrows():
        sym  = str(row['Symbol'])
        info = tech_df[tech_df['Symbol'] == sym]
        w    = (float(info.iloc[0].get('Sector Score', 5) or 5) * 0.5 +
                float(info.iloc[0].get('Cap Score',    5) or 5) * 0.5) \
               if len(info) > 0 else 5.0
        weights.append(w)
        symbols.append(sym)
    total_w = sum(weights) if sum(weights) > 0 else 1
    allocs  = {s: round(SWING_CAPITAL * w / total_w)
               for s, w in zip(symbols, weights)}
    alloc_p = {s: round(w / total_w * 100, 1)
               for s, w in zip(symbols, weights)}
    return allocs, alloc_p

def run_swing_review(swing_df):
    if len(swing_df) == 0:
        return []

    lines = []
    def p(line=''): lines.append(str(line))

    p("=" * 100)
    p("  SWING TRADING PORTFOLIO — HOLDINGS REVIEW")
    p(f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    p(f"  Holdings   : {len(swing_df)} stocks  |  "
      f"Capital: Rs{SWING_CAPITAL:,.0f}  |  "
      f"Drawdown limit: {DRAWDOWN_LIMIT*100:.0f}%  |  "
      f"Cycle: {MAX_HOLD_DAYS} days")
    p()
    p("  Exit  : ML changed OR forecast negative OR EMA broken OR drawdown hit")
    p("  Carry : 30d done + ML conf >= 65% + forecast positive")
    p("=" * 100)

    total_invested = 0
    total_curr_val = 0
    stock_data     = []

    for _, holding in swing_df.iterrows():
        symbol   = str(holding['Symbol'])
        entry_px = float(holding.get('Entry_Price', 0) or 0)
        qty      = int(holding.get('Quantity',       0) or 0)
        invested = entry_px * qty
        total_invested += invested
        info     = tech_df[tech_df['Symbol'] == symbol]
        curr_px  = float(info.iloc[0].get('Current Price', 0) or 0) \
                   if len(info) > 0 else entry_px
        ml_conf  = float(info.iloc[0].get('ML_Confidence', 0) or 0) \
                   if len(info) > 0 else 0.0
        f25      = float(info.iloc[0].get('Forecast_25d_Pct', 0) or 0) \
                   if len(info) > 0 else 0.0
        curr_val = curr_px * qty
        total_curr_val += curr_val
        pl_pct   = ((curr_val - invested) / invested * 100) \
                    if invested > 0 else 0
        stock_data.append({
            'symbol'  : symbol,
            'invested': invested,
            'curr_val': curr_val,
            'pl_pct'  : pl_pct,
            'ml_conf' : ml_conf,
            'f25'     : f25,
            'holding' : holding,
        })

    portfolio_pl_pct = ((total_curr_val - total_invested) /
                         total_invested * 100) \
                        if total_invested > 0 else 0
    drawdown_hit     = portfolio_pl_pct <= -(DRAWDOWN_LIMIT * 100)
    stock_data_sorted= sorted(stock_data, key=lambda x: x['ml_conf'])

    p(f"\n{'─'*100}")
    p(f"  SECTION 1 — PORTFOLIO P&L & DRAWDOWN")
    p(f"{'─'*100}")
    p(f"  Total Invested : Rs{total_invested:,.0f}")
    p(f"  Current Value  : Rs{total_curr_val:,.0f}")
    p(f"  Portfolio P&L  : "
      f"Rs{total_curr_val-total_invested:+,.0f} "
      f"({portfolio_pl_pct:+.1f}%)")
    if drawdown_hit:
        p(f"\n  ⚠️  DRAWDOWN ALERT — Portfolio down "
          f"{abs(portfolio_pl_pct):.1f}%")
        p(f"     Exit weakest: {stock_data_sorted[0]['symbol']} "
          f"(ML conf: {stock_data_sorted[0]['ml_conf']:.1f}%)")
    else:
        room = DRAWDOWN_LIMIT*100 - abs(min(portfolio_pl_pct, 0))
        p(f"  Status : ✅ Within limit  (room: {room:.1f}%)")

    allocs, alloc_p = compute_swing_allocation(swing_df)
    p(f"\n{'─'*100}")
    p(f"  SECTION 2 — ALLOCATION GUIDE  |  Capital: Rs{SWING_CAPITAL:,.0f}")
    p(f"{'─'*100}")
    p(f"  {'Symbol':<14} {'Cap':<16} {'SecRnk':>6}  {'CapRnk':>6}  "
      f"{'Alloc%':>6}  {'Rs Amount':>10}  {'Invested':>10}  {'P&L%':>6}")
    p(f"  {'─'*14} {'─'*16} {'─'*6}  {'─'*6}  "
      f"{'─'*6}  {'─'*10}  {'─'*10}  {'─'*6}")
    for sd in stock_data:
        sym  = sd['symbol']
        info = tech_df[tech_df['Symbol'] == sym]
        sec_r= float(info.iloc[0].get('Sector Score', 0) or 0) \
               if len(info) > 0 else 0
        cap_r= float(info.iloc[0].get('Cap Score',    0) or 0) \
               if len(info) > 0 else 0
        cap_c= str(info.iloc[0].get('Cap Category',   '') or '') \
               if len(info) > 0 else ''
        p(f"  {sym:<14} {cap_c:<16} {sec_r:>6.1f}  {cap_r:>6.1f}  "
          f"{alloc_p.get(sym,0):>6.1f}%  "
          f"Rs{allocs.get(sym,0):>8,.0f}  "
          f"Rs{sd['invested']:>8,.0f}  "
          f"{sd['pl_pct']:>+6.1f}%")

    p(f"\n{'─'*100}")
    p(f"  SECTION 3 — STOCK-BY-STOCK REVIEW")
    p(f"  (Weakest ML confidence first — for drawdown exit decisions)")
    p(f"{'─'*100}")

    weakest_flagged = False
    for sd in stock_data_sorted:
        symbol     = sd['symbol']
        holding    = sd['holding']
        entry_px   = float(holding.get('Entry_Price',  0) or 0)
        qty        = int(holding.get('Quantity',        0) or 0)
        entry_date = str(holding.get('Entry_Date',     '') or '')
        entry_sec  = float(holding.get('Sector_Rank_At_Entry', 0) or 0)
        entry_cap  = float(holding.get('Cap_Rank_At_Entry',    0) or 0)
        sec_chg    = float(holding.get('Sector_Rank_Change',   0) or 0)
        cap_chg    = float(holding.get('Cap_Rank_Change',      0) or 0)
        cycle_start= str(holding.get('Cycle_Start_Date',
                                      entry_date) or entry_date)
        notes      = str(holding.get('Notes', '') or '')
        invested   = sd['invested']
        curr_val   = sd['curr_val']
        pl_pct     = sd['pl_pct']

        try:
            cycle_days = (datetime.now() -
                          datetime.strptime(cycle_start,
                                             '%Y-%m-%d')).days
        except:
            cycle_days = 0

        info = tech_df[tech_df['Symbol'] == symbol]
        if len(info) == 0:
            p(f"\n  {'─'*96}")
            p(f"  {symbol}  ⚠️  Not found in universe")
            p(f"  ► ACTION : 🔴 EXIT")
            continue

        r          = info.iloc[0]
        curr_px    = float(r.get('Current Price',    0) or 0)
        ml_pred    = str(r.get('ML_Prediction',      ''))
        ml_conf    = float(r.get('ML_Confidence',    0) or 0)
        best_setup = str(r.get('Best Setup',         ''))
        tech_score = float(r.get('Tech Score',       0) or 0)
        tier       = str(r.get('Tier',               ''))
        rsi        = float(r.get('RSI',              0) or 0)
        adx        = float(r.get('ADX',              0) or 0)
        macd       = float(r.get('MACD Hist',        0) or 0)
        curr_sec   = float(r.get('Sector Score',     0) or 0)
        curr_cap   = float(r.get('Cap Score',        0) or 0)
        sec_trend  = str(r.get('Sector Trend',       ''))
        forecast25 = float(r.get('Forecast_25d_Pct', 0) or 0)
        forecast45 = float(r.get('Forecast_45d_Pct', 0) or 0)
        bull_prob  = float(r.get('Bullish_Cont_Prob',0) or 0)
        top_prob   = float(r.get('Top_Rev_Prob',     0) or 0)
        ema_ok     = passes_ema_filter(r)
        pct_ema    = pct_above_ema50(r)

        exit_reasons  = []
        add_reasons   = []
        warn_reasons  = []
        carry_forward = False

        if ml_pred != 'Bullish Continual':
            exit_reasons.append(
                f"ML changed: {ml_pred} — no longer Bull Cont")
        if forecast25 < 0 and forecast45 < 0:
            exit_reasons.append(
                f"Forecast negative: 25d={forecast25:+.1f}%  "
                f"45d={forecast45:+.1f}%")
        if not ema_ok and ml_pred == 'Bullish Continual':
            exit_reasons.append(
                f"EMA structure broken — Price no longer > EMA50 > EMA200")
        if drawdown_hit and not weakest_flagged:
            exit_reasons.append(
                f"Portfolio drawdown {abs(portfolio_pl_pct):.1f}% — "
                f"exit weakest stock")
            weakest_flagged = True
        if cycle_days >= MAX_HOLD_DAYS:
            if (ml_conf >= MIN_CARRY_CONF and forecast25 >= 0
                    and ml_pred == 'Bullish Continual' and ema_ok):
                carry_forward = True
            else:
                exit_reasons.append(
                    f"30-day cycle complete ({cycle_days}d) — "
                    f"setup not strong enough to carry forward")
        if sec_chg <= -1.0 or cap_chg <= -1.0:
            warn_reasons.append(
                f"Rank declining: "
                f"SecRnk({sec_chg:+.1f}) CapRnk({cap_chg:+.1f})")
        if top_prob >= 65:
            warn_reasons.append(
                f"Top reversal risk: {top_prob:.1f}% — tighten stop")
        if pct_ema > EARLY_ENTRY_PCT:
            warn_reasons.append(
                f"Price {pct_ema:+.1f}% above EMA50 — "
                f"already run up, watch for pullback")
        if (ml_conf >= 75 and forecast25 > 3
                and ml_pred == 'Bullish Continual'):
            add_reasons.append(
                f"Strong ML: conf={ml_conf:.1f}%  "
                f"forecast={forecast25:+.1f}%")
        if is_sector_bullish(r) and best_setup == 'Momentum':
            add_reasons.append(f"Sector + stock both bullish ✅")
        if tech_score >= 75 and best_setup == 'Momentum':
            add_reasons.append(f"Strong tech: score={tech_score:.0f}")
        if sec_chg >= 1.0 and cap_chg >= 1.0:
            add_reasons.append(
                f"Both ranks improved ({sec_chg:+.1f} / {cap_chg:+.1f})")

        if carry_forward:           action = "🔵 CARRY FORWARD"
        elif exit_reasons:          action = "🔴 EXIT"
        elif len(add_reasons) >= 2: action = "🟢 ADD"
        elif add_reasons:           action = "🟡 HOLD  (mild add)"
        else:                       action = "🟡 HOLD"

        p(f"\n  {'─'*96}")
        p(f"  {symbol}  |  "
          f"{short_sector(r.get('Sector','?'))}  |  "
          f"{r.get('Cap Category','?')}  |  "
          f"MCap {mcap_str(r.get('Market Cap Cr',0))}  |  "
          f"Cycle day: {cycle_days}/{MAX_HOLD_DAYS}  |  "
          f"EMA50: {pct_ema:+.1f}%"
          + (' [HIGH]' if pct_ema > EARLY_ENTRY_PCT else ''))
        p(f"  Entry   : Rs{entry_px:.2f} × {qty} on {entry_date} "
          f"= Rs{invested:,.0f}")
        p(f"  Current : Rs{curr_px:.2f}  |  "
          f"Value: Rs{curr_val:,.0f}  |  "
          f"P&L: Rs{curr_val-invested:+,.0f} ({pl_pct:+.1f}%)")
        p(f"  Ranks   : SecRnk {entry_sec:.1f}→{curr_sec:.1f} "
          f"({sec_chg:+.1f})  |  "
          f"CapRnk {entry_cap:.1f}→{curr_cap:.1f} ({cap_chg:+.1f})")
        p(f"  ML      : {ml_pred}  Conf:{ml_conf:.1f}%  "
          f"BullProb:{bull_prob:.1f}%  TopRevRisk:{top_prob:.1f}%")
        p(f"  Forecast: 25d={forecast25:+.1f}%  45d={forecast45:+.1f}%")
        p(f"  Tech    : Score={tech_score:.0f}  Setup={best_setup}  "
          f"Tier={tier_abbr(tier)}  "
          f"RSI={rsi:.0f}  ADX={adx:.0f}  MACD={macd:+.2f}")
        p(f"  Sector  : {short_trend(sec_trend)}  |  "
          f"EMA: {'✅ OK' if ema_ok else '❌ Broken'}")
        if notes:
            p(f"  Notes   : {notes}")
        p(f"\n  ► ACTION : {action}")
        if carry_forward:
            p(f"    🔵 Carry forward — conf={ml_conf:.1f}%  "
              f"forecast={forecast25:+.1f}%")
        elif exit_reasons:
            for r_ in exit_reasons:
                p(f"    ✗ Exit : {r_}")
        if add_reasons and not carry_forward:
            for r_ in add_reasons:
                p(f"    ✓ Add  : {r_}")
        for r_ in warn_reasons:
            p(f"    ⚠ Warn : {r_}")

    p(f"\n{'─'*100}")
    p(f"  PORTFOLIO SUMMARY")
    p(f"  Total Invested : Rs{total_invested:,.0f}")
    p(f"  Current Value  : Rs{total_curr_val:,.0f}")
    p(f"  Total P&L      : "
      f"Rs{total_curr_val-total_invested:+,.0f} "
      f"({portfolio_pl_pct:+.1f}%)")
    p(f"  Drawdown       : "
      f"{'⚠️  ALERT' if drawdown_hit else '✅ Within limit'}")
    p(f"{'─'*100}")
    p(f"  🔵 CARRY FORWARD  🔴 EXIT  🟢 ADD  🟡 HOLD")
    p(f"  [HIGH] = >{EARLY_ENTRY_PCT}% above EMA50 — watch for pullback")
    p(f"{'─'*100}")
    return lines

# ── MODEL PORTFOLIO ────────────────────────────────────────────
def create_model_portfolio():
    print("\n  CREATE / UPDATE MODEL PORTFOLIO")
    print("  " + "─" * 40)

    existing = [f.replace('.csv','')
                for f in os.listdir(MODEL_PORT_DIR)
                if f.endswith('.csv')]
    if existing:
        print(f"\n  Existing model portfolios:")
        for i, name in enumerate(existing, 1):
            print(f"    {i}. {name}")

    port_name = input(
        "\n  Portfolio name (e.g. 'metals_focus'): ").strip()
    if not port_name:
        print("  Invalid name. Exiting.")
        return

    port_name = port_name.replace(' ', '_').lower()
    filepath  = os.path.join(MODEL_PORT_DIR, f"{port_name}.csv")
    df        = load_portfolio(filepath, MODEL_COLS)

    if len(df) > 0:
        df = update_rank_changes(df)
        print(f"\n  '{port_name}' exists — {len(df)} stocks")
        for i, row in df.iterrows():
            sec_chg = float(row.get('Sector_Rank_Change', 0) or 0)
            cap_chg = float(row.get('Cap_Rank_Change',    0) or 0)
            print(f"    {i+1:>2}. {str(row['Symbol']):<14} "
                  f"Rs{float(row['Entry_Price']):.2f} × "
                  f"{int(row['Quantity'])}  "
                  f"SecRnk({sec_chg:+.1f})  "
                  f"CapRnk({cap_chg:+.1f})")
        print(f"\n  Options:")
        print(f"  1. Add stocks  2. Remove stocks  3. Cancel")
        sub = input("  Choice: ").strip()
        if sub == '2':
            to_remove = input(
                "  Row numbers to remove (comma separated): ").strip()
            try:
                indices = [int(x.strip())-1
                           for x in to_remove.split(',')]
                df      = df.drop(df.index[indices]).reset_index(drop=True)
                df      = update_rank_changes(df)
                save_portfolio(df, filepath)
            except:
                print("  Invalid. No changes.")
            return
        elif sub == '3':
            return

    print(f"\n  Enter stocks for '{port_name}'. Type 'done' when finished.")
    new_records = []
    while True:
        record = input_stock(is_swing=False, is_model=True)
        if record == 'done':
            break
        if record is not None:
            new_records.append(record)

    if new_records:
        new_df = pd.DataFrame(new_records)
        df     = pd.concat([df, new_df], ignore_index=True)
        df     = df.drop_duplicates(subset=['Symbol'], keep='last')
        df     = update_rank_changes(df)
        save_portfolio(df, filepath)
        print(f"\n  ✅ Model portfolio '{port_name}' — {len(df)} stocks")

def review_model_portfolios():
    """Review model portfolios — with selector."""
    files = [f for f in os.listdir(MODEL_PORT_DIR) if f.endswith('.csv')]
    if not files:
        print("\n  No model portfolios found.")
        print(f"  Use Option 4 to create one.")
        return []

    # Portfolio selector
    names = [f.replace('.csv','') for f in sorted(files)]
    print(f"\n  Existing portfolios:")
    for i, name in enumerate(names, 1):
        # Quick P&L preview
        fp  = os.path.join(MODEL_PORT_DIR, f"{name}.csv")
        df_ = load_portfolio(fp, MODEL_COLS)
        print(f"    {i}. {name}  ({len(df_)} stocks)")
    print(f"    {len(names)+1}. All portfolios")
    print()

    sel = input(f"  Enter choice (1-{len(names)+1}): ").strip()
    try:
        sel_idx = int(sel)
    except:
        sel_idx = len(names) + 1  # default to all

    if sel_idx == len(names) + 1:
        selected_files = sorted(files)
    elif 1 <= sel_idx <= len(names):
        selected_files = [f"{names[sel_idx-1]}.csv"]
    else:
        selected_files = sorted(files)

    lines = []
    def p(line=''): lines.append(str(line))

    p("=" * 90)
    p("  MODEL PORTFOLIO REVIEW — P&L COMPARISON")
    p(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    p(f"  Reviewing : {len(selected_files)} portfolio(s)")
    p("=" * 90)

    summary = []

    for fname in selected_files:
        port_name = fname.replace('.csv', '')
        filepath  = os.path.join(MODEL_PORT_DIR, fname)
        df        = load_portfolio(filepath, MODEL_COLS)
        if len(df) == 0:
            continue
        df = update_rank_changes(df)

        total_invested = 0
        total_curr_val = 0
        stock_lines    = []

        for _, row in df.iterrows():
            symbol   = str(row['Symbol'])
            entry_px = float(row.get('Entry_Price', 0) or 0)
            qty      = int(row.get('Quantity',       0) or 0)
            invested = entry_px * qty
            total_invested += invested
            info    = tech_df[tech_df['Symbol'] == symbol]
            curr_px = float(info.iloc[0].get('Current Price', 0) or 0) \
                      if len(info) > 0 else entry_px
            curr_val= curr_px * qty
            total_curr_val += curr_val
            pl_pct  = ((curr_val - invested) / invested * 100) \
                       if invested > 0 else 0
            sec_chg = float(row.get('Sector_Rank_Change', 0) or 0)
            cap_chg = float(row.get('Cap_Rank_Change',    0) or 0)
            ml_pred = str(info.iloc[0].get('ML_Prediction','')) \
                      if len(info) > 0 else '—'
            ml_conf = float(info.iloc[0].get('ML_Confidence', 0) or 0) \
                      if len(info) > 0 else 0
            stock_lines.append(
                f"    {symbol:<14} "
                f"Rs{entry_px:.2f}→Rs{curr_px:.2f}  "
                f"P&L:{pl_pct:+.1f}%  "
                f"SecRnk({sec_chg:+.1f}) CapRnk({cap_chg:+.1f})  "
                f"ML:{short_ml(ml_pred)} {ml_conf:.0f}%")

        port_pl     = total_curr_val - total_invested
        port_pl_pct = (port_pl / total_invested * 100) \
                       if total_invested > 0 else 0
        summary.append({
            'name'   : port_name,
            'invested': total_invested,
            'curr_val': total_curr_val,
            'pl_abs' : port_pl,
            'pl_pct' : port_pl_pct,
            'stocks' : len(df),
        })

        p(f"\n{'─'*90}")
        p(f"  {port_name.upper().replace('_',' ')}")
        p(f"  Stocks: {len(df)}  |  "
          f"Invested: Rs{total_invested:,.0f}  |  "
          f"Current: Rs{total_curr_val:,.0f}  |  "
          f"P&L: Rs{port_pl:+,.0f} ({port_pl_pct:+.1f}%)")
        p(f"{'─'*90}")
        for sl in stock_lines:
            p(sl)

    if len(summary) > 1:
        summary_sorted = sorted(
            summary, key=lambda x: x['pl_pct'], reverse=True)
        p(f"\n{'═'*90}")
        p(f"  PORTFOLIO RANKING (by P&L %)")
        p(f"{'═'*90}")
        p(f"  {'Rank':<5} {'Portfolio':<30} {'Stocks':>6}  "
          f"{'Invested':>12}  {'P&L Rs':>12}  {'P&L%':>7}")
        p(f"  {'─'*5} {'─'*30} {'─'*6}  {'─'*12}  {'─'*12}  {'─'*7}")
        for i, s in enumerate(summary_sorted, 1):
            medal = {1:'🥇', 2:'🥈', 3:'🥉'}.get(i, f'#{i} ')
            p(f"  {medal:<5} {s['name'].replace('_',' '):<30} "
              f"{s['stocks']:>6}  "
              f"Rs{s['invested']:>10,.0f}  "
              f"Rs{s['pl_abs']:>+10,.0f}  "
              f"{s['pl_pct']:>+7.1f}%")
        p(f"{'═'*90}")

    return lines

# ═══════════════════════════════════════════════════════════════
# MAIN FLOW FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def run_lt_full():
    all_lines            = []
    lt_lines, lt_symbols = build_lt_recommendations()
    for line in lt_lines:
        print(line)
    all_lines += lt_lines + ['', '']

    choice = input("\n  Update LT portfolio? (y/n): ").strip().lower()
    if choice == 'y':
        lt_df = manage_portfolio(LT_FILE, LT_COLS,
                                 'Long Term', is_swing=False)
    else:
        lt_df = load_portfolio(LT_FILE, LT_COLS)
        if len(lt_df) > 0:
            lt_df = update_rank_changes(lt_df)

    if len(lt_df) > 0:
        review = run_lt_review(lt_df)
        for line in review:
            print(line)
        all_lines += review

    path = os.path.join(REPORTS_DIR, f'lt_portfolio_{today_file}.txt')
    save_report(all_lines, path)
    return lt_symbols

def run_swing_full(lt_symbols=None):
    if lt_symbols is None:
        _, lt_symbols = build_lt_recommendations()

    all_lines   = []
    swing_lines = build_swing_recommendations(lt_symbols)
    for line in swing_lines:
        print(line)
    all_lines += swing_lines + ['', '']

    choice = input("\n  Update Swing portfolio? (y/n): ").strip().lower()
    if choice == 'y':
        swing_df = manage_portfolio(SWING_FILE, SWING_COLS,
                                    'Swing', is_swing=True)
    else:
        swing_df = load_portfolio(SWING_FILE, SWING_COLS)
        if len(swing_df) > 0:
            swing_df = update_rank_changes(swing_df)

    if len(swing_df) > 0:
        review = run_swing_review(swing_df)
        for line in review:
            print(line)
        all_lines += review

    path = os.path.join(REPORTS_DIR, f'swing_portfolio_{today_file}.txt')
    save_report(all_lines, path)

def run_model_review_full():
    lines = review_model_portfolios()
    for line in lines:
        print(line)
    if lines:
        path = os.path.join(
            REPORTS_DIR, f'model_portfolio_review_{today_file}.txt')
        save_report(lines, path)

# ═══════════════════════════════════════════════════════════════
# MAIN MENU
# ═══════════════════════════════════════════════════════════════
print("\n")
print("  1. Long Term Portfolio Review")
print("  2. Swing Trading Portfolio Review")
print("  3. Both Reviews (LT + Swing)")
print("  4. Create / Update Model Portfolio")
print("  5. Review Model Portfolios (P&L tracking)")
print("  6. All Reviews (LT + Swing + Model)")
print()
choice = input("  Enter choice (1/2/3/4/5/6): ").strip()

if choice == '1':
    run_lt_full()
elif choice == '2':
    run_swing_full()
elif choice == '3':
    lt_symbols = run_lt_full()
    run_swing_full(lt_symbols)
elif choice == '4':
    create_model_portfolio()
elif choice == '5':
    run_model_review_full()
elif choice == '6':
    lt_symbols = run_lt_full()
    run_swing_full(lt_symbols)
    run_model_review_full()
else:
    print("  Invalid choice.")

print()
print("=" * 60)
print("  Portfolio Manager complete!")
print(f"  Finished : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"  Reports  : {REPORTS_DIR}")
print("=" * 60)
