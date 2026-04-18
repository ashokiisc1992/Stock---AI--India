# ============================================================
# run_reversal.py — AI Stock Screener (Indian Markets)
# Reversal Hunter + Fundamental Deep Dive + Rank Movers
#
# Run manually whenever needed:
#   python run_reversal.py
#
# Menu:
#   1. Reversal Candidates     (Best Setup=Reversal, BotProb>=40%, Rank>=6.5)
#   2. Fundamental Deep Dive   (any stock — full data + absolute metrics)
#   3. Score & Rank Movers     (vs previous quarter snapshot)
#   4. All
#
# Reads from existing files — no yfinance calls, very fast.
#
# For Section 3 (Rank Movers) to work next quarter:
#   Before each quarterly run, copy:
#     data/fundamentals/fundamental_scores_full.csv
#       → data/fundamentals/fundamental_scores_prev.csv
# ============================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime
from difflib import get_close_matches

# ── PATHS ─────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, 'data')
SCORES_DIR   = os.path.join(DATA_DIR, 'scores')
FUND_DIR     = os.path.join(DATA_DIR, 'fundamentals')
UNIVERSE_DIR = os.path.join(DATA_DIR, 'universe')
REPORT_DIR   = os.path.join(DATA_DIR, 'reports', 'reversal')

os.makedirs(REPORT_DIR, exist_ok=True)

today_str  = datetime.now().strftime('%Y-%m-%d')
today_file = datetime.now().strftime('%Y%m%d')

# ── CONSTANTS ─────────────────────────────────────────────────
MIN_BOTTOM_PROB = 40.0
MIN_RANK        = 6.5
CAP_ORDER       = ['Large Cap', 'Mini Large Cap', 'Mid Cap', 'Small Cap']

# ── FILE PATHS ────────────────────────────────────────────────
TECH_FILE        = os.path.join(SCORES_DIR,   'technical_report_full.csv')
FUND_FILE        = os.path.join(FUND_DIR,     'fundamental_scores_full.csv')
FUND_PREV_FILE   = os.path.join(FUND_DIR,     'fundamental_scores_prev.csv')
FUND_METRICS_CSV = os.path.join(FUND_DIR,     'fundamental_metrics_full.csv')
PREFILT_FILE     = os.path.join(UNIVERSE_DIR, 'prefilt_passed.csv')

print("=" * 65)
print("  AI Stock Screener — Reversal & Fundamental Analysis")
print(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 65)
print("\nLoading data...")

# ── LOAD DATA ─────────────────────────────────────────────────
for fpath, label in [
    (TECH_FILE,        'technical_report_full.csv'),
    (FUND_FILE,        'fundamental_scores_full.csv'),
    (FUND_METRICS_CSV, 'fundamental_metrics_full.csv'),
    (PREFILT_FILE,     'prefilt_passed.csv'),
]:
    if not os.path.exists(fpath):
        print(f"\n  ERROR: {label} not found.")
        print(f"  Run run_weekly.py first.")
        exit(1)

tech_df         = pd.read_csv(TECH_FILE)
fund_df         = pd.read_csv(FUND_FILE)
fund_metrics_df = pd.read_csv(FUND_METRICS_CSV)
prefilt_df      = pd.read_csv(PREFILT_FILE)

print(f"  Tech report   : {len(tech_df)} stocks")
print(f"  Fund scores   : {len(fund_df)} stocks")
print(f"  Fund metrics  : {len(fund_metrics_df)} stocks")

# ── MERGE Market_Cap_Cr into fund_df if missing ───────────────
if 'Market_Cap_Cr' not in fund_df.columns:
    fund_df = fund_df.merge(
        prefilt_df[['Symbol', 'Market_Cap_Cr']], on='Symbol', how='left')

# ── CLASSIFY MCAP ─────────────────────────────────────────────
def classify_mcap(mcap_cr):
    try:
        v = float(mcap_cr)
        if v >= 20000: return 'Large Cap'
        elif v >= 5000: return 'Mini Large Cap'
        elif v >= 1000: return 'Mid Cap'
        else:           return 'Small Cap'
    except:
        return 'Small Cap'

fund_df['Cap Category'] = fund_df['Market_Cap_Cr'].apply(classify_mcap)

# ── COMPUTE SECTOR SCORE + CAP SCORE ─────────────────────────
# Sector Score: relative rank within same sector (0-10)
fund_df['Sector Score'] = 0.0
for sector in fund_df['Sector'].unique():
    mask      = fund_df['Sector'] == sector
    max_score = fund_df[mask]['Final Score'].max()
    if max_score > 0:
        fund_df.loc[mask, 'Sector Score'] = (
            fund_df[mask]['Final Score'] / max_score * 10).round(1)

# Cap Score: relative rank within same sector + cap category (0-10)
fund_df['Cap Score'] = 0.0
for sector in fund_df['Sector'].unique():
    for cap in CAP_ORDER:
        mask   = (fund_df['Sector'] == sector) & (fund_df['Cap Category'] == cap)
        subset = fund_df[mask]
        if len(subset) == 0:
            continue
        max_score = subset['Final Score'].max()
        if max_score > 0:
            fund_df.loc[mask, 'Cap Score'] = (
                subset['Final Score'] / max_score * 10).round(1)

# ── BUILD WORKING DATAFRAME ───────────────────────────────────
# tech_df is base (has all signals + Sector Score + Cap Score from weekly run)
# Merge fund breakdown columns from fund_df
merge_cols = ['Symbol', 'Historical Score', 'Peer Score', 'Quality Score',
              'Promoter Holding %', 'FII + DII %', 'Final Score', 'Market_Cap_Cr']

work_df = tech_df.copy()
for col in ['Historical Score', 'Peer Score', 'Quality Score',
            'Promoter Holding %', 'FII + DII %', 'Final Score', 'Market_Cap_Cr']:
    if col in work_df.columns:
        work_df = work_df.drop(columns=[col])

work_df = work_df.merge(fund_df[merge_cols], on='Symbol', how='left')
print(f"  Working df    : {work_df.shape[0]} stocks, {work_df.shape[1]} columns")

# ── HELPER FUNCTIONS ──────────────────────────────────────────
def mcap_str(mcap_cr):
    try:
        v = float(mcap_cr)
        if v >= 100000: return f"Rs{v/100000:.1f}L Cr"
        return f"Rs{v:,.0f} Cr"
    except:
        return "—"

def na(val, suffix='', decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '—'
    try:
        return f"{float(val):.{decimals}f}{suffix}"
    except:
        return str(val)

def flag(val, good_above=None, bad_below=None):
    try:
        v = float(val)
        if good_above is not None and v >= good_above: return '✅'
        if bad_below  is not None and v <= bad_below:  return '⚠️'
    except:
        pass
    return ''

def trend_arrow(val):
    try:
        v = float(val)
        if v > 0:   return f'▲{v:.2f}'
        elif v < 0: return f'▼{abs(v):.2f}'
        else:       return f'→{abs(v):.2f}'
    except:
        return '—'

all_symbols = fund_metrics_df['Symbol'].tolist()

def validate_symbol(user_input):
    s = user_input.strip().upper()
    if s in all_symbols:
        return s
    matches = get_close_matches(s, all_symbols, n=3, cutoff=0.6)
    if not matches:
        print(f"  '{s}' not found. No close matches.")
        return None
    print(f"  '{s}' not found. Did you mean:")
    for i, m in enumerate(matches, 1):
        print(f"    {i}. {m}")
    print(f"    0. None / skip")
    choice = input("  Enter number: ").strip()
    if choice in [str(i) for i in range(1, len(matches) + 1)]:
        return matches[int(choice) - 1]
    return None

# ── SECTION 1: REVERSAL CANDIDATES ───────────────────────────
def run_reversal_candidates():
    print("=" * 65)
    print("  REVERSAL HUNTER — Settings")
    print("=" * 65)
    try:
        top_n = int(input(
            "\n  How many stocks per cap category? [default 10]: "
        ).strip() or 10)
    except:
        top_n = 10

    reversal_df = work_df[
        (work_df['Best Setup']       == 'Reversal') &
        (work_df['Bottom_Rev_Prob']  >= MIN_BOTTOM_PROB) &
        (
            (work_df['Sector Score'] >= MIN_RANK) |
            (work_df['Cap Score']    >= MIN_RANK)
        )
    ].copy()
    reversal_df = reversal_df.sort_values('Bottom_Rev_Prob', ascending=False)

    print(f"\n  Total candidates : {len(reversal_df)}")

    if len(reversal_df) == 0:
        print(f"\n  No reversal candidates found with current filters.")
        print(f"  Market may be in strong uptrend — check again next week.")
        return

    print(f"\n  {'#':<3}  {'Symbol':<8}  {'MCap':>11}  {'Price':>8}  "
          f"{'BotProb':>7}  {'RevScr':>6}  {'SecRnk':>6}  {'CapRnk':>6}  "
          f"{'RSI':>5}  {'ADX':>5}  {'MACD':>7}  {'Vol':>5}")
    print(f"  {'─'*102}")

    serial = 1
    for cap in CAP_ORDER:
        cap_df = reversal_df[reversal_df['Cap Category'] == cap].head(top_n)
        if len(cap_df) == 0:
            continue
        cap_short = {'Large Cap': 'L', 'Mini Large Cap': 'ML',
                     'Mid Cap': 'M', 'Small Cap': 'S'}.get(cap, '?')
        print(f"\n  [{cap_short}] {cap}")
        for _, row in cap_df.iterrows():
            print(f"  {serial:<3}  "
                  f"{row['Symbol']:<8}  "
                  f"{mcap_str(row['Market Cap Cr']):>11}  "
                  f"Rs{row['Current Price']:>7.2f}  "
                  f"{row['Bottom_Rev_Prob']:>6.1f}%  "
                  f"{row['Reversal Score']:>6.0f}  "
                  f"{row['Sector Score']:>6.1f}  "
                  f"{row['Cap Score']:>6.1f}  "
                  f"{row['RSI']:>5.1f}  "
                  f"{row['ADX']:>5.1f}  "
                  f"{row['MACD Hist']:>+7.3f}  "
                  f"{row['Vol Ratio']:>4.2f}x")
            serial += 1

    print(f"\n  {'─'*102}")
    print(f"  BotProb=ML reversal prob  RevScr=Reversal tech score  "
          f"SecRnk/CapRnk=Relative rank 0-10")

    # Save report to file
    report_lines = []
    report_lines.append(f"REVERSAL CANDIDATES — {today_str}")
    report_lines.append(
        f"Filter: Best Setup=Reversal | BotProb>={MIN_BOTTOM_PROB}% | "
        f"SecRnk or CapRnk>={MIN_RANK}")
    report_lines.append(f"Total candidates: {len(reversal_df)}")
    report_lines.append("")
    for cap in CAP_ORDER:
        cap_df = reversal_df[reversal_df['Cap Category'] == cap].head(top_n)
        if len(cap_df) == 0:
            continue
        report_lines.append(f"[{cap}]")
        for _, row in cap_df.iterrows():
            report_lines.append(
                f"  {row['Symbol']:<8}  "
                f"{mcap_str(row['Market Cap Cr']):>11}  "
                f"Rs{row['Current Price']:>7.2f}  "
                f"BotProb:{row['Bottom_Rev_Prob']:.1f}%  "
                f"RevScr:{row['Reversal Score']:.0f}  "
                f"SecRnk:{row['Sector Score']:.1f}  "
                f"CapRnk:{row['Cap Score']:.1f}  "
                f"RSI:{row['RSI']:.1f}  "
                f"ADX:{row['ADX']:.1f}  "
                f"MACD:{row['MACD Hist']:+.3f}  "
                f"Vol:{row['Vol Ratio']:.2f}x")
        report_lines.append("")
    report_path = os.path.join(REPORT_DIR, f'reversal_{today_file}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n  Report saved: {report_path}")

# ── SECTION 2: FUNDAMENTAL DEEP DIVE ─────────────────────────
def show_fundamental(symbol):
    m_row = fund_metrics_df[fund_metrics_df['Symbol'] == symbol]
    s_row = fund_df[fund_df['Symbol'] == symbol]
    t_row = work_df[work_df['Symbol'] == symbol]

    if len(m_row) == 0:
        print(f"  {symbol} not found in fundamental metrics.")
        return

    m = m_row.iloc[0]
    s = s_row.iloc[0] if len(s_row) > 0 else None
    t = t_row.iloc[0] if len(t_row) > 0 else None

    sector   = m.get('Sector',   '—')
    industry = m.get('Industry', '—')
    mcap     = mcap_str(t['Market Cap Cr']) if t is not None else '—'
    price    = f"Rs{t['Current Price']:.2f}" if t is not None else '—'
    cap_cat  = t['Cap Category'] if t is not None else '—'

    print(f"\n  {'═'*65}")
    print(f"  {symbol}  |  {sector}  |  {industry}")
    print(f"  {cap_cat}  |  MCap {mcap}  |  Price {price}")
    print(f"  {'═'*65}")

    # SCORES
    if s is not None:
        print(f"\n  FUNDAMENTAL SCORES")
        print(f"  {'─'*65}")
        fin_s  = s.get('Final Score',      '—')
        hist_s = s.get('Historical Score', '—')
        peer_s = s.get('Peer Score',       '—')
        qual_s = s.get('Quality Score',    '—')
        fin_flag = '✅' if float(fin_s or 0) >= 65 else \
                   '⚠️' if float(fin_s or 0) < 50 else ''
        print(f"  Final Score      : {na(fin_s)}/100  {fin_flag}")
        print(f"  Historical Score : {na(hist_s)}/40   "
              f"(Revenue CAGR + Profit CAGR + OPM + ROE)")
        print(f"  Peer Score       : {na(peer_s)}/40   "
              f"(vs sector peers on all metrics)")
        print(f"  Quality Score    : {na(qual_s)}/20   "
              f"(Promoter + FII/DII + CF + consistency)")

    # RELATIVE RANKING
    if t is not None:
        print(f"\n  RELATIVE RANKING")
        print(f"  {'─'*65}")
        print(f"  Sector Rank  : {t['Sector Score']:.1f}/10  "
              f"(vs all {sector} stocks)")
        print(f"  Cap Rank     : {t['Cap Score']:.1f}/10  "
              f"(vs {cap_cat} {sector} stocks)")

    # GROWTH
    print(f"\n  GROWTH")
    print(f"  {'─'*65}")
    print(f"  TTM Revenue      : Rs{na(m.get('TTM Revenue'))} Cr")
    print(f"  Revenue CAGR 5Y  : {na(m.get('Revenue CAGR 5Y'),  suffix='%')}  "
          f"{flag(m.get('Revenue CAGR 5Y'),  good_above=15, bad_below=5)}")
    print(f"  Revenue CAGR 10Y : {na(m.get('Revenue CAGR 10Y'), suffix='%')}")
    print(f"  Revenue YoY Q    : {na(m.get('Revenue YoY Q'),    suffix='%')}  "
          f"{'Accelerating ✅' if m.get('Revenue Accelerating') == True else ''}")
    print(f"  Rev Consec YoY   : "
          f"{na(m.get('Revenue Consecutive YoY'), decimals=0)} quarters")
    print(f"\n  TTM Profit       : Rs{na(m.get('TTM Profit'))} Cr")
    print(f"  Profit CAGR 5Y   : {na(m.get('Profit CAGR 5Y'),  suffix='%')}  "
          f"{flag(m.get('Profit CAGR 5Y'),  good_above=15, bad_below=5)}")
    print(f"  Profit CAGR 10Y  : {na(m.get('Profit CAGR 10Y'), suffix='%')}")
    print(f"  Profit YoY Q     : {na(m.get('Profit YoY Q'),    suffix='%')}")
    print(f"  EPS Latest Q     : {na(m.get('Latest EPS Q'))}")
    print(f"  EPS YoY Growth   : {na(m.get('EPS YoY Growth'),  suffix='%')}")

    # MARGINS & RETURNS
    print(f"\n  MARGINS & RETURNS")
    print(f"  {'─'*65}")
    print(f"  OPM 5Y Avg       : {na(m.get('Avg OPM 5Y'),  suffix='%')}  "
          f"{flag(m.get('Avg OPM 5Y'),  good_above=20, bad_below=10)}")
    print(f"  OPM 10Y Avg      : {na(m.get('Avg OPM 10Y'), suffix='%')}")
    margin_status = 'Improving ✅' if m.get('Margin Improving') == True \
                    else 'Declining ⚠️' if m.get('Margin Improving') == False \
                    else ''
    print(f"  OPM Latest Q     : {na(m.get('Latest OPM Q'), suffix='%')}  "
          f"{margin_status}")
    print(f"  ROE              : {na(m.get('Final ROE'),     suffix='%')}  "
          f"{flag(m.get('Final ROE'), good_above=15, bad_below=8)}")
    latest_roce = m.get('Latest ROCE')
    avg_roce    = m.get('Avg ROCE 5Y')
    roce_status = 'Improving ✅' if m.get('ROCE Improving') == True \
                  else 'Declining ⚠️' if m.get('ROCE Improving') == False \
                  else ''
    print(f"  ROCE Latest      : {na(latest_roce, suffix='%')}  {roce_status}")
    if avg_roce is not None and latest_roce is not None:
        if round(float(avg_roce), 1) != round(float(latest_roce), 1):
            print(f"  ROCE 5Y Avg      : {na(avg_roce, suffix='%')}")
        else:
            print(f"  ROCE 5Y Avg      : — (insufficient historical data)")
    else:
        print(f"  ROCE 5Y Avg      : {na(avg_roce, suffix='%')}")

    # BALANCE SHEET
    print(f"\n  BALANCE SHEET")
    print(f"  {'─'*65}")
    debt_status = 'Reducing ✅' if m.get('Debt Reducing') == True \
                  else 'Rising ⚠️' if m.get('Debt Reducing') == False \
                  else ''
    print(f"  Debt to Equity   : {na(m.get('Debt to Equity'))}  {debt_status}")
    print(f"  Latest Debt      : Rs{na(m.get('Latest Debt'))} Cr")
    print(f"  Latest Equity    : Rs{na(m.get('Latest Equity'))} Cr")

    # CASH FLOW
    print(f"\n  CASH FLOW")
    print(f"  {'─'*65}")
    cf_pos   = float(m.get('CF Positive Years') or 0)
    cf_total = float(m.get('CF Total Years')    or 1)
    cf_pct   = round(cf_pos / cf_total * 100) if cf_total > 0 else 0
    cf_grow  = 'Growing ✅' if m.get('CF Growing') == True else ''
    print(f"  Operating CF     : Rs{na(m.get('Latest Operating CF'))} Cr  "
          f"{cf_grow}")
    print(f"  CF Consistency   : {cf_pos:.0f}/{cf_total:.0f} years positive "
          f"({cf_pct}%)  "
          f"{flag(cf_pct, good_above=90, bad_below=70)}")

    # OWNERSHIP
    print(f"\n  OWNERSHIP")
    print(f"  {'─'*65}")
    fii = float(m.get('FII Holding') or 0)
    dii = float(m.get('DII Holding') or 0)
    print(f"  Promoter Holding : {na(m.get('Promoter Holding'), suffix='%')}  "
          f"4Q: {trend_arrow(m.get('Promoter Change 4Q'))}  "
          f"8Q: {trend_arrow(m.get('Promoter Change 8Q'))}")
    print(f"  FII Holding      : {na(fii, suffix='%')}  "
          f"4Q: {trend_arrow(m.get('FII Change 4Q'))}")
    print(f"  DII Holding      : {na(dii, suffix='%')}  "
          f"4Q: {trend_arrow(m.get('DII Change 4Q'))}")
    print(f"  FII + DII Total  : {na(fii + dii, suffix='%')}")

    # TECHNICAL SNAPSHOT
    if t is not None:
        print(f"\n  TECHNICAL SNAPSHOT")
        print(f"  {'─'*65}")
        ml_conf     = t.get('ML_Confidence')
        ml_conf_str = f"{float(ml_conf):.1f}%" \
                      if ml_conf and float(ml_conf) > 0 else '—'
        print(f"  ML Prediction    : {t.get('ML_Prediction', '—')}  "
              f"({ml_conf_str})")
        print(f"  Best Setup       : {t.get('Best Setup', '—')}")
        print(f"  Sector Trend     : {t.get('Sector Trend', '—')}")
        print(f"  Forecast 25d     : {na(t.get('Forecast_25d_Pct'),  suffix='%')}")
        print(f"  Forecast 45d     : {na(t.get('Forecast_45d_Pct'),  suffix='%')}")
        print(f"  Forecast 180d    : {na(t.get('Forecast_180d_Pct'), suffix='%')}")

    print(f"\n  {'═'*65}")

def run_fundamental_deepdive():
    print("\n" + "=" * 65)
    print("  FUNDAMENTAL DEEP DIVE — Enter any stock symbol")
    print("=" * 65)
    while True:
        user_inp = input("\n  Symbol (or 'done'): ").strip()
        if user_inp.lower() == 'done':
            break
        sym = validate_symbol(user_inp)
        if sym:
            show_fundamental(sym)

# ── SECTION 3: SCORE & RANK MOVERS ───────────────────────────
def run_rank_movers():
    print("=" * 80)
    print("  SCORE & RANK MOVERS")
    print("=" * 80)

    if not os.path.exists(FUND_PREV_FILE):
        print("""
  No previous quarter data found.

  This section compares current vs previous quarter to find:
    3A — Absolute Score change  (Final Score up or down)
    3B — Relative Rank change   (Sector Rank + Cap Rank up or down)

  To enable this section next quarter:
    Before running run_quarterly.py, copy:
      fundamental_scores_full.csv
        → fundamental_scores_prev.csv
    in the data/fundamentals/ folder.

  After that this section will automatically show:
    RELATIVELY STRONGER  — rank improved even if score fell
    PEERS IMPROVED MORE  — score rose but rank fell
        """)
        return

    # Load and process previous quarter data
    fund_prev_df = pd.read_csv(FUND_PREV_FILE)
    if 'Market_Cap_Cr' not in fund_prev_df.columns:
        fund_prev_df = fund_prev_df.merge(
            prefilt_df[['Symbol', 'Market_Cap_Cr']], on='Symbol', how='left')
    fund_prev_df['Cap Category'] = fund_prev_df['Market_Cap_Cr'].apply(classify_mcap)

    # Recompute previous sector + cap ranks
    fund_prev_df['Prev Sector Score'] = 0.0
    for sector in fund_prev_df['Sector'].unique():
        mask      = fund_prev_df['Sector'] == sector
        max_score = fund_prev_df[mask]['Final Score'].max()
        if max_score > 0:
            fund_prev_df.loc[mask, 'Prev Sector Score'] = (
                fund_prev_df[mask]['Final Score'] / max_score * 10).round(1)

    fund_prev_df['Prev Cap Score'] = 0.0
    for sector in fund_prev_df['Sector'].unique():
        for cap in CAP_ORDER:
            mask   = (fund_prev_df['Sector'] == sector) & \
                     (fund_prev_df['Cap Category'] == cap)
            subset = fund_prev_df[mask]
            if len(subset) == 0:
                continue
            max_score = subset['Final Score'].max()
            if max_score > 0:
                fund_prev_df.loc[mask, 'Prev Cap Score'] = (
                    subset['Final Score'] / max_score * 10).round(1)

    # Merge current vs previous
    compare_df = fund_df[['Symbol', 'Sector', 'Final Score',
                           'Sector Score', 'Cap Score']].merge(
        fund_prev_df[['Symbol', 'Final Score',
                      'Prev Sector Score', 'Prev Cap Score']]
        .rename(columns={'Final Score': 'Prev Final Score'}),
        on='Symbol', how='inner')

    compare_df['Score Change']       = (compare_df['Final Score'] -
                                        compare_df['Prev Final Score']).round(1)
    compare_df['Sector Rank Change'] = (compare_df['Sector Score'] -
                                        compare_df['Prev Sector Score']).round(1)
    compare_df['Cap Rank Change']    = (compare_df['Cap Score'] -
                                        compare_df['Prev Cap Score']).round(1)
    compare_df['Rank Sum Change']    = (compare_df['Sector Rank Change'] +
                                        compare_df['Cap Rank Change']).round(1)

    def verdict(row):
        src = row['Sector Rank Change']
        crc = row['Cap Rank Change']
        if src > 0 and crc > 0:   return 'RELATIVELY STRONGER ✅'
        elif src < 0 and crc < 0: return 'RELATIVELY WEAKER ⚠️'
        elif src > 0 or crc > 0:  return 'MIXED →'
        else:                     return 'UNCHANGED →'

    compare_df['Verdict'] = compare_df.apply(verdict, axis=1)

    # ── 3A: Absolute Score Movers ──────────────────────────────
    print(f"\n  3A — ABSOLUTE SCORE CHANGE")
    print(f"  Note: market-wide decline may pull all scores down — "
          f"use 3B for true signal")
    print(f"  {'─'*78}")

    improved = compare_df[compare_df['Score Change'] > 0].sort_values(
        'Score Change', ascending=False).head(20)
    declined = compare_df[compare_df['Score Change'] < 0].sort_values(
        'Score Change').head(20)

    print(f"\n  TOP IMPROVERS (absolute score up)")
    print(f"  {'Symbol':<12} {'Sector':<25} {'Prev':>6}  {'Curr':>6}  "
          f"{'Chg':>6}  Verdict")
    print(f"  {'─'*78}")
    for _, row in improved.iterrows():
        print(f"  {row['Symbol']:<12} {str(row['Sector']):<25} "
              f"{row['Prev Final Score']:>6.1f}  "
              f"{row['Final Score']:>6.1f}  "
              f"▲{row['Score Change']:>5.1f}  "
              f"{row['Verdict']}")

    print(f"\n  TOP DECLINERS (absolute score down)")
    print(f"  {'Symbol':<12} {'Sector':<25} {'Prev':>6}  {'Curr':>6}  "
          f"{'Chg':>6}  Verdict")
    print(f"  {'─'*78}")
    for _, row in declined.iterrows():
        print(f"  {row['Symbol']:<12} {str(row['Sector']):<25} "
              f"{row['Prev Final Score']:>6.1f}  "
              f"{row['Final Score']:>6.1f}  "
              f"▼{abs(row['Score Change']):>5.1f}  "
              f"{row['Verdict']}")

    # ── 3B: Relative Rank Movers ───────────────────────────────
    print(f"\n  3B — RELATIVE RANK CHANGE")
    print(f"  Key insight: score can fall but rank can improve "
          f"if peers fell more")
    print(f"  {'─'*85}")

    rel_improved = compare_df[compare_df['Rank Sum Change'] > 0].sort_values(
        'Rank Sum Change', ascending=False).head(20)
    rel_declined = compare_df[compare_df['Rank Sum Change'] < 0].sort_values(
        'Rank Sum Change').head(20)

    print(f"\n  RELATIVELY STRONGER (rank improved vs peers)")
    print(f"  {'Symbol':<12} {'Sector':<22} {'AbsScr':>7}  "
          f"{'SecRnk':<16}  {'CapRnk':<16}  Verdict")
    print(f"  {'─'*90}")
    for _, row in rel_improved.iterrows():
        sc  = f"▲{row['Score Change']:.1f}" if row['Score Change'] > 0 \
              else f"▼{abs(row['Score Change']):.1f}"
        src = f"▲{row['Sector Rank Change']:.1f}"
        crc = f"▲{row['Cap Rank Change']:.1f}"
        print(f"  {row['Symbol']:<12} {str(row['Sector']):<22} "
              f"{sc:>7}  "
              f"{row['Prev Sector Score']:.1f}→{row['Sector Score']:.1f}({src})  "
              f"{row['Prev Cap Score']:.1f}→{row['Cap Score']:.1f}({crc})  "
              f"{row['Verdict']}")

    print(f"\n  RELATIVELY WEAKER (rank declined vs peers)")
    print(f"  {'Symbol':<12} {'Sector':<22} {'AbsScr':>7}  "
          f"{'SecRnk':<16}  {'CapRnk':<16}  Verdict")
    print(f"  {'─'*90}")
    for _, row in rel_declined.iterrows():
        sc  = f"▲{row['Score Change']:.1f}" if row['Score Change'] > 0 \
              else f"▼{abs(row['Score Change']):.1f}"
        src = f"▼{abs(row['Sector Rank Change']):.1f}"
        crc = f"▼{abs(row['Cap Rank Change']):.1f}"
        print(f"  {row['Symbol']:<12} {str(row['Sector']):<22} "
              f"{sc:>7}  "
              f"{row['Prev Sector Score']:.1f}→{row['Sector Score']:.1f}({src})  "
              f"{row['Prev Cap Score']:.1f}→{row['Cap Score']:.1f}({crc})  "
              f"{row['Verdict']}")

    print(f"\n  {'─'*90}")
    print(f"  AbsScr = Final Score change  |  "
          f"SecRnk/CapRnk = relative rank 0-10")
    print(f"  RELATIVELY STRONGER ✅ = both ranks improved vs peers")
    print(f"  RELATIVELY WEAKER ⚠️  = both ranks declined vs peers")
    print(f"  {'─'*90}")

# ── MAIN MENU ─────────────────────────────────────────────────
print("\n")
print("=" * 65)
print("  AI Stock Screener — Reversal & Fundamental Analysis")
print(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 65)
print()
print("  1. Reversal Candidates")
print("  2. Fundamental Deep Dive  (any stock)")
print("  3. Score & Rank Movers    (vs previous quarter)")
print("  4. All")
print()

choice = input("  Enter choice (1/2/3/4): ").strip()

if choice == '1':
    run_reversal_candidates()
elif choice == '2':
    run_fundamental_deepdive()
elif choice == '3':
    run_rank_movers()
elif choice == '4':
    run_reversal_candidates()
    run_fundamental_deepdive()
    run_rank_movers()
else:
    print("  Invalid choice.")

print()
print("=" * 65)
print("  Done!")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 65)
