# ============================================================
# run_model_portfolio.py — AI Stock Screener (Indian Markets)
# Creates 10 pre-defined model portfolios automatically
# at current prices from technical_report_full.csv
#
# Run once to initialize:
#   python run_model_portfolio.py
#
# Then track weekly using Option 5 in run_portfolio.py
#
# 10 Portfolio Logics:
#   1.  Pure Tier 1 Equal Weight
#   2.  Pure Tier 1 Rank Weighted
#   3.  Tier 1+2 Rank Weighted  (main screener logic)
#   4.  High ML Confidence      (conf >= 75%)
#   5.  High Forecast 25d       (top by 25d forecast)
#   6.  High Tech Score         (tech score >= 80)
#   7.  Cap Diversified         (2-3 per cap category)
#   8.  Sector Concentrated     (best 3 from top 2 bull sectors)
#   9.  Bull Sector Only        (only ★★★ stocks — all 3 aligned)
#   10. Max Forecast 45d        (top by 45d forecast)
#
# All portfolios:
#   - Capital: Rs 1.5 Lakhs
#   - Only Bullish Continual + Momentum + EMA confirmed + ML conf >= 40
#   - Max 10 stocks per portfolio
#   - Saved to data/portfolio/model_portfolios/
# ============================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime

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

today_str  = datetime.now().strftime('%Y-%m-%d')
today_file = datetime.now().strftime('%Y%m%d')

# ── CONSTANTS ─────────────────────────────────────────────────
CAPITAL       = 150000   # Rs 1.5 Lakhs
MAX_STOCKS    = 10
MIN_CONF      = 40.0
CAP_ORDER     = ['Large Cap', 'Mini Large Cap', 'Mid Cap', 'Small Cap']

MODEL_COLS = [
    'Symbol', 'Entry_Price', 'Entry_Date', 'Quantity', 'Cap_Category',
    'Sector_Rank_At_Entry', 'Cap_Rank_At_Entry',
    'Sector_Rank_Change', 'Cap_Rank_Change', 'Notes'
]

# ── LOAD DATA ──────────────────────────────────────────────────
print("=" * 60)
print("  AI Stock Screener — Model Portfolio Creator")
print(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)
print("\nLoading data...")

tech_df    = pd.read_csv(os.path.join(SCORES_DIR,   'technical_report_full.csv'))
prefilt_df = pd.read_csv(os.path.join(UNIVERSE_DIR, 'prefilt_passed.csv'))
quality_df = pd.read_csv(os.path.join(UNIVERSE_DIR, 'quality_passed.csv'))

valid_symbols = set(quality_df['Symbol'].tolist())

print(f"     Tech report  : {len(tech_df)} stocks")
print(f"     Valid symbols: {len(valid_symbols)}")

# ── HELPER FUNCTIONS ───────────────────────────────────────────
def is_sector_bullish(row):
    return 'Uptrend' in str(row.get('Sector Trend', ''))

def is_sector_strong_bull(row):
    return 'Strong Uptrend' in str(row.get('Sector Trend', ''))

def passes_ema_filter(row):
    try:
        price  = float(row.get('Current Price', 0) or 0)
        ema50  = float(row.get('EMA50',  0) or 0)
        ema200 = float(row.get('EMA200', 0) or 0)
        return price > ema50 > ema200
    except:
        return False

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

def get_current_ranks(symbol):
    row = tech_df[tech_df['Symbol'] == symbol]
    if len(row) == 0:
        return 0.0, 0.0
    r = row.iloc[0]
    return (float(r.get('Sector Score', 0) or 0),
            float(r.get('Cap Score',    0) or 0))

def compute_equal_qty(price, n_stocks, capital=CAPITAL):
    """Equal weight — split capital equally."""
    alloc = capital / n_stocks
    qty   = max(1, int(alloc / price))
    return qty

def compute_weighted_qty(row, total_weight, capital=CAPITAL):
    """Rank weighted — proportional to priority score."""
    weight = compute_priority_score(row)
    alloc  = capital * (weight / total_weight) if total_weight > 0 else 0
    price  = float(row.get('Current Price', 0) or 0)
    qty    = max(1, int(alloc / price)) if price > 0 else 1
    return qty

def save_portfolio(df, name):
    filepath = os.path.join(MODEL_PORT_DIR, f"{name}.csv")
    df.to_csv(filepath, index=False)
    return filepath

def build_record(row, qty, notes=''):
    symbol    = str(row['Symbol'])
    price     = float(row.get('Current Price', 0) or 0)
    cap_cat   = str(row.get('Cap Category', '') or '')
    sec_rank  = float(row.get('Sector Score', 0) or 0)
    cap_rank  = float(row.get('Cap Score',    0) or 0)
    return {
        'Symbol'              : symbol,
        'Entry_Price'         : round(price, 2),
        'Entry_Date'          : today_str,
        'Quantity'            : qty,
        'Cap_Category'        : cap_cat,
        'Sector_Rank_At_Entry': sec_rank,
        'Cap_Rank_At_Entry'   : cap_rank,
        'Sector_Rank_Change'  : 0.0,
        'Cap_Rank_Change'     : 0.0,
        'Notes'               : notes,
    }

# ── BUILD BASE UNIVERSE ────────────────────────────────────────
# All portfolios use this filtered base:
# Bullish Continual + Momentum + EMA OK + ML conf >= 40

base = tech_df[
    (tech_df['ML_Prediction'] == 'Bullish Continual') &
    (tech_df['Best Setup']    == 'Momentum') &
    (tech_df['ML_Confidence'].fillna(0) >= MIN_CONF) &
    (tech_df['Symbol'].isin(valid_symbols))
].copy()

base = base[base.apply(passes_ema_filter, axis=1)].copy()
base['Priority_Score'] = base.apply(compute_priority_score, axis=1)

# Separate Tier 1 (sector bullish) and Tier 2
tier1 = base[base.apply(is_sector_bullish, axis=1)].copy()
tier2 = base[~base.apply(is_sector_bullish, axis=1)].copy()

print(f"\n     Base universe (filtered):")
print(f"       Tier 1 (Sector+ML+Tech): {len(tier1)} stocks")
print(f"       Tier 2 (ML+Tech only)  : {len(tier2)} stocks")
print(f"       Total                  : {len(base)} stocks")

# ── CHECK EXISTING PORTFOLIOS ──────────────────────────────────
existing = [f.replace('.csv','') for f in os.listdir(MODEL_PORT_DIR)
            if f.endswith('.csv')]
if existing:
    print(f"\n  ⚠️  Existing model portfolios found:")
    for name in existing:
        print(f"       {name}")
    print()
    choice = input(
        "  Overwrite all existing model portfolios? (y/n): ").strip().lower()
    if choice != 'y':
        print("  Exiting — no changes made.")
        exit()

# ── PORTFOLIO 1: PURE TIER 1 EQUAL WEIGHT ─────────────────────
print("\n[1/10] Pure Tier 1 Equal Weight...")
p1 = tier1.sort_values('Priority_Score', ascending=False).head(MAX_STOCKS)
if len(p1) < MAX_STOCKS:
    extra = MAX_STOCKS - len(p1)
    p1    = pd.concat([p1,
                       tier2.sort_values('Priority_Score', ascending=False)
                            .head(extra)])
records = []
for _, row in p1.iterrows():
    qty = compute_equal_qty(float(row.get('Current Price', 0) or 0), len(p1))
    records.append(build_record(row, qty, 'P1-Tier1-Equal'))
df1  = pd.DataFrame(records)
fp1  = save_portfolio(df1, 'p1_tier1_equal_weight')
print(f"     ✅ {len(df1)} stocks  →  {fp1.split(os.sep)[-1]}")

# ── PORTFOLIO 2: PURE TIER 1 RANK WEIGHTED ────────────────────
print("[2/10] Pure Tier 1 Rank Weighted...")
p2         = tier1.sort_values('Priority_Score', ascending=False).head(MAX_STOCKS)
if len(p2) < MAX_STOCKS:
    extra  = MAX_STOCKS - len(p2)
    p2     = pd.concat([p2,
                        tier2.sort_values('Priority_Score', ascending=False)
                             .head(extra)])
total_w2   = p2['Priority_Score'].sum()
records    = []
for _, row in p2.iterrows():
    qty    = compute_weighted_qty(row, total_w2)
    records.append(build_record(row, qty, 'P2-Tier1-RankWt'))
df2  = pd.DataFrame(records)
fp2  = save_portfolio(df2, 'p2_tier1_rank_weighted')
print(f"     ✅ {len(df2)} stocks  →  {fp2.split(os.sep)[-1]}")

# ── PORTFOLIO 3: TIER 1+2 RANK WEIGHTED (MAIN LOGIC) ──────────
print("[3/10] Tier 1+2 Rank Weighted (main screener logic)...")
p3       = base.sort_values('Priority_Score', ascending=False).head(MAX_STOCKS)
total_w3 = p3['Priority_Score'].sum()
records  = []
for _, row in p3.iterrows():
    qty  = compute_weighted_qty(row, total_w3)
    records.append(build_record(row, qty, 'P3-Tier1+2-RankWt'))
df3  = pd.DataFrame(records)
fp3  = save_portfolio(df3, 'p3_tier1_2_rank_weighted')
print(f"     ✅ {len(df3)} stocks  →  {fp3.split(os.sep)[-1]}")

# ── PORTFOLIO 4: HIGH ML CONFIDENCE (>= 75%) ──────────────────
print("[4/10] High ML Confidence (>= 75%)...")
p4       = base[base['ML_Confidence'].fillna(0) >= 75].copy()
p4       = p4.sort_values('ML_Confidence', ascending=False).head(MAX_STOCKS)
if len(p4) == 0:
    p4   = base.sort_values('ML_Confidence', ascending=False).head(MAX_STOCKS)
total_w4 = p4['Priority_Score'].sum()
records  = []
for _, row in p4.iterrows():
    qty  = compute_weighted_qty(row, total_w4)
    records.append(build_record(row, qty,
                                f"P4-MLConf-{row['ML_Confidence']:.0f}%"))
df4  = pd.DataFrame(records)
fp4  = save_portfolio(df4, 'p4_high_ml_confidence')
print(f"     ✅ {len(df4)} stocks  →  {fp4.split(os.sep)[-1]}")

# ── PORTFOLIO 5: HIGH FORECAST 25D ────────────────────────────
print("[5/10] High Forecast 25d...")
p5       = base.copy()
p5       = p5[p5['Forecast_25d_Pct'].notna() &
              (p5['Forecast_25d_Pct'] > 0)]
p5       = p5.sort_values('Forecast_25d_Pct', ascending=False).head(MAX_STOCKS)
total_w5 = p5['Priority_Score'].sum()
records  = []
for _, row in p5.iterrows():
    qty  = compute_weighted_qty(row, total_w5)
    records.append(build_record(row, qty,
                                f"P5-F25d-{row['Forecast_25d_Pct']:.1f}%"))
df5  = pd.DataFrame(records)
fp5  = save_portfolio(df5, 'p5_high_forecast_25d')
print(f"     ✅ {len(df5)} stocks  →  {fp5.split(os.sep)[-1]}")

# ── PORTFOLIO 6: HIGH TECH SCORE (>= 80) ──────────────────────
print("[6/10] High Tech Score (>= 80)...")
p6       = base[base['Tech Score'].fillna(0) >= 80].copy()
p6       = p6.sort_values('Tech Score', ascending=False).head(MAX_STOCKS)
if len(p6) == 0:
    p6   = base.sort_values('Tech Score', ascending=False).head(MAX_STOCKS)
total_w6 = p6['Priority_Score'].sum()
records  = []
for _, row in p6.iterrows():
    qty  = compute_weighted_qty(row, total_w6)
    records.append(build_record(row, qty,
                                f"P6-TechScore-{row['Tech Score']:.0f}"))
df6  = pd.DataFrame(records)
fp6  = save_portfolio(df6, 'p6_high_tech_score')
print(f"     ✅ {len(df6)} stocks  →  {fp6.split(os.sep)[-1]}")

# ── PORTFOLIO 7: CAP DIVERSIFIED (2-3 per cap category) ───────
print("[7/10] Cap Diversified (2-3 per cap category)...")
p7_records = []
per_cap    = 2  # start with 2 per cap, top up to 10 if needed
for cap in CAP_ORDER:
    cap_base = base[base['Cap Category'] == cap].sort_values(
        'Priority_Score', ascending=False).head(per_cap)
    for _, row in cap_base.iterrows():
        p7_records.append(row)

# If we have less than MAX_STOCKS, add more from top caps
if len(p7_records) < MAX_STOCKS:
    already  = set(r['Symbol'] for r in p7_records)
    more_needed = MAX_STOCKS - len(p7_records)
    extras   = base[~base['Symbol'].isin(already)].sort_values(
        'Priority_Score', ascending=False).head(more_needed)
    for _, row in extras.iterrows():
        p7_records.append(row)

p7       = pd.DataFrame(p7_records).head(MAX_STOCKS)
total_w7 = p7['Priority_Score'].sum()
records  = []
for _, row in p7.iterrows():
    qty  = compute_weighted_qty(row, total_w7)
    records.append(build_record(row, qty,
                                f"P7-CapDiv-{row['Cap Category']}"))
df7  = pd.DataFrame(records)
fp7  = save_portfolio(df7, 'p7_cap_diversified')
print(f"     ✅ {len(df7)} stocks  →  {fp7.split(os.sep)[-1]}")

# ── PORTFOLIO 8: SECTOR CONCENTRATED (top 2 bull sectors) ─────
print("[8/10] Sector Concentrated (best 3 from top 2 bull sectors)...")

# Find top 2 sectors by average Priority Score among Tier 1
if len(tier1) > 0:
    sector_scores = tier1.groupby('Sector')['Priority_Score'].mean()
    top_sectors   = sector_scores.sort_values(ascending=False).head(2).index.tolist()
else:
    sector_scores = base.groupby('Sector')['Priority_Score'].mean()
    top_sectors   = sector_scores.sort_values(ascending=False).head(2).index.tolist()

p8_records = []
for sector in top_sectors:
    sector_stocks = base[base['Sector'] == sector].sort_values(
        'Priority_Score', ascending=False).head(3)
    for _, row in sector_stocks.iterrows():
        p8_records.append(row)

# If less than MAX_STOCKS, fill from other sectors
if len(p8_records) < MAX_STOCKS:
    already     = set(r['Symbol'] for r in p8_records)
    more_needed = MAX_STOCKS - len(p8_records)
    extras      = base[~base['Symbol'].isin(already)].sort_values(
        'Priority_Score', ascending=False).head(more_needed)
    for _, row in extras.iterrows():
        p8_records.append(row)

p8       = pd.DataFrame(p8_records).head(MAX_STOCKS)
total_w8 = p8['Priority_Score'].sum()
records  = []
for _, row in p8.iterrows():
    qty  = compute_weighted_qty(row, total_w8)
    records.append(build_record(
        row, qty, f"P8-SectConc-{row['Sector'][:12]}"))
df8  = pd.DataFrame(records)
fp8  = save_portfolio(df8, 'p8_sector_concentrated')
print(f"     ✅ {len(df8)} stocks  →  {fp8.split(os.sep)[-1]}")
print(f"        Top sectors: {', '.join(top_sectors)}")

# ── PORTFOLIO 9: BULL SECTOR ONLY (★★★ only) ──────────────────
print("[9/10] Bull Sector Only (all 3 aligned — ★★★ stocks)...")
p9       = tier1.sort_values('Priority_Score', ascending=False).head(MAX_STOCKS)
if len(p9) == 0:
    print("     ⚠️  No Tier 1 stocks — using Tier 2")
    p9   = tier2.sort_values('Priority_Score', ascending=False).head(MAX_STOCKS)
total_w9 = p9['Priority_Score'].sum()
records  = []
for _, row in p9.iterrows():
    qty  = compute_weighted_qty(row, total_w9)
    records.append(build_record(row, qty, 'P9-BullSectorOnly'))
df9  = pd.DataFrame(records)
fp9  = save_portfolio(df9, 'p9_bull_sector_only')
print(f"     ✅ {len(df9)} stocks  →  {fp9.split(os.sep)[-1]}")

# ── PORTFOLIO 10: MAX FORECAST 45D ────────────────────────────
print("[10/10] Max Forecast 45d...")
p10      = base.copy()
p10      = p10[p10['Forecast_45d_Pct'].notna() &
               (p10['Forecast_45d_Pct'] > 0)]
p10      = p10.sort_values('Forecast_45d_Pct', ascending=False).head(MAX_STOCKS)
total_w10= p10['Priority_Score'].sum()
records  = []
for _, row in p10.iterrows():
    qty  = compute_weighted_qty(row, total_w10)
    records.append(build_record(row, qty,
                                f"P10-F45d-{row['Forecast_45d_Pct']:.1f}%"))
df10 = pd.DataFrame(records)
fp10 = save_portfolio(df10, 'p10_max_forecast_45d')
print(f"     ✅ {len(df10)} stocks  →  {fp10.split(os.sep)[-1]}")

# ── SUMMARY REPORT ────────────────────────────────────────────
print("\n" + "=" * 70)
print("  MODEL PORTFOLIO SUMMARY")
print(f"  Created : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"  Capital : Rs{CAPITAL:,.0f} per portfolio")
print("=" * 70)

portfolios = [
    ('p1_tier1_equal_weight',      df1,  'Pure Tier 1 Equal Weight'),
    ('p2_tier1_rank_weighted',     df2,  'Pure Tier 1 Rank Weighted'),
    ('p3_tier1_2_rank_weighted',   df3,  'Tier 1+2 Rank Weighted'),
    ('p4_high_ml_confidence',      df4,  'High ML Confidence (>=75%)'),
    ('p5_high_forecast_25d',       df5,  'High Forecast 25d'),
    ('p6_high_tech_score',         df6,  'High Tech Score (>=80)'),
    ('p7_cap_diversified',         df7,  'Cap Diversified'),
    ('p8_sector_concentrated',     df8,  'Sector Concentrated'),
    ('p9_bull_sector_only',        df9,  'Bull Sector Only (★★★)'),
    ('p10_max_forecast_45d',       df10, 'Max Forecast 45d'),
]

lines = []
def p(line=''): lines.append(str(line))

p("=" * 80)
p("  MODEL PORTFOLIO SUMMARY REPORT")
p(f"  Created : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
p(f"  Capital : Rs{CAPITAL:,.0f} per portfolio")
p(f"  Base    : Bullish Continual + Momentum + EMA OK + ML conf >= {MIN_CONF}%")
p("=" * 80)

for name, df_, label in portfolios:
    total_inv = (df_['Entry_Price'] * df_['Quantity']).sum()
    sectors   = df_['Cap_Category'].value_counts().to_dict()
    sect_str  = '  '.join([f"{k[:2]}:{v}" for k, v in sectors.items()])

    print(f"  {'─'*66}")
    print(f"  {label}")
    print(f"    Stocks: {len(df_)}  |  Deployed: Rs{total_inv:,.0f}  |  {sect_str}")

    p(f"\n{'─'*80}")
    p(f"  #{portfolios.index((name,df_,label))+1}  {label}")
    p(f"  Stocks: {len(df_)}  |  Deployed: Rs{total_inv:,.0f}  |  Cap mix: {sect_str}")
    p(f"{'─'*80}")
    p(f"  {'Symbol':<14} {'Cap':<16} {'Price':>8}  {'Qty':>5}  "
      f"{'Invested':>10}  {'SecRnk':>6}  {'CapRnk':>6}  "
      f"{'ML Conf':>7}  {'F25d%':>5}  {'Notes'}")
    p(f"  {'─'*14} {'─'*16} {'─'*8}  {'─'*5}  "
      f"{'─'*10}  {'─'*6}  {'─'*6}  "
      f"{'─'*7}  {'─'*5}  {'─'*20}")

    for _, row in df_.iterrows():
        sym     = str(row['Symbol'])
        price   = float(row['Entry_Price'])
        qty     = int(row['Quantity'])
        inv     = price * qty
        sec_r   = float(row['Sector_Rank_At_Entry'])
        cap_r   = float(row['Cap_Rank_At_Entry'])
        cap_cat = str(row['Cap_Category'])
        notes   = str(row['Notes'])

        # Get ML conf and forecast from tech_df
        trow    = tech_df[tech_df['Symbol'] == sym]
        ml_conf = float(trow.iloc[0].get('ML_Confidence', 0) or 0) \
                  if len(trow) > 0 else 0
        f25     = float(trow.iloc[0].get('Forecast_25d_Pct', 0) or 0) \
                  if len(trow) > 0 else 0

        p(f"  {sym:<14} {cap_cat:<16} {price:>8.2f}  {qty:>5}  "
          f"Rs{inv:>8,.0f}  {sec_r:>6.1f}  {cap_r:>6.1f}  "
          f"{ml_conf:>7.1f}%  {f25:>+5.1f}%  {notes}")

# Check overlaps between portfolios
p(f"\n{'═'*80}")
p(f"  PORTFOLIO OVERLAP ANALYSIS")
p(f"{'═'*80}")
p(f"  {'Portfolio':<35} {'Unique Stocks':>13}  Overlap with P3 (main)")

main_syms = set(df3['Symbol'].tolist())
for name, df_, label in portfolios:
    syms    = set(df_['Symbol'].tolist())
    overlap = len(syms & main_syms)
    unique  = len(syms - main_syms)
    p(f"  {label:<35} {unique:>8} unique  "
      f"{'|' * overlap} {overlap} common with P3")

p(f"\n{'═'*80}")
p(f"  NEXT STEPS")
p(f"{'═'*80}")
p(f"  1. Review this summary — check which stocks appear across portfolios")
p(f"  2. Run run_portfolio.py → Option 5 → to track P&L weekly")
p(f"  3. After 4-8 weeks, Option 5 will rank all 10 by performance")
p(f"  4. The best logic becomes your actual swing trading strategy")
p(f"{'═'*80}")

# Save summary report
report_path = os.path.join(
    REPORTS_DIR, f'model_portfolio_creation_{today_file}.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"\n{'═'*70}")
print(f"  All 10 portfolios created successfully!")
print(f"  Saved to : {MODEL_PORT_DIR}")
print(f"  Report   : {report_path}")
print(f"{'═'*70}")
print(f"\n  Next: run_portfolio.py → Option 5 to track weekly")
print(f"  In 4-8 weeks, you'll know which logic works best.")
