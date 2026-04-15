# ============================================================
# run_quarterly.py — AI Stock Screener (Indian Markets)
# Day 11: Quarterly Automation
#
# Run once per quarter (or when universe needs full refresh):
#   python run_quarterly.py
#
# Force re-run all steps (ignore existing data):
#   python run_quarterly.py --force
#
# What this does:
#   Step 1  — Load master universe & deduplicate NSE/BSE
#   Step 2  — Pre-filter full universe (price/volume/mcap/listing)
#   Step 3  — Scrape fundamentals from Screener.in
#   Step 4  — Tiered quality filter → new quality_passed.csv
#   Step 5  — Fundamental scoring → fundamental_scores_full.csv
#   Step 6  — Full price re-download (incremental + new symbols)
#   Step 7  — Compute technical indicators
#   Step 8  — Build ML features
#   Step 9  — Run ML inference (first pass)
#   Step 10 — Full technical analysis (VP + consolidation)
#   Step 11 — Report data prep (scores, sector trend, breakouts)
#   Step 12 — Generate quarterly reports
#
# Estimated run time: 4–6 hours
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import os
import time
import warnings
import requests
import argparse
from bs4 import BeautifulSoup
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

# ── FORCE FLAG ────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true',
                    help='Force re-run all steps, ignore existing data')
args, _ = parser.parse_known_args()
FORCE_RERUN = args.force

print("=" * 60)
print("  AI Stock Screener — Quarterly Run")
print(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)
print(f"  Mode : {'FORCE RERUN' if FORCE_RERUN else 'Normal (resumes from checkpoint)'}")
print("  Steps 1–5  : Universe refresh + fundamentals (~4 hrs)")
print("  Steps 6–12 : Technical + ML + reports (~1.5 hrs)")
print("=" * 60)

# ── STEP 1: LOAD MASTER UNIVERSE & DEDUPLICATE ───────────────
print("\n[1/12] Loading master universe & deduplicating NSE/BSE...")

master = pd.read_csv(os.path.join(UNIVERSE_DIR, 'master_stocks.csv'))

print(f"     Total in master  : {len(master)}")
print(f"     NSE              : {len(master[master['Exchange'] == 'NSE'])}")
print(f"     BSE              : {len(master[master['Exchange'] == 'BSE'])}")

nse_stocks = master[master['Exchange'] == 'NSE'].copy()
bse_stocks = master[master['Exchange'] == 'BSE'].copy()

def normalize_name(name):
    name = str(name).upper().strip()
    for suffix in [' LIMITED', ' LTD', ' LTD.', ' PRIVATE',
                   ' PVT', ' PVT.', ' INC', ' CORP', ' CO.']:
        name = name.replace(suffix, '')
    name = ''.join(e for e in name if e.isalnum() or e == ' ')
    return name.strip()

nse_stocks['Name_Norm'] = nse_stocks['Company Name'].apply(normalize_name)
bse_stocks['Name_Norm'] = bse_stocks['Company Name'].apply(normalize_name)

nse_names      = set(nse_stocks['Name_Norm'])
bse_duplicates = bse_stocks[bse_stocks['Name_Norm'].isin(nse_names)]
bse_unique     = bse_stocks[~bse_stocks['Name_Norm'].isin(nse_names)]

master_clean = pd.concat([nse_stocks, bse_unique], ignore_index=True)
master_clean = master_clean.drop(columns=['Name_Norm'])

master_clean['Ticker'] = master_clean.apply(
    lambda row: f"{row['Symbol']}.NS" if row['Exchange'] == 'NSE'
                else f"{row['Symbol']}.BO", axis=1
)

print(f"\n     BSE duplicates removed : {len(bse_duplicates)}")
print(f"     BSE unique kept        : {len(bse_unique)}")
print(f"     Final clean universe   : {len(master_clean)}")
print(f"       NSE : {len(master_clean[master_clean['Exchange'] == 'NSE'])}")
print(f"       BSE : {len(master_clean[master_clean['Exchange'] == 'BSE'])}")

# ── STEP 2: PRE-FILTER FULL UNIVERSE ─────────────────────────
print("\n[2/12] Pre-filtering full universe (price/volume/mcap/listing)...")
print(f"       Started : {datetime.now().strftime('%H:%M:%S')}")
print(f"       Stocks  : {len(master_clean)}")

PREFILT_FILE       = os.path.join(UNIVERSE_DIR, 'prefilt_passed.csv')
PREFILT_CHECKPOINT = os.path.join(TEMP_DIR, 'prefilt_checkpoint.pkl')

MIN_PRICE        = 5
MIN_DAILY_VALUE  = 10_00_000
MIN_MARKET_CAP   = 50_00_00_000
MIN_LISTING_DAYS = 365
MIN_DATA_ROWS    = 200

def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def check_stock(symbol, exchange, ticker):
    result = {
        'Symbol'           : symbol,
        'Exchange'         : exchange,
        'Ticker'           : ticker,
        'passed'           : False,
        'fail_reason'      : None,
        'Price'            : None,
        'Avg_Daily_Value_L': None,
        'Market_Cap_Cr'    : None,
        'Data_Rows'        : None,
    }
    try:
        df_60 = yf.download(ticker, period='60d', interval='1d',
                            progress=False, auto_adjust=True)
        df_60 = flatten_df(df_60)
        if df_60 is None or len(df_60) < 5:
            result['fail_reason'] = 'No data'
            return result

        df_full = yf.download(ticker, period='max', interval='1d',
                              progress=False, auto_adjust=True)
        df_full = flatten_df(df_full)

        current_price = float(df_60['Close'].iloc[-1])
        avg_value     = float(df_60['Volume'].mean()) * current_price

        result['Price']             = round(current_price, 2)
        result['Avg_Daily_Value_L'] = round(avg_value / 1e5, 2)
        result['Data_Rows']         = len(df_full)

        if current_price < MIN_PRICE:
            result['fail_reason'] = f'Price Rs{current_price:.1f} < Rs5'
            return result

        if avg_value < MIN_DAILY_VALUE:
            result['fail_reason'] = f'Daily value Rs{avg_value/1e5:.1f}L < Rs10L'
            return result

        market_cap = None
        try:
            info       = yf.Ticker(ticker).fast_info
            market_cap = getattr(info, 'market_cap', None)
        except:
            pass

        if market_cap is not None:
            result['Market_Cap_Cr'] = round(market_cap / 1e7, 2)
            if market_cap < MIN_MARKET_CAP:
                result['fail_reason'] = f'MCap Rs{market_cap/1e7:.1f}Cr < Rs50Cr'
                return result

        if len(df_full) > 0:
            first_date   = df_full.index[0]
            listing_days = (datetime.now() - first_date.to_pydatetime().replace(tzinfo=None)).days
            if listing_days < MIN_LISTING_DAYS:
                result['fail_reason'] = f'Listed only {listing_days} days'
                return result

        if len(df_full) < MIN_DATA_ROWS:
            result['fail_reason'] = f'Only {len(df_full)} rows'
            return result

        result['passed'] = True
        return result

    except Exception as e:
        result['fail_reason'] = f'Error: {str(e)[:60]}'
        return result

if not FORCE_RERUN and os.path.exists(PREFILT_FILE):
    prefilt_df = pd.read_csv(PREFILT_FILE)
    print(f"\n     ⚠️  Pre-filter already done — {len(prefilt_df)} stocks passed")
    print(f"     Run with --force to redo")

else:
    if os.path.exists(PREFILT_CHECKPOINT):
        with open(PREFILT_CHECKPOINT, 'rb') as f:
            ckpt = pickle.load(f)
        results   = ckpt['results']
        start_idx = ckpt['next_idx']
        print(f"\n     Resuming from stock {start_idx}/{len(master_clean)}")
    else:
        results   = []
        start_idx = 0
        print(f"\n     Starting fresh — {len(master_clean)} stocks")

    total  = len(master_clean)
    passed = sum(1 for r in results if r.get('passed', False))
    failed = len(results) - passed
    print(f"     Already processed: {len(results)} | Passed: {passed} | Failed: {failed}")
    print("-" * 60)

    for i in range(start_idx, total):
        row    = master_clean.iloc[i]
        symbol = row['Symbol']
        exch   = row['Exchange']
        ticker = row['Ticker']

        r = check_stock(symbol, exch, ticker)
        results.append(r)

        if r['passed']:
            passed += 1
        else:
            failed += 1

        if (i + 1) % 50 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"     [{i+1:4d}/{total}] {pct:5.1f}% | "
                  f"Passed: {passed} | Failed: {failed} | "
                  f"Time: {datetime.now().strftime('%H:%M:%S')}")

        if (i + 1) % 100 == 0:
            with open(PREFILT_CHECKPOINT, 'wb') as f:
                pickle.dump({
                    'results'   : results,
                    'next_idx'  : i + 1,
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
                }, f)

        time.sleep(0.5)

    results_df = pd.DataFrame(results)
    prefilt_df = results_df[results_df['passed'] == True].copy()
    prefilt_df.to_csv(PREFILT_FILE, index=False)

    if os.path.exists(PREFILT_CHECKPOINT):
        os.remove(PREFILT_CHECKPOINT)

    print(f"\n     ✅ Pre-filter complete")
    print(f"        Total processed : {len(results_df)}")
    print(f"        Passed          : {len(prefilt_df)}")
    print(f"        Failed          : {len(results_df) - len(prefilt_df)}")
    print(f"\n     Top fail reasons:")
    print(results_df[results_df['passed'] == False]['fail_reason']
          .str.split('<').str[0].str.strip()
          .value_counts().head(8).to_string())

# ── STEP 3: FUNDAMENTAL SCRAPING ─────────────────────────────
print("\n[3/12] Scraping fundamentals from Screener.in...")
print(f"       Started : {datetime.now().strftime('%H:%M:%S')}")

FUND_METRICS_FILE = os.path.join(FUND_DIR, 'fundamental_metrics_full.csv')
FUND_RAW_FILE     = os.path.join(FUND_DIR, 'raw_stock_data_full.pkl')
FUND_CHECKPOINT   = os.path.join(TEMP_DIR, 'fund_scrape_checkpoint.pkl')

bse_raw = pd.read_csv(os.path.join(UNIVERSE_DIR, 'bse_stocks.csv'),
                      index_col=False,
                      usecols=['Security Code', 'Security Id',
                               'Issuer Name', 'Status',
                               'Group', 'Face Value',
                               'ISIN No', 'Instrument'])
bse_raw['Security Id'] = bse_raw['Security Id'].str.strip()
bse_code_map = dict(zip(bse_raw['Security Id'], bse_raw['Security Code']))
print(f"     BSE code map loaded: {len(bse_code_map)} symbols")

def extract_table(table):
    try:
        thead   = table.find('thead')
        columns = []
        if thead:
            for th in thead.find_all('th'):
                text = th.get_text(strip=True)
                columns.append(text if text else 'Metric')
        tbody = table.find('tbody')
        if not tbody:
            return None
        rows = tbody.find_all('tr')
        data = {}
        for row in rows:
            cells = row.find_all('td')
            if not cells:
                continue
            metric_name = cells[0].get_text(strip=True)
            metric_name = metric_name.replace('+', '').strip()
            skip_keywords = ['Raw PDF', 'PDF', 'Source']
            if any(kw in metric_name for kw in skip_keywords):
                continue
            values = []
            for cell in cells[1:]:
                val = cell.get_text(strip=True)
                val = val.replace(',', '').replace('%', '').strip()
                try:
                    values.append(float(val))
                except:
                    values.append(val if val else None)
            data[metric_name] = values
        if not data:
            return None
        col_names = columns[1:] if len(columns) > 1 else list(range(len(values)))
        return pd.DataFrame(data, index=col_names).T
    except:
        return None

def scrape_screener(lookup):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        urls = [
            f"https://www.screener.in/company/{lookup}/consolidated/",
            f"https://www.screener.in/company/{lookup}/",
        ]
        soup = None
        for url in urls:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                continue
            temp_soup = BeautifulSoup(response.text, 'html.parser')
            has_data  = False
            for section in temp_soup.find_all('section'):
                heading = section.find('h2')
                if heading and 'Profit' in heading.text:
                    table = section.find('table')
                    if table:
                        thead = table.find('thead')
                        if thead and len(thead.find_all('th')) > 1:
                            has_data = True
                    break
            if has_data:
                soup = temp_soup
                break
        if soup is None:
            return None

        target_sections = [
            'Quarterly Results', 'Profit & Loss', 'Balance Sheet',
            'Cash Flows', 'Ratios', 'Shareholding Pattern',
        ]
        result = {}
        for section in soup.find_all('section'):
            heading = section.find('h2')
            if not heading:
                continue
            section_name = heading.text.strip()
            if section_name not in target_sections:
                continue
            table = section.find('table')
            if not table:
                continue
            df = extract_table(table)
            if df is not None:
                result[section_name] = df
        for section in soup.find_all('section'):
            heading = section.find('h2')
            if heading and 'Peer' in heading.text:
                links       = section.find_all('a')
                sector_info = {}
                titles      = ['Broad Sector', 'Sector', 'Broad Industry', 'Industry']
                for link in links:
                    title = link.get('title', '')
                    if title in titles:
                        sector_info[title] = link.text.strip()
                result['Sector Info'] = sector_info
                break
        return result if result else None
    except:
        return None

def scrape_stock(symbol, exchange):
    result = scrape_screener(symbol)
    if result:
        return result
    if exchange == 'BSE':
        bse_code = bse_code_map.get(symbol)
        if bse_code:
            result = scrape_screener(str(int(bse_code)))
            if result:
                return result
    return None

def safe_cagr(end_val, start_val, years):
    try:
        if end_val and start_val and start_val > 0 and end_val > 0 and years > 0:
            return round(((end_val / start_val) ** (1 / years) - 1) * 100, 2)
    except:
        pass
    return None

def extract_all_metrics(symbol, data):
    m = {'Symbol': symbol}
    try:
        sector_info       = data.get('Sector Info', {})
        m['Sector']       = sector_info.get('Sector', None)
        m['Industry']     = sector_info.get('Industry', None)
        m['Broad Sector'] = sector_info.get('Broad Sector', None)

        financial_keywords = ['Financial', 'Banking', 'Insurance',
                              'NBFC', 'Lending', 'Microfinance']
        is_financial = any(kw in str(m.get('Sector', ''))
                           for kw in financial_keywords)

        if is_financial:
            revenue_row = 'Revenue'
            margin_row  = 'Financing Margin %'
            debt_row    = 'Borrowing'
        else:
            revenue_row = 'Sales'
            margin_row  = 'OPM %'
            debt_row    = 'Borrowings'

        pl = data.get('Profit & Loss')
        if pl is not None:
            pl_clean = pl.drop(columns=['TTM'], errors='ignore')
            if revenue_row in pl_clean.index:
                sales = pl_clean.loc[revenue_row].dropna()
                if len(sales) >= 6:
                    m['Revenue CAGR 5Y']  = safe_cagr(sales.iloc[-1], sales.iloc[-6], 5)
                if len(sales) >= 11:
                    m['Revenue CAGR 10Y'] = safe_cagr(sales.iloc[-1], sales.iloc[-11], 10)
                if len(sales) >= 3:
                    yrs  = min(len(sales) - 1, 10)
                    base = sales.iloc[-yrs - 1]
                    if base > 0:
                        m['Revenue CAGR Max']   = safe_cagr(sales.iloc[-1], base, yrs)
                        m['Revenue CAGR Years'] = yrs
            if margin_row in pl_clean.index:
                opm = pl_clean.loc[margin_row].dropna()
                m['Avg OPM 5Y']  = round(opm.iloc[-5:].mean(),  2) if len(opm) >= 5  else (round(opm.mean(), 2) if len(opm) > 0 else None)
                m['Avg OPM 10Y'] = round(opm.iloc[-10:].mean(), 2) if len(opm) >= 10 else None
            if 'Net Profit' in pl_clean.index:
                profit = pl_clean.loc['Net Profit'].dropna()
                if len(profit) >= 6 and profit.iloc[-6] > 0:
                    m['Profit CAGR 5Y']  = safe_cagr(profit.iloc[-1], profit.iloc[-6], 5)
                if len(profit) >= 11 and profit.iloc[-11] > 0:
                    m['Profit CAGR 10Y'] = safe_cagr(profit.iloc[-1], profit.iloc[-11], 10)
                if len(profit) >= 3:
                    yrs  = min(len(profit) - 1, 10)
                    base = profit.iloc[-yrs - 1]
                    if base > 0:
                        m['Profit CAGR Max']   = safe_cagr(profit.iloc[-1], base, yrs)
                        m['Profit CAGR Years'] = yrs

        qr = data.get('Quarterly Results')
        if qr is not None:
            qr_clean = qr.drop(columns=['TTM'], errors='ignore')
            if revenue_row in qr_clean.index:
                sales_q = qr_clean.loc[revenue_row].dropna()
                if len(sales_q) >= 4:
                    m['TTM Revenue'] = round(sales_q.iloc[-4:].sum(), 2)
                if len(sales_q) >= 5:
                    yr_ago = sales_q.iloc[-5]
                    if yr_ago > 0:
                        m['Revenue YoY Q'] = round(
                            (sales_q.iloc[-1] - yr_ago) / yr_ago * 100, 2)
                yoy = 0
                for i in range(1, min(5, len(sales_q) - 4)):
                    curr = sales_q.iloc[-i]
                    prev = sales_q.iloc[-i - 4]
                    if prev > 0 and curr > prev:
                        yoy += 1
                    else:
                        break
                m['Revenue Consecutive YoY'] = yoy
                if len(sales_q) >= 9:
                    r1 = (sales_q.iloc[-1] - sales_q.iloc[-5]) / sales_q.iloc[-5] * 100
                    r2 = (sales_q.iloc[-5] - sales_q.iloc[-9]) / sales_q.iloc[-9] * 100
                    m['Revenue Accelerating'] = r1 > r2
            if 'Net Profit' in qr_clean.index:
                profit_q = qr_clean.loc['Net Profit'].dropna()
                if len(profit_q) >= 4:
                    m['TTM Profit'] = round(profit_q.iloc[-4:].sum(), 2)
                if len(profit_q) >= 5:
                    yr_ago = profit_q.iloc[-5]
                    if yr_ago > 0:
                        m['Profit YoY Q'] = round(
                            (profit_q.iloc[-1] - yr_ago) / yr_ago * 100, 2)
                m['Profit Positive Q'] = int((profit_q > 0).sum())
                m['Total Quarters']    = len(profit_q)
            if margin_row in qr_clean.index:
                opm_q = qr_clean.loc[margin_row].dropna()
                m['Latest OPM Q'] = opm_q.iloc[-1] if len(opm_q) > 0 else None
                m['Avg OPM 4Q']   = round(opm_q.iloc[-4:].mean(), 2) if len(opm_q) >= 4 else None
                if len(opm_q) >= 5:
                    m['Margin Improving'] = bool(opm_q.iloc[-1] >= opm_q.iloc[-5])
            if 'EPS in Rs' in qr_clean.index:
                eps_q = qr_clean.loc['EPS in Rs'].dropna()
                m['Latest EPS Q'] = eps_q.iloc[-1] if len(eps_q) > 0 else None
                if len(eps_q) >= 5 and eps_q.iloc[-5] > 0:
                    m['EPS YoY Growth'] = round(
                        (eps_q.iloc[-1] - eps_q.iloc[-5]) / eps_q.iloc[-5] * 100, 2)

        bs = data.get('Balance Sheet')
        if bs is not None:
            if debt_row in bs.index:
                debt = bs.loc[debt_row].dropna()
                m['Latest Debt']   = debt.iloc[-1] if len(debt) > 0 else None
                if len(debt) >= 4:
                    m['Debt Reducing'] = bool(debt.iloc[-1] < debt.iloc[-4])
            if 'Reserves' in bs.index and 'Equity Capital' in bs.index:
                reserves = bs.loc['Reserves'].dropna()
                eq_cap   = bs.loc['Equity Capital'].dropna()
                if len(reserves) > 0 and len(eq_cap) > 0:
                    equity             = reserves.iloc[-1] + eq_cap.iloc[-1]
                    m['Latest Equity'] = round(equity, 2)
                    if m.get('TTM Profit') and equity > 0:
                        m['ROE'] = round(m['TTM Profit'] / equity * 100, 2)
                    if m.get('Latest Debt') is not None and equity > 0:
                        m['Debt to Equity'] = round(m['Latest Debt'] / equity, 2)

        cf = data.get('Cash Flows')
        if cf is not None:
            if 'Cash from Operating Activity' in cf.index:
                op_cf = cf.loc['Cash from Operating Activity'].dropna()
                m['Latest Operating CF'] = op_cf.iloc[-1] if len(op_cf) > 0 else None
                m['CF Positive Years']   = int((op_cf > 0).sum())
                m['CF Total Years']      = len(op_cf)
                if len(op_cf) >= 4:
                    m['CF Growing'] = bool(op_cf.iloc[-1] > op_cf.iloc[-4])

        ratios = data.get('Ratios')
        if ratios is not None:
            if 'ROCE %' in ratios.index:
                roce = ratios.loc['ROCE %'].dropna()
                m['Latest ROCE']    = roce.iloc[-1] if len(roce) > 0 else None
                if len(roce) >= 4:
                    m['ROCE Improving'] = bool(roce.iloc[-1] > roce.iloc[-4])
                if len(roce) >= 5:
                    m['Avg ROCE 5Y']    = round(roce.iloc[-5:].mean(), 2)
            if 'ROE %' in ratios.index:
                roe_s = ratios.loc['ROE %'].dropna()
                m['ROE from Ratios'] = roe_s.iloc[-1] if len(roe_s) > 0 else None
                m['ROE Avg 5Y']      = round(roe_s.iloc[-5:].mean(), 2) if len(roe_s) >= 5 else None
                m['ROE Improving']   = bool(roe_s.iloc[-1] > roe_s.iloc[-4]) if len(roe_s) >= 4 else None

        if pd.notna(m.get('ROE from Ratios')):
            m['Final ROE'] = m['ROE from Ratios']
        elif pd.notna(m.get('ROE')):
            m['Final ROE'] = m['ROE']
        else:
            m['Final ROE'] = None

        sh = data.get('Shareholding Pattern')
        if sh is not None:
            if 'Promoters' in sh.index:
                promoter = sh.loc['Promoters'].dropna()
                m['Promoter Holding']   = promoter.iloc[-1] if len(promoter) > 0 else None
                if len(promoter) >= 5:
                    m['Promoter Change 4Q'] = round(promoter.iloc[-1] - promoter.iloc[-5], 2)
                if len(promoter) >= 9:
                    m['Promoter Change 8Q'] = round(promoter.iloc[-1] - promoter.iloc[-9], 2)
            if 'FIIs' in sh.index:
                fii = sh.loc['FIIs'].dropna()
                m['FII Holding']   = fii.iloc[-1] if len(fii) > 0 else None
                if len(fii) >= 5:
                    m['FII Change 4Q'] = round(fii.iloc[-1] - fii.iloc[-5], 2)
            if 'DIIs' in sh.index:
                dii = sh.loc['DIIs'].dropna()
                m['DII Holding']   = dii.iloc[-1] if len(dii) > 0 else None
                if len(dii) >= 5:
                    m['DII Change 4Q'] = round(dii.iloc[-1] - dii.iloc[-5], 2)

    except Exception as e:
        pass
    return m

if not FORCE_RERUN and os.path.exists(FUND_METRICS_FILE):
    fund_metrics_df = pd.read_csv(FUND_METRICS_FILE)
    print(f"\n     ⚠️  Fundamental scrape already done — {len(fund_metrics_df)} stocks")
    print(f"     Run with --force to redo")

else:
    prefilt_df = pd.read_csv(PREFILT_FILE)
    print(f"     Stocks to scrape : {len(prefilt_df)}")
    print(f"     Estimated time   : ~{len(prefilt_df) * 2.5 / 60:.0f} minutes")

    if os.path.exists(FUND_CHECKPOINT):
        with open(FUND_CHECKPOINT, 'rb') as f:
            ckpt = pickle.load(f)
        all_stock_data = ckpt['raw_data']
        all_metrics    = ckpt['metrics']
        start_idx      = ckpt['next_idx']
        print(f"     Resuming from stock {start_idx}/{len(prefilt_df)}")
    else:
        all_stock_data = {}
        all_metrics    = []
        start_idx      = 0
        print(f"     Starting fresh")

    failed_stocks = []
    total         = len(prefilt_df)
    print(f"     Already scraped  : {len(all_metrics)} stocks")
    print("-" * 60)

    for i in range(start_idx, total):
        row      = prefilt_df.iloc[i]
        symbol   = row['Symbol']
        exchange = row['Exchange']

        try:
            data = scrape_stock(symbol, exchange)
            if data:
                all_stock_data[symbol] = data
                metrics                = extract_all_metrics(symbol, data)
                all_metrics.append(metrics)
            else:
                failed_stocks.append(symbol)
        except Exception as e:
            failed_stocks.append(symbol)

        if (i + 1) % 25 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"     [{i+1:4d}/{total}] {pct:5.1f}% | "
                  f"Scraped: {len(all_metrics)} | Failed: {len(failed_stocks)} | "
                  f"Time: {datetime.now().strftime('%H:%M:%S')}")

        if (i + 1) % 50 == 0:
            with open(FUND_CHECKPOINT, 'wb') as f:
                pickle.dump({
                    'raw_data' : all_stock_data,
                    'metrics'  : all_metrics,
                    'next_idx' : i + 1,
                }, f)

        time.sleep(2.5)

    with open(FUND_RAW_FILE, 'wb') as f:
        pickle.dump(all_stock_data, f)

    fund_metrics_df = pd.DataFrame(all_metrics)
    fund_metrics_df.to_csv(FUND_METRICS_FILE, index=False)

    if os.path.exists(FUND_CHECKPOINT):
        os.remove(FUND_CHECKPOINT)

    print(f"\n     ✅ Fundamental scrape complete")
    print(f"        Scraped  : {len(fund_metrics_df)} stocks")
    print(f"        Failed   : {len(failed_stocks)}")
    if failed_stocks:
        print(f"        Failed stocks: {failed_stocks[:20]}")

# ── STEP 4: TIERED QUALITY FILTER ────────────────────────────
print("\n[4/12] Applying tiered quality filter...")

QUALITY_FILE = os.path.join(UNIVERSE_DIR, 'quality_passed.csv')

def passes_tiered_filter(row):
    mcap   = row.get('Market_Cap_Cr', 0) or 0
    sector = str(row.get('Sector', ''))

    if mcap >= 20000:
        tier      = 'Large Cap';    min_roe = 6;  max_de = 2.5; min_promo = 15; min_rev = 200
    elif mcap >= 5000:
        tier      = 'Mini Large';   min_roe = 7;  max_de = 2.0; min_promo = 20; min_rev = 100
    elif mcap >= 1000:
        tier      = 'Mid Cap';      min_roe = 8;  max_de = 1.5; min_promo = 25; min_rev = 50
    else:
        tier      = 'Small Cap';    min_roe = 10; max_de = 1.0; min_promo = 30; min_rev = 20

    financial_sectors = ['Financial Services', 'Banking', 'Insurance']
    is_financial      = any(s in sector for s in financial_sectors)

    promoter = float(row.get('Promoter Holding') or 0)
    fii      = float(row.get('FII Holding')      or 0)
    dii      = float(row.get('DII Holding')      or 0)

    is_mnc           = (promoter < 5 and (fii > 40 or dii > 40))
    is_institutional = (promoter < 20 and (fii + dii) > 65)

    profit = row.get('TTM Profit')
    if profit is None or pd.isna(profit) or profit <= 0:
        return False, 'TTM Profit negative or zero', tier

    roe = row.get('Final ROE') or row.get('ROE from Ratios') or row.get('ROE')
    if roe is None or pd.isna(roe) or roe < min_roe:
        return False, f'ROE {round(roe,1) if roe else None}% < {min_roe}% ({tier})', tier

    if not is_financial:
        de = row.get('Debt to Equity')
        if de is not None and not pd.isna(de) and de > max_de:
            return False, f'D/E {de} > {max_de} ({tier})', tier

    if not is_mnc and not is_institutional:
        promo = row.get('Promoter Holding')
        if promo is None or pd.isna(promo) or promo < min_promo:
            return False, f'Promoter {promo}% < {min_promo}% ({tier})', tier

    rev = row.get('TTM Revenue')
    if rev is None or pd.isna(rev) or rev < min_rev:
        return False, f'Revenue {round(rev,0) if rev else None}Cr < {min_rev}Cr ({tier})', tier

    if not is_financial:
        if 'Consumer Durables' in sector:
            pq = row.get('Profit Positive Q') or 0
            tq = row.get('Total Quarters')    or 1
            if pd.isna(pq): pq = 0
            if (pq / tq) < 0.8:
                return False, f'Inconsistent profits: {pq}/{tq} quarters', tier
        else:
            cf_pos   = row.get('CF Positive Years') or 0
            cf_total = row.get('CF Total Years')    or 1
            if pd.isna(cf_pos): cf_pos = 0
            if (cf_pos / cf_total) < 0.60:
                return False, f'Poor CF: {cf_pos}/{cf_total} yrs', tier

    if not is_mnc and not is_institutional:
        p_change = row.get('Promoter Change 4Q')
        if p_change is not None and not pd.isna(p_change):
            if p_change < -5:
                return False, f'Promoter selling: {p_change}% in 4Q', tier

    pq = row.get('Profit Positive Q') or 0
    tq = row.get('Total Quarters')    or 1
    if pd.isna(pq): pq = 0
    if tq > 0 and (pq / tq) < 0.8:
        return False, f'Inconsistent profits: {pq}/{tq} quarters', tier

    return True, None, tier

if not FORCE_RERUN and os.path.exists(QUALITY_FILE):
    quality_df = pd.read_csv(QUALITY_FILE)
    print(f"\n     ⚠️  Quality filter already done — {len(quality_df)} stocks passed")
    print(f"     Run with --force to redo")

else:
    fund_metrics_df = pd.read_csv(FUND_METRICS_FILE)
    prefilt_df      = pd.read_csv(PREFILT_FILE)
    fund_df = fund_metrics_df.merge(
        prefilt_df[['Symbol', 'Market_Cap_Cr']], on='Symbol', how='left'
    )
    print(f"     Stocks to filter : {len(fund_df)}")

    results = []
    for _, row in fund_df.iterrows():
        passed, reason, tier = passes_tiered_filter(row)
        results.append({
            'Symbol'       : row['Symbol'],
            'Sector'       : row.get('Sector'),
            'Market_Cap_Cr': row.get('Market_Cap_Cr'),
            'Cap_Tier'     : tier,
            'Passed'       : passed,
            'Fail_Reason'  : reason,
        })

    results_df     = pd.DataFrame(results)
    passed_quality = results_df[results_df['Passed'] == True]
    failed_quality = results_df[results_df['Passed'] == False]

    passed_quality[['Symbol']].to_csv(QUALITY_FILE, index=False)
    results_df.to_csv(os.path.join(UNIVERSE_DIR, 'quality_filter_results.csv'), index=False)
    quality_df = passed_quality[['Symbol']].copy()

    print(f"\n     ✅ Quality filter complete")
    print(f"        Total input : {len(results_df)}")
    print(f"        Passed      : {len(passed_quality)}")
    print(f"        Failed      : {len(failed_quality)}")
    print(f"\n     Passed by tier:")
    print(passed_quality['Cap_Tier'].value_counts().to_string())
    print(f"\n     Top fail reasons:")
    print(failed_quality['Fail_Reason']
          .str.split('<').str[0].str.strip()
          .value_counts().head(8).to_string())
    print(f"\n     Passed by sector (top 10):")
    print(passed_quality['Sector'].value_counts().head(10).to_string())

# ── STEP 5: FUNDAMENTAL SCORING ───────────────────────────────
print("\n[5/12] Computing fundamental scores...")

FUND_SCORES_FILE = os.path.join(FUND_DIR, 'fundamental_scores_full.csv')

def score_metric_level(value, thresholds, reverse=False):
    if pd.isna(value):
        return None
    if reverse:
        if value <= thresholds[0]:   return 3
        elif value <= thresholds[1]: return 2
        elif value <= thresholds[2]: return 1
        else:                        return 0
    else:
        if value >= thresholds[0]:   return 3
        elif value >= thresholds[1]: return 2
        elif value >= thresholds[2]: return 1
        else:                        return 0

def calculate_historical_score(row):
    score      = 0
    cagr_years = float(row.get('Revenue CAGR Years') or row.get('Profit CAGR Years') or 0)
    if cagr_years < 5:   discount = 0.50
    elif cagr_years < 8: discount = 0.20
    else:                discount = 0.00

    rev_growth = float(row.get('Revenue CAGR 5Y') or row.get('Revenue CAGR 10Y') or row.get('Revenue CAGR Max') or 0)
    if rev_growth >= 20:   s = 15
    elif rev_growth >= 15: s = 12
    elif rev_growth >= 10: s = 9
    elif rev_growth >= 5:  s = 5
    else:                  s = 0
    score += s

    profit_growth = float(row.get('Profit CAGR 5Y') or row.get('Profit CAGR 10Y') or row.get('Profit CAGR Max') or 0)
    if profit_growth >= 20:   s = 15
    elif profit_growth >= 15: s = 12
    elif profit_growth >= 10: s = 9
    elif profit_growth >= 5:  s = 5
    else:                     s = 0
    score += s

    opm = float(row.get('Avg OPM 5Y') or row.get('Avg OPM 10Y') or 0)
    if opm >= 25:   s = 5
    elif opm >= 20: s = 4
    elif opm >= 15: s = 3
    elif opm >= 10: s = 2
    elif opm >= 5:  s = 1
    else:           s = 0
    score += s

    roe = float(row.get('Final ROE') or row.get('ROE') or 0)
    if roe >= 25:   s = 5
    elif roe >= 20: s = 4
    elif roe >= 15: s = 3
    elif roe >= 10: s = 2
    else:           s = 0
    score += s

    return min(round(score * (1 - discount), 1), 40)

def calculate_financial_score(row):
    score      = 0
    cagr_years = float(row.get('Revenue CAGR Years') or row.get('Profit CAGR Years') or 0)
    if cagr_years < 5:   discount = 0.50
    elif cagr_years < 8: discount = 0.20
    else:                discount = 0.00

    roe = float(row.get('Final ROE') or row.get('ROE') or 0)
    if roe >= 20:   s = 20
    elif roe >= 18: s = 17
    elif roe >= 15: s = 13
    elif roe >= 12: s = 8
    elif roe >= 10: s = 4
    else:           s = 0
    score += s

    profit_growth = float(row.get('Profit CAGR 5Y') or row.get('Profit CAGR 10Y') or row.get('Profit CAGR Max') or 0)
    if profit_growth >= 20:   s = 10
    elif profit_growth >= 15: s = 8
    elif profit_growth >= 10: s = 6
    elif profit_growth >= 5:  s = 3
    else:                     s = 0
    score += s

    rev_growth = float(row.get('Revenue CAGR 5Y') or row.get('Revenue CAGR 10Y') or row.get('Revenue CAGR Max') or 0)
    if rev_growth >= 20:   s = 10
    elif rev_growth >= 15: s = 8
    elif rev_growth >= 10: s = 6
    elif rev_growth >= 5:  s = 3
    else:                  s = 0
    score += s

    return min(round(score * (1 - discount), 1), 40)

def calculate_peer_scores(df):
    peer_scores  = []
    peer_metrics = {
        'Final ROE'        : True,
        'Revenue CAGR Max' : True,
        'Profit CAGR Max'  : True,
        'Avg OPM 5Y'       : True,
        'Latest ROCE'      : True,
        'Debt to Equity'   : False,
        'Revenue YoY Q'    : True,
        'Profit YoY Q'     : True,
    }
    for _, row in df.iterrows():
        symbol       = row['Symbol']
        sector       = row.get('Sector', 'Unknown')
        scores       = {}
        sector_peers = df[df['Sector'] == sector]
        for metric, higher_is_better in peer_metrics.items():
            stock_val = row.get(metric)
            if pd.isna(stock_val):
                scores[metric] = None
                continue
            peer_vals = sector_peers[metric].dropna()
            if len(peer_vals) < 2:
                scores[metric] = 2
                continue
            if higher_is_better:
                percentile = (peer_vals < stock_val).sum() / len(peer_vals) * 100
            else:
                percentile = (peer_vals > stock_val).sum() / len(peer_vals) * 100
            if percentile >= 80:   score = 5
            elif percentile >= 60: score = 4
            elif percentile >= 40: score = 3
            elif percentile >= 20: score = 2
            else:                  score = 1
            scores[metric] = score
        valid = [s for s in scores.values() if s is not None]
        total = round(sum(valid) / (len(valid) * 5) * 40, 1) if valid else 20
        peer_scores.append({'Symbol': symbol, 'Sector': sector, 'Peer Score': total})
    return pd.DataFrame(peer_scores)

def calculate_quality_score(row):
    scores = {}

    promoter = row.get('Promoter Holding')
    level    = score_metric_level(promoter, [60, 50, 40])
    scores['Promoter Holding'] = level if level is not None else 0

    promoter_change = float(row.get('Promoter Change 4Q') or 0)
    if pd.isna(promoter_change): promoter_change = 0
    if promoter_change > 1:      pt_score = 4
    elif promoter_change > 0:    pt_score = 3
    elif promoter_change >= -1:  pt_score = 2
    elif promoter_change >= -3:  pt_score = 1
    else:                        pt_score = 0
    scores['Promoter Trend'] = pt_score

    fii = float(row.get('FII Holding') or 0)
    dii = float(row.get('DII Holding') or 0)
    if pd.isna(fii): fii = 0
    if pd.isna(dii): dii = 0
    institutional = fii + dii
    if institutional < 10:   inst_score = 4
    elif institutional < 20: inst_score = 3
    elif institutional < 35: inst_score = 2
    else:                    inst_score = 1
    scores['Institutional'] = inst_score

    profit_q    = row.get('Profit Positive Q', 0) or 0
    total_q     = row.get('Total Quarters',    1) or 1
    consistency = (profit_q / total_q * 100) if total_q > 0 else 0
    if consistency >= 95:   pc_score = 4
    elif consistency >= 85: pc_score = 3
    elif consistency >= 75: pc_score = 2
    else:                   pc_score = 1
    scores['Profit Consistency'] = pc_score

    cf_years   = row.get('CF Positive Years', 0) or 0
    cf_total   = row.get('CF Total Years',    1) or 1
    cf_ratio   = (cf_years / cf_total * 100) if cf_total > 0 else 0
    cf_growing = row.get('CF Growing')
    if cf_ratio >= 90 and cf_growing == True: cq_score = 4
    elif cf_ratio >= 80:                      cq_score = 3
    elif cf_ratio >= 65:                      cq_score = 2
    else:                                     cq_score = 1
    scores['CF Quality'] = cq_score

    return {
        'Symbol'            : row['Symbol'],
        'Sector'            : row.get('Sector'),
        'Quality Score'     : sum(scores.values()),
        'Promoter Holding %': promoter,
        'FII + DII %'       : round(fii + dii, 2),
    }

if not FORCE_RERUN and os.path.exists(FUND_SCORES_FILE):
    fund_scores_df = pd.read_csv(FUND_SCORES_FILE)
    print(f"\n     ⚠️  Fundamental scoring already done — {len(fund_scores_df)} stocks")
    print(f"     Run with --force to redo")

else:
    fund_metrics_df = pd.read_csv(FUND_METRICS_FILE)
    quality_df      = pd.read_csv(QUALITY_FILE)
    prefilt_df      = pd.read_csv(PREFILT_FILE)

    fund_df = fund_metrics_df[
        fund_metrics_df['Symbol'].isin(quality_df['Symbol'])
    ].copy().reset_index(drop=True)
    fund_df = fund_df.merge(
        prefilt_df[['Symbol', 'Market_Cap_Cr']], on='Symbol', how='left'
    )
    print(f"     Stocks to score : {len(fund_df)}")

    financial_sectors = ['Financial Services', 'Banking', 'Insurance']

    historical_scores = []
    for _, row in fund_df.iterrows():
        sector = str(row.get('Sector', ''))
        is_fin = any(kw in sector for kw in financial_sectors)
        hist_score = calculate_financial_score(row) if is_fin else calculate_historical_score(row)
        historical_scores.append({'Symbol': row['Symbol'], 'Sector': sector, 'Historical Score': hist_score})

    historical_df     = pd.DataFrame(historical_scores)
    peer_df           = calculate_peer_scores(fund_df)
    quality_scores    = [calculate_quality_score(row) for _, row in fund_df.iterrows()]
    quality_df_scores = pd.DataFrame(quality_scores)

    final_df = historical_df[['Symbol', 'Sector', 'Historical Score']].merge(
        peer_df[['Symbol', 'Peer Score']], on='Symbol'
    ).merge(
        quality_df_scores[['Symbol', 'Quality Score', 'Promoter Holding %', 'FII + DII %']], on='Symbol'
    )

    final_df['Final Score'] = (
        final_df['Historical Score'] +
        final_df['Peer Score']       +
        final_df['Quality Score']
    ).round(1)

    final_df = final_df.sort_values('Final Score', ascending=False).reset_index(drop=True)
    final_df.to_csv(FUND_SCORES_FILE, index=False)
    fund_scores_df = final_df

    print(f"\n     ✅ Fundamental scoring complete")
    print(f"        Stocks scored : {len(final_df)}")
    print(f"\n     Top 10:")
    for i, row in final_df.head(10).iterrows():
        print(f"       {i+1:3}. {row['Symbol']:<15} {row['Final Score']:>6}/100  {row['Sector']}")

print(f"\n     Done — {len(fund_scores_df)} stocks scored")

# ── STEP 6: PRICE DATA ────────────────────────────────────────
print("\n[6/12] Updating price data...")
print(f"       Started : {datetime.now().strftime('%H:%M:%S')}")

PRICE_FILE = os.path.join(PRICES_DIR, 'price_data_full.pkl')

quality_df = pd.read_csv(QUALITY_FILE)
prefilt_df = pd.read_csv(PREFILT_FILE)

ticker_map = {}
for _, row in prefilt_df.iterrows():
    sym = row['Symbol']
    ticker_map[sym] = f"{sym}.NS" if row['Exchange'] == 'NSE' else f"{sym}.BO"

symbols = quality_df['Symbol'].tolist()
print(f"       Stocks  : {len(symbols)}")

if not FORCE_RERUN and os.path.exists(PRICE_FILE):
    with open(PRICE_FILE, 'rb') as f:
        price_data = pickle.load(f)
    existing_symbols = set(price_data.keys())
    new_symbols      = [s for s in symbols if s not in existing_symbols]
    print(f"\n       Existing price data  : {len(existing_symbols)} stocks")
    print(f"       New symbols to fetch : {len(new_symbols)}")

    if len(new_symbols) == 0:
        print(f"       ✅ All symbols have price data — running incremental update")
        # Incremental update for all existing symbols
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
                today = datetime.now().strftime('%Y-%m-%d')
                if start_date >= today:
                    skipped += 1
                    continue
                new_df = yf.download(ticker, start=start_date, end=today,
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
                        price_data[symbol] = combined.sort_index()
                    else:
                        price_data[symbol] = new_df
                    updated += 1
                else:
                    skipped += 1
            except Exception as e:
                failed.append(symbol)
            if (i + 1) % 50 == 0 or (i + 1) == len(symbols):
                pct = (i + 1) / len(symbols) * 100
                print(f"       [{i+1:4d}/{len(symbols)}] {pct:5.1f}% | "
                      f"Updated: {updated} | Skipped: {skipped} | "
                      f"Failed: {len(failed)} | "
                      f"Time: {datetime.now().strftime('%H:%M:%S')}")
            time.sleep(0.5)
        with open(PRICE_FILE, 'wb') as f:
            pickle.dump(price_data, f)
        print(f"\n       ✅ Incremental update done")
        print(f"          Updated: {updated} | Skipped: {skipped} | Failed: {len(failed)}")
    else:
        # Fetch only new symbols, keep existing
        print(f"       Fetching {len(new_symbols)} new symbols...")
        failed = []
        for i, symbol in enumerate(new_symbols):
            ticker = ticker_map.get(symbol, f"{symbol}.NS")
            try:
                df = yf.download(ticker, period='max', interval='1d',
                                 progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                df = df.loc[:, ~df.columns.duplicated()]
                if df is not None and len(df) >= 50:
                    price_data[symbol] = df
                else:
                    failed.append(symbol)
            except Exception as e:
                failed.append(symbol)
            if (i + 1) % 50 == 0 or (i + 1) == len(new_symbols):
                pct = (i + 1) / len(new_symbols) * 100
                print(f"       [{i+1:4d}/{len(new_symbols)}] {pct:5.1f}% | "
                      f"Failed: {len(failed)} | "
                      f"Time: {datetime.now().strftime('%H:%M:%S')}")
            time.sleep(0.5)
        with open(PRICE_FILE, 'wb') as f:
            pickle.dump(price_data, f)
        print(f"\n       ✅ New symbols fetched")
        print(f"          Total: {len(price_data)} | Failed: {len(failed)}")

else:
    # FORCE_RERUN — full fresh download
    print(f"\n       Full re-download — {len(symbols)} stocks")
    print(f"       Estimated time : ~{len(symbols) * 0.5 / 60:.0f} minutes")
    print("-" * 60)
    price_data = {}
    failed     = []
    for i, symbol in enumerate(symbols):
        ticker = ticker_map.get(symbol, f"{symbol}.NS")
        try:
            df = yf.download(ticker, period='max', interval='1d',
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df = df.loc[:, ~df.columns.duplicated()]
            if df is not None and len(df) >= 50:
                price_data[symbol] = df
            else:
                failed.append(symbol)
        except Exception as e:
            failed.append(symbol)
        if (i + 1) % 50 == 0 or (i + 1) == len(symbols):
            pct = (i + 1) / len(symbols) * 100
            print(f"       [{i+1:4d}/{len(symbols)}] {pct:5.1f}% | "
                  f"Fetched: {len(price_data)} | Failed: {len(failed)} | "
                  f"Time: {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(0.5)
    with open(PRICE_FILE, 'wb') as f:
        pickle.dump(price_data, f)
    print(f"\n       ✅ Price data saved")
    print(f"          Fetched: {len(price_data)} | Failed: {len(failed)}")
    if failed:
        print(f"          Failed : {failed[:20]}")

print(f"\n     Done — {len(price_data)} stocks in price_data")

# ── STEP 7: COMPUTE TECHNICAL INDICATORS ─────────────────────
print("\n[7/12] Computing technical indicators...")
print(f"       Started : {datetime.now().strftime('%H:%M:%S')}")

INDICATOR_FILE = os.path.join(PRICES_DIR, 'indicator_data_full.pkl')

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
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr14      = tr.ewm(com=13,    adjust=False).mean()
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
failed_ind     = []
for symbol, df in price_data.items():
    try:
        indicator_data[symbol] = compute_indicators(df)
    except Exception as e:
        failed_ind.append(symbol)

with open(INDICATOR_FILE, 'wb') as f:
    pickle.dump(indicator_data, f)

print(f"     Done — {len(indicator_data)} stocks computed")
if failed_ind:
    print(f"     Failed : {len(failed_ind)} — {failed_ind[:10]}")

# ── STEP 8: BUILD ML FEATURES ─────────────────────────────────
print("\n[8/12] Building ML features...")
print(f"       Started : {datetime.now().strftime('%H:%M:%S')}")

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
    df['return_1d']     = df['Close'].pct_change(1)
    df['return_5d']     = df['Close'].pct_change(5)
    df['return_20d']    = df['Close'].pct_change(20)
    df['return_60d']    = df['Close'].pct_change(60)
    df['52w_high']      = df['Close'].rolling(252).max()
    df['52w_low']       = df['Close'].rolling(252).min()
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
failed_feat  = []
for symbol, df in indicator_data.items():
    try:
        all_features[symbol] = build_features(symbol, df)
    except Exception as e:
        failed_feat.append(symbol)

print(f"     Done — {len(all_features)} stocks")
if failed_feat:
    print(f"     Failed : {len(failed_feat)} — {failed_feat[:10]}")

# ── STEP 9: LOAD ML MODELS + FIRST PASS INFERENCE ────────────
print("\n[9/12] Loading ML models and running inference...")
print(f"       Started : {datetime.now().strftime('%H:%M:%S')}")

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

print(f"     Models loaded for {len(group_stocks)} groups")

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
        bottom_prob          = bottom_models[group].predict_proba(X)[0][1]
        result['Bottom_Rev_Prob'] = round(float(bottom_prob) * 100, 1)
        result['Bottom_Rev_Flag'] = bottom_prob >= 0.60
        top_prob                  = top_models[group].predict_proba(X)[0][1]
        result['Top_Rev_Prob']    = round(float(top_prob) * 100, 1)
        result['Top_Rev_Flag']    = top_prob >= 0.60

    if group in bullish_cont_models:
        sym_enc              = get_symbol_enc(symbol, bullish_cont_encoders[group])
        latest['symbol_enc'] = sym_enc
        X                    = latest[feat_cols]
        bc_prob              = bullish_cont_models[group].predict_proba(X)[0][1]
        result['Bullish_Cont_Prob'] = round(float(bc_prob) * 100, 1)

    if group in bearish_cont_models:
        sym_enc              = get_symbol_enc(symbol, bearish_cont_encoders[group])
        latest['symbol_enc'] = sym_enc
        X                    = latest[feat_cols]
        dc_prob              = bearish_cont_models[group].predict_proba(X)[0][1]
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
            result[f'Forecast_{horizon}_Pct']   = round(float(pred_return) * 100, 1)
            result[f'Forecast_{horizon}_Price'] = round(float(pred_price), 2)

    forecast_25d            = result['Forecast_25d_Pct'] or 0
    ml_label                = assign_ml_label(best_setup, forecast_25d)
    result['ML_Prediction'] = ml_label
    result['ML_Confidence'] = assign_ml_confidence(ml_label, result)
    return result

try:
    old_scores = pd.read_csv(os.path.join(SCORES_DIR, 'technical_report_full.csv'))
    setup_map  = dict(zip(old_scores['Symbol'], old_scores['Best Setup']))
    print(f"     Using last known Best Setup from existing technical report")
except:
    setup_map  = {}
    print(f"     No existing technical report — using Watching as default")

all_ml_scores = []
for symbol in symbols:
    try:
        best_setup = setup_map.get(symbol, 'Watching')
        all_ml_scores.append(run_inference(symbol, best_setup))
    except:
        pass

print(f"     Done — {len(all_ml_scores)} stocks scored")
print(f"     (Second pass will run after technical analysis in Step 10)")

# ── STEP 10: FULL TECHNICAL ANALYSIS ─────────────────────────
print("\n[10/12] Running full technical analysis (VP + consolidation)...")
print(f"        Started : {datetime.now().strftime('%H:%M:%S')}")
print(f"        Estimated time: ~6 minutes")

TECH_CHECKPOINT = os.path.join(TEMP_DIR, 'tech_checkpoint.pkl')

fund_full  = pd.read_csv(FUND_SCORES_FILE)
prefilt_df = pd.read_csv(PREFILT_FILE)
fund_full  = fund_full.merge(
    prefilt_df[['Symbol', 'Market_Cap_Cr']], on='Symbol', how='left'
)

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
                overlap_pct         = (overlap_high - overlap_low) / candle_range
                volume_at_price[i] += candle_vol * overlap_pct
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

def score_momentum(df, vp):
    scores   = {}
    latest   = df.iloc[-1]
    close    = latest['Close']
    ema20    = latest['EMA20']
    ema50    = latest['EMA50']
    ema200   = latest['EMA200']
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
    macd_v   = latest.get('MACD', None)
    signal_v = latest.get('Signal', None)
    if macd_v is not None and signal_v is not None:
        if macd_v > signal_v and hist > 0 and macd_v > 0:  scores['MACD'] = 20
        elif macd_v > signal_v and hist > 0:               scores['MACD'] = 15
        elif macd_v > signal_v:                            scores['MACD'] = 10
        else:                                              scores['MACD'] = 0
    else:
        scores['MACD'] = 20 if hist > 0 else (10 if hist > -0.5 else 0)
    adx      = latest['ADX']
    di_plus  = latest['DI_Plus']
    di_minus = latest['DI_Minus']
    if adx > 25 and di_plus > di_minus:    scores['ADX'] = 15
    elif adx > 20 and di_plus > di_minus:  scores['ADX'] = 10
    elif adx > 25:                         scores['ADX'] = 5
    else:                                  scores['ADX'] = 0
    if vp:
        status = vp['breakout_status']
        if status == 'BREAKOUT_UP':     scores['VP'] = 15
        elif status == 'AT_RESISTANCE': scores['VP'] = 10
        elif status == 'INSIDE':        scores['VP'] = 5
        else:                           scores['VP'] = 0
    else:
        scores['VP'] = 5
    return sum(scores.values()), scores

def score_reversal(df, vp, lookback=20):
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
    print(f"        Resuming from checkpoint — {len(done_syms)} already done")
else:
    tech_reports = []
    done_syms    = set()

remaining = [s for s in symbols if s in indicator_data and s not in done_syms]
print(f"        Remaining : {len(remaining)} stocks")
print("-" * 60)

for i, symbol in enumerate(remaining):
    try:
        df       = indicator_data[symbol]
        latest   = df.iloc[-1]
        fund_row = fund_full[fund_full['Symbol'] == symbol]

        fund_score = float(fund_row['Final Score'].values[0]) \
                     if len(fund_row) > 0 and 'Final Score' in fund_row.columns else 50.0
        sector     = fund_row['Sector'].values[0] \
                     if len(fund_row) > 0 else 'Unknown'
        mcap_cr    = float(fund_row['Market_Cap_Cr'].values[0]) \
                     if len(fund_row) > 0 and 'Market_Cap_Cr' in fund_row.columns else 0

        macd_col = 'MACD_Hist' if 'MACD_Hist' in df.columns else 'MACD Hist'
        vol_col  = 'Vol_Ratio'  if 'Vol_Ratio'  in df.columns else 'Vol Ratio'

        close    = latest['Close']
        rsi      = round(latest['RSI'],      2)
        adx      = round(latest['ADX'],      2)
        ema20    = round(latest['EMA20'],    2)
        ema50    = round(latest['EMA50'],    2)
        ema200   = round(latest['EMA200'],   2)
        di_plus  = round(latest['DI_Plus'],  2)
        di_minus = round(latest['DI_Minus'], 2)
        hist     = round(latest[macd_col],   4)
        vol_r    = round(latest[vol_col],    2)

        vp           = calculate_volume_profile(df)
        mom_score, _ = score_momentum(df, vp)
        rev_score, _ = score_reversal(df, vp)
        consol_info  = get_consolidation_info(df)

        if mom_score >= rev_score and mom_score >= 50:
            best_setup = 'Momentum'
            tech_score = mom_score
        elif rev_score > mom_score and rev_score >= 50:
            best_setup = 'Reversal'
            tech_score = rev_score
        else:
            best_setup = 'Watching'
            tech_score = max(mom_score, rev_score)

        in_consol       = consol_info['consolidating']
        consol_days     = consol_info['consol_days']
        consol_label    = consol_info['duration_label']
        pct_to_breakout = consol_info['pct_to_breakout'] or 0
        consol_volume   = consol_info['breakout_volume']  or 0
        cap_category    = classify_mcap(mcap_cr)

        if fund_score >= 60 and tech_score >= 65 and best_setup != 'Watching':
            tier = f'TIER 1 — BUY NOW ({best_setup})'
        elif fund_score >= 60 and in_consol and pct_to_breakout > -5:
            tier = 'TIER 1 — BREAKOUT IMMINENT'
        elif fund_score >= 60 and tech_score >= 40 and best_setup != 'Watching':
            tier = 'TIER 2 — WATCHLIST'
        elif fund_score >= 60 and in_consol:
            tier = 'TIER 2 — BASE BUILDING'
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
            'In Consolidation': in_consol, 'Consol Days': consol_days,
            'Consol Label': consol_label,
            'Pct to Breakout': round(pct_to_breakout, 2),
            'Consol Volume': consol_volume,
        })
        done_syms.add(symbol)

    except Exception:
        pass

    if (i + 1) % 50 == 0 or (i + 1) == len(remaining):
        pct = (i + 1) / len(remaining) * 100
        print(f"        [{i+1:4d}/{len(remaining)}] {pct:5.1f}% | "
              f"Done: {len(tech_reports)} | "
              f"Time: {datetime.now().strftime('%H:%M:%S')}")
        with open(TECH_CHECKPOINT, 'wb') as f:
            pickle.dump({'reports': tech_reports, 'done': done_syms}, f)

# Second pass ML with correct Best Setup
tech_df       = pd.DataFrame(tech_reports)
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
tech_final.to_csv(os.path.join(SCORES_DIR, 'technical_report_full.csv'), index=False)
ml_scores_df.to_csv(os.path.join(SCORES_DIR, 'ml_scores_full.csv'), index=False)

if os.path.exists(TECH_CHECKPOINT):
    os.remove(TECH_CHECKPOINT)

print(f"\n     Done — {len(tech_final)} stocks analysed")
print(f"     Tier distribution:")
for tier, cnt in tech_final['Tier'].value_counts().items():
    print(f"       {tier}: {cnt}")

# ── STEP 11: REPORT DATA PREP ─────────────────────────────────
print("\n[11/12] Preparing report data...")
print(f"        Started : {datetime.now().strftime('%H:%M:%S')}")

tech_df    = pd.read_csv(os.path.join(SCORES_DIR, 'technical_report_full.csv'))
fund_full  = pd.read_csv(FUND_SCORES_FILE)
prefilt_df = pd.read_csv(PREFILT_FILE)

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
    fund_full[['Symbol', 'Sector Score', 'Cap Score']],
    on='Symbol', how='left'
)

SCORES_FILE    = os.path.join(SCORES_DIR, 'last_week_scores.csv')
rank_jumpers   = set()
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
print(f"        Rank jumpers : {len(rank_jumpers)}")

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

breakout_vol_data = {}
for _, row in tech_df[tech_df['In Consolidation'] == True].iterrows():
    symbol      = row['Symbol']
    consol_days = int(row['Consol Days'])
    if symbol not in price_data or consol_days < 5:
        continue
    df = price_data[symbol]
    if len(df) < consol_days + 5:
        breakout_vol_data[symbol] = 1.0
        continue
    week_vol   = df['Volume'].iloc[-5:].mean()
    consol_avg = df['Volume'].iloc[-consol_days:-5].mean()
    breakout_vol_data[symbol] = round(
        week_vol / consol_avg if consol_avg > 0 else 1.0, 2)
tech_df['Breakout Vol'] = tech_df['Symbol'].map(breakout_vol_data).fillna(0.0)

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

SECTOR_INDEX_MAP = {
    'Information Technology'            : '^CNXIT',
    'Healthcare'                        : '^CNXPHARMA',
    'Financial Services'                : 'FINIETF.NS',
    'Capital Goods'                     : '^CNXINFRA',
    'Chemicals'                         : 'MID150BEES.NS',
    'Consumer Durables'                 : '^CNXCONSUM',
    'Oil, Gas & Consumable Fuels'       : '^CNXENERGY',
    'Automobile and Auto Components'    : '^CNXAUTO',
    'Textiles'                          : 'MID150BEES.NS',
    'Consumer Services'                 : 'MID150BEES.NS',
    'Banking'                           : '^NSEBANK',
    'Fast Moving Consumer Goods'        : '^CNXFMCG',
    'Metals & Mining'                   : '^CNXMETAL',
    'Realty'                            : '^CNXREALTY',
    'Services'                          : '^CNXSERVICE',
    'Construction'                      : '^CNXINFRA',
    'Construction Materials'            : '^CNXINFRA',
    'Power'                             : '^CNXENERGY',
    'Pharmaceuticals'                   : '^CNXPHARMA',
    'Cement'                            : '^CNXCMDT',
    'Defence'                           : 'MID150BEES.NS',
    'Telecommunication'                 : 'MID150BEES.NS',
    'Diversified'                       : 'MID150BEES.NS',
    'Utilities'                         : '^CNXENERGY',
    'Media Entertainment & Publication' : 'MID150BEES.NS',
    'Media, Entertainment & Publication': 'MID150BEES.NS',
    'Agriculture'                       : 'MID150BEES.NS',
    'Forest Materials'                  : 'MID150BEES.NS',
}

print(f"        Fetching sector index data...")
ticker_data = {}
for ticker in set(SECTOR_INDEX_MAP.values()):
    try:
        df = yf.download(ticker, period='1y', interval='1d',
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        if len(df) > 50:
            ticker_data[ticker] = df
    except:
        pass

fallback     = 'MID150BEES.NS'
sector_price = {}
for sector, ticker in SECTOR_INDEX_MAP.items():
    if ticker in ticker_data:
        sector_price[sector] = ticker_data[ticker]
    elif fallback in ticker_data:
        sector_price[sector] = ticker_data[fallback]

def calculate_sector_indicators(df):
    data = df.copy()
    data['EMA20']  = data['Close'].ewm(span=20,  adjust=False).mean()
    data['EMA50']  = data['Close'].ewm(span=50,  adjust=False).mean()
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
    delta          = data['Close'].diff()
    gain           = delta.where(delta > 0, 0)
    loss           = -delta.where(delta < 0, 0)
    rs             = gain.ewm(span=14, adjust=False).mean() / (
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
                          (low  - close.shift()).abs()], axis=1).max(axis=1)
    dm_plus  = high.diff()
    dm_minus = -low.diff()
    dm_plus  = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    atr      = tr.ewm(span=14, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(span=14,  adjust=False).mean() / (atr + 1e-9)
    di_minus = 100 * dm_minus.ewm(span=14, adjust=False).mean() / (atr + 1e-9)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    data['ADX'] = dx.ewm(span=14, adjust=False).mean()
    return data

def get_sector_trend(sector):
    if sector not in sector_price:
        return 'No data', '—'
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
    bull_score   = sum([above_ema200, above_ema50, ema_aligned, macd_bull, rsi > 55])
    bear_score   = sum([not above_ema200, not above_ema50, ema_bearish, macd_bear, rsi < 45])
    if   bull_score >= 5 and adx > 25: label = 'Strong Uptrend ↑↑'
    elif bull_score >= 4 and adx > 20: label = 'Uptrend ↑'
    elif bull_score >= 3 and adx <= 20:label = 'Weak Uptrend →↑'
    elif bull_score == bear_score:     label = 'Sideways →'
    elif bear_score >= 5 and adx > 25: label = 'Strong Downtrend ↓↓'
    elif bear_score >= 4 and adx > 20: label = 'Downtrend ↓'
    elif bear_score >= 3 and adx <= 20:label = 'Weak Downtrend →↓'
    else:                              label = 'Sideways →'
    ema_str  = '↑↑↑' if ema_aligned else '↓↓↓' if ema_bearish else 'mixed'
    macd_str = '▲' if macd_bull else '▼' if macd_bear else '~'
    detail   = (f"RSI {rsi} | ADX {adx} | MACD {macd_str} | "
                f"EMA {ema_str} | {'Above' if above_ema200 else 'Below'} EMA200")
    return label, detail

tech_df['Sector Trend']  = ''
tech_df['Sector Detail'] = ''
for idx, row in tech_df.iterrows():
    s_label, s_detail = get_sector_trend(str(row['Sector']))
    tech_df.at[idx, 'Sector Trend']  = s_label
    tech_df.at[idx, 'Sector Detail'] = s_detail

BREAKOUT_FILE = os.path.join(SCORES_DIR, 'breakout_tracker.csv')
today_str     = datetime.now().strftime('%Y-%m-%d')

if os.path.exists(BREAKOUT_FILE):
    bk_df = pd.read_csv(BREAKOUT_FILE)
else:
    bk_df = pd.DataFrame(columns=[
        'Symbol', 'First_Breakout_Date', 'Weeks_Count',
        'Last_Pct', 'Cap_Category', 'Sector', 'Fund_Score'
    ])

current_breakouts = tech_df[
    (tech_df['In Consolidation'] == True) &
    (tech_df['Pct to Breakout'] > 0)
].copy()

updated_rows = []
for _, row in current_breakouts.iterrows():
    sym      = row['Symbol']
    existing = bk_df[bk_df['Symbol'] == sym]
    if len(existing) > 0:
        weeks = int(existing.iloc[0]['Weeks_Count']) + 1
        updated_rows.append({
            'Symbol'            : sym,
            'First_Breakout_Date': existing.iloc[0]['First_Breakout_Date'],
            'Weeks_Count'       : weeks,
            'Last_Pct'          : round(row['Pct to Breakout'], 2),
            'Cap_Category'      : row['Cap Category'],
            'Sector'            : row['Sector'],
            'Fund_Score'        : row.get('Fund Score', 0),
        })
    else:
        updated_rows.append({
            'Symbol'            : sym,
            'First_Breakout_Date': today_str,
            'Weeks_Count'       : 1,
            'Last_Pct'          : round(row['Pct to Breakout'], 2),
            'Cap_Category'      : row['Cap Category'],
            'Sector'            : row['Sector'],
            'Fund_Score'        : row.get('Fund Score', 0),
        })

for _, row in bk_df.iterrows():
    sym           = row['Symbol']
    already_added = any(r['Symbol'] == sym for r in updated_rows)
    if not already_added and int(row['Weeks_Count']) < 3:
        tech_row = tech_df[tech_df['Symbol'] == sym]
        if len(tech_row) > 0:
            pct = float(tech_row.iloc[0]['Pct to Breakout'])
            if pct > -5:
                updated_rows.append({
                    'Symbol'            : sym,
                    'First_Breakout_Date': row['First_Breakout_Date'],
                    'Weeks_Count'       : int(row['Weeks_Count']),
                    'Last_Pct'          : round(pct, 2),
                    'Cap_Category'      : row['Cap_Category'],
                    'Sector'            : row['Sector'],
                    'Fund_Score'        : row['Fund_Score'],
                })

new_bk_df = pd.DataFrame(updated_rows)
new_bk_df.to_csv(BREAKOUT_FILE, index=False)

if len(new_bk_df) > 0:
    tech_df = tech_df.merge(
        new_bk_df[['Symbol', 'Weeks_Count', 'First_Breakout_Date']],
        on='Symbol', how='left'
    )
else:
    tech_df['Weeks_Count']         = None
    tech_df['First_Breakout_Date'] = None

tech_df.to_csv(os.path.join(SCORES_DIR, 'technical_report_full.csv'), index=False)

print(f"        Done — {len(tech_df)} rows, {len(tech_df.columns)} cols")
print(f"        Breakout tracker : {len(new_bk_df)} stocks")
print(f"        Rank jumpers     : {len(rank_jumpers)}")

# ── STEP 12: GENERATE QUARTERLY REPORTS ──────────────────────
print("\n[12/12] Generating quarterly reports...")
print(f"        Started : {datetime.now().strftime('%H:%M:%S')}")

today_file = datetime.now().strftime('%Y%m%d')
CAP_ORDER  = ['Large Cap', 'Mini Large Cap', 'Mid Cap', 'Small Cap']

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
    if 'BUY NOW (Momentum)' in tier:  return 'T1-Mom'
    if 'BUY NOW (Reversal)' in tier:  return 'T1-Rev'
    if 'BREAKOUT IMMINENT'  in tier:  return 'T1-Brk'
    if 'WATCHLIST'          in tier:  return 'T2-Wtch'
    if 'BASE BUILDING'      in tier:  return 'T2-Base'
    return 'T3'

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
    sec = row.get('Sector Score', 0) or 0
    cap = row.get('Cap Score',    0) or 0
    sym = row['Symbol']
    return sec >= 7 or cap >= 7 or sym in rank_jumpers

def get_tech_trend(row):
    try:
        price  = float(row['Current Price'])
        ema20  = float(row['EMA20'])
        ema50  = float(row['EMA50'])
        ema200 = float(row['EMA200'])
        adx    = float(row['ADX'])
        rsi    = float(row['RSI'])
        macd   = float(row['MACD Hist'])
        vol    = float(row['Vol Ratio'])
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

def build_sector_summary(tech_df):
    lines = []
    def p(line=''): lines.append(str(line))
    p(f"{'─'*78}")
    p(f"  SECTOR TREND SUMMARY")
    p(f"{'─'*78}")
    p(f"  {'Sector':<44} {'Trend':<28} {'Detail'}")
    p(f"  {'─'*44} {'─'*28} {'─'*30}")
    sectors_seen = {}
    for _, row in tech_df.iterrows():
        sec = str(row['Sector'])
        if sec not in sectors_seen:
            sectors_seen[sec] = (row['Sector Trend'], row['Sector Detail'])
    for sec in sorted(sectors_seen.keys()):
        trend, detail = sectors_seen[sec]
        p(f"  {sec:<44} {trend:<28} {detail}")
    p()
    return lines

def generate_tech_report(is_short=True):
    fmt   = 'SHORT' if is_short else 'LONG'
    now   = datetime.now().strftime('%d %B %Y')
    lines = []
    sep   = "=" * 78
    def p(line=''): lines.append(str(line))

    p(sep)
    p(f"  AI STOCK SCREENER — QUARTERLY REPORT ({fmt})")
    p(f"  {now}  |  Universe: {len(tech_df)} stocks")
    p(sep)
    p(f"  SecRank  = rank vs same sector  |  CapRank = rank vs same sector + cap")
    p(f"  Week Vol = this week avg vs prior 15-day avg")
    p(f"  Break Vol= this week avg vs consolidation period avg")
    p(f"  ↑ = jumped 10%+ in rank vs last quarter")
    p()

    for line in build_sector_summary(tech_df):
        p(line)

    # ── TIER 1 MOMENTUM ───────────────────────────────────────
    t1_mom = tech_df[tech_df['Tier'] == 'TIER 1 — BUY NOW (Momentum)'].copy()
    t1_mom = t1_mom.sort_values('Fund Score', ascending=False)
    p(f"{'─'*78}")
    p(f"  TIER 1 — MOMENTUM  ({len(t1_mom)} stocks)")
    p(f"  Fund >=60 + Tech >=65 + Momentum setup confirmed")
    p(f"{'─'*78}")
    serial = 1
    for cap in CAP_ORDER:
        cap_grp = t1_mom[t1_mom['Cap Category'] == cap]
        if len(cap_grp) == 0:
            continue
        cap_lbl = {'Large Cap':'L','Mini Large Cap':'ML','Mid Cap':'M','Small Cap':'S'}[cap]
        p(f"\n  [{cap_lbl}] {cap}")
        p(f"  {'─'*66}")
        for _, row in cap_grp.iterrows():
            in_consol   = row.get('In Consolidation', False)
            week_vol    = float(row.get('Vol 5D Ratio', 1.0))
            break_vol   = float(row.get('Breakout Vol', 0.0))
            weeks_count = row.get('Weeks_Count')
            if is_short:
                p(f"\n  #{serial}  {row['Symbol']}  ({row['Sector']})  "
                  f"MCap {mcap_str(row['Market Cap Cr'])}  "
                  f"Rs{row['Current Price']:.2f}")
                p(f"    Setup   : {row['Best Setup']:<12} Tech {row['Tech Score']:>3.0f}/100")
                p(f"    Scores  : Fund {row['Fund Score']:.1f}/100  "
                  f"Sector Rank {row.get('Sector Score',0):.1f}/10  "
                  f"Cap Rank {row.get('Cap Score',0):.1f}/10")
                if in_consol:
                    p(f"    Base    : {row['Consol Label']} | "
                      f"{row['Pct to Breakout']:+.1f}% to breakout")
                    p(f"    Break   : {get_breakout_vol_inference(week_vol, break_vol)}")
                p(f"    Tech    : RSI {row['RSI']:.0f}  ADX {row['ADX']:.0f}  "
                  f"MACD {round(row['MACD Hist']):+d}  → {get_reason(row)}")
                p(f"    Volume  : Week Vol:{week_vol:.2f}x "
                  f"({row['Vol Label']}) → {row['Vol Inference']}")
                p(f"    Sector  : {row['Sector Trend']} | {row['Sector Detail']}")
            else:
                p(f"\n  #{serial}  {row['Symbol']}  ({row['Sector']})  "
                  f"MCap {mcap_str(row['Market Cap Cr'])}  "
                  f"Rs{row['Current Price']:.2f}")
                p(f"    Setup   : {row['Best Setup']:<12} Tech {row['Tech Score']:>3.0f}/100  "
                  f"Tier: {tier_abbr(row['Tier'])}")
                p(f"    Scores  : Fund {row['Fund Score']:.1f}  "
                  f"Mom:{row['Momentum Score']:.0f}  "
                  f"Rev:{row['Reversal Score']:.0f}  "
                  f"Sec:{row.get('Sector Score',0):.1f}  "
                  f"Cap:{row.get('Cap Score',0):.1f}")
                if in_consol:
                    p(f"    Base    : {row['Consol Label']} | "
                      f"{row['Pct to Breakout']:+.1f}% to breakout")
                    p(f"    Break   : {get_breakout_vol_inference(week_vol, break_vol)}")
                p(f"    Tech    : RSI:{row['RSI']:3.0f}  ADX:{row['ADX']:3.0f}  "
                  f"MACD:{round(row['MACD Hist']):+d}  → {get_reason(row)}")
                p(f"    Volume  : Week Vol:{week_vol:.2f}x "
                  f"({row['Vol Label']}) → {row['Vol Inference']}")
                p(f"    Sector  : {row['Sector Trend']} | {row['Sector Detail']}")
                if weeks_count and not pd.isna(weeks_count):
                    p(f"    Breakout: {int(weeks_count)} weeks above resistance | "
                      f"First: {row.get('First_Breakout_Date','—')}")
            serial += 1

    # ── TIER 1 REVERSAL ───────────────────────────────────────
    t1_rev = tech_df[tech_df['Tier'] == 'TIER 1 — BUY NOW (Reversal)'].copy()
    t1_rev = t1_rev.sort_values('Fund Score', ascending=False)
    if len(t1_rev) > 0:
        p(f"\n{'─'*78}")
        p(f"  TIER 1 — REVERSAL  ({len(t1_rev)} stocks)")
        p(f"  Fund >=60 + Tech >=65 + Reversal setup confirmed")
        p(f"{'─'*78}")
        for cap in CAP_ORDER:
            cap_grp = t1_rev[t1_rev['Cap Category'] == cap]
            if len(cap_grp) == 0:
                continue
            cap_lbl = {'Large Cap':'L','Mini Large Cap':'ML','Mid Cap':'M','Small Cap':'S'}[cap]
            p(f"\n  [{cap_lbl}] {cap}")
            p(f"  {'─'*66}")
            for _, row in cap_grp.iterrows():
                week_vol  = float(row.get('Vol 5D Ratio', 1.0))
                break_vol = float(row.get('Breakout Vol', 0.0))
                if is_short:
                    p(f"\n  #{serial}  {row['Symbol']}  ({row['Sector']})  "
                      f"MCap {mcap_str(row['Market Cap Cr'])}  "
                      f"Rs{row['Current Price']:.2f}")
                    p(f"    Setup   : {row['Best Setup']:<12} Tech {row['Tech Score']:>3.0f}/100")
                    p(f"    Scores  : Fund {row['Fund Score']:.1f}/100  "
                      f"Sector Rank {row.get('Sector Score',0):.1f}/10  "
                      f"Cap Rank {row.get('Cap Score',0):.1f}/10")
                    p(f"    Tech    : RSI {row['RSI']:.0f}  ADX {row['ADX']:.0f}  "
                      f"MACD {round(row['MACD Hist']):+d}  → {get_reason(row)}")
                    p(f"    Volume  : Week Vol:{week_vol:.2f}x "
                      f"({row['Vol Label']}) → {row['Vol Inference']}")
                    p(f"    Sector  : {row['Sector Trend']} | {row['Sector Detail']}")
                else:
                    p(f"\n  #{serial}  {row['Symbol']}  ({row['Sector']})  "
                      f"MCap {mcap_str(row['Market Cap Cr'])}  "
                      f"Rs{row['Current Price']:.2f}")
                    p(f"    Setup   : {row['Best Setup']:<12} Tech {row['Tech Score']:>3.0f}/100  "
                      f"Tier: {tier_abbr(row['Tier'])}")
                    p(f"    Scores  : Fund {row['Fund Score']:.1f}  "
                      f"Mom:{row['Momentum Score']:.0f}  "
                      f"Rev:{row['Reversal Score']:.0f}  "
                      f"Sec:{row.get('Sector Score',0):.1f}  "
                      f"Cap:{row.get('Cap Score',0):.1f}")
                    p(f"    Tech    : RSI:{row['RSI']:3.0f}  ADX:{row['ADX']:3.0f}  "
                      f"MACD:{round(row['MACD Hist']):+d}  → {get_reason(row)}")
                    p(f"    Volume  : Week Vol:{week_vol:.2f}x "
                      f"({row['Vol Label']}) → {row['Vol Inference']}")
                    p(f"    Sector  : {row['Sector Trend']} | {row['Sector Detail']}")
                serial += 1

    # ── TIER 1 BREAKOUT IMMINENT ──────────────────────────────
    t1_brk = tech_df[tech_df['Tier'] == 'TIER 1 — BREAKOUT IMMINENT'].copy()
    t1_brk = t1_brk.sort_values('Fund Score', ascending=False)
    if len(t1_brk) > 0:
        p(f"\n{'─'*78}")
        p(f"  TIER 1 — BREAKOUT IMMINENT  ({len(t1_brk)} stocks)")
        p(f"  Fund >=60 + In consolidation + within 5% of breakout")
        p(f"{'─'*78}")
        for _, row in t1_brk.iterrows():
            week_vol  = float(row.get('Vol 5D Ratio', 1.0))
            break_vol = float(row.get('Breakout Vol', 0.0))
            p(f"\n  #{serial}  {row['Symbol']}  ({row['Sector']})  "
              f"MCap {mcap_str(row['Market Cap Cr'])}  "
              f"Rs{row['Current Price']:.2f}")
            p(f"    Scores  : Fund {row['Fund Score']:.1f}/100  "
              f"Sector Rank {row.get('Sector Score',0):.1f}/10  "
              f"Cap Rank {row.get('Cap Score',0):.1f}/10")
            p(f"    Base    : {row['Consol Label']} | "
              f"{row['Pct to Breakout']:+.1f}% to breakout")
            p(f"    Break   : {get_breakout_vol_inference(week_vol, break_vol)}")
            p(f"    Tech    : RSI {row['RSI']:.0f}  ADX {row['ADX']:.0f}  "
              f"MACD {round(row['MACD Hist']):+d}  → {get_reason(row)}")
            p(f"    Sector  : {row['Sector Trend']} | {row['Sector Detail']}")
            serial += 1

    # ── TIER 2A WATCHLIST ─────────────────────────────────────
    t2_watch    = tech_df[tech_df['Tier'] == 'TIER 2 — WATCHLIST'].copy()
    t2_watch    = t2_watch.sort_values('Fund Score', ascending=False)
    t2_filtered = t2_watch[t2_watch.apply(passes_rank_filter, axis=1)]
    p(f"\n{'─'*78}")
    p(f"  TIER 2A — WATCHLIST  (Top {len(t2_filtered)} | Cap/Sec Rank >=7 or ↑)")
    p(f"  Setup forming — wait for confirmation before entering")
    p(f"{'─'*78}")
    for _, row in t2_filtered.iterrows():
        week_vol    = float(row.get('Vol 5D Ratio', 1.0))
        in_consol   = row.get('In Consolidation', False)
        jump_marker = ' ↑' if row['Symbol'] in rank_jumpers else ''
        base_str    = f"  [{int(row['Consol Days'])}d base]" if in_consol else ''
        p(f"\n  #{serial}  {row['Symbol']}  "
          f"Rs{row['Current Price']:.2f}  "
          f"MCap {mcap_str(row['Market Cap Cr'])}  "
          f"Fund:{row['Fund Score']:.1f}  "
          f"Sec:{row.get('Sector Score',0):4.1f}  "
          f"Cap:{row.get('Cap Score',0):4.1f}"
          f"{jump_marker}{base_str}")
        p(f"      Technical : [{row['Best Setup']}|{row['Tech Score']:.0f}]  "
          f"RSI:{row['RSI']:3.0f}  ADX:{row['ADX']:3.0f}  "
          f"MACD:{round(row['MACD Hist']):+d}  → {get_reason(row)}")
        p(f"      Volume    : Week Vol:{week_vol:.2f}x "
          f"({row['Vol Label']}) → {row['Vol Inference']}")
        p(f"      Sector    : {row['Sector Trend']} | {row['Sector Detail']}")
        serial += 1

    # ── TIER 2B BASE BUILDING ─────────────────────────────────
    t2_base      = tech_df[tech_df['Tier'] == 'TIER 2 — BASE BUILDING'].copy()
    t2_base      = t2_base.sort_values('Fund Score', ascending=False)
    t2b_filtered = t2_base[t2_base.apply(passes_rank_filter, axis=1)]
    if len(t2b_filtered) > 0:
        p(f"\n{'─'*78}")
        p(f"  TIER 2B — BASE BUILDING  ({len(t2b_filtered)} stocks | Cap/Sec Rank >=7)")
        p(f"  In consolidation — waiting for breakout trigger")
        p(f"{'─'*78}")
        for _, row in t2b_filtered.iterrows():
            week_vol    = float(row.get('Vol 5D Ratio', 1.0))
            break_vol   = float(row.get('Breakout Vol', 0.0))
            jump_marker = ' ↑' if row['Symbol'] in rank_jumpers else ''
            p(f"\n  #{serial}  {row['Symbol']}  "
              f"Rs{row['Current Price']:.2f}  "
              f"MCap {mcap_str(row['Market Cap Cr'])}  "
              f"Fund:{row['Fund Score']:.1f}  "
              f"Sec:{row.get('Sector Score',0):4.1f}  "
              f"Cap:{row.get('Cap Score',0):4.1f}"
              f"{jump_marker}")
            p(f"      Base    : {row['Consol Label']} | "
              f"{row['Pct to Breakout']:+.1f}% to breakout")
            p(f"      Break   : {get_breakout_vol_inference(week_vol, break_vol)}")
            p(f"      Tech    : RSI:{row['RSI']:3.0f}  ADX:{row['ADX']:3.0f}  "
              f"MACD:{round(row['MACD Hist']):+d}  → {get_reason(row)}")
            p(f"      Sector  : {row['Sector Trend']} | {row['Sector Detail']}")
            serial += 1

    # ── TIER 3 REVERSAL WATCH ─────────────────────────────────
    t3_rev = tech_df[
        (tech_df['Tier'] == 'TIER 3 — WAITING') &
        (tech_df['Best Setup'] == 'Reversal')
    ].copy()
    t3_rev          = t3_rev.sort_values('Reversal Score', ascending=False)
    t3_rev_filtered = t3_rev[t3_rev.apply(passes_rank_filter, axis=1)].head(20)
    if len(t3_rev_filtered) > 0:
        p(f"\n{'─'*78}")
        p(f"  REVERSAL WATCH  (Top {len(t3_rev_filtered)} | Cap/Sec Rank >=7)")
        p(f"  Reversal setup forming — not yet Tier 1/2 quality")
        p(f"{'─'*78}")
        for _, row in t3_rev_filtered.iterrows():
            week_vol    = float(row.get('Vol 5D Ratio', 1.0))
            jump_marker = ' ↑' if row['Symbol'] in rank_jumpers else ''
            tier_tag    = f"[{tier_abbr(row['Tier'])}]"
            p(f"\n  #{serial}  {row['Symbol']}  "
              f"Rs{row['Current Price']:.2f}  "
              f"MCap {mcap_str(row['Market Cap Cr'])}  "
              f"{tier_tag}  "
              f"Fund:{row['Fund Score']:.1f}  "
              f"Rev:{row['Reversal Score']:.0f}  "
              f"Sec:{row.get('Sector Score',0):4.1f}  "
              f"Cap:{row.get('Cap Score',0):4.1f}"
              f"{jump_marker}")
            p(f"      Tech    : RSI:{row['RSI']:3.0f}  ADX:{row['ADX']:3.0f}  "
              f"MACD:{round(row['MACD Hist']):+d}  → {get_reason(row)}")
            p(f"      Volume  : Week Vol:{week_vol:.2f}x "
              f"({row['Vol Label']}) → {row['Vol Inference']}")
            p(f"      Sector  : {row['Sector Trend']} | {row['Sector Detail']}")
            serial += 1

    # ── RANK JUMPERS ──────────────────────────────────────────
    if rank_jumpers:
        p(f"\n{'─'*78}")
        p(f"  ↑ RANK JUMPERS THIS QUARTER  ({len(rank_jumpers)} stocks)")
        p(f"{'─'*78}")
        jumper_df = tech_df[tech_df['Symbol'].isin(rank_jumpers)].sort_values(
            'Fund Score', ascending=False)
        for _, row in jumper_df.iterrows():
            p(f"  {row['Symbol']:12}  Rs{row['Current Price']:.2f}  "
              f"MCap {mcap_str(row['Market Cap Cr'])}  "
              f"Fund:{row['Fund Score']:4.1f}  "
              f"Sec:{row.get('Sector Score',0):4.1f}  "
              f"Cap:{row.get('Cap Score',0):4.1f}  "
              f"{tier_abbr(row['Tier'])}  {row['Sector']}")

    p(f"\n{'─'*78}")
    p(f"  L=Large Cap  ML=Mini Large Cap  M=Mid Cap  S=Small Cap")
    p(f"  SecRank  = rank vs same sector | CapRank = rank vs same sector + cap")
    p(f"  Week Vol = this week avg vs prior 15-day avg")
    p(f"  Break Vol= this week avg vs consolidation period avg")
    p(f"  ↑ = jumped 10%+ in Cap or Sector rank vs last quarter")
    p(f"{'─'*78}")
    return lines

# ── ML REPORT ─────────────────────────────────────────────────
tech_for_ml = pd.read_csv(os.path.join(SCORES_DIR, 'technical_report_full.csv'))

CONF_THRESHOLD = {
    'Bullish Continual': 40.0,
    'Bullish'          : 50.0,
    'Reversal'         : 50.0,
}
buy_labels = ['Bullish Continual', 'Bullish', 'Reversal']

filtered = tech_for_ml[
    tech_for_ml['ML_Prediction'].isin(buy_labels) &
    tech_for_ml['ML_Confidence'].notna()
].copy()
filtered = filtered[
    filtered.apply(
        lambda r: r['ML_Confidence'] >= CONF_THRESHOLD.get(r['ML_Prediction'], 50),
        axis=1)
].copy()
filtered = filtered.sort_values('ML_Confidence', ascending=False).reset_index(drop=True)

def build_ml_report(df, fmt='long'):
    now   = datetime.now().strftime('%Y-%m-%d %H:%M')
    lines = []
    sep   = "=" * 74
    def p(line=''): lines.append(str(line))
    p(sep)
    p(f"  AI STOCK SCREENER — QUARTERLY ML REPORT ({fmt.upper()})")
    p(f"  Generated : {now}")
    p(f"  Universe  : {len(tech_for_ml)} stocks | Buy signals: {len(df)}")
    p(sep)
    for label in buy_labels:
        section = df[df['ML_Prediction'] == label].copy()
        if len(section) == 0:
            continue
        p(f"\n{'─' * 74}")
        p(f"  {label.upper()}  ({len(section)} stocks)")
        if label == 'Bullish Continual':
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
                p(f"  {i:>3}  {row['Symbol']:<14} "
                  f"{mcap_str(row['Market Cap Cr']):>12}  "
                  f"{row.get('Sector Score', 0):>6.1f}  "
                  f"{row.get('Cap Score', 0):>6.1f}  "
                  f"{row['ML_Confidence']:>5.1f}  "
                  f"{row['Forecast_25d_Pct']:>+5.1f}  "
                  f"{row['Forecast_45d_Pct']:>+5.1f}  "
                  f"{get_tech_trend(row):<6}  "
                  f"{tier_abbr(str(row.get('Tier', '')))}")
        else:
            for i, (_, row) in enumerate(section.iterrows(), 1):
                p(f"\n  {i}. {row['Symbol']}  |  {row['Sector']}  |  "
                  f"{row['Cap Category']}  |  {mcap_str(row['Market Cap Cr'])}")
                p(f"     Confidence : {row['ML_Confidence']:.1f}%  |  "
                  f"Trend: {get_tech_trend(row)}  |  "
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

# ── SAVE ALL 4 REPORTS ────────────────────────────────────────
tech_short = generate_tech_report(is_short=True)
tech_long  = generate_tech_report(is_short=False)
ml_short   = build_ml_report(filtered, fmt='short')
ml_long    = build_ml_report(filtered, fmt='long')

with open(os.path.join(REPORTS_TECH, f'quarterly_report_short_{today_file}.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(tech_short))
with open(os.path.join(REPORTS_TECH, f'quarterly_report_long_{today_file}.txt'),  'w', encoding='utf-8') as f:
    f.write('\n'.join(tech_long))
with open(os.path.join(REPORTS_ML,   f'quarterly_ml_report_short_{today_file}.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(ml_short))
with open(os.path.join(REPORTS_ML,   f'quarterly_ml_report_long_{today_file}.txt'),  'w', encoding='utf-8') as f:
    f.write('\n'.join(ml_long))

print(f"     Reports saved:")
print(f"       {REPORTS_TECH}/quarterly_report_short_{today_file}.txt")
print(f"       {REPORTS_TECH}/quarterly_report_long_{today_file}.txt")
print(f"       {REPORTS_ML}/quarterly_ml_report_short_{today_file}.txt")
print(f"       {REPORTS_ML}/quarterly_ml_report_long_{today_file}.txt")
print()
print("=" * 60)
print("  Quarterly run complete!")
print(f"  Finished : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)
