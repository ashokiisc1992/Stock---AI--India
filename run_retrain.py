# ============================================================
# run_retrain.py — AI Stock Screener (Indian Markets)
# Day 12: ML Model Retraining
#
# Run AFTER quarterly run, once you're satisfied with the
# updated universe:
#   python run_retrain.py
#
# What this does:
#   Step 1 — Load price, indicator, fundamental, tech data
#   Step 2 — Assign stocks to 6 groups
#   Step 3 — Build training features + labels
#   Step 4 — Train Model 1: Reversal classifier (bottom + top)
#   Step 5 — Train Model 2: Trend classifier (3-class)
#   Step 6 — Train Model 3: Price forecast regressor (25d/45d/180d)
#   Step 7 — Train Model 4: Continuation classifier (bullish + bearish)
#   Step 8 — Save all models + group mappings
#
# Estimated run time: ~1.5–2 hours
#
# SAFETY: Backs up existing models before overwriting.
# If retraining fails mid-way, existing models are preserved.
# ============================================================

import pandas as pd
import numpy as np
import pickle
import warnings
import os
import shutil
from collections import defaultdict
from datetime import datetime

from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

warnings.filterwarnings('ignore')

# ── PATHS ─────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, 'data')
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
MODELS_BAK   = os.path.join(BASE_DIR, 'models_backup')
UNIVERSE_DIR = os.path.join(DATA_DIR, 'universe')
FUND_DIR     = os.path.join(DATA_DIR, 'fundamentals')
PRICES_DIR   = os.path.join(DATA_DIR, 'prices')
SCORES_DIR   = os.path.join(DATA_DIR, 'scores')

os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("  AI Stock Screener — ML Retraining")
print(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)
print("  Trains all 4 models on updated universe")
print("  Backs up existing models before overwriting")
print("=" * 60)

# ── SAFETY: BACKUP EXISTING MODELS ────────────────────────────
print("\n[0/8] Backing up existing models...")
if os.path.exists(MODELS_BAK):
    shutil.rmtree(MODELS_BAK)
shutil.copytree(MODELS_DIR, MODELS_BAK)
print(f"     ✅ Backup saved to: models_backup/")
print(f"        Files backed up: {len(os.listdir(MODELS_BAK))}")

# ── STEP 1: LOAD DATA ─────────────────────────────────────────
print("\n[1/8] Loading data...")

with open(os.path.join(PRICES_DIR, 'price_data_full.pkl'), 'rb') as f:
    price_data = pickle.load(f)

with open(os.path.join(PRICES_DIR, 'indicator_data_full.pkl'), 'rb') as f:
    indicator_data = pickle.load(f)

fund_scores = pd.read_csv(os.path.join(FUND_DIR,    'fundamental_scores_full.csv'))
tech_report = pd.read_csv(os.path.join(SCORES_DIR,  'technical_report_full.csv'))
prefilt_df  = pd.read_csv(os.path.join(UNIVERSE_DIR,'prefilt_passed.csv'))

# Merge market cap into fund_scores if not present
if 'Market_Cap_Cr' not in fund_scores.columns:
    fund_scores = fund_scores.merge(
        prefilt_df[['Symbol', 'Market_Cap_Cr']], on='Symbol', how='left'
    )

# Build lookup: symbol → {Sector, Cap Category, Market Cap Cr, Final Score}
fund_lookup = tech_report.set_index('Symbol')[
    ['Sector', 'Cap Category', 'Market Cap Cr']
].to_dict('index')

for sym, row in fund_scores.set_index('Symbol').iterrows():
    if sym in fund_lookup:
        fund_lookup[sym]['Final Score'] = row.get('Final Score', 50)

print(f"     Price data     : {len(price_data)} stocks")
print(f"     Indicator data : {len(indicator_data)} stocks")
print(f"     Fund scores    : {len(fund_scores)} stocks")
print(f"     Tech report    : {len(tech_report)} stocks")
print(f"     Fund lookup    : {len(fund_lookup)} stocks")
print(f"\n     Sectors in universe:")
print(tech_report['Sector'].value_counts().to_string())

# ── STEP 2: GROUP ASSIGNMENT ──────────────────────────────────
print("\n[2/8] Assigning stocks to groups...")

SECTOR_GROUP_MAP = {
    'Information Technology'            : 'IT',
    'Financial Services'                : 'Financial',
    'Chemicals'                         : 'Chemicals',
    'Healthcare'                        : 'Healthcare',
    'Consumer Durables'                 : 'Consumer',
    'Consumer Services'                 : 'Consumer',
    'Fast Moving Consumer Goods'        : 'Consumer',
    'Capital Goods'                     : 'Industrial',
    'Automobile and Auto Components'    : 'Industrial',
    'Construction'                      : 'Industrial',
    'Construction Materials'            : 'Industrial',
    'Textiles'                          : 'Industrial',
    'Services'                          : 'Industrial',
    'Metals & Mining'                   : 'Industrial',
    'Oil, Gas & Consumable Fuels'       : 'Industrial',
    'Power'                             : 'Industrial',
    'Realty'                            : 'Industrial',
    'Utilities'                         : 'Industrial',
    'Telecommunication'                 : 'Industrial',
    'Media, Entertainment & Publication': 'Industrial',
    'Media Entertainment & Publication' : 'Industrial',
    'Forest Materials'                  : 'Industrial',
    'Diversified'                       : 'Industrial',
    'Pharmaceuticals'                   : 'Healthcare',
    'Banking'                           : 'Financial',
}

symbol_group = {}
unmapped     = []
for sym, info in fund_lookup.items():
    sector = info.get('Sector', '')
    grp    = SECTOR_GROUP_MAP.get(sector)
    if grp is None:
        unmapped.append((sym, sector))
        grp = 'Industrial'
    symbol_group[sym] = grp

group_stocks = defaultdict(list)
for sym, grp in symbol_group.items():
    group_stocks[grp].append(sym)

print(f"     Group assignments:")
for grp, stocks in sorted(group_stocks.items()):
    print(f"       {grp:12} : {len(stocks):3d} stocks")
print(f"\n     Total mapped : {len(symbol_group)} stocks")
if unmapped:
    print(f"     Unmapped (→ Industrial): {len(unmapped)}")
    for sym, sec in unmapped[:10]:
        print(f"       {sym}: '{sec}'")

# ── STEP 3: BUILD FEATURES + LABELS ──────────────────────────
print("\n[3/8] Building features and labels...")
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

TARGET_BOTTOM = 'bottom_reversal'
TARGET_TOP    = 'top_reversal'
TARGET_TREND  = 'trend_label'
TARGET_25D    = 'target_25d'
TARGET_45D    = 'target_45d'
TARGET_180D   = 'target_180d'

def build_features(symbol, price_df, ind_df):
    df = price_df.copy()

    # Drop overlapping columns before joining
    overlap_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'Dividends', 'Stock Splits']
    ind_clean = ind_df.drop(
        columns=[c for c in overlap_cols if c in ind_df.columns]
    )
    df = df.join(ind_clean, how='left')

    # ── PRICE FEATURES ────────────────────────────────────────
    df['return_1d']  = df['Close'].pct_change(1)
    df['return_5d']  = df['Close'].pct_change(5)
    df['return_20d'] = df['Close'].pct_change(20)
    df['return_60d'] = df['Close'].pct_change(60)

    df['high_52w']      = df['High'].rolling(252).max()
    df['low_52w']       = df['Low'].rolling(252).min()
    df['dist_52w_high'] = (df['Close'] - df['high_52w']) / df['high_52w']
    df['dist_52w_low']  = (df['Close'] - df['low_52w'])  / df['low_52w']

    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low']  - df['Close'].shift(1))
        )
    )
    df['atr_14']         = df['tr'].rolling(14).mean()
    df['atr_pct']        = df['atr_14'] / df['Close']
    df['volatility_20d'] = df['return_1d'].rolling(20).std()
    df['volatility_60d'] = df['return_1d'].rolling(60).std()

    # ── VOLUME FEATURES ───────────────────────────────────────
    df['vol_ma20']      = df['Volume'].rolling(20).mean()
    df['vol_ratio_5d']  = df['Volume'].rolling(5).mean()  / df['vol_ma20']
    df['vol_ratio_20d'] = df['Volume'].rolling(20).mean() / df['Volume'].rolling(60).mean()
    df['obv']           = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['obv_slope_10d'] = df['obv'].diff(10) / (df['obv'].abs().rolling(10).mean() + 1e-9)
    df['vol_spike']     = (df['Volume'] > df['vol_ma20'] * 2).astype(int)

    # ── RSI FEATURES ──────────────────────────────────────────
    if 'RSI' in df.columns:
        df['rsi']            = df['RSI']
        df['rsi_slope_5d']   = df['RSI'].diff(5)
        df['rsi_oversold']   = (df['RSI'] < 30).astype(int)
        df['rsi_overbought'] = (df['RSI'] > 70).astype(int)

    # ── MACD FEATURES ─────────────────────────────────────────
    macd_col = 'MACD_Hist' if 'MACD_Hist' in df.columns else 'MACD Hist'
    if macd_col in df.columns:
        df['macd_hist']     = df[macd_col]
        df['macd_slope_3d'] = df[macd_col].diff(3)
        df['macd_slope_5d'] = df[macd_col].diff(5)
        df['macd_cross']    = (
            (df[macd_col] > 0) & (df[macd_col].shift(1) <= 0)
        ).astype(int)

    # ── ADX FEATURES ──────────────────────────────────────────
    if 'ADX' in df.columns:
        df['adx']       = df['ADX']
        df['adx_slope'] = df['ADX'].diff(5)
        if 'DI_Plus' in df.columns and 'DI_Minus' in df.columns:
            df['di_spread'] = df['DI_Plus'] - df['DI_Minus']
        elif 'DI+' in df.columns and 'DI-' in df.columns:
            df['di_spread'] = df['DI+'] - df['DI-']
        else:
            df['di_spread'] = 0

    # ── EMA FEATURES ──────────────────────────────────────────
    if 'EMA20' in df.columns:
        df['price_vs_ema20']  = (df['Close'] - df['EMA20'])  / df['EMA20']
        df['price_vs_ema50']  = (df['Close'] - df['EMA50'])  / df['EMA50']
        df['price_vs_ema200'] = (df['Close'] - df['EMA200']) / df['EMA200']
        df['ema20_vs_ema50']  = (df['EMA20'] - df['EMA50'])  / df['EMA50']
        df['ema50_vs_ema200'] = (df['EMA50'] - df['EMA200']) / df['EMA200']

    # ── TARGET VARIABLES ──────────────────────────────────────
    df['future_return_20d'] = df['Close'].shift(-20) / df['Close'] - 1
    df['future_return_5d']  = df['Close'].shift(-5)  / df['Close'] - 1

    df['bottom_reversal'] = (
        (df['return_20d'] < -0.05) &
        (df['future_return_20d'] > 0.08)
    ).astype(int)

    df['top_reversal'] = (
        (df['return_20d'] > 0.05) &
        (df['future_return_20d'] < -0.08)
    ).astype(int)

    # Trend label (3-class)
    def assign_trend(row):
        f20 = row.get('future_return_20d', 0)
        if pd.isna(f20):    return 'Sideways'
        if f20 > 0.05:      return 'Uptrend'
        elif f20 < -0.05:   return 'Downtrend'
        else:               return 'Sideways'
    df['trend_label'] = df.apply(assign_trend, axis=1)

    # Forecast targets
    df['target_25d']  = df['Close'].shift(-25)  / df['Close'] - 1
    df['target_45d']  = df['Close'].shift(-45)  / df['Close'] - 1
    df['target_180d'] = df['Close'].shift(-180) / df['Close'] - 1

    # ── CONTINUATION LABELS ───────────────────────────────────
    required = ['Close', 'EMA200', 'EMA20', 'EMA50', 'return_60d', 'future_return_20d']
    if all(c in df.columns for c in required):
        price_above_ema200 = df['Close'] > df['EMA200']
        ema_bullish_align  = (df['EMA20'] > df['EMA50']) & (df['EMA50'] > df['EMA200'])
        ema_bearish_align  = (df['EMA20'] < df['EMA50']) & (df['EMA50'] < df['EMA200'])
        strong_uptrend_60d = df['return_60d'] > 0.05
        strong_dntrend_60d = df['return_60d'] < -0.05
        future_up          = df['future_return_20d'] > 0.03
        future_down        = df['future_return_20d'] < -0.03

        df['bullish_cont'] = (
            price_above_ema200 & ema_bullish_align &
            strong_uptrend_60d & future_up
        ).astype(int)

        df['bearish_cont'] = (
            ~price_above_ema200 & ema_bearish_align &
            strong_dntrend_60d  & future_down
        ).astype(int)
    else:
        df['bullish_cont'] = 0
        df['bearish_cont'] = 0

    return df

# Build features for all stocks
all_features = {}
failed_feat  = []
for symbol in price_data.keys():
    price_df = price_data[symbol]
    ind_df   = indicator_data.get(symbol, pd.DataFrame())
    try:
        all_features[symbol] = build_features(symbol, price_df, ind_df)
    except Exception as e:
        failed_feat.append(symbol)

print(f"     Features built : {len(all_features)} stocks")
if failed_feat:
    print(f"     Failed         : {len(failed_feat)} — {failed_feat[:10]}")

# Verify di_spread
sample    = list(all_features.keys())[0]
df_s      = all_features[sample]
di_ok     = 'di_spread' in df_s.columns and (df_s['di_spread'] != 0).sum() > 0
print(f"     di_spread OK   : {di_ok}")
print(f"     FEATURE_COLS   : {len(FEATURE_COLS)} features")

# ── WALK-FORWARD SETUP ────────────────────────────────────────
WF_FOLDS = [
    ('2015-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
    ('2015-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
    ('2015-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    ('2015-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
    ('2015-01-01', '2024-12-31', '2025-01-01', '2025-12-31'),
]

def prepare_group_data(group_name, target_col):
    group_syms = group_stocks[group_name]
    frames     = []
    for sym in group_syms:
        if sym not in all_features:
            continue
        df = all_features[sym].copy()
        df['symbol_id'] = sym
        keep = FEATURE_COLS + [target_col, 'symbol_id']
        df   = df[keep].copy()
        df   = df.replace([np.inf, -np.inf], np.nan)
        df   = df.dropna()
        df.index = all_features[sym].dropna(
            subset=FEATURE_COLS + [target_col]
        ).index
        frames.append(df)
    if not frames:
        return None
    combined           = pd.concat(frames).sort_index()
    le                 = LabelEncoder()
    combined['symbol_enc'] = le.fit_transform(combined['symbol_id'])
    return combined, le

def walk_forward_classify(group_name, target_col):
    """Binary classifier — reversal and continuation."""
    result = prepare_group_data(group_name, target_col)
    if result is None:
        return None, [], None
    combined, le  = result
    feat_cols     = FEATURE_COLS + ['symbol_enc']
    fold_metrics  = []

    for tr_start, tr_end, te_start, te_end in WF_FOLDS:
        train_df = combined[tr_start:tr_end]
        test_df  = combined[te_start:te_end]
        if len(train_df) < 100 or len(test_df) < 10:
            continue
        if train_df[target_col].nunique() < 2:
            continue
        X_train   = train_df[feat_cols]
        y_train   = train_df[target_col]
        X_test    = test_df[feat_cols]
        y_test    = test_df[target_col]
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos = neg_count / pos_count if pos_count > 0 else 1

        model = XGBClassifier(
            n_estimators     = 200,
            max_depth        = 4,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            scale_pos_weight = scale_pos,
            use_label_encoder= False,
            eval_metric      = 'aucpr',
            random_state     = 42,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
        preds = model.predict(X_test)
        fold_metrics.append({
            'fold'      : te_start[:4],
            'precision' : round(precision_score(y_test, preds, zero_division=0), 3),
            'recall'    : round(recall_score(y_test, preds,    zero_division=0), 3),
            'f1'        : round(f1_score(y_test, preds,        zero_division=0), 3),
            'pos_events': int(y_test.sum()),
            'test_rows' : len(y_test),
        })

    # Final model on ALL data
    neg_all   = (combined[target_col] == 0).sum()
    pos_all   = (combined[target_col] == 1).sum()
    scale_all = neg_all / pos_all if pos_all > 0 else 1
    final_model = XGBClassifier(
        n_estimators     = 200,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = scale_all,
        use_label_encoder= False,
        eval_metric      = 'aucpr',
        random_state     = 42,
    )
    final_model.fit(combined[feat_cols], combined[target_col], verbose=False)
    return final_model, fold_metrics, le

def walk_forward_trend(group_name):
    """3-class trend classifier."""
    result = prepare_group_data(group_name, TARGET_TREND)
    if result is None:
        return None, [], None, None
    combined, le_sym  = result
    feat_cols         = FEATURE_COLS + ['symbol_enc']
    le_trend          = LabelEncoder()
    combined['trend_enc'] = le_trend.fit_transform(combined[TARGET_TREND])
    fold_metrics      = []

    for tr_start, tr_end, te_start, te_end in WF_FOLDS:
        train_df = combined[tr_start:tr_end]
        test_df  = combined[te_start:te_end]
        if len(train_df) < 100 or len(test_df) < 10:
            continue
        if train_df['trend_enc'].nunique() < 2:
            continue
        X_train = train_df[feat_cols]
        y_train = train_df['trend_enc']
        X_test  = test_df[feat_cols]
        y_test  = test_df['trend_enc']

        model = XGBClassifier(
            n_estimators     = 300,
            max_depth        = 5,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            use_label_encoder= False,
            eval_metric      = 'mlogloss',
            random_state     = 42,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
        preds = model.predict(X_test)
        fold_metrics.append({
            'fold'     : te_start[:4],
            'accuracy' : round(accuracy_score(y_test, preds), 3),
            'test_rows': len(y_test),
        })

    # Final model on ALL data
    final_model = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        use_label_encoder= False,
        eval_metric      = 'mlogloss',
        random_state     = 42,
    )
    final_model.fit(combined[feat_cols], combined['trend_enc'], verbose=False)
    return final_model, fold_metrics, le_sym, le_trend

def walk_forward_regress(group_name, target_col):
    """Forecast regressor."""
    result = prepare_group_data(group_name, target_col)
    if result is None:
        return None, [], None
    combined, le  = result
    feat_cols     = FEATURE_COLS + ['symbol_enc']
    fold_metrics  = []

    for tr_start, tr_end, te_start, te_end in WF_FOLDS:
        train_df = combined[tr_start:tr_end]
        test_df  = combined[te_start:te_end]
        if len(train_df) < 100 or len(test_df) < 10:
            continue
        X_train = train_df[feat_cols]
        y_train = train_df[target_col]
        X_test  = test_df[feat_cols]
        y_test  = test_df[target_col]

        model = XGBRegressor(
            n_estimators     = 200,
            max_depth        = 4,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            random_state     = 42,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
        preds   = model.predict(X_test)
        dir_acc = np.mean(np.sign(preds) == np.sign(y_test))
        mae     = np.mean(np.abs(preds - y_test))
        fold_metrics.append({
            'fold'     : te_start[:4],
            'dir_acc'  : round(dir_acc, 3),
            'mae'      : round(mae, 3),
            'test_rows': len(y_test),
        })

    # Final model on ALL data
    final_model = XGBRegressor(
        n_estimators     = 200,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = 42,
    )
    final_model.fit(combined[feat_cols], combined[target_col], verbose=False)
    return final_model, fold_metrics, le

# ── STEP 4: MODEL 1 — REVERSAL CLASSIFIER ────────────────────
print("\n[4/8] Training Model 1 — Reversal Classifier...")
print(f"       Started : {datetime.now().strftime('%H:%M:%S')}")
print("=" * 60)

bottom_models   = {}
top_models      = {}
bottom_encoders = {}
top_encoders    = {}

for group in sorted(group_stocks.keys()):
    print(f"\n── {group} ({len(group_stocks[group])} stocks) ──")

    model_b, metrics_b, le_b = walk_forward_classify(group, TARGET_BOTTOM)
    if model_b:
        bottom_models[group]   = model_b
        bottom_encoders[group] = le_b
        print(f"  Bottom reversal:")
        for m in metrics_b:
            print(f"    {m['fold']}: prec={m['precision']} | "
                  f"rec={m['recall']} | f1={m['f1']} | "
                  f"events={m['pos_events']}/{m['test_rows']}")

    model_t, metrics_t, le_t = walk_forward_classify(group, TARGET_TOP)
    if model_t:
        top_models[group]   = model_t
        top_encoders[group] = le_t
        print(f"  Top reversal:")
        for m in metrics_t:
            print(f"    {m['fold']}: prec={m['precision']} | "
                  f"rec={m['recall']} | f1={m['f1']} | "
                  f"events={m['pos_events']}/{m['test_rows']}")

print(f"\n✅ Model 1 trained for {len(bottom_models)} groups")

# ── STEP 5: MODEL 2 — TREND CLASSIFIER ───────────────────────
print("\n[5/8] Training Model 2 — Trend Classifier (3-class)...")
print(f"       Started : {datetime.now().strftime('%H:%M:%S')}")
print("=" * 60)
print("Classes: Uptrend / Sideways / Downtrend")
print("Random baseline: 33.3%  |  Target: >45%")

trend_models         = {}
trend_encoders       = {}
trend_label_encoders = {}

for group in sorted(group_stocks.keys()):
    print(f"\n── {group} ({len(group_stocks[group])} stocks) ──")

    model_t, metrics_t, le_sym, le_trend = walk_forward_trend(group)
    if model_t:
        trend_models[group]         = model_t
        trend_encoders[group]       = le_sym
        trend_label_encoders[group] = le_trend
        print(f"  Classes: {list(le_trend.classes_)}")
        for m in metrics_t:
            flag = '✅' if m['accuracy'] > 0.45 else '⚠️ '
            print(f"  {flag} {m['fold']}: acc={m['accuracy']} | test={m['test_rows']} rows")

print(f"\n✅ Model 2 trained for {len(trend_models)} groups")

# ── STEP 6: MODEL 3 — PRICE FORECAST REGRESSOR ───────────────
print("\n[6/8] Training Model 3 — Price Forecast Regressor...")
print(f"       Started : {datetime.now().strftime('%H:%M:%S')}")
print("=" * 60)
print("Horizons: 25d / 45d / 180d per group")
print("Metric: Directional accuracy (sign match)")
print("Baseline: 50%")

forecast_models   = {}
forecast_encoders = {}

for group in sorted(group_stocks.keys()):
    print(f"\n── {group} ({len(group_stocks[group])} stocks) ──")
    forecast_models[group]   = {}
    forecast_encoders[group] = {}

    for horizon, target_col in [
        ('25d',  TARGET_25D),
        ('45d',  TARGET_45D),
        ('180d', TARGET_180D),
    ]:
        model_f, metrics_f, le_f = walk_forward_regress(group, target_col)
        if model_f:
            forecast_models[group][horizon]   = model_f
            forecast_encoders[group][horizon] = le_f
            dir_accs = [m['dir_acc'] for m in metrics_f]
            avg_dir  = round(np.mean(dir_accs), 3) if dir_accs else 0
            flag     = '✅' if avg_dir > 0.55 else '⚠️ '
            print(f"  {flag} {horizon}: avg dir_acc={avg_dir} | folds={len(metrics_f)}")

print(f"\n✅ Model 3 trained for {len(forecast_models)} groups")

# ── STEP 7: MODEL 4 — CONTINUATION CLASSIFIER ────────────────
print("\n[7/8] Training Model 4 — Continuation Classifier...")
print(f"       Started : {datetime.now().strftime('%H:%M:%S')}")
print("=" * 60)
print("Bullish Cont: in uptrend + predicted to continue up")
print("Bearish Cont: in downtrend + predicted to continue down")

bullish_cont_models   = {}
bearish_cont_models   = {}
bullish_cont_encoders = {}
bearish_cont_encoders = {}

for group in sorted(group_stocks.keys()):
    print(f"\n── {group} ({len(group_stocks[group])} stocks) ──")

    model_bc, metrics_bc, le_bc = walk_forward_classify(group, 'bullish_cont')
    if model_bc:
        bullish_cont_models[group]   = model_bc
        bullish_cont_encoders[group] = le_bc
        print(f"  Bullish continuation:")
        for m in metrics_bc:
            print(f"    {m['fold']}: prec={m['precision']} | "
                  f"rec={m['recall']} | f1={m['f1']} | "
                  f"events={m['pos_events']}/{m['test_rows']}")

    model_dc, metrics_dc, le_dc = walk_forward_classify(group, 'bearish_cont')
    if model_dc:
        bearish_cont_models[group]   = model_dc
        bearish_cont_encoders[group] = le_dc
        print(f"  Bearish continuation:")
        for m in metrics_dc:
            print(f"    {m['fold']}: prec={m['precision']} | "
                  f"rec={m['recall']} | f1={m['f1']} | "
                  f"events={m['pos_events']}/{m['test_rows']}")

print(f"\n✅ Model 4 trained for {len(bullish_cont_models)} groups")

# ── STEP 8: SAVE ALL MODELS ───────────────────────────────────
print("\n[8/8] Saving models...")

def save_pkl(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

save_pkl(bottom_models,         os.path.join(MODELS_DIR, 'bottom_models.pkl'))
save_pkl(bottom_encoders,       os.path.join(MODELS_DIR, 'bottom_encoders.pkl'))
save_pkl(top_models,            os.path.join(MODELS_DIR, 'top_models.pkl'))
save_pkl(top_encoders,          os.path.join(MODELS_DIR, 'top_encoders.pkl'))
save_pkl(trend_models,          os.path.join(MODELS_DIR, 'trend_models.pkl'))
save_pkl(trend_encoders,        os.path.join(MODELS_DIR, 'trend_encoders.pkl'))
save_pkl(trend_label_encoders,  os.path.join(MODELS_DIR, 'trend_label_encoders.pkl'))
save_pkl(forecast_models,       os.path.join(MODELS_DIR, 'forecast_models.pkl'))
save_pkl(forecast_encoders,     os.path.join(MODELS_DIR, 'forecast_encoders.pkl'))
save_pkl(bullish_cont_models,   os.path.join(MODELS_DIR, 'bullish_cont_models.pkl'))
save_pkl(bullish_cont_encoders, os.path.join(MODELS_DIR, 'bullish_cont_encoders.pkl'))
save_pkl(bearish_cont_models,   os.path.join(MODELS_DIR, 'bearish_cont_models.pkl'))
save_pkl(bearish_cont_encoders, os.path.join(MODELS_DIR, 'bearish_cont_encoders.pkl'))
save_pkl(symbol_group,          os.path.join(MODELS_DIR, 'symbol_group.pkl'))
save_pkl(dict(group_stocks),    os.path.join(MODELS_DIR, 'group_stocks.pkl'))

print(f"\n✅ Models saved to models/")
saved = os.listdir(MODELS_DIR)
for fname in sorted(saved):
    size = os.path.getsize(os.path.join(MODELS_DIR, fname)) / 1024
    print(f"   {fname:45} {size:6.1f} KB")

print(f"\n   Backup preserved at: models_backup/")
print(f"   To restore backup  : delete models/ and rename models_backup/ to models/")

print()
print("=" * 60)
print("  Retraining complete!")
print(f"  Finished : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)
print()
print("  Next steps:")
print("  1. Review fold metrics above — check for degradation")
print("  2. Run python run_weekly.py to generate a fresh report")
print("  3. Compare new report vs last week's to validate models")
print("  4. If models look worse, restore backup:")
print("     → Delete models/ folder")
print("     → Rename models_backup/ to models/")
