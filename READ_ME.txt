============================================================
  AI STOCK SCREENER — INDIAN MARKETS
  Complete Reference Guide
  Last updated: April 2026
============================================================

This file explains every script, folder, when to run each,
what it does, and where it saves output. Read this if you
haven't touched the project in a while.

============================================================
  PROJECT LOCATION
============================================================

  E:\learn\Project 1 AI Screener\stock-ai-india\

  GitHub : https://github.com/ashokiisc1992/Stock---AI--India
  Python : 3.12.3 (Anaconda)
  IDE    : JetBrains PyCharm / Jupyter Notebook

  Always open Anaconda Prompt and navigate here first:
    cd E:\learn\Project 1 AI Screener\stock-ai-india


============================================================
  FOLDER STRUCTURE
============================================================

stock-ai-india\
│
├── data\
│   ├── universe\              ← Stock lists and filters
│   │   ├── master_stocks.csv      All NSE+BSE symbols
│   │   ├── bse_stocks.csv         BSE symbol→code map
│   │   ├── prefilt_passed.csv     Pre-filtered universe (~2000+ stocks)
│   │   ├── quality_passed.csv     Quality filtered universe (752 stocks)
│   │   └── quality_filter_results.csv  Why each stock passed/failed
│   │
│   ├── fundamentals\          ← Screener.in scraped data
│   │   ├── raw_stock_data_full.pkl     Raw scraped data (large file)
│   │   ├── fundamental_metrics_full.csv  All computed metrics
│   │   └── fundamental_scores_full.csv   Final scores (0-100) per stock
│   │
│   ├── prices\                ← Price + indicator data
│   │   ├── price_data_full.pkl         OHLCV data for all 752 stocks
│   │   └── indicator_data_full.pkl     EMA, RSI, MACD, ADX etc.
│   │
│   ├── scores\                ← Final scored output
│   │   ├── technical_report_full.csv   Master output — all 752 stocks
│   │   │                               with tech score, ML prediction,
│   │   │                               sector trend, ranks etc.
│   │   ├── ml_scores_full.csv          ML inference details
│   │   ├── last_week_scores.csv        Previous week scores (for rank change)
│   │   └── breakout_tracker.csv        Stocks near breakout, week count
│   │
│   ├── reports\
│   │   ├── technical\         ← Weekly/quarterly text reports
│   │   │   ├── weekly_report_short_YYYYMMDD.txt
│   │   │   ├── weekly_report_long_YYYYMMDD.txt
│   │   │   ├── quarterly_report_short_YYYYMMDD.txt
│   │   │   └── quarterly_report_long_YYYYMMDD.txt
│   │   │
│   │   ├── ml\                ← ML specific reports
│   │   │   ├── weekly_ml_report_short_YYYYMMDD.txt
│   │   │   ├── weekly_ml_report_long_YYYYMMDD.txt
│   │   │   ├── quarterly_ml_report_short_YYYYMMDD.txt
│   │   │   └── quarterly_ml_report_long_YYYYMMDD.txt
│   │   │
│   │   └── portfolio\         ← Portfolio reports
│   │       ├── lt_portfolio_YYYYMMDD.txt
│   │       ├── swing_portfolio_YYYYMMDD.txt
│   │       ├── model_portfolio_review_YYYYMMDD.txt
│   │       └── model_portfolio_creation_YYYYMMDD.txt
│   │
│   ├── portfolio\             ← Your actual portfolio files
│   │   ├── long_term_portfolio.csv     Your LT holdings
│   │   ├── swing_portfolio.csv         Your swing holdings
│   │   └── model_portfolios\           10 auto-created model portfolios
│   │       ├── p1_tier1_equal_weight.csv
│   │       ├── p2_tier1_rank_weighted.csv
│   │       ├── p3_tier1_2_rank_weighted.csv
│   │       ├── p4_high_ml_confidence.csv
│   │       ├── p5_high_forecast_25d.csv
│   │       ├── p6_high_tech_score.csv
│   │       ├── p7_cap_diversified.csv
│   │       ├── p8_sector_concentrated.csv
│   │       ├── p9_bull_sector_only.csv
│   │       ├── p10_max_forecast_45d.csv
│   │       └── <your_custom_name>.csv  ← your own model portfolios
│   │
│   └── temp\                  ← Checkpoints during long runs
│       ├── prefilt_checkpoint.pkl      (auto-deleted after run)
│       ├── fund_scrape_checkpoint.pkl  (auto-deleted after run)
│       └── tech_checkpoint.pkl         (auto-deleted after run)
│
├── models\                    ← Trained ML model files
│   ├── bottom_models.pkl           Reversal classifier (bottom)
│   ├── bottom_encoders.pkl
│   ├── top_models.pkl              Reversal classifier (top)
│   ├── top_encoders.pkl
│   ├── trend_models.pkl            3-class trend classifier
│   ├── trend_encoders.pkl
│   ├── trend_label_encoders.pkl
│   ├── forecast_models.pkl         Price forecast regressor (25d/45d/180d)
│   ├── forecast_encoders.pkl
│   ├── bullish_cont_models.pkl     Continuation classifier
│   ├── bullish_cont_encoders.pkl
│   ├── bearish_cont_models.pkl
│   ├── bearish_cont_encoders.pkl
│   ├── symbol_group.pkl            Symbol → group mapping
│   └── group_stocks.pkl            Group → [symbols] mapping
│
├── models_backup\             ← Auto-backup before retraining
│   └── (same files as models\)
│
├── notebook\                  ← Development Jupyter notebooks
│   ├── day1 to day13 notebooks     All development history
│   └── day13_portfolio.ipynb       Portfolio manager development
│
├── run_weekly.py              ← Run every Saturday
├── run_quarterly.py           ← Run every quarter
├── run_retrain.py             ← Run after quarterly
├── run_portfolio.py           ← Run when needed (LT/Swing/Model review)
├── run_model_portfolio.py     ← Run once to create 10 model portfolios
├── .gitignore                 ← models\ folder is ignored
├── README.md                  ← GitHub readme
└── READ_ME.txt                ← This file


============================================================
  SCRIPT 1: run_weekly.py
============================================================

  PURPOSE  : Weekly stock screening — generates fresh reports
             every Saturday with updated prices + ML signals

  WHEN     : Every Saturday morning
             Set up in Windows Task Scheduler to run automatically

  COMMAND  :
    python run_weekly.py

  WHAT IT DOES (9 steps, ~25 minutes):
    Step 1  Load quality_passed.csv (752 stocks)
    Step 2  Download incremental price updates from yfinance
    Step 3  Compute technical indicators (EMA, RSI, MACD, ADX)
    Step 4  Build ML features from price/indicator data
    Step 5  Load ML models from models\ folder
    Step 6  Run ML inference (2-pass — tech analysis first, then ML)
    Step 7  Technical analysis (volume profile + consolidation detection)
    Step 8  Report data preparation (sector trend, rank scores, breakouts)
    Step 9  Generate 4 text reports

  FILES IT READS:
    data\universe\quality_passed.csv
    data\universe\master_stocks.csv
    data\fundamentals\fundamental_scores_full.csv
    data\prices\price_data_full.pkl         (updates this)
    data\prices\indicator_data_full.pkl     (updates this)
    models\*.pkl

  FILES IT SAVES:
    data\prices\price_data_full.pkl         ← updated with latest prices
    data\prices\indicator_data_full.pkl     ← updated indicators
    data\scores\technical_report_full.csv   ← MASTER OUTPUT FILE
    data\scores\ml_scores_full.csv
    data\scores\last_week_scores.csv        ← for rank change tracking
    data\scores\breakout_tracker.csv        ← updated breakout tracker
    data\reports\technical\weekly_report_short_YYYYMMDD.txt
    data\reports\technical\weekly_report_long_YYYYMMDD.txt
    data\reports\ml\weekly_ml_report_short_YYYYMMDD.txt
    data\reports\ml\weekly_ml_report_long_YYYYMMDD.txt

  IMPORTANT:
    - Never run in parallel with other yfinance scripts
    - Models are loaded fresh each run — any retraining is auto-picked up
    - If run fails mid-way, just re-run — price data is incremental


============================================================
  SCRIPT 2: run_quarterly.py
============================================================

  PURPOSE  : Full universe refresh every quarter —
             re-scrapes fundamentals, re-filters universe,
             re-scores all stocks

  WHEN     : Once per quarter, after result season is complete
             Recommended: Mid-May, Mid-Aug, Mid-Nov, Mid-Feb
             Always run BEFORE run_retrain.py

  COMMAND  :
    python run_quarterly.py --force

  (Without --force it skips steps 2-5 if files already exist)
  (Without --force after a crash it resumes from checkpoint)

  WHAT IT DOES (12 steps, ~4-6 hours):
    Step 1  Load master universe, deduplicate NSE/BSE
    Step 2  Pre-filter all 4,953 stocks (price/volume/mcap/listing)
    Step 3  Scrape fundamentals from Screener.in for ~2000+ stocks
    Step 4  Apply tiered quality filter → new quality_passed.csv
    Step 5  Compute fundamental scores (Historical + Peer + Quality)
    Step 6  Update price data (incremental for existing, full for new)
    Step 7  Compute technical indicators
    Step 8  Build ML features
    Step 9  Run ML inference (pass 1)
    Step 10 Full technical analysis (VP + consolidation)
    Step 11 Report data preparation
    Step 12 Generate 4 quarterly reports

  FILES IT READS:
    data\universe\master_stocks.csv
    data\universe\bse_stocks.csv
    models\*.pkl

  FILES IT SAVES (overwrites everything):
    data\universe\prefilt_passed.csv        ← new pre-filtered universe
    data\universe\quality_passed.csv        ← new quality universe
    data\universe\quality_filter_results.csv
    data\fundamentals\fundamental_metrics_full.csv
    data\fundamentals\fundamental_scores_full.csv
    data\fundamentals\raw_stock_data_full.pkl
    data\prices\price_data_full.pkl
    data\prices\indicator_data_full.pkl
    data\scores\technical_report_full.csv
    data\scores\ml_scores_full.csv
    data\scores\last_week_scores.csv
    data\scores\breakout_tracker.csv
    data\reports\technical\quarterly_report_short_YYYYMMDD.txt
    data\reports\technical\quarterly_report_long_YYYYMMDD.txt
    data\reports\ml\quarterly_ml_report_short_YYYYMMDD.txt
    data\reports\ml\quarterly_ml_report_long_YYYYMMDD.txt

  IF IT CRASHES:
    Checkpoints are saved every 50-100 stocks in data\temp\
    Just re-run WITHOUT --force → it will resume from checkpoint
    Checkpoint files are auto-deleted after successful completion

  IMPORTANT:
    - Screener.in may slow down or block mid-run
      If Step 3 shows many failures, let it finish then re-run
      without --force to retry failed stocks
    - Step 2 takes ~40 mins (4953 stocks × 0.5s)
    - Step 3 takes ~90 mins (2000+ stocks × 2.5s)


============================================================
  SCRIPT 3: run_retrain.py
============================================================

  PURPOSE  : Retrain all 4 ML models on the updated universe
             after a quarterly run

  WHEN     : After run_quarterly.py completes
             Run BEFORE next run_weekly.py for fresh ML signals

  COMMAND  :
    python run_retrain.py

  WHAT IT DOES (8 steps, ~1.5-2 hours):
    Step 0  Backs up existing models\ to models_backup\
    Step 1  Load price, indicator, fundamental, tech data
    Step 2  Assign stocks to 6 groups (IT/Financial/Chemicals/
            Healthcare/Consumer/Industrial)
    Step 3  Build training features + labels for all stocks
    Step 4  Train Model 1: Reversal classifier (bottom + top)
    Step 5  Train Model 2: 3-class trend classifier
    Step 6  Train Model 3: Price forecast regressor (25d/45d/180d)
    Step 7  Train Model 4: Continuation classifier (bullish + bearish)
    Step 8  Save all models + group mappings to models\

  MODELS TRAINED:
    Model 1 — Reversal Classifier
              Predicts bottom reversal probability
              Predicts top reversal probability
              XGBoost, scale_pos_weight, aucpr metric

    Model 2 — Trend Classifier (3-class)
              Predicts Uptrend / Sideways / Downtrend
              XGBoost, mlogloss metric

    Model 3 — Price Forecast Regressor
              Predicts 25d, 45d, 180d price return
              XGBoost regressor, directional accuracy metric

    Model 4 — Continuation Classifier
              Predicts bullish continuation probability
              Predicts bearish continuation probability
              XGBoost, scale_pos_weight, aucpr metric

  TRAINING METHOD:
    Walk-forward validation — 5 folds (2021→2025)
    Final model trained on ALL historical data
    6 separate models per model type (one per sector group)
    = 4 × 6 = 24 model objects + encoders

  FILES IT READS:
    data\prices\price_data_full.pkl
    data\prices\indicator_data_full.pkl
    data\fundamentals\fundamental_scores_full.csv
    data\scores\technical_report_full.csv
    data\universe\prefilt_passed.csv

  FILES IT SAVES:
    models_backup\*.pkl         ← backup of old models (before overwrite)
    models\*.pkl                ← 15 new model files

  IF MODELS GET WORSE:
    Delete models\ folder
    Rename models_backup\ to models\
    Weekly and quarterly will use old models again

  IMPORTANT:
    - Always run after quarterly, before weekly
    - Weekly and quarterly auto-use new models (no config change needed)
    - obv_slope_10d uses obv.abs().rolling(10).mean() as denominator
      (matches training — do not change)


============================================================
  SCRIPT 4: run_portfolio.py
============================================================

  PURPOSE  : Portfolio manager — recommendations + tracking
             for Long Term, Swing, and Model portfolios

  WHEN     : Run manually whenever needed
             Long Term  → monthly review
             Swing      → weekly review (every Saturday after run_weekly)
             Model      → weekly review (Option 5)

  COMMAND  :
    python run_portfolio.py

  MENU OPTIONS:
    1. Long Term Portfolio Review
       Shows top stock recommendations (Tier 1/2/3) +
       reviews your existing LT holdings (Hold/Add/Exit)
       Saves: data\reports\portfolio\lt_portfolio_YYYYMMDD.txt

    2. Swing Trading Portfolio Review
       Shows top swing recommendations (Bull Cont only) +
       reviews your existing swing holdings (Hold/Add/Exit/Carry)
       Saves: data\reports\portfolio\swing_portfolio_YYYYMMDD.txt

    3. Both Reviews (LT + Swing)
       Runs options 1 and 2 together

    4. Create / Update Model Portfolio
       Enter any portfolio name + stocks + prices
       Saves: data\portfolio\model_portfolios\<name>.csv
       Use this for YOUR OWN custom model portfolios

    5. Review Model Portfolios (P&L tracking)
       Shows selector → pick one or all portfolios
       Shows P&L per stock + ranking across portfolios (🥇🥈🥉)
       Saves: data\reports\portfolio\model_portfolio_review_YYYYMMDD.txt

    6. All Reviews (LT + Swing + Model)
       Runs options 1, 2, and 5 together

  PRICE FILTER (asked at startup):
    1. Standard     → Price > EMA50 > EMA200 (all qualifying stocks)
    2. Early Entry  → within X% of EMA50 (stocks not yet run up)
    3. Both         → show all, mark [HIGH] if > X% above EMA50
    Default: 3 (Both), Default threshold: 10%

  RECOMMENDATION LOGIC:
    ★★★ Tier 1: Sector bullish + ML Bullish Continual +
                Tech Momentum + Price > EMA50 > EMA200
                ML Confidence >= 40%
    ★★☆ Tier 2: ML Bullish Continual + Tech Momentum +
                Price > EMA50 > EMA200 + ML Confidence >= 40%
                (sector not bullish — stock holding in bear market)
    ★☆☆ Tier 3: ML Reversal + Tech Reversal +
                Fund Score >= 65 (LT only, max 2 per cap category)

  HOLD/ADD/EXIT LOGIC:
    Long Term:
      EXIT  → Both ranks dropped >= 1.5 pts OR
               ML turned bearish (conf >= 50%) OR
               Both forecasts < -3%
      ADD   → Both ranks improved >= 1.0 pts AND
               ML conf >= 65% AND Tech score >= 70
      [A]   → Accumulation mode — price exits suppressed
               Only rank + ML triggers exit

    Swing:
      EXIT  → ML no longer Bullish Continual OR
               Both forecasts negative OR
               EMA structure broken OR
               30-day cycle complete (weak setup) OR
               Portfolio drawdown >= 8% (exit weakest first)
      CARRY → 30 days done but ML conf >= 65% + forecast > 0
      ADD   → ML conf >= 75% + forecast > 3% +
               Sector + stock both bullish

  PORTFOLIO FILES:
    data\portfolio\long_term_portfolio.csv
      Columns: Symbol, Entry_Price, Entry_Date, Quantity,
               Cap_Category, Sector_Rank_At_Entry, Cap_Rank_At_Entry,
               Sector_Rank_Change, Cap_Rank_Change,
               Accumulation_Mode, Notes

    data\portfolio\swing_portfolio.csv
      Same columns + Cycle_Start_Date (no Accumulation_Mode)

    data\portfolio\model_portfolios\<name>.csv
      Same columns (no Accumulation_Mode, no Cycle_Start_Date)

  SWING ALLOCATION:
    Rs 1.5 Lakhs capital
    Weight = 50% Sector Rank + 50% Cap Rank
    Normalized to 100% across holdings
    Rs amount shown per stock

  RANK TRACKING:
    Sector_Rank_At_Entry — recorded when stock is first added
    Cap_Rank_At_Entry    — recorded when stock is first added
    Sector_Rank_Change   — recomputed every run vs entry baseline
    Cap_Rank_Change      — recomputed every run vs entry baseline
    These ranks come from fundamental scores within sector/cap group


============================================================
  SCRIPT 5: run_model_portfolio.py
============================================================

  PURPOSE  : Automatically creates 10 pre-defined model portfolios
             at current prices using different selection logics
             Used to test which selection logic performs best

  WHEN     : Run ONCE to initialize (already done: 15-Apr-2026)
             Re-run only if you want to reset all portfolios
             (it will ask before overwriting)

  COMMAND  :
    python run_model_portfolio.py

  10 PORTFOLIO LOGICS (all use Rs 1.5L capital, max 10 stocks):
    Base filter for all: Bullish Continual + Momentum +
                         EMA OK + ML Confidence >= 40%

    P1  Pure Tier 1 Equal Weight
        Only ★★★ stocks (sector+ML+tech all aligned)
        Equal allocation across all 10 stocks

    P2  Pure Tier 1 Rank Weighted
        Only ★★★ stocks
        50% Sector Rank + 50% Cap Rank weighting

    P3  Tier 1+2 Rank Weighted  ← Main screener logic
        ★★★ first, then ★★☆ to fill up to 10
        50% Sector Rank + 50% Cap Rank weighting

    P4  High ML Confidence
        Top 10 by ML confidence (>= 75% preferred)
        Rank weighted allocation

    P5  High Forecast 25d
        Top 10 by 25-day price forecast %
        Rank weighted allocation

    P6  High Tech Score
        Top 10 by technical score (>= 80 preferred)
        Rank weighted allocation

    P7  Cap Diversified
        2-3 stocks per cap category (Large/MiniLarge/Mid/Small)
        Rank weighted allocation

    P8  Sector Concentrated
        Best 3 stocks from top 2 bullish sectors
        Rank weighted allocation

    P9  Bull Sector Only
        Only ★★★ stocks — all three aligned
        Same as P1 but rank weighted (P1 is equal weight)

    P10 Max Forecast 45d
        Top 10 by 45-day price forecast %
        Rank weighted allocation

  FILES IT SAVES:
    data\portfolio\model_portfolios\p1_tier1_equal_weight.csv
    data\portfolio\model_portfolios\p2_tier1_rank_weighted.csv
    data\portfolio\model_portfolios\p3_tier1_2_rank_weighted.csv
    data\portfolio\model_portfolios\p4_high_ml_confidence.csv
    data\portfolio\model_portfolios\p5_high_forecast_25d.csv
    data\portfolio\model_portfolios\p6_high_tech_score.csv
    data\portfolio\model_portfolios\p7_cap_diversified.csv
    data\portfolio\model_portfolios\p8_sector_concentrated.csv
    data\portfolio\model_portfolios\p9_bull_sector_only.csv
    data\portfolio\model_portfolios\p10_max_forecast_45d.csv
    data\reports\portfolio\model_portfolio_creation_YYYYMMDD.txt

  HOW TO TRACK:
    Every Saturday after run_weekly.py:
    python run_portfolio.py → Option 5 → All portfolios
    Ranking table shows which logic is outperforming


============================================================
  QUARTERLY WORKFLOW (May 2026 and beyond)
============================================================

  Run in this exact order:

  Step 1 — Quarterly refresh (mid-May, after result season)
    python run_quarterly.py --force
    Time: ~4-6 hours
    What: Fresh fundamentals, new universe, new scores

  Step 2 — ML retraining (immediately after quarterly)
    python run_retrain.py
    Time: ~1.5-2 hours
    What: Retrain all 4 models on updated universe

  Step 3 — Validate with weekly run
    python run_weekly.py
    Time: ~25 minutes
    What: Fresh weekly report using new models

  Step 4 — Review portfolio
    python run_portfolio.py → Option 3
    What: Updated recommendations + portfolio review


============================================================
  WEEKLY WORKFLOW (every Saturday)
============================================================

  Step 1 — Update prices + generate report
    python run_weekly.py
    Time: ~25 minutes

  Step 2 — Review portfolio + model portfolios
    python run_portfolio.py → Option 6 (All reviews)
    OR
    python run_portfolio.py → Option 2 (Swing only, weekly)
    python run_portfolio.py → Option 5 (Model portfolio P&L)


============================================================
  ML MODEL DETAILS
============================================================

  6 Sector Groups:
    IT         : Information Technology (60 stocks)
    Financial  : Financial Services (91 stocks)
    Chemicals  : Chemicals (57 stocks)
    Healthcare : Healthcare (68 stocks)
    Consumer   : Consumer Durables + Services + FMCG (120 stocks)
    Industrial : Everything else (356 stocks)

  Feature Columns (29 total):
    Price    : return_1d, return_5d, return_20d, return_60d,
               dist_52w_high, dist_52w_low, atr_pct,
               volatility_20d, volatility_60d
    Volume   : vol_ratio_5d, vol_ratio_20d, obv_slope_10d, vol_spike
    RSI      : rsi, rsi_slope_5d, rsi_oversold, rsi_overbought
    MACD     : macd_hist, macd_slope_3d, macd_slope_5d, macd_cross
    ADX      : adx, adx_slope, di_spread
    EMA      : price_vs_ema20, price_vs_ema50, price_vs_ema200,
               ema20_vs_ema50, ema50_vs_ema200
    + symbol_enc (added during training/inference)

  ML Prediction Labels:
    Bullish Continual → stock in uptrend, ML confirms continuation
    Bearish Continual → stock in downtrend, likely continues down
    Reversal          → stock turning around (oversold bounce)
    Bullish           → positive forecast but no strong setup
    Bearish           → negative outlook
    No Signal         → no clear direction

  ML Confidence:
    For Bullish Continual → Bullish_Cont_Prob from model
    For Reversal          → Bottom_Rev_Prob from model
    >= 40% = model validated signal (used in portfolio)
    < 40%  = label from rule-based logic only (not in portfolio)

  Important Note:
    232 of 362 Bullish Continual stocks have zero ML confidence.
    These are label-only stocks (Best Setup=Momentum + Forecast>0)
    NOT model validated.
    Weekly/quarterly reports show all stocks including zero confidence.
    Portfolio recommendations only show >= 40% confidence stocks.


============================================================
  FUNDAMENTAL SCORING
============================================================

  Final Score = Historical Score (max 40)
              + Peer Score (max 40, percentile within sector)
              + Quality Score (max 20)
              = Total max 100

  Sector Score: Final Score / max_in_sector × 10  (0-10)
  Cap Score   : Final Score / max_in_cap × 10     (0-10)
  Both scores are relative — not percentile

  Quality Filter Thresholds (tiered by market cap):
    Large Cap   (>= Rs 20,000 Cr) : ROE >= 6%,  D/E <= 2.5
    Mini Large  (>= Rs 5,000 Cr)  : ROE >= 7%,  D/E <= 2.0
    Mid Cap     (>= Rs 1,000 Cr)  : ROE >= 8%,  D/E <= 1.5
    Small Cap   (< Rs 1,000 Cr)   : ROE >= 10%, D/E <= 1.0
    Financials exempt D/E and CF rules
    MNCs/institutional holders exempt promoter holding rule


============================================================
  TECHNICAL SCORING
============================================================

  Momentum Score (0-100):
    EMA alignment (30 pts) + RSI (20 pts) + MACD (20 pts)
    + ADX (15 pts) + Volume Profile (15 pts)

  Reversal Score (0-100):
    RSI level (20) + RSI divergence (15) + MACD cross (20)
    + MACD divergence (15) + ADX (15) + Volume (15)

  Best Setup assignment:
    Momentum score >= Reversal score AND >= 50 → Momentum
    Reversal score > Momentum score AND >= 50  → Reversal
    Both < 50                                  → Watching

  Tier assignment:
    TIER 1 BUY NOW (Momentum)    : Fund>=60, Tech>=65, Setup=Momentum
    TIER 1 BUY NOW (Reversal)    : Fund>=60, Tech>=65, Setup=Reversal
    TIER 1 BREAKOUT IMMINENT     : Fund>=60, In consol, within 5% breakout
    TIER 2 WATCHLIST             : Fund>=60, Tech>=40, Setup≠Watching
    TIER 2 BASE BUILDING         : Fund>=60, In consolidation
    TIER 3 WAITING               : Everything else


============================================================
  SECTOR INDEX MAP
============================================================

  Sector trend is computed from index data (yfinance):
    IT            → ^CNXIT
    Healthcare    → ^CNXPHARMA
    Financial     → FINIETF.NS
    Capital Goods → ^CNXINFRA
    Oil & Gas     → ^CNXENERGY
    Auto          → ^CNXAUTO
    FMCG          → ^CNXFMCG
    Metals        → ^CNXMETAL
    Realty        → ^CNXREALTY
    Banking       → ^NSEBANK
    Power         → ^CNXENERGY
    Others        → MID150BEES.NS (proxy)

  Sector Trend Labels:
    Strong Uptrend ↑↑  → All 5 bull signals confirmed, ADX > 25
    Uptrend ↑          → 4 bull signals, ADX > 20
    Weak Uptrend →↑    → 3 bull signals, ADX <= 20
    Sideways →         → Mixed signals
    Weak Downtrend →↓  → 3 bear signals
    Downtrend ↓        → 4 bear signals
    Strong Downtrend↓↓ → All 5 bear signals confirmed


============================================================
  COLUMN REFERENCE (technical_report_full.csv)
============================================================

  Symbol              Stock symbol (NSE or BSE)
  Sector              Sector name from Screener.in
  Fund Score          Fundamental score (0-100)
  Market Cap Cr       Market cap in Crores
  Cap Category        Large/Mini Large/Mid/Small Cap
  Current Price       Last available closing price
  RSI                 14-day RSI
  ADX                 14-day ADX
  MACD Hist           MACD histogram value
  Vol Ratio           Volume / 20-day avg volume
  EMA20/50/200        Exponential moving averages
  DI_Plus/DI_Minus    Directional indicators
  Momentum Score      Technical momentum score (0-100)
  Reversal Score      Technical reversal score (0-100)
  Best Setup          Momentum / Reversal / Watching
  Tech Score          Best of momentum/reversal score
  Tier                Tier classification
  In Consolidation    True/False
  Consol Days         Days in consolidation
  Consol Label        Duration label (e.g. "365d (1-1.5 years)")
  Pct to Breakout     % price is above consolidation resistance
  ML_Prediction       ML label (Bullish Continual etc.)
  ML_Confidence       Model confidence % (0 = rule-based, not model)
  Forecast_25d_Pct    Predicted return in 25 trading days
  Forecast_45d_Pct    Predicted return in 45 trading days
  Forecast_180d_Pct   Predicted return in 180 trading days
  Bottom_Rev_Prob     Probability of bottom reversal (%)
  Top_Rev_Prob        Probability of top reversal (%)
  Bullish_Cont_Prob   Probability of bullish continuation (%)
  Bearish_Cont_Prob   Probability of bearish continuation (%)
  Sector Score        Rank within sector (0-10, relative)
  Cap Score           Rank within sector+cap (0-10, relative)
  Vol 5D Ratio        5-day avg vol / prior 15-day avg vol
  Sector Trend        Sector index trend label
  Sector Detail       RSI, ADX, MACD, EMA details for sector


============================================================
  GITIGNORE — WHAT IS NOT PUSHED TO GITHUB
============================================================

  models\             ← Large pkl files, not versioned
  models_backup\      ← Same
  data\               ← All data files (large, not versioned)
  __pycache__\

  What IS pushed:
    run_weekly.py
    run_quarterly.py
    run_retrain.py
    run_portfolio.py
    run_model_portfolio.py
    .gitignore
    README.md
    READ_ME.txt  ← this file


============================================================
  TROUBLESHOOTING
============================================================

  Q: run_quarterly.py crashes mid-way
  A: Re-run WITHOUT --force. It resumes from checkpoint.
     data\temp\ folder has checkpoint files.

  Q: yfinance not downloading data
  A: Wait and retry. NSE/BSE data has rate limits.
     Never run multiple scripts simultaneously.

  Q: ML confidence is 0 for many stocks
  A: Normal — 232/362 Bullish Continual stocks have zero confidence.
     These got label from rule-based logic, not model validation.
     Portfolio manager filters these out (requires >= 40%).
     Weekly/quarterly reports show all of them — that is correct.

  Q: Models giving worse results after retraining
  A: Restore backup:
     1. Delete models\ folder
     2. Rename models_backup\ to models\
     Weekly and quarterly will use old models.

  Q: Stock not found in portfolio input
  A: Script uses fuzzy matching. Type approximate symbol,
     it shows 3 closest matches with sector/price/mcap.
     Enter 0 to skip if none match.

  Q: Screener.in scraping fails for many stocks
  A: Normal — some stocks don't have full data on Screener.
     Failed stocks are logged. Run without --force to retry.

  Q: run_portfolio.py prices are not live
  A: Prices come from technical_report_full.csv which is
     updated by run_weekly.py. Run weekly first, then portfolio.

============================================================
  END OF READ_ME
  Last updated: 15-April-2026
============================================================
