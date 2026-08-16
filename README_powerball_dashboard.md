# Powerball Analytics Dashboard

This Streamlit dashboard loads a historical Powerball CSV and normalizes mixed historical rule periods before computing frequency, deviation, recency, and structural metrics.

## Files
- `powerball_dashboard_app.py` — main Streamlit app
- `powerball_core.py` — pure forecast and walk-forward validation logic
- `test_powerball_core.py` — unit tests for the forecast core
- `powerball.csv` — sample historical file
- `POWERBALL_ANALYTICAL_STUDY.md` — formulas, methodology, and findings for the uploaded file
- `requirements_powerball_dashboard.txt` — Python dependencies

## Why this version is better
This historical file is **not a single regime**. It mixes at least these eras:
- `2010-2011 | 5/59 + PB39`
- `2012-2015 | 5/59 + PB35`
- `2015-present | 5/69 + PB26`

Because of that, a fixed expected value such as `5T/69` for all white balls or `T/26` for all Powerballs is mathematically wrong over the full dataset.

This app uses **era-aware expected values**:
- White ball `n`: `E[n] = Σ_t 5/M_t` for draws where `n <= M_t`
- Powerball `n`: `E[n] = Σ_t 1/P_t` for draws where `n <= P_t`

where:
- `M_t` = white-ball pool size in draw `t`
- `P_t` = Powerball pool size in draw `t`

## Features
- Numbered sidebar navigation with independent sections, including a dedicated historical-validation page.
- Upload CSV directly in the app
- Automatic official data sync from Texas Lottery CSV with a six-hour cache and manual refresh:
  - Source: `https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/powerball.csv`
  - Bundled local CSV is used as a fallback if the official source is unavailable
- Auto-parse rows with and without `Power Play`
- Filter by era, year, weekday, and date range
- Observed vs expected counts
- 95% confidence bands for expected counts
- Chi-square uniformity tests
- Z-scores versus era-aware expectation
- p-values and FDR-adjusted q-values for multiple testing control
- Overdue / recency tables
- Sum / parity / range / consecutive-pair structure
- Pair and triplet frequencies
- Pair co-occurrence heatmap (top white numbers)
- Rolling 52-draw hit counts
- Data-only diagnostics:
  - Number vs z-score relation
  - Bucket deviations (1-10, 11-20, ...)
  - Last-digit deviations
  - Era-stability consistency table + heatmap
- Data-quality checks:
  - Duplicate white numbers in a draw
  - Out-of-range white numbers by era
  - Out-of-range Powerball by era
- Composite exploration score with tunable weights for descriptive views only
- Physical Bias Simulator (experimental):
  - Uniform mode
  - Weight bias mode
  - Weight + wear mode
  - Optional measured weights upload (`number,weight`)
- Forecast section now includes:
  - Filterable views (`Mas probables`, `Menos probables`, `Mas atrasadas`, `Mas frias`)
  - White number range filter
  - Forecast trained independently from visualization filters
  - Current-matrix-only training with Bayesian smoothing
  - No overdue/gap signal in the default forecast
  - Model strength calibrated on early history and evaluated on a later holdout
  - White inclusion rates estimated from the same without-replacement ticket simulation
  - Data-clear block with winning/losing combinations:
    - Top exact tickets (5+PB)
    - One-hit losing tickets
    - Top white pairs and triplets
  - Filter-aware candidate ticket simulation
- Rolling view now includes quick modes:
  - Manual
  - Top forecast
  - Bottom forecast
  - Most overdue
- CSV export of filtered data
- Excel export with multi-sheet analytical outputs
- Walk-forward backtest with Brier score, log-loss, top-k hits, yearly stability, and uniform baseline
- Era-aware expected values for historical pairs and triplets
- Unit tests for the pure forecast core in `powerball_core.py`

## Install
```bash
pip install -r requirements_powerball_dashboard.txt
```

## Run
```bash
streamlit run powerball_dashboard_app.py
```

## Test
```bash
python3 -m unittest -v test_powerball_core.py
```

## GitHub + Live (Streamlit Community Cloud)
1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, create app from that repo.
3. Set main file path to `streamlit_app.py` (or `powerball_dashboard_app.py`).
4. Deploy.

Texas CSV manual sync in-app uses:
- `https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/powerball.csv`

## Notes
- The exploration score is experimental and should be treated as a ranking aid, not a prediction engine.
- The Physical Bias Simulator is sensitivity analysis only (uniform vs hypothetical/measured micro-bias), not predictive proof.
- This dashboard is strongest for descriptive statistics, anomaly detection, and historical segmentation.
