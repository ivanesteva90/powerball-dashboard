# Powerball Analytics Dashboard

This Streamlit dashboard loads a historical Powerball CSV and normalizes mixed historical rule periods before computing frequency, deviation, recency, and structural metrics.

## Files
- `powerball_dashboard_app.py` — main Streamlit app
- `powerball_core.py` — pure forecast and walk-forward validation logic
- `powerball_equipment.py` — official pre-test parser and equipment analysis
- `test_powerball_core.py` — unit tests for the forecast core
- `test_powerball_equipment.py` — unit tests for equipment parsing and validation
- `powerball.csv` — sample historical file
- `powerball_pretests.csv` — normalized local fallback for the official pre-test report
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
- Automatic Powerball pre-test/equipment sync with the same refresh control:
  - Source: `https://cdn.powerball.com/v01/media/powerball-pre-test.pdf`
  - Extracts draw order, machine IDs, ball-set IDs, pre-tests, official draws, and post-tests
  - Audits number agreement against the Texas winning-number CSV
  - Bundled normalized CSV is used as a fallback
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
- Equipment and pre-test section:
  - Machine/set usage history
  - Number deviations by machine or set with FDR correction
  - Pre-test overlap compared with its uniform expectation
  - Walk-forward machine+set validation with hierarchical shrinkage
  - Equipment signals remain outside POP unless future out-of-sample evidence supports them
- Physical Bias Simulator (experimental):
  - Uniform mode with no invented per-number weights
  - Explicit hypothetical target-ball stress scenarios around an 80 g nominal mass
  - Optional measured weights upload (`number,weight`)
  - Sensitivity coefficient is user-controlled and never enters the production POP
- Forecast section now includes:
  - Filterable views (`Mas probables`, `Menos probables`, `Mas atrasadas`, `Mas frias`)
  - White number range filter
  - Forecast trained independently from visualization filters
  - Current-matrix-only training with Bayesian smoothing
  - No overdue/gap signal in the default forecast
  - Model strength and historical weight calibrated on early history and evaluated on a later holdout
  - Exact Conditional Bernoulli white-ball POP for fixed-size selections of five numbers
  - Automatic shrinkage toward the uniform baseline when holdout evidence is weak or negative
  - Moving-block bootstrap 95% intervals for white and Powerball POP
  - Separate model POP, official uniform POP, simulation rate, and full-ticket official probability
  - Holdout improvement intervals, model win rates, and explicit evidence labels
  - Data-clear block with winning/losing combinations:
    - Top exact tickets (5+PB)
    - One-hit losing tickets
    - Top white pairs and triplets
  - Filter-aware candidate ticket simulation using the same fixed-size model as the forecast
  - Ticket portfolio diversity using overlap penalties and incremental number coverage
- Rolling view now includes quick modes:
  - Manual
  - Top forecast
  - Bottom forecast
  - Most overdue
- CSV export of filtered data
- Excel export with multi-sheet analytical outputs
- Walk-forward backtest with Brier score, exact white subset log-loss, top-k hits, yearly stability, bootstrap uncertainty, and uniform baseline
- Era-aware expected values for historical pairs and triplets
- Unit tests for forecast and equipment parsing/validation

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
python3 -m unittest discover -v
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
- Accuracy v3 deliberately keeps forecasts close to uniform unless out-of-sample improvement is statistically supported.
- POP is a model estimate, while official ticket odds remain unchanged under a fair drawing.
- The current equipment walk-forward test does not beat the uniform reference, so machine/set data is diagnostic only.
- The Physical Bias Simulator is sensitivity analysis only (uniform vs explicit hypothetical/measured micro-bias), not predictive proof.
- This dashboard is strongest for descriptive statistics, anomaly detection, and historical segmentation.
