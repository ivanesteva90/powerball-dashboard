# Powerball Analytical Study — Archived Snapshot

> This document preserves the March 2026 study and is not refreshed automatically. The live dashboard and its
> `Validación histórica` page are the current source for data range, forecast calibration, and backtest results.

## 1. Data profile
Parsed historical draws: **1,921**
Date range: **2010-02-03** to **2026-03-16**
Rows without `Power Play`: **210**
Observed draw weekdays: **Wednesday, Saturday, Monday**
First Monday draw in this file: **2021-08-23**

### Inferred matrix periods from the uploaded file
1. **2010-2011 | 5/59 + PB39** — 204 draws
2. **2012-2015 | 5/59 + PB35** — 388 draws
3. **2015-2026 | 5/69 + PB26** — 1,329 draws

## 2. Why a fixed expected value is wrong for this file
If you compute white-ball expectation with `5T/69` across the full history, you bias the analysis because white balls `60-69` were not available in the older periods.

The correct mixed-era expectation is:

### White balls
For number `n`:
`E[n] = Σ_t I(n <= M_t) * (5 / M_t)`

### Powerball
For number `n`:
`E_pb[n] = Σ_t I(n <= P_t) * (1 / P_t)`

where:
- `M_t` = white-ball pool size in draw `t`
- `P_t` = Powerball pool size in draw `t`
- `I(.)` = indicator function

## 3. Core formulas worth combining with Python

### A. Frequency
`f_n = Σ_t I(n appears in draw t)`

### B. Variance across mixed eras
White ball:
`Var[n] = Σ_t p_t (1 - p_t)` where `p_t = 5/M_t` if `n <= M_t`, else `0`

Powerball:
`Var_pb[n] = Σ_t q_t (1 - q_t)` where `q_t = 1/P_t` if `n <= P_t`, else `0`

### C. Z-score
`z_n = (f_n - E[n]) / sqrt(Var[n])`

### D. Chi-square goodness-of-fit
`χ² = Σ_n (O_n - E_n)^2 / E_n`

Use this against the era-aware expected values rather than a fixed uniform vector.

### E. Overdue / gap
`gap_n = current_draw_index - last_draw_index_where_n_appeared`

### F. Rolling hit count
For a window `w`:
`rolling_n(t) = Σ_{k=t-w+1}^{t} I(n appears in draw k)`

### G. Pair frequency
`c_{ij} = Σ_t I(i and j appear together in draw t)`

### H. Structural metrics by draw
- White-ball sum
- Odd count / even count
- Low count / high count
- Range width = `max - min`
- Consecutive pairs
- Repeats from previous draw

## 4. Statistical findings from this exact file

## Era 1: 2010-2011 | 5/59 + PB39
- White balls chi-square: **65.02**, p-value **0.2455**
- Powerball chi-square: **30.76**, p-value **0.7915**

## Era 2: 2012-2015 | 5/59 + PB35
- White balls chi-square: **37.11**, p-value **0.9852**
- Powerball chi-square: **32.54**, p-value **0.5391**

## Era 3: 2015-2026 | 5/69 + PB26
- White balls chi-square: **82.31**, p-value **0.1138**
- Powerball chi-square: **23.57**, p-value **0.5445**

### Interpretation
Within each era, the observed frequencies are **not strongly inconsistent with the expected uniform behavior**. That means the file is much more useful for descriptive analysis and anomaly screening than for true prediction.

## 5. Most useful practical outputs from the dashboard

### Current-era hot white balls (2015-10-07 onward)
Top observed counts in the uploaded file:
- 61 → 119
- 21 → 117
- 23 → 116
- 28 → 116
- 33 → 114
- 27 → 113
- 32 → 113
- 64 → 113
- 69 → 112

### Current-era hot Powerballs
- 4 → 64
- 21 → 63
- 14 → 61
- 24 → 61
- 18 → 59

### Current-era most overdue white balls as of 2026-03-16
- 67 → 61 draws since seen
- 12 → 52
- 13 → 43
- 44 → 43
- 32 → 42

### Current-era most overdue Powerballs as of 2026-03-16
- 8 → 98 draws since seen
- 25 → 86
- 9 → 81
- 3 → 47
- 22 → 41

### Current-era most frequent pairs
- 21-32 → 15
- 61-69 → 15
- 51-61 → 14
- 52-64 → 14
- 37-44 → 14

## 6. Best modeling strategy
Use a 3-layer approach:

### Layer 1 — Descriptive
Frequencies, gaps, rolling counts, pairs, triplets, structural metrics.

### Layer 2 — Inferential
Era-aware expectations, z-scores, chi-square tests.

### Layer 3 — Experimental ranking
A composite score such as:
`Score[n] = 0.45 * z_long + 0.35 * z_recent_52 + 0.20 * z_gap`

This is an **exploration score**, not a prediction claim.

## 7. Recommended Python stack
- `pandas` — parsing and reshaping
- `numpy` — vectorized math
- `scipy` — chi-square and p-values
- `plotly` — interactive charts
- `streamlit` — dashboard and CSV upload

## 8. Practical conclusion
This uploaded file supports a strong **statistical exploration system**, not a reliable prediction engine. The mathematically solid move is to build a dashboard that is:
- upload-driven,
- era-aware,
- expectation-aware,
- and focused on deviation, recency, and structure.

## 9. Physical-bias module policy
The dashboard includes a **Physical Bias Simulator** only as a controlled sensitivity module:
- `Uniform` baseline (era-aware expected values)
- An explicit target-ball stress scenario using an approximate nominal mass of 80 g
- Measured `number,weight` values when supplied by the user

This module is explicitly exploratory:
- It does **not** claim predictive power.
- It is meant to compare how rankings would change under small hypothetical perturbations.
- It does not infer physical weight from the ball number or historical frequency.
- If no measured weights or explicit scenario are supplied, the result remains uniform.

## 10. Accuracy v3 forecast policy
The production forecast treats the five white balls as one unordered, fixed-size subset. Given positive
weights `w`, the probability of a five-number set `S` is:

`P(S | |S|=5) = product(w_i for i in S) / e_5(w)`

where `e_5(w)` is the fifth elementary-symmetric sum. This provides exact marginal inclusion probabilities
whose total is five and an exact sampler consistent with the displayed POP values.

The forecast is mixed with the official uniform reference:

`P_final = model_weight * P_model + (1 - model_weight) * P_uniform`

Model strength and the initial weight are selected on the calibration period. A later holdout determines the
evidence label from a bootstrap interval of Brier improvement:
- `Evidencia de mejora`: retain the calibrated model weight.
- `Mejora incierta`: retain only 25% of the calibrated model weight.
- `Sin mejora`: use the uniform model.

The dashboard also reports moving-block bootstrap intervals for next-draw POP. These ranges describe model
uncertainty; they do not alter the official game odds.

## 11. Equipment and pre-test evidence
The official MUSL pre-test report is parsed into a normalized table containing draw order, machine IDs,
ball-set IDs, pre-tests, official draws, and post-tests. The dashboard joins official draw rows to the Texas
winning-number CSV by date and verifies both the unordered white set and Powerball value.

For equipment-conditioned probabilities, the walk-forward estimator uses only rows before each target draw.
Machine-only and set-only rates are independently shrunk toward the uniform model with a 100-draw prior and
then averaged. This avoids treating small equipment groups as strong evidence.

Result on the current 5/69 + PB26 era through the available equipment report:
- White equipment Brier: `0.067288` versus uniform `0.067213`.
- Powerball equipment Brier: `0.037027` versus uniform `0.036982`.
- Both evidence labels are `Sin mejora` because their mean improvements are negative.
- No white machine-number comparison survives 5% FDR in the current analysis.
- Mean white overlap between pre-tests and the official draw is approximately `1.398`, versus an expected
  `1.395` under independence and the applicable matrix.

Therefore machine/set and pre-test information remains a retrospective integrity diagnostic. It is not part
of the next-draw POP unless a future preregistered out-of-sample test produces stable positive evidence and
the equipment selection is available before the relevant ticket-sales cutoff.

## 12. Play-plan mathematics
For `k` distinct full tickets in one draw under a `5/69 + 1/26` matrix:

`P(jackpot in one draw) = k / (C(69,5) * 26)`

For `d` independent draws:

`P(at least one jackpot) = 1 - (1 - P(draw))^d`

The dashboard uses stable `log1p`/`expm1` evaluation for this cumulative probability and computes cost from
the user-selected ticket price and schedule. Candidate portfolios penalize overlap and reward incremental
coverage, but the dashboard explicitly keeps that portfolio score separate from the jackpot probability.
Because the validated forecast has no stable positive edge, zero tickets remains the financial optimum;
positive ticket counts are treated as an entertainment budget rather than an investment recommendation.
