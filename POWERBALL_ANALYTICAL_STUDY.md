# Powerball Analytical Study for Uploaded Historical CSV

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
- `Weight bias` (hypothetical or measured `number,weight`)
- `Weight + wear` (adds a wear proxy term)

This module is explicitly exploratory:
- It does **not** claim predictive power.
- It is meant to compare how rankings would change under small hypothetical perturbations.
- If no measured weights are uploaded, the app uses a transparent hypothetical signal and labels it as such.
