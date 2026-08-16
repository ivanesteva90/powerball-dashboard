from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd


WHITE_COLS = ["num1", "num2", "num3", "num4", "num5"]
DEFAULT_RECENT_WINDOW = 52
DEFAULT_PRIOR_STRENGTH = 20.0
WHITE_LONG_WEIGHT = 0.65
PB_LONG_WEIGHT = 0.25


@dataclass(frozen=True)
class BacktestConfig:
    white_strength: float
    powerball_strength: float
    white_model_weight: float
    powerball_model_weight: float
    white_calibrated_weight: float
    powerball_calibrated_weight: float
    white_evidence: str
    powerball_evidence: str
    calibration_draws: int
    holdout_draws: int
    holdout_start: pd.Timestamp


def calculate_play_plan(
    tickets_per_draw: int,
    ticket_cost: float = 2.0,
    draws_per_week: int = 3,
    years: int = 1,
    white_pool_size: int = 69,
    powerball_pool_size: int = 26,
) -> dict[str, float | int]:
    """Calculate cost and jackpot coverage for distinct full tickets."""
    tickets = max(0, int(tickets_per_draw))
    cost = max(0.0, float(ticket_cost))
    weekly_draws = max(0, int(draws_per_week))
    horizon_years = max(0, int(years))
    total_combinations = comb(int(white_pool_size), 5) * int(powerball_pool_size)
    tickets = min(tickets, total_combinations)
    annual_draws = 52 * weekly_draws
    horizon_draws = annual_draws * horizon_years
    probability_per_draw = tickets / total_combinations

    def cumulative_probability(draw_count: int) -> float:
        if probability_per_draw <= 0 or draw_count <= 0:
            return 0.0
        if probability_per_draw >= 1:
            return 1.0
        return float(-np.expm1(draw_count * np.log1p(-probability_per_draw)))

    annual_probability = cumulative_probability(annual_draws)
    horizon_probability = cumulative_probability(horizon_draws)
    expected_years = (
        total_combinations / (tickets * annual_draws)
        if tickets > 0 and annual_draws > 0
        else np.inf
    )
    return {
        "tickets_per_draw": tickets,
        "total_combinations": total_combinations,
        "annual_draws": annual_draws,
        "horizon_draws": horizon_draws,
        "probability_per_draw": probability_per_draw,
        "probability_per_year": annual_probability,
        "probability_horizon": horizon_probability,
        "one_in_per_draw": 1 / probability_per_draw if probability_per_draw > 0 else np.inf,
        "one_in_per_year": 1 / annual_probability if annual_probability > 0 else np.inf,
        "one_in_horizon": 1 / horizon_probability if horizon_probability > 0 else np.inf,
        "cost_per_draw": tickets * cost,
        "cost_per_week": tickets * cost * weekly_draws,
        "cost_per_year": tickets * cost * annual_draws,
        "cost_horizon": tickets * cost * horizon_draws,
        "expected_years_to_jackpot": expected_years,
    }


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values, ddof=0))
    if not np.isfinite(std) or std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - float(np.max(logits))
    exp_values = np.exp(shifted)
    total = float(exp_values.sum())
    if total <= 0 or not np.isfinite(total):
        return np.repeat(1 / len(logits), len(logits))
    return exp_values / total


def _elementary_symmetric_table(weights: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return prefix/suffix elementary-symmetric sums through the requested order."""
    weights = np.asarray(weights, dtype=float)
    n_items = len(weights)
    prefix = np.zeros((n_items + 1, order + 1), dtype=float)
    suffix = np.zeros((n_items + 1, order + 1), dtype=float)
    prefix[:, 0] = 1.0
    suffix[:, 0] = 1.0
    for i, weight in enumerate(weights):
        prefix[i + 1] = prefix[i]
        for degree in range(1, min(order, i + 1) + 1):
            prefix[i + 1, degree] += weight * prefix[i, degree - 1]
    for i in range(n_items - 1, -1, -1):
        suffix[i] = suffix[i + 1]
        available = n_items - i
        for degree in range(1, min(order, available) + 1):
            suffix[i, degree] += weights[i] * suffix[i + 1, degree - 1]
    return prefix, suffix


def conditional_bernoulli_inclusion_probabilities(weights: np.ndarray, sample_size: int = 5) -> np.ndarray:
    """Exact marginal inclusion probabilities for a fixed-size weighted subset."""
    weights = np.asarray(weights, dtype=float)
    if sample_size < 0 or sample_size > len(weights):
        raise ValueError("sample_size must be between zero and the number of weights")
    if sample_size == 0:
        return np.zeros(len(weights), dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("conditional Bernoulli weights must be finite and positive")
    prefix, suffix = _elementary_symmetric_table(weights, sample_size)
    normalizer = float(prefix[len(weights), sample_size])
    if normalizer <= 0 or not np.isfinite(normalizer):
        raise ValueError("conditional Bernoulli normalizer is invalid")
    inclusion = np.zeros(len(weights), dtype=float)
    for i, weight in enumerate(weights):
        excluding_i = 0.0
        for left_order in range(sample_size):
            right_order = sample_size - 1 - left_order
            excluding_i += prefix[i, left_order] * suffix[i + 1, right_order]
        inclusion[i] = weight * excluding_i / normalizer
    return np.clip(inclusion, 0.0, 1.0)


def conditional_bernoulli_subset_probability(
    weights: np.ndarray,
    selected_indices: np.ndarray | list[int] | tuple[int, ...],
    sample_size: int = 5,
) -> float:
    """Exact probability of one unordered subset under conditional Bernoulli sampling."""
    weights = np.asarray(weights, dtype=float)
    selected = np.asarray(selected_indices, dtype=int)
    if len(selected) != sample_size or len(np.unique(selected)) != sample_size:
        raise ValueError("selected_indices must contain sample_size unique positions")
    if np.any(selected < 0) or np.any(selected >= len(weights)):
        raise ValueError("selected index is outside the weight vector")
    prefix, _ = _elementary_symmetric_table(weights, sample_size)
    normalizer = float(prefix[len(weights), sample_size])
    return float(np.prod(weights[selected]) / normalizer)


def conditional_bernoulli_normalizer(weights: np.ndarray, sample_size: int = 5) -> float:
    """Return the fixed-size subset normalizer for repeated probability calculations."""
    weights = np.asarray(weights, dtype=float)
    if sample_size < 0 or sample_size > len(weights):
        raise ValueError("sample_size must be between zero and the number of weights")
    prefix, _ = _elementary_symmetric_table(weights, sample_size)
    return float(prefix[len(weights), sample_size])


def sample_conditional_bernoulli(
    weights: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw an exact fixed-size subset using suffix elementary-symmetric sums."""
    weights = np.asarray(weights, dtype=float)
    if sample_size < 0 or sample_size > len(weights):
        raise ValueError("sample_size must be between zero and the number of weights")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("conditional Bernoulli weights must be finite and positive")
    _, suffix = _elementary_symmetric_table(weights, sample_size)
    selected: list[int] = []
    remaining = int(sample_size)
    for i, weight in enumerate(weights):
        if remaining == 0:
            break
        items_left = len(weights) - i
        if items_left == remaining:
            selected.extend(range(i, len(weights)))
            break
        denominator = float(suffix[i, remaining])
        probability = weight * suffix[i + 1, remaining - 1] / denominator
        if rng.random() < probability:
            selected.append(i)
            remaining -= 1
    return np.asarray(selected, dtype=int)


def current_matrix_draws(df: pd.DataFrame) -> pd.DataFrame:
    """Return all draws from the matrix used by the latest available draw."""
    if df.empty:
        return df.copy()
    ordered = df.sort_values("draw_date").reset_index(drop=True)
    latest = ordered.iloc[-1]
    mask = (
        ordered["white_pool_max"].eq(int(latest["white_pool_max"]))
        & ordered["powerball_pool_max"].eq(int(latest["powerball_pool_max"]))
    )
    return ordered.loc[mask].reset_index(drop=True)


def _draw_gaps(draw_values: np.ndarray, numbers: np.ndarray) -> np.ndarray:
    gaps = []
    draw_count = len(draw_values)
    for number in numbers:
        if draw_values.ndim == 2:
            hits = np.where(np.any(draw_values == number, axis=1))[0]
        else:
            hits = np.where(draw_values == number)[0]
        gaps.append(draw_count - 1 - int(hits[-1]) if len(hits) else draw_count)
    return np.asarray(gaps, dtype=int)


def _white_signal(
    draw_values: np.ndarray,
    pool_size: int,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    numbers = np.arange(1, pool_size + 1)
    draw_count = len(draw_values)
    recent_count = min(int(recent_window), draw_count)
    recent = draw_values[-recent_count:]
    counts = np.bincount(draw_values.ravel(), minlength=pool_size + 1)[1 : pool_size + 1].astype(float)
    recent_counts = np.bincount(recent.ravel(), minlength=pool_size + 1)[1 : pool_size + 1].astype(float)
    uniform_rate = 5 / pool_size
    long_rate = (counts + prior_strength * uniform_rate) / (draw_count + prior_strength)
    recent_rate = (recent_counts + prior_strength * uniform_rate) / (recent_count + prior_strength)
    signal = WHITE_LONG_WEIGHT * _standardize(long_rate) + (1 - WHITE_LONG_WEIGHT) * _standardize(recent_rate)
    gaps = _draw_gaps(draw_values, numbers)
    return counts, long_rate, recent_rate, gaps, signal


def _powerball_signal(
    draw_values: np.ndarray,
    pool_size: int,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    numbers = np.arange(1, pool_size + 1)
    draw_count = len(draw_values)
    recent_count = min(int(recent_window), draw_count)
    recent = draw_values[-recent_count:]
    counts = np.bincount(draw_values, minlength=pool_size + 1)[1 : pool_size + 1].astype(float)
    recent_counts = np.bincount(recent, minlength=pool_size + 1)[1 : pool_size + 1].astype(float)
    uniform_rate = 1 / pool_size
    long_rate = (counts + prior_strength * uniform_rate) / (draw_count + prior_strength)
    recent_rate = (recent_counts + prior_strength * uniform_rate) / (recent_count + prior_strength)
    signal = PB_LONG_WEIGHT * _standardize(long_rate) + (1 - PB_LONG_WEIGHT) * _standardize(recent_rate)
    gaps = _draw_gaps(draw_values, numbers)
    return counts, long_rate, recent_rate, gaps, signal


def build_white_forecast(
    df: pd.DataFrame,
    strength: float,
    model_weight: float = 1.0,
    recent_window: int = DEFAULT_RECENT_WINDOW,
) -> pd.DataFrame:
    active = current_matrix_draws(df)
    if active.empty:
        return pd.DataFrame()
    pool_size = int(active.iloc[-1]["white_pool_max"])
    numbers = np.arange(1, pool_size + 1)
    values = active[WHITE_COLS].to_numpy(dtype=int)
    counts, long_rate, recent_rate, gaps, signal = _white_signal(values, pool_size, recent_window)
    draw_weight = _softmax(float(strength) * signal)
    model_pop = conditional_bernoulli_inclusion_probabilities(draw_weight, sample_size=5)
    uniform_pop = np.repeat(5 / pool_size, pool_size)
    blend = float(np.clip(model_weight, 0.0, 1.0))
    final_pop = blend * model_pop + (1 - blend) * uniform_pop
    result = pd.DataFrame(
        {
            "number": numbers,
            "observed_active_era": counts.astype(int),
            "long_rate_active": long_rate,
            "recent_rate_52": recent_rate,
            "draws_since_seen": gaps,
            "forecast_signal": signal,
            "draw_weight": draw_weight,
            "model_pop_next_draw": model_pop,
            "pop_next_draw": final_pop,
            "uniform_pop_next_draw": uniform_pop,
            "model_weight": np.repeat(blend, pool_size),
            "lift_vs_uniform_pct": (final_pop / uniform_pop - 1) * 100,
        }
    )
    result = result.sort_values("pop_next_draw", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def build_powerball_forecast(
    df: pd.DataFrame,
    strength: float,
    model_weight: float = 1.0,
    recent_window: int = DEFAULT_RECENT_WINDOW,
) -> pd.DataFrame:
    active = current_matrix_draws(df)
    if active.empty:
        return pd.DataFrame()
    pool_size = int(active.iloc[-1]["powerball_pool_max"])
    numbers = np.arange(1, pool_size + 1)
    values = active["powerball"].to_numpy(dtype=int)
    counts, long_rate, recent_rate, gaps, signal = _powerball_signal(values, pool_size, recent_window)
    model_pop = _softmax(float(strength) * signal)
    uniform_pop = np.repeat(1 / pool_size, pool_size)
    blend = float(np.clip(model_weight, 0.0, 1.0))
    final_pop = blend * model_pop + (1 - blend) * uniform_pop
    result = pd.DataFrame(
        {
            "number": numbers,
            "observed_active_era": counts.astype(int),
            "long_rate_active": long_rate,
            "recent_rate_52": recent_rate,
            "draws_since_seen": gaps,
            "forecast_signal": signal,
            "draw_weight": model_pop,
            "model_pop_next_draw": model_pop,
            "pop_next_draw": final_pop,
            "uniform_pop_next_draw": uniform_pop,
            "model_weight": np.repeat(blend, pool_size),
            "lift_vs_uniform_pct": (final_pop / uniform_pop - 1) * 100,
        }
    )
    result = result.sort_values("pop_next_draw", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def _mean_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def _bootstrap_mean_interval(values: np.ndarray, seed: int, repetitions: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(repetitions), len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _evidence_label(mean_improvement: float, ci_low: float) -> str:
    if mean_improvement <= 0:
        return "Sin mejora"
    if ci_low > 0:
        return "Evidencia de mejora"
    return "Mejora incierta"


def _evidence_gate(calibrated_weight: float, evidence: str) -> float:
    if evidence == "Evidencia de mejora":
        return float(calibrated_weight)
    if evidence == "Mejora incierta":
        return float(calibrated_weight) * 0.25
    return 0.0


def walk_forward_backtest(
    df: pd.DataFrame,
    warmup_draws: int = 104,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    calibration_fraction: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, BacktestConfig]:
    """Calibrate model strength on early history and evaluate on a later holdout."""
    active = current_matrix_draws(df)
    if len(active) <= warmup_draws + 20:
        empty = pd.DataFrame()
        config = BacktestConfig(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Sin datos", "Sin datos", 0, 0, pd.NaT)
        return empty, empty, empty, config

    white_pool = int(active.iloc[-1]["white_pool_max"])
    pb_pool = int(active.iloc[-1]["powerball_pool_max"])
    white_values = active[WHITE_COLS].to_numpy(dtype=int)
    pb_values = active["powerball"].to_numpy(dtype=int)
    records: list[dict] = []

    for index in range(int(warmup_draws), len(active)):
        _, _, _, _, white_signal = _white_signal(white_values[:index], white_pool, recent_window)
        _, _, _, _, pb_signal = _powerball_signal(pb_values[:index], pb_pool, recent_window)
        white_labels = np.zeros(white_pool, dtype=float)
        white_labels[white_values[index] - 1] = 1.0
        pb_labels = np.zeros(pb_pool, dtype=float)
        pb_labels[pb_values[index] - 1] = 1.0
        records.append(
            {
                "draw_date": active.iloc[index]["draw_date"],
                "white_signal": white_signal,
                "white_labels": white_labels,
                "pb_signal": pb_signal,
                "pb_labels": pb_labels,
            }
        )

    split = max(1, min(len(records) - 1, int(len(records) * float(calibration_fraction))))
    calibration = records[:split]
    holdout = records[split:]
    strength_grid = np.array([0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20])
    blend_grid = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 1.0])
    white_uniform = np.repeat(5 / white_pool, white_pool)
    pb_uniform = np.repeat(1 / pb_pool, pb_pool)

    def best_parameters(rows: list[dict], kind: str) -> tuple[float, float]:
        labels = np.stack([row[f"{kind}_labels"] for row in rows])
        scored: list[tuple[float, float, float]] = []
        for strength in strength_grid:
            if kind == "white":
                raw = np.stack(
                    [
                        conditional_bernoulli_inclusion_probabilities(
                            _softmax(float(strength) * row["white_signal"]), sample_size=5
                        )
                        for row in rows
                    ]
                )
                uniform = white_uniform
            else:
                raw = np.stack([_softmax(float(strength) * row["pb_signal"]) for row in rows])
                uniform = pb_uniform
            for blend in blend_grid:
                probabilities = float(blend) * raw + (1 - float(blend)) * uniform
                loss = float(np.mean((probabilities - labels) ** 2))
                scored.append((loss, float(blend), float(strength)))
        _, best_blend, best_strength = min(scored)
        return best_strength, best_blend

    white_strength, white_calibrated_weight = best_parameters(calibration, "white")
    pb_strength, pb_calibrated_weight = best_parameters(calibration, "pb")

    detail_rows = []
    for row in holdout:
        white_draw_weight = _softmax(white_strength * row["white_signal"])
        white_model_inclusion = conditional_bernoulli_inclusion_probabilities(white_draw_weight, sample_size=5)
        white_inclusion = (
            white_calibrated_weight * white_model_inclusion
            + (1 - white_calibrated_weight) * white_uniform
        )
        actual_white = np.flatnonzero(row["white_labels"] > 0)
        white_order = np.argsort(white_inclusion)[::-1]
        white_top5 = set(white_order[:5])
        white_top10 = set(white_order[:10])
        white_model_subset = conditional_bernoulli_subset_probability(
            white_draw_weight, actual_white, sample_size=5
        )
        white_subset_probability = (
            white_calibrated_weight * white_model_subset
            + (1 - white_calibrated_weight) / comb(white_pool, 5)
        )

        pb_model_prob = _softmax(pb_strength * row["pb_signal"])
        pb_prob = pb_calibrated_weight * pb_model_prob + (1 - pb_calibrated_weight) * pb_uniform
        actual_pb = int(np.argmax(row["pb_labels"]))
        pb_order = np.argsort(pb_prob)[::-1]
        pb_rank = int(np.where(pb_order == actual_pb)[0][0]) + 1

        detail_rows.append(
            {
                "draw_date": row["draw_date"],
                "year": int(pd.Timestamp(row["draw_date"]).year),
                "white_brier_model": _mean_brier(white_inclusion, row["white_labels"]),
                "white_brier_uniform": _mean_brier(white_uniform, row["white_labels"]),
                "white_log_loss_model": float(-np.log(max(white_subset_probability, 1e-15))),
                "white_log_loss_uniform": float(np.log(comb(white_pool, 5))),
                "white_top5_hits": len(set(actual_white).intersection(white_top5)),
                "white_top10_hits": len(set(actual_white).intersection(white_top10)),
                "pb_brier_model": _mean_brier(pb_prob, row["pb_labels"]),
                "pb_brier_uniform": _mean_brier(pb_uniform, row["pb_labels"]),
                "pb_log_loss_model": float(-np.log(max(pb_prob[actual_pb], 1e-12))),
                "pb_log_loss_uniform": float(-np.log(1 / pb_pool)),
                "pb_rank": pb_rank,
                "pb_top3_hit": int(pb_rank <= 3),
                "pb_top5_hit": int(pb_rank <= 5),
            }
        )

    detail = pd.DataFrame(detail_rows)
    metric_specs = [
        ("White Brier", "white_brier_uniform", "white_brier_model", "lower"),
        ("White log-loss", "white_log_loss_uniform", "white_log_loss_model", "lower"),
        ("White hits en top 5", 25 / white_pool, "white_top5_hits", "higher"),
        ("White hits en top 10", 50 / white_pool, "white_top10_hits", "higher"),
        ("Powerball Brier", "pb_brier_uniform", "pb_brier_model", "lower"),
        ("Powerball log-loss", "pb_log_loss_uniform", "pb_log_loss_model", "lower"),
        ("Powerball top 3", 3 / pb_pool, "pb_top3_hit", "higher"),
        ("Powerball top 5", 5 / pb_pool, "pb_top5_hit", "higher"),
    ]
    summary_rows = []
    for metric_index, (metric, uniform_ref, model_col, direction) in enumerate(metric_specs):
        uniform_values = (
            detail[uniform_ref].to_numpy(dtype=float)
            if isinstance(uniform_ref, str)
            else np.repeat(float(uniform_ref), len(detail))
        )
        model_values = detail[model_col].to_numpy(dtype=float)
        improvement_values = uniform_values - model_values if direction == "lower" else model_values - uniform_values
        ci_low, ci_high = _bootstrap_mean_interval(improvement_values, seed=202603 + metric_index)
        mean_improvement = float(improvement_values.mean())
        summary_rows.append(
            {
                "metric": metric,
                "uniform": float(uniform_values.mean()),
                "model": float(model_values.mean()),
                "direction": direction,
                "improvement": mean_improvement,
                "improvement_ci_low": ci_low,
                "improvement_ci_high": ci_high,
                "model_win_rate": float(np.mean(improvement_values > 0)),
                "evidence": _evidence_label(mean_improvement, ci_low),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["delta"] = summary["model"] - summary["uniform"]
    summary["beats_uniform"] = np.where(
        summary["direction"].eq("lower"), summary["model"] < summary["uniform"], summary["model"] > summary["uniform"]
    )
    summary["relative_improvement_pct"] = np.where(
        summary["direction"].eq("lower"),
        (summary["uniform"] - summary["model"]) / summary["uniform"] * 100,
        (summary["model"] - summary["uniform"]) / summary["uniform"] * 100,
    )

    white_brier_row = summary.loc[summary["metric"].eq("White Brier")].iloc[0]
    pb_brier_row = summary.loc[summary["metric"].eq("Powerball Brier")].iloc[0]
    white_evidence = str(white_brier_row["evidence"])
    pb_evidence = str(pb_brier_row["evidence"])
    white_model_weight = _evidence_gate(white_calibrated_weight, white_evidence)
    pb_model_weight = _evidence_gate(pb_calibrated_weight, pb_evidence)

    yearly = (
        detail.groupby("year", as_index=False)
        .agg(
            draws=("draw_date", "size"),
            white_brier_model=("white_brier_model", "mean"),
            white_brier_uniform=("white_brier_uniform", "mean"),
            white_top5_hits=("white_top5_hits", "mean"),
            pb_brier_model=("pb_brier_model", "mean"),
            pb_brier_uniform=("pb_brier_uniform", "mean"),
            pb_top5_rate=("pb_top5_hit", "mean"),
        )
    )
    yearly["white_brier_improvement"] = yearly["white_brier_uniform"] - yearly["white_brier_model"]
    yearly["pb_brier_improvement"] = yearly["pb_brier_uniform"] - yearly["pb_brier_model"]
    config = BacktestConfig(
        white_strength=white_strength,
        powerball_strength=pb_strength,
        white_model_weight=white_model_weight,
        powerball_model_weight=pb_model_weight,
        white_calibrated_weight=white_calibrated_weight,
        powerball_calibrated_weight=pb_calibrated_weight,
        white_evidence=white_evidence,
        powerball_evidence=pb_evidence,
        calibration_draws=len(calibration),
        holdout_draws=len(holdout),
        holdout_start=pd.Timestamp(holdout[0]["draw_date"]),
    )
    return detail, summary, yearly, config


def forecast_pop_intervals(
    df: pd.DataFrame,
    white_strength: float,
    powerball_strength: float,
    white_model_weight: float,
    powerball_model_weight: float,
    repetitions: int = 200,
    block_size: int = 13,
    seed: int = 202604,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Moving-block bootstrap intervals for the final next-draw POP estimates."""
    active = current_matrix_draws(df)
    if active.empty:
        return pd.DataFrame(), pd.DataFrame()
    white_pool = int(active.iloc[-1]["white_pool_max"])
    pb_pool = int(active.iloc[-1]["powerball_pool_max"])
    white_values = active[WHITE_COLS].to_numpy(dtype=int)
    pb_values = active["powerball"].to_numpy(dtype=int)
    draw_count = len(active)
    block = max(1, min(int(block_size), draw_count))
    rng = np.random.default_rng(int(seed))
    white_samples = np.zeros((int(repetitions), white_pool), dtype=float)
    pb_samples = np.zeros((int(repetitions), pb_pool), dtype=float)
    white_uniform = np.repeat(5 / white_pool, white_pool)
    pb_uniform = np.repeat(1 / pb_pool, pb_pool)

    for repetition in range(int(repetitions)):
        sampled_indices: list[int] = []
        while len(sampled_indices) < draw_count:
            start = int(rng.integers(0, draw_count - block + 1))
            sampled_indices.extend(range(start, start + block))
        sampled_indices = sampled_indices[:draw_count]
        _, _, _, _, white_signal = _white_signal(white_values[sampled_indices], white_pool)
        _, _, _, _, pb_signal = _powerball_signal(pb_values[sampled_indices], pb_pool)
        white_weights = _softmax(float(white_strength) * white_signal)
        white_model_pop = conditional_bernoulli_inclusion_probabilities(white_weights, sample_size=5)
        pb_model_pop = _softmax(float(powerball_strength) * pb_signal)
        white_samples[repetition] = (
            float(white_model_weight) * white_model_pop
            + (1 - float(white_model_weight)) * white_uniform
        )
        pb_samples[repetition] = (
            float(powerball_model_weight) * pb_model_pop
            + (1 - float(powerball_model_weight)) * pb_uniform
        )

    white_intervals = pd.DataFrame(
        {
            "number": np.arange(1, white_pool + 1),
            "pop_ci_low": np.quantile(white_samples, 0.025, axis=0),
            "pop_ci_high": np.quantile(white_samples, 0.975, axis=0),
        }
    )
    pb_intervals = pd.DataFrame(
        {
            "number": np.arange(1, pb_pool + 1),
            "pop_ci_low": np.quantile(pb_samples, 0.025, axis=0),
            "pop_ci_high": np.quantile(pb_samples, 0.975, axis=0),
        }
    )
    return white_intervals, pb_intervals
