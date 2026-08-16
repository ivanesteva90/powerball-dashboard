from __future__ import annotations

from dataclasses import dataclass

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
    calibration_draws: int
    holdout_draws: int
    holdout_start: pd.Timestamp


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
    result = pd.DataFrame(
        {
            "number": numbers,
            "observed_active_era": counts.astype(int),
            "long_rate_active": long_rate,
            "recent_rate_52": recent_rate,
            "draws_since_seen": gaps,
            "forecast_signal": signal,
            "draw_weight": draw_weight,
            "lift_vs_uniform_pct": (draw_weight / (1 / pool_size) - 1) * 100,
        }
    )
    result = result.sort_values("forecast_signal", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def build_powerball_forecast(
    df: pd.DataFrame,
    strength: float,
    recent_window: int = DEFAULT_RECENT_WINDOW,
) -> pd.DataFrame:
    active = current_matrix_draws(df)
    if active.empty:
        return pd.DataFrame()
    pool_size = int(active.iloc[-1]["powerball_pool_max"])
    numbers = np.arange(1, pool_size + 1)
    values = active["powerball"].to_numpy(dtype=int)
    counts, long_rate, recent_rate, gaps, signal = _powerball_signal(values, pool_size, recent_window)
    draw_weight = _softmax(float(strength) * signal)
    result = pd.DataFrame(
        {
            "number": numbers,
            "observed_active_era": counts.astype(int),
            "long_rate_active": long_rate,
            "recent_rate_52": recent_rate,
            "draws_since_seen": gaps,
            "forecast_signal": signal,
            "draw_weight": draw_weight,
            "lift_vs_uniform_pct": (draw_weight / (1 / pool_size) - 1) * 100,
        }
    )
    result = result.sort_values("forecast_signal", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def _mean_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


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
        config = BacktestConfig(0.0, 0.0, 0, 0, pd.NaT)
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

    def best_strength(rows: list[dict], signal_key: str, label_key: str, multiplier: int) -> float:
        scored = []
        for strength in strength_grid:
            losses = []
            for row in rows:
                probs = multiplier * _softmax(float(strength) * row[signal_key])
                losses.append(_mean_brier(probs, row[label_key]))
            scored.append((float(np.mean(losses)), float(strength)))
        return min(scored)[1]

    white_strength = best_strength(calibration, "white_signal", "white_labels", 5)
    pb_strength = best_strength(calibration, "pb_signal", "pb_labels", 1)

    detail_rows = []
    for row in holdout:
        white_draw_weight = _softmax(white_strength * row["white_signal"])
        white_inclusion = 5 * white_draw_weight
        white_uniform = np.repeat(5 / white_pool, white_pool)
        actual_white = np.flatnonzero(row["white_labels"] > 0)
        white_top5 = set(np.argsort(row["white_signal"])[-5:])
        white_top10 = set(np.argsort(row["white_signal"])[-10:])

        pb_prob = _softmax(pb_strength * row["pb_signal"])
        pb_uniform = np.repeat(1 / pb_pool, pb_pool)
        actual_pb = int(np.argmax(row["pb_labels"]))
        pb_order = np.argsort(row["pb_signal"])[::-1]
        pb_rank = int(np.where(pb_order == actual_pb)[0][0]) + 1

        detail_rows.append(
            {
                "draw_date": row["draw_date"],
                "year": int(pd.Timestamp(row["draw_date"]).year),
                "white_brier_model": _mean_brier(white_inclusion, row["white_labels"]),
                "white_brier_uniform": _mean_brier(white_uniform, row["white_labels"]),
                "white_log_loss_model": float(-np.mean(np.log(np.clip(white_draw_weight[actual_white], 1e-12, None)))),
                "white_log_loss_uniform": float(-np.log(1 / white_pool)),
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
    summary = pd.DataFrame(
        [
            ("White Brier", detail["white_brier_uniform"].mean(), detail["white_brier_model"].mean(), "lower"),
            ("White log-loss", detail["white_log_loss_uniform"].mean(), detail["white_log_loss_model"].mean(), "lower"),
            ("White hits en top 5", 25 / white_pool, detail["white_top5_hits"].mean(), "higher"),
            ("White hits en top 10", 50 / white_pool, detail["white_top10_hits"].mean(), "higher"),
            ("Powerball Brier", detail["pb_brier_uniform"].mean(), detail["pb_brier_model"].mean(), "lower"),
            ("Powerball log-loss", detail["pb_log_loss_uniform"].mean(), detail["pb_log_loss_model"].mean(), "lower"),
            ("Powerball top 3", 3 / pb_pool, detail["pb_top3_hit"].mean(), "higher"),
            ("Powerball top 5", 5 / pb_pool, detail["pb_top5_hit"].mean(), "higher"),
        ],
        columns=["metric", "uniform", "model", "direction"],
    )
    summary["delta"] = summary["model"] - summary["uniform"]
    summary["beats_uniform"] = np.where(
        summary["direction"].eq("lower"), summary["model"] < summary["uniform"], summary["model"] > summary["uniform"]
    )
    summary["relative_improvement_pct"] = np.where(
        summary["direction"].eq("lower"),
        (summary["uniform"] - summary["model"]) / summary["uniform"] * 100,
        (summary["model"] - summary["uniform"]) / summary["uniform"] * 100,
    )

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
    config = BacktestConfig(
        white_strength=white_strength,
        powerball_strength=pb_strength,
        calibration_draws=len(calibration),
        holdout_draws=len(holdout),
        holdout_start=pd.Timestamp(holdout[0]["draw_date"]),
    )
    return detail, summary, yearly, config
