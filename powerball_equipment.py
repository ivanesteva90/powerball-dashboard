from __future__ import annotations

import io
import re
from collections import defaultdict
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy.stats import norm


POWERBALL_PRETEST_PDF_URL = "https://cdn.powerball.com/v01/media/powerball-pre-test.pdf"
WHITE_COLS = ["num1", "num2", "num3", "num4", "num5"]
EQUIPMENT_WHITE_COLS = ["white_1", "white_2", "white_3", "white_4", "white_5"]


def _matrix_for_date(draw_date: pd.Timestamp) -> tuple[int, int]:
    if draw_date <= pd.Timestamp("2009-01-03"):
        return 55, 42
    if draw_date <= pd.Timestamp("2012-01-14"):
        return 59, 39
    if draw_date <= pd.Timestamp("2015-10-03"):
        return 59, 35
    return 69, 26


def download_pretest_pdf(
    url: str = POWERBALL_PRETEST_PDF_URL,
    timeout_sec: int = 45,
) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 PowerballAnalytics/1.0"})
    with urlopen(request, timeout=timeout_sec) as response:
        payload = response.read()
    if not payload.startswith(b"%PDF"):
        raise ValueError("Powerball pre-test source did not return a PDF.")
    return payload


def _canonical_draw_type(raw_type: str) -> str:
    normalized = raw_type.strip().lower()
    if normalized.startswith("pre-test"):
        return "Pre-test"
    if normalized.startswith("post-test") or normalized.startswith("posttest"):
        return "Post-test"
    if normalized.startswith("draw"):
        return "Draw"
    return raw_type.strip()


def _nullable_int(token: str) -> int | None:
    cleaned = token.strip()
    if cleaned in {"-", "--", "---", "N/A"}:
        return None
    return int(cleaned)


def parse_pretest_text(text: str) -> pd.DataFrame:
    """Parse text extracted from MUSL's Powerball pre-test PDF."""
    records: list[dict] = []
    date_pattern = re.compile(r"^\d{2}/\d{2}/\d{2}$")
    value_pattern = re.compile(r"^(?:\d{1,2}|-+|N/A)$")

    last_date: pd.Timestamp | None = None
    for raw_line in text.splitlines():
        tokens = raw_line.replace("\u00a0", " ").split()
        if not tokens or not date_pattern.match(tokens[0]):
            continue

        value_tokens: list[str] = []
        type_tokens: list[str] = []
        for token in tokens[1:]:
            if not type_tokens and value_pattern.match(token):
                value_tokens.append(token)
            else:
                type_tokens.append(token)

        # Some PDF rows lose the Power Play dash during text extraction.
        if len(value_tokens) == 10:
            value_tokens.insert(8, "--")
        if len(value_tokens) != 11 or not type_tokens:
            continue

        values = [_nullable_int(token) for token in value_tokens]
        raw_type = " ".join(type_tokens).strip()
        month, day, short_year = (int(part) for part in tokens[0].split("/"))
        parsed_date = pd.Timestamp(year=2000 + short_year, month=month, day=day)
        # Repair an obvious PDF typo when a row jumps years but repeats the prior month/day.
        if (
            last_date is not None
            and parsed_date > last_date + pd.Timedelta(days=370)
            and parsed_date.month == last_date.month
            and parsed_date.day == last_date.day
        ):
            parsed_date = last_date
        last_date = parsed_date

        records.append(
            {
                "draw_date": parsed_date,
                "white_1": values[0],
                "white_2": values[1],
                "white_3": values[2],
                "white_4": values[3],
                "white_5": values[4],
                "white_machine_id": values[5],
                "white_ball_set_id": values[6],
                "powerball": values[7],
                "power_play": values[8],
                "powerball_machine_id": values[9],
                "powerball_ball_set_id": values[10],
                "draw_type": _canonical_draw_type(raw_type),
                "draw_type_raw": raw_type,
            }
        )

    if not records:
        raise ValueError("No Powerball pre-test rows were detected in the extracted text.")

    result = pd.DataFrame(records).dropna(subset=["draw_date", *EQUIPMENT_WHITE_COLS, "powerball"])
    integer_cols = [
        *EQUIPMENT_WHITE_COLS,
        "white_machine_id",
        "white_ball_set_id",
        "powerball",
        "power_play",
        "powerball_machine_id",
        "powerball_ball_set_id",
    ]
    for column in integer_cols:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")

    matrix = result["draw_date"].apply(_matrix_for_date)
    result["white_pool_max"] = matrix.apply(lambda item: item[0]).astype(int)
    result["powerball_pool_max"] = matrix.apply(lambda item: item[1]).astype(int)
    result = result.sort_values(["draw_date", "draw_type"]).reset_index(drop=True)
    result["test_sequence"] = (
        result.groupby(["draw_date", "draw_type"], sort=False).cumcount() + 1
    ).astype(int)
    return result


def parse_pretest_pdf_bytes(pdf_bytes: bytes) -> pd.DataFrame:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf to read the Powerball pre-test report.") from exc

    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return parse_pretest_text(text)


def join_winning_draws_with_equipment(
    winning_draws: pd.DataFrame,
    equipment_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join official draw equipment to winning rows and audit number agreement."""
    draw_rows = equipment_rows[equipment_rows["draw_type"].eq("Draw")].copy()
    draw_rows = draw_rows.sort_values("draw_date").drop_duplicates("draw_date", keep="last")
    equipment_columns = [
        "draw_date",
        *EQUIPMENT_WHITE_COLS,
        "white_machine_id",
        "white_ball_set_id",
        "powerball",
        "powerball_machine_id",
        "powerball_ball_set_id",
        "draw_type_raw",
    ]
    equipment_draws = draw_rows[equipment_columns].rename(columns={"powerball": "equipment_powerball"})
    merged = winning_draws.merge(equipment_draws, on="draw_date", how="left")

    winning_sets = merged[WHITE_COLS].apply(
        lambda row: tuple(sorted(int(value) for value in row)) if row.notna().all() else tuple(), axis=1
    )
    equipment_sets = merged[EQUIPMENT_WHITE_COLS].apply(
        lambda row: tuple(sorted(int(value) for value in row)) if row.notna().all() else tuple(), axis=1
    )
    merged["equipment_available"] = merged["white_machine_id"].notna()
    merged["white_numbers_match"] = merged["equipment_available"] & winning_sets.eq(equipment_sets)
    merged["powerball_matches"] = (
        merged["equipment_available"]
        & merged["powerball"].astype("Int64").eq(merged["equipment_powerball"].astype("Int64"))
    )

    available = merged["equipment_available"]
    quality = pd.DataFrame(
        {
            "metric": [
                "winning_draws",
                "equipment_matches_by_date",
                "coverage_pct",
                "white_number_mismatches",
                "powerball_mismatches",
                "latest_equipment_date",
            ],
            "value": [
                len(merged),
                int(available.sum()),
                float(available.mean() * 100),
                int((available & ~merged["white_numbers_match"]).sum()),
                int((available & ~merged["powerball_matches"]).sum()),
                draw_rows["draw_date"].max(),
            ],
        }
    )
    return merged, quality


def equipment_usage_summary(equipment_rows: pd.DataFrame) -> pd.DataFrame:
    draws = equipment_rows[equipment_rows["draw_type"].eq("Draw")].copy()
    specs = [
        ("White machine", "white_machine_id"),
        ("White set", "white_ball_set_id"),
        ("Powerball machine", "powerball_machine_id"),
        ("Powerball set", "powerball_ball_set_id"),
    ]
    frames = []
    for label, column in specs:
        grouped = (
            draws.dropna(subset=[column])
            .groupby(column, as_index=False)
            .agg(draws=("draw_date", "size"), first_draw=("draw_date", "min"), last_draw=("draw_date", "max"))
            .rename(columns={column: "equipment_id"})
        )
        grouped.insert(0, "equipment_type", label)
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(len(values), np.nan)
    valid_indices = np.flatnonzero(np.isfinite(values))
    if len(valid_indices) == 0:
        return result
    valid = values[valid_indices]
    order = np.argsort(valid)
    ranked = valid[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    result[valid_indices] = restored
    return result


def equipment_number_deviation(
    equipment_rows: pd.DataFrame,
    ball_type: str,
    group_column: str,
    min_draws: int = 25,
) -> pd.DataFrame:
    """Observed-vs-uniform deviations for each equipment group and number."""
    if group_column not in equipment_rows.columns:
        raise ValueError(f"Unknown equipment column: {group_column}")
    draws = equipment_rows[equipment_rows["draw_type"].eq("Draw")].dropna(subset=[group_column]).copy()
    records: list[dict] = []

    for equipment_id, group in draws.groupby(group_column):
        if len(group) < int(min_draws):
            continue
        if ball_type == "white":
            pool_size = int(group["white_pool_max"].max())
            values = group[EQUIPMENT_WHITE_COLS].to_numpy(dtype=int)
            counts = np.bincount(values.ravel(), minlength=pool_size + 1)[1 : pool_size + 1]
            for number in range(1, pool_size + 1):
                probabilities = np.where(
                    group["white_pool_max"].to_numpy(dtype=int) >= number,
                    5 / group["white_pool_max"].to_numpy(dtype=float),
                    0.0,
                )
                expected = float(probabilities.sum())
                variance = float((probabilities * (1 - probabilities)).sum())
                observed = int(counts[number - 1])
                z_score = (observed - expected) / np.sqrt(variance) if variance > 0 else np.nan
                records.append(
                    {
                        "equipment_id": int(equipment_id),
                        "number": number,
                        "draws": len(group),
                        "observed": observed,
                        "expected": expected,
                        "z_score": z_score,
                    }
                )
        elif ball_type == "powerball":
            pool_size = int(group["powerball_pool_max"].max())
            values = group["powerball"].to_numpy(dtype=int)
            counts = np.bincount(values, minlength=pool_size + 1)[1 : pool_size + 1]
            for number in range(1, pool_size + 1):
                probabilities = np.where(
                    group["powerball_pool_max"].to_numpy(dtype=int) >= number,
                    1 / group["powerball_pool_max"].to_numpy(dtype=float),
                    0.0,
                )
                expected = float(probabilities.sum())
                variance = float((probabilities * (1 - probabilities)).sum())
                observed = int(counts[number - 1])
                z_score = (observed - expected) / np.sqrt(variance) if variance > 0 else np.nan
                records.append(
                    {
                        "equipment_id": int(equipment_id),
                        "number": number,
                        "draws": len(group),
                        "observed": observed,
                        "expected": expected,
                        "z_score": z_score,
                    }
                )
        else:
            raise ValueError("ball_type must be 'white' or 'powerball'.")

    result = pd.DataFrame(records)
    if result.empty:
        return result
    result["p_value_two_sided"] = 2 * norm.sf(result["z_score"].abs())
    result["q_value_fdr"] = _benjamini_hochberg(result["p_value_two_sided"].to_numpy())
    result["is_fdr_5pct"] = result["q_value_fdr"] < 0.05
    result["delta"] = result["observed"] - result["expected"]
    return result.sort_values("z_score", ascending=False).reset_index(drop=True)


def pretest_draw_comparison(equipment_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for draw_date, day in equipment_rows.groupby("draw_date", sort=True):
        draws = day[day["draw_type"].eq("Draw")]
        pretests = day[day["draw_type"].eq("Pre-test")]
        if draws.empty or pretests.empty:
            continue
        draw = draws.iloc[-1]
        actual_white = set(int(draw[column]) for column in EQUIPMENT_WHITE_COLS)
        pretest_white = set(int(value) for value in pretests[EQUIPMENT_WHITE_COLS].to_numpy().ravel())
        pretest_pb = set(pretests["powerball"].dropna().astype(int))
        test_count = len(pretests)
        white_pool = int(draw["white_pool_max"])
        pb_pool = int(draw["powerball_pool_max"])
        records.append(
            {
                "draw_date": draw_date,
                "pretests": test_count,
                "white_draw_numbers_seen_in_pretests": len(actual_white & pretest_white),
                "white_expected_seen": 5 * (1 - (1 - 5 / white_pool) ** test_count),
                "powerball_seen_in_pretests": int(int(draw["powerball"]) in pretest_pb),
                "powerball_expected_seen": 1 - (1 - 1 / pb_pool) ** test_count,
                "white_machine_id": int(draw["white_machine_id"]),
                "white_ball_set_id": int(draw["white_ball_set_id"]),
                "powerball_machine_id": int(draw["powerball_machine_id"]),
                "powerball_ball_set_id": int(draw["powerball_ball_set_id"]),
            }
        )
    return pd.DataFrame(records)


def _bootstrap_interval(values: np.ndarray, seed: int = 202608, repetitions: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(repetitions), len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def equipment_walk_forward_backtest(
    equipment_rows: pd.DataFrame,
    warmup_draws: int = 104,
    prior_draws: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retrospective walk-forward test using only prior rows for each machine and set."""
    draws = equipment_rows[equipment_rows["draw_type"].eq("Draw")].sort_values("draw_date").copy()
    if draws.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    latest = draws.iloc[-1]
    draws = draws[
        draws["white_pool_max"].eq(int(latest["white_pool_max"]))
        & draws["powerball_pool_max"].eq(int(latest["powerball_pool_max"]))
    ].reset_index(drop=True)
    if len(draws) <= warmup_draws:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    white_pool = int(draws.iloc[-1]["white_pool_max"])
    pb_pool = int(draws.iloc[-1]["powerball_pool_max"])
    white_uniform = np.repeat(5 / white_pool, white_pool)
    pb_uniform = np.repeat(1 / pb_pool, pb_pool)

    white_counts = {
        "machine": defaultdict(lambda: np.zeros(white_pool, dtype=float)),
        "set": defaultdict(lambda: np.zeros(white_pool, dtype=float)),
    }
    pb_counts = {
        "machine": defaultdict(lambda: np.zeros(pb_pool, dtype=float)),
        "set": defaultdict(lambda: np.zeros(pb_pool, dtype=float)),
    }
    group_draws = {
        "white_machine": defaultdict(int),
        "white_set": defaultdict(int),
        "pb_machine": defaultdict(int),
        "pb_set": defaultdict(int),
    }

    def smoothed(counts: np.ndarray, count: int, baseline: np.ndarray) -> np.ndarray:
        return (counts + float(prior_draws) * baseline) / (count + float(prior_draws))

    records: list[dict] = []
    for index, row in draws.iterrows():
        wm, ws = int(row["white_machine_id"]), int(row["white_ball_set_id"])
        pm, ps = int(row["powerball_machine_id"]), int(row["powerball_ball_set_id"])
        actual_white = np.zeros(white_pool, dtype=float)
        actual_white[row[EQUIPMENT_WHITE_COLS].to_numpy(dtype=int) - 1] = 1.0
        actual_pb = np.zeros(pb_pool, dtype=float)
        actual_pb[int(row["powerball"]) - 1] = 1.0

        if index >= int(warmup_draws):
            white_machine_prob = smoothed(
                white_counts["machine"][wm], group_draws["white_machine"][wm], white_uniform
            )
            white_set_prob = smoothed(
                white_counts["set"][ws], group_draws["white_set"][ws], white_uniform
            )
            pb_machine_prob = smoothed(
                pb_counts["machine"][pm], group_draws["pb_machine"][pm], pb_uniform
            )
            pb_set_prob = smoothed(pb_counts["set"][ps], group_draws["pb_set"][ps], pb_uniform)
            white_model = (white_machine_prob + white_set_prob) / 2
            pb_model = (pb_machine_prob + pb_set_prob) / 2
            records.append(
                {
                    "draw_date": row["draw_date"],
                    "year": int(pd.Timestamp(row["draw_date"]).year),
                    "white_machine_id": wm,
                    "white_ball_set_id": ws,
                    "powerball_machine_id": pm,
                    "powerball_ball_set_id": ps,
                    "white_brier_equipment": float(np.mean((white_model - actual_white) ** 2)),
                    "white_brier_uniform": float(np.mean((white_uniform - actual_white) ** 2)),
                    "pb_brier_equipment": float(np.mean((pb_model - actual_pb) ** 2)),
                    "pb_brier_uniform": float(np.mean((pb_uniform - actual_pb) ** 2)),
                }
            )

        white_indices = row[EQUIPMENT_WHITE_COLS].to_numpy(dtype=int) - 1
        white_counts["machine"][wm][white_indices] += 1
        white_counts["set"][ws][white_indices] += 1
        pb_index = int(row["powerball"]) - 1
        pb_counts["machine"][pm][pb_index] += 1
        pb_counts["set"][ps][pb_index] += 1
        group_draws["white_machine"][wm] += 1
        group_draws["white_set"][ws] += 1
        group_draws["pb_machine"][pm] += 1
        group_draws["pb_set"][ps] += 1

    detail = pd.DataFrame(records)
    if detail.empty:
        return detail, pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    for index, (label, uniform_col, model_col) in enumerate(
        [
            ("White: máquina + set", "white_brier_uniform", "white_brier_equipment"),
            ("Powerball: máquina + set", "pb_brier_uniform", "pb_brier_equipment"),
        ]
    ):
        improvements = detail[uniform_col].to_numpy() - detail[model_col].to_numpy()
        ci_low, ci_high = _bootstrap_interval(improvements, seed=202608 + index)
        mean_improvement = float(improvements.mean())
        evidence = "Sin mejora"
        if mean_improvement > 0:
            evidence = "Evidencia de mejora" if ci_low > 0 else "Mejora incierta"
        summary_rows.append(
            {
                "metric": label,
                "uniform_brier": float(detail[uniform_col].mean()),
                "equipment_brier": float(detail[model_col].mean()),
                "improvement": mean_improvement,
                "improvement_ci_low": ci_low,
                "improvement_ci_high": ci_high,
                "model_win_rate": float(np.mean(improvements > 0)),
                "evidence": evidence,
            }
        )
    summary = pd.DataFrame(summary_rows)
    yearly = (
        detail.groupby("year", as_index=False)
        .agg(
            draws=("draw_date", "size"),
            white_brier_equipment=("white_brier_equipment", "mean"),
            white_brier_uniform=("white_brier_uniform", "mean"),
            pb_brier_equipment=("pb_brier_equipment", "mean"),
            pb_brier_uniform=("pb_brier_uniform", "mean"),
        )
    )
    yearly["white_improvement"] = yearly["white_brier_uniform"] - yearly["white_brier_equipment"]
    yearly["pb_improvement"] = yearly["pb_brier_uniform"] - yearly["pb_brier_equipment"]
    return detail, summary, yearly


def build_hypothetical_weight_table(
    max_number: int,
    target_numbers: list[int],
    nominal_weight_g: float = 80.0,
    relative_delta_pct: float = -0.375,
) -> pd.DataFrame:
    numbers = np.arange(1, int(max_number) + 1)
    weights = np.repeat(float(nominal_weight_g), len(numbers))
    target_mask = np.isin(numbers, np.asarray(target_numbers, dtype=int))
    weights[target_mask] *= 1 + float(relative_delta_pct) / 100
    return pd.DataFrame(
        {
            "number": numbers,
            "weight": weights,
            "weight_delta_pct": (weights / float(nominal_weight_g) - 1) * 100,
            "weight_source": "escenario hipotético",
        }
    )
