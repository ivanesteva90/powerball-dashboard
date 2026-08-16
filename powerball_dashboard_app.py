import io
from collections import Counter
from datetime import datetime, timedelta
from itertools import combinations
from math import comb
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from powerball_core import (
    build_powerball_forecast,
    build_white_forecast,
    conditional_bernoulli_normalizer,
    current_matrix_draws,
    forecast_pop_intervals,
    sample_conditional_bernoulli,
    walk_forward_backtest,
)
from powerball_equipment import (
    POWERBALL_PRETEST_PDF_URL,
    build_hypothetical_weight_table,
    download_pretest_pdf,
    equipment_number_deviation,
    equipment_usage_summary,
    equipment_walk_forward_backtest,
    join_winning_draws_with_equipment,
    parse_pretest_pdf_bytes,
    pretest_draw_comparison,
)

try:
    from scipy.stats import chi2, norm
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import openpyxl  # noqa: F401
    OPENPYXL_OK = True
except Exception:
    OPENPYXL_OK = False

st.set_page_config(page_title="Powerball Analytics Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1500px;}
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff 0%, #f4f7fb 100%);
        border: 1px solid #dce3ec;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 8px 24px rgba(24, 39, 75, 0.05);
    }
    [data-testid="stSidebar"] {background: #f4f7fb;}
    .accuracy-note {
        border-left: 4px solid #e63946;
        background: #fff5f5;
        padding: 0.8rem 1rem;
        border-radius: 0 10px 10px 0;
        margin: 0.5rem 0 1.1rem 0;
    }
    .data-ok {
        border-left: 4px solid #16855b;
        background: #f0fbf6;
        padding: 0.8rem 1rem;
        border-radius: 0 10px 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

WHITE_COLS = ["num1", "num2", "num3", "num4", "num5"]
TEXAS_POWERBALL_CSV_URL = "https://www.texaslottery.com/export/sites/lottery/Games/Powerball/Winning_Numbers/powerball.csv"
PRETEST_LOCAL_FILENAME = "powerball_pretests.csv"
POWERBALL_DRAW_WEEKDAYS = {0, 2, 5}  # Monday, Wednesday, Saturday (Texas Lottery schedule)


def infer_matrix(draw_date: pd.Timestamp) -> tuple[int, int, str]:
    """Infer the game matrix from the observed historical file structure.
    This matches the uploaded Texas Lottery dataset periods.
    """
    if draw_date <= pd.Timestamp("2012-01-14"):
        return 59, 39, "2010-2011 | 5/59 + PB39"
    if draw_date <= pd.Timestamp("2015-10-03"):
        return 59, 35, "2012-2015 | 5/59 + PB35"
    return 69, 26, "2015-presente | 5/69 + PB26"


@st.cache_data(show_spinner=False)
def parse_powerball_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    raw = raw_bytes.decode("utf-8-sig", errors="replace")
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) not in (10, 11):
            continue
        # If a header ever appears, skip it.
        if parts[0].lower() in {"game", "game name", "game_name"}:
            continue

        if len(parts) == 11:
            game_name, month, day, year, n1, n2, n3, n4, n5, powerball, power_play = parts
        else:
            game_name, month, day, year, n1, n2, n3, n4, n5, powerball = parts
            power_play = np.nan

        rows.append(
            {
                "game_name": game_name,
                "month": month,
                "day": day,
                "year": year,
                "num1": n1,
                "num2": n2,
                "num3": n3,
                "num4": n4,
                "num5": n5,
                "powerball": powerball,
                "power_play": power_play,
            }
        )

    if not rows:
        raise ValueError("No valid Powerball rows were detected in the uploaded CSV.")

    df = pd.DataFrame(rows)
    numeric_cols = ["month", "day", "year", *WHITE_COLS, "powerball", "power_play"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["draw_date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"]), errors="coerce"
    )
    df = df.dropna(subset=["draw_date", *WHITE_COLS, "powerball"]).copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values("draw_date").reset_index(drop=True)

    matrix_info = df["draw_date"].apply(infer_matrix)
    df["white_pool_max"] = matrix_info.apply(lambda x: x[0])
    df["powerball_pool_max"] = matrix_info.apply(lambda x: x[1])
    df["era"] = matrix_info.apply(lambda x: x[2])

    df["weekday"] = df["draw_date"].dt.day_name()
    df["white_sorted"] = df[WHITE_COLS].apply(lambda r: tuple(sorted(int(v) for v in r)), axis=1)
    df["white_sum"] = df[WHITE_COLS].sum(axis=1)
    df["white_min"] = df[WHITE_COLS].min(axis=1)
    df["white_max"] = df[WHITE_COLS].max(axis=1)
    df["white_range"] = df["white_max"] - df["white_min"]
    df["odd_count"] = df[WHITE_COLS].apply(lambda r: sum(int(v) % 2 for v in r), axis=1)
    df["even_count"] = 5 - df["odd_count"]
    # Era-aware low/high split per draw
    df["low_count"] = df.apply(lambda r: sum(int(v) <= (r["white_pool_max"] // 2) for v in [r[c] for c in WHITE_COLS]), axis=1)
    df["high_count"] = 5 - df["low_count"]
    df["consecutive_pairs"] = df["white_sorted"].apply(
        lambda nums: sum(1 for a, b in zip(nums[:-1], nums[1:]) if b - a == 1)
    )
    prev_sets = [set()] + [set(nums) for nums in df["white_sorted"].iloc[:-1]]
    curr_sets = [set(nums) for nums in df["white_sorted"]]
    df["repeat_from_prev_draw"] = [len(c & p) for c, p in zip(curr_sets, prev_sets)]
    return df


@st.cache_data(show_spinner=False)
def load_default_sample(file_mtime_ns: int | None = None, file_size: int | None = None) -> pd.DataFrame | None:
    # file_mtime_ns and file_size are used only to invalidate cache when local CSV changes.
    sample_path = Path(__file__).with_name("powerball.csv")
    if sample_path.exists():
        return parse_powerball_csv_bytes(sample_path.read_bytes())
    return None


def explode_white(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["draw_date", "era", "weekday", "white_pool_max", *WHITE_COLS]].melt(
        id_vars=["draw_date", "era", "weekday", "white_pool_max"],
        value_vars=WHITE_COLS,
        var_name="slot",
        value_name="white_ball",
    )
    out["white_ball"] = out["white_ball"].astype(int)
    return out.sort_values("draw_date").reset_index(drop=True)


def white_counts(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df[WHITE_COLS].to_numpy().ravel()).astype(int).value_counts().sort_index()


def mixed_expected_white(df: pd.DataFrame, max_number: int | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["number", "observed", "expected", "variance", "z_score"])
    if max_number is None:
        max_number = int(df["white_pool_max"].max())
    observed = white_counts(df).reindex(range(1, max_number + 1), fill_value=0)
    expected, variance = [], []
    for n in range(1, max_number + 1):
        p = np.where(df["white_pool_max"].to_numpy() >= n, 5 / df["white_pool_max"].to_numpy(), 0.0)
        expected.append(p.sum())
        variance.append((p * (1 - p)).sum())
    out = pd.DataFrame(
        {
            "number": range(1, max_number + 1),
            "observed": observed.values,
            "expected": expected,
            "variance": variance,
        }
    )
    std = np.sqrt(out["variance"].clip(lower=0))
    out["expected_ci_low"] = (out["expected"] - 1.96 * std).clip(lower=0)
    out["expected_ci_high"] = out["expected"] + 1.96 * std
    out["z_score"] = np.where(out["variance"] > 0, (out["observed"] - out["expected"]) / np.sqrt(out["variance"]), np.nan)
    out["delta"] = out["observed"] - out["expected"]
    return out


def mixed_expected_powerball(df: pd.DataFrame, max_number: int | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["number", "observed", "expected", "variance", "z_score"])
    if max_number is None:
        max_number = int(df["powerball_pool_max"].max())
    observed = df["powerball"].astype(int).value_counts().sort_index().reindex(range(1, max_number + 1), fill_value=0)
    expected, variance = [], []
    for n in range(1, max_number + 1):
        p = np.where(df["powerball_pool_max"].to_numpy() >= n, 1 / df["powerball_pool_max"].to_numpy(), 0.0)
        expected.append(p.sum())
        variance.append((p * (1 - p)).sum())
    out = pd.DataFrame(
        {
            "number": range(1, max_number + 1),
            "observed": observed.values,
            "expected": expected,
            "variance": variance,
        }
    )
    std = np.sqrt(out["variance"].clip(lower=0))
    out["expected_ci_low"] = (out["expected"] - 1.96 * std).clip(lower=0)
    out["expected_ci_high"] = out["expected"] + 1.96 * std
    out["z_score"] = np.where(out["variance"] > 0, (out["observed"] - out["expected"]) / np.sqrt(out["variance"]), np.nan)
    out["delta"] = out["observed"] - out["expected"]
    return out


def chi_square_from_expected(expected_df: pd.DataFrame) -> dict:
    valid = expected_df[expected_df["expected"] > 0].copy()
    if valid.empty:
        return {"chi2": np.nan, "df": np.nan, "p_value": np.nan}
    chi2_stat = float((((valid["observed"] - valid["expected"]) ** 2) / valid["expected"]).sum())
    dof = max(len(valid) - 1, 1)
    if SCIPY_OK:
        p_value = float(chi2.sf(chi2_stat, dof))
    else:
        p_value = np.nan
    return {"chi2": chi2_stat, "df": dof, "p_value": p_value}


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    valid = p_values.notna()
    if not valid.any():
        return pd.Series(np.nan, index=p_values.index)
    pv = p_values[valid].clip(lower=0, upper=1)
    m = len(pv)
    order = np.argsort(pv.to_numpy())
    ranked = pv.to_numpy()[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    restored = np.empty(m)
    restored[order] = q
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    out.loc[pv.index] = restored
    return out


def add_significance_columns(expected_df: pd.DataFrame) -> pd.DataFrame:
    out = expected_df.copy()
    if out.empty or "z_score" not in out.columns:
        out["p_value_two_sided"] = np.nan
        out["q_value_fdr"] = np.nan
        out["is_fdr_5pct"] = False
        return out
    if SCIPY_OK:
        out["p_value_two_sided"] = 2 * norm.sf(np.abs(out["z_score"]))
    else:
        out["p_value_two_sided"] = np.nan
    out["q_value_fdr"] = benjamini_hochberg(out["p_value_two_sided"])
    out["is_fdr_5pct"] = out["q_value_fdr"] <= 0.05
    return out


def draw_quality_report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        summary = pd.DataFrame(columns=["check", "rows_flagged", "pct_rows"])
        detail = pd.DataFrame()
        return summary, detail

    detail = df[["draw_date", *WHITE_COLS, "powerball", "white_pool_max", "powerball_pool_max", "era"]].copy()
    detail["duplicate_white_in_draw"] = detail[WHITE_COLS].apply(lambda r: len(set(int(v) for v in r)) < 5, axis=1)
    detail["white_out_of_range"] = detail.apply(
        lambda r: any((int(r[c]) < 1) or (int(r[c]) > int(r["white_pool_max"])) for c in WHITE_COLS),
        axis=1,
    )
    detail["powerball_out_of_range"] = detail.apply(
        lambda r: (int(r["powerball"]) < 1) or (int(r["powerball"]) > int(r["powerball_pool_max"])),
        axis=1,
    )
    detail["duplicate_draw_date"] = detail["draw_date"].duplicated(keep=False)
    detail["any_issue"] = detail[
        ["duplicate_white_in_draw", "white_out_of_range", "powerball_out_of_range", "duplicate_draw_date"]
    ].any(axis=1)

    total = len(detail)
    summary = pd.DataFrame(
        [
            ("Duplicate white numbers in same draw", int(detail["duplicate_white_in_draw"].sum())),
            ("Duplicate draw date", int(detail["duplicate_draw_date"].sum())),
            ("White ball outside era range", int(detail["white_out_of_range"].sum())),
            ("Powerball outside era range", int(detail["powerball_out_of_range"].sum())),
            ("Any issue", int(detail["any_issue"].sum())),
        ],
        columns=["check", "rows_flagged"],
    )
    summary["pct_rows"] = np.where(total > 0, 100 * summary["rows_flagged"] / total, 0.0)
    issue_rows = detail[detail["any_issue"]].copy()
    return summary, issue_rows


def pair_cooccurrence_matrix(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    wc = white_counts(df).sort_values(ascending=False)
    top_numbers = sorted(wc.head(top_n).index.tolist())
    matrix = pd.DataFrame(0, index=top_numbers, columns=top_numbers, dtype=int)
    for nums in df["white_sorted"]:
        in_top = [n for n in nums if n in matrix.index]
        for a, b in combinations(in_top, 2):
            matrix.loc[a, b] += 1
            matrix.loc[b, a] += 1
        for a in in_top:
            matrix.loc[a, a] += 1
    return matrix


def texas_now_ct() -> datetime:
    return datetime.now(ZoneInfo("America/Chicago"))


def is_powerball_draw_day_ct(now_ct: datetime | None = None) -> bool:
    if now_ct is None:
        now_ct = texas_now_ct()
    return now_ct.weekday() in POWERBALL_DRAW_WEEKDAYS


def next_powerball_draw_day_ct(now_ct: datetime | None = None) -> datetime:
    if now_ct is None:
        now_ct = texas_now_ct()
    for i in range(0, 8):
        candidate = now_ct + timedelta(days=i)
        if candidate.weekday() in POWERBALL_DRAW_WEEKDAYS:
            return candidate
    return now_ct


def download_texas_powerball_csv(url: str = TEXAS_POWERBALL_CSV_URL, timeout_sec: int = 25) -> bytes:
    request = Request(url, headers={"User-Agent": "powerball-dashboard/1.0"})
    with urlopen(request, timeout=timeout_sec) as response:
        return response.read()


@st.cache_data(ttl=21600, show_spinner=False)
def load_official_powerball() -> pd.DataFrame:
    return parse_powerball_csv_bytes(download_texas_powerball_csv())


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_official_pretests() -> pd.DataFrame:
    return parse_pretest_pdf_bytes(download_pretest_pdf())


@st.cache_data(show_spinner=False)
def load_local_pretests(file_mtime_ns: int | None = None, file_size: int | None = None) -> pd.DataFrame:
    del file_mtime_ns, file_size
    path = Path(__file__).with_name(PRETEST_LOCAL_FILENAME)
    if not path.exists():
        return pd.DataFrame()
    result = pd.read_csv(path, parse_dates=["draw_date"])
    integer_columns = [
        "white_1",
        "white_2",
        "white_3",
        "white_4",
        "white_5",
        "white_machine_id",
        "white_ball_set_id",
        "powerball",
        "power_play",
        "powerball_machine_id",
        "powerball_ball_set_id",
    ]
    for column in integer_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    return result


def overdue_white(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["number", "draws_since_seen", "last_seen"])
    df = df.reset_index(drop=True)
    max_number = int(df["white_pool_max"].max())
    rows = []
    for n in range(1, max_number + 1):
        mask = df[WHITE_COLS].isin([n]).any(axis=1)
        hit_idx = np.where(mask.to_numpy())[0]
        if len(hit_idx):
            last_idx = int(hit_idx.max())
            rows.append((n, len(df) - 1 - last_idx, df.loc[last_idx, "draw_date"]))
        else:
            rows.append((n, len(df), pd.NaT))
    return pd.DataFrame(rows, columns=["number", "draws_since_seen", "last_seen"]).sort_values(
        ["draws_since_seen", "number"], ascending=[False, True]
    )


def overdue_powerball(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["number", "draws_since_seen", "last_seen"])
    df = df.reset_index(drop=True)
    max_number = int(df["powerball_pool_max"].max())
    rows = []
    for n in range(1, max_number + 1):
        mask = df["powerball"].eq(n)
        hit_idx = np.where(mask.to_numpy())[0]
        if len(hit_idx):
            last_idx = int(hit_idx.max())
            rows.append((n, len(df) - 1 - last_idx, df.loc[last_idx, "draw_date"]))
        else:
            rows.append((n, len(df), pd.NaT))
    return pd.DataFrame(rows, columns=["number", "draws_since_seen", "last_seen"]).sort_values(
        ["draws_since_seen", "number"], ascending=[False, True]
    )


def rolling_hits_white(df: pd.DataFrame, number: int, window: int = 52) -> pd.DataFrame:
    hit = df[WHITE_COLS].isin([number]).any(axis=1).astype(int)
    out = df[["draw_date"]].copy()
    out["hit"] = hit
    out["rolling_hits"] = out["hit"].rolling(window=window, min_periods=1).sum()
    return out


def pair_frequency(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    counter = Counter()
    for nums in df["white_sorted"]:
        for pair in combinations(nums, 2):
            counter[pair] += 1
    pool_counts = df["white_pool_max"].value_counts().to_dict()
    rows = []
    for (a, b), observed in counter.items():
        expected = 0.0
        variance = 0.0
        for pool_size, draw_count in pool_counts.items():
            if int(pool_size) < max(a, b):
                continue
            probability = 20 / (int(pool_size) * (int(pool_size) - 1))
            expected += draw_count * probability
            variance += draw_count * probability * (1 - probability)
        z_score = (observed - expected) / np.sqrt(variance) if variance > 0 else np.nan
        lift = (observed / expected - 1) * 100 if expected > 0 else np.nan
        rows.append((f"{a}-{b}", observed, expected, lift, z_score))
    result = pd.DataFrame(rows, columns=["pair", "count", "expected", "lift_vs_expected_pct", "z_score"])
    return result.sort_values(["z_score", "count"], ascending=[False, False]).head(top_n).reset_index(drop=True)


def triplet_frequency(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    counter = Counter()
    for nums in df["white_sorted"]:
        for tri in combinations(nums, 3):
            counter[tri] += 1
    pool_counts = df["white_pool_max"].value_counts().to_dict()
    rows = []
    for (a, b, c), observed in counter.items():
        expected = 0.0
        variance = 0.0
        for pool_size, draw_count in pool_counts.items():
            pool_size = int(pool_size)
            if pool_size < max(a, b, c):
                continue
            probability = 60 / (pool_size * (pool_size - 1) * (pool_size - 2))
            expected += draw_count * probability
            variance += draw_count * probability * (1 - probability)
        z_score = (observed - expected) / np.sqrt(variance) if variance > 0 else np.nan
        lift = (observed / expected - 1) * 100 if expected > 0 else np.nan
        rows.append((f"{a}-{b}-{c}", observed, expected, lift, z_score))
    result = pd.DataFrame(rows, columns=["triplet", "count", "expected", "lift_vs_expected_pct", "z_score"])
    return result.sort_values(["z_score", "count"], ascending=[False, False]).head(top_n).reset_index(drop=True)


def ticket_frequency(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["white_numbers", "powerball", "count", "last_seen", "draws_since_seen", "ticket"])

    work = df.reset_index(drop=True).copy()
    work["draw_idx"] = np.arange(len(work))
    grouped = (
        work.groupby(["white_sorted", "powerball"], as_index=False)
        .agg(
            count=("draw_idx", "size"),
            last_seen=("draw_date", "max"),
            last_idx=("draw_idx", "max"),
        )
    )
    grouped["draws_since_seen"] = (len(work) - 1 - grouped["last_idx"]).astype(int)
    grouped["white_numbers"] = grouped["white_sorted"].apply(lambda t: " - ".join(str(int(n)) for n in t))
    grouped["powerball"] = grouped["powerball"].astype(int)
    grouped["ticket"] = grouped["white_numbers"] + " | PB " + grouped["powerball"].astype(str)
    grouped = grouped.drop(columns=["white_sorted", "last_idx"]).sort_values(
        ["count", "draws_since_seen"], ascending=[False, True]
    )
    return grouped.reset_index(drop=True)


def bucket_deviation(expected_df: pd.DataFrame, bucket_size: int = 10) -> pd.DataFrame:
    if expected_df.empty:
        return pd.DataFrame(columns=["bucket", "observed", "expected", "delta", "pct_vs_expected", "z_score"])
    max_number = int(expected_df["number"].max())
    work = expected_df.copy()
    work["bucket_start"] = ((work["number"] - 1) // bucket_size) * bucket_size + 1
    work["bucket_end"] = (work["bucket_start"] + bucket_size - 1).clip(upper=max_number)
    work["bucket"] = work["bucket_start"].astype(int).astype(str) + "-" + work["bucket_end"].astype(int).astype(str)
    grouped = (
        work.groupby(["bucket_start", "bucket_end", "bucket"], as_index=False)
        .agg(
            observed=("observed", "sum"),
            expected=("expected", "sum"),
            variance=("variance", "sum"),
        )
        .sort_values("bucket_start")
    )
    grouped["delta"] = grouped["observed"] - grouped["expected"]
    grouped["pct_vs_expected"] = np.where(
        grouped["expected"] > 0,
        (grouped["observed"] / grouped["expected"] - 1) * 100,
        np.nan,
    )
    grouped["z_score"] = np.where(
        grouped["variance"] > 0,
        grouped["delta"] / np.sqrt(grouped["variance"]),
        np.nan,
    )
    return grouped[["bucket", "observed", "expected", "delta", "pct_vs_expected", "z_score"]]


def last_digit_deviation(expected_df: pd.DataFrame) -> pd.DataFrame:
    if expected_df.empty:
        return pd.DataFrame(columns=["last_digit", "observed", "expected", "delta", "pct_vs_expected", "z_score"])
    work = expected_df.copy()
    work["last_digit"] = work["number"] % 10
    grouped = (
        work.groupby("last_digit", as_index=False)
        .agg(
            observed=("observed", "sum"),
            expected=("expected", "sum"),
            variance=("variance", "sum"),
        )
        .sort_values("last_digit")
    )
    grouped["delta"] = grouped["observed"] - grouped["expected"]
    grouped["pct_vs_expected"] = np.where(
        grouped["expected"] > 0,
        (grouped["observed"] / grouped["expected"] - 1) * 100,
        np.nan,
    )
    grouped["z_score"] = np.where(
        grouped["variance"] > 0,
        grouped["delta"] / np.sqrt(grouped["variance"]),
        np.nan,
    )
    return grouped[["last_digit", "observed", "expected", "delta", "pct_vs_expected", "z_score"]]


def era_stability_white(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return (
            pd.DataFrame(columns=["number", "active_eras", "mean_z", "std_z", "consistent_sign"]),
            pd.DataFrame(),
        )
    max_number = int(df["white_pool_max"].max())
    era_frames = []
    for era_name, era_df in df.groupby("era", sort=False):
        era_exp = mixed_expected_white(era_df, max_number=max_number)
        era_exp["era"] = era_name
        era_frames.append(era_exp[["number", "era", "expected", "z_score"]])
    long_df = pd.concat(era_frames, ignore_index=True)
    z_matrix = long_df.pivot(index="number", columns="era", values="z_score")
    expected_matrix = long_df.pivot(index="number", columns="era", values="expected")
    active_mask = expected_matrix > 0
    z_active = z_matrix.where(active_mask)

    active_eras = active_mask.sum(axis=1)
    pos_count = (z_active > 0).sum(axis=1)
    neg_count = (z_active < 0).sum(axis=1)
    stability = pd.DataFrame(
        {
            "number": z_active.index.astype(int),
            "active_eras": active_eras.astype(int),
            "mean_z": z_active.mean(axis=1, skipna=True).to_numpy(),
            "std_z": z_active.std(axis=1, skipna=True, ddof=0).fillna(0).to_numpy(),
            "consistent_sign": ((pos_count == active_eras) | (neg_count == active_eras)).to_numpy(),
        }
    ).sort_values(["mean_z", "std_z"], ascending=[False, True])
    return stability.reset_index(drop=True), z_matrix.reset_index()


def parse_weights_csv_bytes(raw_bytes: bytes, max_number: int) -> tuple[pd.DataFrame, int]:
    try:
        raw_df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise ValueError("Unable to parse weights CSV.") from exc
    if raw_df.empty:
        raise ValueError("Weights CSV is empty.")

    normalized_cols = {str(c).strip().lower(): c for c in raw_df.columns}
    if "number" in normalized_cols and "weight" in normalized_cols:
        number_col = normalized_cols["number"]
        weight_col = normalized_cols["weight"]
    elif len(raw_df.columns) >= 2:
        number_col = raw_df.columns[0]
        weight_col = raw_df.columns[1]
    else:
        raise ValueError("Weights CSV must contain at least two columns: number, weight.")

    weights = raw_df[[number_col, weight_col]].rename(columns={number_col: "number", weight_col: "weight"})
    weights["number"] = pd.to_numeric(weights["number"], errors="coerce").astype("Int64")
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce")
    weights = weights.dropna(subset=["number", "weight"]).copy()
    weights["number"] = weights["number"].astype(int)
    weights = weights[(weights["number"] >= 1) & (weights["number"] <= max_number)]
    if weights.empty:
        raise ValueError("No valid rows in weights CSV after cleaning.")

    weights = weights.groupby("number", as_index=False)["weight"].mean()
    full = pd.DataFrame({"number": range(1, max_number + 1)})
    full = full.merge(weights, on="number", how="left")
    missing_count = int(full["weight"].isna().sum())
    fill_value = full["weight"].mean()
    if pd.isna(fill_value):
        fill_value = 1.0
    full["weight"] = full["weight"].fillna(fill_value)
    std = full["weight"].std(ddof=0) or 1.0
    full["weight_z"] = (full["weight"] - full["weight"].mean()) / std
    full["weight_delta_pct"] = np.where(
        full["weight"].mean() > 0,
        (full["weight"] / full["weight"].mean() - 1) * 100,
        0.0,
    )
    full["weight_source"] = "CSV de pesos medidos"
    return full, missing_count


def physical_bias_projection(
    white_exp: pd.DataFrame,
    include_weight: bool,
    include_wear: bool,
    beta: float,
    gamma: float,
    measured_weights: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    if white_exp.empty:
        return white_exp.copy(), {}

    sim = white_exp[["number", "observed", "expected", "variance", "z_score"]].copy()
    total_expected = sim["expected"].sum()
    total_observed = sim["observed"].sum()
    sim["baseline_prob"] = np.where(total_expected > 0, sim["expected"] / total_expected, 1 / len(sim))

    weight_delta_pct = np.zeros(len(sim))
    source = "uniforme: sin mediciones"
    if include_weight and measured_weights is not None:
        weight_columns = ["number", "weight"]
        if "weight_delta_pct" in measured_weights.columns:
            weight_columns.append("weight_delta_pct")
        merged = sim[["number"]].merge(measured_weights[weight_columns], on="number", how="left")
        if "weight_delta_pct" in merged.columns:
            weight_delta_pct = merged["weight_delta_pct"].fillna(0).to_numpy(dtype=float)
        else:
            mean_weight = float(merged["weight"].mean())
            weight_delta_pct = np.where(
                mean_weight > 0,
                (merged["weight"].fillna(mean_weight).to_numpy(dtype=float) / mean_weight - 1) * 100,
                0.0,
            )
        if "weight_source" in measured_weights.columns and measured_weights["weight_source"].notna().any():
            source = str(measured_weights["weight_source"].dropna().iloc[0])
        else:
            source = "CSV de pesos medidos"

    wear_delta_pct = np.zeros(len(sim))
    if include_wear and measured_weights is not None and "wear_delta_pct" in measured_weights.columns:
        wear_merged = sim[["number"]].merge(
            measured_weights[["number", "wear_delta_pct"]], on="number", how="left"
        )
        wear_delta_pct = wear_merged["wear_delta_pct"].fillna(0).to_numpy(dtype=float)

    # beta/gamma are sensitivity coefficients per percentage point, not measured physical laws.
    logits = (
        np.log(np.clip(sim["baseline_prob"].to_numpy(), 1e-12, None))
        - beta * weight_delta_pct
        - gamma * wear_delta_pct
    )
    shifted = logits - logits.max()
    raw = np.exp(shifted)
    adjusted_prob = raw / raw.sum()

    sim["weight_delta_pct"] = weight_delta_pct
    sim["wear_delta_pct"] = wear_delta_pct
    sim["adjusted_prob"] = adjusted_prob
    sim["adjusted_expected"] = adjusted_prob * total_observed
    sim["prob_lift_pct"] = np.where(
        sim["baseline_prob"] > 0,
        (sim["adjusted_prob"] / sim["baseline_prob"] - 1) * 100,
        np.nan,
    )
    sim["expected_delta"] = sim["adjusted_expected"] - sim["expected"]
    sim["adjusted_z_proxy"] = np.where(
        sim["variance"] > 0,
        (sim["observed"] - sim["adjusted_expected"]) / np.sqrt(sim["variance"]),
        np.nan,
    )
    sim = sim.sort_values("adjusted_prob", ascending=False).reset_index(drop=True)

    valid_uniform = sim["expected"] > 0
    valid_adjusted = sim["adjusted_expected"] > 0
    chi2_uniform = float((((sim.loc[valid_uniform, "observed"] - sim.loc[valid_uniform, "expected"]) ** 2) / sim.loc[valid_uniform, "expected"]).sum())
    chi2_adjusted = float((((sim.loc[valid_adjusted, "observed"] - sim.loc[valid_adjusted, "adjusted_expected"]) ** 2) / sim.loc[valid_adjusted, "adjusted_expected"]).sum())

    metrics = {
        "weight_source": source,
        "chi2_uniform": chi2_uniform,
        "chi2_adjusted": chi2_adjusted,
        "chi2_delta": chi2_uniform - chi2_adjusted,
        "mae_uniform": float((sim["observed"] - sim["expected"]).abs().mean()),
        "mae_adjusted": float((sim["observed"] - sim["adjusted_expected"]).abs().mean()),
    }
    return sim, metrics


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    expv = np.exp(shifted)
    denom = expv.sum()
    if denom <= 0:
        return np.repeat(1 / len(logits), len(logits))
    return expv / denom


def statistical_forecast_white(
    filtered: pd.DataFrame,
    white_exp: pd.DataFrame,
    white_score: pd.DataFrame,
    strength: float = 0.025,
    model_weight: float = 1.0,
) -> pd.DataFrame:
    del white_exp, white_score
    result = build_white_forecast(filtered, strength=float(strength), model_weight=float(model_weight))
    if not result.empty:
        result["draw_prob"] = result["draw_weight"]
        result["inclusion_prob_next_draw"] = result["pop_next_draw"]
        result["z_score"] = result["forecast_signal"]
        result["recent_52_z"] = np.nan
        result["gap_z"] = np.nan
    return result


def statistical_forecast_pb(
    filtered: pd.DataFrame,
    pb_exp: pd.DataFrame,
    pb_over: pd.DataFrame,
    strength: float = 0.025,
    model_weight: float = 1.0,
) -> pd.DataFrame:
    del pb_exp, pb_over
    result = build_powerball_forecast(filtered, strength=float(strength), model_weight=float(model_weight))
    if not result.empty:
        result["draw_prob"] = result["pop_next_draw"]
        result["z_score"] = result["forecast_signal"]
    return result


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros(len(values))
    return (values - vmin) / (vmax - vmin)


def _build_sim_count_df(counter: Counter, key_name: str, n_samples: int) -> pd.DataFrame:
    if not counter:
        return pd.DataFrame(columns=[key_name, "sim_count", "appearance_rate"])
    rows = [(k, v) for k, v in counter.items()]
    df = pd.DataFrame(rows, columns=[key_name, "sim_count"]).sort_values("sim_count", ascending=False)
    df["appearance_rate"] = np.where(n_samples > 0, df["sim_count"] / n_samples, 0.0)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def run_ticket_simulation_bundle(
    white_forecast: pd.DataFrame,
    pb_forecast: pd.DataFrame,
    n_samples: int = 8000,
    top_n: int = 12,
    seed: int = 42,
    overlap_lambda: float = 0.35,
) -> dict[str, pd.DataFrame]:
    if white_forecast.empty or pb_forecast.empty:
        empty_tickets = pd.DataFrame(
            columns=[
                "ticket",
                "white_numbers",
                "powerball",
                "sim_count",
                "sim_prob",
                "empirical_freq_score",
                "statistical_weight_score",
                "overlap_penalty",
                "coverage_gain",
                "model_ticket_prob",
                "official_ticket_prob",
                "ticket_score",
            ]
        )
        return {
            "tickets": empty_tickets,
            "white_number_freq": pd.DataFrame(columns=["number", "sim_count", "appearance_rate"]),
            "powerball_freq": pd.DataFrame(columns=["number", "sim_count", "appearance_rate"]),
            "pair_freq": pd.DataFrame(columns=["pair", "sim_count", "appearance_rate"]),
            "triplet_freq": pd.DataFrame(columns=["triplet", "sim_count", "appearance_rate"]),
        }

    rng = np.random.default_rng(int(seed))
    white_numbers = white_forecast["number"].to_numpy(dtype=int)
    white_p = white_forecast["draw_weight"].to_numpy(dtype=float)
    white_p = white_p / white_p.sum()
    white_model_weight = float(white_forecast["model_weight"].iloc[0])
    white_normalizer = conditional_bernoulli_normalizer(white_p, sample_size=5)
    white_index_map = {int(number): index for index, number in enumerate(white_numbers)}
    candidate_uniform_probability = 1 / comb(len(white_numbers), 5)
    white_emp_map = white_forecast.set_index("number")["observed_active_era"].astype(float).to_dict()

    pb_numbers = pb_forecast["number"].to_numpy(dtype=int)
    pb_p = pb_forecast["pop_next_draw"].to_numpy(dtype=float)
    pb_p = pb_p / pb_p.sum()
    pb_prob_map = pb_forecast.set_index("number")["pop_next_draw"].to_dict()
    pb_emp_map = pb_forecast.set_index("number")["observed_active_era"].astype(float).to_dict()

    ticket_counter = Counter()
    white_counter = Counter()
    pb_counter = Counter()
    pair_counter = Counter()
    triplet_counter = Counter()

    full_white_pool = int(round(5 / float(white_forecast["uniform_pop_next_draw"].iloc[0])))
    full_pb_pool = int(round(1 / float(pb_forecast["uniform_pop_next_draw"].iloc[0])))
    official_ticket_probability = 1 / (comb(full_white_pool, 5) * full_pb_pool)

    for _ in range(int(n_samples)):
        if rng.random() < white_model_weight:
            selected_indices = sample_conditional_bernoulli(white_p, sample_size=5, rng=rng)
        else:
            selected_indices = rng.choice(len(white_numbers), size=5, replace=False)
        w = tuple(sorted(white_numbers[selected_indices].tolist()))
        pb = int(rng.choice(pb_numbers, size=1, replace=True, p=pb_p)[0])
        ticket_counter[(w, pb)] += 1
        for n in w:
            white_counter[n] += 1
        pb_counter[pb] += 1
        for p in combinations(w, 2):
            pair_counter[p] += 1
        for t in combinations(w, 3):
            triplet_counter[t] += 1

    candidate_rows = []
    for (w, pb), cnt in ticket_counter.items():
        empirical_raw = float(np.mean([white_emp_map.get(n, 0.0) for n in w] + [pb_emp_map.get(pb, 0.0)]))
        subset_indices = np.asarray([white_index_map[int(number)] for number in w], dtype=int)
        white_model_subset_probability = float(np.prod(white_p[subset_indices]) / white_normalizer)
        white_final_subset_probability = (
            white_model_weight * white_model_subset_probability
            + (1 - white_model_weight) * candidate_uniform_probability
        )
        model_ticket_probability = white_final_subset_probability * float(pb_prob_map.get(pb, 0.0))
        stat_log = float(np.log(max(model_ticket_probability, 1e-15)))
        candidate_rows.append(
            {
                "white_tuple": w,
                "powerball": pb,
                "sim_count": int(cnt),
                "sim_prob": cnt / n_samples,
                "empirical_raw": empirical_raw,
                "statistical_raw": stat_log,
                "model_ticket_prob": model_ticket_probability,
                "official_ticket_prob": official_ticket_probability,
            }
        )
    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        return {
            "tickets": pd.DataFrame(),
            "white_number_freq": pd.DataFrame(),
            "powerball_freq": pd.DataFrame(),
            "pair_freq": pd.DataFrame(),
            "triplet_freq": pd.DataFrame(),
        }

    candidates["empirical_freq_score"] = _minmax_scale(candidates["empirical_raw"].to_numpy(dtype=float))
    candidates["statistical_weight_score"] = _minmax_scale(candidates["statistical_raw"].to_numpy(dtype=float))
    candidates["base_score"] = 0.45 * candidates["empirical_freq_score"] + 0.55 * candidates["statistical_weight_score"]
    candidate_pool = max(int(top_n) * 200, 2000)
    candidates = candidates.nlargest(min(candidate_pool, len(candidates)), "base_score").copy()

    selected_rows = []
    selected_tuples: list[tuple[tuple[int, ...], int]] = []
    covered_white: set[int] = set()
    covered_pb: set[int] = set()
    remaining = candidates.copy()
    select_n = min(int(top_n), len(remaining))
    for _ in range(select_n):
        if remaining.empty:
            break
        if not selected_tuples:
            remaining["overlap_penalty"] = 0.0
            remaining["coverage_gain"] = 1.0
            remaining["ticket_score"] = remaining["base_score"] + 0.15
        else:
            penalties = []
            coverage_gains = []
            for _, row in remaining.iterrows():
                w = set(row["white_tuple"])
                pb = int(row["powerball"])
                max_overlap = 0.0
                for sw, spb in selected_tuples:
                    white_overlap = len(w.intersection(set(sw))) / 5.0
                    pb_overlap = 0.25 if pb == int(spb) else 0.0
                    max_overlap = max(max_overlap, white_overlap + pb_overlap)
                penalties.append(max_overlap)
                coverage_gains.append(len(w.difference(covered_white)) / 5.0 + (0.25 if pb not in covered_pb else 0.0))
            remaining["overlap_penalty"] = penalties
            remaining["coverage_gain"] = coverage_gains
            remaining["ticket_score"] = (
                remaining["base_score"]
                - float(overlap_lambda) * remaining["overlap_penalty"]
                + 0.15 * remaining["coverage_gain"]
            )

        best_idx = remaining["ticket_score"].idxmax()
        best = remaining.loc[best_idx].copy()
        selected_rows.append(best)
        selected_tuples.append((tuple(int(n) for n in best["white_tuple"]), int(best["powerball"])))
        covered_white.update(int(n) for n in best["white_tuple"])
        covered_pb.add(int(best["powerball"]))
        remaining = remaining.drop(index=best_idx)

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    if selected.empty:
        ticket_df = pd.DataFrame(columns=["ticket", "white_numbers", "powerball", "sim_count", "sim_prob", "empirical_freq_score", "statistical_weight_score", "overlap_penalty", "coverage_gain", "model_ticket_prob", "official_ticket_prob", "ticket_score"])
    else:
        selected["white_numbers"] = selected["white_tuple"].apply(lambda t: " - ".join(str(int(n)) for n in t))
        selected["ticket"] = selected["white_numbers"] + " | PB " + selected["powerball"].astype(int).astype(str)
        ticket_df = selected[
            [
                "ticket",
                "white_numbers",
                "powerball",
                "sim_count",
                "sim_prob",
                "empirical_freq_score",
                "statistical_weight_score",
                "overlap_penalty",
                "coverage_gain",
                "model_ticket_prob",
                "official_ticket_prob",
                "ticket_score",
            ]
        ].copy()
        ticket_df["powerball"] = ticket_df["powerball"].astype(int)
        ticket_df["sim_count"] = ticket_df["sim_count"].astype(int)

    white_freq = _build_sim_count_df(white_counter, "number", int(n_samples))
    pb_freq = _build_sim_count_df(pb_counter, "number", int(n_samples))
    pair_freq = _build_sim_count_df(
        Counter({f"{a}-{b}": c for (a, b), c in pair_counter.items()}), "pair", int(n_samples)
    )
    triplet_freq = _build_sim_count_df(
        Counter({f"{a}-{b}-{c}": c for (a, b, c), c in triplet_counter.items()}), "triplet", int(n_samples)
    )
    if not white_freq.empty:
        white_freq["number"] = white_freq["number"].astype(int)
    if not pb_freq.empty:
        pb_freq["number"] = pb_freq["number"].astype(int)

    return {
        "tickets": ticket_df.sort_values("ticket_score", ascending=False).reset_index(drop=True),
        "white_number_freq": white_freq,
        "powerball_freq": pb_freq,
        "pair_freq": pair_freq,
        "triplet_freq": triplet_freq,
    }


def simulate_forecast_tickets(
    white_forecast: pd.DataFrame,
    pb_forecast: pd.DataFrame,
    n_samples: int = 8000,
    top_n: int = 12,
    seed: int = 42,
    overlap_lambda: float = 0.35,
) -> pd.DataFrame:
    bundle = run_ticket_simulation_bundle(
        white_forecast=white_forecast,
        pb_forecast=pb_forecast,
        n_samples=n_samples,
        top_n=top_n,
        seed=seed,
        overlap_lambda=overlap_lambda,
    )
    return bundle["tickets"]


def build_excel_export(
    filtered: pd.DataFrame,
    white_exp: pd.DataFrame,
    pb_exp: pd.DataFrame,
    white_over: pd.DataFrame,
    pb_over: pd.DataFrame,
    white_score: pd.DataFrame,
    bucket_stats: pd.DataFrame,
    digit_stats: pd.DataFrame,
    stability: pd.DataFrame,
    quality_summary: pd.DataFrame,
    quality_issues: pd.DataFrame,
    white_forecast: pd.DataFrame,
    pb_forecast: pd.DataFrame,
    tickets_forecast: pd.DataFrame,
    sim_white_number_freq: pd.DataFrame,
    sim_powerball_freq: pd.DataFrame,
    sim_pair_freq: pd.DataFrame,
    sim_triplet_freq: pd.DataFrame,
    sim_df: pd.DataFrame,
    equipment_rows: pd.DataFrame,
    equipment_quality: pd.DataFrame,
    equipment_backtest: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        filtered[["draw_date", *WHITE_COLS, "powerball", "power_play", "weekday", "era", "year"]].to_excel(
            writer, sheet_name="filtered_draws", index=False
        )
        white_exp.to_excel(writer, sheet_name="white_expected", index=False)
        pb_exp.to_excel(writer, sheet_name="pb_expected", index=False)
        white_over.to_excel(writer, sheet_name="white_overdue", index=False)
        pb_over.to_excel(writer, sheet_name="pb_overdue", index=False)
        white_score.to_excel(writer, sheet_name="composite_score", index=False)
        bucket_stats.to_excel(writer, sheet_name="bucket_stats", index=False)
        digit_stats.to_excel(writer, sheet_name="last_digit_stats", index=False)
        stability.to_excel(writer, sheet_name="era_stability", index=False)
        quality_summary.to_excel(writer, sheet_name="quality_summary", index=False)
        if not quality_issues.empty:
            quality_issues.to_excel(writer, sheet_name="quality_issues", index=False)
        if not white_forecast.empty:
            white_forecast.to_excel(writer, sheet_name="forecast_white", index=False)
        if not pb_forecast.empty:
            pb_forecast.to_excel(writer, sheet_name="forecast_pb", index=False)
        if not tickets_forecast.empty:
            tickets_forecast.to_excel(writer, sheet_name="forecast_tickets", index=False)
        if not sim_white_number_freq.empty:
            sim_white_number_freq.to_excel(writer, sheet_name="sim_white_number_freq", index=False)
        if not sim_powerball_freq.empty:
            sim_powerball_freq.to_excel(writer, sheet_name="sim_pb_number_freq", index=False)
        if not sim_pair_freq.empty:
            sim_pair_freq.to_excel(writer, sheet_name="sim_pair_freq", index=False)
        if not sim_triplet_freq.empty:
            sim_triplet_freq.to_excel(writer, sheet_name="sim_triplet_freq", index=False)
        if not sim_df.empty:
            sim_df.to_excel(writer, sheet_name="physical_sim", index=False)
        if not equipment_rows.empty:
            equipment_rows.to_excel(writer, sheet_name="equipment_pretests", index=False)
        if not equipment_quality.empty:
            equipment_quality.to_excel(writer, sheet_name="equipment_quality", index=False)
        if not equipment_backtest.empty:
            equipment_backtest.to_excel(writer, sheet_name="equipment_backtest", index=False)
    output.seek(0)
    return output.getvalue()


def trend_score_white(
    df: pd.DataFrame,
    weight_long: float = 0.45,
    weight_recent: float = 0.35,
    weight_gap: float = 0.20,
) -> pd.DataFrame:
    max_number = int(df["white_pool_max"].max()) if not df.empty else 0
    mixed = mixed_expected_white(df, max_number=max_number)
    if df.empty:
        return mixed
    total_weight = weight_long + weight_recent + weight_gap
    if total_weight <= 0:
        weight_long, weight_recent, weight_gap = 0.45, 0.35, 0.20
        total_weight = 1.0
    weight_long, weight_recent, weight_gap = (
        weight_long / total_weight,
        weight_recent / total_weight,
        weight_gap / total_weight,
    )

    recent_n = min(52, len(df))
    recent = df.tail(recent_n)
    recent_counts = white_counts(recent).reindex(range(1, max_number + 1), fill_value=0)
    recent_expected = []
    recent_var = []
    for n in range(1, max_number + 1):
        p = np.where(recent["white_pool_max"].to_numpy() >= n, 5 / recent["white_pool_max"].to_numpy(), 0.0)
        recent_expected.append(p.sum())
        recent_var.append((p * (1 - p)).sum())
    recent_z = np.where(np.array(recent_var) > 0, (recent_counts.to_numpy() - np.array(recent_expected)) / np.sqrt(np.array(recent_var)), np.nan)
    overdue = overdue_white(df).set_index("number")
    gap_std = overdue["draws_since_seen"].std(ddof=0) or 1.0
    mixed["recent_52_z"] = recent_z
    mixed["draws_since_seen"] = overdue.loc[mixed["number"], "draws_since_seen"].to_numpy()
    mixed["gap_z"] = (mixed["draws_since_seen"] - mixed["draws_since_seen"].mean()) / gap_std
    mixed["exploration_score"] = (
        weight_long * mixed["z_score"].fillna(0)
        + weight_recent * pd.Series(mixed["recent_52_z"]).fillna(0)
        + weight_gap * mixed["gap_z"].fillna(0)
    )
    return mixed.sort_values("exploration_score", ascending=False)


def format_p(p):
    if pd.isna(p):
        return "N/A"
    return f"{p:.4f}"


@st.cache_data(show_spinner=False)
def cached_walk_forward_backtest(df: pd.DataFrame):
    return walk_forward_backtest(df)


@st.cache_data(show_spinner=False)
def cached_forecast_pop_intervals(
    df: pd.DataFrame,
    white_strength: float,
    powerball_strength: float,
    white_model_weight: float,
    powerball_model_weight: float,
):
    return forecast_pop_intervals(
        df,
        white_strength=white_strength,
        powerball_strength=powerball_strength,
        white_model_weight=white_model_weight,
        powerball_model_weight=powerball_model_weight,
    )


@st.cache_data(show_spinner=False)
def cached_equipment_backtest(equipment_rows: pd.DataFrame):
    return equipment_walk_forward_backtest(equipment_rows)


@st.cache_data(show_spinner=False)
def cached_pretest_comparison(equipment_rows: pd.DataFrame):
    return pretest_draw_comparison(equipment_rows)


def build_navigation_guide() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Resumen y combinaciones", "Ranking, extremos y tickets candidatos."),
            ("Validación histórica", "Compara el modelo con el uniforme fuera de muestra."),
            ("Frecuencias", "Observado vs esperado, z-score y significancia."),
            ("Recencia", "Cuántos sorteos lleva cada número sin aparecer."),
            ("Patrones y combinaciones", "Pares, tríos, sumas y estructura histórica."),
            ("Diagnósticos avanzados", "Buckets, últimos dígitos y estabilidad entre eras."),
            ("Calidad de datos", "Filas inválidas, rangos y perfil del archivo."),
            ("Evolución temporal", "Frecuencia móvil de un número."),
            ("Equipos y pre-tests", "Máquinas, sets, pruebas previas y validación retrospectiva."),
            ("Laboratorio experimental", "Escenarios físicos hipotéticos, separados del forecast."),
            ("Datos y descargas", "Datos filtrados y exportación CSV/Excel."),
        ],
        columns=["Seccion", "Que muestra"],
    )


st.title("Powerball: análisis y forecast")
st.caption("Datos oficiales, validación histórica y combinaciones exploratorias en una sola vista.")

sample_path = Path(__file__).with_name("powerball.csv")
now_ct = texas_now_ct()
source_mode = st.radio(
    "Fuente de datos",
    options=["Texas Lottery (automática)", "Subir mi CSV", "Copia local incluida"],
    horizontal=True,
    key="data_source_mode",
)
uploaded = None
if source_mode == "Subir mi CSV":
    uploaded = st.file_uploader("Selecciona el CSV de números ganadores", type=["csv"])

with st.sidebar.expander("Datos oficiales", expanded=True):
    st.markdown(f"[Abrir CSV oficial de Texas Lottery]({TEXAS_POWERBALL_CSV_URL})")
    st.markdown(f"[Abrir reporte oficial de pre-tests]({POWERBALL_PRETEST_PDF_URL})")
    st.caption("Resultados y pre-tests se renuevan cada 6 horas. El botón fuerza ambas consultas.")
    if st.button("Buscar actualización ahora", key="sync_texas_csv_button", use_container_width=True):
        load_official_powerball.clear()
        st.cache_data.clear()
        st.success("Caché limpiada. Se consultará nuevamente la fuente oficial.")
    st.caption(f"Hora Texas (CT): {now_ct.strftime('%Y-%m-%d %H:%M')}")

data_source_label = source_mode
if uploaded is not None:
    df = parse_powerball_csv_bytes(uploaded.getvalue())
elif source_mode == "Texas Lottery (automática)":
    try:
        df = load_official_powerball()
    except Exception as exc:
        df = load_default_sample(
            file_mtime_ns=int(sample_path.stat().st_mtime_ns) if sample_path.exists() else None,
            file_size=int(sample_path.stat().st_size) if sample_path.exists() else None,
        )
        data_source_label = "Copia local de respaldo"
        st.warning(f"No se pudo consultar Texas Lottery; se usa la copia local. Detalle: {exc}")
elif source_mode == "Copia local incluida":
    df = load_default_sample(
        file_mtime_ns=int(sample_path.stat().st_mtime_ns) if sample_path.exists() else None,
        file_size=int(sample_path.stat().st_size) if sample_path.exists() else None,
    )
else:
    df = None

if df is None or df.empty:
    st.info("Upload a CSV to begin.")
    st.stop()

pretest_path = Path(__file__).with_name(PRETEST_LOCAL_FILENAME)
try:
    equipment_df = load_official_pretests()
    equipment_source_label = "Powerball Pre-Test oficial"
except Exception as exc:
    equipment_df = load_local_pretests(
        file_mtime_ns=int(pretest_path.stat().st_mtime_ns) if pretest_path.exists() else None,
        file_size=int(pretest_path.stat().st_size) if pretest_path.exists() else None,
    )
    equipment_source_label = "Copia local de pre-tests"
    if equipment_df.empty:
        st.sidebar.warning(f"Pre-tests no disponibles: {exc}")

if not equipment_df.empty:
    draws_with_equipment, equipment_quality = join_winning_draws_with_equipment(df, equipment_df)
else:
    draws_with_equipment, equipment_quality = pd.DataFrame(), pd.DataFrame()

PAGE_LABELS = {
    "Inicio (Forecast)": "1. Resumen y combinaciones",
    "Validacion (Backtest)": "2. Validación histórica",
    "Frecuencia y Significancia": "3. Frecuencias",
    "Recencia (Overdue)": "4. Recencia",
    "Estructura y Combinaciones": "5. Patrones y combinaciones",
    "Diagnosticos": "6. Diagnósticos avanzados",
    "Perfil y Calidad": "7. Calidad de datos",
    "Rolling": "8. Evolución temporal",
    "Equipos y Pre-tests": "9. Equipos y pre-tests",
    "Simulador Fisico": "10. Laboratorio experimental",
    "Datos y Exportes": "11. Datos y descargas",
}
st.sidebar.header("Secciones")
page = st.sidebar.radio(
    "Navegar",
    options=list(PAGE_LABELS),
    format_func=lambda value: PAGE_LABELS[value],
    label_visibility="collapsed",
)

min_date, max_date = df["draw_date"].min().date(), df["draw_date"].max().date()
csv_signature = (len(df), str(min_date), str(max_date))
if st.session_state.get("date_range_csv_signature") != csv_signature:
    st.session_state["start_date_filter"] = min_date
    st.session_state["end_date_filter"] = max_date
    st.session_state["date_range_csv_signature"] = csv_signature
if "start_date_filter" not in st.session_state:
    st.session_state["start_date_filter"] = min_date
if "end_date_filter" not in st.session_state:
    st.session_state["end_date_filter"] = max_date

era_options = ["All"] + list(df["era"].dropna().unique())
weekday_options = ["All"] + sorted(df["weekday"].dropna().unique().tolist())
year_options = ["All"] + [int(y) for y in sorted(df["year"].dropna().unique().tolist())]
with st.sidebar.expander("Filtros de análisis", expanded=False):
    st.caption("Afectan gráficos y tablas, no el forecast del próximo sorteo.")
    selected_eras = st.multiselect("Era", options=era_options, default=["All"])
    selected_weekdays = st.multiselect("Día de la semana", options=weekday_options, default=["All"])
    selected_years = st.multiselect("Año", options=year_options, default=["All"])
    start_date_input = st.date_input(
        "Fecha de inicio",
        min_value=min_date,
        max_value=max_date,
        value=st.session_state["start_date_filter"],
        key="start_date_filter",
    )
    end_date_input = st.date_input(
        "Fecha de fin",
        min_value=min_date,
        max_value=max_date,
        value=st.session_state["end_date_filter"],
        key="end_date_filter",
    )
    show_missing_pp = st.checkbox("Conservar filas sin Power Play", value=True)

with st.sidebar.expander("Ajustes avanzados", expanded=False):
    bucket_size = st.slider("Tamaño de buckets", min_value=5, max_value=20, value=10, step=1)
    weight_long = st.slider("Score descriptivo: largo plazo", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
    weight_recent = st.slider("Score descriptivo: últimos 52", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    weight_gap = st.slider("Score descriptivo: atraso", min_value=0.0, max_value=1.0, value=0.20, step=0.05)
    forecast_samples = st.slider("Simulaciones de tickets", min_value=10000, max_value=200000, value=50000, step=10000)
    forecast_top_tickets = st.slider("Tickets candidatos", min_value=5, max_value=25, value=12, step=1)
    forecast_seed = st.number_input("Semilla reproducible", min_value=1, max_value=999999, value=42, step=1)
    overlap_penalty_lambda = st.slider("Diversidad entre tickets", min_value=0.0, max_value=1.0, value=0.35, step=0.05)

filtered = df.copy()
if "All" not in selected_eras:
    filtered = filtered[filtered["era"].isin(selected_eras)]
if "All" not in selected_weekdays:
    filtered = filtered[filtered["weekday"].isin(selected_weekdays)]
if "All" not in selected_years:
    filtered = filtered[filtered["year"].isin(selected_years)]
start_dt = pd.to_datetime(start_date_input)
end_dt = pd.to_datetime(end_date_input)
if start_dt > end_dt:
    st.sidebar.warning("La fecha de inicio es mayor que la fecha de fin. Se ajustaron automaticamente.")
    start_dt, end_dt = end_dt, start_dt
filtered = filtered[(filtered["draw_date"] >= start_dt) & (filtered["draw_date"] <= end_dt)]
if not show_missing_pp:
    filtered = filtered[filtered["power_play"].notna()]
filtered = filtered.reset_index(drop=True)

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

if not equipment_df.empty:
    selected_draw_dates = set(filtered["draw_date"])
    equipment_filtered = equipment_df[equipment_df["draw_date"].isin(selected_draw_dates)].reset_index(drop=True)
else:
    equipment_filtered = pd.DataFrame()

white_exp = add_significance_columns(mixed_expected_white(filtered))
pb_exp = add_significance_columns(mixed_expected_powerball(filtered))
white_chi = chi_square_from_expected(white_exp)
pb_chi = chi_square_from_expected(pb_exp)
white_over = overdue_white(filtered)
pb_over = overdue_powerball(filtered)
quality_summary, quality_issues = draw_quality_report(filtered)
pair_matrix = pair_cooccurrence_matrix(filtered, top_n=20)
pair_top = pair_frequency(filtered, top_n=40)
triplet_top = triplet_frequency(filtered, top_n=30)
ticket_stats = ticket_frequency(filtered)
white_score = trend_score_white(
    filtered,
    weight_long=weight_long,
    weight_recent=weight_recent,
    weight_gap=weight_gap,
)
bucket_stats = bucket_deviation(white_exp, bucket_size=bucket_size)
digit_stats = last_digit_deviation(white_exp)
stability_df, stability_matrix = era_stability_white(filtered)
forecast_source = current_matrix_draws(df)
backtest_detail, backtest_summary, backtest_yearly, backtest_config = cached_walk_forward_backtest(df)
forecast_strength_white = float(backtest_config.white_strength)
forecast_strength_pb = float(backtest_config.powerball_strength)
forecast_model_weight_white = float(backtest_config.white_model_weight)
forecast_model_weight_pb = float(backtest_config.powerball_model_weight)
forecast_white_exp = add_significance_columns(mixed_expected_white(forecast_source))
forecast_pb_exp = add_significance_columns(mixed_expected_powerball(forecast_source))
forecast_white_score = trend_score_white(forecast_source, weight_long=0.65, weight_recent=0.35, weight_gap=0.0)
white_forecast = statistical_forecast_white(
    filtered=forecast_source,
    white_exp=forecast_white_exp,
    white_score=forecast_white_score,
    strength=forecast_strength_white,
    model_weight=forecast_model_weight_white,
)
pb_forecast = statistical_forecast_pb(
    filtered=forecast_source,
    pb_exp=forecast_pb_exp,
    pb_over=overdue_powerball(forecast_source),
    strength=forecast_strength_pb,
    model_weight=forecast_model_weight_pb,
)
white_pop_intervals, pb_pop_intervals = cached_forecast_pop_intervals(
    forecast_source,
    white_strength=forecast_strength_white,
    powerball_strength=forecast_strength_pb,
    white_model_weight=forecast_model_weight_white,
    powerball_model_weight=forecast_model_weight_pb,
)
if not white_forecast.empty and not white_pop_intervals.empty:
    white_forecast = white_forecast.merge(white_pop_intervals, on="number", how="left")
if not pb_forecast.empty and not pb_pop_intervals.empty:
    pb_forecast = pb_forecast.merge(pb_pop_intervals, on="number", how="left")
ticket_sim_bundle = run_ticket_simulation_bundle(
    white_forecast=white_forecast,
    pb_forecast=pb_forecast,
    n_samples=int(forecast_samples),
    top_n=int(forecast_top_tickets),
    seed=int(forecast_seed),
    overlap_lambda=float(overlap_penalty_lambda),
)
tickets_forecast = ticket_sim_bundle["tickets"]
sim_white_number_freq = ticket_sim_bundle["white_number_freq"]
sim_powerball_freq = ticket_sim_bundle["powerball_freq"]
sim_pair_freq = ticket_sim_bundle["pair_freq"]
sim_triplet_freq = ticket_sim_bundle["triplet_freq"]
if not sim_white_number_freq.empty:
    white_forecast = white_forecast.merge(
        sim_white_number_freq[["number", "appearance_rate"]].rename(
            columns={"appearance_rate": "simulated_appearance_rate"}
        ),
        on="number",
        how="left",
    )
    white_forecast = white_forecast.sort_values("rank").reset_index(drop=True)
if not white_forecast.empty:
    white_forecast["inclusion_prob_next_draw"] = white_forecast["pop_next_draw"]
    white_forecast["pop_delta_pp"] = (
        white_forecast["pop_next_draw"] - white_forecast["uniform_pop_next_draw"]
    ) * 100
if not sim_powerball_freq.empty:
    pb_forecast = pb_forecast.merge(
        sim_powerball_freq[["number", "appearance_rate"]].rename(
            columns={"appearance_rate": "simulated_probability"}
        ),
        on="number",
        how="left",
    )
    pb_forecast = pb_forecast.sort_values("rank").reset_index(drop=True)
if not pb_forecast.empty:
    pb_forecast["pop_delta_pp"] = (
        pb_forecast["pop_next_draw"] - pb_forecast["uniform_pop_next_draw"]
    ) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fuente activa", data_source_label)
c2.metric("Último sorteo disponible", str(df["draw_date"].max().date()))
c3.metric("Sorteos para forecast", f"{len(forecast_source):,}")
c4.metric("Sorteos en la vista", f"{len(filtered):,}")

guide_df = build_navigation_guide()
default_sim_df, default_sim_metrics = physical_bias_projection(
    white_exp=white_exp,
    include_weight=False,
    include_wear=False,
    beta=0.0,
    gamma=0.0,
    measured_weights=None,
)

if page == "Inicio (Forecast)":
    st.subheader("Resumen y combinaciones")
    st.markdown(
        "<div class='accuracy-note'><b>Cómo leer esta página:</b> el ranking resume señales históricas débiles. "
        "No cambia las probabilidades oficiales del juego. La confianza del modelo se calibra con sorteos "
        "anteriores y el atraso se muestra solo como dato descriptivo. <b>POP</b> significa probabilidad "
        "estimada de que el número aparezca en el próximo sorteo. Accuracy v3 calcula el POP white con una "
        "selección exacta de cinco bolas y reduce automáticamente el peso histórico cuando el backtest no es "
        "concluyente.</div>",
        unsafe_allow_html=True,
    )
    status1, status2, status3, status4 = st.columns(4)
    status1.metric("Próximo sorteo (CT)", str(next_powerball_draw_day_ct(now_ct).date()))
    status2.metric("Evidencia white", backtest_config.white_evidence)
    status3.metric("Evidencia Powerball", backtest_config.powerball_evidence)
    status4.metric(
        "Peso histórico final",
        f"W {backtest_config.white_model_weight:.0%} | PB {backtest_config.powerball_model_weight:.0%}",
    )
    with st.expander("Qué contiene cada sección"):
        st.dataframe(guide_df, width="stretch", hide_index=True)

    if white_forecast.empty or pb_forecast.empty:
        st.info("No hay suficientes datos para construir forecast en el filtro actual.")
    else:
        active_pool_white = int(forecast_source.iloc[-1]["white_pool_max"])
        white_range_filter = st.slider(
            "Limitar números white para las combinaciones",
            min_value=1,
            max_value=active_pool_white,
            value=(1, active_pool_white),
            key="home_forecast_white_range",
        )
        view_mode_white = st.radio(
            "Qué números white quieres ver",
            options=["Mayor señal", "Menor señal", "Más atrasados"],
            horizontal=True,
            key="home_forecast_white_mode",
        )
        view_mode_pb = st.radio(
            "Qué Powerball quieres ver",
            options=["Mayor señal", "Menor señal", "Más atrasados"],
            horizontal=True,
            key="home_forecast_pb_mode",
        )
        view_n = st.slider(
            "Cantidad a mostrar",
            min_value=5,
            max_value=30,
            value=15,
            step=1,
            key="home_forecast_view_n",
        )

        white_filtered = white_forecast[
            (white_forecast["number"] >= white_range_filter[0]) & (white_forecast["number"] <= white_range_filter[1])
        ].copy()
        if view_mode_white == "Mayor señal":
            white_view = white_filtered.sort_values("pop_next_draw", ascending=False)
        elif view_mode_white == "Menor señal":
            white_view = white_filtered.sort_values("pop_next_draw", ascending=True)
        else:
            white_view = white_filtered.sort_values("draws_since_seen", ascending=False)

        if view_mode_pb == "Mayor señal":
            pb_view = pb_forecast.sort_values("pop_next_draw", ascending=False)
        elif view_mode_pb == "Menor señal":
            pb_view = pb_forecast.sort_values("pop_next_draw", ascending=True)
        else:
            pb_view = pb_forecast.sort_values("draws_since_seen", ascending=False)

        top_white = white_forecast.sort_values("pop_next_draw", ascending=False).head(5)["number"].tolist()
        low_white = white_forecast.sort_values("pop_next_draw", ascending=True).head(5)["number"].tolist()
        top_pb = pb_forecast.sort_values("pop_next_draw", ascending=False).head(3)["number"].tolist()
        low_pb = pb_forecast.sort_values("pop_next_draw", ascending=True).head(3)["number"].tolist()
        fw1, fw2, fw3 = st.columns(3)
        fw1.metric("Matriz actual", f"5/1-{int(forecast_source.iloc[-1]['white_pool_max'])}")
        fw2.metric("Mayor señal white", ", ".join(str(n) for n in top_white))
        fw3.metric("Menor señal white", ", ".join(str(n) for n in low_white))
        fp1, fp2 = st.columns(2)
        fp1.metric("Mayor señal PB", ", ".join(str(n) for n in top_pb))
        fp2.metric("Menor señal PB", ", ".join(str(n) for n in low_pb))

        st.markdown("### POP del próximo sorteo")
        st.caption(
            "POP es una estimación del modelo basada en el histórico. La referencia uniforme oficial es "
            f"{5 / active_pool_white:.2%} para cada bola white y "
            f"{1 / int(forecast_source.iloc[-1]['powerball_pool_max']):.2%} para cada Powerball."
        )
        pop_white_display = white_view.head(view_n)[
            [
                "number",
                "pop_next_draw",
                "pop_ci_low",
                "pop_ci_high",
                "uniform_pop_next_draw",
                "pop_delta_pp",
                "draws_since_seen",
            ]
        ].copy()
        pop_white_display.columns = [
            "Número white",
            "POP próximo sorteo",
            "Rango bajo 95%",
            "Rango alto 95%",
            "POP uniforme",
            "Diferencia",
            "Sorteos sin salir",
        ]
        pop_white_display["Número white"] = pop_white_display["Número white"].astype(int)
        pop_white_display["POP próximo sorteo"] = pop_white_display["POP próximo sorteo"].map("{:.2%}".format)
        pop_white_display["Rango bajo 95%"] = pop_white_display["Rango bajo 95%"].map("{:.2%}".format)
        pop_white_display["Rango alto 95%"] = pop_white_display["Rango alto 95%"].map("{:.2%}".format)
        pop_white_display["POP uniforme"] = pop_white_display["POP uniforme"].map("{:.2%}".format)
        pop_white_display["Diferencia"] = pop_white_display["Diferencia"].map(lambda value: f"{value:+.3f} pp")

        pop_pb_display = pb_view.head(min(view_n, 20))[
            [
                "number",
                "pop_next_draw",
                "pop_ci_low",
                "pop_ci_high",
                "uniform_pop_next_draw",
                "pop_delta_pp",
                "draws_since_seen",
            ]
        ].copy()
        pop_pb_display.columns = [
            "Powerball",
            "POP próximo sorteo",
            "Rango bajo 95%",
            "Rango alto 95%",
            "POP uniforme",
            "Diferencia",
            "Sorteos sin salir",
        ]
        pop_pb_display["Powerball"] = pop_pb_display["Powerball"].astype(int)
        pop_pb_display["POP próximo sorteo"] = pop_pb_display["POP próximo sorteo"].map("{:.2%}".format)
        pop_pb_display["Rango bajo 95%"] = pop_pb_display["Rango bajo 95%"].map("{:.2%}".format)
        pop_pb_display["Rango alto 95%"] = pop_pb_display["Rango alto 95%"].map("{:.2%}".format)
        pop_pb_display["POP uniforme"] = pop_pb_display["POP uniforme"].map("{:.2%}".format)
        pop_pb_display["Diferencia"] = pop_pb_display["Diferencia"].map(lambda value: f"{value:+.3f} pp")

        pop_col_white, pop_col_pb = st.columns(2)
        with pop_col_white:
            st.markdown(f"**White: {view_mode_white}**")
            st.dataframe(pop_white_display, width="stretch", hide_index=True)
        with pop_col_pb:
            st.markdown(f"**Powerball: {view_mode_pb}**")
            st.dataframe(pop_pb_display, width="stretch", hide_index=True)

        white_ticket_base = white_filtered.copy()
        if len(white_ticket_base) < 5:
            white_ticket_base = white_forecast.copy()
            st.info("El rango seleccionado tiene menos de cinco números; se usa la matriz white completa.")
        home_bundle = run_ticket_simulation_bundle(
            white_forecast=white_ticket_base,
            pb_forecast=pb_forecast,
            n_samples=int(forecast_samples),
            top_n=int(forecast_top_tickets),
            seed=int(forecast_seed),
            overlap_lambda=float(overlap_penalty_lambda),
        )
        tickets_view = home_bundle["tickets"]
        white_sim_view = home_bundle["white_number_freq"]
        pb_sim_view = home_bundle["powerball_freq"]
        pair_sim_view = home_bundle["pair_freq"]
        triplet_sim_view = home_bundle["triplet_freq"]

        st.divider()
        st.markdown("### Combinaciones candidatas")
        tickets_display = tickets_view.rename(
            columns={
                "ticket": "combinación",
                "white_numbers": "white",
                "powerball": "PB",
                "sim_count": "apariciones_simuladas",
                "sim_prob": "tasa_ticket",
                "empirical_freq_score": "score_histórico",
                "statistical_weight_score": "score_modelo",
                "overlap_penalty": "solapamiento",
                "coverage_gain": "ganancia_cobertura",
                "model_ticket_prob": "prob_modelo_condicional",
                "official_ticket_prob": "prob_oficial_ticket",
                "ticket_score": "score_final",
            }
        )
        tickets_display["tasa_ticket"] = tickets_display["tasa_ticket"].map("{:.5%}".format)
        tickets_display["prob_modelo_condicional"] = tickets_display["prob_modelo_condicional"].map("{:.3e}".format)
        tickets_display["prob_oficial_ticket"] = tickets_display["prob_oficial_ticket"].map("{:.3e}".format)
        st.dataframe(tickets_display, width="stretch", hide_index=True)
        st.caption(
            "La probabilidad del modelo es condicional al rango white seleccionado. La probabilidad oficial de "
            "cada ticket completo sigue siendo la misma. Solapamiento y ganancia de cobertura diversifican el "
            "grupo de tickets; no aumentan la probabilidad oficial individual."
        )
        with st.expander("Ver frecuencias internas de la simulación"):
            col_simfreq_a, col_simfreq_b = st.columns(2)
            with col_simfreq_a:
                st.markdown("**Aparición por simulación: números white**")
                st.dataframe(white_sim_view.head(view_n), width="stretch", hide_index=True)
                st.markdown("**Aparición por simulación: Powerball**")
                st.dataframe(pb_sim_view.head(min(view_n, 20)), width="stretch", hide_index=True)
            with col_simfreq_b:
                st.markdown("**Aparición por simulación: pares**")
                st.dataframe(pair_sim_view.head(view_n), width="stretch", hide_index=True)
                st.markdown("**Aparición por simulación: tripletas**")
                st.dataframe(triplet_sim_view.head(view_n), width="stretch", hide_index=True)
        st.download_button(
            "Descargar combinaciones candidatas",
            tickets_display.to_csv(index=False).encode("utf-8"),
            file_name="powerball_forecast_tickets.csv",
            mime="text/csv",
        )

        st.divider()
        st.markdown("### Histórico: combinaciones que ya aparecieron")
        st.caption(
            "Esta sección describe el archivo filtrado; no implica que una combinación repetida tenga más opción futura."
        )

        repeated_tickets = ticket_stats[ticket_stats["count"] >= 2].copy()
        if repeated_tickets.empty:
            repeated_tickets = ticket_stats.copy()

        one_hit_tickets = ticket_stats[ticket_stats["count"] == 1].copy()
        one_hit_tickets = one_hit_tickets.sort_values("draws_since_seen", ascending=False)

        col_combo_a, col_combo_b = st.columns(2)
        with col_combo_a:
            st.markdown("**Top tickets exactos (5 + PB)**")
            st.dataframe(
                repeated_tickets[["ticket", "count", "last_seen", "draws_since_seen"]].head(view_n),
                width="stretch",
                hide_index=True,
            )
        with col_combo_b:
            st.markdown("**Tickets vistos una sola vez**")
            st.dataframe(
                one_hit_tickets[["ticket", "count", "last_seen", "draws_since_seen"]].head(view_n),
                width="stretch",
                hide_index=True,
            )

        col_combo_c, col_combo_d = st.columns(2)
        with col_combo_c:
            st.markdown("**Pares con mayor desviación vs esperado por era**")
            st.dataframe(pair_top.head(view_n), width="stretch", hide_index=True)
        with col_combo_d:
            st.markdown("**Tríos con mayor desviación vs esperado por era**")
            st.dataframe(triplet_top.head(view_n), width="stretch", hide_index=True)

        col_fw_a, col_fw_b = st.columns(2)
        with col_fw_a:
            fig = px.bar(
                white_view.head(view_n),
                x="number",
                y="pop_next_draw",
                title=f"White POP: {view_mode_white}",
                hover_data={
                    "pop_ci_low": ":.2%",
                    "pop_ci_high": ":.2%",
                    "lift_vs_uniform_pct": ":.2f",
                    "z_score": ":.2f",
                },
            )
            fig.add_hline(
                y=5 / active_pool_white,
                line_dash="dash",
                annotation_text=f"Uniforme {5 / active_pool_white:.2%}",
            )
            fig.update_layout(height=340, yaxis_title="POP próximo sorteo")
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig, width="stretch")
        with col_fw_b:
            fig = px.bar(
                pb_view.head(min(20, view_n)),
                x="number",
                y="pop_next_draw",
                title=f"Powerball POP: {view_mode_pb}",
                hover_data={
                    "pop_ci_low": ":.2%",
                    "pop_ci_high": ":.2%",
                    "lift_vs_uniform_pct": ":.2f",
                    "z_score": ":.2f",
                    "draws_since_seen": True,
                },
            )
            pb_uniform_pop = 1 / int(forecast_source.iloc[-1]["powerball_pool_max"])
            fig.add_hline(
                y=pb_uniform_pop,
                line_dash="dash",
                annotation_text=f"Uniforme {pb_uniform_pop:.2%}",
            )
            fig.update_layout(height=340, yaxis_title="POP próximo sorteo")
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig, width="stretch")

        col_fw_c, col_fw_d = st.columns(2)
        with col_fw_c:
            st.dataframe(
                white_view[
                    [
                        "rank",
                        "number",
                        "pop_next_draw",
                        "pop_ci_low",
                        "pop_ci_high",
                        "uniform_pop_next_draw",
                        "pop_delta_pp",
                        "draw_prob",
                        "lift_vs_uniform_pct",
                        "z_score",
                        "draws_since_seen",
                    ]
                ].head(view_n),
                width="stretch",
                hide_index=True,
            )
        with col_fw_d:
            st.dataframe(
                pb_view[
                    [
                        "rank",
                        "number",
                        "pop_next_draw",
                        "pop_ci_low",
                        "pop_ci_high",
                        "uniform_pop_next_draw",
                        "pop_delta_pp",
                        "lift_vs_uniform_pct",
                        "z_score",
                        "draws_since_seen",
                    ]
                ].head(view_n),
                width="stretch",
                hide_index=True,
            )

elif page == "Validacion (Backtest)":
    st.subheader("Validación histórica")
    st.caption(
        "El modelo se recalcula antes de cada sorteo usando únicamente información disponible hasta ese momento. "
        "La primera parte calibra la intensidad y la última parte funciona como prueba fuera de muestra."
    )
    if backtest_summary.empty:
        st.warning("No hay suficientes sorteos de la matriz actual para ejecutar el backtest.")
    else:
        white_brier = backtest_summary.loc[backtest_summary["metric"].eq("White Brier")].iloc[0]
        pb_brier = backtest_summary.loc[backtest_summary["metric"].eq("Powerball Brier")].iloc[0]
        bt1, bt2, bt3, bt4 = st.columns(4)
        bt1.metric("Sorteos de calibración", f"{backtest_config.calibration_draws:,}")
        bt2.metric("Sorteos fuera de muestra", f"{backtest_config.holdout_draws:,}")
        bt3.metric("Intensidad white", f"{backtest_config.white_strength:.3f}")
        bt4.metric("Intensidad Powerball", f"{backtest_config.powerball_strength:.3f}")
        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
        result_col1.metric("Evidencia white", str(white_brier["evidence"]), f"Δ Brier {white_brier['delta']:+.6f}")
        result_col2.metric("Evidencia Powerball", str(pb_brier["evidence"]), f"Δ Brier {pb_brier['delta']:+.6f}")
        result_col3.metric(
            "Peso white",
            f"{backtest_config.white_model_weight:.0%}",
            f"Calibrado {backtest_config.white_calibrated_weight:.0%}",
        )
        result_col4.metric(
            "Peso Powerball",
            f"{backtest_config.powerball_model_weight:.0%}",
            f"Calibrado {backtest_config.powerball_calibrated_weight:.0%}",
        )

        st.markdown(
            "<div class='accuracy-note'><b>Regla de lectura:</b> Brier y log-loss deben ser menores que el "
            "uniforme. El intervalo de mejora debe quedar sobre cero para marcar evidencia. Si incluye cero, "
            "Accuracy v3 conserva solo 25% del peso calibrado; si el promedio no mejora, usa el uniforme.</div>",
            unsafe_allow_html=True,
        )

        summary_view = backtest_summary[
            [
                "metric",
                "uniform",
                "model",
                "delta",
                "improvement_ci_low",
                "improvement_ci_high",
                "model_win_rate",
                "relative_improvement_pct",
                "evidence",
            ]
        ].copy()
        summary_view[["uniform", "model", "delta", "improvement_ci_low", "improvement_ci_high"]] = summary_view[
            ["uniform", "model", "delta", "improvement_ci_low", "improvement_ci_high"]
        ].round(6)
        summary_view["model_win_rate"] = summary_view["model_win_rate"].map("{:.1%}".format)
        summary_view = summary_view.rename(
            columns={
                "metric": "métrica",
                "uniform": "uniforme",
                "model": "modelo",
                "delta": "diferencia",
                "improvement_ci_low": "mejora_intervalo_bajo",
                "improvement_ci_high": "mejora_intervalo_alto",
                "model_win_rate": "ventanas_ganadas",
                "relative_improvement_pct": "mejora_relativa_pct",
                "evidence": "evidencia",
            }
        )
        st.dataframe(summary_view, width="stretch", hide_index=True)

        chart_yearly = backtest_yearly.melt(
            id_vars=["year"],
            value_vars=["white_brier_uniform", "white_brier_model"],
            var_name="serie",
            value_name="brier",
        )
        fig = px.line(
            chart_yearly,
            x="year",
            y="brier",
            color="serie",
            markers=True,
            title="Brier white por año: modelo vs uniforme",
        )
        fig.update_layout(height=360, yaxis_title="Brier (menor es mejor)")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(backtest_yearly.round(6), width="stretch", hide_index=True)

        with st.expander("Ver resultados sorteo por sorteo"):
            st.dataframe(backtest_detail.round(6), width="stretch", hide_index=True)
            st.download_button(
                "Descargar backtest CSV",
                backtest_detail.to_csv(index=False).encode("utf-8"),
                file_name="powerball_backtest.csv",
                mime="text/csv",
            )

elif page == "Perfil y Calidad":
    st.subheader("Perfil y Calidad")
    profile = pd.DataFrame(
        {
            "metric": [
                "Rows",
                "First draw",
                "Last draw",
                "Era count",
                "Power Play missing rows",
                "White number max observed",
                "Powerball max observed",
            ],
            "value": [
                len(filtered),
                filtered["draw_date"].min().date(),
                filtered["draw_date"].max().date(),
                filtered["era"].nunique(),
                int(filtered["power_play"].isna().sum()),
                int(filtered[WHITE_COLS].max().max()),
                int(filtered["powerball"].max()),
            ],
        }
    )
    profile["value"] = profile["value"].astype(str)
    st.dataframe(profile, width="stretch", hide_index=True)
    era_counts = filtered["era"].value_counts().rename_axis("era").reset_index(name="draws")
    st.dataframe(era_counts, width="stretch", hide_index=True)
    st.markdown("**Data quality checks**")
    st.dataframe(quality_summary, width="stretch", hide_index=True)
    if not quality_issues.empty:
        st.warning("Rows with issues detected. Review before interpreting results.")
        st.dataframe(
            quality_issues[
                [
                    "draw_date",
                    *WHITE_COLS,
                    "powerball",
                    "era",
                    "duplicate_white_in_draw",
                    "duplicate_draw_date",
                    "white_out_of_range",
                    "powerball_out_of_range",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("No duplicate/out-of-range issues found in the current filtered dataset.")

elif page == "Frecuencia y Significancia":
    st.subheader("Frecuencia y Significancia")
    st.caption(
        "Los numeros jugables son enteros (`number`, `observed`). "
        "Los decimales (`expected`, `z_score`, `p_value`, `q_value`) son metricas estadisticas."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**White balls chi-square**: χ²={white_chi['chi2']:.2f}, df={int(white_chi['df'])}, p={format_p(white_chi['p_value'])}")
        fig = go.Figure()
        fig.add_bar(x=white_exp["number"], y=white_exp["observed"], name="Observed")
        fig.add_scatter(x=white_exp["number"], y=white_exp["expected_ci_high"], mode="lines", line=dict(width=0), showlegend=False)
        fig.add_scatter(
            x=white_exp["number"],
            y=white_exp["expected_ci_low"],
            name="95% CI",
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.15)",
            line=dict(width=0),
        )
        fig.add_scatter(x=white_exp["number"], y=white_exp["expected"], name="Expected", mode="lines+markers")
        fig.update_layout(height=360, xaxis_title="White ball", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")
        white_table = white_exp.sort_values("z_score", ascending=False)[
            ["number", "observed", "expected", "z_score", "p_value_two_sided", "q_value_fdr", "is_fdr_5pct"]
        ].head(15).copy()
        white_table["number"] = white_table["number"].astype(int)
        white_table["observed"] = white_table["observed"].astype(int)
        white_table["expected"] = white_table["expected"].round(2)
        white_table["z_score"] = white_table["z_score"].round(3)
        white_table["p_value_two_sided"] = white_table["p_value_two_sided"].round(4)
        white_table["q_value_fdr"] = white_table["q_value_fdr"].round(4)
        st.dataframe(white_table, width="stretch", hide_index=True)
    with col_b:
        st.markdown(f"**Powerball chi-square**: χ²={pb_chi['chi2']:.2f}, df={int(pb_chi['df'])}, p={format_p(pb_chi['p_value'])}")
        fig = go.Figure()
        fig.add_bar(x=pb_exp["number"], y=pb_exp["observed"], name="Observed")
        fig.add_scatter(x=pb_exp["number"], y=pb_exp["expected_ci_high"], mode="lines", line=dict(width=0), showlegend=False)
        fig.add_scatter(
            x=pb_exp["number"],
            y=pb_exp["expected_ci_low"],
            name="95% CI",
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(239, 85, 59, 0.15)",
            line=dict(width=0),
        )
        fig.add_scatter(x=pb_exp["number"], y=pb_exp["expected"], name="Expected", mode="lines+markers")
        fig.update_layout(height=360, xaxis_title="Powerball", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")
        pb_table = pb_exp.sort_values("z_score", ascending=False)[
            ["number", "observed", "expected", "z_score", "p_value_two_sided", "q_value_fdr", "is_fdr_5pct"]
        ].head(12).copy()
        pb_table["number"] = pb_table["number"].astype(int)
        pb_table["observed"] = pb_table["observed"].astype(int)
        pb_table["expected"] = pb_table["expected"].round(2)
        pb_table["z_score"] = pb_table["z_score"].round(3)
        pb_table["p_value_two_sided"] = pb_table["p_value_two_sided"].round(4)
        pb_table["q_value_fdr"] = pb_table["q_value_fdr"].round(4)
        st.dataframe(pb_table, width="stretch", hide_index=True)

elif page == "Diagnosticos":
    st.subheader("Diagnosticos")
    corr = white_exp["number"].corr(white_exp["z_score"]) if not white_exp.empty else np.nan
    top_bucket = bucket_stats.sort_values("pct_vs_expected", ascending=False).head(1)
    top_digit = digit_stats.sort_values("pct_vs_expected", ascending=False).head(1)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Corr(number, z-score)", "N/A" if pd.isna(corr) else f"{corr:.3f}")
    m2.metric("Top bucket", "N/A" if top_bucket.empty else f"{top_bucket.iloc[0]['bucket']} ({top_bucket.iloc[0]['pct_vs_expected']:+.2f}%)")
    m3.metric("Top last digit", "N/A" if top_digit.empty else f"{int(top_digit.iloc[0]['last_digit'])} ({top_digit.iloc[0]['pct_vs_expected']:+.2f}%)")
    m4.metric("Consistent sign (2+ eras)", f"{int(stability_df[(stability_df['active_eras'] >= 2) & (stability_df['consistent_sign'])].shape[0]):,}")
    col_diag_a, col_diag_b = st.columns(2)
    with col_diag_a:
        fig = px.scatter(white_exp.sort_values("number"), x="number", y="z_score", hover_data={"observed": True, "expected": ":.2f", "delta": ":.2f"})
        fig.update_layout(height=320, title="White number vs z-score")
        st.plotly_chart(fig, width="stretch")
    with col_diag_b:
        fig = px.bar(bucket_stats, x="bucket", y="pct_vs_expected", title="Bucket deviation vs expected (%)")
        fig.update_layout(height=320, yaxis_title="% vs expected")
        st.plotly_chart(fig, width="stretch")
    col_diag_c, col_diag_d = st.columns(2)
    with col_diag_c:
        fig = px.bar(digit_stats, x="last_digit", y="pct_vs_expected", title="Last-digit deviation vs expected (%)")
        fig.update_layout(height=320, yaxis_title="% vs expected")
        st.plotly_chart(fig, width="stretch")
    with col_diag_d:
        if stability_matrix.shape[1] > 2:
            heat = stability_matrix.set_index("number")
            fig = go.Figure(data=go.Heatmap(z=heat.to_numpy(), x=heat.columns.tolist(), y=heat.index.tolist(), colorscale="RdBu", zmid=0))
            fig.update_layout(height=320, title="Era stability heatmap (z-score)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Era-stability heatmap requires at least 2 active eras in current filters.")

elif page == "Recencia (Overdue)":
    st.subheader("Recencia (Overdue)")
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**White mas atrasadas**")
        st.dataframe(white_over.head(25), width="stretch", hide_index=True)
    with col_d:
        st.markdown("**Powerball mas atrasadas**")
        st.dataframe(pb_over.head(25), width="stretch", hide_index=True)

elif page == "Estructura y Combinaciones":
    st.subheader("Estructura y Combinaciones")
    col_e, col_f, col_g = st.columns(3)
    with col_e:
        fig = px.histogram(filtered, x="white_sum", nbins=25, title="Distribution of white-ball sum")
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")
    with col_f:
        odd_mix = filtered["odd_count"].value_counts().sort_index().rename_axis("odd_count").reset_index(name="draws")
        fig = px.bar(odd_mix, x="odd_count", y="draws", title="Odd/Even mix")
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")
    with col_g:
        cons = filtered["consecutive_pairs"].value_counts().sort_index().rename_axis("consecutive_pairs").reset_index(name="draws")
        fig = px.bar(cons, x="consecutive_pairs", y="draws", title="Consecutive pairs per draw")
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")
    col_h, col_i = st.columns(2)
    with col_h:
        repeat_dist = filtered["repeat_from_prev_draw"].value_counts().sort_index().rename_axis("repeat_from_prev_draw").reset_index(name="draws")
        fig = px.bar(repeat_dist, x="repeat_from_prev_draw", y="draws", title="Repeats from previous draw")
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")
    with col_i:
        range_dist = filtered["white_range"].value_counts().sort_index().rename_axis("white_range").reset_index(name="draws")
        fig = px.bar(range_dist, x="white_range", y="draws", title="Range width of white numbers")
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")
    col_j, col_k = st.columns(2)
    with col_j:
        st.markdown("**Pares ajustados por la matriz de cada era**")
        st.dataframe(pair_frequency(filtered, top_n=20), width="stretch", hide_index=True)
    with col_k:
        st.markdown("**Tríos ajustados por la matriz de cada era**")
        st.dataframe(triplet_frequency(filtered, top_n=15), width="stretch", hide_index=True)
    if not pair_matrix.empty:
        fig = go.Figure(
            data=go.Heatmap(
                z=pair_matrix.to_numpy(),
                x=pair_matrix.columns.tolist(),
                y=pair_matrix.index.tolist(),
                colorscale="Blues",
            )
        )
        fig.update_layout(height=430, title="Pair co-occurrence heatmap (top 20)")
        st.plotly_chart(fig, width="stretch")

elif page == "Equipos y Pre-tests":
    st.subheader("Equipos y pre-tests")
    st.markdown(
        "<div class='accuracy-note'><b>Uso correcto:</b> estos datos permiten buscar desviaciones por máquina "
        "y set, pero normalmente se publican después del sorteo. Por eso su validación es retrospectiva y no "
        "modifica el POP del próximo sorteo.</div>",
        unsafe_allow_html=True,
    )
    if equipment_df.empty:
        st.info("No fue posible cargar el reporte de pre-tests ni su copia local.")
    else:
        quality_map = equipment_quality.set_index("metric")["value"].to_dict()
        eq1, eq2, eq3, eq4 = st.columns(4)
        eq1.metric("Fuente", equipment_source_label)
        eq2.metric("Última fecha de equipos", str(pd.Timestamp(equipment_df["draw_date"].max()).date()))
        eq3.metric("Cobertura del CSV", f"{float(quality_map.get('coverage_pct', 0)):.1f}%")
        mismatch_total = int(quality_map.get("white_number_mismatches", 0)) + int(
            quality_map.get("powerball_mismatches", 0)
        )
        eq4.metric("Discordancias", f"{mismatch_total:,}")
        st.caption(
            "La fecha del reporte de equipos puede quedar detrás del CSV ganador por el tiempo de publicación del PDF."
        )
        if pd.Timestamp(equipment_df["draw_date"].max()) < pd.Timestamp(df["draw_date"].max()):
            st.info(
                f"El CSV ganador llega a {df['draw_date'].max().date()}, mientras el reporte de equipos llega a "
                f"{equipment_df['draw_date'].max().date()}. Los sorteos pendientes quedan sin equipo, no se imputan."
            )
        mismatch_rows = draws_with_equipment[
            draws_with_equipment["equipment_available"]
            & (~draws_with_equipment["white_numbers_match"] | ~draws_with_equipment["powerball_matches"])
        ]
        if not mismatch_rows.empty:
            with st.expander("Auditar discordancias entre las dos fuentes"):
                st.caption(
                    "Se conservan como discrepancias de fuente y no se corrigen automáticamente con valores inventados."
                )
                st.dataframe(
                    mismatch_rows[
                        [
                            "draw_date",
                            *WHITE_COLS,
                            "powerball",
                            "white_1",
                            "white_2",
                            "white_3",
                            "white_4",
                            "white_5",
                            "equipment_powerball",
                            "white_numbers_match",
                            "powerball_matches",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

        usage = equipment_usage_summary(equipment_filtered)
        if usage.empty:
            st.info("No hay filas de equipos para el rango filtrado.")
        else:
            st.markdown("### Uso de máquinas y sets")
            usage_type = st.selectbox(
                "Equipo a visualizar",
                options=usage["equipment_type"].drop_duplicates().tolist(),
                key="equipment_usage_type",
            )
            usage_view = usage[usage["equipment_type"].eq(usage_type)].sort_values("equipment_id")
            usage_view["equipment_id"] = usage_view["equipment_id"].astype(int).astype(str)
            fig = px.bar(
                usage_view,
                x="equipment_id",
                y="draws",
                hover_data=["first_draw", "last_draw"],
                title=f"Sorteos por {usage_type.lower()}",
            )
            fig.update_layout(height=340, xaxis_title="Identificador", yaxis_title="Sorteos")
            st.plotly_chart(fig, width="stretch")

            st.markdown("### Desviaciones observadas por equipo")
            dev_a, dev_b, dev_c = st.columns(3)
            with dev_a:
                deviation_ball_type = st.radio(
                    "Campo",
                    options=["White", "Powerball"],
                    horizontal=True,
                    key="equipment_deviation_ball_type",
                )
            if deviation_ball_type == "White":
                group_options = {
                    "Máquina white": "white_machine_id",
                    "Set white": "white_ball_set_id",
                }
                internal_ball_type = "white"
            else:
                group_options = {
                    "Máquina Powerball": "powerball_machine_id",
                    "Set Powerball": "powerball_ball_set_id",
                }
                internal_ball_type = "powerball"
            with dev_b:
                deviation_group_label = st.selectbox(
                    "Agrupar por",
                    options=list(group_options),
                    key="equipment_deviation_group",
                )
            with dev_c:
                min_equipment_draws = st.slider(
                    "Mínimo de sorteos por equipo",
                    min_value=10,
                    max_value=100,
                    value=30,
                    step=5,
                    key="equipment_min_draws",
                )
            deviation = equipment_number_deviation(
                equipment_filtered,
                ball_type=internal_ball_type,
                group_column=group_options[deviation_group_label],
                min_draws=int(min_equipment_draws),
            )
            if deviation.empty:
                st.info("Ningún equipo alcanza el mínimo de sorteos dentro del filtro actual.")
            else:
                equipment_ids = sorted(deviation["equipment_id"].astype(int).unique().tolist())
                selected_equipment_id = st.selectbox(
                    "Identificador a inspeccionar",
                    options=equipment_ids,
                    key="equipment_deviation_id",
                )
                deviation_view = deviation[deviation["equipment_id"].eq(selected_equipment_id)].copy()
                sig_count = int(deviation_view["is_fdr_5pct"].sum())
                st.metric("Números significativos después de FDR", sig_count)
                fig = px.bar(
                    deviation_view.sort_values("number"),
                    x="number",
                    y="z_score",
                    color="is_fdr_5pct",
                    title=f"Desviación por número: {deviation_group_label} {selected_equipment_id}",
                    hover_data={"observed": True, "expected": ":.2f", "q_value_fdr": ":.4f"},
                )
                fig.add_hline(y=0, line_width=1, line_color="#4b5563")
                fig.update_layout(height=360, yaxis_title="z-score")
                st.plotly_chart(fig, width="stretch")
                deviation_display = deviation_view.sort_values("z_score", ascending=False)[
                    ["number", "observed", "expected", "z_score", "q_value_fdr", "is_fdr_5pct"]
                ].copy()
                deviation_display["number"] = deviation_display["number"].astype(int)
                deviation_display["observed"] = deviation_display["observed"].astype(int)
                deviation_display[["expected", "z_score", "q_value_fdr"]] = deviation_display[
                    ["expected", "z_score", "q_value_fdr"]
                ].round(4)
                st.dataframe(deviation_display, width="stretch", hide_index=True)

        st.markdown("### Pre-tests frente al sorteo oficial")
        comparison = cached_pretest_comparison(equipment_filtered)
        if comparison.empty:
            st.info("No hay fechas con pre-tests y sorteo oficial dentro del filtro.")
        else:
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Fechas comparables", f"{len(comparison):,}")
            pc2.metric(
                "White vistos antes: observado",
                f"{comparison['white_draw_numbers_seen_in_pretests'].mean():.2f}",
            )
            pc3.metric("White vistos antes: esperado", f"{comparison['white_expected_seen'].mean():.2f}")
            overlap_distribution = (
                comparison["white_draw_numbers_seen_in_pretests"]
                .value_counts()
                .sort_index()
                .rename_axis("números_del_draw_vistos_en_pretests")
                .reset_index(name="sorteos")
            )
            fig = px.bar(
                overlap_distribution,
                x="números_del_draw_vistos_en_pretests",
                y="sorteos",
                title="Coincidencias entre pre-tests y draw",
            )
            fig.update_layout(height=330)
            st.plotly_chart(fig, width="stretch")
            with st.expander("Ver comparaciones recientes"):
                st.dataframe(comparison.tail(30).sort_values("draw_date", ascending=False), width="stretch", hide_index=True)

        st.markdown("### Validación walk-forward de máquina + set")
        st.caption(
            "Usa la matriz actual completa. Para cada sorteo estima tasas con datos anteriores solamente y las "
            "suaviza hacia el uniforme con un prior de 100 sorteos. Menor Brier es mejor."
        )
        equipment_detail, equipment_summary, equipment_yearly = cached_equipment_backtest(equipment_df)
        if equipment_summary.empty:
            st.info("No hay suficientes sorteos para validar la señal de equipos.")
        else:
            summary_display = equipment_summary.copy()
            summary_display[
                [
                    "uniform_brier",
                    "equipment_brier",
                    "improvement",
                    "improvement_ci_low",
                    "improvement_ci_high",
                ]
            ] = summary_display[
                [
                    "uniform_brier",
                    "equipment_brier",
                    "improvement",
                    "improvement_ci_low",
                    "improvement_ci_high",
                ]
            ].round(7)
            summary_display["model_win_rate"] = summary_display["model_win_rate"].map("{:.1%}".format)
            st.dataframe(summary_display, width="stretch", hide_index=True)
            equipment_yearly_chart = equipment_yearly.melt(
                id_vars=["year"],
                value_vars=["white_improvement", "pb_improvement"],
                var_name="campo",
                value_name="mejora_brier",
            )
            fig = px.bar(
                equipment_yearly_chart,
                x="year",
                y="mejora_brier",
                color="campo",
                barmode="group",
                title="Mejora anual frente al uniforme",
            )
            fig.add_hline(y=0, line_width=1, line_color="#4b5563")
            fig.update_layout(height=360, yaxis_title="Uniforme - equipo")
            st.plotly_chart(fig, width="stretch")
            with st.expander("Ver backtest de equipos sorteo por sorteo"):
                st.dataframe(equipment_detail.tail(250), width="stretch", hide_index=True)
                st.download_button(
                    "Descargar backtest de equipos",
                    equipment_detail.to_csv(index=False).encode("utf-8"),
                    file_name="powerball_equipment_backtest.csv",
                    mime="text/csv",
                )

elif page == "Simulador Fisico":
    st.subheader("Laboratorio físico")
    st.markdown(
        "<div class='accuracy-note'><b>No es forecast:</b> este laboratorio responde qué ocurriría bajo una "
        "diferencia de masa elegida o medida. Sin pesos por número, el resultado permanece uniforme y no entra "
        "en el POP.</div>",
        unsafe_allow_html=True,
    )
    sim_mode = st.radio(
        "Fuente del escenario",
        options=["Uniforme", "Escenario hipotético", "CSV de pesos medidos"],
        horizontal=True,
        key="sim_mode_nav",
    )
    measured_weights_df = None
    max_sim_number = int(white_exp["number"].max())
    beta = 0.0
    if sim_mode == "Escenario hipotético":
        scenario_numbers = st.multiselect(
            "Bolas afectadas en el escenario",
            options=list(range(1, max_sim_number + 1)),
            default=[1],
            key="physical_scenario_numbers",
        )
        scenario_direction = st.radio(
            "Diferencia de masa",
            options=["Más ligera", "Más pesada"],
            horizontal=True,
            key="physical_scenario_direction",
        )
        scenario_delta = st.slider(
            "Diferencia respecto a 80 g (%)",
            min_value=0.10,
            max_value=1.00,
            value=0.375,
            step=0.025,
            key="physical_scenario_delta",
        )
        signed_delta = -float(scenario_delta) if scenario_direction == "Más ligera" else float(scenario_delta)
        measured_weights_df = build_hypothetical_weight_table(
            max_number=max_sim_number,
            target_numbers=[int(number) for number in scenario_numbers],
            nominal_weight_g=80.0,
            relative_delta_pct=signed_delta,
        )
        beta = st.slider(
            "Sensibilidad asumida (log-odds por 1% de masa)",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
            key="physical_beta",
        )
    elif sim_mode == "CSV de pesos medidos":
        weight_upload = st.file_uploader(
            "CSV medido: columnas number,weight",
            type=["csv"],
            key="weights_upload_nav",
        )
        if weight_upload is not None:
            try:
                measured_weights_df, missing_weight_rows = parse_weights_csv_bytes(
                    weight_upload.getvalue(), max_sim_number
                )
                st.success(
                    f"Pesos válidos: {len(measured_weights_df) - missing_weight_rows}; "
                    f"valores imputados con la media: {missing_weight_rows}."
                )
            except ValueError as exc:
                st.warning(f"El archivo de pesos fue ignorado: {exc}")
        beta = st.slider(
            "Sensibilidad asumida (log-odds por 1% de masa)",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
            key="physical_beta_measured",
        )
    include_weight = sim_mode != "Uniforme" and measured_weights_df is not None
    sim_df, sim_metrics = physical_bias_projection(
        white_exp,
        include_weight=include_weight,
        include_wear=False,
        beta=float(beta),
        gamma=0.0,
        measured_weights=measured_weights_df,
    )
    st.caption(
        "Fórmula de estrés: log(p ajustada) = log(p uniforme) - sensibilidad × diferencia de masa (%). "
        "El coeficiente es hipotético; no se ha medido para la máquina Halogen de Powerball."
    )
    sim_m1, sim_m2, sim_m3, sim_m4 = st.columns(4)
    sim_m1.metric("Fuente de masa", sim_metrics.get("weight_source", "N/A"))
    sim_m2.metric("χ² uniforme", f"{sim_metrics.get('chi2_uniform', np.nan):.2f}")
    sim_m3.metric("χ² escenario", f"{sim_metrics.get('chi2_adjusted', np.nan):.2f}")
    sim_m4.metric("Cambio de ajuste χ²", f"{sim_metrics.get('chi2_delta', np.nan):+.2f}")
    col_sim_a, col_sim_b = st.columns(2)
    with col_sim_a:
        sim_plot = sim_df.sort_values("number")
        fig = go.Figure()
        fig.add_scatter(x=sim_plot["number"], y=sim_plot["baseline_prob"], name="Uniforme", mode="lines")
        fig.add_scatter(x=sim_plot["number"], y=sim_plot["adjusted_prob"], name="Escenario", mode="lines")
        fig.update_layout(height=320, title="Uniforme frente al escenario", yaxis_title="Probabilidad relativa")
        st.plotly_chart(fig, width="stretch")
    with col_sim_b:
        top_lift = sim_df.reindex(sim_df["prob_lift_pct"].abs().sort_values(ascending=False).index).head(15)
        fig = px.bar(top_lift, x="number", y="prob_lift_pct", title="Mayor cambio absoluto (%)")
        fig.update_layout(height=320, yaxis_title="% frente al uniforme")
        st.plotly_chart(fig, width="stretch")
    physical_display = sim_df[
        [
            "number",
            "weight_delta_pct",
            "observed",
            "expected",
            "adjusted_expected",
            "prob_lift_pct",
            "expected_delta",
        ]
    ].copy()
    physical_display["number"] = physical_display["number"].astype(int)
    physical_display[["weight_delta_pct", "expected", "adjusted_expected", "prob_lift_pct", "expected_delta"]] = (
        physical_display[
            ["weight_delta_pct", "expected", "adjusted_expected", "prob_lift_pct", "expected_delta"]
        ].round(4)
    )
    st.dataframe(physical_display, width="stretch", hide_index=True)
    default_sim_df = sim_df

elif page == "Rolling":
    st.subheader("Rolling")
    max_white_num = int(filtered["white_pool_max"].max())
    default_num = min(21, max_white_num)
    rolling_window = st.slider("Rolling window (draws)", min_value=26, max_value=156, value=52, step=13, key="rolling_window_nav")
    rolling_mode = st.radio(
        "Numero para rolling",
        options=["Manual", "Top forecast", "Bottom forecast", "Most overdue"],
        horizontal=True,
        key="rolling_mode_nav",
    )
    if rolling_mode == "Manual" or white_forecast.empty:
        number_selected = st.slider("Inspect white-ball rolling hits", min_value=1, max_value=max_white_num, value=default_num, key="rolling_manual_nav")
    elif rolling_mode == "Top forecast":
        top_options = white_forecast.sort_values("inclusion_prob_next_draw", ascending=False).head(20)["number"].tolist()
        number_selected = st.selectbox("Pick from top forecast", options=top_options, index=0, key="rolling_top_pick_nav")
    elif rolling_mode == "Bottom forecast":
        bottom_options = white_forecast.sort_values("inclusion_prob_next_draw", ascending=True).head(20)["number"].tolist()
        number_selected = st.selectbox("Pick from bottom forecast", options=bottom_options, index=0, key="rolling_bottom_pick_nav")
    else:
        overdue_options = white_over.head(20)["number"].astype(int).tolist()
        number_selected = st.selectbox("Pick from most overdue", options=overdue_options, index=0, key="rolling_overdue_pick_nav")
    roll_df = rolling_hits_white(filtered, int(number_selected), window=int(rolling_window))
    fig = px.line(
        roll_df,
        x="draw_date",
        y="rolling_hits",
        title=f"White ball {int(number_selected)}: rolling hits over last {int(rolling_window)} draws",
    )
    fig.update_layout(height=340)
    st.plotly_chart(fig, width="stretch")

else:
    st.subheader("Datos y Exportes")
    st.dataframe(
        filtered[["draw_date", *WHITE_COLS, "powerball", "power_play", "weekday", "era", "year"]],
        width="stretch",
        hide_index=True,
    )
    csv_export = filtered[["draw_date", *WHITE_COLS, "powerball", "power_play", "weekday", "era", "year"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered dataset", csv_export, file_name="powerball_filtered.csv", mime="text/csv")
    equipment_export_detail = pd.DataFrame()
    if not equipment_df.empty:
        equipment_export_detail, _, _ = cached_equipment_backtest(equipment_df)
        st.download_button(
            "Descargar equipos y pre-tests normalizados",
            equipment_filtered.to_csv(index=False).encode("utf-8"),
            file_name="powerball_equipment_pretests.csv",
            mime="text/csv",
        )
    if OPENPYXL_OK:
        excel_bytes = build_excel_export(
            filtered=filtered,
            white_exp=white_exp,
            pb_exp=pb_exp,
            white_over=white_over,
            pb_over=pb_over,
            white_score=white_score,
            bucket_stats=bucket_stats,
            digit_stats=digit_stats,
            stability=stability_df,
            quality_summary=quality_summary,
            quality_issues=quality_issues,
            white_forecast=white_forecast,
            pb_forecast=pb_forecast,
            tickets_forecast=tickets_forecast,
            sim_white_number_freq=sim_white_number_freq,
            sim_powerball_freq=sim_powerball_freq,
            sim_pair_freq=sim_pair_freq,
            sim_triplet_freq=sim_triplet_freq,
            sim_df=default_sim_df.sort_values("number").reset_index(drop=True),
            equipment_rows=equipment_filtered,
            equipment_quality=equipment_quality,
            equipment_backtest=equipment_export_detail,
        )
        st.download_button(
            "Download full analysis (Excel)",
            excel_bytes,
            file_name="powerball_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Install `openpyxl` to enable Excel export.")

st.stop()
