#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a 2x2 combined figure (data loaded from CSV files):
 - top-left: Statement (smoothed mean curve passing through yearly means + ribbon)
 - top-right: Theory   (same)
 - bottom-left: Policy (shows three mean curves: red=Statement, green=Theory, blue=Policy; CI/ribbon only for Policy)
 - bottom-right: histogram (y = frequency, x = sentiment score)

Notes
- Sentiment series are NOT rescaled; original numeric values are used.
- Y-axis limits for time-series panels are fixed to [-0.85, 0.85].
- Histogram x-axis range is set to [-0.85, 0.85].
- Uses monthly sentiment CSV with columns: date, statment, Theory, Policy. (FSR/SLOOS can exist but are unused here.)
- CPI CSV is read (to follow your I/O pattern) but not used for this figure.

Input files (edit the paths below if needed):
- Sentiment: C:\\Users\\HUFS_MATH\\IdeaProjects\\FOMC_Graphrag\\data\\sentiment_13000.csv
- CPI:       C:\\Users\\HUFS_MATH\\IdeaProjects\\FOMC_Graphrag\\data\\cpi.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, UnivariateSpline

# --------------- Configuration ---------------
SENTIMENT_CSV = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\sentiment_13+19_llm.csv"
CPI_CSV       = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\cpi.csv"

OUT_FIG = "combined_2x2.png"

# Panel/plot constants
DENSE_POINTS = 300
HALFWIDTH_S_FACTOR = 8.0
HIST_BINS = 20
HIST_ALPHA = 0.7
HIST_COLOR = "grey"
DEFAULT_Y_LIMS = (-0.85, 0.85)
XTICK_START, XTICK_END = 2017, 2025
XLIM_MIN, XLIM_MAX = 2016.9, 2025.1

# --------------- Utilities ---------------
def read_csv_kr(path: str | Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV {path} with tried encodings. Last error: {last_err}")

def clean_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = re.sub(r",", "", s)
    s = re.sub(r"[\u2212\u2012\u2013\u2014]", "-", s)  # normalize unicode minus
    s = re.sub(r"[^0-9\-\.\+]", "", s)
    if s == "" or s in ["+", "-"]:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def normalize_month_to_day1(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    s = re.sub(r"[./]", "-", s)
    # If YYYY-MM only, append -01
    if re.match(r"^\d{4}-\d{2}$", s):
        s = s + "-01"
    return s

def compute_yearly_stats(df, col):
    """Compute yearly means, ci95 for a column, return dict with yearly df and raw points."""
    pts = df[["Year", col]].dropna().rename(columns={col: "val"})
    if pts.empty:
        return None
    yr = pts.groupby("Year", as_index=False).agg(
        n=("val", "size"),
        mean=("val", "mean"),
        sd=("val", "std")
    )
    yr["sd"] = yr["sd"].fillna(0.0)
    yr["se"] = yr["sd"] / np.sqrt(yr["n"])
    yr["ci95"] = 1.96 * yr["se"]
    yr["ymin"] = yr["mean"] - yr["ci95"]
    yr["ymax"] = yr["mean"] + yr["ci95"]
    yr = yr.sort_values("Year").reset_index(drop=True)
    return {"yr": yr, "pts": pts}

def eval_mean_on_dense(yr_df, dense_x):
    """PCHIP curve through yearly means; exact at integer years."""
    years = yr_df["Year"].to_numpy()
    mean_vals = yr_df["mean"].to_numpy()
    if len(years) == 1:
        mean_dense = np.full_like(dense_x, mean_vals[0], dtype=float)
    elif len(years) == 2:
        mean_dense = np.interp(dense_x, years, mean_vals)
    else:
        pchip = PchipInterpolator(years, mean_vals, extrapolate=True)
        mean_dense = pchip(dense_x)
    # enforce exact match at integer years
    for yv, mv in zip(years, mean_vals):
        idx = np.abs(dense_x - yv).argmin()
        mean_dense[idx] = mv
    return mean_dense

def smooth_halfwidth_on_dense(years, half_vals, dense_x, s_factor=HALFWIDTH_S_FACTOR):
    """Smooth the CI half-width over dense_x using a spline; non-negative; exact at integer years."""
    if len(years) < 2:
        return np.full_like(dense_x, half_vals[0] if len(half_vals) == 1 else 0.0, dtype=float)
    v = np.var(half_vals) if np.isfinite(np.var(half_vals)) else 0.0
    s_param = v * len(years) * s_factor
    k = min(3, max(1, len(years) - 1))
    try:
        sp = UnivariateSpline(years, half_vals, s=s_param, k=k)
        half_dense = sp(dense_x)
    except Exception:
        half_dense = np.interp(dense_x, years, half_vals)
    for yv, hv in zip(years, half_vals):
        idx = np.abs(dense_x - yv).argmin()
        half_dense[idx] = hv
    half_dense = np.maximum(half_dense, 0.0)
    return half_dense

def plot_yearly_mean_ci_single(ax, yr_stats, dense_x, mean_dense, half_dense, title, line_color, show_points=True, y_limits=DEFAULT_Y_LIMS):
    """Ribbon + mean line + yearly points + raw jittered points, with set limits and labels."""
    ymin_dense = mean_dense - half_dense
    ymax_dense = mean_dense + half_dense
    ax.fill_between(dense_x, ymin_dense, ymax_dense, color="grey", alpha=0.45, linewidth=0)
    ax.plot(dense_x, mean_dense, color=line_color, linewidth=2.2)
    # yearly markers
    years = yr_stats["yr"]["Year"].to_numpy()
    mean_vals = yr_stats["yr"]["mean"].to_numpy()
    if show_points:
        ax.scatter(years, mean_vals, color=line_color, edgecolor="white", zorder=5, s=54)
    # jittered raw points
    rng = np.random.default_rng(2024)
    jitter = rng.uniform(-0.08, 0.08, size=len(yr_stats["pts"]))
    ax.scatter(yr_stats["pts"]["Year"] + jitter, yr_stats["pts"]["val"], color="black", s=14, alpha=0.9)
    ax.set_xticks(np.arange(XTICK_START, XTICK_END + 1))
    ax.set_xlim(XLIM_MIN, XLIM_MAX)
    ax.set_ylim(y_limits)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Sentiment Score")

def plot_mean_line_only(ax, yr_stats, dense_x, mean_dense, line_color, y_limits=DEFAULT_Y_LIMS):
    """Only mean line + yearly markers, for auxiliary series in the combined panel."""
    ax.plot(dense_x, mean_dense, color=line_color, linewidth=2.0)
    years = yr_stats["yr"]["Year"].to_numpy()
    mean_vals = yr_stats["yr"]["mean"].to_numpy()
    ax.scatter(years, mean_vals, color=line_color, edgecolor="white", zorder=5, s=40)
    ax.set_xticks(np.arange(XTICK_START, XTICK_END + 1))
    ax.set_xlim(XLIM_MIN, XLIM_MAX)
    ax.set_ylim(y_limits)

def plot_hist(ax, df, cols, bins=HIST_BINS, color=HIST_COLOR, alpha=HIST_ALPHA, x_limits=DEFAULT_Y_LIMS):
    vals = []
    for c in cols:
        if c in df.columns:
            vals.append(df[c].dropna().values)
    if not vals:
        ax.text(0.5, 0.5, "No data for histogram", ha="center", va="center")
        return
    all_vals = np.concatenate(vals)
    ax.hist(all_vals, bins=bins, color=color, alpha=alpha, edgecolor="black")
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of All Series")
    ax.set_xlim(x_limits)

# --------------- Load data ---------------
# Sentiment: requires date, statment (or Statement), Theory, Policy
sent_df = read_csv_kr(SENTIMENT_CSV)
sent_df.columns = [c.strip() for c in sent_df.columns]

# Normalize expected columns
col_map = {}
if "statment" in sent_df.columns:
    col_map["statment"] = "Statement"
elif "Statement" in sent_df.columns:
    col_map["Statement"] = "Statement"
if "Theory" in sent_df.columns:
    col_map["Theory"] = "Theory"
if "Policy" in sent_df.columns:
    col_map["Policy"] = "Policy"
if "date" in sent_df.columns:
    col_map["date"] = "date"

sent_df = sent_df.rename(columns=col_map)

required_cols = ["date", "Statement", "Theory", "Policy"]
missing = [c for c in required_cols if c not in sent_df.columns]
if missing:
    raise ValueError(f"Missing required columns in sentiment CSV: {missing}")

# Keep only needed columns and clean
sent_df = sent_df[required_cols].copy()
for c in ["Statement", "Theory", "Policy"]:
    sent_df[c] = sent_df[c].apply(clean_num)

# Parse month string and compute Year
sent_df["date_chr"] = sent_df["date"].apply(normalize_month_to_day1)
sent_df["Date"] = pd.to_datetime(sent_df["date_chr"], errors="coerce")
sent_df["Year"] = sent_df["Date"].dt.year
sent_df = sent_df[(~sent_df["Year"].isna()) & (sent_df["Year"] >= XTICK_START) & (sent_df["Year"] <= XTICK_END)].copy()

# CPI: load (not used for plotting; read to match your I/O expectations)
try:
    cpi_df = read_csv_kr(CPI_CSV)
    # Optional normalization if you wish to use later; not required here.
except Exception as e:
    print(f"[WARN] Could not read CPI CSV ({CPI_CSV}): {e}")

# --------------- Compute yearly stats ---------------
stmt_stats = compute_yearly_stats(sent_df, "Statement")
theory_stats = compute_yearly_stats(sent_df, "Theory")
policy_stats = compute_yearly_stats(sent_df, "Policy")

# Common dense x-domain
dense_x = np.linspace(float(XTICK_START), float(XTICK_END), DENSE_POINTS)

# --------------- Build figure ---------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ax00, ax01 = axes[0, 0], axes[0, 1]
ax10, ax11 = axes[1, 0], axes[1, 1]

# Top-left: Statement (red) with ribbon
if stmt_stats is not None:
    mean_dense_stmt = eval_mean_on_dense(stmt_stats["yr"], dense_x)
    half_dense_stmt = smooth_halfwidth_on_dense(
        stmt_stats["yr"]["Year"].to_numpy(), stmt_stats["yr"]["ci95"].to_numpy(), dense_x
    )
    plot_yearly_mean_ci_single(
        ax00, stmt_stats, dense_x, mean_dense_stmt, half_dense_stmt,
        title="LLM[Statement(St)]", line_color="red", show_points=True, y_limits=DEFAULT_Y_LIMS
    )
else:
    ax00.text(0.5, 0.5, "No data: Statement", ha="center", va="center")
    ax00.set_ylim(DEFAULT_Y_LIMS)

# Top-right: Theory (green) with ribbon
if theory_stats is not None:
    mean_dense_theory = eval_mean_on_dense(theory_stats["yr"], dense_x)
    half_dense_theory = smooth_halfwidth_on_dense(
        theory_stats["yr"]["Year"].to_numpy(), theory_stats["yr"]["ci95"].to_numpy(), dense_x
    )
    plot_yearly_mean_ci_single(
        ax01, theory_stats, dense_x, mean_dense_theory, half_dense_theory,
        title="GraphRAG[St+Theory(Th)]", line_color="green", show_points=True, y_limits=DEFAULT_Y_LIMS
    )
else:
    ax01.text(0.5, 0.5, "No data: Theory", ha="center", va="center")
    ax01.set_ylim(DEFAULT_Y_LIMS)

# Bottom-left: Combined (red=Statement mean, green=Theory mean, blue=Policy with ribbon)
if stmt_stats is None and theory_stats is None and policy_stats is None:
    ax10.text(0.5, 0.5, "No data for any series", ha="center", va="center")
    ax10.set_ylim(DEFAULT_Y_LIMS)
else:
    # Statement mean (red)
    if stmt_stats is not None:
        mean_dense_stmt = eval_mean_on_dense(stmt_stats["yr"], dense_x)
        plot_mean_line_only(ax10, stmt_stats, dense_x, mean_dense_stmt, line_color="red")
    # Theory mean (green)
    if theory_stats is not None:
        mean_dense_theory = eval_mean_on_dense(theory_stats["yr"], dense_x)
        plot_mean_line_only(ax10, theory_stats, dense_x, mean_dense_theory, line_color="green")
    # Policy with ribbon (blue)
    if policy_stats is not None:
        mean_dense_policy = eval_mean_on_dense(policy_stats["yr"], dense_x)
        half_dense_policy = smooth_halfwidth_on_dense(
            policy_stats["yr"]["Year"].to_numpy(), policy_stats["yr"]["ci95"].to_numpy(), dense_x
        )
        plot_yearly_mean_ci_single(
            ax10, policy_stats, dense_x, mean_dense_policy, half_dense_policy,
            title="GraphRAG[St+Th+Policy]", line_color="blue", show_points=True, y_limits=DEFAULT_Y_LIMS
        )
    # Legend
    legend_items = []
    if stmt_stats is not None:
        legend_items.append((plt.Line2D([0], [0], color="red", lw=2), "LLM[Statement(St)]; mean"))
    if theory_stats is not None:
        legend_items.append((plt.Line2D([0], [0], color="green", lw=2), "GraphRAG[St+Theory(Th)]; mean"))
    if policy_stats is not None:
        legend_items.append((plt.Line2D([0], [0], color="blue", lw=2), "GraphRAG[St+Th+Policy]; mean+CI"))
    if legend_items:
        handles, labels = zip(*legend_items)
        ax10.legend(handles, labels, fontsize=9, loc="upper right")

# Bottom-right: Histogram of Statement/Theory/Policy values
plot_hist(ax11, sent_df, ["Statement", "Theory", "Policy"],
          bins=HIST_BINS, color=HIST_COLOR, alpha=HIST_ALPHA, x_limits=DEFAULT_Y_LIMS)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
print(f"[DONE] Saved figure to: {OUT_FIG}")
plt.show()