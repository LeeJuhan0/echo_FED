#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a 2x2 combined figure:
 - top-left: Statement (smoothed mean curve passing through yearly means + ribbon)
 - top-right: P&B  (same)
 - bottom-left: Policy (shows three mean curves: red, green, blue; CI/ribbon only for blue)
 - bottom-right: histogram (y = frequency, x = sentiment score)

Modifications in this version:
 - DO NOT rescale sentiment series to [0,1]; use original cleaned numeric values.
 - Set y-axis limits for the three time-series panels to [-0.85, 0.85].
 - Histogram x-axis range is set to [-0.85, 0.85] so it matches the sentiment value range.
 - All other behavior preserved (PCHIP through yearly means, CI ribbon for Policy only).
 - CHANGE: Histogram now uses only the data used in subplots 2 and 3 (P&B and Policy),
           and its title is updated to "Distribution of Series Used in Subplots 2 and 3".
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy import stats
import io
from textwrap import dedent

# ---------- Inline data (unchanged; uses exactly your inputs) ----------
policy_text = """date\tstatment\tP&B\tPolicy
date\tstatment\tP&B\tPolicy
2017-02\t0.3\t0.32\t0.29
2017-03\t0.42\t0.4\t0.35
2017-05\t0.35\t0.36\t0.33
2017-06\t0.43\t0.35\t0.3
2017-07\t0.36\t0.32\t0.24
2017-09\t0.4\t0.36\t0.28
2017-11\t0.4\t0.43\t0.38
2017-12\t0.55\t0.55\t0.5
2018-01\t0.64\t0.65\t0.55
2018-03\t0.45\t0.52\t0.52
2018-05\t0.45\t0.4\t0.3
2018-06\t0.74\t0.7\t0.66
2018-08\t0.75\t0.67\t0.63
2018-09\t0.84\t0.72\t0.76
2018-11\t0.7\t0.65\t0.5
2018-12\t0.7\t0.6\t0.43
2019-01\t0.45\t0.48\t0.36
2019-06\t0.2\t0.12\t0.04
2019-07\t0.3\t0.2\t0.08
2019-09\t0.2\t0.2\t0.08
2019-10\t0.25\t0.22\t0.08
2019-12\t0.24\t0.25\t0.12
2020-01\t0.2\t0.25\t0.12
2020-03\t0.3\t-0.1\t-0.24
2020-03\t0.3\t-0.4\t-0.65
2020-04\t-0.18\t-0.45\t-0.65
2020-06\t-0.4\t-0.5\t-0.55
2020-07\t-0.82\t-0.22\t-0.35
2020-09\t-0.62\t-0.1\t-0.3
2020-11\t-0.4\t-0.1\t-0.2
2020-12\t-0.3\t-0.1\t-0.12
2021-01\t-0.3\t-0.15\t-0.09
2021-03\t-0.3\t0.2\t0.2
2021-04\t-0.3\t0.22\t0.3
2021-06\t0.1\t0.435\t0.45
2021-07\t0.22\t0.45\t0.55
2021-09\t0.44\t0.25\t0.25
2021-11\t0.45\t0.25\t0.3
2021-12\t0.35\t0.27\t0.27
2022-01\t0.35\t0.2\t0.18
2022-03\t0.35\t0.04\t-0.01
2022-05\t0.3\t-0.18\t-0.12
2022-06\t0.1\t-0.2\t-0.19
2022-07\t-0.12\t-0.4\t-0.48
2022-09\t-0.12\t-0.35\t-0.5
2022-11\t-0.32\t-0.25\t-0.4
2022-12\t-0.23\t-0.25\t-0.4
2023-02\t-0.25\t-0.14\t-0.15
2023-03\t-0.15\t-0.18\t-0.3
2023-05\t-0.15\t0.04\t-0.2
2023-06\t-0.1\t0.05\t-0.12
2023-07\t0.1\t0.05\t-0.04
2023-09\t0.15\t0.1\t-0.05
2023-11\t0.12\t0.1\t-0.05
2023-12\t0.2\t-0.08\t-0.2
2024-01\t0.2\t0.2\t0.05
2024-03\t-0.1\t0.29\t0.1
2024-05\t0.25\t0.1\t-0.02
2024-06\t0.35\t0.27\t0.14
2024-07\t0.25\t0.2\t0.07
2024-09\t0.35\t0.4\t0.24
2024-11\t0.25\t0.25\t0.15
2024-12\t0.32\t0.32\t0.22
2025-01\t0.35\t0.25\t0.1
2025-03\t0.3\t0.25\t0.16
2025-05\t0.34\t0.1\t-0.05
2025-06\t0.3\t0.22\t0.1
2025-07\t0.15\t-0.08\t-0.12
"""
cpi_text = """Prev_CPI_Value\tNext_CPI_Value\tFOMC_date
250.083\t251.143\t2017-02-01
251.143\t251.290\t2017-03-01
251.290\t251.642\t2017-04-01
251.642\t251.835\t2017-05-01
251.835\t252.014\t2017-06-01
252.014\t251.936\t2017-07-01
251.936\t252.460\t2017-08-01
252.460\t252.941\t2017-09-01
252.941\t253.638\t2017-10-01
253.638\t253.492\t2017-11-01
253.492\t253.558\t2017-12-01
253.558\t254.638\t2018-01-01
254.638\t255.783\t2018-02-01
255.783\t256.610\t2018-03-01
256.610\t257.025\t2018-04-01
257.025\t257.469\t2018-05-01
257.469\t257.697\t2018-06-01
257.697\t257.867\t2018-07-01
257.867\t258.012\t2018-08-01
258.012\t258.429\t2018-09-01
258.429\t259.063\t2018-10-01
259.063\t259.105\t2018-11-01
259.105\t259.083\t2018-12-01
259.083\t260.122\t2019-01-01
260.122\t261.114\t2019-02-01
261.114\t261.836\t2019-03-01
261.836\t262.332\t2019-04-01
262.332\t262.590\t2019-05-01
262.590\t263.177\t2019-06-01
263.177\t263.566\t2019-07-01
263.566\t264.169\t2019-08-01
264.169\t264.522\t2019-09-01
264.522\t265.059\t2019-10-01
265.059\t265.108\t2019-11-01
265.108\t264.935\t2019-12-01
264.935\t266.004\t2020-01-01
266.004\t267.268\t2020-02-01
267.268\t267.312\t2020-03-01
267.268\t267.312\t2020-03-01
267.312\t266.089\t2020-04-01
266.089\t265.799\t2020-05-01
265.799\t266.302\t2020-06-01
266.302\t267.703\t2020-07-01
267.703\t268.756\t2020-08-01
268.756\t269.054\t2020-09-01
269.054\t269.328\t2020-10-01
269.328\t269.473\t2020-11-01
269.473\t269.226\t2020-12-01
269.226\t269.755\t2021-01-01
269.755\t270.696\t2021-02-01
270.696\t271.713\t2021-03-01
271.713\t273.968\t2021-04-01
273.968\t275.893\t2021-05-01
275.893\t278.218\t2021-06-01
278.218\t279.146\t2021-07-01
279.146\t279.507\t2021-08-01
279.507\t279.884\t2021-09-01
279.884\t281.617\t2021-10-01
281.617\t282.754\t2021-11-01
282.754\t283.908\t2021-12-01
283.908\t285.996\t2022-01-01
285.996\t288.059\t2022-02-01
288.059\t289.305\t2022-03-01
289.305\t290.846\t2022-04-01
290.846\t292.506\t2022-05-01
292.506\t294.680\t2022-06-01
294.680\t295.646\t2022-07-01
295.646\t297.178\t2022-08-01
297.178\t298.442\t2022-09-01
298.442\t299.315\t2022-10-01
299.315\t299.600\t2022-11-01
299.600\t300.113\t2022-12-01
300.113\t301.962\t2023-01-01
301.962\t304.011\t2023-02-01
304.011\t305.476\t2023-03-01
305.476\t306.899\t2023-04-01
306.899\t308.096\t2023-05-01
308.096\t308.910\t2023-06-01
308.910\t309.402\t2023-07-01
309.402\t310.103\t2023-08-01
310.103\t310.817\t2023-09-01
310.817\t311.380\t2023-10-01
311.380\t311.606\t2023-11-01
311.606\t311.907\t2023-12-01
311.907\t313.623\t2024-01-01
313.623\t315.419\t2024-02-01
315.419\t317.088\t2024-03-01
317.088\t317.978\t2024-04-01
317.978\t318.629\t2024-05-01
318.629\t319.003\t2024-06-01
319.003\t319.214\t2024-07-01
319.214\t320.017\t2024-08-01
320.017\t321.109\t2024-09-01
321.109\t321.758\t2024-10-01
321.758\t321.947\t2024-11-01
321.947\t322.007\t2024-12-01
322.007\t323.842\t2025-01-01
323.842\t325.252\t2025-02-01
325.252\t325.933\t2025-03-01
325.933\t326.815\t2025-04-01
326.815\t327.509\t2025-05-01
327.509\t328.364\t2025-06-01
328.364\t328.980\t2025-07-01
"""
policy_df_raw = pd.read_csv(io.StringIO(dedent(policy_text)), sep=r"\s+")
cpi_df = pd.read_csv(io.StringIO(dedent(cpi_text)), sep=r"\s+")
cpi_df['FOMC_date'] = pd.to_datetime(cpi_df['FOMC_date'])

# ---------- Build merged monthly mapping (month-level aggregation for CPI) ----------
cpi_df['month'] = cpi_df['FOMC_date'].dt.strftime('%Y-%m')
cpi_month = cpi_df.groupby('month', as_index=False).agg({
    'Prev_CPI_Value': 'mean',
    'Next_CPI_Value': 'mean',
    'FOMC_date': 'min'
}).rename(columns={'month': 'date'})

policy_df = policy_df_raw.copy()
policy_df.columns = policy_df.columns.str.strip()
policy_df = policy_df.rename(columns={'statment': 'Statement', 'date': 'date'})
policy_df['date'] = policy_df['date'].str.strip()

merged = policy_df.merge(cpi_month[['date', 'Prev_CPI_Value', 'Next_CPI_Value', 'FOMC_date']],
                         on='date', how='left')

if merged['Prev_CPI_Value'].isna().any():
    print("Warning: some policy rows did not find CPI match. See rows:")
    print(merged[merged['Prev_CPI_Value'].isna()][['date']])

merged['FOMC_date'] = pd.to_datetime(merged['FOMC_date'])
merged = merged.sort_values('FOMC_date').reset_index(drop=True)

# ---------- USER EDITABLE: subplot titles and x-axis labels ----------
prev_scatter_titles = [
    "Statement(St) vs Previous Month Core CPI",
    "St+Theory(Th) vs Previous Month Core CPI",
    "St+Th+Policy vs Previous Month Core CPI"
]
prev_scatter_xlabels = ["Sentiment Score", "Sentiment Score", "Sentiment Score"]
prev_hist_title = "Previous Month Core CPI Distribution (all observations)"
prev_hist_xlabel = "Previous Month Core CPI"

next_scatter_titles = [
    "Statement(St) vs Next Month Core CPI",
    "St+Theory(Th) vs Next Month Core CPI",
    "St+Th+Policy vs Next Month Core CPI"
]
next_scatter_xlabels = ["Sentiment Score", "Sentiment Score", "Sentiment Score"]
next_hist_title = "Next Core CPI Distribution (all observations)"
next_hist_xlabel = "Next Month Core CPI"

# ---------- plotting helpers and new compute function ----------
DENSE_POINTS = 300
HALFWIDTH_S_FACTOR = 8.0
HIST_BINS = 20
HIST_ALPHA = 0.7
HIST_COLOR = "grey"

def clean_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = re.sub(r",", "", s)
    s = re.sub(r"[\u2212\u2012\u2013\u2014]", "-", s)
    s = re.sub(r"[^0-9\-\.\+]", "", s)
    if s == "" or s in ["+", "-"]:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def normalize_date_char(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    s = re.sub(r"[./]", "-", s)
    s = re.sub(r"^([0-9]{4})-([0-9])$", r"\1-0\2", s)
    s = re.sub(r"^([0-9]{4})-([0-9]{2})$", r"\1-\2-01", s)
    return s

def smooth_halfwidth_on_dense(years, half_vals, dense_x, s_factor=HALFWIDTH_S_FACTOR):
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

def compute_yearly_stats(df, col):
    """Compute yearly means, ci95, and return arrays plus original pts."""
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

def eval_mean_on_dense(yr, dense_x):
    """Given yearly df (with Year and mean), evaluate a smooth mean on dense_x that passes through yearly means."""
    years = yr["Year"].to_numpy()
    mean_vals = yr["mean"].to_numpy()
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

# Set default y-limits to the user-requested range (-0.85, 0.85)
DEFAULT_Y_LIMS = (-0.85, 0.85)

def plot_yearly_mean_ci_single(ax, yr_stats, dense_x, mean_dense, half_dense, title, line_color, show_points=True, y_limits=DEFAULT_Y_LIMS):
    """Plot ribbon (if half_dense provided), mean_dense line and optionally yearly points & raw jittered points."""
    ymin_dense = mean_dense - half_dense
    ymax_dense = mean_dense + half_dense
    ax.fill_between(dense_x, ymin_dense, ymax_dense, color="grey", alpha=0.45, linewidth=0)
    ax.plot(dense_x, mean_dense, color=line_color, linewidth=2.2)
    # yearly markers
    years = yr_stats["yr"]["Year"].to_numpy()
    mean_vals = yr_stats["yr"]["mean"].to_numpy()
    if show_points:
        ax.scatter(years, mean_vals, color=line_color, edgecolor="white", zorder=5, s=54)
    # jittered raw points (from pts)
    rng = np.random.default_rng(2024)
    jitter = rng.uniform(-0.08, 0.08, size=len(yr_stats["pts"]))
    ax.scatter(yr_stats["pts"]["Year"] + jitter, yr_stats["pts"]["val"], color="black", s=14, alpha=0.9)
    ax.set_xticks(np.arange(2017, 2026))
    ax.set_xlim(2016.9, 2025.1)
    ax.set_ylim(y_limits)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Sentiment Score")

def plot_mean_line_only(ax, yr_stats, dense_x, mean_dense, line_color, y_limits=DEFAULT_Y_LIMS):
    """Plot only a mean line and (optionally) yearly mean markers (we will not draw jittered raw for these auxiliary lines)."""
    ax.plot(dense_x, mean_dense, color=line_color, linewidth=2.0)
    # show yearly mean markers for this series (smaller)
    years = yr_stats["yr"]["Year"].to_numpy()
    mean_vals = yr_stats["yr"]["mean"].to_numpy()
    ax.scatter(years, mean_vals, color=line_color, edgecolor="white", zorder=5, s=40)
    ax.set_xticks(np.arange(2017, 2026))
    ax.set_xlim(2016.9, 2025.1)
    ax.set_ylim(y_limits)

def plot_hist(ax, df, cols, bins=HIST_BINS, color=HIST_COLOR, alpha=HIST_ALPHA, x_limits=DEFAULT_Y_LIMS):
    vals = []
    for c in cols:
        if c in df.columns:
            vals.append(df[c].dropna().values)
    if len(vals) == 0:
        ax.text(0.5, 0.5, "No data for histogram", ha="center", va="center")
        return
    all_vals = np.concatenate(vals)
    ax.hist(all_vals, bins=bins, color=color, alpha=alpha, edgecolor="black")
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of All Series")
    # Set histogram x-axis to match sentiment range requested
    ax.set_xlim(x_limits)

# ---------- Preprocess input-like data frame (mimic your earlier cleaning) ----------
# For this self-contained script we will create a small "df" by parsing dates from 'date' in policy_df.
policy_df = policy_df.copy()
# Build Date and Year similar to your normalize_date_char pipeline (policy 'date' is YYYY-MM)
def normalize_date_char_local(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    s = re.sub(r"[./]", "-", s)
    # add day if missing
    if re.match(r"^\d{4}-\d{2}$", s):
        s = s + "-01"
    return s

policy_df['date_chr'] = policy_df['date'].apply(normalize_date_char_local)
policy_df['Date'] = pd.to_datetime(policy_df['date_chr'], errors='coerce', infer_datetime_format=True)
policy_df['Year'] = policy_df['Date'].dt.year

# Clean numeric sentiment columns (DO NOT rescale to [0,1]; keep original numeric range)
for col in ["Statement", "P&B", "Policy"]:
    if col in policy_df.columns:
        policy_df[col] = policy_df[col].apply(clean_num)
    else:
        policy_df[col] = np.nan

policy_df = policy_df[(~policy_df["Year"].isna()) & (policy_df["Year"] >= 2017) & (policy_df["Year"] <= 2025)].copy()

# ---------- Create combined panel figure ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ax00 = axes[0, 0]
ax01 = axes[0, 1]
ax10 = axes[1, 0]
ax11 = axes[1, 1]

# compute yearly stats for each series
stmt_stats = compute_yearly_stats(policy_df, "Statement")
pnb_stats = compute_yearly_stats(policy_df, "P&B")
pol_stats = compute_yearly_stats(policy_df, "Policy")

# define common dense_x over the whole plotting period (keeps alignment)
dense_x = np.linspace(2017.0, 2025.0, DENSE_POINTS)

# PANEL 1: Statement (red) - single series plotting with its ribbon
if stmt_stats is not None:
    mean_dense_stmt = eval_mean_on_dense(stmt_stats["yr"], dense_x)
    half_dense_stmt = smooth_halfwidth_on_dense(stmt_stats["yr"]["Year"].to_numpy(), stmt_stats["yr"]["ci95"].to_numpy(), dense_x)
    plot_yearly_mean_ci_single(ax00, stmt_stats, dense_x, mean_dense_stmt, half_dense_stmt, title="LLM[Statement(St)]", line_color="red", show_points=True)
else:
    ax00.text(0.5, 0.5, "No data: Statement", ha="center", va="center")
    ax00.set_ylim(DEFAULT_Y_LIMS)

# PANEL 2: P&B (green) - single series plotting with its ribbon
if pnb_stats is not None:
    mean_dense_pnb = eval_mean_on_dense(pnb_stats["yr"], dense_x)
    half_dense_pnb = smooth_halfwidth_on_dense(pnb_stats["yr"]["Year"].to_numpy(), pnb_stats["yr"]["ci95"].to_numpy(), dense_x)
    plot_yearly_mean_ci_single(ax01, pnb_stats, dense_x, mean_dense_pnb, half_dense_pnb, title="GraphRAG[St+Theory(Th)]", line_color="green", show_points=True)
else:
    ax01.text(0.5, 0.5, "No data: P&B", ha="center", va="center")
    ax01.set_ylim(DEFAULT_Y_LIMS)

# PANEL 3 (bottom-left): combine three mean lines (red, green, blue), keep ribbon only for Policy (blue)
if (stmt_stats is None) and (pnb_stats is None) and (pol_stats is None):
    ax10.text(0.5, 0.5, "No data for any series", ha="center", va="center")
    ax10.set_ylim(DEFAULT_Y_LIMS)
else:
    # plot Statement mean (red) if exists
    if stmt_stats is not None:
        mean_dense_stmt = eval_mean_on_dense(stmt_stats["yr"], dense_x)
        plot_mean_line_only(ax10, stmt_stats, dense_x, mean_dense_stmt, line_color="red")
    # plot P&B mean (green) if exists
    if pnb_stats is not None:
        mean_dense_pnb = eval_mean_on_dense(pnb_stats["yr"], dense_x)
        plot_mean_line_only(ax10, pnb_stats, dense_x, mean_dense_pnb, line_color="green")
    # plot Policy with ribbon (blue)
    if pol_stats is not None:
        mean_dense_pol = eval_mean_on_dense(pol_stats["yr"], dense_x)
        half_dense_pol = smooth_halfwidth_on_dense(pol_stats["yr"]["Year"].to_numpy(), pol_stats["yr"]["ci95"].to_numpy(), dense_x)
        plot_yearly_mean_ci_single(ax10, pol_stats, dense_x, mean_dense_pol, half_dense_pol, title="GraphRAG[St+Th+Policy]", line_color="blue", show_points=True)
    # adjust legend for combined panel: create custom legend entries
    legend_items = []
    if stmt_stats is not None:
        l1 = plt.Line2D([0], [0], color="red", lw=2)
        legend_items.append((l1, "LLM[Statement(St)]; mean"))
    if pnb_stats is not None:
        l2 = plt.Line2D([0], [0], color="green", lw=2)
        legend_items.append((l2, "GraphRAG[St+Theory(Th)]; mean"))
    if pol_stats is not None:
        l3 = plt.Line2D([0], [0], color="blue", lw=2)
        legend_items.append((l3, "GraphRAG[St+Th+Policy]; mean+CI"))
    if legend_items:
        handles, labels = zip(*legend_items)
        ax10.legend(handles, labels, fontsize=9, loc="upper right")

# PANEL 4: histogram including only data used in subplots 2 and 3 (P&B and Policy)
plot_hist(ax11, policy_df, ["P&B", "Policy"], bins=HIST_BINS, color=HIST_COLOR, alpha=HIST_ALPHA, x_limits=DEFAULT_Y_LIMS)
ax11.set_title("Distribution of Series Used in Subplots (1,2) and (2,1)")

plt.tight_layout()
plt.show()
plt.close(fig)