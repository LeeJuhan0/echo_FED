#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create two 2x2 figures from external CSVs:
 - Prev group (Prev_CPI_Value): 3 scatter/regression panels + histogram
 - Next group (Next_CPI_Value): 3 scatter/regression panels + histogram

Updates from the inline-data version:
- Read CPI from:    C:\\Users\\HUFS_MATH\\IdeaProjects\\FOMC_Graphrag\\data\\cpi.csv
- Read sentiment from: C:\\Users\\HUFS_MATH\\IdeaProjects\\FOMC_Graphrag\\data\\sentiment_13000.csv
- Use columns date, statment(->Statement), Theory, Policy (logic kept; second x-axis uses Theory)
- The 4th-panel histogram shows the combined distribution of the three x-variables (Statement, Theory, Policy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# --------------- Configuration (edit if paths differ) ---------------
SENTIMENT_CSV = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\sentiment_13+19.csv"
CPI_CSV       = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\cpi.csv"

# Output images
OUT_PREV = "prev_group_2x2.png"
OUT_NEXT = "next_group_2x2.png"

# --------------- Helpers ---------------
def read_csv_kr(path: str | Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV {path} with tried encodings. Last error: {last_err}")

def fit_line(x, y):
    m, b = np.polyfit(x, y, 1)
    r, p = stats.pearsonr(x, y)
    return m, b, r, p

def plot_scatter_reg(ax, df, xcol, ycol, title, xlabel, ylabel, early_mask, late_mask,
                     early_color='#1f77b4', late_color='#d62728', overall_color='#2ca02c'):
    # scatter by segments
    ax.scatter(df.loc[early_mask, xcol], df.loc[early_mask, ycol],
               color=early_color, edgecolors='white', linewidth=0.5, alpha=0.85, label='2017-2022')
    ax.scatter(df.loc[late_mask, xcol], df.loc[late_mask, ycol],
               color=late_color, edgecolors='white', linewidth=0.5, alpha=0.85, label='2023-2025')

    # segment fits
    def plot_seg(mask, color, label):
        xs = df.loc[mask, xcol].to_numpy(dtype=float)
        ys = df.loc[mask, ycol].to_numpy(dtype=float)
        xs = xs[~np.isnan(xs)]
        ys = ys[~np.isnan(ys)]
        if len(xs) < 2 or len(ys) < 2:
            return None
        m, b, r, p = fit_line(xs, ys)
        xs_plot = np.linspace(np.nanmin(xs), np.nanmax(xs), 120)
        ax.plot(xs_plot, m * xs_plot + b, color=color, lw=1.6, label=f"{label} (r={r:.2f})")
        return (m, b, r, p)

    stats_early = plot_seg(early_mask, early_color, "regression 17-22")
    stats_late = plot_seg(late_mask, late_color, "regression 23-25")

    # overall fit
    xs_all = df[xcol].to_numpy(dtype=float)
    ys_all = df[ycol].to_numpy(dtype=float)
    xs_all = xs_all[~np.isnan(xs_all)]
    ys_all = ys_all[~np.isnan(ys_all)]
    if len(xs_all) >= 2 and len(ys_all) >= 2:
        m_all, b_all, r_all, p_all = fit_line(xs_all, ys_all)
        xs_plot = np.linspace(np.nanmin(xs_all), np.nanmax(xs_all), 200)
        ax.plot(xs_plot, m_all * xs_plot + b_all, color=overall_color, lw=1.8, linestyle='--',
                label=f"regression 17-25 (r={r_all:.2f})")

    # labels, title, limits, grid
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.85, 0.85)
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    return {"early": stats_early, "late": stats_late}

def plot_hist(ax, values, bins=20, color='grey', xlabel='Sentiment Score', title='Histogram'):
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    ax.hist(vals, bins=bins, color=color, alpha=0.8, edgecolor='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(alpha=0.2)

# --------------- Load data ---------------
# Sentiment: expects columns: date, statment, Theory, Policy (FSR/SLOOS are optional and unused here)
sent_df = read_csv_kr(SENTIMENT_CSV)
sent_df.columns = [c.strip() for c in sent_df.columns]
# normalize expected columns
col_map = {}
if "statment" in sent_df.columns:
    col_map["statment"] = "Statement"
elif "Statement" in sent_df.columns:
    col_map["Statement"] = "Statement"
# Keep Theory and Policy as-is
if "Theory" in sent_df.columns:
    col_map["Theory"] = "Theory"
if "Policy" in sent_df.columns:
    col_map["Policy"] = "Policy"
if "date" in sent_df.columns:
    col_map["date"] = "date"

sent_df = sent_df.rename(columns=col_map)
needed = ["date", "Statement", "Theory", "Policy"]
missing = [c for c in needed if c not in sent_df.columns]
if missing:
    raise ValueError(f"Missing required columns in sentiment CSV: {missing}")

# keep only needed cols
sent_df = sent_df[needed].copy()
# coerce numerics
for c in ["Statement", "Theory", "Policy"]:
    sent_df[c] = pd.to_numeric(sent_df[c], errors="coerce")
# trim/normalize date to YYYY-MM
sent_df["date"] = sent_df["date"].astype(str).str.strip()

# CPI: expects columns: Prev_CPI_Value, Next_CPI_Value, FOMC_date
cpi_df = read_csv_kr(CPI_CSV)
cpi_df.columns = [c.strip() for c in cpi_df.columns]
# standardize CPI column names if slightly different
cpi_col_map = {}
for cand in cpi_df.columns:
    key = cand.strip().lower().replace(" ", "_")
    if key in {"prev_cpi_value", "prev_cpi"}:
        cpi_col_map[cand] = "Prev_CPI_Value"
    elif key in {"next_cpi_value", "next_cpi"}:
        cpi_col_map[cand] = "Next_CPI_Value"
    elif key in {"fomc_date", "fomcdate", "date"}:
        cpi_col_map[cand] = "FOMC_date"
cpi_df = cpi_df.rename(columns=cpi_col_map)

cpi_needed = ["Prev_CPI_Value", "Next_CPI_Value", "FOMC_date"]
cpi_missing = [c for c in cpi_needed if c not in cpi_df.columns]
if cpi_missing:
    raise ValueError(f"Missing required columns in CPI CSV: {cpi_missing}")

cpi_df["FOMC_date"] = pd.to_datetime(cpi_df["FOMC_date"], errors="coerce")
# Build month aggregation (YYYY-MM)
cpi_df["month"] = cpi_df["FOMC_date"].dt.strftime("%Y-%m")
cpi_month = (
    cpi_df.groupby("month", as_index=False)
    .agg({
        "Prev_CPI_Value": "mean",
        "Next_CPI_Value": "mean",
        "FOMC_date": "min"
    })
    .rename(columns={"month": "date"})
)

# Merge on YYYY-MM
merged = sent_df.merge(
    cpi_month[["date", "Prev_CPI_Value", "Next_CPI_Value,? FOMC_date".split(",? ")[0], "FOMC_date"]],
    on="date", how="left"
)

# Warn if unmatched
if merged["Prev_CPI_Value"].isna().any():
    print("Warning: some sentiment rows did not find CPI match. Rows:")
    print(merged.loc[merged["Prev_CPI_Value"].isna(), ["date"]].drop_duplicates())

# Ensure datetime and sort by FOMC_date
merged["FOMC_date"] = pd.to_datetime(merged["FOMC_date"], errors="coerce")
merged = merged.sort_values("FOMC_date").reset_index(drop=True)

# --------------- Plot settings (user-editable text only) ---------------
prev_scatter_titles = [
    "LLM[Statement(St)] vs Previous Month CPI",
    "GraphRAG[St+Theory(Th)] vs Previous Month CPI",
    "GraphRAG[St+Th+Policy] vs Previous Month CPI"
]
prev_scatter_xlabels = ["Sentiment Score", "Sentiment Score", "Sentiment Score"]
prev_hist_title = "Distribution of Statement/Theory/Policy scores"
prev_hist_xlabel = "Sentiment Score"

next_scatter_titles = [
    "LLM[Statement(St)] vs Next Month CPI",
    "GraphRAG[St+Theory(Th)] vs Next Month CPI",
    "GraphRAG[St+Th+Policy] vs Next Month CPI"
]
next_scatter_xlabels = ["Sentiment Score", "Sentiment Score", "Sentiment Score"]
next_hist_title = "Distribution of Statement/Theory/Policy scores"
next_hist_xlabel = "Sentiment Score"

# --------------- Build masks ---------------
merged["year"] = merged["FOMC_date"].dt.year
early_mask = merged["year"] <= 2022
late_mask  = merged["year"] >= 2023

# --------------- Figure generator ---------------
def make_2x2_for(ycol, scatter_titles, scatter_xlabels, hist_title, hist_xlabel, out_fname):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()
    stats_dict = {}

    # three scatter panels: Statement, Theory, Policy
    xcols = ["Statement", "Theory", "Policy"]
    for i, xcol in enumerate(xcols):
        title = scatter_titles[i]
        xlabel = scatter_xlabels[i]
        ylabel = "Previous Month CPI" if ycol.startswith("Prev") else "Next Month CPI"
        stats_dict[f"panel_{i+1}"] = plot_scatter_reg(
            axes_flat[i], merged, xcol, ycol, title, xlabel, ylabel, early_mask, late_mask
        )

    # 4th panel: histogram of combined x variables
    combined_x = np.concatenate([
        merged["Statement"].to_numpy(dtype=float),
        merged["Theory"].to_numpy(dtype=float),
        merged["Policy"].to_numpy(dtype=float)
    ])
    plot_hist(axes_flat[3], combined_x, bins=20, color='grey', xlabel=hist_xlabel, title=hist_title)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_fname, dpi=300)
    print(f"Saved {out_fname}")
    plt.show()
    return stats_dict

# --------------- Create and save figures ---------------
prev_stats = make_2x2_for(
    ycol="Prev_CPI_Value",
    scatter_titles=prev_scatter_titles,
    scatter_xlabels=prev_scatter_xlabels,
    hist_title=prev_hist_title,
    hist_xlabel=prev_hist_xlabel,
    out_fname=OUT_PREV
)
next_stats = make_2x2_for(
    ycol="Next_CPI_Value",
    scatter_titles=next_scatter_titles,
    scatter_xlabels=next_scatter_xlabels,
    hist_title=next_hist_title,
    hist_xlabel=next_hist_xlabel,
    out_fname=OUT_NEXT
)

# Optional: print a small summary of stats availability
print("\nPanel fit availability summary (None means not enough data for segment fit):")
for k, v in prev_stats.items():
    print("Prev", k, v)
for k, v in next_stats.items():
    print("Next", k, v)