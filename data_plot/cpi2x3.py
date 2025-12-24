#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# --------------- Configuration (edit if paths differ) ---------------
SENTIMENT_CSV = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\sentiment_13+19_llm.csv"
CPI_CSV       = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\cpi.csv"

# Output image
OUT_3x2 = "cpi_prev_next_3x2.png"

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

def fit_line(x: np.ndarray, y: np.ndarray):
    """Return slope/intercept + Pearson r,p for valid paired data."""
    m, b = np.polyfit(x, y, 1)
    r, p = stats.pearsonr(x, y)
    return m, b, r, p

def plot_scatter_reg(ax, df, xcol, ycol, title, xlabel, ylabel, early_mask, late_mask,
                     early_color='#1f77b4', late_color='#d62728', overall_color='#2ca02c'):
    # ----- scatter by segments -----
    ax.scatter(df.loc[early_mask, xcol], df.loc[early_mask, ycol],
               color=early_color, edgecolors='white', linewidth=0.5, alpha=0.85, label='2017-2022')
    ax.scatter(df.loc[late_mask, xcol], df.loc[late_mask, ycol],
               color=late_color, edgecolors='white', linewidth=0.5, alpha=0.85, label='2023-2025')

    # ----- segment fit helper (IMPORTANT: filter x,y together) -----
    def plot_seg(mask, color, label):
        xs = df.loc[mask, xcol].to_numpy(dtype=float)
        ys = df.loc[mask, ycol].to_numpy(dtype=float)

        valid = (~np.isnan(xs)) & (~np.isnan(ys))
        xs = xs[valid]
        ys = ys[valid]

        if len(xs) < 2:
            return None

        m, b, r, p = fit_line(xs, ys)
        xs_plot = np.linspace(np.min(xs), np.max(xs), 120)
        ax.plot(xs_plot, m * xs_plot + b, color=color, lw=1.6, label=f"{label} (r={r:.2f})")
        return (m, b, r, p)

    stats_early = plot_seg(early_mask, early_color, "regression 17-22")
    stats_late  = plot_seg(late_mask,  late_color,  "regression 23-25")

    # ----- overall fit -----
    xs_all = df[xcol].to_numpy(dtype=float)
    ys_all = df[ycol].to_numpy(dtype=float)
    valid_all = (~np.isnan(xs_all)) & (~np.isnan(ys_all))
    xs_all = xs_all[valid_all]
    ys_all = ys_all[valid_all]

    if len(xs_all) >= 2:
        m_all, b_all, r_all, p_all = fit_line(xs_all, ys_all)
        xs_plot = np.linspace(np.min(xs_all), np.max(xs_all), 200)
        ax.plot(xs_plot, m_all * xs_plot + b_all, color=overall_color, lw=1.8, linestyle='--',
                label=f"regression 17-25 (r={r_all:.2f})")

    # ----- cosmetics -----
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.85, 0.85)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    return {"early": stats_early, "late": stats_late}

# --------------- Load data ---------------
# Sentiment: expects columns: date, statment(->Statement), Theory, Policy
sent_df = read_csv_kr(SENTIMENT_CSV)
sent_df.columns = [c.strip() for c in sent_df.columns]

# normalize expected columns
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

needed = ["date", "Statement", "Theory", "Policy"]
missing = [c for c in needed if c not in sent_df.columns]
if missing:
    raise ValueError(f"Missing required columns in sentiment CSV: {missing}")

sent_df = sent_df[needed].copy()

# coerce numerics
for c in ["Statement", "Theory", "Policy"]:
    sent_df[c] = pd.to_numeric(sent_df[c], errors="coerce")

# normalize date (YYYY-MM)
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

# Merge on YYYY-MM  (FIXED)
merged = sent_df.merge(
    cpi_month[["date", "Prev_CPI_Value", "Next_CPI_Value", "FOMC_date"]],
    on="date", how="left"
)

# Warn if unmatched
if merged["Prev_CPI_Value"].isna().any():
    print("Warning: some sentiment rows did not find CPI match. Rows:")
    print(merged.loc[merged["Prev_CPI_Value"].isna(), ["date"]].drop_duplicates())

# Ensure datetime and sort
merged["FOMC_date"] = pd.to_datetime(merged["FOMC_date"], errors="coerce")
merged = merged.sort_values("FOMC_date").reset_index(drop=True)

# --------------- Build masks ---------------
merged["year"] = merged["FOMC_date"].dt.year
early_mask = merged["year"] <= 2022
late_mask  = merged["year"] >= 2023

# --------------- 3x2 Plot (Prev on left, Next on right) ---------------
def make_3x2_prev_next(out_fname: str):
    # rows: Statement -> Theory -> Policy
    row_specs = [
        ("Statement", "LLM[Statement(St)]"),
        ("Theory",     "GraphRAG[St+Theory(Th)]"),
        ("Policy",     "GraphRAG[St+Th+Policy]")
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))  # 3 rows x 2 cols

    for r, (xcol, row_title_prefix) in enumerate(row_specs):
        # ---- Left column: Previous CPI ----
        plot_scatter_reg(
            ax=axes[r, 0],
            df=merged,
            xcol=xcol,
            ycol="Prev_CPI_Value",
            title=f"{row_title_prefix} vs Previous Month CPI",
            xlabel="Sentiment Score",
            ylabel="Previous Month CPI",
            early_mask=early_mask,
            late_mask=late_mask
        )

        # ---- Right column: Next CPI ----
        plot_scatter_reg(
            ax=axes[r, 1],
            df=merged,
            xcol=xcol,
            ycol="Next_CPI_Value",
            title=f"{row_title_prefix} vs Next Month CPI",
            xlabel="Sentiment Score",
            ylabel="Next Month CPI",
            early_mask=early_mask,
            late_mask=late_mask
        )

    plt.tight_layout()
    fig.savefig(out_fname, dpi=300)
    print(f"Saved {out_fname}")
    plt.show()

# --------------- Run ---------------
make_3x2_prev_next(OUT_3x2)
