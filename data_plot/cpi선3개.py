#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create two 2x2 figures from external CSVs:
 - Prev group (Prev_CPI_Value)
 - Next group (Next_CPI_Value)

Changes applied:
- Remove histogram panel
- Panel layout:
    (1,1) Statement
    (1,2) lg
    (2,1) Theory
    (2,2) Policy
- (1,2) uses lg column with same plotting logic as (1,1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ---------------- Configuration ----------------
SENTIMENT_CSV = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\lg_sentiment.csv"
CPI_CSV       = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\cpi.csv"

OUT_PREV = "prev_group_2x2.png"
OUT_NEXT = "next_group_2x2.png"

# ---------------- Helpers ----------------
def read_csv_kr(path: str | Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV {path}. Last error: {last_err}")

def fit_line(x, y):
    m, b = np.polyfit(x, y, 1)
    r, p = stats.pearsonr(x, y)
    return m, b, r, p

def plot_scatter_reg(ax, df, xcol, ycol, title, xlabel, ylabel,
                     early_mask, late_mask,
                     early_color='#1f77b4',
                     late_color='#d62728',
                     overall_color='#2ca02c'):

    ax.scatter(df.loc[early_mask, xcol], df.loc[early_mask, ycol],
               color=early_color, edgecolors='white', linewidth=0.5,
               alpha=0.85, label='2017–2022')

    ax.scatter(df.loc[late_mask, xcol], df.loc[late_mask, ycol],
               color=late_color, edgecolors='white', linewidth=0.5,
               alpha=0.85, label='2023–2025')

    def plot_seg(mask, color, label):
        xs = df.loc[mask, xcol].to_numpy(dtype=float)
        ys = df.loc[mask, ycol].to_numpy(dtype=float)
        msk = ~np.isnan(xs) & ~np.isnan(ys)
        xs, ys = xs[msk], ys[msk]
        if len(xs) < 2:
            return None
        m, b, r, p = fit_line(xs, ys)
        xp = np.linspace(xs.min(), xs.max(), 120)
        ax.plot(xp, m * xp + b, color=color, lw=1.6,
                label=f"{label} (r={r:.2f})")
        return (m, b, r, p)

    stats_early = plot_seg(early_mask, early_color, "regression 17–22")
    stats_late  = plot_seg(late_mask, late_color,  "regression 23–25")

    xs_all = df[xcol].to_numpy(dtype=float)
    ys_all = df[ycol].to_numpy(dtype=float)
    msk = ~np.isnan(xs_all) & ~np.isnan(ys_all)
    if msk.sum() >= 2:
        m, b, r, p = fit_line(xs_all[msk], ys_all[msk])
        xp = np.linspace(xs_all[msk].min(), xs_all[msk].max(), 200)
        ax.plot(xp, m * xp + b, color=overall_color, lw=1.8,
                linestyle='--', label=f"regression 17–25 (r={r:.2f})")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.85, 0.85)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    return {"early": stats_early, "late": stats_late}

# ---------------- Load sentiment data ----------------
sent_df = read_csv_kr(SENTIMENT_CSV)
sent_df.columns = [c.strip() for c in sent_df.columns]

sent_df = sent_df.rename(columns={
    "statment": "Statement",
    "date": "date"
})

needed = ["date", "Statement", "Theory", "Policy", "lg"]
missing = [c for c in needed if c not in sent_df.columns]
if missing:
    raise ValueError(f"Missing columns in sentiment CSV: {missing}")

sent_df = sent_df[needed].copy()
for c in ["Statement", "Theory", "Policy", "lg"]:
    sent_df[c] = pd.to_numeric(sent_df[c], errors="coerce")

sent_df["date"] = sent_df["date"].astype(str).str.strip()

# ---------------- Load CPI data ----------------
cpi_df = read_csv_kr(CPI_CSV)
cpi_df.columns = [c.strip() for c in cpi_df.columns]

cpi_df = cpi_df.rename(columns={
    "Prev_CPI_Value": "Prev_CPI_Value",
    "Next_CPI_Value": "Next_CPI_Value",
    "FOMC_date": "FOMC_date"
})

cpi_df["FOMC_date"] = pd.to_datetime(cpi_df["FOMC_date"], errors="coerce")
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

merged = sent_df.merge(
    cpi_month,
    on="date",
    how="left"
)

merged["FOMC_date"] = pd.to_datetime(merged["FOMC_date"])
merged = merged.sort_values("FOMC_date").reset_index(drop=True)

merged["year"] = merged["FOMC_date"].dt.year
early_mask = merged["year"] <= 2022
late_mask  = merged["year"] >= 2023

# ---------------- Plot generator ----------------
def make_2x2_for(ycol, titles, out_fname):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    xcols = ["Statement", "lg", "Theory", "Policy"]

    stats_out = {}
    for i, xcol in enumerate(xcols):
        stats_out[f"panel_{i+1}"] = plot_scatter_reg(
            axes[i],
            merged,
            xcol,
            ycol,
            titles[i],
            "Sentiment Score",
            "Previous Month CPI" if ycol.startswith("Prev") else "Next Month CPI",
            early_mask,
            late_mask
        )

    plt.tight_layout()
    fig.savefig(out_fname, dpi=300)
    plt.show()
    return stats_out

# ---------------- Titles ----------------
prev_titles = [
    "LLM(gpt5)[Statement(St)] vs Previous Month CPI",
    "LLM(lg)[Statement(St)] vs Previous Month CPI",
    "GraphRAG(gpt5)[St+Theory(Th)] vs Previous Month CPI",
    "GraphRAG(gpt5)[St+Th+Policy] vs Previous Month CPI"
]

next_titles = [
    "LLM(gpt5)[Statement(St)] vs Next Month CPI",
    "LLM(lg)[Statement(St)] vs Next Month CPI",
    "GraphRAG(gpt5)[St+Theory(Th)] vs Next Month CPI",
    "GraphRAG(gpt5)[St+Th+Policy] vs Next Month CPI"
]

# ---------------- Run ----------------
prev_stats = make_2x2_for("Prev_CPI_Value", prev_titles, OUT_PREV)
next_stats = make_2x2_for("Next_CPI_Value", next_titles, OUT_NEXT)

print("\nPanel fit availability summary:")
for k, v in prev_stats.items():
    print("Prev", k, v)
for k, v in next_stats.items():
    print("Next", k, v)
