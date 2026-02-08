# -*- coding: utf-8 -*-
"""
Create two 2x2 figures from sentiment.csv (monthly LLM metrics) + nfci_anfci_monthly.xlsx

Applied changes:
- Remove histogram panel
- Panel layout:
    (1,1) Statement
    (1,2) lg
    (2,1) Theory
    (2,2) Policy
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# -------------------- Paths --------------------
BASE_DIR = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data")
NFCI_ANFCI_PATH = BASE_DIR / "nfci_anfci_monthly.xlsx"
SENTIMENT_PATH = BASE_DIR / "lg_sentiment.csv"

# -------------------- Helpers --------------------
def fit_line(x, y):
    m, b = np.polyfit(x, y, 1)
    if SCIPY_OK:
        r, p = stats.pearsonr(x, y)
    else:
        r = np.corrcoef(x, y)[0, 1]
        p = None
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
        valid = (~np.isnan(xs)) & (~np.isnan(ys))
        if valid.sum() < 2:
            return None
        m, b, r, p = fit_line(xs[valid], ys[valid])
        xp = np.linspace(xs[valid].min(), xs[valid].max(), 120)
        ax.plot(xp, m * xp + b, color=color, lw=1.6,
                label=f"{label} (r={r:.2f})")
        return (m, b, r, p)

    stats_early = plot_seg(early_mask, early_color, "regression 17–22")
    stats_late  = plot_seg(late_mask,  late_color,  "regression 23–25")

    xs_all = df[xcol].to_numpy(dtype=float)
    ys_all = df[ycol].to_numpy(dtype=float)
    valid_all = (~np.isnan(xs_all)) & (~np.isnan(ys_all))
    if valid_all.sum() >= 2:
        m, b, r, p = fit_line(xs_all[valid_all], ys_all[valid_all])
        xp = np.linspace(xs_all[valid_all].min(), xs_all[valid_all].max(), 200)
        ax.plot(xp, m * xp + b, color=overall_color, lw=1.8,
                linestyle='--', label=f"regression 17–25 (r={r:.2f})")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.85, 0.85)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    return {"early": stats_early, "late": stats_late}

# -------------------- Main --------------------
def main():
    # Load NFCI / ANFCI
    nfci_df = pd.read_excel(NFCI_ANFCI_PATH)
    nfci_df.columns = [c.strip().lower() for c in nfci_df.columns]

    for col in ("date", "nfci", "anfci"):
        if col not in nfci_df.columns:
            raise ValueError("nfci_anfci_monthly.xlsx must contain date, nfci, anfci")

    # Load sentiment
    sent_df = pd.read_csv(SENTIMENT_PATH)
    sent_df.columns = [c.strip().lower() for c in sent_df.columns]

    required = ["date", "statement", "theory", "policy", "lg"]
    missing = [c for c in required if c not in sent_df.columns]
    if missing:
        raise ValueError(f"Missing columns in sentiment CSV: {missing}")

    sent_df = sent_df[required].drop_duplicates(subset=["date"]).reset_index(drop=True)

    # Merge
    merged = sent_df.merge(
        nfci_df[["date", "nfci", "anfci"]],
        on="date", how="left"
    )

    merged["month_date"] = pd.to_datetime(merged["date"] + "-01", errors="coerce")
    merged = merged.dropna(subset=["month_date"]).reset_index(drop=True)
    merged["year"] = merged["month_date"].dt.year

    early_mask = merged["year"] <= 2022
    late_mask  = merged["year"] >= 2023

    # ---------------- NFCI figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    xcols = ["statement", "lg", "theory", "policy"]
    titles_nfci = [
        "LLM(gpt)[Statement] vs NFCI",
        "LLM(lg)[Statement] vs NFCI",
        "GraphRAG(gpt)[St+Theory] vs NFCI",
        "GraphRAG(gpt)[St+Th+Policy] vs NFCI"
    ]

    for i, xcol in enumerate(xcols):
        plot_scatter_reg(
            axes[i], merged, xcol, "nfci",
            titles_nfci[i],
            "Sentiment Score",
            "NFCI",
            early_mask,
            late_mask
        )

    plt.tight_layout()
    plt.savefig(BASE_DIR / "sentiment_vs_nfci_2x2.png", dpi=300)
    plt.show()

    # ---------------- ANFCI figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    titles_anfci = [
        "LLM[Statement] vs ANFCI",
        "LLM[lg] vs ANFCI",
        "GraphRAG[St+Theory] vs ANFCI",
        "GraphRAG[St+Th+Policy] vs ANFCI"
    ]

    for i, xcol in enumerate(xcols):
        plot_scatter_reg(
            axes[i], merged, xcol, "anfci",
            titles_anfci[i],
            "Sentiment Score",
            "ANFCI",
            early_mask,
            late_mask
        )

    plt.tight_layout()
    plt.savefig(BASE_DIR / "sentiment_vs_anfci_2x2.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
