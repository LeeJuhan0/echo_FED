# -*- coding: utf-8 -*-
"""
One 3x2 figure:
- Left column:  Monthly NFCI  (Statement -> Theory -> Policy)
- Right column: Monthly ANFCI (Statement -> Theory -> Policy)

Rows:    Statement, St+Theory, St+Th+Policy
Columns: NFCI (left), ANFCI (right)
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
NFCI_ANFCI_PATH = BASE_DIR / "nfci_anfci_input_data_for_analysis.xlsx"
SENTIMENT_PATH  = BASE_DIR / "sentiment_13+19_llm.csv"

OUT_3x2 = BASE_DIR / "sentiment_nfci_anfci_3x2.png"

# -------------------- Helpers --------------------
def fit_line(x, y):
    """Simple linear regression y = m x + b and Pearson correlation."""
    m, b = np.polyfit(x, y, 1)
    if SCIPY_OK:
        r, p = stats.pearsonr(x, y)
    else:
        r = np.corrcoef(x, y)[0, 1]
        p = None
    return m, b, r, p

def plot_scatter_reg(ax, df, xcol, ycol, title, xlabel, ylabel, early_mask, late_mask,
                     early_color='#1f77b4', late_color='#d62728', overall_color='#2ca02c'):
    # Scatter by segments
    ax.scatter(df.loc[early_mask, xcol], df.loc[early_mask, ycol],
               color=early_color, edgecolors='white', linewidth=0.5, alpha=0.85, label='2017-2022')
    ax.scatter(df.loc[late_mask, xcol], df.loc[late_mask, ycol],
               color=late_color, edgecolors='white', linewidth=0.5, alpha=0.85, label='2023-2025')

    def plot_seg(mask, color, label):
        xs = df.loc[mask, xcol].to_numpy(dtype=float)
        ys = df.loc[mask, ycol].to_numpy(dtype=float)

        # IMPORTANT: keep paired valid points
        valid = (~np.isnan(xs)) & (~np.isnan(ys))
        if valid.sum() < 2:
            return None

        m, b, r, p = fit_line(xs[valid], ys[valid])
        xs_plot = np.linspace(np.nanmin(xs[valid]), np.nanmax(xs[valid]), 120)
        ax.plot(xs_plot, m * xs_plot + b, color=color, lw=1.6, label=f"{label} (r={r:.2f})")
        return (m, b, r, p)

    stats_early = plot_seg(early_mask, '#1f77b4', "regression 17-22")
    stats_late  = plot_seg(late_mask,  '#d62728', "regression 23-25")

    # Overall regression
    xs_all = df[xcol].to_numpy(dtype=float)
    ys_all = df[ycol].to_numpy(dtype=float)
    valid_all = (~np.isnan(xs_all)) & (~np.isnan(ys_all))

    if valid_all.sum() >= 2:
        m_all, b_all, r_all, p_all = fit_line(xs_all[valid_all], ys_all[valid_all])
        xs_plot = np.linspace(np.nanmin(xs_all[valid_all]), np.nanmax(xs_all[valid_all]), 200)
        ax.plot(xs_plot, m_all * xs_plot + b_all,
                color=overall_color, lw=1.8, linestyle='--',
                label=f"regression 17-25 (r={r_all:.2f})")

    # Formatting
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.85, 0.85)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    return {"early": stats_early, "late": stats_late}

def resolve_statement_column(df: pd.DataFrame) -> str:
    """Return the actual column name for Statement metric (handles typo)."""
    if "statment" in df.columns:
        return "statment"
    if "statement" in df.columns:
        return "statement"
    # fallback: try case variants
    for c in df.columns:
        if c.lower() in ("statment", "statement"):
            return c
    raise ValueError("Sentiment CSV must contain 'statment' or 'statement' column.")

# -------------------- Main --------------------
def main():
    # Load NFCI/ANFCI monthly
    if not NFCI_ANFCI_PATH.exists():
        raise FileNotFoundError(f"Not found: {NFCI_ANFCI_PATH}")

    nfci_df = pd.read_excel(NFCI_ANFCI_PATH)
    nfci_df.columns = [c.strip() for c in nfci_df.columns]

    # normalize to lowercase for targets
    nfci_df = nfci_df.rename(columns={c: c.lower() for c in nfci_df.columns})
    for col in ("date", "nfci", "anfci"):
        if col not in nfci_df.columns:
            raise ValueError(f"{NFCI_ANFCI_PATH.name} must contain columns ['date','nfci','anfci'].")

    # Load sentiment
    if not SENTIMENT_PATH.exists():
        raise FileNotFoundError(f"Not found: {SENTIMENT_PATH}")

    sent_df = pd.read_csv(SENTIMENT_PATH)
    sent_df.columns = [c.strip() for c in sent_df.columns]
    if "date" not in sent_df.columns:
        raise ValueError("sentiment CSV must contain 'date' column (YYYY-MM).")

    # Decide x columns (fixed 3 panels)
    st_col = resolve_statement_column(sent_df)
    if "Theory" not in sent_df.columns:
        raise ValueError("Sentiment CSV must contain 'Theory' column.")
    if "Policy" not in sent_df.columns:
        raise ValueError("Sentiment CSV must contain 'Policy' column.")

    # Use these in fixed order
    xcols = [st_col, "Theory", "Policy"]

    # Drop dup months; keep first
    sent_df = sent_df.drop_duplicates(subset=["date"]).reset_index(drop=True)

    # Merge on date (YYYY-MM)
    merged = sent_df[["date"] + xcols].merge(
        nfci_df[["date", "nfci", "anfci"]],
        on="date", how="left"
    )

    # Parse date to datetime (first day of month)
    merged["month_date"] = pd.to_datetime(merged["date"] + "-01", errors="coerce")
    merged = merged.dropna(subset=["month_date"]).reset_index(drop=True)
    merged["year"] = merged["month_date"].dt.year

    # Coerce numeric
    for c in xcols + ["nfci", "anfci"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # Warn on missing targets
    missing = merged[merged["nfci"].isna() | merged["anfci"].isna()]["date"].tolist()
    if missing:
        warnings.warn(f"Missing NFCI/ANFCI for months: {missing}")

    # Masks
    early_mask = merged["year"] <= 2022
    late_mask  = merged["year"] >= 2023

    # -------------------- 3x2 figure --------------------
    row_specs = [
        (xcols[0], "LLM[Statement(St)]"),
        (xcols[1], "GraphRAG[St+Theory(Th)]"),
        (xcols[2], "GraphRAG[St+Th+Policy]")
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))  # 3 rows x 2 cols

    for r, (xcol, prefix) in enumerate(row_specs):
        # Left column: NFCI
        plot_scatter_reg(
            ax=axes[r, 0],
            df=merged,
            xcol=xcol,
            ycol="nfci",
            title=f"{prefix} vs Previous Week ANFCI",
            xlabel="Sentiment Score",
            ylabel="NFCI",
            early_mask=early_mask,
            late_mask=late_mask
        )

        # Right column: ANFCI
        plot_scatter_reg(
            ax=axes[r, 1],
            df=merged,
            xcol=xcol,
            ycol="anfci",
            title=f"{prefix} vs Previous Month ANFCI",
            xlabel="Sentiment Score",
            ylabel="ANFCI",
            early_mask=early_mask,
            late_mask=late_mask
        )

    plt.tight_layout()
    fig.savefig(OUT_3x2, dpi=300)
    print(f"Saved {OUT_3x2}")
    plt.show()

if __name__ == "__main__":
    main()
