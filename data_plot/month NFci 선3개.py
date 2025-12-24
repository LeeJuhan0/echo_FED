# -*- coding: utf-8 -*-
"""
Create two 2x2 figures from sentiment.csv (monthly LLM metrics) + nfci_anfci_monthly.xlsx:
 - NFCI group: 3 scatter/regression panels (x in [statment, Theory, Policy]) + combined histogram
 - ANFCI group: 3 scatter/regression panels (same x's) + combined histogram

Assumptions:
- nfci_anfci_monthly.xlsx has columns: date (YYYY-MM), nfci, anfci
- sentiment.csv has columns: date (YYYY-MM), and LLM metrics such as:
  - 'statment' (typo), optionally 'statement'
  - 'Theory'
  - 'Policy'
  (If some are missing, the script tries reasonable fallbacks.)

Install:
pip install pandas numpy matplotlib scipy openpyxl
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
SENTIMENT_PATH = BASE_DIR / "sentiment_13+19.csv"

# Preferred x-columns in order; we will pick the first 3 that exist.
XCOL_CANDIDATES = [
    "statment", "statement",  # common typo vs correct
    "Theory", "theory",
    "Policy", "policy",
    "FSR", "SLOOS", "llm"
]

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
        valid = (~np.isnan(xs)) & (~np.isnan(ys))
        if valid.sum() < 2:
            return None
        m, b, r, p = fit_line(xs[valid], ys[valid])
        xs_plot = np.linspace(np.nanmin(xs[valid]), np.nanmax(xs[valid]), 120)
        ax.plot(xs_plot, m*xs_plot + b, color=color, lw=1.6, label=f"{label} (r={r:.2f})")
        return (m, b, r, p)

    stats_early = plot_seg(early_mask, early_color, "regression 17-22")
    stats_late  = plot_seg(late_mask,  late_color,  "regression 23-25")

    # Overall regression
    xs_all = df[xcol].to_numpy(dtype=float)
    ys_all = df[ycol].to_numpy(dtype=float)
    valid_all = (~np.isnan(xs_all)) & (~np.isnan(ys_all))
    if valid_all.sum() >= 2:
        m_all, b_all, r_all, p_all = fit_line(xs_all[valid_all], ys_all[valid_all])
        xs_plot = np.linspace(np.nanmin(xs_all[valid_all]), np.nanmax(xs_all[valid_all]), 200)
        ax.plot(xs_plot, m_all*xs_plot + b_all, color=overall_color, lw=1.8, linestyle='--',
                label=f"regression 17-25 (r={r_all:.2f})")

    # Formatting
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.85, 0.85)  # keep same x-range as reference
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)

    return {"early": stats_early, "late": stats_late}

def plot_hist(ax, values, bins=20, color='grey', xlabel='Sentiment', title='Histogram'):
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    ax.hist(vals, bins=bins, color=color, alpha=0.8, edgecolor='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(alpha=0.2)

def select_xcols(df, want=3):
    """Pick up to 'want' x columns from candidates; return list preserving order."""
    cols = []
    seen_base = set()
    for cand in XCOL_CANDIDATES:
        if cand in df.columns:
            base = cand.lower()
            if base not in seen_base:
                cols.append(cand)
                seen_base.add(base)
        if len(cols) >= want:
            break
    if len(cols) < want:
        warnings.warn(f"Only found {len(cols)} x-columns: {cols}. The figure will include fewer scatter panels.")
    return cols

def make_2x2_for(merged, ycol, xcols, titles, xlabels, hist_title, hist_xlabel, out_path):
    # Create up to 3 scatter; remaining panels (if any) will be blank but we will fill histogram in last.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    # Year-based masks
    early_mask = merged['year'] <= 2022
    late_mask  = merged['year'] >= 2023

    stats_dict = {}
    # Scatter panels
    for i, xcol in enumerate(xcols[:3]):
        title = titles[i] if i < len(titles) else f"{xcol} vs {ycol.upper()}"
        xlabel = xlabels[i] if i < len(xlabels) else "Sentiment Score"
        ylabel = "NFCI" if ycol == "nfci" else "ANFCI"
        stats = plot_scatter_reg(axes_flat[i], merged, xcol, ycol, title, xlabel, ylabel,
                                 early_mask, late_mask)
        stats_dict[f"panel_{i+1}"] = stats

    # If fewer than 3 xcols, hide unused scatter axes
    for j in range(len(xcols), 3):
        axes_flat[j].axis('off')

    # Combined histogram of the used x-columns
    combined = np.concatenate([merged[c].to_numpy(dtype=float) for c in xcols]) if xcols else np.array([])
    plot_hist(axes_flat[3], combined, bins=20, color='grey', xlabel=hist_xlabel, title=hist_title)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")
    plt.show()
    return stats_dict

# -------------------- Main --------------------
def main():
    # Load NFCI/ANFCI monthly
    if not NFCI_ANFCI_PATH.exists():
        raise FileNotFoundError(f"Not found: {NFCI_ANFCI_PATH}")
    nfci_df = pd.read_excel(NFCI_ANFCI_PATH)
    nfci_df.columns = [c.strip() for c in nfci_df.columns]
    # normalize to expected names
    col_map = {c: c.lower() for c in nfci_df.columns}
    nfci_df = nfci_df.rename(columns=col_map)
    for col in ("date", "nfci", "anfci"):
        if col not in nfci_df.columns:
            raise ValueError(f"{NFCI_ANFCI_PATH.name} must contain columns ['date','nfci','anfci']. Found: {list(nfci_df.columns)}")

    # Load sentiment
    if not SENTIMENT_PATH.exists():
        raise FileNotFoundError(f"Not found: {SENTIMENT_PATH}")
    sent_df = pd.read_csv(SENTIMENT_PATH)
    sent_df.columns = [c.strip() for c in sent_df.columns]
    if "date" not in sent_df.columns:
        raise ValueError("sentiment.csv must contain 'date' column (YYYY-MM).")
    # Drop dup months; keep first
    sent_df = sent_df.drop_duplicates(subset=["date"]).reset_index(drop=True)

    # Select up to 3 x columns
    xcols = select_xcols(sent_df, want=3)
    if not xcols:
        raise ValueError(f"No usable x columns from candidates: {XCOL_CANDIDATES}. Found columns: {list(sent_df.columns)}")
    print(f"Using x-columns: {xcols}")

    # Merge on date (YYYY-MM)
    merged = sent_df[["date"] + xcols].merge(
        nfci_df[["date", "nfci", "anfci"]],
        on="date", how="left"
    )

    # Parse date to datetime (first day of month)
    merged["month_date"] = pd.to_datetime(merged["date"] + "-01", errors="coerce")
    merged = merged.dropna(subset=["month_date"]).reset_index(drop=True)
    merged["year"] = merged["month_date"].dt.year

    # Warn on missing targets
    missing = merged[merged["nfci"].isna() | merged["anfci"].isna()]["date"].tolist()
    if missing:
        warnings.warn(f"Missing NFCI/ANFCI for months: {missing}")

    # NFCI 2x2
    prev_titles = [
        f"LLM[Statement(St)] vs Monthly NFCI",
        f"GraphRAG[St+Theory(Th)] vs Monthly NFCI",
        f"GraphRAG[St+Th+Policy] vs Monthly NFCI",
    ]
    prev_xlabels = ["Sentiment Score"] * 3
    prev_hist_title = "Combined distribution of selected sentiment metrics"
    prev_hist_xlabel = "Sentiment Score(s)"
    nfci_png = BASE_DIR / "sentiment_vs_nfci_2x2.png"
    nfci_stats = make_2x2_for(
        merged=merged, ycol="nfci", xcols=xcols,
        titles=prev_titles, xlabels=prev_xlabels,
        hist_title=prev_hist_title, hist_xlabel=prev_hist_xlabel,
        out_path=nfci_png
    )

    # ANFCI 2x2
    next_titles = [
        f"LLM[Statement(St)] vs Monthly ANFCI",
        f"GraphRAG[St+Theory(Th)] vs Monthly ANFCI",
        f"GraphRAG[St+Th+Policy] vs Monthly ANFCI",
    ]
    next_xlabels = ["Sentiment Score"] * 3
    next_hist_title = "Combined distribution of selected sentiment metrics"
    next_hist_xlabel = "Sentiment Score(s)"
    anfci_png = BASE_DIR / "sentiment_vs_anfci_2x2.png"
    anfci_stats = make_2x2_for(
        merged=merged, ycol="anfci", xcols=xcols,
        titles=next_titles, xlabels=next_xlabels,
        hist_title=next_hist_title, hist_xlabel=next_hist_xlabel,
        out_path=anfci_png
    )

    # Summary
    print("\nPanel fit availability summary (None means not enough points for segment fit):")
    for k, v in nfci_stats.items():
        print("NFCI", k, v)
    for k, v in anfci_stats.items():
        print("ANFCI", k, v)

if __name__ == "__main__":
    main()