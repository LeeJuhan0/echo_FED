# -*- coding: utf-8 -*-

"""
2x3 figure:

Row 1 (LG):   lg_llm | lg_statement | lg_policy
Row 2 (GPT):  gpt_llm | gpt_statement | gpt_policy

Y-axis: prev_month_anfci
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

BASE_DIR = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\results\evaluation")

NFCI_ANFCI_PATH = Path(
    r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\external\fomc_nfci_anfci_prev_next_week_month.xlsx"
)

SENTIMENT_PATH = Path(
    r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\results\pipeline_output\score_lg_ex_th_full_pipeline.csv"
)

OUT_2x3 = BASE_DIR / "sentiment_prev_month_anfci_2x3.png"


# -------------------- Helpers --------------------

def fit_line(x, y):

    m, b = np.polyfit(x, y, 1)

    if SCIPY_OK:
        r, p = stats.pearsonr(x, y)
    else:
        r = np.corrcoef(x, y)[0, 1]
        p = None

    return m, b, r, p


def plot_scatter_reg(
        ax,
        df,
        xcol,
        ycol,
        title,
        xlabel,
        ylabel,
        early_mask,
        late_mask,
        early_color="#1f77b4",
        late_color="#d62728",
        overall_color="#2ca02c"
):

    ax.scatter(
        df.loc[early_mask, xcol],
        df.loc[early_mask, ycol],
        color=early_color,
        edgecolors="white",
        linewidth=0.5,
        alpha=0.85,
        label="2017-2022"
    )

    ax.scatter(
        df.loc[late_mask, xcol],
        df.loc[late_mask, ycol],
        color=late_color,
        edgecolors="white",
        linewidth=0.5,
        alpha=0.85,
        label="2023-2025"
    )

    def plot_seg(mask, color, label):

        xs = df.loc[mask, xcol].to_numpy(float)
        ys = df.loc[mask, ycol].to_numpy(float)

        valid = (~np.isnan(xs)) & (~np.isnan(ys))

        if valid.sum() < 2:
            return

        m, b, r, p = fit_line(xs[valid], ys[valid])

        xs_plot = np.linspace(xs[valid].min(), xs[valid].max(), 120)

        ax.plot(
            xs_plot,
            m * xs_plot + b,
            color=color,
            lw=1.5,
            label=f"{label} (r={r:.2f})"
        )


    plot_seg(early_mask, early_color, "17-22")
    plot_seg(late_mask, late_color, "23-25")

    xs_all = df[xcol].to_numpy(float)
    ys_all = df[ycol].to_numpy(float)

    valid = (~np.isnan(xs_all)) & (~np.isnan(ys_all))

    if valid.sum() >= 2:

        m, b, r, p = fit_line(xs_all[valid], ys_all[valid])

        xs_plot = np.linspace(xs_all[valid].min(), xs_all[valid].max(), 200)

        ax.plot(
            xs_plot,
            m * xs_plot + b,
            color=overall_color,
            lw=1.7,
            linestyle="--",
            label=f"17-25 (r={r:.2f})"
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.85, 0.85)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


# -------------------- Main --------------------

def main():

    # ---------- Load Macro ----------

    nfci_df = pd.read_excel(NFCI_ANFCI_PATH)
    nfci_df.columns = [c.strip().lower() for c in nfci_df.columns]

    if "prev_month_anfci" not in nfci_df.columns:
        raise ValueError("Missing column: prev_month_anfci")

    # 날짜 처리
    if "fomc_date" in nfci_df.columns:
        nfci_df["date"] = pd.to_datetime(
            nfci_df["fomc_date"],
            errors="coerce"
        ).dt.strftime("%Y-%m")
    else:
        raise ValueError("Missing column: fomc_date")


    # ---------- Load Sentiment ----------

    sent_df = pd.read_csv(SENTIMENT_PATH)
    sent_df.columns = [c.strip() for c in sent_df.columns]

    needed = [
        "date",
        "lg_llm",
        "lg_statement",
        "lg_policy",
        "gpt_llm",
        "gpt_statement",
        "gpt_policy"
    ]

    for c in needed:
        if c not in sent_df.columns:
            raise ValueError(f"Missing column: {c}")

    sent_df = sent_df[needed].copy()
    sent_df["date"] = sent_df["date"].astype(str).str[:7]

    # ---------- Merge ----------

    merged = sent_df.merge(
        nfci_df[["date", "prev_month_anfci"]],
        on="date",
        how="left"
    )

    merged["month_date"] = pd.to_datetime(
        merged["date"] + "-01",
        errors="coerce"
    )

    merged = merged.dropna(subset=["month_date"])
    merged["year"] = merged["month_date"].dt.year

    for c in needed[1:] + ["prev_month_anfci"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    early_mask = merged["year"] <= 2022
    late_mask  = merged["year"] >= 2023

    # ---------- Plot ----------

    grid = [

        # Row 1: LG
        [
            ("lg_llm", "LG_LLM"),
            ("lg_statement", "LG_GRAG[St]"),
            ("lg_policy", "LG_GRAG[Policy]")
        ],

        # Row 2: GPT
        [
            ("gpt_llm", "GPT_LLM"),
            ("gpt_statement", "GPT_GRAG[St]"),
            ("gpt_policy", "GPT_GRAG[St]")
        ]
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for r in range(2):
        for c in range(3):

            xcol, title = grid[r][c]

            plot_scatter_reg(
                ax=axes[r, c],
                df=merged,
                xcol=xcol,
                ycol="prev_month_anfci",
                title=f"{title} vs Prev Month ANFCI",
                xlabel="Sentiment Score",
                ylabel="Prev Month ANFCI",
                early_mask=early_mask,
                late_mask=late_mask
            )

    plt.tight_layout()
    fig.savefig(OUT_2x3, dpi=300)
    print(f"Saved: {OUT_2x3}")
    plt.show()


# -------------------- Run --------------------

if __name__ == "__main__":
    main()
