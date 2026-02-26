#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


# ---------------- Configuration ----------------
SENTIMENT_CSV = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\results\pipeline_output\score_lg_ex_th_full_pipeline.csv"
CPI_CSV       = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\external\cpi.csv"

OUT_2x3 = "cpi_sentiment_2x3.png"


# ---------------- Helpers ----------------
def read_csv_kr(path: str | Path) -> pd.DataFrame:

    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    last_err = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"CSV load failed: {last_err}")


def fit_line(x: np.ndarray, y: np.ndarray):

    m, b = np.polyfit(x, y, 1)
    r, p = stats.pearsonr(x, y)

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
        early_color='#1f77b4',
        late_color='#d62728',
        overall_color='#2ca02c'
):

    # ----- Scatter -----
    ax.scatter(
        df.loc[early_mask, xcol],
        df.loc[early_mask, ycol],
        color=early_color,
        edgecolors='white',
        linewidth=0.5,
        alpha=0.85,
        label='2017-2022'
    )

    ax.scatter(
        df.loc[late_mask, xcol],
        df.loc[late_mask, ycol],
        color=late_color,
        edgecolors='white',
        linewidth=0.5,
        alpha=0.85,
        label='2023-2025'
    )


    # ----- Segment Regression -----
    def plot_seg(mask, color, label):

        xs = df.loc[mask, xcol].to_numpy(float)
        ys = df.loc[mask, ycol].to_numpy(float)

        valid = (~np.isnan(xs)) & (~np.isnan(ys))
        xs = xs[valid]
        ys = ys[valid]

        if len(xs) < 2:
            return

        m, b, r, p = fit_line(xs, ys)

        xs_plot = np.linspace(xs.min(), xs.max(), 120)

        ax.plot(
            xs_plot,
            m * xs_plot + b,
            color=color,
            lw=1.5,
            label=f"{label} (r={r:.2f})"
        )


    plot_seg(early_mask, early_color, "17-22")
    plot_seg(late_mask, late_color, "23-25")


    # ----- Overall Regression -----
    xs_all = df[xcol].to_numpy(float)
    ys_all = df[ycol].to_numpy(float)

    valid = (~np.isnan(xs_all)) & (~np.isnan(ys_all))
    xs_all = xs_all[valid]
    ys_all = ys_all[valid]

    if len(xs_all) >= 2:

        m, b, r, p = fit_line(xs_all, ys_all)

        xs_plot = np.linspace(xs_all.min(), xs_all.max(), 200)

        ax.plot(
            xs_plot,
            m * xs_plot + b,
            color=overall_color,
            lw=1.7,
            linestyle='--',
            label=f"17-25 (r={r:.2f})"
        )


    # ----- Style -----
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xlim(-0.85, 0.85)

    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)



# ---------------- Load Sentiment ----------------
sent_df = read_csv_kr(SENTIMENT_CSV)

sent_df.columns = [c.strip() for c in sent_df.columns]

needed_cols = [
    "date",
    "lg_llm",
    "lg_statement",
    "lg_policy",
    "gpt_llm",
    "gpt_statement",
    "gpt_policy"
]

missing = [c for c in needed_cols if c not in sent_df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")


sent_df = sent_df[needed_cols].copy()


# numeric coercion
for c in needed_cols[1:]:
    sent_df[c] = pd.to_numeric(sent_df[c], errors="coerce")


# date normalize
sent_df["date"] = sent_df["date"].astype(str).str[:7]



# ---------------- Load CPI ----------------
cpi_df = read_csv_kr(CPI_CSV)

cpi_df.columns = [c.strip() for c in cpi_df.columns]

cpi_df = cpi_df.rename(columns={
    "Prev_CPI_Value": "Prev_CPI_Value",
    "Next_CPI_Value": "Next_CPI_Value",
    "FOMC_date": "FOMC_date"
})


cpi_df["FOMC_date"] = pd.to_datetime(cpi_df["FOMC_date"], errors="coerce")


# Monthly aggregation
cpi_df["month"] = cpi_df["FOMC_date"].dt.strftime("%Y-%m")

cpi_month = (
    cpi_df
    .groupby("month", as_index=False)
    .agg({
        "Prev_CPI_Value": "mean",
        "Next_CPI_Value": "mean",
        "FOMC_date": "min"
    })
    .rename(columns={"month": "date"})
)



# ---------------- Merge ----------------
merged = sent_df.merge(
    cpi_month,
    on="date",
    how="left"
)


merged["FOMC_date"] = pd.to_datetime(merged["FOMC_date"])
merged = merged.sort_values("FOMC_date").reset_index(drop=True)


# ---------------- Masks ----------------
merged["year"] = merged["FOMC_date"].dt.year

early_mask = merged["year"] <= 2022
late_mask  = merged["year"] >= 2023



# ---------------- 2x3 Plot ----------------
def make_2x3_plot(out_fname):

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

                ycol="Prev_CPI_Value",

                title=f"{title} vs Prev CPI",

                xlabel="Sentiment Score",

                ylabel="Prev CPI",

                early_mask=early_mask,

                late_mask=late_mask
            )


    plt.tight_layout()

    fig.savefig(out_fname, dpi=300)

    print(f"Saved: {out_fname}")

    plt.show()



# ---------------- Run ----------------
make_2x3_plot(OUT_2x3)
