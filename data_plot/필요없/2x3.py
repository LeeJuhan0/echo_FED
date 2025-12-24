#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2x3 Figures (Previous / Next Month CPI sets)

REQUEST:
 - ONLY the (Row1 Col1) LLM panel must be replaced not just in style but in the
   underlying DATA + correlation logic so that its scatter / regressions /
   r values match EXACTLY what the separate 1x3 LLM script would produce.
 - All the other five subplots (Statement / P&B / Policy panels and the two
   histograms) stay as they were in the existing 2x3 script that used the policy
   text (Statement, P&B, Policy) dataset.
 - The LLM histogram (Row3 Col1) should also use the 1x3-panel data values.
 - The combined sentiment histogram (Row3 Col2) remains based on Statement/P&B/Policy.

IMPLEMENTATION STRATEGY:
 1. Build two independent merged datasets:
    (A) policy_merged: from policy_text + CPI (for Statement / P&B / Policy & combined sentiment)
    (B) llm_merged: from (dates_llm + llm_values) + CPI (exact logic from the 1x3 script).
       This dataset is used ONLY for Row1 Col1 scatter and Row3 Col1 LLM distribution.
 2. The early/late segmentation is computed separately for each dataset:
      early = year <= 2022, late = year >= 2023
 3. Regression & r for LLM panel use only llm_merged (so numbers match 1x3 script).
 4. Regression & r for Statement / P&B / Policy panels use policy_merged (unchanged).
 5. X-limits fixed at (-0.85, 0.85) everywhere for consistency.
 6. Row3 Col2 histogram shows combined sentiment (Statement, P&B, Policy) from policy_merged
    but keeps title "Previous/Next Month CPI Distribution ..." per prior request even though
    the data are sentiment scores (clarified in code comments).

OUTPUT FILES:
  previous_month_cpi_2x3_llm_panel_exact.png
  next_month_cpi_2x3_llm_panel_exact.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import io
from textwrap import dedent
import warnings

# ---------- (A) POLICY DATA (Statement / P&B / Policy) ----------
policy_text = """date\tstatment\tP&B\tPolicy
2017-02\t0.33\t0.32\t0.29
2017-03\t0.4\t0.4\t0.35
2017-05\t0.34\t0.36\t0.33
2017-06\t0.4\t0.35\t0.3
2017-07\t0.35\t0.32\t0.24
2017-09\t0.38\t0.36\t0.28
2017-11\t0.4\t0.43\t0.38
2017-12\t0.58\t0.55\t0.5
2018-01\t0.66\t0.65\t0.55
2018-03\t0.4\t0.52\t0.52
2018-05\t0.44\t0.4\t0.3
2018-06\t0.75\t0.7\t0.66
2018-08\t0.65\t0.67\t0.63
2018-09\t0.79\t0.72\t0.76
2018-11\t0.66\t0.65\t0.5
2018-12\t0.64\t0.6\t0.43
2019-01\t0.4\t0.48\t0.36
2019-06\t0.2\t0.12\t0.04
2019-07\t0.21\t0.2\t0.08
2019-09\t0.23\t0.2\t0.08
2019-10\t0.2\t0.22\t0.08
2019-12\t0.3\t0.25\t0.12
2020-01\t0.28\t0.25\t0.12
2020-03\t-0.2\t-0.1\t-0.24
2020-03\t-0.62\t-0.4\t-0.65
2020-04\t-0.75\t-0.45\t-0.65
2020-06\t-0.66\t-0.5\t-0.55
2020-07\t-0.4\t-0.22\t-0.35
2020-09\t-0.25\t-0.1\t-0.3
2020-11\t-0.25\t-0.1\t-0.2
2020-12\t-0.22\t-0.1\t-0.12
2021-01\t-0.3\t-0.15\t-0.09
2021-03\t0.1\t0.2\t0.2
2021-04\t0.2\t0.22\t0.3
2021-06\t0.4\t0.435\t0.45
2021-07\t0.4\t0.45\t0.55
2021-09\t0.3\t0.25\t0.25
2021-11\t0.3\t0.25\t0.3
2021-12\t0.32\t0.27\t0.27
2022-01\t0.26\t0.2\t0.18
2022-03\t0.1\t0.04\t-0.01
2022-05\t-0.15\t-0.18\t-0.12
2022-06\t-0.15\t-0.2\t-0.19
2022-07\t-0.3\t-0.4\t-0.48
2022-09\t-0.25\t-0.35\t-0.5
2022-11\t-0.2\t-0.25\t-0.4
2022-12\t-0.2\t-0.25\t-0.4
2023-02\t-0.1\t-0.14\t-0.15
2023-03\t-0.14\t-0.18\t-0.3
2023-05\t0.1\t0.04\t-0.2
2023-06\t0.1\t0.05\t-0.12
2023-07\t0.1\t0.05\t-0.04
2023-09\t0.18\t0.1\t-0.05
2023-11\t0.15\t0.1\t-0.05
2023-12\t-0.1\t-0.08\t-0.2
2024-01\t0.3\t0.2\t0.05
2024-03\t0.32\t0.29\t0.1
2024-05\t0.2\t0.1\t-0.02
2024-06\t0.3\t0.27\t0.14
2024-07\t0.25\t0.2\t0.07
2024-09\t0.3\t0.4\t0.24
2024-11\t0.3\t0.25\t0.15
2024-12\t0.3\t0.32\t0.22
2025-01\t0.3\t0.25\t0.1
2025-03\t0.22\t0.25\t0.16
2025-05\t0.15\t0.1\t-0.05
2025-06\t0.3\t0.22\t0.1
2025-07\t-0.15\t-0.08\t-0.12
"""

# ---------- CPI DATA ----------
cpi_text = """Prev_CPI_Value\tNext_CPI_Value\tFOMC_date
241.432\t243.603\t2017-02-01
243.603\t243.801\t2017-03-15
243.801\t244.733\t2017-05-03
244.733\t244.955\t2017-06-14
244.955\t244.786\t2017-07-26
245.519\t246.819\t2017-09-20
246.819\t246.669\t2017-11-01
246.669\t246.524\t2017-12-13
246.524\t247.867\t2018-01-31
248.991\t249.554\t2018-03-21
249.554\t251.588\t2018-05-02
251.588\t251.989\t2018-06-13
251.989\t252.146\t2018-08-01
252.146\t252.439\t2018-09-26
252.439\t252.038\t2018-11-08
252.038\t251.233\t2018-12-19
251.233\t251.712\t2019-01-30
252.776\t254.202\t2019-03-20
254.202\t256.092\t2019-05-01
256.092\t256.143\t2019-06-19
256.143\t256.571\t2019-07-31
256.558\t256.759\t2019-09-18
256.759\t257.346\t2019-10-30
257.208\t256.974\t2019-12-11
256.974\t257.971\t2020-01-29
257.971\t258.115\t2020-03-03
258.678\t258.115\t2020-03-15
258.115\t256.389\t2020-04-29
256.394\t257.797\t2020-06-10
257.797\t259.101\t2020-07-29
259.918\t260.28\t2020-09-16
260.28\t260.229\t2020-11-05
260.229\t260.474\t2020-12-16
260.474\t261.582\t2021-01-27
263.014\t264.877\t2021-03-17
264.877\t267.054\t2021-04-28
269.195\t271.696\t2021-06-16
271.696\t273.003\t2021-07-28
273.567\t274.31\t2021-09-22
274.31\t277.948\t2021-11-03
277.948\t278.802\t2021-12-15
278.802\t281.148\t2022-01-26
283.716\t287.504\t2022-03-16
287.504\t292.296\t2022-05-04
292.296\t296.311\t2022-06-15
296.311\t296.276\t2022-07-27
296.171\t296.808\t2022-09-21
296.808\t297.711\t2022-11-02
297.711\t296.797\t2022-12-14
296.797\t300.84\t2023-02-01
300.84\t301.836\t2023-03-22
301.836\t304.127\t2023-05-03
304.127\t305.109\t2023-06-14
305.109\t305.691\t2023-07-26
307.026\t307.789\t2023-09-20
307.789\t307.051\t2023-11-01
307.051\t306.746\t2023-12-13
306.746\t308.417\t2024-01-31
310.326\t312.332\t2024-03-20
312.332\t314.069\t2024-05-01
314.069\t314.175\t2024-06-12
314.175\t314.54\t2024-07-31
314.796\t315.301\t2024-09-18
315.301\t315.493\t2024-11-07
315.493\t315.605\t2024-12-18
315.605\t317.671\t2025-01-29
319.082\t319.799\t2025-03-19
319.799\t320.795\t2025-05-30
321.465\t322.561\t2025-06-18
322.561\t323.048\t2025-07-30
"""

# ---------- (B) LLM DATA (EXACT from 1x3 approach) ----------
dates_llm = [
    "2017-02","2017-03","2017-05","2017-06","2017-07","2017-09","2017-11","2017-12",
    "2018-01","2018-03","2018-05","2018-06","2018-08","2018-09","2018-11","2018-12",
    "2019-01","2019-06","2019-07","2019-09","2019-10","2019-12","2020-01","2020-03",
    "2020-03","2020-04","2020-06","2020-07","2020-09","2020-11","2020-12","2021-01",
    "2021-03","2021-04","2021-06","2021-07","2021-09","2021-11","2021-12","2022-01",
    "2022-03","2022-05","2022-06","2022-07","2022-09","2022-11","2022-12","2023-02",
    "2023-03","2023-05","2023-06","2023-07","2023-09","2023-11","2023-12","2024-01",
    "2024-03","2024-05","2024-06","2024-07","2024-09","2024-11","2024-12","2025-01",
    "2025-03","2025-05","2025-06","2025-07"
]
llm_values = [
    0.3,0.42,0.35,0.43,0.36,0.4,0.4,0.55,0.64,0.45,0.45,0.74,0.75,0.84,0.7,0.7,0.45,
    0.2,0.3,0.2,0.25,0.24,0.2,0.3,0.3,-0.18,-0.4,-0.82,-0.62,-0.4,-0.3,-0.3,-0.3,
    -0.3,0.1,0.22,0.44,0.45,0.35,0.35,0.35,0.3,0.1,-0.12,-0.12,-0.32,-0.23,-0.25,
    -0.15,-0.15,-0.1,0.1,0.15,0.12,0.2,0.2,-0.1,0.25,0.35,0.25,0.35,0.25,0.32,0.35,
    0.3,0.34,0.3,0.15,0.3,-0.1
]

# ---------- Load DataFrames ----------
policy_df_raw = pd.read_csv(io.StringIO(dedent(policy_text)), sep=r"\s+")
cpi_df = pd.read_csv(io.StringIO(dedent(cpi_text)), sep=r"\s+")
cpi_df['FOMC_date'] = pd.to_datetime(cpi_df['FOMC_date'])

policy_df = policy_df_raw.rename(columns={'statment': 'Statement'})
policy_df['date'] = policy_df['date'].str.strip()

# Monthly CPI mapping
cpi_df['month'] = cpi_df['FOMC_date'].dt.strftime('%Y-%m')
cpi_month = cpi_df.groupby('month', as_index=False).agg({
    'Prev_CPI_Value': 'mean',
    'Next_CPI_Value': 'mean',
    'FOMC_date': 'min'
}).rename(columns={'month':'date'})

# (A) policy_merged
policy_merged = policy_df.merge(
    cpi_month[['date','Prev_CPI_Value','Next_CPI_Value','FOMC_date']],
    on='date', how='left'
)
policy_merged['FOMC_date'] = pd.to_datetime(policy_merged['FOMC_date'])
policy_merged = policy_merged.sort_values('FOMC_date').reset_index(drop=True)
policy_merged['year'] = policy_merged['FOMC_date'].dt.year
policy_early_mask = policy_merged['year'] <= 2022
policy_late_mask  = policy_merged['year'] >= 2023

# (B) llm_merged (ONLY for LLM panel replicated exactly from 1x3 logic)
if len(dates_llm) != len(llm_values):
    warnings.warn("LLM dates and values length mismatch; truncating.")
n_llm = min(len(dates_llm), len(llm_values))
llm_df = pd.DataFrame({'date': dates_llm[:n_llm], 'LLM': llm_values[:n_llm]})

llm_merged = llm_df.merge(
    cpi_month[['date','Prev_CPI_Value','Next_CPI_Value','FOMC_date']],
    on='date', how='left'
)
llm_merged['FOMC_date'] = pd.to_datetime(llm_merged['FOMC_date'])
llm_merged = llm_merged.sort_values('FOMC_date').reset_index(drop=True)
llm_merged['year'] = llm_merged['FOMC_date'].dt.year
llm_early_mask = llm_merged['year'] <= 2022
llm_late_mask  = llm_merged['year'] >= 2023

# ---------- Helper functions ----------
def fit_line(x, y):
    m, b = np.polyfit(x, y, 1)
    r, p = stats.pearsonr(x, y)
    return m, b, r, p

def llm_panel_from_1x3(ax, df, ycol, early_mask, late_mask,
                       early_color='#1f77b4', late_color='#d62728', overall_color='#2ca02c'):
    """
    EXACT style & data logic as in the standalone 1x3 script for LLM panel.
    """
    xcol = 'LLM'
    title = f"LLM (gpt-5) vs {'Previous Month' if ycol.startswith('Prev') else 'Next Month'} CPI"

    ax.scatter(df.loc[early_mask, xcol], df.loc[early_mask, ycol],
               color=early_color, edgecolors='white', linewidth=0.5,
               alpha=0.85, label='2017-2022')
    ax.scatter(df.loc[late_mask, xcol], df.loc[late_mask, ycol],
               color=late_color, edgecolors='white', linewidth=0.5,
               alpha=0.85, label='2023-2025')

    def segment(mask, color, lbl):
        xs = df.loc[mask, xcol].to_numpy(dtype=float)
        ys = df.loc[mask, ycol].to_numpy(dtype=float)
        valid = (~np.isnan(xs)) & (~np.isnan(ys))
        if valid.sum() < 2:
            return None
        m,b,r,p = fit_line(xs[valid], ys[valid])
        xs_line = np.linspace(xs[valid].min(), xs[valid].max(), 120)
        ax.plot(xs_line, m*xs_line + b, color=color, lw=1.6, label=f"{lbl} (r={r:.2f})")
        return (m,b,r,p)

    stats_early = segment(early_mask, early_color, "Regression 17-22")
    stats_late  = segment(late_mask,  late_color,  "Regression 23-25")

    xs_all = df[xcol].to_numpy(dtype=float)
    ys_all = df[ycol].to_numpy(dtype=float)
    valid_all = (~np.isnan(xs_all)) & (~np.isnan(ys_all))
    if valid_all.sum() >= 2:
        m_all,b_all,r_all,p_all = fit_line(xs_all[valid_all], ys_all[valid_all])
        xs_line = np.linspace(xs_all[valid_all].min(), xs_all[valid_all].max(), 200)
        ax.plot(xs_line, m_all*xs_line + b_all, color=overall_color, lw=1.8, ls='--',
                label=f"Regression 17-25 (r={r_all:.2f})")

    ax.set_title(title)
    ax.set_xlabel("Sentiment Score (LLM)")
    ax.set_ylabel("Previous Month CPI" if ycol.startswith('Prev') else "Next Month CPI")
    ax.set_xlim(-0.85, 0.85)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return {"early": stats_early, "late": stats_late}

def standard_policy_panel(ax, df, xcol, ycol, title,
                          early_mask, late_mask,
                          early_color='#1f77b4', late_color='#d62728', overall_color='#2ca02c'):
    """
    The existing 2x3 logic for Statement / P&B / Policy panels (unchanged).
    """
    ax.scatter(df.loc[early_mask, xcol], df.loc[early_mask, ycol],
               color=early_color, edgecolors='white', linewidth=0.5,
               alpha=0.85, label='2017-2022')
    ax.scatter(df.loc[late_mask, xcol], df.loc[late_mask, ycol],
               color=late_color, edgecolors='white', linewidth=0.5,
               alpha=0.85, label='2023-2025')

    def seg(mask, color, lbl):
        xs = df.loc[mask, xcol].to_numpy(dtype=float)
        ys = df.loc[mask, ycol].to_numpy(dtype=float)
        ok = (~np.isnan(xs)) & (~np.isnan(ys))
        if ok.sum() < 2:
            return None
        m,b,r,p = fit_line(xs[ok], ys[ok])
        xs_line = np.linspace(xs[ok].min(), xs[ok].max(), 140)
        ax.plot(xs_line, m*xs_line + b, color=color, lw=1.6,
                label=f"{lbl} (r={r:.2f})")
        return (m,b,r,p)

    stats_early = seg(early_mask, early_color, "Reg 17-22")
    stats_late  = seg(late_mask,  late_color,  "Reg 23-25")

    xs_all = df[xcol].to_numpy(dtype=float)
    ys_all = df[ycol].to_numpy(dtype=float)
    ok_all = (~np.isnan(xs_all)) & (~np.isnan(ys_all))
    if ok_all.sum() >= 2:
        m_all,b_all,r_all,p_all = fit_line(xs_all[ok_all], ys_all[ok_all])
        xs_line = np.linspace(xs_all[ok_all].min(), xs_all[ok_all].max(), 200)
        ax.plot(xs_line, m_all*xs_line + b_all, color=overall_color, lw=1.8, ls='--',
                label=f"Reg 17-25 (r={r_all:.2f})")

    ax.set_title(title)
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Previous Month CPI" if ycol.startswith('Prev') else "Next Month CPI")
    ax.set_xlim(-0.85, 0.85)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return {"early": stats_early, "late": stats_late}

def hist_panel(ax, values, title, xlabel):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    ax.hist(arr, bins=20, color='grey', edgecolor='black', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25)

def combined_sentiment(policy_df):
    return np.concatenate([
        policy_df['Statement'].to_numpy(dtype=float),
        policy_df['P&B'].to_numpy(dtype=float),
        policy_df['Policy'].to_numpy(dtype=float)
    ])

def make_2x3_set(ycol, group_label, outfile):
    """
    Build a figure using:
      - LLM panel from llm_merged (DATA + style from 1x3)
      - Statement/P&B/Policy panels from policy_merged
      - Hist row3 col1: LLM (from llm_merged)
      - Hist row3 col2: combined sentiment (policy_merged)
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    ax_llm, ax_statement = axes[0,0], axes[0,1]
    ax_pb, ax_policy     = axes[1,0], axes[1,1]
    ax_hist_llm, ax_hist_combo = axes[2,0], axes[2,1]

    stats = {}
    # 1) LLM panel (1x3 data + style)
    stats['LLM'] = llm_panel_from_1x3(
        ax_llm, llm_merged, ycol,
        llm_early_mask, llm_late_mask
    )

    # 2) Statement / P&B / Policy (policy data)
    stats['Statement'] = standard_policy_panel(
        ax_statement, policy_merged, 'Statement', ycol,
        f"Statement vs {group_label} Month CPI",
        policy_early_mask, policy_late_mask
    )
    stats['P&B'] = standard_policy_panel(
        ax_pb, policy_merged, 'P&B', ycol,
        f"P&B vs {group_label} Month CPI",
        policy_early_mask, policy_late_mask
    )
    stats['Policy'] = standard_policy_panel(
        ax_policy, policy_merged, 'Policy', ycol,
        f"Policy vs {group_label} Month CPI",
        policy_early_mask, policy_late_mask
    )

    # 3) Histograms
    # LLM distribution (exact llm_merged data)
    hist_panel(
        ax_hist_llm,
        llm_merged['LLM'],
        "LLM (gpt-5) Distribution",
        "LLM Sentiment Score"
    )

    # Combined sentiment (Statement/P&B/Policy) from policy_merged
    combo = combined_sentiment(policy_merged)
    hist_panel(
        ax_hist_combo,
        combo,
        f"{group_label} Month CPI Distribution (all observations)",
        f"{group_label} Month Sentiment Score"
    )
    # NOTE: Title says CPI Distribution per earlier request, but data is combined sentiment.

    plt.tight_layout()
    fig.savefig(outfile, dpi=300)
    print(f"Saved {outfile}")
    return stats

# ---------- Generate both sets ----------
prev_stats = make_2x3_set('Prev_CPI_Value', 'Previous', 'previous_month_cpi_2x3_llm_panel_exact.png')
next_stats = make_2x3_set('Next_CPI_Value', 'Next', 'next_month_cpi_2x3_llm_panel_exact.png')

# ---------- Summary printout ----------
print("\n--- LLM Panel (1x3 data) Regression Segment Stats ---")
print("Previous Month LLM panel:", prev_stats['LLM'])
print("Next Month LLM panel:", next_stats['LLM'])

print("\n--- Policy Panels (Statement / P&B / Policy) Regression Segment Stats ---")
print("Previous Month:")
for k in ['Statement','P&B','Policy']:
    print(" ", k, prev_stats[k])
print("Next Month:")
for k in ['Statement','P&B','Policy']:
    print(" ", k, next_stats[k])

print("\nNOTE: If r still differs from your original 1x3 script, verify the CPI mapping "
      "and date lists are EXACTLY the same as in that script (including duplicate months).")