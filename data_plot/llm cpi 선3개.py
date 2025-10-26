#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a single figure with 3 subplots:
 - Left:  LLM vs Prev_CPI_Value (scatter + regressions)
 - Middle:LLM vs Next_CPI_Value (scatter + regressions)
 - Right: Histogram of LLM (shared)

This version keeps x-axis limits (-0.85, 0.85) for scatter panels and sets the
y-axis labels explicitly to "Previous Week NFCI" and "Previous Week ANFCI".
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import io
from textwrap import dedent
import warnings

# ---------- Inline CPI data (unchanged) ----------
cpi_text = """
Prev_CPI_Value\tNext_CPI_Value\tFOMC_date
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
cpi_df = pd.read_csv(io.StringIO(dedent(cpi_text)), sep=r"\s+")
cpi_df['FOMC_date'] = pd.to_datetime(cpi_df['FOMC_date'])

# ---------- Original policy dates (kept in same order as original script) ----------
dates = [
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

# ---------- User-provided llm values (as given) ----------
llm_values = [
    0.3,0.42,0.35,0.43,0.36,0.4,0.4,0.55,0.64,0.45,0.45,0.74,0.75,0.84,0.7,0.7,0.45,
    0.2,0.3,0.2,0.25,0.24,0.2,0.3,0.3,-0.18,-0.4,-0.82,-0.62,-0.4,-0.3,-0.3,-0.3,
    -0.3,0.1,0.22,0.44,0.45,0.35,0.35,0.35,0.3,0.1,-0.12,-0.12,-0.32,-0.23,-0.25,
    -0.15,-0.15,-0.1,0.1,0.15,0.12,0.2,0.2,-0.1,0.25,0.35,0.25,0.35,0.25,0.32,0.35,
    0.3,0.34,0.3,0.15,0.3,-0.1
]

# Pair dates and llm values; truncate if mismatch
n_dates = len(dates)
n_llm = len(llm_values)
if n_dates != n_llm:
    warnings.warn(f"Number of dates ({n_dates}) != number of llm values ({n_llm}). Truncating to min length.")
n = min(n_dates, n_llm)
policy_df = pd.DataFrame({'date': dates[:n], 'llm': llm_values[:n]})

# ---------- Build merged monthly mapping (month-level aggregation for CPI) ----------
cpi_df['month'] = cpi_df['FOMC_date'].dt.strftime('%Y-%m')
cpi_month = cpi_df.groupby('month', as_index=False).agg({
    'Prev_CPI_Value': 'mean',
    'Next_CPI_Value': 'mean',
    'FOMC_date': 'min'
}).rename(columns={'month': 'date'})

merged = policy_df.merge(cpi_month[['date', 'Prev_CPI_Value', 'Next_CPI_Value', 'FOMC_date']],
                         on='date', how='left')

if merged['Prev_CPI_Value'].isna().any() or merged['Next_CPI_Value'].isna().any():
    print("Warning: some rows did not find CPI match. Unmatched rows:")
    print(merged[merged['Prev_CPI_Value'].isna() | merged['Next_CPI_Value'].isna()][['date']])

merged['FOMC_date'] = pd.to_datetime(merged['FOMC_date'])
merged = merged.sort_values('FOMC_date').reset_index(drop=True)

# ---------- helpers ----------
def fit_line(x, y):
    m, b = np.polyfit(x, y, 1)
    r, p = stats.pearsonr(x, y)
    return m, b, r, p

def plot_scatter_with_regs(ax, df, xcol, ycol, title, ylabel, early_mask, late_mask,
                           early_color='#1f77b4', late_color='#d62728', overall_color='#2ca02c'):
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

    stats_early = plot_seg(early_mask, early_color, "Regression 17-22")
    stats_late = plot_seg(late_mask, late_color, "Regression 23-25")

    xs_all = df[xcol].to_numpy(dtype=float)
    ys_all = df[ycol].to_numpy(dtype=float)
    valid_all = (~np.isnan(xs_all)) & (~np.isnan(ys_all))
    if valid_all.sum() >= 2:
        m_all, b_all, r_all, p_all = fit_line(xs_all[valid_all], ys_all[valid_all])
        xs_plot = np.linspace(np.nanmin(xs_all[valid_all]), np.nanmax(xs_all[valid_all]), 200)
        ax.plot(xs_plot, m_all*xs_plot + b_all, color=overall_color, lw=1.8, linestyle='--',
                label=f"Regression 17-25 (r={r_all:.2f})")

    # enforce requested x-axis limits for scatter panels
    ax.set_xlim(-0.85, 0.85)

    ax.set_title(title)
    ax.set_xlabel('Sentiment Score (LLM)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    return {"early": stats_early, "late": stats_late}

def plot_hist(ax, values, bins=20, color='grey', xlabel='LLM', title='Histogram'):
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    ax.hist(vals, bins=bins, color=color, alpha=0.8, edgecolor='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(alpha=0.2)

# ---------- masks ----------
merged['year'] = merged['FOMC_date'].dt.year
early_mask = merged['year'] <= 2022
late_mask  = merged['year'] >= 2023

# ---------- Create combined 1x3 figure ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
ax_prev, ax_next, ax_hist = axes

prev_stats = plot_scatter_with_regs(
    ax_prev, merged, 'llm', 'Prev_CPI_Value',
    title='LLM (gpt-5) vs Previous Month CPI',
    ylabel='Previous Month CPI',
    early_mask=early_mask, late_mask=late_mask
)

next_stats = plot_scatter_with_regs(
    ax_next, merged, 'llm', 'Next_CPI_Value',
    title='LLM (gpt-5) vs Next Month CPI',
    ylabel='Next Month CPI',
    early_mask=early_mask, late_mask=late_mask
)

plot_hist(ax_hist, merged['llm'], bins=20, color='grey', xlabel='Sentiment Score (LLM)',
          title='LLM (gpt-5) Distribution (all observations)')

out_fname = 'llm_combined_1x3_ylabel_fixed.png'
fig.savefig(out_fname, dpi=300)
print(f"Saved {out_fname}")
plt.show()

# ---------- summary ----------
print("\nRegression fit availability summary (None means not enough points for that segment):")
print("Prev panel stats:", prev_stats)
print("Next panel stats:", next_stats)