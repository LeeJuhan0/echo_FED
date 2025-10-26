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
250.083\t251.143\t2017-02-01
251.143\t251.290\t2017-03-01
251.290\t251.642\t2017-04-01
251.642\t251.835\t2017-05-01
251.835\t252.014\t2017-06-01
252.014\t251.936\t2017-07-01
251.936\t252.460\t2017-08-01
252.460\t252.941\t2017-09-01
252.941\t253.638\t2017-10-01
253.638\t253.492\t2017-11-01
253.492\t253.558\t2017-12-01
253.558\t254.638\t2018-01-01
254.638\t255.783\t2018-02-01
255.783\t256.610\t2018-03-01
256.610\t257.025\t2018-04-01
257.025\t257.469\t2018-05-01
257.469\t257.697\t2018-06-01
257.697\t257.867\t2018-07-01
257.867\t258.012\t2018-08-01
258.012\t258.429\t2018-09-01
258.429\t259.063\t2018-10-01
259.063\t259.105\t2018-11-01
259.105\t259.083\t2018-12-01
259.083\t260.122\t2019-01-01
260.122\t261.114\t2019-02-01
261.114\t261.836\t2019-03-01
261.836\t262.332\t2019-04-01
262.332\t262.590\t2019-05-01
262.590\t263.177\t2019-06-01
263.177\t263.566\t2019-07-01
263.566\t264.169\t2019-08-01
264.169\t264.522\t2019-09-01
264.522\t265.059\t2019-10-01
265.059\t265.108\t2019-11-01
265.108\t264.935\t2019-12-01
264.935\t266.004\t2020-01-01
266.004\t267.268\t2020-02-01
267.268\t267.312\t2020-03-01
267.268\t267.312\t2020-03-01
267.312\t266.089\t2020-04-01
266.089\t265.799\t2020-05-01
265.799\t266.302\t2020-06-01
266.302\t267.703\t2020-07-01
267.703\t268.756\t2020-08-01
268.756\t269.054\t2020-09-01
269.054\t269.328\t2020-10-01
269.328\t269.473\t2020-11-01
269.473\t269.226\t2020-12-01
269.226\t269.755\t2021-01-01
269.755\t270.696\t2021-02-01
270.696\t271.713\t2021-03-01
271.713\t273.968\t2021-04-01
273.968\t275.893\t2021-05-01
275.893\t278.218\t2021-06-01
278.218\t279.146\t2021-07-01
279.146\t279.507\t2021-08-01
279.507\t279.884\t2021-09-01
279.884\t281.617\t2021-10-01
281.617\t282.754\t2021-11-01
282.754\t283.908\t2021-12-01
283.908\t285.996\t2022-01-01
285.996\t288.059\t2022-02-01
288.059\t289.305\t2022-03-01
289.305\t290.846\t2022-04-01
290.846\t292.506\t2022-05-01
292.506\t294.680\t2022-06-01
294.680\t295.646\t2022-07-01
295.646\t297.178\t2022-08-01
297.178\t298.442\t2022-09-01
298.442\t299.315\t2022-10-01
299.315\t299.600\t2022-11-01
299.600\t300.113\t2022-12-01
300.113\t301.962\t2023-01-01
301.962\t304.011\t2023-02-01
304.011\t305.476\t2023-03-01
305.476\t306.899\t2023-04-01
306.899\t308.096\t2023-05-01
308.096\t308.910\t2023-06-01
308.910\t309.402\t2023-07-01
309.402\t310.103\t2023-08-01
310.103\t310.817\t2023-09-01
310.817\t311.380\t2023-10-01
311.380\t311.606\t2023-11-01
311.606\t311.907\t2023-12-01
311.907\t313.623\t2024-01-01
313.623\t315.419\t2024-02-01
315.419\t317.088\t2024-03-01
317.088\t317.978\t2024-04-01
317.978\t318.629\t2024-05-01
318.629\t319.003\t2024-06-01
319.003\t319.214\t2024-07-01
319.214\t320.017\t2024-08-01
320.017\t321.109\t2024-09-01
321.109\t321.758\t2024-10-01
321.758\t321.947\t2024-11-01
321.947\t322.007\t2024-12-01
322.007\t323.842\t2025-01-01
323.842\t325.252\t2025-02-01
325.252\t325.933\t2025-03-01
325.933\t326.815\t2025-04-01
326.815\t327.509\t2025-05-01
327.509\t328.364\t2025-06-01
328.364\t328.980\t2025-07-01
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
    title='LLM (gpt-5) vs Previous Month Core CPI',
    ylabel='Previous Month Core CPI',
    early_mask=early_mask, late_mask=late_mask
)

next_stats = plot_scatter_with_regs(
    ax_next, merged, 'llm', 'Next_CPI_Value',
    title='LLM (gpt-5) vs Next Month Core CPI',
    ylabel='Next Month Core CPI',
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