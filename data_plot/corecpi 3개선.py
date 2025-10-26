#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create two 2x2 figures from provided policy + CPI inline data:
 - Prev group (Prev_CPI_Value): 3 scatter/regression panels + histogram
 - Next group (Next_CPI_Value): 3 scatter/regression panels + histogram
This variant changes the 4th-panel histogram so it plots the combined frequency
of the three x-variables used in the scatter panels (Statement, P&B, Policy).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import io
from textwrap import dedent

# ---------- Inline data (unchanged; uses exactly your inputs) ----------
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
cpi_text = """Prev_CPI_Value\tNext_CPI_Value\tFOMC_date
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
policy_df_raw = pd.read_csv(io.StringIO(dedent(policy_text)), sep=r"\s+")
cpi_df = pd.read_csv(io.StringIO(dedent(cpi_text)), sep=r"\s+")
cpi_df['FOMC_date'] = pd.to_datetime(cpi_df['FOMC_date'])

# ---------- Build merged monthly mapping (month-level aggregation for CPI) ----------
cpi_df['month'] = cpi_df['FOMC_date'].dt.strftime('%Y-%m')
# monthly aggregation: mean Prev/Next, earliest FOMC_date
cpi_month = cpi_df.groupby('month', as_index=False).agg({
    'Prev_CPI_Value': 'mean',
    'Next_CPI_Value': 'mean',
    'FOMC_date': 'min'
}).rename(columns={'month': 'date'})

# normalize policy columns and merge on 'YYYY-MM'
policy_df = policy_df_raw.copy()
policy_df.columns = policy_df.columns.str.strip()
policy_df = policy_df.rename(columns={'statment': 'Statement', 'date': 'date'})
policy_df['date'] = policy_df['date'].str.strip()

merged = policy_df.merge(cpi_month[['date', 'Prev_CPI_Value', 'Next_CPI_Value', 'FOMC_date']],
                         on='date', how='left')

# check mapping
if merged['Prev_CPI_Value'].isna().any():
    print("Warning: some policy rows did not find CPI match. See rows:")
    print(merged[merged['Prev_CPI_Value'].isna()][['date']])

# ensure FOMC_date is datetime
merged['FOMC_date'] = pd.to_datetime(merged['FOMC_date'])
merged = merged.sort_values('FOMC_date').reset_index(drop=True)

# ---------- USER EDITABLE: subplot titles and x-axis labels ----------
# Prev group (top-3 panels)
prev_scatter_titles = [
    "Statement(St) vs Previous Month Core CPI",   # panel 1 title (top-left)
    "St+Theory(Th) vs Previous Month Core CPI",         # panel 2 title (top-right)
    "St+Th+Policy vs Previous Month Core CPI"       # panel 3 title (bottom-left)
]
prev_scatter_xlabels = [
    "Sentiment Score",         # panel 1 xlabel
    "Sentiment Score",               # panel 2 xlabel
    "Sentiment Score"             # panel 3 xlabel
]
prev_hist_title = "Previous Month Core CPI Distribution (all observations)"
prev_hist_xlabel = "Previous Month Core CPI"

# Next group (bottom-3 panels)
next_scatter_titles = [
    "Statement(St) vs Next Month Core CPI",
    "St+Theory(Th) vs Next Month Core CPI",
    "St+Th+Policy vs Next Month Core CPI"
]
next_scatter_xlabels = [
    "Sentiment Score",
    "Sentiment Score",
    "Sentiment Score"
]
next_hist_title = "Next Month Core CPI Distribution (all observations)"
next_hist_xlabel = "Next Month Core CPI"

# ---------- plotting helpers ----------
def fit_line(x, y):
    m, b = np.polyfit(x, y, 1)
    r, p = stats.pearsonr(x, y)
    return m, b, r, p

def plot_scatter_reg(ax, df, xcol, ycol, title, xlabel, ylabel, early_mask, late_mask,
                     early_color='#1f77b4', late_color='#d62728', overall_color='#2ca02c'):
    # scatter
    ax.scatter(df.loc[early_mask, xcol], df.loc[early_mask, ycol],
               color=early_color, edgecolors='white', linewidth=0.5, alpha=0.85, label='2017-2022')
    ax.scatter(df.loc[late_mask, xcol], df.loc[late_mask, ycol],
               color=late_color, edgecolors='white', linewidth=0.5, alpha=0.85, label='2023-2025')
    # segment fits
    def plot_seg(mask, color, label):
        xs = df.loc[mask, xcol].to_numpy(dtype=float)
        ys = df.loc[mask, ycol].to_numpy(dtype=float)
        if len(xs[~np.isnan(xs)]) < 2 or len(ys[~np.isnan(ys)]) < 2:
            return None
        m,b,r,p = fit_line(xs, ys)
        xs_plot = np.linspace(np.nanmin(xs), np.nanmax(xs), 120)
        ax.plot(xs_plot, m*xs_plot + b, color=color, lw=1.6, label=f"{label} (r={r:.2f})")
        return (m,b,r,p)
    stats_early = plot_seg(early_mask, early_color, "regression 17-22")
    stats_late = plot_seg(late_mask, late_color, "regression 23-25")
    # overall fit
    xs_all = df[xcol].to_numpy(dtype=float)
    ys_all = df[ycol].to_numpy(dtype=float)
    if len(xs_all[~np.isnan(xs_all)]) >= 2:
        m_all,b_all,r_all,p_all = fit_line(xs_all, ys_all)
        xs_plot = np.linspace(np.nanmin(xs_all), np.nanmax(xs_all), 200)
        ax.plot(xs_plot, m_all*xs_plot + b_all, color=overall_color, lw=1.8, linestyle='--',
                label=f"regression 17-25 (r={r_all:.2f})")
    # labels, title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # --- enforce uniform x-axis limits for scatter panels ---
    ax.set_xlim(-0.85, 0.85)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    return {"early": stats_early, "late": stats_late}

def plot_hist(ax, values, bins=20, color='grey', xlabel='CPI', title='Histogram'):
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    ax.hist(vals, bins=bins, color=color, alpha=0.8, edgecolor='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(alpha=0.2)

# ---------- Create Prev 2x2 figure ----------
merged['year'] = merged['FOMC_date'].dt.year
early_mask = merged['year'] <= 2022
late_mask  = merged['year'] >= 2023

def make_2x2_for(ycol, scatter_titles, scatter_xlabels, hist_title, hist_xlabel, out_fname):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()
    stats_dict = {}
    # three scatter panels (positions 0,1,2)
    xcols = ['Statement', 'P&B', 'Policy']
    for i, xcol in enumerate(xcols):
        title = scatter_titles[i]
        xlabel = scatter_xlabels[i]
        ylabel = "Previous Month Core CPI" if ycol.startswith("Prev") else "Next Month Core CPI"
        stats_dict[f"panel_{i+1}"] = plot_scatter_reg(axes_flat[i], merged, xcol, ycol, title, xlabel, ylabel, early_mask, late_mask)
    # histogram in 4th panel
    # NOTE: change here: instead of plotting the CPI values, we plot the combined frequency
    # of the three x-columns (Statement, P&B, Policy) as requested.
    combined_x = np.concatenate([
        merged['Statement'].to_numpy(dtype=float),
        merged['P&B'].to_numpy(dtype=float),
        merged['Policy'].to_numpy(dtype=float)
    ])
    plot_hist(axes_flat[3], combined_x, bins=20, color='grey', xlabel=hist_xlabel, title=hist_title)
    #plt.suptitle(f"{ycol} group — 2x2 summary", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_fname, dpi=300)
    print(f"Saved {out_fname}")
    plt.show()
    return stats_dict

# Prev figure
prev_stats = make_2x2_for(
    ycol='Prev_CPI_Value',
    scatter_titles=prev_scatter_titles,
    scatter_xlabels=prev_scatter_xlabels,
    hist_title=prev_hist_title,
    hist_xlabel=prev_hist_xlabel,
    out_fname='prev_group_2x2.png'
)

# Next figure
next_stats = make_2x2_for(
    ycol='Next_CPI_Value',
    scatter_titles=next_scatter_titles,
    scatter_xlabels=next_scatter_xlabels,
    hist_title=next_hist_title,
    hist_xlabel=next_hist_xlabel,
    out_fname='next_group_2x2.png'
)

# Optional: print a small summary of stats availability
print("\nPanel fit availability summary (None means not enough data for segment fit):")
for k,v in prev_stats.items():
    print("Prev", k, v)
for k,v in next_stats.items():
    print("Next", k, v)

# End of script