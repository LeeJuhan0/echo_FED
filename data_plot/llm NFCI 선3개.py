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
-0.5051747427\t-0.4116244131\t2017-02-01
-0.5012278561\t-0.4264627969\t2017-03-15
-0.5248851848\t-0.4657276594\t2017-05-03
-0.5725607943\t-0.5263562456\t2017-06-14
-0.5893044337\t-0.5676246210\t2017-07-26
-0.5717574109\t-0.5050250777\t2017-09-20
-0.6076308315\t-0.5960663231\t2017-11-01
-0.6149953549\t-0.6316020939\t2017-12-13
-0.6143644059\t-0.6210717312\t2018-01-31
-0.5140657618\t-0.4957534277\t2018-03-21
-0.5469950403\t-0.5193089202\t2018-05-02
-0.5635640586\t-0.5382515628\t2018-06-13
-0.5719724724\t-0.5717436054\t2018-08-01
-0.6059188001\t-0.6352729661\t2018-09-26
-0.5439605423\t-0.5871509654\t2018-11-08
-0.4351837331\t-0.4555196331\t2018-12-19
-0.5139149293\t-0.5237359383\t2019-01-30
-0.6038582901\t-0.6013068435\t2019-03-20
-0.6196408433\t-0.6228027926\t2019-05-01
-0.5752064269\t-0.5905090711\t2019-06-19
-0.5813783612\t-0.5996808333\t2019-07-31
-0.5182709171\t-0.5659224441\t2019-09-18
-0.5529726217\t-0.5805069649\t2019-10-30
-0.5755378651\t-0.6257507225\t2019-11-11
-0.5673448594\t-0.5622399729\t2019-12-11
-0.6286392543\t-0.6075513360\t2020-01-29
0.04145253481\t0.06903925894\t2020-03-03
0.04145253481\t0.06903925894\t2020-03-15
0.1643774046\t0.2282937624\t2020-04-29
-0.3184473367\t-0.4144029943\t2020-06-10
-0.4635388763\t-0.5464601277\t2020-07-29
-0.5081334801\t-0.6991041804\t2020-09-16
-0.5116844951\t-0.5437072596\t2020-11-05
-0.5976715841\t-0.5305703901\t2020-12-16
-0.6255278526\t-0.5495374118\t2021-01-27
-0.6411606016\t-0.5881440857\t2021-03-17
-0.6820721653\t-0.5885676005\t2021-04-28
-0.6992240740\t-0.6158025896\t2021-06-16
-0.6660163319\t-0.6158368941\t2021-07-28
-0.6578002505\t-0.6626167635\t2021-09-22
-0.6263272400\t-0.6235932035\t2021-11-03
-0.5400240858\t-0.5023109474\t2021-12-15
-0.5577510432\t-0.5359246677\t2022-01-26
-0.3873114785\t-0.2755368823\t2022-03-16
-0.3317972676\t-0.2150508919\t2022-05-04
-0.2047625971\t-0.08634414817\t2022-06-15
-0.1941450165\t-0.1648524178\t2022-07-27
-0.1420047721\t-0.1679423134\t2022-09-21
-0.1204636447\t-0.1167039561\t2022-11-02
-0.1863351264\t-0.1916614742\t2022-12-14
-0.3065797690\t-0.3004480292\t2023-02-01
-0.1723375172\t-0.1306616195\t2023-03-22
-0.1915992989\t-0.2231477601\t2023-05-03
-0.2173785466\t-0.2425657858\t2023-06-14
-0.2666628768\t-0.2710770734\t2023-07-26
-0.3362841511\t-0.2871016529\t2023-09-20
-0.2851610288\t-0.2753451041\t2023-11-01
-0.3425432627\t-0.3715241523\t2023-12-13
-0.3993766902\t-0.4281602687\t2024-01-31
-0.4414308974\t-0.4853750538\t2024-03-20
-0.4150108365\t-0.4340774975\t2024-05-01
-0.4066587716\t-0.4023535485\t2024-06-12
-0.3659056084\t-0.3649585910\t2024-07-31
-0.4167244652\t-0.4566038506\t2024-09-18
-0.4535968780\t-0.5078742234\t2024-11-07
-0.4825094617\t-0.5182652175\t2024-12-18
-0.5086765442\t-0.5261303212\t2025-01-29
-0.4468250678\t-0.4547258862\t2025-03-19
-0.4048410170\t-0.4334718001\t2025-05-07
-0.4823948827\t-0.4792118789\t2025-06-18
-0.5320184188\t-0.5071333834\t2025-07-30
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
    title='LLM (gpt-5) vs Previous Week NFCI',
    ylabel='Previous Week NFCI',
    early_mask=early_mask, late_mask=late_mask
)

next_stats = plot_scatter_with_regs(
    ax_next, merged, 'llm', 'Next_CPI_Value',
    title='LLM (gpt-5) vs Previous Week ANFCI',
    ylabel='Previous Week ANFCI',
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