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
    "Statement(St) vs Previous Week NFCI",   # panel 1 title (top-left)
    "St+Theory(Th) vs Previous Week NFCI",         # panel 2 title (top-right)
    "St+Th+Policy vs Previous Week NFCI"       # panel 3 title (bottom-left)
]
prev_scatter_xlabels = [
    "Sentiment Score",         # panel 1 xlabel
    "Sentiment Score",               # panel 2 xlabel
    "Sentiment Score"             # panel 3 xlabel
]
prev_hist_title = "Previous Week NFCI Distribution (all observations)"
prev_hist_xlabel = "Previous Week NFCI"

# Next group (bottom-3 panels)
next_scatter_titles = [
    "Statement(St) vs Previous Week ANFCI",
    "St+Theory(Th) vs Previous Week ANFCI",
    "St+Th+Policy vs Previous Week ANFCI"
]
next_scatter_xlabels = [
    "Sentiment Score",
    "Sentiment Score",
    "Sentiment Score"
]
next_hist_title = "Previous Week ANFCI Distribution (all observations)"
next_hist_xlabel = "Previous Week ANFCI"

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
        ylabel = "Previous Week NFCI" if ycol.startswith("Prev") else "Previous Week ANFCI"
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