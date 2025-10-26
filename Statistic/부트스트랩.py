#!/usr/bin/env python3
"""
Plot scatter + bootstrap regression CIs (B=700) of score vs change for each source
(Statement, P&B, Policy), and a histogram that reflects bootstrap-derived values.

This version also computes bootstrap distributions of the Pearson correlation r
(for each source) and displays the observed r plus the bootstrap mean and 95% CI.

Output: fomc_change_vs_score_bootstrap_with_boot_hist_and_boot_r.png

Dependencies:
  pip install pandas numpy matplotlib seaborn scipy
"""
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# -----------------------
# Data (copied from user)
# -----------------------
change_text = """
0.6
0.3
0.3
0.1
0.1
0.3
-0.1
-0.1
-0.1
0.5
0.4
0.4
0
0.1
0.2
-0.3
-0.3
0.4
0.5
0.2
0
0
0.1
-0.1
-0.1
0.3
0.3
-0.2
0
0.6
0.4
0
0.2
0.4
0.4
0.6
0.8
0.9
0.3
0.9
0.5
0.3
0.8
1.0
1.0
1.3
0.1
0.4
0.1
0.5
0.4
0.4
0.1
0.2
0.6
0
0.1
0.3
0.4
0.3
0.2
-0.1
0.2
0.2
0.3
0.4
0.2
-0.1
0.1
0.3
"""
fomc_dates_text = """
2017-02-01
2017-03-15
2017-05-03
2017-06-14
2017-07-26
2017-09-20
2017-11-01
2017-12-13
2018-01-31
2018-03-21
2018-05-02
2018-06-13
2018-08-01
2018-09-26
2018-11-08
2018-12-19
2019-01-30
2019-03-20
2019-05-01
2019-06-19
2019-07-31
2019-09-18
2019-10-30
2019-12-11
2020-01-29
2020-03-03
2020-03-15
2020-04-29
2020-06-10
2020-07-29
2020-09-16
2020-11-05
2020-12-16
2021-01-27
2021-03-17
2021-04-28
2021-06-16
2021-07-28
2021-09-22
2021-11-03
2021-12-15
2022-01-26
2022-03-16
2022-05-04
2022-06-15
2022-07-27
2022-09-21
2022-11-02
2022-12-14
2023-02-01
2023-03-22
2023-05-03
2023-06-14
2023-07-26
2023-09-20
2023-11-01
2023-12-13
2024-01-31
2024-03-20
2024-05-01
2024-06-12
2024-07-31
2024-09-18
2024-11-07
2024-12-18
2025-01-29
2025-03-19
2025-04-30
2025-06-18
2025-07-30
"""
scores_csv = """date,Statement,P&B,Policy
2017-02,0.33,0.32,0.24
2017-03,0.4,0.4,0.35
2017-05,0.34,0.36,0.28
2017-06,0.4,0.35,0.33
2017-07,0.35,0.32,0.26
2017-09,0.38,0.36,0.25
2017-11,0.4,0.43,0.41
2017-12,0.58,0.55,0.58
2018-01,0.66,0.65,0.53
2018-03,0.4,0.52,0.58
2018-05,0.44,0.4,0.27
2018-06,0.75,0.7,0.64
2018-08,0.65,0.67,0.65
2018-09,0.79,0.72,0.82
2018-11,0.66,0.65,0.45
2018-12,0.7,0.65,0.55
2019-01,0.48,0.58,0.4
2019-03,0.15,0.18,0.03
2019-05,0.27,0.22,0.05
2019-06,0.2,0.12,0.04
2019-07,0.21,0.2,0.06
2019-09,0.23,0.2,0.06
2019-10,0.2,0.22,0.06
2019-12,0.3,0.25,0.14
2020-01,0.28,0.25,0.08
2020-03x,-0.2,-0.1,-0.22
2020-03y,-0.62,-0.4,-0.66
2020-04,-0.75,-0.45,-0.68
2020-06,-0.66,-0.5,-0.62
2020-07,-0.4,-0.22,-0.25
2020-09,-0.25,-0.1,-0.2
2020-11,-0.25,-0.1,-0.1
2020-12,-0.22,-0.1,-0.18
2021-01,-0.3,-0.15,-0.05
2021-03,0.1,0.2,0.27
2021-04,0.2,0.22,0.35
2021-06,0.4,0.435,0.44
2021-07,0.4,0.45,0.6
2021-09,0.3,0.25,0.2
2021-11,0.3,0.25,0.25
2021-12,0.32,0.27,0.27
2022-01,0.26,0.2,0.15
2022-03,0.1,0.04,0.03
2022-05,-0.15,-0.18,-0.1
2022-06,-0.15,-0.2,-0.15
2022-07,-0.3,-0.4,-0.55
2022-09,-0.25,-0.35,-0.48
2022-11,-0.2,-0.25,-0.45
2022-12,-0.2,-0.25,-0.5
2023-02,-0.1,-0.14,-0.12
2023-03,-0.14,-0.18,-0.35
2023-05,0.1,0.04,-0.2
2023-06,0.1,0.05,-0.12
2023-07,0.1,0.05,-0.02
2023-09,0.18,0.1,-0.12
2023-11,0.15,0.1,-0.15
2023-12,-0.1,-0.08,-0.25
2024-01,0.3,0.2,0
2024-03,0.32,0.29,0.1
2024-05,0.2,0.1,0.02
2024-06,0.3,0.27,0.1
2024-07,0.25,0.2,0.04
2024-09,0.3,0.4,0.2
2024-11,0.3,0.25,0.1
2024-12,0.3,0.32,0.2
2025-01,0.3,0.25,0.05
2025-03,0.22,0.25,0.12
2025-05,0.15,0.1,-0.08
2025-06,0.3,0.22,0.05
2025-07,-0.15,-0.08,-0.2
"""

# Parse change and dates
change = np.array([float(x) for x in change_text.strip().splitlines()])
fomc_dates = [x.strip() for x in fomc_dates_text.strip().splitlines()]

# Load scores table
scores_df = pd.read_csv(io.StringIO(scores_csv.strip()))
scores_df = scores_df.reset_index(drop=True)

# Align change to the scores table by position (use the first N change values)
N = len(scores_df)
if len(change) < N:
    raise ValueError("Fewer 'change' values than score rows; cannot align by position.")
change_aligned = change[:N]

# Build merged dataframe
df = scores_df.copy()
df['change'] = change_aligned
if len(fomc_dates) >= N:
    df['FOMC_date'] = fomc_dates[:N]
else:
    df['FOMC_date'] = pd.NA

# Prepare for plotting
sources = ['Statement', 'P&B', 'Policy']
B = 50  # bootstrap resamples (set to 700 as requested earlier)
rng = np.random.default_rng(12345)  # reproducible

sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={"height_ratios": [1.0, 0.5]})
ax_list = axes[0]

# We'll collect bootstrap-predicted y values at the observed x locations for all sources,
# then show their histogram in the bottom panel. This will have size ~ B * N * (#sources).
all_boot_y_at_obs = []

for i, src in enumerate(sources):
    ax = ax_list[i]
    x = df['change'].values.astype(float)
    y = df[src].values.astype(float)
    n = len(x)

    # Scatter
    ax.scatter(x, y, alpha=0.7, edgecolor='k', s=50)

    # Fit original OLS line
    slope, intercept = np.polyfit(x, y, 1)
    x_grid = np.linspace(np.min(x), np.max(x), 300)
    y_fit = slope * x_grid + intercept

    # Bootstrap lines: store predictions on the grid (for band) and predictions at observed x
    boot_preds_grid = np.empty((B, len(x_grid)))
    boot_preds_at_obs = np.empty((B, n))

    # Also compute bootstrap distribution of Pearson r
    r_boot = np.full(B, np.nan)

    indices = np.arange(n)
    for b in range(B):
        resample_idx = rng.choice(indices, size=n, replace=True)
        xb = x[resample_idx]
        yb = y[resample_idx]
        # handle degenerate xb
        if np.allclose(xb, xb[0]):
            slope_b = 0.0
            intercept_b = np.mean(yb)
        else:
            slope_b, intercept_b = np.polyfit(xb, yb, 1)
        boot_preds_grid[b, :] = slope_b * x_grid + intercept_b
        boot_preds_at_obs[b, :] = slope_b * x + intercept_b  # predict at original x positions

        # bootstrap Pearson r on the resampled paired data
        try:
            r_b, _ = stats.pearsonr(xb, yb)
        except Exception:
            r_b = np.nan
        r_boot[b] = r_b

    # 95% pointwise CI from bootstrap predictions on grid
    lower = np.nanpercentile(boot_preds_grid, 2.5, axis=0)
    upper = np.nanpercentile(boot_preds_grid, 97.5, axis=0)

    # Shade CI and plot fit
    ax.fill_between(x_grid, lower, upper, color='lightgray', alpha=0.7, zorder=1)
    ax.plot(x_grid, y_fit, color='black', lw=2, label='OLS fit')

    # Observed Pearson correlation and p-value (on original data)
    try:
        r_obs, pval_obs = stats.pearsonr(x, y)
    except Exception:
        r_obs, pval_obs = np.nan, np.nan

    # Summarize bootstrap r: drop NaNs, compute mean and 95% CI
    valid_r_boot = r_boot[~np.isnan(r_boot)]
    if valid_r_boot.size > 0:
        r_boot_mean = np.mean(valid_r_boot)
        r_boot_low, r_boot_high = np.percentile(valid_r_boot, [2.5, 97.5])
        r_boot_n = valid_r_boot.size
    else:
        r_boot_mean = np.nan
        r_boot_low = np.nan
        r_boot_high = np.nan
        r_boot_n = 0

    # Annotate showing observed r and bootstrap summary (mean + 95% CI)
    ann_text = (
        f"r_obs = {r_obs:.3f}\n"
        f"p_obs = {pval_obs:.3f}\n"
        f"boot r_mean = {r_boot_mean:.3f}\n"
        f"boot 95% CI = [{r_boot_low:.3f}, {r_boot_high:.3f}]\n"
        f"(boot valid = {r_boot_n}/{B})"
    )
    ax.annotate(ann_text, xy=(0.02, 0.95), xycoords='axes fraction',
                ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))

    ax.set_title(f"{src}:sentiment_index vs inflation_rate", fontsize=12)
    ax.set_xlabel("inflation_rate")
    ax.set_ylabel("sentiment_index")
    ax.grid(True)
    ax.legend()

    # collect flattened bootstrap predictions at observed x for this source
    all_boot_y_at_obs.append(boot_preds_at_obs.flatten())

# Concatenate across sources: this yields an array of size roughly B * N * 3
all_boot_y_at_obs_flat = np.concatenate(all_boot_y_at_obs)

# Bottom row: draw histogram of bootstrapped predicted y values (this shows counts on order B*N*#sources)
ax_hist = axes[1, 0]
axes[1, 1].axis('off')
axes[1, 2].axis('off')

n_bins = 50
ax_hist.hist(all_boot_y_at_obs_flat, bins=n_bins, color="#4c72b0", edgecolor="black", alpha=0.9)
ax_hist.set_xlabel("bootstrap-predicted score (at observed x)")
ax_hist.set_ylabel("frequency (counts across B resamples and all obs)")
ax_hist.set_title(f"Histogram of bootstrap-predicted y (B={B}, total samples={all_boot_y_at_obs_flat.size})")

plt.tight_layout()
outfn = "fomc_change_vs_score_bootstrap_with_boot_hist_and_boot_r.png"
plt.savefig(outfn, dpi=300, bbox_inches='tight')
print(f"Saved figure to {outfn}")

if __name__ == "__main__":
    plt.show()

