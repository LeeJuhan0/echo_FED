#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import UnivariateSpline, PchipInterpolator

# =====================================================
# Paths
# =====================================================
BASE_DIR = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data")

NFCI_ANFCI_PATH = BASE_DIR / "nfci_anfci_monthly.xlsx"
SENTIMENT_CSV   = BASE_DIR / "sentiment_13+19_llm.csv"
CPI_CSV         = BASE_DIR / "cpi.csv"

OUT_FIG = BASE_DIR / "hist_panel_2x2_combined.png"

YEAR_START, YEAR_END = 2017, 2025
DENSE_POINTS = 300
BOOT_B = 2000
HALFWIDTH_S_FACTOR = 8.0

# =====================================================
# Helpers (shared)
# =====================================================
def read_csv_kr(path):
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Failed to read {path}")

def apply_white_style(ax):
    ax.set_facecolor("white")
    ax.grid(False)
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    ax.tick_params(colors="black")

def plot_hist(ax, values, bins=20, xlabel="", title=""):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    ax.hist(v, bins=bins, color="grey", alpha=0.8, edgecolor="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(alpha=0.2)

# =====================================================
# Panel-code helpers (⚠ 그대로 유지)
# =====================================================
def clean_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = re.sub(r",", "", s)
    s = re.sub(r"[^0-9\-\.\+]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan

def normalize_month_to_day1(s):
    if pd.isna(s):
        return np.nan
    s = str(s).replace(".", "-").replace("/", "-")
    if re.match(r"^\d{4}-\d{2}$", s):
        s += "-01"
    return s

def yearly_median_ci_bootstrap(df, year_col, val_col, B=BOOT_B):
    rng = np.random.default_rng(2025)
    out = []
    for y, vals in df[[year_col, val_col]].dropna().groupby(year_col)[val_col]:
        vals = vals.to_numpy(dtype=float)
        med = np.median(vals)
        if len(vals) > 1:
            meds = [np.median(rng.choice(vals, size=len(vals), replace=True)) for _ in range(B)]
            lo, hi = np.percentile(meds, [2.5, 97.5])
        else:
            lo = hi = med
        out.append((y, med, lo, hi))
    return pd.DataFrame(out, columns=["Year", "median", "lo", "hi"])

def eval_mean_on_dense(yr_df, dense_x):
    x = yr_df["Year"].to_numpy()
    y = yr_df["mean"].to_numpy()
    if len(x) <= 2:
        return np.interp(dense_x, x, y)
    return PchipInterpolator(x, y)(dense_x)

def smooth_halfwidth_on_dense(years, half_vals, dense_x):
    if len(years) <= 2:
        return np.interp(dense_x, years, half_vals)
    s = np.var(half_vals) * len(years) * HALFWIDTH_S_FACTOR
    sp = UnivariateSpline(years, half_vals, s=s)
    return np.maximum(sp(dense_x), 0)

# =====================================================
# Load data
# =====================================================
nf = pd.read_excel(NFCI_ANFCI_PATH)
nf.columns = nf.columns.str.lower()
nf["anfci"] = pd.to_numeric(nf["anfci"], errors="coerce")

cpi = read_csv_kr(CPI_CSV)
cpi.columns = cpi.columns.str.lower()
diff_col = [c for c in cpi.columns if "diff" in c][0]
cpi_diff = pd.to_numeric(cpi[diff_col], errors="coerce")

sent = read_csv_kr(SENTIMENT_CSV)
sent = sent.rename(columns={"statment": "Statement"})
for c in ["Statement", "Theory", "Policy"]:
    sent[c] = pd.to_numeric(sent[c], errors="coerce")

combined_sentiment = np.concatenate([
    sent["Statement"].to_numpy(float),
    sent["Theory"].to_numpy(float),
    sent["Policy"].to_numpy(float),
])

sent["date_chr"] = sent["date"].apply(normalize_month_to_day1)
sent["Date"] = pd.to_datetime(sent["date_chr"], errors="coerce")
sent["Year"] = sent["Date"].dt.year
sent = sent[(sent["Year"] >= YEAR_START) & (sent["Year"] <= YEAR_END)]

pts = sent[["Year", "Policy"]].dropna().rename(columns={"Policy": "val"})
yr = pts.groupby("Year").agg(
    mean=("val", "mean"),
    sd=("val", "std"),
    n=("val", "size")
).reset_index()
yr["sd"] = yr["sd"].fillna(0.0)
yr["se"] = yr["sd"] / np.sqrt(yr["n"])
yr["ci95"] = 1.96 * yr["se"]

# =====================================================
# Plot 2×2 (FINAL)
# =====================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax11, ax12, ax21, ax22 = axes.flat

# (1,1) ANFCI
plot_hist(
    ax11,
    nf["anfci"],
    xlabel="Monthly ANFCI",
    title="Distribution of Monthly ANFCI"
)

# (1,2) CPI diff
plot_hist(
    ax12,
    cpi_diff,
    xlabel="CPI diff",
    title="Distribution of CPI diff"
)

# (2,1) Sentiment histogram
plot_hist(
    ax21,
    combined_sentiment,
    xlabel="Sentiment score",
    title="Distribution of Statement / Theory / Policy"
)

# (2,2) Panel-code figure (⚠ 로직 그대로)
apply_white_style(ax22)

dense_x = np.linspace(YEAR_START, YEAR_END, DENSE_POINTS)
mean_dense = eval_mean_on_dense(yr, dense_x)
half_dense = smooth_halfwidth_on_dense(
    yr["Year"].to_numpy(),
    yr["ci95"].to_numpy(),
    dense_x
)

ax22.fill_between(
    dense_x,
    mean_dense - half_dense,
    mean_dense + half_dense,
    color="#2C6BED",
    alpha=0.30
)
ax22.plot(dense_x, mean_dense, color="#2C6BED", linewidth=2.2)

ax22.scatter(
    yr["Year"], yr["mean"],
    color="#2C6BED", edgecolor="white", s=54, zorder=5
)

rng = np.random.default_rng(2024)
jitter = rng.uniform(-0.08, 0.08, size=len(pts))
ax22.scatter(
    pts["Year"] + jitter,
    pts["val"],
    color="black", s=18, alpha=0.8
)

ax22.set_title("GraphRAG Policy sentiment")
ax22.set_xlabel("Year")
ax22.set_ylabel("Sentiment score")

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=300)
print(f"Saved: {OUT_FIG}")
plt.show()
