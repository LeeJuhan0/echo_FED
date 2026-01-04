#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import UnivariateSpline, PchipInterpolator

# -------------------- Paths --------------------
BASE_DIR = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data")
NFCI_ANFCI_PATH = BASE_DIR / "nfci_anfci_monthly.xlsx"

SENTIMENT_CSV = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\sentiment_13+19_llm.csv"
CPI_CSV       = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\cpi.csv"

OUT_FIG = str(BASE_DIR / "panel_2x2_nfci_anfci_cpi_policy.png")

YEAR_START, YEAR_END = 2017, 2025
DENSE_N = 400
BOOT_B = 2000  # bootstrap reps for median CI

# ---- (for Policy panel EXACT reuse) ----
DENSE_POINTS = 300
HALFWIDTH_S_FACTOR = 8.0
DEFAULT_Y_LIMS = (-0.85, 0.85)
XTICK_START, XTICK_END = 2017, 2025
XLIM_MIN, XLIM_MAX = 2016.9, 2025.1

# -------------------- Helpers --------------------
def read_csv_kr(path: str | Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV {path} with tried encodings. Last error: {last_err}")

def clean_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = re.sub(r",", "", s)
    s = re.sub(r"[\u2212\u2012\u2013\u2014]", "-", s)  # normalize unicode minus
    s = re.sub(r"[^0-9\-\.\+]", "", s)
    if s == "" or s in ["+", "-"]:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def parse_yyyymm_to_date(s: str) -> pd.Timestamp:
    s = str(s).strip()
    s = re.sub(r"[./]", "-", s)
    if re.match(r"^\d{4}-\d{2}$", s):
        s = s + "-01"
    return pd.to_datetime(s, errors="coerce")

def spline_smooth(x_years: np.ndarray, y_vals: np.ndarray, dense_x: np.ndarray, s_factor: float = 0.8):
    """
    ggplot-like smooth: do NOT force passing exactly through yearly points.
    """
    x_years = np.asarray(x_years, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)

    if len(x_years) <= 1:
        return np.full_like(dense_x, y_vals[0] if len(y_vals) else np.nan, dtype=float)
    if len(x_years) == 2:
        return np.interp(dense_x, x_years, y_vals)

    v = float(np.nanvar(y_vals)) if np.isfinite(np.nanvar(y_vals)) else 0.0
    s = v * len(x_years) * s_factor
    k = min(3, len(x_years) - 1)

    try:
        sp = UnivariateSpline(x_years, y_vals, s=s, k=k)
        return sp(dense_x)
    except Exception:
        return np.interp(dense_x, x_years, y_vals)

def apply_gg_style(ax):
    ax.set_facecolor("#EBEBEB")              # ggplot panel background
    ax.grid(True, color="white", linewidth=1)
    ax.grid(True, which="minor", color="white", linewidth=0.6)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(color="gray", labelcolor="black")


def yearly_median_ci_bootstrap(df: pd.DataFrame, year_col: str, val_col: str, B: int = BOOT_B, seed: int = 2025):
    rng = np.random.default_rng(seed)
    out = []
    g = df[[year_col, val_col]].dropna().groupby(year_col)[val_col]

    for y, vals in g:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        n = len(vals)
        if n == 0:
            continue

        med = float(np.median(vals))
        if n == 1:
            lo, hi = med, med
        else:
            meds = np.empty(B, dtype=float)
            for b in range(B):
                sample = rng.choice(vals, size=n, replace=True)
                meds[b] = np.median(sample)
            lo = float(np.percentile(meds, 2.5))
            hi = float(np.percentile(meds, 97.5))

        out.append((int(y), n, med, lo, hi))

    res = pd.DataFrame(out, columns=["Year", "n", "median", "lo", "hi"]).sort_values("Year").reset_index(drop=True)
    return res

def plot_year_strip_median(ax, df, year_col, val_col, title, ylabel):
    """
    ggplot-like panel:
    - grey background + white grid
    - black points at x=Year (no jitter)
    - blue smooth median line
    - grey 95% CI ribbon
    """
    apply_gg_style(ax)

    d = df[[year_col, val_col]].dropna().copy()
    d = d[np.isfinite(d[val_col])]
    ax.scatter(d[year_col].astype(int), d[val_col].astype(float),
               s=18, color="black", alpha=0.9, zorder=3)

    ym = yearly_median_ci_bootstrap(d, year_col, val_col, B=BOOT_B)
    ym = ym[(ym["Year"] >= YEAR_START) & (ym["Year"] <= YEAR_END)]
    if ym.empty:
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(YEAR_START, YEAR_END + 1))
        ax.set_xlim(YEAR_START - 0.2, YEAR_END + 0.2)
        return

    years = ym["Year"].to_numpy(dtype=float)
    dense_x = np.linspace(YEAR_START, YEAR_END, DENSE_N)

    med_dense = spline_smooth(years, ym["median"].to_numpy(dtype=float), dense_x, s_factor=0.6)
    lo_dense  = spline_smooth(years, ym["lo"].to_numpy(dtype=float), dense_x, s_factor=0.8)
    hi_dense  = spline_smooth(years, ym["hi"].to_numpy(dtype=float), dense_x, s_factor=0.8)

    lo_dense2 = np.minimum(lo_dense, hi_dense)
    hi_dense2 = np.maximum(lo_dense, hi_dense)

    ax.fill_between(dense_x, lo_dense2, hi_dense2, color="grey", alpha=0.25, linewidth=0, zorder=1)
    ax.plot(dense_x, med_dense, color="#2C6BED", linewidth=2.2, zorder=2)  # blue smooth line

    ax.set_xticks(np.arange(YEAR_START, YEAR_END + 1))
    ax.set_xlim(YEAR_START - 0.2, YEAR_END + 0.2)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)


# -------------------- EXACT reuse from your 2nd code (Policy panel) --------------------
def normalize_month_to_day1(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    s = re.sub(r"[./]", "-", s)
    if re.match(r"^\d{4}-\d{2}$", s):
        s = s + "-01"
    return s

def compute_yearly_stats(df, col):
    pts = df[["Year", col]].dropna().rename(columns={col: "val"})
    if pts.empty:
        return None
    yr = pts.groupby("Year", as_index=False).agg(
        n=("val", "size"),
        mean=("val", "mean"),
        sd=("val", "std")
    )
    yr["sd"] = yr["sd"].fillna(0.0)
    yr["se"] = yr["sd"] / np.sqrt(yr["n"])
    yr["ci95"] = 1.96 * yr["se"]
    yr["ymin"] = yr["mean"] - yr["ci95"]
    yr["ymax"] = yr["mean"] + yr["ci95"]
    yr = yr.sort_values("Year").reset_index(drop=True)
    return {"yr": yr, "pts": pts}

def eval_mean_on_dense(yr_df, dense_x):
    years = yr_df["Year"].to_numpy()
    mean_vals = yr_df["mean"].to_numpy()
    if len(years) == 1:
        mean_dense = np.full_like(dense_x, mean_vals[0], dtype=float)
    elif len(years) == 2:
        mean_dense = np.interp(dense_x, years, mean_vals)
    else:
        pchip = PchipInterpolator(years, mean_vals, extrapolate=True)
        mean_dense = pchip(dense_x)
    for yv, mv in zip(years, mean_vals):
        idx = np.abs(dense_x - yv).argmin()
        mean_dense[idx] = mv
    return mean_dense

def smooth_halfwidth_on_dense(years, half_vals, dense_x, s_factor=HALFWIDTH_S_FACTOR):
    if len(years) < 2:
        return np.full_like(dense_x, half_vals[0] if len(half_vals) == 1 else 0.0, dtype=float)
    v = np.var(half_vals) if np.isfinite(np.var(half_vals)) else 0.0
    s_param = v * len(years) * s_factor
    k = min(3, max(1, len(years) - 1))
    try:
        sp = UnivariateSpline(years, half_vals, s=s_param, k=k)
        half_dense = sp(dense_x)
    except Exception:
        half_dense = np.interp(dense_x, years, half_vals)
    for yv, hv in zip(years, half_vals):
        idx = np.abs(dense_x - yv).argmin()
        half_dense[idx] = hv
    half_dense = np.maximum(half_dense, 0.0)
    return half_dense

def plot_yearly_mean_ci_single(ax, yr_stats, dense_x, mean_dense, half_dense, title, line_color, show_points=True, y_limits=DEFAULT_Y_LIMS):
    ymin_dense = mean_dense - half_dense
    ymax_dense = mean_dense + half_dense
    ax.fill_between(dense_x, ymin_dense, ymax_dense, color="grey", alpha=0.45, linewidth=0)
    ax.plot(dense_x, mean_dense, color=line_color, linewidth=2.2)
    years = yr_stats["yr"]["Year"].to_numpy()
    mean_vals = yr_stats["yr"]["mean"].to_numpy()
    if show_points:
        ax.scatter(years, mean_vals, color=line_color, edgecolor="white", zorder=5, s=54)
    rng = np.random.default_rng(2024)
    jitter = rng.uniform(-0.08, 0.08, size=len(yr_stats["pts"]))
    ax.scatter(yr_stats["pts"]["Year"] + jitter, yr_stats["pts"]["val"], color="black", s=14, alpha=0.9)
    ax.set_xticks(np.arange(XTICK_START, XTICK_END + 1))
    ax.set_xlim(XLIM_MIN, XLIM_MAX)
    ax.set_ylim(y_limits)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Sentiment Score")

# -------------------- Load NFCI/ANFCI --------------------
nf = pd.read_excel(NFCI_ANFCI_PATH)
nf.columns = [c.strip() for c in nf.columns]
nf = nf.rename(columns={c: c.lower() for c in nf.columns})

for col in ["date", "nfci", "anfci"]:
    if col not in nf.columns:
        raise ValueError(f"nfci_anfci_monthly.xlsx must have columns: date, nfci, anfci. Found: {list(nf.columns)}")

nf["Date"] = nf["date"].apply(parse_yyyymm_to_date)
nf["Year"] = nf["Date"].dt.year
nf["nfci"] = pd.to_numeric(nf["nfci"], errors="coerce")
nf["anfci"] = pd.to_numeric(nf["anfci"], errors="coerce")
nf = nf[(nf["Year"] >= YEAR_START) & (nf["Year"] <= YEAR_END)].copy()

# -------------------- Load CPI (Next_CPI_Value -> CPI monthly -> yearly strip) --------------------
cpi = read_csv_kr(CPI_CSV)
cpi.columns = [c.strip() for c in cpi.columns]
lower_map = {c.lower().strip(): c for c in cpi.columns}

def pick_col(*cands):
    for cand in cands:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

fomc_col = pick_col("FOMC_date", "fomc_date", "date")
next_col = pick_col("diff", "diff", "next_cpi")

if fomc_col is None or next_col is None:
    raise ValueError(f"cpi.csv must have FOMC_date and diff columns. Found: {list(cpi.columns)}")

cpi["FOMC_date"] = pd.to_datetime(cpi[fomc_col], errors="coerce")
cpi["diff"] = pd.to_numeric(cpi[next_col], errors="coerce")
cpi["month"] = cpi["FOMC_date"].dt.to_period("M").dt.to_timestamp()
cpi_month = (
    cpi.dropna(subset=["month", "diff"])
    .groupby("month", as_index=False)["diff"].mean()
    .rename(columns={"diff": "CPI", "month": "Date"})
)
cpi_month["Year"] = cpi_month["Date"].dt.year
cpi_month = cpi_month[(cpi_month["Year"] >= YEAR_START) & (cpi_month["Year"] <= YEAR_END)].copy()

# -------------------- Load Sentiment (for Policy panel EXACT) --------------------
sent_df = read_csv_kr(SENTIMENT_CSV)
sent_df.columns = [c.strip() for c in sent_df.columns]

col_map = {}
if "statment" in sent_df.columns:
    col_map["statment"] = "Statement"
elif "Statement" in sent_df.columns:
    col_map["Statement"] = "Statement"
if "Theory" in sent_df.columns:
    col_map["Theory"] = "Theory"
if "Policy" in sent_df.columns:
    col_map["Policy"] = "Policy"
if "date" in sent_df.columns:
    col_map["date"] = "date"

sent_df = sent_df.rename(columns=col_map)

required_cols = ["date", "Statement", "Theory", "Policy"]
missing = [c for c in required_cols if c not in sent_df.columns]
if missing:
    raise ValueError(f"Missing required columns in sentiment CSV: {missing}")

sent_df = sent_df[required_cols].copy()
for c in ["Statement", "Theory", "Policy"]:
    sent_df[c] = sent_df[c].apply(clean_num)

sent_df["date_chr"] = sent_df["date"].apply(normalize_month_to_day1)
sent_df["Date"] = pd.to_datetime(sent_df["date_chr"], errors="coerce")
sent_df["Year"] = sent_df["Date"].dt.year
sent_df = sent_df[(~sent_df["Year"].isna()) & (sent_df["Year"] >= XTICK_START) & (sent_df["Year"] <= XTICK_END)].copy()

policy_stats = compute_yearly_stats(sent_df, "Policy")
dense_x_policy = np.linspace(float(XTICK_START), float(XTICK_END), DENSE_POINTS)

# -------------------- Plot 2x2 --------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax00, ax01 = axes[0, 0], axes[0, 1]
ax10, ax11 = axes[1, 0], axes[1, 1]

# ✅ SWAP TOP PANELS:
# (0,0) should be NFCI now
plot_year_strip_median(
    ax00, nf, "Year", "nfci",
    title="NFCI (National Financial Conditions Index)",
    ylabel="NFCI"
)

# (0,1) should be ANFCI now
plot_year_strip_median(
    ax01, nf, "Year", "anfci",
    title="ANFCI (Adjusted National Financial Conditions Index)",
    ylabel="ANFCI"
)

# (1,0) CPI unchanged
plot_year_strip_median(
    ax10, cpi_month, "Year", "CPI",
    title="CPI (Consumer Price Index diff)",
    ylabel="CPI"
)

# ✅ (1,1) RIGHT-BOTTOM = EXACT reuse of your 2nd code's (2,1) Policy panel
if policy_stats is not None:
    mean_dense_policy = eval_mean_on_dense(policy_stats["yr"], dense_x_policy)
    half_dense_policy = smooth_halfwidth_on_dense(
        policy_stats["yr"]["Year"].to_numpy(),
        policy_stats["yr"]["ci95"].to_numpy(),
        dense_x_policy
    )
    plot_yearly_mean_ci_single(
        ax11, policy_stats, dense_x_policy,
        mean_dense_policy, half_dense_policy,
        title="GraphRAG[St+Th+Policy]", line_color="blue",
        show_points=True, y_limits=DEFAULT_Y_LIMS
    )
else:
    ax11.text(0.5, 0.5, "No data: Policy", ha="center", va="center")
    ax11.set_ylim(DEFAULT_Y_LIMS)

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=300)
print(f"Saved: {OUT_FIG}")
plt.show()
