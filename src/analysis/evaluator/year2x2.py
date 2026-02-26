#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import UnivariateSpline, PchipInterpolator
plt.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 17,
})

# -------------------- Paths --------------------
BASE_DIR = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\results\external")
NFCI_ANFCI_PATH = BASE_DIR / "nfci_anfci_monthly.xlsx"

SENTIMENT_CSV = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\results\experiment\except_theory_Simulation_score\Simulation_except_theory.csv"
CPI_CSV       = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\cpi.csv"
BASE_DIR_out = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\evalution")
OUT_FIG = str(BASE_DIR_out / "panel_2x2_nfci_anfci_cpi_wo_th.png")

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

def apply_white_style(ax):
    ax.set_facecolor("white")
    ax.grid(False)
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    ax.tick_params(colors="black")



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
# -------------------- (여기부터 대체) Plot single combined panel --------------------
fig, ax = plt.subplots(figsize=(13, 9))
apply_white_style(ax)

dense_x = np.linspace(YEAR_START, YEAR_END, DENSE_N)

# --- NFCI (빨강, 점선) on primary y-axis ---
# --- ANFCI (red, dashed) on primary y-axis ---
d_af = nf[["Year", "anfci"]].dropna().copy()
d_af = d_af[np.isfinite(d_af["anfci"])]

if not d_af.empty:
    ym_af = yearly_median_ci_bootstrap(d_af, "Year", "anfci", B=BOOT_B)
    ym_af = ym_af[(ym_af["Year"] >= YEAR_START) & (ym_af["Year"] <= YEAR_END)]

    if not ym_af.empty:
        years_af = ym_af["Year"].to_numpy(dtype=float)
        med_af = spline_smooth(
            years_af,
            ym_af["median"].to_numpy(dtype=float),
            dense_x,
            s_factor=0.6
        )
        lo_af = spline_smooth(years_af, ym_af["lo"].to_numpy(dtype=float), dense_x, 0.8)
        hi_af = spline_smooth(years_af, ym_af["hi"].to_numpy(dtype=float), dense_x, 0.8)

        lo_af2 = np.minimum(lo_af, hi_af)
        hi_af2 = np.maximum(lo_af, hi_af)

        ax.fill_between(
            dense_x, lo_af2, hi_af2,
            color="red", alpha=0.18, linewidth=0, zorder=1
        )
        ax.plot(
            dense_x, med_af,
            color="red", linewidth=2.2, linestyle="--",
            label="ANFCI (median)", zorder=3
        )

        # points
        ax.scatter(d_af["Year"].astype(int), d_af["anfci"].astype(float), s=24, color="red", edgecolor="white", zorder=4)

# --- Policy (파랑, 실선) on primary y-axis ---
if policy_stats is not None:
    mean_dense_policy = eval_mean_on_dense(policy_stats["yr"], dense_x_policy)
    half_dense_policy = smooth_halfwidth_on_dense(
        policy_stats["yr"]["Year"].to_numpy(),
        policy_stats["yr"]["ci95"].to_numpy(),
        dense_x_policy
    )
    # align dense_x lengths (dense_x_policy may differ); we used dense_x_policy earlier
    ax.fill_between(dense_x_policy, mean_dense_policy - half_dense_policy, mean_dense_policy + half_dense_policy,
                    color="#2C6BED", alpha=0.30, linewidth=0, zorder=1)
    ax.plot(dense_x_policy, mean_dense_policy, color="#2C6BED", linewidth=2.2, linestyle="-", label="GraphRAG[Policy] (mean)", zorder=3)
    # points for yearly means
    years_pol = policy_stats["yr"]["Year"].to_numpy()
    mean_vals_pol = policy_stats["yr"]["mean"].to_numpy()
    ax.scatter(years_pol, mean_vals_pol, color="#2C6BED", edgecolor="white", zorder=5, s=54)

    # --- GraphRAG distribution points (black, jittered) ---
    rng = np.random.default_rng(2024)
    jitter = rng.uniform(-0.08, 0.08, size=len(policy_stats["pts"]))

    ax.scatter(
        policy_stats["pts"]["Year"] + jitter,
        policy_stats["pts"]["val"],
        color="black",
        s=18,
        alpha=0.8,
        zorder=4,
        label="_nolegend_"
    )


# --- CPI (초록, dash-dot) on secondary y-axis (오른쪽 보조축) ---
ax2 = ax.twinx()
apply_white_style(ax2)
d_cpi = cpi_month[["Year", "CPI"]].dropna().copy()
d_cpi = d_cpi[np.isfinite(d_cpi["CPI"])]
if not d_cpi.empty:
    ym_cpi = yearly_median_ci_bootstrap(d_cpi, "Year", "CPI", B=BOOT_B)
    ym_cpi = ym_cpi[(ym_cpi["Year"] >= YEAR_START) & (ym_cpi["Year"] <= YEAR_END)]
    if not ym_cpi.empty:
        years_cpi = ym_cpi["Year"].to_numpy(dtype=float)
        med_cpi = spline_smooth(years_cpi, ym_cpi["median"].to_numpy(dtype=float), dense_x, s_factor=0.6)
        lo_cpi  = spline_smooth(years_cpi, ym_cpi["lo"].to_numpy(dtype=float), dense_x, s_factor=0.8)
        hi_cpi  = spline_smooth(years_cpi, ym_cpi["hi"].to_numpy(dtype=float), dense_x, s_factor=0.8)
        lo_cpi2 = np.minimum(lo_cpi, hi_cpi)
        hi_cpi2 = np.maximum(lo_cpi, hi_cpi)
        ax2.fill_between(dense_x, lo_cpi2, hi_cpi2, color="green", alpha=0.18, linewidth=0, zorder=1)
        ax2.plot(dense_x, med_cpi, color="green", linewidth=2.2, linestyle="-.", label="CPI Change(median)", zorder=3)
        ax2.scatter(d_cpi["Year"].astype(int), d_cpi["CPI"].astype(float), s=24, color="green", edgecolor="white", zorder=4)

# --- Axis formatting & legend ---
ax.set_xticks(np.arange(YEAR_START, YEAR_END + 1))
ax.set_xlim(XLIM_MIN, XLIM_MAX)
ax.set_xlabel("Year")
ax.set_ylabel("NFCI / GraphRAG[Policy]")

ax2.set_ylabel("CPI change")

# create a combined legend (handles from both axes)
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=True, fontsize=15)

ax.set_title("NFCI (red), CPI (green), Policy (blue)")

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=300)
print(f"Saved: {OUT_FIG}")
plt.show()
# -------------------- (여기까지 대체) --------------------
