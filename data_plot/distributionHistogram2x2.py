#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------- Paths --------------------
BASE_DIR = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data")
NFCI_ANFCI_PATH = BASE_DIR / "nfci_anfci_monthly.xlsx"

SENTIMENT_CSV = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\sentiment_13+19_llm.csv"
CPI_CSV       = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\cpi.csv"

OUT_FIG = str(BASE_DIR / "hist_2x2_nfci_anfci_cpidiff_sentiment.png")

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

def plot_hist(ax, values, bins=20, color="grey", xlabel="", title=""):
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    ax.hist(vals, bins=bins, color=color, alpha=0.8, edgecolor="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(alpha=0.2)

# -------------------- Load NFCI / ANFCI --------------------
nfci_df = pd.read_excel(NFCI_ANFCI_PATH)
nfci_df.columns = [c.strip() for c in nfci_df.columns]
# 표준화: date/nfci/anfci 를 소문자로
nfci_df = nfci_df.rename(columns={c: c.lower() for c in nfci_df.columns})

need_cols = {"date", "nfci", "anfci"}
if not need_cols.issubset(set(nfci_df.columns)):
    raise ValueError(f"nfci_anfci_monthly.xlsx must contain columns {need_cols}. "
                     f"Found: {list(nfci_df.columns)}")

# 숫자형 강제
nfci_df["nfci"] = pd.to_numeric(nfci_df["nfci"], errors="coerce")
nfci_df["anfci"] = pd.to_numeric(nfci_df["anfci"], errors="coerce")

# -------------------- Load CPI (diff column) --------------------
cpi_df = read_csv_kr(CPI_CSV)
cpi_df.columns = [c.strip() for c in cpi_df.columns]

# 'diff' 컬럼 이름이 다를 가능성 대비 (대소문자/공백)
lower_map = {c.lower().strip(): c for c in cpi_df.columns}
if "diff" in lower_map:
    diff_col = lower_map["diff"]
else:
    # 혹시 'cpi_diff' 같은 이름이면 여기에 추가
    cand = None
    for k, orig in lower_map.items():
        if "diff" == k or k.endswith("_diff") or "diff" in k:
            cand = orig
            break
    if cand is None:
        raise ValueError(f"cpi.csv must contain a 'diff' column. Found: {list(cpi_df.columns)}")
    diff_col = cand

cpi_diff = pd.to_numeric(cpi_df[diff_col], errors="coerce").to_numpy(dtype=float)

# -------------------- Load Sentiment (Statement/Theory/Policy) --------------------
sent_df = read_csv_kr(SENTIMENT_CSV)
sent_df.columns = [c.strip() for c in sent_df.columns]

# statment -> Statement 정규화
col_map = {}
if "statment" in sent_df.columns:
    col_map["statment"] = "Statement"
elif "Statement" in sent_df.columns:
    col_map["Statement"] = "Statement"

if "Theory" in sent_df.columns:
    col_map["Theory"] = "Theory"
if "Policy" in sent_df.columns:
    col_map["Policy"] = "Policy"

sent_df = sent_df.rename(columns=col_map)

needed = ["Statement", "Theory", "Policy"]
missing = [c for c in needed if c not in sent_df.columns]
if missing:
    raise ValueError(f"Missing required columns in sentiment CSV: {missing}")

for c in needed:
    sent_df[c] = pd.to_numeric(sent_df[c], errors="coerce")

combined_sentiment = np.concatenate([
    sent_df["Statement"].to_numpy(dtype=float),
    sent_df["Theory"].to_numpy(dtype=float),
    sent_df["Policy"].to_numpy(dtype=float),
])

# -------------------- Plot 2x2 --------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ax11, ax12, ax21, ax22 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# (1,1) NFCI histogram
plot_hist(
    ax11,
    nfci_df["nfci"].to_numpy(dtype=float),
    bins=20,
    xlabel="Monthly NFCI",
    title="Distribution of Monthly NFCI"
)

# (1,2) ANFCI histogram
plot_hist(
    ax12,
    nfci_df["anfci"].to_numpy(dtype=float),
    bins=20,
    xlabel="Monthly ANFCI",
    title="Distribution of Monthly ANFCI"
)

# (2,1) CPI diff histogram
plot_hist(
    ax21,
    cpi_diff,
    bins=20,
    xlabel="CPI diff",
    title=f"Distribution of CPI diff ({diff_col})"
)

# (2,2) combined sentiment histogram
plot_hist(
    ax22,
    combined_sentiment,
    bins=20,
    xlabel="Sentiment Score",
    title="Distribution of Statement/Theory/Policy scores"
)

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=300)
print(f"Saved: {OUT_FIG}")
plt.show()
