#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# -------------------- Paths --------------------
FOMC_CSV_PATH = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\cpi.csv"          # FOMC_date 열
CPI_CSV_PATH  = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\CPI2016~2025.csv" # observation_date + CPI + Release Dates

OUT_CSV_PATH  = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\cpi_fomc_window.csv"

DATE_FMT = "%Y-%m-%d"

# -------------------- Rule toggle --------------------
# ✅ 정보집합(일반적으로 타당): FOMC가 (직전월 CPI 발표일) 이후면 previous = -1, 아니면 -2
USE_STANDARD_INFOSET_RULE = True

# 네가 텍스트로 적어준 규칙(반대 방향)을 그대로 쓰려면:
# USE_STANDARD_INFOSET_RULE = False
#
# - False일 때:
#   FOMC_date > release_date 이면 previous = -2
#   FOMC_date < release_date 이면 previous = -1


# -------------------- Helpers --------------------
def read_csv_kr(path: str | Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV {path}. Last error: {last_err}")

def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    """해당 날짜가 속한 월의 1일(월 자료 key)"""
    return ts.to_period("M").to_timestamp(how="start")

def pick_cols_cpi(df: pd.DataFrame) -> tuple[str, str, str]:
    """
    CPI CSV에서 observation_date / release_date / cpi value 컬럼을 자동 탐지.
    기대 예시:
      observation_date, CPIAUCSL, Release Dates
    """
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    lower = {c: c.lower().replace(" ", "_") for c in cols}

    # observation_date
    obs_col = None
    for c in cols:
        if lower[c] in ("observation_date", "observationdate"):
            obs_col = c
            break
    if obs_col is None:
        # fallback: date 포함 컬럼 중 observation 우선
        date_like = [c for c in cols if "date" in lower[c]]
        if date_like:
            obs_col = date_like[0]
        else:
            obs_col = cols[0]

    # release_date
    rel_col = None
    for c in cols:
        if lower[c] in ("release_dates", "release_date", "releasedates", "releasedate"):
            rel_col = c
            break
    if rel_col is None:
        # fallback: 'release' 포함
        rel_like = [c for c in cols if "release" in lower[c]]
        if not rel_like:
            raise ValueError("CPI CSV에서 'Release Dates' (release date) 컬럼을 찾지 못했습니다.")
        rel_col = rel_like[0]

    # value (CPI)
    value_candidates = [c for c in cols if c not in (obs_col, rel_col)]
    if not value_candidates:
        raise ValueError("CPI CSV에서 CPI 값 컬럼을 찾지 못했습니다.")
    val_col = value_candidates[0]  # 보통 CPIAUCSL

    return obs_col, rel_col, val_col


# -------------------- Load FOMC dates --------------------
fomc_df = read_csv_kr(FOMC_CSV_PATH)
fomc_df.columns = [c.strip() for c in fomc_df.columns]
if "FOMC_date" not in fomc_df.columns:
    raise ValueError("cpi.csv에서 'FOMC_date' 컬럼을 찾지 못했습니다.")

fomc_df["FOMC_date"] = pd.to_datetime(fomc_df["FOMC_date"], errors="coerce")
fomc_df = fomc_df.dropna(subset=["FOMC_date"]).sort_values("FOMC_date").reset_index(drop=True)

# -------------------- Load CPI monthly + release dates --------------------
cpi_raw = read_csv_kr(CPI_CSV_PATH)
obs_col, rel_col, val_col = pick_cols_cpi(cpi_raw)

cpi_df = cpi_raw[[obs_col, rel_col, val_col]].copy()
cpi_df = cpi_df.rename(columns={obs_col: "observation_date", rel_col: "release_date", val_col: "cpi"})

# 날짜/값 파싱
cpi_df["observation_date"] = pd.to_datetime(cpi_df["observation_date"], format=DATE_FMT, errors="coerce")
cpi_df["release_date"] = pd.to_datetime(cpi_df["release_date"], format=DATE_FMT, errors="coerce")
cpi_df["cpi"] = pd.to_numeric(cpi_df["cpi"], errors="coerce")

cpi_df = cpi_df.dropna(subset=["observation_date"]).sort_values("observation_date").reset_index(drop=True)

# month key 생성
cpi_df["month"] = cpi_df["observation_date"].dt.to_period("M").dt.to_timestamp(how="start")

# ✅ CPI 값은 observation_date(month)로만 가져온다
cpi_map = cpi_df.dropna(subset=["cpi"]).set_index("month")["cpi"]

# release_date도 month(=observation month) 기준으로 매핑
release_map = cpi_df.dropna(subset=["release_date"]).drop_duplicates("month").set_index("month")["release_date"]

# -------------------- Build output --------------------
rows = []
for d in fomc_df["FOMC_date"]:
    base_month = month_start(d)  # FOMC가 속한 월 (YYYY-MM-01)

    # 비교에 쓸 release_date를 무엇으로 잡을지?
    # ✅ 일반적으로: "직전월 CPI"(base_month-1)의 release_date와 비교
    prev1_month = base_month + pd.DateOffset(months=-1)
    rel = release_map.get(prev1_month, pd.NaT)

    # ---- previous offset 결정 ----
    # standard: d > rel 이면 prev=-1, 아니면 prev=-2
    if USE_STANDARD_INFOSET_RULE:
        if pd.notna(rel) and (d >= rel):
            prev_month = prev1_month                       # -1
        else:
            prev_month = base_month + pd.DateOffset(months=-2)  # -2
    else:
        # ✅ 네가 텍스트로 적어준 규칙 그대로(반대 방향)
        if pd.notna(rel) and (d > rel):
            prev_month = base_month + pd.DateOffset(months=-2)  # -2
        else:
            prev_month = prev1_month                             # -1

    # ---- next j months (항상 observation_date 기준으로 월 이동) ----
    n1 = base_month + pd.DateOffset(months=+1)
    n2 = base_month + pd.DateOffset(months=+2)
    n3 = base_month + pd.DateOffset(months=+3)
    n4 = base_month + pd.DateOffset(months=+4)
    n5 = base_month + pd.DateOffset(months=+5)

    rows.append({
        "FOMC_date": d.strftime(DATE_FMT),
        "previous_CPI": cpi_map.get(prev_month, np.nan),
        "next_1month_CPI": cpi_map.get(n1, np.nan),
        "next_2month_CPI": cpi_map.get(n2, np.nan),
        "next_3month_CPI": cpi_map.get(n3, np.nan),
        "next_4month_CPI": cpi_map.get(n4, np.nan),
        "next_5month_CPI": cpi_map.get(n5, np.nan),
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")
print(f"Saved: {OUT_CSV_PATH}")
print(out_df.head(10))
