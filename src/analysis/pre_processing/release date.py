#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build CPI release-date table from BLS annual release schedules.

Outputs: cpi_release_dates_2016_2025.csv
Columns:
- observation_date (YYYY-MM-01): reference month identifier for CPI
- release_date (YYYY-MM-DD): BLS announcement date
- source_url: BLS schedule page used as source

Source pages (BLS official):
- https://www.bls.gov/schedule/<YEAR>/home.htm
"""

from __future__ import annotations

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path

OUT_CSV = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\cpi_release_dates_2016_2025.csv")

YEARS = list(range(2016, 2026))
BASE_URL = "https://www.bls.gov/schedule/{year}/home.htm"

# Example line pattern on BLS schedule pages:
# "Consumer Price Index for January 2017"
CPI_TITLE_RE = re.compile(r"^Consumer Price Index for\s+([A-Za-z]+)\s+(\d{4})\s*$", re.I)

# Date line pattern:
# "Wednesday, February 15, 2017"
DATE_RE = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})\s*$", re.I)

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}

def fetch_text_lines(url: str) -> list[str]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text("\n")
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]  # drop empty
    return lines

def parse_year(year: int) -> pd.DataFrame:
    url = BASE_URL.format(year=year)
    lines = fetch_text_lines(url)

    rows = []
    i = 0
    while i < len(lines):
        m = CPI_TITLE_RE.match(lines[i])
        if m:
            ref_month_name = m.group(1).lower()
            ref_year = int(m.group(2))
            ref_month = MONTH_MAP.get(ref_month_name)

            # The schedule typically has the announcement date within the next few lines.
            # We'll scan forward up to 10 lines to find a date line.
            release_date = None
            for j in range(1, 11):
                if i + j >= len(lines):
                    break
                dm = DATE_RE.match(lines[i + j])
                if dm:
                    rel_month_name = dm.group(2).lower()
                    rel_month = MONTH_MAP[rel_month_name]
                    rel_day = int(dm.group(3))
                    rel_year = int(dm.group(4))
                    release_date = datetime(rel_year, rel_month, rel_day).date()
                    break

            if ref_month and release_date:
                observation_date = datetime(ref_year, ref_month, 1).date()
                rows.append({
                    "observation_date": observation_date.isoformat(),
                    "release_date": release_date.isoformat(),
                    "source_url": url
                })
        i += 1

    return pd.DataFrame(rows)

def main():
    all_df = []
    for y in YEARS:
        print(f"[INFO] parsing {y} ...")
        df_y = parse_year(y)
        all_df.append(df_y)

    out = pd.concat(all_df, ignore_index=True).drop_duplicates()

    # Sort by observation_date
    out["observation_date"] = pd.to_datetime(out["observation_date"], format="%Y-%m-%d")
    out = out.sort_values("observation_date")
    out["observation_date"] = out["observation_date"].dt.strftime("%Y-%m-%d")

    # sanity: keep only 2016-01 ~ 2025-12 range (adjust if you want)
    out = out[(out["observation_date"] >= "2016-01-01") & (out["observation_date"] <= "2025-12-01")]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[DONE] saved: {OUT_CSV} (rows={len(out)})")

if __name__ == "__main__":
    main()
