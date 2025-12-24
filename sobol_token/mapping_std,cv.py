#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
정리_표준편차2.xlsx의 1행 파일명에서 (gid_index, round_adj_token, simscore_percentage)를 추출하고,
2행(표준편차), 3행(표준편차/평균)을 각각 std, cv로 매핑하며,
추가로 평균(mean = std / cv)도 함께 계산하여
fomc_token_simscore_results.csv에 std, cv, mean 열을 병합한 새 CSV를 생성합니다.
"""

import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


EXCEL_PATH = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\bundle\정리_표준편차2.xlsx")
BASE_CSV_PATH = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\fomc_token_simscore_results.csv")
OUTPUT_CSV_PATH = BASE_CSV_PATH.with_name(BASE_CSV_PATH.stem + "_with_std_cv.csv")


def read_excel_matrix(xlsx_path: Path) -> pd.DataFrame:
    # 헤더 없이 통째로 읽기 (첫 행에 파일명이 가로로 있음)
    return pd.read_excel(xlsx_path, header=None, engine="openpyxl")


def detect_rows(df: pd.DataFrame) -> Tuple[int, int, int]:
    """
    - 파일명(.csv)이 가장 많이 있는 행을 파일명 행으로 선정
    - 그 다음 행을 표준편차, 다다음 행을 표준편차/평균으로 가정
    필요 시 '표준편차' 텍스트 검색으로 보정
    """
    csv_counts = {}
    for i in range(len(df)):
        row = df.iloc[i].astype(str).fillna("")
        csv_counts[i] = sum(x.strip().endswith(".csv") for x in row)

    file_row = max(csv_counts, key=csv_counts.get)

    # 기본 가정
    std_row = file_row + 1
    cv_row = file_row + 2

    # 보정: '표준편차'가 있는 행 찾기
    for i in range(file_row + 1, min(file_row + 5, len(df))):
        if df.iloc[i].astype(str).str.contains("표준편차").any():
            std_row = i
            # '표준편차/평균'은 std_row 다음 행일 확률이 높음
            if i + 1 < len(df):
                cv_row = i + 1
            break

    return file_row, std_row, cv_row


def parse_params_from_filename(name: str) -> Optional[Tuple[int, int, float]]:
    """
    예: 2017__03__FOMC201703_(1,11000,15.0%).csv -> (1, 11000, 15.0)
    """
    name = str(name).strip()
    m = re.search(r"\((\d+)\s*,\s*(\d+)\s*,\s*([0-9]+(?:\.[0-9]+)?)%\)\.csv$", name)
    if not m:
        return None
    gid = int(m.group(1))
    token = int(m.group(2))
    sim_pct = float(m.group(3))  # 퍼센트 단위 (예: 15.0)
    return gid, token, sim_pct


def build_metrics_from_excel(df: pd.DataFrame, file_row: int, std_row: int, cv_row: int) -> pd.DataFrame:
    """
    엑셀에서 (gid, token, sim_pct) -> (std, cv, mean) 매핑 테이블 생성
    첫 번째 열(A열)은 라벨(예: '표준편차')일 수 있으므로 B열부터 사용
    mean은 std / cv로 계산 (여기서 cv는 '표준편차/평균' 값)
    """
    filenames: List[str] = (
        df.iloc[file_row, 1:].dropna().astype(str).str.strip().tolist()
    )
    std_values = pd.to_numeric(df.iloc[std_row, 1: 1 + len(filenames)], errors="coerce")
    cv_values = pd.to_numeric(df.iloc[cv_row, 1: 1 + len(filenames)], errors="coerce")

    # 평균(mean) 계산: mean = std / (std/mean) = std / cv
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_values = std_values.astype(float) / cv_values.astype(float)
        mean_values = mean_values.where(np.isfinite(mean_values), np.nan)

    rows = []
    for name, std, cv, mean in zip(filenames, std_values, cv_values, mean_values):
        params = parse_params_from_filename(name)
        if params is None:
            continue
        gid, token, sim_pct = params
        rows.append((gid, token, sim_pct, std, cv, mean))

    metrics_df = pd.DataFrame(
        rows,
        columns=["gid_index", "round_adj_token", "simscore_pct", "std", "cv", "mean"],
    )
    # 중복 키가 있다면 마지막 값 우선으로 정리
    metrics_df = (
        metrics_df.sort_index()
        .drop_duplicates(subset=["gid_index", "round_adj_token", "simscore_pct"], keep="last")
        .reset_index(drop=True)
    )
    return metrics_df


def try_read_csv(path: Path) -> pd.DataFrame:
    # 한글/윈도 인코딩 호환
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 마지막 시도 기본값
    return pd.read_csv(path)


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalize_simscore_to_pct(series: pd.Series) -> pd.Series:
    """
    simscore를 퍼센트(0~100 스케일)로 통일
    - 문자열에 '%'가 있으면 제거 후 float로
    - 숫자인 경우 값의 범위로 추정: 최대값 <= 1.0이면 100배, 아니면 그대로
    """
    s = series.copy()
    if s.dtype == object:
        s = s.astype(str).str.strip()
        s = s.str.replace("%", "", regex=False)
        s = pd.to_numeric(s, errors="coerce")
    # 숫자 스케일 추정
    if pd.api.types.is_numeric_dtype(s):
        max_val = pd.to_numeric(s, errors="coerce").dropna().max()
        if pd.isna(max_val):
            return s
        if max_val <= 1.0000001:
            s = s * 100.0
    return s


def main():
    # 1) 엑셀 로드 및 매핑 테이블 생성
    xl_df = read_excel_matrix(EXCEL_PATH)
    file_row, std_row, cv_row = detect_rows(xl_df)
    metrics_df = build_metrics_from_excel(xl_df, file_row, std_row, cv_row)

    if metrics_df.empty:
        raise RuntimeError("엑셀에서 (gid, token, sim_pct) → (std, cv, mean) 매핑을 추출하지 못했습니다. 파일명 형식을 확인하세요.")

    # 2) 기반 CSV 로드
    base_df = try_read_csv(BASE_CSV_PATH)

    # 3) 컬럼 탐색 (유연 매핑)
    gid_col = find_col(base_df, ["gid_index", "gid", "gid_idx", "gidIndex"])
    token_col = find_col(base_df, ["round_adj_token", "token", "round_token", "adj_token", "roundAdjToken"])
    sim_col = find_col(base_df, ["simscore_percentage", "simscore_percent", "simscore_pct", "simscore", "similarity_score", "simScore"])

    missing = [name for name, col in [("gid_index", gid_col), ("round_adj_token", token_col), ("simscore", sim_col)] if col is None]
    if missing:
        raise RuntimeError(f"기반 CSV에 다음 컬럼을 찾을 수 없습니다: {', '.join(missing)}")

    # 4) simscore를 퍼센트 스케일로 정규화 후 병합 키 생성
    base_df["_simscore_pct"] = normalize_simscore_to_pct(base_df[sim_col])

    # 병합 시 부동소수 오차를 줄이기 위해 소수 3자리로 반올림
    base_df["_sim_key"] = np.round(base_df["_simscore_pct"].astype(float), 3)
    metrics_df["_sim_key"] = np.round(metrics_df["simscore_pct"].astype(float), 3)

    # 5) 병합
    merged = base_df.merge(
        metrics_df[["gid_index", "round_adj_token", "_sim_key", "std", "cv", "mean"]],
        left_on=[gid_col, token_col, "_sim_key"],
        right_on=["gid_index", "round_adj_token", "_sim_key"],
        how="left",
        suffixes=("", "_metrics"),
    )

    # 6) 정리: 불필요한 조인 키 제거
    merged = merged.drop(columns=["gid_index", "round_adj_token", "_sim_key"])
    # 열 보장: std, cv, mean (없으면 NaN으로)
    for col in ("std", "cv", "mean"):
        if col not in merged.columns:
            merged[col] = np.nan

    # 7) 저장
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    matched_std = merged["std"].notna().sum()
    matched_cv = merged["cv"].notna().sum()
    matched_mean = merged["mean"].notna().sum()
    total = len(merged)
    print(f"완료: std {matched_std}/{total}, cv {matched_cv}/{total}, mean {matched_mean}/{total} 행 매핑")
    print(f"저장: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()