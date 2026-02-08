# -*- coding: utf-8 -*-
"""
sentiment.csv의 date(YYYY-MM) 월 목록에 맞춰
nfci-data-series.excel.xlsx의 주간 데이터를 월평균(NFCI, ANFCI)으로 집계하여
date, nfci, anfci 컬럼으로 엑셀 파일로 저장합니다.

필요 라이브러리:
- pandas
- openpyxl (xlsx 저장/읽기용)

pip install pandas openpyxl
"""

import pandas as pd
from pathlib import Path

def main():
    # 파일 경로 설정
    excel_path = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\nfci-data-series.excel.xlsx")
    sentiment_path = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\sentiment.csv")
    output_path = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\nfci_anfci_monthly.xlsx")

    # 1) 주간 NFCI 데이터 로드
    df_weekly = pd.read_excel(excel_path)  # engine='openpyxl' 지정 가능
    df_weekly.columns = [c.strip() for c in df_weekly.columns]

    # 필수 컬럼 확인
    required_cols = {"Friday_of_Week", "NFCI", "ANFCI"}
    if not required_cols.issubset(set(df_weekly.columns)):
        raise ValueError(f"엑셀 파일에 {required_cols} 컬럼이 모두 존재해야 합니다. 현재 컬럼: {df_weekly.columns.tolist()}")

    # 날짜 파싱 (MM/DD/YYYY 형식)
    df_weekly["Friday_of_Week"] = pd.to_datetime(df_weekly["Friday_of_Week"], errors="coerce")
    df_weekly = df_weekly.dropna(subset=["Friday_of_Week"])

    # 월 키(YYYY-MM) 생성
    df_weekly["date"] = df_weekly["Friday_of_Week"].dt.to_period("M").astype(str)

    # 월별 평균(NFCI, ANFCI)
    monthly_avg = (
        df_weekly.groupby("date", as_index=False)[["NFCI", "ANFCI"]]
        .mean()
    )

    # 2) sentiment.csv 로드 (월 목록)
    df_sent = pd.read_csv(sentiment_path)
    df_sent.columns = [c.strip() for c in df_sent.columns]
    if "date" not in df_sent.columns:
        raise ValueError("sentiment.csv에는 'date' 컬럼(YYYY-MM 형식)이 필요합니다.")

    # 좌측(기준) 월 목록을 중복 없이 유지하며 추출
    months = df_sent[["date"]].drop_duplicates().copy()

    # 월 목록 기준으로 평균값 매칭 (좌측 기준으로 순서 유지)
    out = months.merge(monthly_avg, how="left", on="date")

    # 컬럼명 소문자 변경 및 정렬
    out = out.rename(columns={"NFCI": "nfci", "ANFCI": "anfci"})
    out = out[["date", "nfci", "anfci"]]

    # 엑셀로 저장
    out.to_excel(output_path, index=False)
    print(f"저장 완료: {output_path}")

    # 누락된 월(매칭 실패) 경고 출력
    missing = out[out["nfci"].isna() | out["anfci"].isna()]["date"].tolist()
    if missing:
        print(f"경고: NFCI/ANFCI 데이터가 없어 비어있는 월이 {len(missing)}개 있습니다: {missing}")

if __name__ == "__main__":
    main()