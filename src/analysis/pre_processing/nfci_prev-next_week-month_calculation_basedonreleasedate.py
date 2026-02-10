import pandas as pd
from pathlib import Path

"""
입력 데이터: NFCI 주별 데이터 (Friday_of_Week 기준), ex: nfci-data-series.excel.xlsx

프로그램 역할:
- FOMC 발표일(fomc_date, "MM/DD/YYYY")을 기준으로
  1) prev_week  : fomc_date 이전(또는 당일) 가장 가까운 Friday_of_Week 1개 값
  2) next_week  : fomc_date 이후 가장 가까운 Friday_of_Week 1개 값
  3) prev_month : fomc_date 이전(또는 당일) 주간값 중 최근 4개(=4주) 평균
  4) next_month : fomc_date 이후 주간값 중 다음 4개(=4주) 평균
- 위를 NFCI/ANFCI 모두 계산
- 결과를 엑셀로 저장

출력 컬럼(예):
fomc_date, prev_week_date, next_week_date,
prev_week_nfci, prev_week_anfci, next_week_nfci, next_week_anfci,
prev_month_nfci, prev_month_anfci, next_month_nfci, next_month_anfci,
count_prev4, count_next4
"""

# =========================
# 1) 입력 파일 경로
# =========================
input_path = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\nfci-data-series.excel.xlsx")

# =========================
# 2) 엑셀 읽기
# =========================
df = pd.read_excel(input_path, sheet_name=0)

# 컬럼명 공백/탭 정리
df.columns = [str(c).strip() for c in df.columns]

# 필수 컬럼 체크
required = ["Friday_of_Week", "NFCI", "ANFCI"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"필수 컬럼이 없습니다: {missing}\n현재 컬럼: {list(df.columns)}")

# 날짜 파싱
df["Friday_of_Week"] = pd.to_datetime(df["Friday_of_Week"], errors="coerce")
df = df.dropna(subset=["Friday_of_Week"]).copy()

# 날짜 정렬 + 중복 제거(있을 경우 대비)
df = df.sort_values("Friday_of_Week").drop_duplicates(subset=["Friday_of_Week"], keep="last").reset_index(drop=True)

# =========================
# 3) FOMC date 리스트 (문자열) -> datetime
# =========================
statement_dates_str = [
    "02/01/2017","03/15/2017","05/03/2017","06/14/2017","07/26/2017","09/20/2017","11/01/2017","12/13/2017",
    "01/31/2018","03/21/2018","05/02/2018","06/13/2018","08/01/2018","09/26/2018","11/08/2018","12/19/2018",
    "01/30/2019","03/20/2019","05/01/2019","06/19/2019","07/31/2019","09/18/2019","10/30/2019","12/11/2019",
    "01/29/2020","03/15/2020","04/29/2020","06/10/2020","07/29/2020","09/16/2020","11/05/2020","12/16/2020",
    "01/27/2021","03/17/2021","04/28/2021","06/16/2021","07/28/2021","09/22/2021","11/03/2021","12/15/2021",
    "01/26/2022","03/16/2022","05/04/2022","06/15/2022","07/27/2022","09/21/2022","11/02/2022","12/14/2022",
    "02/01/2023","03/22/2023","05/03/2023","06/14/2023","07/26/2023","09/20/2023","11/01/2023","12/13/2023",
    "01/31/2024","03/20/2024","05/01/2024","06/12/2024","07/31/2024","09/18/2024","11/07/2024","12/18/2024",
    "01/29/2025","03/19/2025","05/07/2025","06/18/2025","07/30/2025"
]
fomc_dates = pd.to_datetime(statement_dates_str, format="%m/%d/%Y", errors="coerce")
fomc_dates = fomc_dates.dropna()

# =========================
# 4) 핵심 함수: 한 fomc_date에 대해 prev/next week + prev/next 4주 평균 계산
# =========================
def compute_windows(df_in: pd.DataFrame, fomc_date: pd.Timestamp) -> dict:
    # (A) prev_week: fomc_date 이전(또는 당일) 가장 가까운 금요일
    prev_mask = df_in["Friday_of_Week"] <= fomc_date
    prev_week_date = df_in.loc[prev_mask, "Friday_of_Week"].max() if prev_mask.any() else pd.NaT

    # (B) next_week: fomc_date 이후 가장 가까운 금요일
    next_mask = df_in["Friday_of_Week"] > fomc_date
    next_week_date = df_in.loc[next_mask, "Friday_of_Week"].min() if next_mask.any() else pd.NaT

    # (C) prev_month(=4주): fomc_date 이전(또는 당일) 주간 데이터 중 최근 4개 평균
    prev4 = df_in.loc[prev_mask].tail(4)  # 이미 날짜 오름차순이므로 tail(4)=최근 4개
    prev_month_nfci = prev4["NFCI"].mean() if len(prev4) > 0 else pd.NA
    prev_month_anfci = prev4["ANFCI"].mean() if len(prev4) > 0 else pd.NA

    # (D) next_month(=4주): fomc_date 이후 주간 데이터 중 다음 4개 평균
    next4 = df_in.loc[next_mask].head(4)
    next_month_nfci = next4["NFCI"].mean() if len(next4) > 0 else pd.NA
    next_month_anfci = next4["ANFCI"].mean() if len(next4) > 0 else pd.NA

    # (E) prev_week / next_week 값
    if pd.notna(prev_week_date):
        prev_row = df_in.loc[df_in["Friday_of_Week"] == prev_week_date].iloc[-1]
        prev_week_nfci = prev_row["NFCI"]
        prev_week_anfci = prev_row["ANFCI"]
    else:
        prev_week_nfci, prev_week_anfci = pd.NA, pd.NA

    if pd.notna(next_week_date):
        next_row = df_in.loc[df_in["Friday_of_Week"] == next_week_date].iloc[0]
        next_week_nfci = next_row["NFCI"]
        next_week_anfci = next_row["ANFCI"]
    else:
        next_week_nfci, next_week_anfci = pd.NA, pd.NA

    return {
        "fomc_date": fomc_date.date(),
        "prev_week_date": prev_week_date.date() if pd.notna(prev_week_date) else pd.NA,
        "next_week_date": next_week_date.date() if pd.notna(next_week_date) else pd.NA,

        "prev_week_nfci": prev_week_nfci,
        "prev_week_anfci": prev_week_anfci,
        "next_week_nfci": next_week_nfci,
        "next_week_anfci": next_week_anfci,

        "prev_month_nfci": prev_month_nfci,
        "prev_month_anfci": prev_month_anfci,
        "next_month_nfci": next_month_nfci,
        "next_month_anfci": next_month_anfci,

        "count_prev4": len(prev4),
        "count_next4": len(next4),
    }

# =========================
# 5) FOMC 전체에 대해 계산
# =========================
rows = []
for d in fomc_dates:
    rows.append(compute_windows(df, d))

out = pd.DataFrame(rows)

# =========================
# 6) 엑셀로 저장
# =========================
output_path = input_path.with_name("fomc_nfci_anfci_prev_next_week_month.xlsx")
out.to_excel(output_path, index=False)

print("✅ 저장 완료:", output_path)
print(out.head(10))
