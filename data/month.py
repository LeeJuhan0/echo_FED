import pandas as pd
from pathlib import Path

# =========================
# 1) 입력 파일 경로
# =========================
input_path = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\nfci-data-series.excel.xlsx")

# =========================
# 2) 2번째 시트 읽기 (0-based라 sheet_name=1)
# =========================
df = pd.read_excel(input_path, sheet_name=1)

# 컬럼명 공백/탭 정리
df.columns = [str(c).strip() for c in df.columns]

# 필수 컬럼 체크
required = ["Friday_of_Week", "NFCI", "ANFCI"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"필수 컬럼이 없습니다: {missing}\n현재 컬럼: {list(df.columns)}")

# 날짜 파싱 (예: 01/08/1971)
df["Friday_of_Week"] = pd.to_datetime(df["Friday_of_Week"], errors="coerce")
df = df.dropna(subset=["Friday_of_Week"])

# =========================
# 3) 월별 '4개(주간) 평균' 만들기
#    - 각 월에서 날짜순으로 앞 4개 평균 (4개 미만이면 있는 만큼)
# =========================
df["month"] = df["Friday_of_Week"].dt.to_period("M")

def first4_mean(g: pd.DataFrame) -> pd.Series:
    g2 = g.sort_values("Friday_of_Week").head(4)
    return pd.Series({
        "nfci": g2["NFCI"].mean(),
        "anfci": g2["ANFCI"].mean(),
        "count_used": len(g2),
    })

monthly = df.groupby("month", as_index=True).apply(first4_mean)

# =========================
# 4) FOMC 월(YYYY-MM) 리스트
# =========================
fomc_dates = [
    "2017-02","2017-03","2017-05","2017-06","2017-07","2017-09","2017-11","2017-12",
    "2018-01","2018-03","2018-05","2018-06","2018-08","2018-09","2018-11","2018-12",
    "2019-01","2019-03","2019-05","2019-06","2019-07","2019-09","2019-10","2019-12",
    "2020-01","2020-03","2020-04","2020-06","2020-07","2020-09","2020-11","2020-12",
    "2021-01","2021-03","2021-04","2021-06","2021-07","2021-09","2021-11","2021-12",
    "2022-01","2022-03","2022-05","2022-06","2022-07","2022-09","2022-11","2022-12",
    "2023-02","2023-03","2023-05","2023-06","2023-07","2023-09","2023-11","2023-12",
    "2024-01","2024-03","2024-05","2024-06","2024-07","2024-09","2024-11","2024-12",
    "2025-01","2025-03","2025-05","2025-06","2025-07",
]
fomc_periods = pd.PeriodIndex(fomc_dates, freq="M")

# =========================
# 5) FOMC 월과 (0~5개월 후) 월평균 매핑
# =========================
rows = []
for p in fomc_periods:
    row = {"date": p.strftime("%Y-%m")}
    for k in range(0, 7):  # 0~5
        pk = p + k
        nf = monthly.loc[pk, "nfci"] if pk in monthly.index else pd.NA
        af = monthly.loc[pk, "anfci"] if pk in monthly.index else pd.NA

        if k == 0:
            row["nfci"] = nf
            row["anfci"] = af
        else:
            row[f"Next{k}monthnfci"] = nf
            row[f"Next{k}monthanfci"] = af
    rows.append(row)

out = pd.DataFrame(rows)

# =========================
# 6) 엑셀로 저장
# =========================
output_path = input_path.with_name("fomc_nfci_anfci_monthly_0to6.xlsx")
out.to_excel(output_path, index=False)

print("✅ 저장 완료:", output_path)
print(out.head(10))
