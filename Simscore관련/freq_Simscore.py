import pandas as pd
import os

# 4개의 폴더 경로 설정
folder_paths = [
    r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Theory\alldata"
]

# 키워드 설정
keywords = [
    # 경제 및 금융 관련 단어
    "economy", "market", "growth", "inflation", "employment", "investment",
    "spending", "activity", "conditions", "stability", "demand", "supply",
    "prices", "labor", "unemployment", "output", "trade", "income", "production",
    "jobs", "wages", "revenue", "consumption", "profit", "exchange", "balance",
    "budget", "cost", "savings", "capital", "credit", "debt", "wealth",
    "equity", "assets", "liabilities", "fund", "cash", "liquidity",
    "interest", "rate", "returns", "loan", "borrowing", "lending",
    "spreads", "yield", "bonds", "stocks", "securities", "mortgages",
    "reserves", "currency", "valuation", "risk", "uncertainty", "volatility",
    "diversity", "dividends", "losses", "gains", "profitability",

    # 정책 및 FOMC 관련 단어
    "policy", "committee", "federal", "reserve", "decision", "action",
    "mandate", "objective", "targets", "range", "guidance", "adjustments",
    "framework", "indicators", "monitoring", "assessment", "evaluation",
    "measures", "tools", "implementation", "approach", "strategy", "response",
    "oversight", "regulation", "framework", "review", "plan", "projection",
    "forecast", "expectations", "goals", "realization", "support",

    # 글로벌 및 경제적 요인
    "global", "international", "domestic", "regional", "sector", "industry",
    "households", "businesses", "firms", "corporations", "partnerships",
    "entities", "investors", "consumers", "producers", "suppliers", "retailers",
    "distributors", "exporters", "importers", "banks", "institutions",
    "markets", "systems", "economies", "resources", "labor force",
    "population", "trends", "patterns", "cycles", "shocks", "developments",
]
# 키워드를 모두 소문자로 변환
keywords = [keyword.lower() for keyword in keywords]

# 결과를 저장할 리스트
file_keyword_sums = []

# 각 폴더를 순회하면서 파일 처리
for folder_path in folder_paths:
    folder_name = os.path.basename(folder_path)  # 폴더 이름 추출
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            try:
                # 에러 발생 시 문제 있는 줄 건너뛰기
                data = pd.read_csv(file_path, on_bad_lines='skip')
                # 데이터 처리 로직 (키워드 빈도 계산)
                text_data = data.astype(str).apply(lambda x: ' '.join(x), axis=1).str.cat(sep=' ').lower()
                total_count = sum(text_data.count(keyword) for keyword in keywords)
                # 폴더 이름과 파일 이름 결합
                file_keyword_sums.append((f"{folder_name}_{file_name}", total_count))
            except Exception as e:
                print(f"Error reading {file_name} in folder {folder_name}: {e}")

# 키워드 빈도 합계 기준으로 정렬
sorted_files = sorted(file_keyword_sums, key=lambda x: x[1], reverse=True)

# 정렬된 데이터를 DataFrame으로 변환
df = pd.DataFrame(sorted_files, columns=["File Name", "Keyword Frequency"])

# 엑셀 파일로 저장
output_excel_path = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\frequency.xlsx"
df.to_excel(output_excel_path, index=False)

# 출력
print(f"Sorted keyword frequencies have been saved to {output_excel_path}")
for file_name, total_count in sorted_files:
    print(f"{file_name}: {total_count}")