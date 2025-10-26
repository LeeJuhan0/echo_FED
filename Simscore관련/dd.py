import pandas as pd
import math
import os

# ===== 설정 =====
EXCEL_PATH = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simscore관련\openai_paragraph_vs_statements_only_economic.xlsx"
SHEET_NAME = 0          # 첫 번째 시트
HAS_HEADER = False      # 첫 행이 헤더면 True로 변경
OVERWRITE = True        # True면 원본 덮어쓰기, False면 새 파일에 저장
# =================

def is_nan(x):
    try:
        return x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x)
    except Exception:
        return False

def add_csv_suffix(val):
    if is_nan(val):
        return val
    s = str(val).strip()
    if not s:
        return s
    if s.lower().endswith(".csv"):
        return s
    return s + ".csv"

def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=0 if HAS_HEADER else None)

    # A열(첫 번째 열)에 .csv 접미사 추가
    df.iloc[:, 0] = df.iloc[:, 0].apply(add_csv_suffix)

    # 저장 경로 결정
    if OVERWRITE:
        out_path = EXCEL_PATH
    else:
        root, ext = os.path.splitext(EXCEL_PATH)
        out_path = f"{root}_with_csv_suffix{ext}"

    # 저장
    df.to_excel(out_path, index=False)
    print(f"저장 완료: {out_path}")

if __name__ == "__main__":
    main()