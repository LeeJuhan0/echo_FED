# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd

DEFAULT_PATH = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simscore관련\combined_book_paragraphs.xlsx"

def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        # 엑셀 첫 시트를 읽습니다. 필요 시 engine="openpyxl" 사용
        try:
            return pd.read_excel(path)
        except Exception:
            return pd.read_excel(path, engine="openpyxl")
    else:
        # CSV/TSV 자동 구분자 추론
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")

def main():
    ap = argparse.ArgumentParser(description="paragraph 열의 글자 수를 1000자 단위로 구간화하여 카운트")
    ap.add_argument("--path", type=str, default=DEFAULT_PATH, help="입력 파일 경로 (엑셀/CSV)")
    ap.add_argument("--sheet", type=str, default=None, help="엑셀 시트명(미지정 시 첫 시트)")
    args = ap.parse_args()

    # 파일 읽기
    ext = os.path.splitext(args.path)[1].lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        try:
            df = pd.read_excel(args.path, sheet_name=args.sheet if args.sheet else 0)
        except Exception:
            df = pd.read_excel(args.path, sheet_name=args.sheet if args.sheet else 0, engine="openpyxl")
    else:
        df = pd.read_csv(args.path, sep=None, engine="python", encoding="utf-8-sig")

    if "paragraph" not in df.columns:
        raise ValueError(f"입력 파일에 'paragraph' 열이 없습니다. 실제 컬럼: {list(df.columns)}")

    # 공백 정리 및 빈 값 제거 후 길이 계산
    s = df["paragraph"].dropna().astype(str).str.strip()
    s = s[s.str.len() > 0]
    lengths = s.str.len()

    if lengths.empty:
        print("paragraph 데이터가 비어 있습니다.")
        return

    # 1000자 단위 구간으로 카운트
    bin_idx = (lengths // 1000).astype(int)
    counts = bin_idx.value_counts().sort_index()

    # 결과 출력
    print("=== paragraph 길이 분포 (1000자 단위) ===")
    for k, c in counts.items():
        low = k * 1000
        high = k * 1000 + 999
        print(f"{low:>5}-{high:<5} : {c}")

    # 요약 통계
    print("\n=== 요약 통계 ===")
    print(f"총 문단 수       : {len(lengths)}")
    print(f"최소 길이        : {int(lengths.min())}")
    print(f"최대 길이        : {int(lengths.max())}")
    print(f"평균 길이        : {round(lengths.mean(), 2)}")
    print(f"표준편차         : {round(lengths.std(ddof=1), 2)}")

if __name__ == "__main__":
    main()