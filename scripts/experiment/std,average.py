from pathlib import Path
import pandas as pd
import numpy as np
import math

# 설정
DIR = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\bundle")
COLUMN_LETTER = "B"     # 사용할 열 (엑셀 기준)
START_ROW = 2           # 시작 행(엑셀 기준, 포함)
END_ROW = 31            # 끝 행(엑셀 기준, 포함)
OUTPUT_XLSX = DIR / "정리_표준편차2.xlsx"

# 표준편차: 샘플 기준(Excel STDEV.S). 모집단으로 원하시면 ddof=0으로 변경.
STDEV_DDOF = 1

def col_letter_to_index(letter: str) -> int:
    letter = letter.strip().upper()
    # 단일 문자만 처리 (B -> 1). 필요시 확장 가능.
    return ord(letter) - ord('A')

def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    # 한글 Windows 환경을 고려해 인코딩 fallback
    encodings = ["utf-8-sig", "cp949", "utf-8"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def main():
    csv_files = sorted(DIR.glob("*.csv"))
    if not csv_files:
        print(f"[정보] CSV 파일을 찾지 못했습니다: {DIR}")
        return

    col_idx = col_letter_to_index(COLUMN_LETTER)
    results = {}  # filename -> {"표준편차": val, "표준편차/평균": val}

    for csv_path in csv_files:
        try:
            df = read_csv_with_fallback(csv_path)
        except Exception as e:
            print(f"[경고] 파일 읽기 실패: {csv_path.name} ({e})")
            continue

        # 열 개수 확인
        if df.shape[1] <= col_idx:
            print(f"[경고] {csv_path.name}: 열 {COLUMN_LETTER}가 없습니다. 건너뜀.")
            continue

        # 행 슬라이싱: 엑셀 기준 2~31행 -> pandas iloc[1:31]
        series = df.iloc[:, col_idx]
        series_slice = series.iloc[START_ROW - 1: END_ROW]

        # 숫자 변환(천단위 쉼표 제거 후 변환)
        series_clean = pd.to_numeric(
            series_slice.astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        ).dropna()

        if series_clean.empty:
            print(f"[경고] {csv_path.name}: 지정 구간(B{START_ROW}:B{END_ROW})에 유효한 숫자 데이터가 없습니다.")
            std_val = np.nan
            mean_val = np.nan
        else:
            mean_val = float(series_clean.mean())
            std_val = float(series_clean.std(ddof=STDEV_DDOF))

        ratio = (std_val / abs(mean_val)) if (mean_val is not None and mean_val != 0) else np.nan

        results[csv_path.name] = {
            "표준편차": std_val,
            "표준편차/평균": ratio,
            "평균" : mean_val,
        }

    if not results:
        print("[정보] 집계할 결과가 없습니다.")
        return

    # 결과를 엑셀 표 형태로 정리: 행 = [표준편차, 표준편차/평균], 열 = 파일이름
    out_df = pd.DataFrame(results)
    # 행 순서 고정
    out_df = out_df.reindex(index=["표준편차", "표준편차/평균","평균"])

    # 엑셀로 저장
    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="Summary")

    print(f"[완료] 엑셀 파일 생성: {OUTPUT_XLSX}")
    print(f"[참고] 사용 설정: 열={COLUMN_LETTER}, 행={START_ROW}~{END_ROW}, 표준편차(ddof)={STDEV_DDOF}")

if __name__ == "__main__":
    main()