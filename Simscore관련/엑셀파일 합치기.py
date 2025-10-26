import os
import glob
import re
import sys
from typing import List, Tuple, Optional
import pandas as pd


def natural_key(s: str):
    """
    자연 정렬 키 (file2 < file10 같은 정렬을 위해).
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def gather_csv_files(path: str, filename_contains: Optional[str] = None) -> List[str]:
    """
    디렉토리에서 .csv 파일 수집. 파일 경로가 들어오면:
    - .csv면 그대로 반환
    - 아니면 경고 후 빈 리스트
    filename_contains가 주어지면, 파일명에 해당 문자열이 포함된 파일만 수집.
    """
    files: List[str] = []

    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.csv"))
    elif os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            files = [path]
        else:
            print(f"[WARN] Unsupported file (not .csv): {path}", file=sys.stderr)
            return []
    else:
        print(f"[WARN] Path does not exist: {path}", file=sys.stderr)
        return []

    # 파일명 필터(선택)
    if filename_contains:
        files = [f for f in files if filename_contains in os.path.basename(f)]

    files = sorted(files, key=natural_key)
    return files


def try_read_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """
    다양한 인코딩을 시도하여 CSV 읽기.
    """
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    print(f"[WARN] Failed to read CSV with common encodings: {csv_path} ({last_err})", file=sys.stderr)
    return None


def read_paragraphs_from_csv(csv_path: str) -> pd.DataFrame:
    """
    단일 CSV에서 paragraph 열만 반환.
    - 'id'가 있으면 숫자 기준 오름차순 정렬.
    - paragraph를 문자열로 변환, 공백 트림, 빈 값 제거.
    """
    df = try_read_csv(csv_path)
    if df is None:
        return pd.DataFrame(columns=["paragraph"])

    if "paragraph" not in df.columns:
        print(f"[WARN] Missing 'paragraph' column: {csv_path}", file=sys.stderr)
        return pd.DataFrame(columns=["paragraph"])

    # id가 있으면 정렬 (숫자 변환 → 보조키로 원래 id 유지)
    if "id" in df.columns:
        id_numeric = pd.to_numeric(df["id"], errors="coerce")
        df = df.assign(_id_num=id_numeric)
        df = df.sort_values(by=["_id_num", "id"], na_position="last").drop(columns=["_id_num"])

    out = df[["paragraph"]].copy()
    out["paragraph"] = out["paragraph"].astype(str).str.strip()
    out = out[out["paragraph"].astype(bool)]
    out.reset_index(drop=True, inplace=True)

    print(f"[INFO] {os.path.basename(csv_path)} -> {len(out)} paragraphs", file=sys.stderr)
    return out


def build_book_dataframe(source_path: str, book_name: str, filename_contains: Optional[str] = None) -> pd.DataFrame:
    """
    source_path(폴더 또는 단일 CSV) 하의 CSV들을 모아:
    - book_name_index: f\"{book_name} 문단_{k}\"
    - paragraph
    를 생성. 같은 책 내에서 k는 1부터 시작해서 파일을 넘어 연속 증가.
    filename_contains가 주어지면 파일명에 해당 문자열이 포함된 CSV만 사용.
    """
    files = gather_csv_files(source_path, filename_contains=filename_contains)
    if not files:
        print(f"[WARN] No CSV files found for book '{book_name}' at: {source_path}", file=sys.stderr)
        return pd.DataFrame(columns=["book_name_index", "paragraph"])

    paragraphs: List[str] = []
    for fp in files:
        df = read_paragraphs_from_csv(fp)
        if df.empty:
            print(f"[INFO] No valid paragraphs in: {fp}", file=sys.stderr)
            continue
        paragraphs.extend(df["paragraph"].tolist())

    rows = []
    for i, para in enumerate(paragraphs, start=1):
        rows.append({
            "book_name_index": f"{book_name} 문단_{i}",
            "paragraph": para
        })

    book_df = pd.DataFrame(rows, columns=["book_name_index", "paragraph"])
    print(f"[INFO] Book '{book_name}' total paragraphs: {len(book_df)}", file=sys.stderr)
    return book_df


def combine_books(sources: List[Tuple[str, str, Optional[str]]]) -> pd.DataFrame:
    """
    sources: (source_path, book_name, filename_contains) 목록
    반환: book_name_index, paragraph 두 열을 갖는 병합 DataFrame
    """
    all_books = []
    for source_path, book_name, filename_contains in sources:
        df = build_book_dataframe(source_path, book_name, filename_contains=filename_contains)
        if not df.empty:
            all_books.append(df)

    if not all_books:
        return pd.DataFrame(columns=["book_name_index", "paragraph"])
    return pd.concat(all_books, ignore_index=True)


def main():
    # 1) Walsh 책 폴더
    folder1 = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simscore관련\책_문단"
    book1 = r"100219~[Carl_E._Walsh]_Monetary_Theory_and_Policy,_Third_(b-ok.org)"

    # 2) Mankiw 책 폴더 (예시 경로: Simscore관련\책_문단_2)
    folder2 = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simscore관련\책_문단_2"
    book2 = r"N. Gregory Mankiw. Macroeconomics  9th edition"

    # 출력 엑셀 경로
    output_path = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\combined_book_paragraphs.xlsx"

    # 필요 시 특정 파일만 고르려면 filename_contains를 사용하세요(아래 None 대신 문자열).
    sources = [
        (folder1, book1, None),
        (folder2, book2, None),
    ]

    combined_df = combine_books(sources)

    if combined_df.empty:
        print("[WARN] No data combined. Please check your folders/files and filters.", file=sys.stderr)
        return

    # 열 순서 고정 및 저장
    combined_df = combined_df[["book_name_index", "paragraph"]]
    try:
        combined_df.to_excel(output_path, index=False)
        print(f"[DONE] Saved combined Excel to: {output_path}")
        print(f"[DONE] Total rows: {len(combined_df)}")
    except Exception as e:
        print(f"[ERROR] Failed to save Excel: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()