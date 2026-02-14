from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


CANONICAL_COLS = ["statment", "papers", "fsr", "sloos", "beigebook"]

# 원본(소스) 열명(소문자/공백제거 기준) → 캐노니컬 키 매핑
SOURCE_TO_CANONICAL: Dict[str, str] = {
    # statment
    "statment": "statment",
    "statement": "statment",  # 오타 보정
    # papers
    "papers": "papers",
    "paper": "papers",
    # fsr
    "fsr": "fsr",
    # sloos
    "sloos": "sloos",
    # beigebook
    "beigebook": "beigebook",
    "beige": "beigebook",
    "beige-book": "beigebook",
    "beige_book": "beigebook",
}

# 최종 출력 컬럼명 매핑
CANONICAL_TO_OUTPUT: Dict[str, str] = {
    "statment": "statment",
    "papers": "Theory",
    "fsr": "FSR",
    "sloos": "SLOOS",
    "beigebook": "Policy",
}


def read_table(path: Path) -> Optional[pd.DataFrame]:
    """
    path가 csv면 read_csv, xlsx면 read_excel로 읽고,
    열 이름은 그대로 두되 반환. 실패 시 None.
    """
    try:
        if path.suffix.lower() in [".xlsx", ".xls"]:
            # 엑셀 파일 읽기
            return pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            # CSV 인코딩 여러 개 시도
            encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
            last_err = None
            for enc in encodings:
                try:
                    return pd.read_csv(path, encoding=enc)
                except Exception as e:
                    last_err = e
            print(f"[WARN] Failed to read CSV {path} with tried encodings. Last error: {last_err}")
            return None
        else:
            return None
    except Exception as e:
        print(f"[WARN] Failed to read file {path}: {e}")
        return None


def normalize_and_select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    열 이름을 소문자/공백제거/특수문자 일부 제거하여 캐노니컬 열로 매핑하고,
    필요한 캐노니컬 열만 선택하여 반환.
    """
    # 원본 → 캐노니컬 이름 생성
    new_cols = {}
    for col in df.columns:
        key = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
        )
        # beige book 같이 붙이지 않은 변형도 처리
        key = key.replace("beigebook", "beigebook").replace("beigebook", "beigebook")
        # 매핑 시도
        if key in SOURCE_TO_CANONICAL:
            new_cols[col] = SOURCE_TO_CANONICAL[key]
        else:
            # 매핑이 없으면 무시
            pass

    # 필요한 컬럼만 리네임 및 선택
    candidate = df.rename(columns=new_cols)
    present = [c for c in CANONICAL_COLS if c in candidate.columns]
    if not present:
        return pd.DataFrame(columns=CANONICAL_COLS)

    # 숫자 변환
    for c in present:
        candidate[c] = pd.to_numeric(candidate[c], errors="coerce")

    return candidate[present]


def collect_monthly_median(month_dir: Path) -> Optional[Dict[str, float]]:
    """
    해당 월 폴더 내의 모든 csv/xlsx를 읽어 필요한 열을 수집하고,
    열별 중간값을 계산하여 dict 반환. 데이터 없으면 None.
    """
    files = list(month_dir.rglob("*.csv")) + list(month_dir.rglob("*.xlsx")) + list(month_dir.rglob("*.xls"))
    if not files:
        return None

    dfs: List[pd.DataFrame] = []
    for f in files:
        df = read_table(f)
        if df is None or df.empty:
            continue
        df_norm = normalize_and_select_columns(df)
        if not df_norm.empty:
            dfs.append(df_norm)

    if not dfs:
        return None

    all_df = pd.concat(dfs, ignore_index=True)
    # 모든 캐노니컬 컬럼 보장
    for c in CANONICAL_COLS:
        if c not in all_df.columns:
            all_df[c] = pd.Series(dtype="float64")

    med = all_df[CANONICAL_COLS].median(skipna=True).to_dict()

    # 최종 출력 컬럼 이름으로 변환
    out = {CANONICAL_TO_OUTPUT[k]: v for k, v in med.items()}
    return out


def is_year_dir(p: Path) -> bool:
    return p.is_dir() and p.name.isdigit() and len(p.name) == 4


def is_month_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    name = p.name
    if not name.isdigit():
        return False
    m = int(name)
    return 1 <= m <= 12


def main():
    # 기본 베이스 경로 (필요 시 커맨드라인 인자로 덮어쓰기)
    default_base = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\results\experiment\simulation"
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(default_base)

    if not base_dir.exists():
        print(f"[ERROR] Base directory not found: {base_dir}")
        sys.exit(1)

    rows = []

    for ydir in sorted([p for p in base_dir.iterdir() if is_year_dir(p)], key=lambda x: x.name):
        year = int(ydir.name)

        for mdir in sorted([p for p in ydir.iterdir() if is_month_dir(p)], key=lambda x: int(x.name)):
            month = int(mdir.name)
            date_str = f"{year:04d}-{month:02d}"

            monthly = collect_monthly_median(mdir)
            if monthly is None:
                print(f"[INFO] No valid data found in {mdir}")
                continue

            row = {"date": date_str}
            row.update(monthly)
            rows.append(row)
            print(f"[OK] {date_str} processed.")

    if not rows:
        print("[WARN] No rows collected. Nothing to write.")
        sys.exit(0)

    # DataFrame 생성 및 정렬, 컬럼 순서 지정
    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values("date")
    desired_cols = ["date", "statment", "Theory", "FSR", "SLOOS", "Policy"]
    for col in desired_cols:
        if col not in df_out.columns:
            df_out[col] = pd.NA
    df_out = df_out[desired_cols]

    out_path = base_dir / "Simulation_19+13_score_summary.xlsx"
    df_out.to_excel(out_path, index=False)
    print(f"[DONE] Saved summary to: {out_path}")


if __name__ == "__main__":
    main()