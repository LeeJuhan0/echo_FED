import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# 파일 경로
corr_path = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\fomc_corr.csv")
nfci_path = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\nfci_anfci_monthly.xlsx")
out_path = corr_path.with_name("fomc_corr_nfci.png")

# nfci/anfci를 0-1로 맞추기 위한 스케일링 사용 여부
scale_indices = True


def parse_fomc_yyyymm(label: str) -> pd.Timestamp:
    """
    '201702~' 같은 라벨을 '2017-02-01'로 파싱.
    비숫자 제거 후 YYYYMM으로 변환.
    """
    digits = re.sub(r"\D", "", str(label))  # 숫자만 남김
    return pd.to_datetime(digits, format="%Y%m")


def load_corr_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 예상 컬럼: ['corr', 'FOMC_date']
    cols = {c.strip(): c for c in df.columns}
    if "corr" not in cols or "FOMC_date" not in cols:
        raise ValueError(f"Unexpected columns in {path}. Found: {df.columns.tolist()}")
    df = df.rename(columns={cols["corr"]: "corr", cols["FOMC_date"]: "FOMC_date"})
    df["date"] = df["FOMC_date"].map(parse_fomc_yyyymm)
    # 같은 달이 중복되면 평균으로 집계
    df = df.groupby("date", as_index=False, sort=True)["corr"].mean()
    df = df.sort_values("date")
    return df


def load_nfci_df(path: Path) -> pd.DataFrame:
    # 첫 시트를 읽고 컬럼 소문자화
    df = pd.read_excel(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    # 예상 컬럼: date, nfci, anfci
    required = {"date", "nfci", "anfci"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Unexpected columns in {path}. Found: {df.columns.tolist()}")
    df = df[["date", "nfci", "anfci"]].copy()
    # 날짜 파싱: 'YYYY-MM' -> 해당 월의 1일
    df["date"] = pd.to_datetime(df["date"])
    # pandas 일부 버전 호환: 'MS' 전달 대신 기본값(월 시작) 사용
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()  # == .dt.start_time
    df = df.sort_values("date")
    return df


def min_max_scale(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    vmin = s.min(skipna=True)
    vmax = s.max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        # 상수열이거나 전부 결측이면 0.5로 반환
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - vmin) / (vmax - vmin)


def plot_corr_nfci(df_corr: pd.DataFrame, df_idx: pd.DataFrame, out_path: Path, scale_indices: bool = True):
    # corr 시프트 컬럼 (현재 날짜에 이전/다음 행의 corr 값을 정렬하여 비교)
    df_corr = df_corr.copy()
    df_corr["corr_prev"] = df_corr["corr"].shift(1)   # 이전 달 corr
    df_corr["corr_next"] = df_corr["corr"].shift(-1)  # 다음 달 corr

    # 스케일링 옵션
    if scale_indices:
        anfci_scaled = min_max_scale(df_idx["anfci"])
        y_label_anfci = "anfci (min–max 0–1)"
    else:
        anfci_scaled = df_idx["anfci"]
        y_label_anfci = "anfci (raw)"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    # corr (원본)
    l_corr = ax.plot(
        df_corr["date"], df_corr["corr"],
        color="tab:blue", linewidth=2, marker="o", markersize=3, label="corr (0–1)"
    )
    # prev corr (이전 행 값)
    l_corr_prev = ax.plot(
        df_corr["date"], df_corr["corr_prev"],
        color="tab:blue", linewidth=1.8, linestyle="--", alpha=0.8, label="prev corr (shift +1)"
    )
    # next corr (다음 행 값)
    l_corr_next = ax.plot(
        df_corr["date"], df_corr["corr_next"],
        color="tab:blue", linewidth=1.8, linestyle=":", alpha=0.8, label="next corr (shift -1)"
    )

    # anfci (그대로)
    l_anfci = ax.plot(
        df_idx["date"], anfci_scaled,
        color="tab:orange", linewidth=2, marker=None, label=y_label_anfci
    )

    # y축 0~1로 고정
    ax.set_ylim(0, 1)
    ax.set_ylabel("Scaled value (0–1)")

    # X축: 연도 기준 메이저, 분기 마이너
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(3, 6, 9, 12)))
    fig.autofmt_xdate()

    # 범례
    lines = l_corr + l_corr_prev + l_corr_next + l_anfci
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="best")

    ax.set_title("FOMC corr (prev/next) vs ANFCI (all in 0–1 scale)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")


def main():
    df_corr = load_corr_df(corr_path)
    df_idx = load_nfci_df(nfci_path)
    plot_corr_nfci(df_corr, df_idx, out_path, scale_indices=scale_indices)


if __name__ == "__main__":
    main()