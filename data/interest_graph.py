import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# 파일 경로
corr_path = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\fomc_corr.csv")
interest_path = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\interest_rate.csv")
out_path = corr_path.with_name("fomc_corr_interest.png")


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
    if "FOMC_date" not in df.columns or "corr" not in df.columns:
        raise ValueError(f"Unexpected columns in {path}. Found: {df.columns.tolist()}")
    df["date"] = df["FOMC_date"].map(parse_fomc_yyyymm)
    # 같은 달이 중복되면 평균으로 집계
    df = df.groupby("date", as_index=False, sort=True)["corr"].mean()
    df = df.sort_values("date")
    return df


def load_interest_df(path: Path) -> pd.DataFrame:
    # 쉼표/공백/탭 구분 어떤 것이든 자동 인식
    df = pd.read_csv(path, engine="python", sep=r"[, \t]+")
    # 예상 컬럼: ['fomc-date', 'interest'] 혹은 대소문자 차이
    # 컬럼 소문자화 및 공백 제거
    df.columns = [c.strip().lower() for c in df.columns]
    if "fomc-date" not in df.columns or "interest" not in df.columns:
        raise ValueError(f"Unexpected columns in {path}. Found: {df.columns.tolist()}")
    df = df.rename(columns={"fomc-date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def plot_corr_interest(df_corr: pd.DataFrame, df_int: pd.DataFrame, out_path: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # corr (좌측 y축: 0~1)
    l1 = ax1.plot(
        df_corr["date"], df_corr["corr"], color="tab:blue", linewidth=2, label="corr (0–1)"
    )
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("corr (0–1)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # interest (우측 y축: 0~6)
    ax2 = ax1.twinx()
    l2 = ax2.plot(
        df_int["date"], df_int["interest"], color="tab:red", linewidth=2, label="interest (0–6)"
    )
    ax2.set_ylim(0, 6)
    ax2.set_ylabel("interest (0–6)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # X축: 월 단위 포맷터
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(3, 6, 9, 12)))  # 분기 표시
    fig.autofmt_xdate()

    # 범례 결합
    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left")

    ax1.set_title("FOMC corr vs Interest over time")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")


def main():
    df_corr = load_corr_df(corr_path)
    df_int = load_interest_df(interest_path)
    plot_corr_interest(df_corr, df_int, out_path)


if __name__ == "__main__":
    main()

    import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# 파일 경로
corr_path = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\fomc_corr.csv")
interest_path = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\interest_rate.csv")
out_path = corr_path.with_name("fomc_corr_interest.png")


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
    if "FOMC_date" not in df.columns or "corr" not in df.columns:
        raise ValueError(f"Unexpected columns in {path}. Found: {df.columns.tolist()}")
    df["date"] = df["FOMC_date"].map(parse_fomc_yyyymm)
    # 같은 달이 중복되면 평균으로 집계
    df = df.groupby("date", as_index=False, sort=True)["corr"].mean()
    df = df.sort_values("date")
    return df


def load_interest_df(path: Path) -> pd.DataFrame:
    # 쉼표/공백/탭 구분 어떤 것이든 자동 인식
    df = pd.read_csv(path, engine="python", sep=r"[, \t]+")
    # 예상 컬럼: ['fomc-date', 'interest'] 혹은 대소문자 차이
    # 컬럼 소문자화 및 공백 제거
    df.columns = [c.strip().lower() for c in df.columns]
    if "fomc-date" not in df.columns or "interest" not in df.columns:
        raise ValueError(f"Unexpected columns in {path}. Found: {df.columns.tolist()}")
    df = df.rename(columns={"fomc-date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def plot_corr_interest(df_corr: pd.DataFrame, df_int: pd.DataFrame, out_path: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # corr (좌측 y축: 0~1)
    l1 = ax1.plot(
        df_corr["date"], df_corr["corr"], color="tab:blue", linewidth=2, label="corr (0–1)"
    )
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("corr (0–1)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # interest (우측 y축: 0~6)
    ax2 = ax1.twinx()
    l2 = ax2.plot(
        df_int["date"], df_int["interest"], color="tab:red", linewidth=2, label="interest (0–6)"
    )
    ax2.set_ylim(0, 6)
    ax2.set_ylabel("interest (0–6)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # X축: 월 단위 포맷터
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(3, 6, 9, 12)))  # 분기 표시
    fig.autofmt_xdate()

    # 범례 결합
    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left")

    ax1.set_title("FOMC corr vs Interest over time")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")


def main():
    df_corr = load_corr_df(corr_path)
    df_int = load_interest_df(interest_path)
    plot_corr_interest(df_corr, df_int, out_path)


if __name__ == "__main__":
    main()