#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fomc_token_simscore_results_with_std_cv.csv에서
round_adj_token, std, cv 컬럼을 읽어 두 개의 그래프를 그립니다.

그래프 1 (STD):
- 산점도: x=round_adj_token, y=std (개별 점)
- 선 4개: round_adj_token 기준으로 std의 median, average(mean), highest(max), lowest(min)

그래프 2 (CV):
- 산점도: x=round_adj_token, y=cv (개별 점)
- 선 4개: round_adj_token 기준으로 cv의 median, average(mean), highest(max), lowest(min)

결과물:
- 화면에 표시
- 같은 폴더에 plot_std_by_token.png, plot_cv_by_token.png 저장
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys


DEFAULT_CSV = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\fomc_token_simscore_results_with_std_cv.csv"


def try_read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 마지막 시도: 기본 인코딩
    return pd.read_csv(path)


def coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.copy()
    # 문자열 내 % 등 제거
    if s.dtype == object:
        s = s.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼명 유연 매칭
    cols = {c.lower(): c for c in df.columns}
    token_col = cols.get("round_adj_token", None)
    std_col = cols.get("std", None)
    cv_col = cols.get("cv", None)

    if token_col is None or std_col is None or cv_col is None:
        raise ValueError("CSV에서 필요한 컬럼(round_adj_token, std, cv)을 찾을 수 없습니다.")

    out = pd.DataFrame({
        "round_adj_token": coerce_numeric(df[token_col]),
        "std": coerce_numeric(df[std_col]),
        "cv": coerce_numeric(df[cv_col]),
    })
    # 유효 값만
    out = out.dropna(subset=["round_adj_token"])
    # 토큰을 정수로 정규화 (필요 시 반올림)
    out["round_adj_token"] = out["round_adj_token"].round().astype(int)
    return out


def compute_group_stats(d: pd.DataFrame, y_col: str) -> pd.DataFrame:
    g = (
        d.groupby("round_adj_token")[y_col]
        #.agg(median="median", mean="mean", highest="max", lowest="min")
        .agg(median="median", mean="mean")
        .sort_index()
    )
    return g.reset_index()


def style_matplotlib():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.ylim(0,1.5)
    mpl.rcParams["axes.titlesize"] = 13
    mpl.rcParams["axes.labelsize"] = 11
    mpl.rcParams["legend.fontsize"] = 10
    mpl.rcParams["figure.dpi"] = 120
    mpl.rcParams["savefig.dpi"] = 150
    mpl.rcParams["font.family"] = mpl.rcParams.get("font.family", ["DejaVu Sans"])


def plot_one_metric(df: pd.DataFrame, metric: str, title: str, save_path: Path):
    # 산점도 데이터
    x = df["round_adj_token"].values
    y = df[metric].values

    # 집계 선 4개
    stats = compute_group_stats(df.dropna(subset=[metric]), metric)
    tokens = stats["round_adj_token"].values

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(x, y, s=22, alpha=0.35, color="#555", label=f"{metric} points")

    # 선 스타일
    linespecs = [
        ("median", "#1f77b4", "-"),
        ("mean",   "#ff7f0e", "--"),
        #("highest","#d62728", "-."),
        #("lowest", "#2ca02c", ":"),
    ]

    for col, color, ls in linespecs:
        ax.plot(tokens, stats[col].values, label=col, color=color, linestyle=ls, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("round_adj_token")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 0.4)
    ax.legend(ncol=2, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5)

    # x축 눈금이 많으면 간격 조정
    uniq_tokens = np.unique(tokens)
    if len(uniq_tokens) > 15:
        # 토큰 간격 추정 후 적정 개수로 줄이기
        step = max(1, len(uniq_tokens) // 12)
        xticks = uniq_tokens[::step]
        ax.set_xticks(xticks)

    fig.tight_layout()
    fig.savefig(save_path)
    print(f"저장: {save_path}")
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description="round_adj_token에 따른 std, cv 통계 그래프 생성")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="입력 CSV 경로 (기본: 시스템 경로)")
    parser.add_argument("--outdir", default="", help="이미지 저장 폴더 (기본: 입력 CSV와 동일 폴더)")
    parser.add_argument("--show", action="store_true", help="그래프를 화면에 표시")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"입력 CSV가 존재하지 않습니다: {csv_path}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    style_matplotlib()

    df_raw = try_read_csv(csv_path)
    df = prepare_data(df_raw)

    # STD 그래프
    std_path = outdir / "plot_std_by_token.png"
    plot_one_metric(df, "std", "STD by round_adj_token (points + median/mean/highest/lowest)", std_path)

    # CV 그래프
    cv_path = outdir / "plot_cv_by_token.png"
    plot_one_metric(df, "cv", "CV by round_adj_token (points + median/mean/highest/lowest)", cv_path)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()