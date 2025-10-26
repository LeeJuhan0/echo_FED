#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV(fomc_token_simscore_results_with_std_cv.csv)에서
round_adj_token, std, gid_index 컬럼을 읽어 다음과 같이 그래프를 표시합니다.

- 그래프 총 개수 = (조건 만족 gid_index 개수) + 1
  1) 집계 그래프 1개: x=round_adj_token, y=std
     - 산점도: 모든 표본 점(회색)
     - 선 3개: round_adj_token 기준 median, mean, lowest(min)
  2) gid_index별 개별 그래프 N개:
     - '데이터 행 수'가 min-n(기본 3) 이상인 gid_index만 생성
     - 산점도: 해당 gid_index의 표본 점(고유 색)
     - 연결선: 같은 token에서 평균 낸 std 값을 token 오름차순으로 연결(같은 색)

주의: 파일로 저장하지 않고, 창으로만 표시합니다.
"""

from pathlib import Path
import argparse
from typing import Optional, Dict, Iterable
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
    return pd.read_csv(path)


def coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype == object:
        s = s.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    token_col = find_col(df, ["round_adj_token", "token", "round_token", "adj_token"])
    std_col = find_col(df, ["std"])
    gid_index_col = find_col(df, ["gid_index"])  # 반드시 gid_index 사용

    if token_col is None or std_col is None:
        raise ValueError("CSV에서 필요한 컬럼(round_adj_token, std)을 찾을 수 없습니다.")
    if gid_index_col is None:
        raise ValueError("CSV에서 gid_index 컬럼을 찾을 수 없습니다. 스크립트는 gid_index를 반드시 필요로 합니다.")

    out = pd.DataFrame({
        "round_adj_token": coerce_numeric(df[token_col]),
        "std": coerce_numeric(df[std_col]),
        "gid_index": coerce_numeric(df[gid_index_col]),
    })

    # 필수 값 정리
    out = out.dropna(subset=["round_adj_token", "gid_index", "std"]).copy()
    out["round_adj_token"] = out["round_adj_token"].round().astype(int)
    out["gid_index"] = out["gid_index"].round().astype(int)

    return out


def compute_group_stats_std(d: pd.DataFrame) -> pd.DataFrame:
    """
    token별 std 통계를 계산: median, mean, lowest(=min)
    """
    g = (
        d.groupby("round_adj_token")["std"]
        .agg(median="median", mean="mean", lowest="min")
        .sort_index()
    )
    return g.reset_index()


def style_matplotlib():
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams["axes.titlesize"] = 13
    mpl.rcParams["axes.labelsize"] = 11
    mpl.rcParams["legend.fontsize"] = 10
    mpl.rcParams["figure.dpi"] = 120
    mpl.rcParams["font.family"] = mpl.rcParams.get("font.family", ["DejaVu Sans"])


def build_gid_index_color_map(gids: np.ndarray) -> Dict[int, tuple]:
    """gid_index별 일관된 색상 매핑을 생성"""
    uniq = [int(g) for g in pd.unique(gids) if pd.notna(g)]
    uniq = sorted(set(uniq))
    if not uniq:
        return {}
    # gid가 많아질 수 있으므로 색상 수 자동 스케일링
    cmap = mpl.cm.get_cmap("tab20", max(20, len(uniq)))
    return {gid: cmap(i % cmap.N) for i, gid in enumerate(uniq)}


def throttle_xticks(ax, tokens: np.ndarray, max_ticks: int = 12):
    uniq_tokens = np.unique(tokens)
    if len(uniq_tokens) > max_ticks:
        step = int(np.ceil(len(uniq_tokens) / max_ticks))
        ax.set_xticks(uniq_tokens[::step])


def plot_aggregate_std(df: pd.DataFrame):
    """
    모든 표본을 회색 점으로 표시하고,
    token별 median, mean, lowest(min) 3개 선을 그리는 집계 그래프
    """
    stats = compute_group_stats_std(df)
    tokens = stats["round_adj_token"].values

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # 모든 표본 산점도
    ax.scatter(df["round_adj_token"].values, df["std"].values, s=20, alpha=0.35, color="#666666", label="points")

    # 집계선 3개
    linespecs = [
        ("median", "#1f77b4", "-"),
        ("mean",   "#ff7f0e", "--"),
        ("lowest", "#2ca02c", ":"),
    ]
    for col, color, ls in linespecs:
        ax.plot(tokens, stats[col].values, label=col, color=color, linestyle=ls, linewidth=2.2, zorder=3)

    ax.set_title("Aggregate STD by round_adj_token (points + median/mean/lowest)")
    ax.set_xlabel("round_adj_token")
    ax.set_ylabel("std")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, ncol=3)

    throttle_xticks(ax, tokens)
    fig.tight_layout()


def plot_per_gid_std(df: pd.DataFrame, gid_to_color: Dict[int, tuple], min_points: int = 3):
    """
    gid_index별 개별 그래프 생성:
    - 산점도: 해당 gid_index 점
    - 연결선: token별 std 평균을 token 순으로 연결
    - 단, 각 gid_index의 '데이터 행 수'가 min_points 이상일 때만 그래프 생성
    """
    counts = df.groupby("gid_index").size()
    eligible_gids = sorted([gid for gid, n in counts.items() if n >= min_points])

    if not eligible_gids:
        print(f"조건을 만족하는 gid_index가 없습니다. (min_points={min_points})")
        return

    print(f"개별 그래프 생성 대상 gid_index: {eligible_gids} (기준: 행 수 >= {min_points})")

    for gid_idx in eligible_gids:
        sub = df[df["gid_index"] == gid_idx].copy()
        color = gid_to_color.get(int(gid_idx), "#1f77b4")

        # token별 평균으로 연결선용 데이터 준비
        agg = (
            sub.groupby("round_adj_token", as_index=False)["std"]
            .mean()
            .sort_values("round_adj_token")
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(
            sub["round_adj_token"].values, sub["std"].values,
            s=26, alpha=0.6, color=color, label=f"gid_index {gid_idx} points", zorder=3
        )

        if len(agg) >= 2:
            ax.plot(
                agg["round_adj_token"].values, agg["std"].values,
                color=color, linewidth=1.6, alpha=0.9, label="token-mean line", zorder=2
            )

        ax.set_title(f"GID_INDEX {gid_idx} - STD by round_adj_token (n={len(sub)})")
        ax.set_xlabel("round_adj_token")
        ax.set_ylabel("std")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, ncol=2)

        throttle_xticks(ax, agg["round_adj_token"].values if len(agg) else sub["round_adj_token"].values)
        fig.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="STD 그래프: 집계(1개) + gid_index별 개별 그래프(N개, 행 수 기준 필터) 표시")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="입력 CSV 경로")
    parser.add_argument("--min-n", type=int, default=3, help="gid_index별 개별 그래프를 생성할 최소 데이터 행 수 (기본: 3)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"입력 CSV가 존재하지 않습니다: {csv_path}", file=sys.stderr)
        sys.exit(1)

    min_points = max(1, int(args.min_n))

    style_matplotlib()

    df_raw = try_read_csv(csv_path)
    df = prepare_data(df_raw)

    # 색상 매핑 생성(개별 그래프에 사용)
    gid_to_color = build_gid_index_color_map(df["gid_index"].values)

    # 1) 집계 그래프(1개)
    plot_aggregate_std(df)

    # 2) gid_index별 개별 그래프(N개) — 데이터가 min_points 이상인 gid만 생성
    plot_per_gid_std(df, gid_to_color, min_points=min_points)

    # 화면 표시
    plt.show()


if __name__ == "__main__":
    main()