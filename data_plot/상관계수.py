import io
from textwrap import dedent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. 패널 데이터 (Statement, P&B, Policy) 70행 (2020-03x,y 포함)
panel_text = """
date Statement P&B Policy
2017-02 0.33 0.32 0.24
2017-03 0.4 0.4 0.35
2017-05 0.34 0.36 0.28
2017-06 0.4 0.35 0.33
2017-07 0.35 0.32 0.26
2017-09 0.38 0.36 0.25
2017-11 0.4 0.43 0.41
2017-12 0.58 0.55 0.58
2018-01 0.66 0.65 0.53
2018-03 0.4 0.52 0.58
2018-05 0.44 0.4 0.27
2018-06 0.75 0.7 0.64
2018-08 0.65 0.67 0.65
2018-09 0.79 0.72 0.82
2018-11 0.66 0.65 0.45
2018-12 0.7 0.65 0.55
2019-01 0.48 0.58 0.4
2019-03 0.15 0.18 0.03
2019-05 0.27 0.22 0.05
2019-06 0.2 0.12 0.04
2019-07 0.21 0.2 0.06
2019-09 0.23 0.2 0.06
2019-10 0.2 0.22 0.06
2019-12 0.3 0.25 0.14
2020-01 0.28 0.25 0.08
2020-03x -0.2 -0.1 -0.22
2020-03y -0.62 -0.4 -0.66
2020-04 -0.75 -0.45 -0.68
2020-06 -0.66 -0.5 -0.62
2020-07 -0.4 -0.22 -0.25
2020-09 -0.25 -0.1 -0.2
2020-11 -0.25 -0.1 -0.1
2020-12 -0.22 -0.1 -0.18
2021-01 -0.3 -0.15 -0.05
2021-03 0.1 0.2 0.27
2021-04 0.2 0.22 0.35
2021-06 0.4 0.435 0.44
2021-07 0.4 0.45 0.6
2021-09 0.3 0.25 0.2
2021-11 0.3 0.25 0.25
2021-12 0.32 0.27 0.27
2022-01 0.26 0.2 0.15
2022-03 0.1 0.04 0.03
2022-05 -0.15 -0.18 -0.1
2022-06 -0.15 -0.2 -0.15
2022-07 -0.3 -0.4 -0.55
2022-09 -0.25 -0.35 -0.48
2022-11 -0.2 -0.25 -0.45
2022-12 -0.2 -0.25 -0.5
2023-02 -0.1 -0.14 -0.12
2023-03 -0.14 -0.18 -0.35
2023-05 0.1 0.04 -0.2
2023-06 0.1 0.05 -0.12
2023-07 0.1 0.05 -0.02
2023-09 0.18 0.1 -0.12
2023-11 0.15 0.1 -0.15
2023-12 -0.1 -0.08 -0.25
2024-01 0.3 0.2 0
2024-03 0.32 0.29 0.1
2024-05 0.2 0.1 0.02
2024-06 0.3 0.27 0.1
2024-07 0.25 0.2 0.04
2024-09 0.3 0.4 0.2
2024-11 0.3 0.25 0.1
2024-12 0.3 0.32 0.2
2025-01 0.3 0.25 0.05
2025-03 0.22 0.25 0.12
2025-05 0.15 0.1 -0.08
2025-06 0.3 0.22 0.05
2025-07 -0.15 -0.08 -0.2
"""

panel_df = pd.read_csv(io.StringIO(dedent(panel_text)), sep=r"\s+")

# 2. current_NFCI / current_ANFCI (70개, 위에 제공 순서가 panel_df 순서와 동일하다고 가정)
current_NFCI = [
    -0.5073162889,-0.498900146,-0.5345577434,-0.576233883,-0.5890146154,-0.5762304314,-0.6114358547,-0.6172927693,
    -0.6003158534,-0.5065885905,-0.5590744135,-0.5569997549,-0.5809238498,-0.6014510741,-0.5278605915,-0.4294617337,
    -0.5372206097,-0.6078907629,-0.6158033914,-0.5754488867,-0.5733096643,-0.5140849666,-0.5654633677,-0.581100561,
    -0.5595858833,-0.6266003643,0.1905448161,0.06469800709,-0.3476490395,-0.4817770637,-0.5040296121,-0.5256422883,
    -0.6049729999,-0.6256723068,-0.6465029652,-0.6874533138,-0.6961187566,-0.6606308802,-0.6589078872,-0.6082420611,
    -0.5427375615,-0.5431556955,-0.375239836,-0.3136649387,-0.1799246593,-0.2130513498,-0.1158412697,-0.1395695187,
    -0.1923099235,-0.3143503557,-0.1533339362,-0.2003105134,-0.2196035597,-0.2822889859,-0.3269355275,-0.290176549,
    -0.351784513,-0.4062790467,-0.4393908337,-0.4149527158,-0.3993118734,-0.3635358444,-0.4250465649,-0.46181325,
    -0.4835530735,-0.5115423135,-0.428517535,-0.422630087,-0.4890766323,-0.5390830343
]
current_ANFCI = [
    -0.4112827601,-0.4268621027,-0.4757480598,-0.53028488,-0.5667895914,-0.5157316719,-0.6118339429,-0.6334498473,
    -0.5997084683,-0.4806290897,-0.5329844605,-0.5336845388,-0.5912868072,-0.6371260483,-0.5784764048,-0.4454581552,
    -0.5389394667,-0.5977404794,-0.6202166348,-0.5893885842,-0.5919840214,-0.5722800593,-0.6050618503,-0.6311501574,
    -0.5487490741,-0.6187178722,0.285396402,0.123103101,-0.4656220601,-0.5716914647,-0.7151405835,-0.543227901,
    -0.5360314471,-0.5524918845,-0.5856877755,-0.5863578918,-0.6147993152,-0.6123854994,-0.662488878,-0.6027439197,
    -0.5167997512,-0.5229107756,-0.2581064538,-0.1796698881,-0.04858961819,-0.2051518135,-0.1305429824,-0.1505394383,
    -0.2036622442,-0.309254874,-0.1126755174,-0.2320200782,-0.2370881141,-0.2844575478,-0.2815903872,-0.2959845922,
    -0.3799233218,-0.4339383999,-0.4846480295,-0.430704484,-0.3895348865,-0.3640806465,-0.4661617459,-0.5159157459,
    -0.5148350355,-0.5212361142,-0.4447566396,-0.4549990198,-0.4801508551,-0.5120188578
]

assert len(panel_df) == len(current_NFCI) == len(current_ANFCI), "길이 불일치: 순서/데이터 확인 필요"
panel_df['current_NFCI'] = current_NFCI
panel_df['current_ANFCI'] = current_ANFCI

# --- 추가: year 컬럼 및 마스크 생성 (필수) ---
# 안전하게 date 칼럼 확인/정리: 첫 칼럼이 date인지 확실치 않으면 강제로 이름 지정
if 'date' not in panel_df.columns:
    panel_df.rename(columns={panel_df.columns[0]: 'date'}, inplace=True)

# date의 앞 4문자를 연도로 사용
panel_df['year'] = panel_df['date'].astype(str).str[:4].astype(int)
early_mask = panel_df['year'] <= 2022   # 2017~2022
late_mask  = panel_df['year'] >= 2023   # 2023~2025

# 3. 6개 산점도 페어 정의
pairs = [
    ('Statement','current_NFCI'),
    ('P&B','current_NFCI'),
    ('Policy','current_NFCI'),
    ('Statement','current_ANFCI'),
    ('P&B','current_ANFCI'),
    ('Policy','current_ANFCI'),
]

# 색상 정의
early_color = '#1f77b4'   # 파랑
late_color  = '#d62728'   # 빨강
overall_color = '#2ca02c' # 초록

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

def safe_regression_and_plot(ax, x, y, color, label, lw=1.8, linestyle='-'):
    """x,y are 1d arrays or series. Fit linear reg if possible and plot. Return (r,p) or None."""
    x = np.asarray(x); y = np.asarray(y)
    # need at least 2 distinct x to fit; require length>=2 and non-constant x
    if len(x) < 2:
        return None
    if np.all(x == x[0]) or np.all(np.isclose(x, x[0])):
        # vertical degeneracy: cannot fit slope; plot horizontal mean line
        y_mean = np.nanmean(y)
        xs = np.linspace(x.min() - 0.01, x.max() + 0.01, 120)
        ax.plot(xs, np.full_like(xs, y_mean), color=color, lw=lw, linestyle=linestyle, label=label)
        try:
            r, p = stats.pearsonr(x, y)
        except Exception:
            r, p = np.nan, np.nan
        return r, p
    # normal fit
    try:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 130)
        ax.plot(xs, m*xs + b, color=color, lw=lw, linestyle=linestyle, label=label)
        r, p = stats.pearsonr(x, y)
        return r, p
    except Exception:
        return None

def add_panel(ax, df, xcol, ycol):
    # plot points by period
    ax.scatter(df.loc[early_mask, xcol], df.loc[early_mask, ycol],
               color=early_color, edgecolors='white', linewidth=0.6, alpha=0.85, label='2017-2022', zorder=3)
    ax.scatter(df.loc[late_mask, xcol], df.loc[late_mask, ycol],
               color=late_color, edgecolors='white', linewidth=0.6, alpha=0.85, label='2023-2025', zorder=3)

    # regression lines: early, late, all
    stats_early = None
    stats_late = None
    stats_all = None

    # early
    xe = df.loc[early_mask, xcol].to_numpy()
    ye = df.loc[early_mask, ycol].to_numpy()
    if len(xe) >= 2:
        stats_early = safe_regression_and_plot(ax, xe, ye, early_color, 'regression 17-22', lw=2.0, linestyle='-')

    # late
    xl = df.loc[late_mask, xcol].to_numpy()
    yl = df.loc[late_mask, ycol].to_numpy()
    if len(xl) >= 2:
        stats_late = safe_regression_and_plot(ax, xl, yl, late_color, 'regression 23-25', lw=2.0, linestyle='-')

    # all
    xa = df[xcol].to_numpy()
    ya = df[ycol].to_numpy()
    if len(xa) >= 2:
        stats_all = safe_regression_and_plot(ax, xa, ya, overall_color, 'regression 17-25', lw=2.4, linestyle='--')

    # annotation box with r/p values when available
    lines = []
    if stats_early is not None:
        lines.append(f"17-22 r={stats_early[0]:.2f} p={stats_early[1]:.1e}")
    if stats_late is not None:
        lines.append(f"23-25 r={stats_late[0]:.2f} p={stats_late[1]:.1e}")
    if stats_all is not None:
        lines.append(f"All r={stats_all[0]:.2f} p={stats_all[1]:.1e}")
    if lines:
        ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes,
                va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.75), zorder=10)

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    if xcol == 'Statement' :
        ax.set_title(f"{xcol} vs {ycol}")
    else :
        ax.set_title(f"add_{xcol} vs {ycol}")
    ax.grid(alpha=0.3)

# draw panels
for ax, (xcol, ycol) in zip(axes, pairs):
    add_panel(ax, panel_df, xcol, ycol)

# figure-level legend (unique labels)
handles, labels = axes[0].get_legend_handles_labels()
# keep unique by label
uniq = {}
for h, l in zip(handles, labels):
    if l not in uniq:
        uniq[l] = h
fig.legend(list(uniq.values()), list(uniq.keys()), loc='upper center', ncol=3, frameon=True, fontsize=9, bbox_to_anchor=(0.5, 1.02))

plt.suptitle("Scatter Plots (period-colored points) with 3 regression lines (17-22 blue, 23-25 red, All green)", fontsize=13, y=0.98)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

# 4. Spearman / Kendall 상관 요약 (변경 없음)
import pprint
summary = {}
for xcol,ycol in pairs:
    x = panel_df[xcol]; y = panel_df[ycol]
    pear = stats.pearsonr(x,y)
    spear = stats.spearmanr(x,y)
    kend = stats.kendalltau(x,y)
    summary[f"{xcol}~{ycol}"] = {
        'n': len(x),
        'pearson_r': pear.statistic, 'pearson_p': pear.pvalue,
        'spearman_rho': spear.statistic, 'spearman_p': spear.pvalue,
        'kendall_tau': kend.statistic, 'kendall_p': kend.pvalue
    }

print("\n=== Correlation Summary (Method B) ===")
pprint.pprint(summary)