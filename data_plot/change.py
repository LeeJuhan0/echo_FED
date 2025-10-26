import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import io
from textwrap import dedent

# ---------- 1. 원시 데이터 읽기 ----------
policy_text = """date	Statement	P&B	Policy
2017-02	0.33	0.32	0.24
2017-03	0.4	0.4	0.35
2017-05	0.34	0.36	0.28
2017-06	0.4	0.35	0.33
2017-07	0.35	0.32	0.26
2017-09	0.38	0.36	0.25
2017-11	0.4	0.43	0.41
2017-12	0.58	0.55	0.58
2018-01	0.66	0.65	0.53
2018-03	0.4	0.52	0.58
2018-05	0.44	0.4	0.27
2018-06	0.75	0.7	0.64
2018-08	0.65	0.67	0.65
2018-09	0.79	0.72	0.82
2018-11	0.66	0.65	0.45
2018-12	0.7	0.65	0.55
2019-01	0.48	0.58	0.4
2019-03	0.15	0.18	0.03
2019-05	0.27	0.22	0.05
2019-06	0.2	0.12	0.04
2019-07	0.21	0.2	0.06
2019-09	0.23	0.2	0.06
2019-10	0.2	0.22	0.06
2019-12	0.3	0.25	0.14
2020-01	0.28	0.25	0.08
2020-03x	-0.2	-0.1	-0.22
2020-03y	-0.62	-0.4	-0.66
2020-04	-0.75	-0.45	-0.68
2020-06	-0.66	-0.5	-0.62
2020-07	-0.4	-0.22	-0.25
2020-09	-0.25	-0.1	-0.2
2020-11	-0.25	-0.1	-0.1
2020-12	-0.22	-0.1	-0.18
2021-01	-0.3	-0.15	-0.05
2021-03	0.1	0.2	0.27
2021-04	0.2	0.22	0.35
2021-06	0.4	0.435	0.44
2021-07	0.4	0.45	0.6
2021-09	0.3	0.25	0.2
2021-11	0.3	0.25	0.25
2021-12	0.32	0.27	0.27
2022-01	0.26	0.2	0.15
2022-03	0.1	0.04	0.03
2022-05	-0.15	-0.18	-0.1
2022-06	-0.15	-0.2	-0.15
2022-07	-0.3	-0.4	-0.55
2022-09	-0.25	-0.35	-0.48
2022-11	-0.2	-0.25	-0.45
2022-12	-0.2	-0.25	-0.5
2023-02	-0.1	-0.14	-0.12
2023-03	-0.14	-0.18	-0.35
2023-05	0.1	0.04	-0.2
2023-06	0.1	0.05	-0.12
2023-07	0.1	0.05	-0.02
2023-09	0.18	0.1	-0.12
2023-11	0.15	0.1	-0.15
2023-12	-0.1	-0.08	-0.25
2024-01	0.3	0.2	0
2024-03	0.32	0.29	0.1
2024-05	0.2	0.1	0.02
2024-06	0.3	0.27	0.1
2024-07	0.25	0.2	0.04
2024-09	0.3	0.4	0.2
2024-11	0.3	0.25	0.1
2024-12	0.3	0.32	0.2
2025-01	0.3	0.25	0.05
2025-03	0.22	0.25	0.12
2025-05	0.15	0.1	-0.08
2025-06	0.3	0.22	0.05
2025-07	-0.15	-0.08	-0.2
"""
cpi_text = """Prev_CPI_Value	Next_CPI_Value	FOMC_date
241.432	243.603	2017-02-01
243.603	243.801	2017-03-15
243.801	244.733	2017-05-03
244.733	244.955	2017-06-14
244.955	244.786	2017-07-26
245.519	246.819	2017-09-20
246.819	246.669	2017-11-01
246.669	246.524	2017-12-13
246.524	247.867	2018-01-31
248.991	249.554	2018-03-21
249.554	251.588	2018-05-02
251.588	251.989	2018-06-13
251.989	252.146	2018-08-01
252.146	252.439	2018-09-26
252.439	252.038	2018-11-08
252.038	251.233	2018-12-19
251.233	251.712	2019-01-30
252.776	254.202	2019-03-20
254.202	256.092	2019-05-01
256.092	256.143	2019-06-19
256.143	256.571	2019-07-31
256.558	256.759	2019-09-18
256.759	257.346	2019-10-30
257.208	256.974	2019-12-11
256.974	257.971	2020-01-29
257.971	258.115	2020-03-03
258.678	258.115	2020-03-15
258.115	256.389	2020-04-29
256.394	257.797	2020-06-10
257.797	259.101	2020-07-29
259.918	260.28	2020-09-16
260.28	260.229	2020-11-05
260.229	260.474	2020-12-16
260.474	261.582	2021-01-27
263.014	264.877	2021-03-17
264.877	267.054	2021-04-28
269.195	271.696	2021-06-16
271.696	273.003	2021-07-28
273.567	274.31	2021-09-22
274.31	277.948	2021-11-03
277.948	278.802	2021-12-15
278.802	281.148	2022-01-26
283.716	287.504	2022-03-16
287.504	292.296	2022-05-04
292.296	296.311	2022-06-15
296.311	296.276	2022-07-27
296.171	296.808	2022-09-21
296.808	297.711	2022-11-02
297.711	296.797	2022-12-14
296.797	300.84	2023-02-01
300.84	301.836	2023-03-22
301.836	304.127	2023-05-03
304.127	305.109	2023-06-14
305.109	305.691	2023-07-26
307.026	307.789	2023-09-20
307.789	307.051	2023-11-01
307.051	306.746	2023-12-13
306.746	308.417	2024-01-31
310.326	312.332	2024-03-20
312.332	314.069	2024-05-01
314.069	314.175	2024-06-12
314.175	314.54	2024-07-31
314.796	315.301	2024-09-18
315.301	315.493	2024-11-07
315.493	315.605	2024-12-18
315.605	317.671	2025-01-29
319.082	319.799	2025-03-19
321.465	322.561	2025-05-18
321.465	322.561	2025-06-18
322.561	323.048	2025-07-30
"""
policy_df_raw = pd.read_csv(io.StringIO(dedent(policy_text)), sep=r"\s+")
cpi_df = pd.read_csv(io.StringIO(dedent(cpi_text)), sep=r"\s+")
cpi_df['FOMC_date'] = pd.to_datetime(cpi_df['FOMC_date'])

# ---------- 2. CPI 월별 회의 순번 라벨링 ----------
cpi_df['month'] = cpi_df['FOMC_date'].dt.strftime('%Y-%m')
cpi_df['idx_in_month'] = cpi_df.groupby('month').cumcount()

def make_date_key(row):
    if row['month'] == '2020-03':
        return f"{row['month']}{'x' if row['idx_in_month']==0 else 'y'}"
    else:
        if row['idx_in_month'] == 0:
            return row['month']
        else:
            return f"{row['month']}_{row['idx_in_month']}"
cpi_df['date_key'] = cpi_df.apply(make_date_key, axis=1)

# ---------- 3. Policy 데이터 전처리 ----------
policy_df = policy_df_raw.copy()
policy_df['date'] = policy_df['date'].str.strip()

has_x = (policy_df['date'] == '2020-03x').any()
has_y = (policy_df['date'] == '2020-03y').any()
has_plain_mar = (policy_df['date'] == '2020-03').any()

if has_plain_mar and (has_x or has_y):
    raise ValueError("2020-03 / 2020-03x / 2020-03y 혼재. 하나의 방식만 유지하세요.")

# ---------- 4A / 4B ----------
if has_x and has_y:
    policy_df['date_key'] = policy_df['date']
    method = "B"
    merged = policy_df.merge(
        cpi_df[['date_key','Prev_CPI_Value','Next_CPI_Value','FOMC_date']],
        on='date_key', how='left'
    )
elif has_plain_mar:
    method = "A"
    mar_rows = cpi_df[cpi_df['month'] == '2020-03']
    if len(mar_rows) != 2:
        raise ValueError("CPI 데이터에서 2020-03 두 회의를 찾지 못했습니다.")
    policy_df['date_key'] = policy_df['date']
    cpi_month = cpi_df.groupby('month', as_index=False).agg({
        'Prev_CPI_Value':'mean',
        'Next_CPI_Value':'mean',
        'FOMC_date':'min'
    }).rename(columns={'month':'date_key'})
    merged = policy_df.merge(
        cpi_month[['date_key','Prev_CPI_Value','Next_CPI_Value','FOMC_date']],
        on='date_key', how='left'
    )
else:
    raise ValueError("2020-03 형식을 인식하지 못했습니다. (x,y 또는 단일 2020-03 필요)")

# ---------- 5. 매핑 실패 검사 ----------
missing = merged[merged['Prev_CPI_Value'].isna()]
if not missing.empty:
    print("매핑 실패 행:")
    print(missing[['date','date_key']])
    raise AssertionError("일부 날짜 매핑 실패. 규칙 업데이트 필요.")

print(f"Method {method} 적용. 관측수 = {len(merged)}")

# ---------- 6. 6개 산점도 (회귀선 3개, 구간별 색상) ----------
pairs = [
    ('Statement', 'Prev_CPI_Value'),
    ('P&B', 'Prev_CPI_Value'),
    ('Policy', 'Prev_CPI_Value'),
    ('Statement', 'Next_CPI_Value'),
    ('P&B', 'Next_CPI_Value'),
    ('Policy', 'Next_CPI_Value'),
]

# 연도 구간 분할
merged['year'] = merged['FOMC_date'].dt.year
early_mask = merged['year'] <= 2022      # 2017~2022
late_mask  = merged['year'] >= 2023      # 2023~2025

early_color = '#1f77b4'  # 파랑
late_color  = '#d62728'  # 빨강
overall_color = '#2ca02c'  # 초록 (전체 회귀선)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

def fit_line(x, y):
    # x, y: 1d numpy arrays
    m, b = np.polyfit(x, y, 1)
    r, p = stats.pearsonr(x, y)
    return m, b, r, p

def add_panel(ax, df, xcol, ycol):
    # 데이터
    x_all = df[xcol].to_numpy()
    y_all = df[ycol].to_numpy()

    # 구간별 데이터
    x_early = df.loc[early_mask, xcol].to_numpy()
    y_early = df.loc[early_mask, ycol].to_numpy()
    x_late  = df.loc[late_mask,  xcol].to_numpy()
    y_late  = df.loc[late_mask,  ycol].to_numpy()

    # 산점도 (점 색상: 두 구간)
    ax.scatter(x_early, y_early, color=early_color, edgecolors='white',
               linewidth=0.5, alpha=0.85, label='2017-2022')
    ax.scatter(x_late, y_late, color=late_color, edgecolors='white',
               linewidth=0.5, alpha=0.85, label='2023-2025')

    # 회귀선 x 구간 (각 subset 범위에 맞춰)
    def plot_reg(x_subset, y_subset, color, label):
        if len(x_subset) < 2:
            return None
        m, b, r, p = fit_line(x_subset, y_subset)
        xs = np.linspace(x_subset.min(), x_subset.max(), 120)
        ax.plot(xs, m*xs + b, color=color, lw=1.6, label=f"{label} (r={r:.2f})")
        return (m, b, r, p)

    stats_early = plot_reg(x_early, y_early, early_color, "회귀 17-22")
    stats_late  = plot_reg(x_late,  y_late,  late_color,  "회귀 23-25")

    # 전체 회귀선 (전체 x 범위)
    m_all, b_all, r_all, p_all = fit_line(x_all, y_all)
    xs_all = np.linspace(x_all.min(), x_all.max(), 200)
    ax.plot(xs_all, m_all*xs_all + b_all, color=overall_color, lw=1.8,
            linestyle='--', label=f"회귀 전체 (r={r_all:.2f})")

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"{xcol} vs {ycol}")

    # 요약 텍스트 (왼쪽 위 박스) - 구간별 r, p
    lines = []
    if stats_early:
        lines.append(f"17-22 r={stats_early[2]:.2f} p={stats_early[3]:.1e}")
    if stats_late:
        lines.append(f"23-25 r={stats_late[2]:.2f} p={stats_late[3]:.1e}")
    lines.append(f"All r={r_all:.2f} p={p_all:.1e}")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.75))

    ax.grid(alpha=0.3)

for ax, (xcol, ycol) in zip(axes, pairs):
    add_panel(ax, merged, xcol, ycol)

# 범례 (한 번만)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=True, fontsize=9, bbox_to_anchor=(0.5, 1.02))

plt.suptitle(f"Scatter with 3 Regression Lines (Method {method})", fontsize=14, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.965])
plt.show()

# ---------- 7. 상관 요약 (원래 전체 구간 기준 + 구간별도 원하면 추가 가능) ----------
from scipy import stats
from pprint import pprint

summary = {}
def corr_block(x, y):
    pear = stats.pearsonr(x, y)
    spear = stats.spearmanr(x, y)
    kend = stats.kendalltau(x, y)
    return {
        'n': len(x),
        'pearson_r': pear.statistic, 'pearson_p': pear.pvalue,
        'spearman_rho': spear.statistic, 'spearman_p': spear.pvalue,
        'kendall_tau': kend.statistic, 'kendall_p': kend.pvalue
    }

for xcol, ycol in pairs:
    summary[f"{xcol}~{ycol}_ALL"] = corr_block(merged[xcol], merged[ycol])
    summary[f"{xcol}~{ycol}_2017_2022"] = corr_block(merged.loc[early_mask, xcol],
                                                     merged.loc[early_mask, ycol])
    summary[f"{xcol}~{ycol}_2023_2025"] = corr_block(merged.loc[late_mask, xcol],
                                                     merged.loc[late_mask, ycol])

print("\n=== Correlation Summary (ALL / 2017-2022 / 2023-2025) ===")
pprint(summary)