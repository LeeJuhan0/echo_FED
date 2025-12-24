import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import PchipInterpolator, UnivariateSpline
import io
from textwrap import dedent
from collections import OrderedDict
import pprint

# ================== 0. 설정 파라미터 ==================
B_BOOT = 700           # 부트스트랩 반복 횟수
CI_LEVEL = 0.95        # 신뢰구간 수준
band_type = 'mean'     # 'mean' 또는 'predictive'
grid_points = 220
grid_mode = 'quantile' # 'quantile' or 'linear'
quantile_smoothing = True
optional_spline_smoothing = True
smooth_factor = 0.25   # Bagged mean 추가 스플라인 후처리 강도(0~1 근사)
min_points_copula = 5

# ================== 1. 데이터 읽기 ==================
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

# ================== 2. CPI date key ==================
cpi_df['month'] = cpi_df['FOMC_date'].dt.strftime('%Y-%m')
cpi_df['idx_in_month'] = cpi_df.groupby('month').cumcount()

def make_date_key(row):
    if row['month'] == '2020-03':
        return f"{row['month']}{'x' if row['idx_in_month']==0 else 'y'}"
    return row['month'] if row['idx_in_month'] == 0 else f"{row['month']}_{row['idx_in_month']}"
cpi_df['date_key'] = cpi_df.apply(make_date_key, axis=1)

# ================== 3. Policy 전처리 ==================
policy_df = policy_df_raw.copy()
policy_df['date'] = policy_df['date'].str.strip()
has_x = (policy_df['date'] == '2020-03x').any()
has_y = (policy_df['date'] == '2020-03y').any()
has_plain_mar = (policy_df['date'] == '2020-03').any()
if has_plain_mar and (has_x or has_y):
    raise ValueError("2020-03 / 2020-03x / 2020-03y 혼재.")

if has_x and has_y:
    method = "B"
    policy_df['date_key'] = policy_df['date']
    merged = policy_df.merge(
        cpi_df[['date_key','Prev_CPI_Value','Next_CPI_Value','FOMC_date']],
        on='date_key', how='left'
    )
elif has_plain_mar:
    method = "A"
    mar_rows = cpi_df[cpi_df['month'] == '2020-03']
    if len(mar_rows) != 2:
        raise ValueError("2020-03 두 회의 못찾음.")
    policy_df['date_key'] = policy_df['date']
    cpi_month = cpi_df.groupby('month', as_index=False).agg({
        'Prev_CPI_Value':'mean','Next_CPI_Value':'mean','FOMC_date':'min'
    }).rename(columns={'month':'date_key'})
    merged = policy_df.merge(
        cpi_month[['date_key','Prev_CPI_Value','Next_CPI_Value','FOMC_date']],
        on='date_key', how='left'
    )
else:
    raise ValueError("2020-03 표기 인식 실패.")

missing = merged[merged['Prev_CPI_Value'].isna()]
if not missing.empty:
    print(missing[['date','date_key']])
    raise AssertionError("일부 날짜 매핑 실패.")
print(f"Method {method} 적용. 관측수={len(merged)}")

# ================== 4. 분석 준비 ==================
pairs = [
    ('Statement', 'Prev_CPI_Value'),
    ('P&B', 'Prev_CPI_Value'),
    ('Policy', 'Prev_CPI_Value'),
    ('Statement', 'Next_CPI_Value'),
    ('P&B', 'Next_CPI_Value'),
    ('Policy', 'Next_CPI_Value'),
]
merged['year'] = merged['FOMC_date'].dt.year
early_mask = merged['year'] <= 2022
late_mask  = merged['year'] >= 2023

# 색상
early_point_color = '#1f77b4'
late_point_color  = '#d62728'
subset_colors = {
    '17-22': '#1f77b4',
    '23-25': '#d62728',
    '전체'  : '#2ca02c'
}
ci_gray_levels = {
    '17-22': '#555555',
    '23-25': '#888888',
    '전체' : '#BBBBBB'
}
ci_gray_alpha = {
    '17-22': 0.30,
    '23-25': 0.30,
    '전체' : 0.25
}

alpha = 1 - CI_LEVEL
z_crit = stats.norm.ppf(1 - alpha/2)

# ================== 5. 보조 함수 ==================
def collapse_duplicate_x(x, y):
    x = np.asarray(x); y = np.asarray(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]; y = y[mask]
    if len(x)==0: return x,y
    ux, inv = np.unique(x, return_inverse=True)
    y_sum = np.zeros_like(ux, dtype=float); cnt = np.zeros_like(ux, dtype=int)
    for i,v in zip(inv,y):
        y_sum[i]+=v; cnt[i]+=1
    return ux, y_sum/cnt

def build_smoothed_inverse_cdf(values):
    """단조 cubic (PCHIP) 기반 역CDF quantile 함수 생성"""
    v = np.sort(values)
    n = len(v)
    ranks = np.arange(1, n+1)
    # Blom 보정
    u = (ranks - 0.375) / (n + 0.25)
    # 양끝 보정
    u_ext = np.concatenate(([0.0], u, [1.0]))
    v_ext = np.concatenate(([v[0]], v, [v[-1]]))
    return PchipInterpolator(u_ext, v_ext, extrapolate=True)

def gaussian_copula_single(x, y,
                           grid_points=200,
                           grid_mode='quantile',
                           band_type='mean',
                           quantile_smoothing=True):
    """
    단일 (x,y)에서 가우시안 코플라 평균 및 (mean or predictive) 밴드(분산/표준편차는 여기서 X 의존 없음).
    여기서는 base mean만 (후처리 bootstrap bagging은 별도 함수) 반환.
    """
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < min_points_copula:
        return None
    # 순위 변환 (Blom)
    rx = stats.rankdata(x, method='average')
    ry = stats.rankdata(y, method='average')
    u_x = (rx - 0.375) / (len(x) + 0.25)
    u_y = (ry - 0.375) / (len(y) + 0.25)

    z_x = stats.norm.ppf(np.clip(u_x,1e-6,1-1e-6))
    z_y = stats.norm.ppf(np.clip(u_y,1e-6,1-1e-6))
    rho = np.corrcoef(z_x, z_y)[0,1]
    rho = np.clip(rho, -0.9999, 0.9999)

    # grid
    if grid_mode == 'quantile':
        q = np.linspace(0,1,grid_points)
        x_sorted = np.sort(x)
        grid_x = np.quantile(x_sorted, q)
        u_grid = q
    else:
        grid_x = np.linspace(x.min(), x.max(), grid_points)
        # ECDF 근사
        x_sorted = np.sort(x)
        ranks = np.searchsorted(x_sorted, grid_x, side='right')
        ranks = np.clip(ranks,1,len(x))
        u_grid = (ranks - 0.375)/(len(x)+0.25)

    z_grid = stats.norm.ppf(np.clip(u_grid,1e-6,1-1e-6))
    z_mean = rho * z_grid
    var_cond = 1 - rho**2
    sd_cond = np.sqrt(var_cond)

    if band_type == 'predictive':
        z_low = z_mean - z_crit*sd_cond
        z_high= z_mean + z_crit*sd_cond
    elif band_type == 'mean':
        # mean band
        sd_mean = sd_cond / np.sqrt(len(x))
        z_low = z_mean - z_crit*sd_mean
        z_high= z_mean + z_crit*sd_mean
    else:
        raise ValueError("band_type must be 'predictive' or 'mean'")

    if quantile_smoothing:
        qfun = build_smoothed_inverse_cdf(y)
        def z_to_y(zv):
            uu = stats.norm.cdf(zv)
            return qfun(uu)
    else:
        y_sorted = np.sort(y)
        n = len(y)
        def z_to_y(zv):
            uu = stats.norm.cdf(zv)
            pos = uu*(n+1)
            pos = np.clip(pos,1,n)
            lo = np.floor(pos).astype(int)
            hi = np.ceil(pos).astype(int)
            w = pos - lo
            lo = np.clip(lo,1,n); hi = np.clip(hi,1,n)
            return (1-w)*y_sorted[lo-1] + w*y_sorted[hi-1]

    mean_y = z_to_y(z_mean)
    low_y  = z_to_y(z_low)
    high_y = z_to_y(z_high)
    return dict(grid_x=grid_x, mean=mean_y, lower=low_y, upper=high_y,
                rho=rho, n=len(x))

def bootstrap_copula_bagging(x, y,
                             B=500,
                             grid_points=200,
                             grid_mode='quantile',
                             band_type='mean',
                             quantile_smoothing=True,
                             seed=2024):
    """
    부트스트랩: 각 bootstrap에서 가우시안 코플라 mean/band 계산
    - bagged_mean: 모든 부트스트랩 mean들의 평균
    - CI: 각 grid에서 (2.5%, 97.5%) (mean band)
    base(원자료) 결과도 함께 반환 (비교용)
    """
    rng = np.random.default_rng(seed)
    base = gaussian_copula_single(x, y,
                                  grid_points=grid_points,
                                  grid_mode=grid_mode,
                                  band_type=band_type,
                                  quantile_smoothing=quantile_smoothing)
    if base is None:
        return None
    n = len(x)
    means = []
    lowers = []
    uppers = []
    for b in range(B):
        idx = rng.integers(0, n, n)
        xb = x[idx]; yb = y[idx]
        xb2, yb2 = collapse_duplicate_x(xb, yb)
        res_b = gaussian_copula_single(xb2, yb2,
                                       grid_points=grid_points,
                                       grid_mode=grid_mode,
                                       band_type=band_type,
                                       quantile_smoothing=quantile_smoothing)
        if res_b is None:
            continue
        means.append(res_b['mean'])
        lowers.append(res_b['lower'])
        uppers.append(res_b['upper'])
    if len(means) < max(30, B//5):
        # 부트스트랩 실패가 많아 신뢰도 낮음
        return dict(**base,
                    bagged_mean=base['mean'],
                    bagged_lower=base['lower'],
                    bagged_upper=base['upper'],
                    B_eff=len(means),
                    warning="부트스트랩 유효 반복 수 적음")

    means_arr = np.vstack(means)
    lowers_arr= np.vstack(lowers)
    uppers_arr= np.vstack(uppers)

    bagged_mean = means_arr.mean(axis=0)
    # mean band CI (mean 자체의 변동) -> means_arr 분위수
    q_low = np.percentile(means_arr, 2.5, axis=0)
    q_high= np.percentile(means_arr,97.5, axis=0)

    # (선택) 추가 약한 후처리 스플라인
    if optional_spline_smoothing:
        # heuristic s: var * len * smooth_factor
        v = np.var(bagged_mean)
        s_param = v * len(bagged_mean) * smooth_factor
        try:
            sp = UnivariateSpline(base['grid_x'], bagged_mean, s=s_param, k=3)
            bagged_mean_smooth = sp(base['grid_x'])
            # 구간도 약간 스무딩 (너무 과하진 않게 동일 sp 의 residual scale 사용)
            sp_l = UnivariateSpline(base['grid_x'], q_low, s=s_param*0.8, k=3)
            sp_u = UnivariateSpline(base['grid_x'], q_high, s=s_param*0.8, k=3)
            q_low = sp_l(base['grid_x'])
            q_high= sp_u(base['grid_x'])
            bagged_mean = bagged_mean_smooth
        except Exception:
            pass

    return dict(grid_x=base['grid_x'],
                base_mean=base['mean'],
                base_lower=base['lower'],
                base_upper=base['upper'],
                bagged_mean=bagged_mean,
                bagged_lower=q_low,
                bagged_upper=q_high,
                rho=base['rho'],
                n=base['n'],
                B_eff=len(means))

# ================== 6. 패널 함수 ==================
def add_panel(ax, df, xcol, ycol):
    # 산점도 (전체 점)
    ax.scatter(df.loc[early_mask, xcol], df.loc[early_mask, ycol],
               color=early_point_color, edgecolors='white',
               linewidth=0.5, alpha=0.85, label='2017-2022 points')
    ax.scatter(df.loc[late_mask, xcol], df.loc[late_mask, ycol],
               color=late_point_color, edgecolors='white',
               linewidth=0.5, alpha=0.85, label='2023-2025 points')

    subsets = [
        ('17-22', early_mask),
        ('23-25', late_mask),
        ('전체', slice(None)),
    ]
    info_lines = []

    for label_text, mask in subsets:
        x_raw = df.loc[mask, xcol].to_numpy()
        y_raw = df.loc[mask, ycol].to_numpy()
        x_proc, y_proc = collapse_duplicate_x(x_raw, y_raw)
        if len(x_proc) < min_points_copula:
            continue

        res = bootstrap_copula_bagging(
            x_proc, y_proc,
            B=B_BOOT,
            grid_points=grid_points,
            grid_mode=grid_mode,
            band_type=band_type,
            quantile_smoothing=quantile_smoothing,
            seed=2024
        )
        if res is None:
            continue

        # CI 채우기 (bagged mean CI)
        ci_color = ci_gray_levels[label_text]
        ax.fill_between(res['grid_x'], res['bagged_lower'], res['bagged_upper'],
                        color=ci_color, alpha=ci_gray_alpha[label_text], linewidth=0,
                        label=f"{label_text} {int(CI_LEVEL*100)}% Copula (boot mean CI)")

        # Bagged mean 곡선
        ax.plot(res['grid_x'], res['bagged_mean'],
                color=subset_colors[label_text], lw=2.2,
                label=f"{label_text} Bagged mean")

        # Base mean (참고용 점선)
        ax.plot(res['grid_x'], res['base_mean'],
                color=subset_colors[label_text], lw=1.2, ls='--', alpha=0.65,
                label=f"{label_text} Base mean(단일)")

        # 상관/요약
        if len(x_raw) >= 3:
            try:
                r, p = stats.pearsonr(x_raw, y_raw)
                info_lines.append(
                    f"{label_text} r={r:.2f} p={p:.1e} ρ={res['rho']:.2f} B={res['B_eff']}"
                )
            except Exception:
                pass

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"{xcol} vs {ycol}")
    ax.grid(alpha=0.3)
    if info_lines:
        ax.text(0.02, 0.98, "\n".join(info_lines),
                transform=ax.transAxes, va='top', ha='left',
                fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3',
                          fc='white', ec='gray', alpha=0.75))

# ================== 7. 플로팅 ==================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for ax, (xcol, ycol) in zip(axes, pairs):
    add_panel(ax, merged, xcol, ycol)

# 범례 중복 제거
handles_all, labels_all = [], []
for ax in axes:
    h,l = ax.get_legend_handles_labels()
    handles_all.extend(h); labels_all.extend(l)
uniq = OrderedDict()
for h,l in zip(handles_all, labels_all):
    if l not in uniq:
        uniq[l] = h

fig.legend(uniq.values(), uniq.keys(),
           loc='upper center', ncol=4,
           frameon=True, fontsize=8,
           bbox_to_anchor=(0.5, 1.02))

title_band = "Mean" if band_type=='mean' else "Predictive"
plt.suptitle(
    f"Gaussian Copula (Bagged {title_band} Curve, {int(CI_LEVEL*100)}% CI, B={B_BOOT}, Method {method})",
    fontsize=14, y=0.995
)
plt.tight_layout(rect=[0,0,1,0.965])
plt.show()

# ================== 8. 기간별 상관 요약 ==================
def corr_block(x, y):
    pear = stats.pearsonr(x, y)
    spear = stats.spearmanr(x, y)
    kend = stats.kendalltau(x, y)
    return dict(
        n=len(x),
        pearson_r=pear.statistic, pearson_p=pear.pvalue,
        spearman_rho=spear.statistic, spearman_p=spear.pvalue,
        kendall_tau=kend.statistic, kendall_p=kend.pvalue
    )

summary = {}
for xcol, ycol in pairs:
    summary[f"{xcol}~{ycol}_ALL"] = corr_block(merged[xcol], merged[ycol])
    summary[f"{xcol}~{ycol}_2017_2022"] = corr_block(merged.loc[early_mask, xcol],
                                                     merged.loc[early_mask, ycol])
    summary[f"{xcol}~{ycol}_2023_2025"] = corr_block(merged.loc[late_mask, xcol],
                                                     merged.loc[late_mask, ycol])

print("\n=== Correlation Summary (ALL / 2017-2022 / 2023-2025) ===")
pprint.pprint(summary)