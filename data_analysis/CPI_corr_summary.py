import pandas as pd
from io import StringIO
from scipy.stats import pearsonr, spearmanr, kendalltau
import os
from typing import Dict, List

policy_text = """date,statment,P&B,Policy
2017-02-01,0.3,0.32,0.25
2017-03-01,0.42,0.445,0.33
2017-05-01,0.35,0.325,0.22
2017-06-01,0.43,0.35,0.295
2017-07-01,0.36,0.325,0.285
2017-09-01,0.4,0.35,0.205
2017-11-01,0.4,0.41,0.335
2017-12-01,0.55,0.5,0.435
2018-01-01,0.64,0.6,0.52
2018-03-01,0.45,0.525,0.57
2018-05-01,0.45,0.46,0.425
2018-06-01,0.74,0.7,0.59
2018-08-01,0.75,0.73,0.61
2018-09-01,0.84,0.7,0.715
2018-11-01,0.7,0.665,0.48
2018-12-01,0.7,0.585,0.46
2019-01-01,0.45,0.41,0.185
2019-03-01,0.2,0.155,-0.005
2019-05-01,0.3,0.265,0.075
2019-06-01,0.2,0.11,0.035
2019-07-01,0.25,0.2,0.1
2019-09-01,0.24,0.15,0.065
2019-10-01,0.2,0.18,0.04
2019-12-01,0.3,0.245,0.11
2020-01-01,0.3,0.2,0.035
2020-03-01,-0.18,-0.1,-0.255
2020-04-01,-0.82,-0.51,-0.65
2020-06-01,-0.62,-0.41,-0.635
2020-07-01,-0.4,-0.25,-0.3
2020-09-01,-0.3,-0.135,-0.2
2020-11-01,-0.3,-0.12,-0.135
2020-12-01,-0.3,-0.11,-0.21
2021-01-01,-0.3,-0.15,-0.04
2021-03-01,0.1,0.18,0.265
2021-04-01,0.22,0.35,0.41
2021-06-01,0.44,0.425,0.465
2021-07-01,0.45,0.405,0.55
2021-09-01,0.35,0.295,0.27
2021-11-01,0.35,0.28,0.235
2021-12-01,0.35,0.275,0.245
2022-01-01,0.3,0.21,0.135
2022-03-01,0.1,0.055,0.04
2022-05-01,-0.12,-0.155,-0.1
2022-06-01,-0.12,-0.13,-0.12
2022-07-01,-0.32,-0.25,-0.525
2022-09-01,-0.23,-0.22,-0.375
2022-11-01,-0.25,-0.165,-0.425
2022-12-01,-0.15,-0.225,-0.425
2023-02-01,-0.15,-0.11,-0.055
2023-03-01,-0.1,-0.18,-0.4
2023-05-01,0.1,-0.05,-0.285
2023-06-01,0.15,0.05,-0.12
2023-07-01,0.12,0.065,-0.045
2023-09-01,0.2,0.11,-0.1
2023-11-01,0.2,0.1,-0.12
2023-12-01,-0.1,-0.075,-0.22
2024-01-01,0.25,0.175,0.02
2024-03-01,0.35,0.345,0.225
2024-05-01,0.25,0.135,0.03
2024-06-01,0.35,0.25,0.05
2024-07-01,0.25,0.2,0.055
2024-09-01,0.32,0.4,0.21
2024-11-01,0.35,0.345,0.15
2024-12-01,0.3,0.31,0.2
2025-01-01,0.34,0.27,0.085
2025-03-01,0.3,0.2,0.055
2025-05-01,0.15,0.09,-0.1
2025-06-01,0.3,0.2,0.03
2025-07-01,-0.1,-0.055,-0.1
"""

cpi_text = """Prev_CPI_Value\tNext_CPI_Value\tFOMC_date
241.432\t243.603\t2017-02-01
243.603\t243.801\t2017-03-15
243.801\t244.733\t2017-05-03
244.733\t244.955\t2017-06-14
244.955\t244.786\t2017-07-26
245.519\t246.819\t2017-09-20
246.819\t246.669\t2017-11-01
246.669\t246.524\t2017-12-13
246.524\t247.867\t2018-01-31
248.991\t249.554\t2018-03-21
249.554\t251.588\t2018-05-02
251.588\t251.989\t2018-06-13
251.989\t252.146\t2018-08-01
252.146\t252.439\t2018-09-26
252.439\t252.038\t2018-11-08
252.038\t251.233\t2018-12-19
251.233\t251.712\t2019-01-30
252.776\t254.202\t2019-03-20
254.202\t256.092\t2019-05-01
256.092\t256.143\t2019-06-19
256.143\t256.571\t2019-07-31
256.558\t256.759\t2019-09-18
256.759\t257.346\t2019-10-30
257.208\t256.974\t2019-12-11
256.974\t257.971\t2020-01-29
257.971\t258.115\t2020-03-03
258.678\t258.115\t2020-03-15
258.115\t256.389\t2020-04-29
256.394\t257.797\t2020-06-10
257.797\t259.101\t2020-07-29
259.918\t260.28\t2020-09-16
260.28\t260.229\t2020-11-05
260.229\t260.474\t2020-12-16
260.474\t261.582\t2021-01-27
263.014\t264.877\t2021-03-17
264.877\t267.054\t2021-04-28
269.195\t271.696\t2021-06-16
271.696\t273.003\t2021-07-28
273.567\t274.31\t2021-09-22
274.31\t277.948\t2021-11-03
277.948\t278.802\t2021-12-15
278.802\t281.148\t2022-01-26
283.716\t287.504\t2022-03-16
287.504\t292.296\t2022-05-04
292.296\t296.311\t2022-06-15
296.311\t296.276\t2022-07-27
296.171\t296.808\t2022-09-21
296.808\t297.711\t2022-11-02
297.711\t296.797\t2022-12-14
296.797\t300.84\t2023-02-01
300.84\t301.836\t2023-03-22
301.836\t304.127\t2023-05-03
304.127\t305.109\t2023-06-14
305.109\t305.691\t2023-07-26
307.026\t307.789\t2023-09-20
307.789\t307.051\t2023-11-01
307.051\t306.746\t2023-12-13
306.746\t308.417\t2024-01-31
310.326\t312.332\t2024-03-20
312.332\t314.069\t2024-05-01
314.069\t314.175\t2024-06-12
314.175\t314.54\t2024-07-31
314.796\t315.301\t2024-09-18
315.301\t315.493\t2024-11-07
315.493\t315.605\t2024-12-18
315.605\t317.671\t2025-01-29
319.082\t319.799\t2025-03-19
319.799\t320.795\t2025-05-30
321.465\t322.561\t2025-06-18
322.561\t323.048\t2025-07-30
"""

# ---------- 읽기 ----------
df_policy = pd.read_csv(StringIO(policy_text), sep=',')
df_policy.columns = ['date', 'statement', 'p_b', 'policy']  # 오타(statment) 정리 포함

df_cpi = pd.read_csv(StringIO(cpi_text), sep='\t')

# ---------- 날짜 파싱 & ym/occ 생성(양쪽 동일 규칙) ----------
# policy
df_policy['date_dt'] = pd.to_datetime(df_policy['date'], errors='coerce')
df_policy = df_policy.sort_values('date_dt')
df_policy['ym'] = df_policy['date_dt'].dt.strftime('%Y-%m')
df_policy['occ'] = df_policy.groupby('ym').cumcount()

# cpi
df_cpi['FOMC_date'] = pd.to_datetime(df_cpi['FOMC_date'], errors='coerce')
df_cpi = df_cpi.sort_values('FOMC_date')
df_cpi['ym'] = df_cpi['FOMC_date'].dt.strftime('%Y-%m')
df_cpi['occ'] = df_cpi.groupby('ym').cumcount()

# ---------- 머지 ----------
df = pd.merge(df_policy, df_cpi, on=['ym', 'occ'], how='inner')

print(f"[DEBUG] merged rows = {len(df)}")
if df.empty:
    print("[DEBUG] merge가 비었습니다. 월별 건수를 확인하세요.")
    print("policy 월별 개수:\n", df_policy['ym'].value_counts().sort_index().head(18))
    print("cpi    월별 개수:\n", df_cpi['ym'].value_counts().sort_index().head(18))

# ---------- 분석 ----------
df['year'] = df['FOMC_date'].dt.year

def compute_correlations(sub: pd.DataFrame, sentiment_col: str, cpi_col: str) -> Dict[str, float]:
    data = sub[[sentiment_col, cpi_col]].dropna()
    if len(data) < 3:
        return {'pearson_r': float('nan'), 'pearson_p': float('nan'),
                'spearman_rho': float('nan'), 'spearman_p': float('nan'),
                'kendall_tau': float('nan'), 'kendall_p': float('nan'),
                'n': len(data)}
    pr, pp = pearsonr(data[sentiment_col], data[cpi_col])
    sr, sp = spearmanr(data[sentiment_col], data[cpi_col])
    kt, kp = kendalltau(data[sentiment_col], data[cpi_col])
    return {'pearson_r': pr, 'pearson_p': pp,
            'spearman_rho': sr, 'spearman_p': sp,
            'kendall_tau': kt, 'kendall_p': kp,
            'n': len(data)}

sentiment_cols = [('statement', 'Statement'), ('p_b', 'P&B'), ('policy', 'Policy')]
cpi_types = [('Prev_CPI_Value', 'prev'), ('Next_CPI_Value', 'next')]
periods = [((2017, 2022), '2017-2022'), ((2023, 2025), '2023-2025')]

rows: List[Dict[str, float]] = []
for (start_end, period_label) in periods:
    start, end = start_end
    sub_period = df[(df['year'] >= start) & (df['year'] <= end)]
    for s_col, s_label in sentiment_cols:
        for c_col, c_label in cpi_types:
            stats = compute_correlations(sub_period, s_col, c_col)
            rows.append({'period': period_label,
                         'sentiment_type': s_label,
                         'CPI_type': c_label,
                         **stats})

result_df = pd.DataFrame(rows, columns=[
    'period','sentiment_type','CPI_type','n',
    'pearson_r','pearson_p','spearman_rho','spearman_p','kendall_tau','kendall_p'
])

# ---------- 저장 ----------
output_file = '../data_analysis/CPI_correlation_results.xlsx'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
result_df.to_excel(output_file, index=False)

print("Correlation results:")
print(result_df)
print(f"\nSaved to {output_file}")