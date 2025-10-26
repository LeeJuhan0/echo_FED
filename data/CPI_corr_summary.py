import pandas as pd
from io import StringIO
from scipy.stats import pearsonr, spearmanr, kendalltau

policy_text = """date\tstatment\tP&B\tPolicy
2017-02\t0.33\t0.32\t0.29
2017-03\t0.4\t0.4\t0.35
2017-05\t0.34\t0.36\t0.33
2017-06\t0.4\t0.35\t0.3
2017-07\t0.35\t0.32\t0.24
2017-09\t0.38\t0.36\t0.28
2017-11\t0.4\t0.43\t0.38
2017-12\t0.58\t0.55\t0.5
2018-01\t0.66\t0.65\t0.55
2018-03\t0.4\t0.52\t0.52
2018-05\t0.44\t0.4\t0.3
2018-06\t0.75\t0.7\t0.66
2018-08\t0.65\t0.67\t0.63
2018-09\t0.79\t0.72\t0.76
2018-11\t0.66\t0.65\t0.5
2018-12\t0.64\t0.6\t0.43
2019-01\t0.4\t0.48\t0.36
2019-06\t0.2\t0.12\t0.04
2019-07\t0.21\t0.2\t0.08
2019-09\t0.23\t0.2\t0.08
2019-10\t0.2\t0.22\t0.08
2019-12\t0.3\t0.25\t0.12
2020-01\t0.28\t0.25\t0.12
2020-03\t-0.2\t-0.1\t-0.24
2020-03\t-0.62\t-0.4\t-0.65
2020-04\t-0.75\t-0.45\t-0.65
2020-06\t-0.66\t-0.5\t-0.55
2020-07\t-0.4\t-0.22\t-0.35
2020-09\t-0.25\t-0.1\t-0.3
2020-11\t-0.25\t-0.1\t-0.2
2020-12\t-0.22\t-0.1\t-0.12
2021-01\t-0.3\t-0.15\t-0.09
2021-03\t0.1\t0.2\t0.2
2021-04\t0.2\t0.22\t0.3
2021-06\t0.4\t0.435\t0.45
2021-07\t0.4\t0.45\t0.55
2021-09\t0.3\t0.25\t0.25
2021-11\t0.3\t0.25\t0.3
2021-12\t0.32\t0.27\t0.27
2022-01\t0.26\t0.2\t0.18
2022-03\t0.1\t0.04\t-0.01
2022-05\t-0.15\t-0.18\t-0.12
2022-06\t-0.15\t-0.2\t-0.19
2022-07\t-0.3\t-0.4\t-0.48
2022-09\t-0.25\t-0.35\t-0.5
2022-11\t-0.2\t-0.25\t-0.4
2022-12\t-0.2\t-0.25\t-0.4
2023-02\t-0.1\t-0.14\t-0.15
2023-03\t-0.14\t-0.18\t-0.3
2023-05\t0.1\t0.04\t-0.2
2023-06\t0.1\t0.05\t-0.12
2023-07\t0.1\t0.05\t-0.04
2023-09\t0.18\t0.1\t-0.05
2023-11\t0.15\t0.1\t-0.05
2023-12\t-0.1\t-0.08\t-0.2
2024-01\t0.3\t0.2\t0.05
2024-03\t0.32\t0.29\t0.1
2024-05\t0.2\t0.1\t-0.02
2024-06\t0.3\t0.27\t0.14
2024-07\t0.25\t0.2\t0.07
2024-09\t0.3\t0.4\t0.24
2024-11\t0.3\t0.25\t0.15
2024-12\t0.3\t0.32\t0.22
2025-01\t0.3\t0.25\t0.1
2025-03\t0.22\t0.25\t0.16
2025-05\t0.15\t0.1\t-0.05
2025-06\t0.3\t0.22\t0.1
2025-07\t-0.15\t-0.08\t-0.12
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

# 1. Read data
df_policy = pd.read_csv(StringIO(policy_text), sep='\t')
# Fix column names
df_policy.columns = ['date', 'statement', 'p_b', 'policy']

df_cpi = pd.read_csv(StringIO(cpi_text), sep='\t')

# 2. Prepare keys for joining
# For CPI keep original meeting date; derive ym and occurrence within month (for months with multiple meetings).
df_cpi['FOMC_date'] = pd.to_datetime(df_cpi['FOMC_date'])
df_cpi['ym'] = df_cpi['FOMC_date'].dt.strftime('%Y-%m')
df_cpi['occ'] = df_cpi.groupby('ym').cumcount()

# Policy dates only have year-month; create synthetic datetime (1st of month) and add occurrence
df_policy['ym'] = df_policy['date']
df_policy['occ'] = df_policy.groupby('ym').cumcount()
df_policy['date_dt'] = pd.to_datetime(df_policy['ym'] + '-01')

# 3. Merge on (ym, occ)
df = pd.merge(df_policy, df_cpi, on=['ym', 'occ'], how='inner')

# 4. Create year for filtering
df['year'] = df['FOMC_date'].dt.year

# 5. Helper to compute correlations
from typing import Dict, List, Tuple

def compute_correlations(sub: pd.DataFrame,
                         sentiment_col: str,
                         cpi_col: str) -> Dict[str, float]:
    # Drop rows with NA in required columns
    data = sub[[sentiment_col, cpi_col]].dropna()
    if len(data) < 3:
        return {
            'pearson_r': float('nan'), 'pearson_p': float('nan'),
            'spearman_rho': float('nan'), 'spearman_p': float('nan'),
            'kendall_tau': float('nan'), 'kendall_p': float('nan'),
            'n': len(data)
        }
    x = data[sentiment_col]
    y = data[cpi_col]
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    kt, kp = kendalltau(x, y)
    return {
        'pearson_r': pr, 'pearson_p': pp,
        'spearman_rho': sr, 'spearman_p': sp,
        'kendall_tau': kt, 'kendall_p': kp,
        'n': len(data)
    }

# 6. Define configurations
sentiment_cols = [
    ('statement', 'Statement'),
    ('p_b', 'P&B'),
    ('policy', 'Policy')
]
cpi_types = [
    ('Prev_CPI_Value', 'prev'),
    ('Next_CPI_Value', 'next')
]
periods = [
    ((2017, 2022), '2017-2022'),
    ((2023, 2025), '2023-2025')
]

rows: List[Dict[str, float]] = []

for (start_end, period_label) in periods:
    start, end = start_end
    sub_period = df[(df['year'] >= start) & (df['year'] <= end)]
    for sentiment_col, sentiment_label in sentiment_cols:
        for cpi_col, cpi_label in cpi_types:
            stats = compute_correlations(sub_period, sentiment_col, cpi_col)
            row = {
                'period': period_label,
                'sentiment_type': sentiment_label,
                'CPI_type': cpi_label,
                **stats
            }
            rows.append(row)

result_df = pd.DataFrame(rows, columns=[
    'period', 'sentiment_type', 'CPI_type',
    'n',
    'pearson_r', 'pearson_p',
    'spearman_rho', 'spearman_p',
    'kendall_tau', 'kendall_p'
])

# 7. Save to Excel
output_file = 'correlation_results.xlsx'
result_df.to_excel(output_file, index=False)

print("Correlation results:")
print(result_df)
print(f"\nSaved to {output_file}")