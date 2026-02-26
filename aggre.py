import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# ==============================
# 1. 엑셀 파일 불러오기
# ==============================
file_path = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\results\experiment\simulation2014_16\Simulation_19+13_score_summary.xlsx"
df = pd.read_excel(file_path)

# 숫자형 컬럼만 사용 (날짜/문자 제외)
cols = df.select_dtypes(include=[np.number]).columns.tolist()
n = len(cols)

print("사용 컬럼:", cols)

# ==============================
# 2. 결과 테이블 생성
# ==============================
result = pd.DataFrame(
    np.full((n, n), "", dtype=object),
    index=cols,
    columns=cols
)

# ==============================
# 3. 쌍별 t-test (단측, 등분산)
# ==============================
for i in range(n):
    for j in range(n):

        if i == j:
            continue  # 대각선은 비움

        x = df[cols[i]].dropna()
        y = df[cols[j]].dropna()

        # 등분산 가정 두 표본 t-test (양측 기준)
        t_stat, p_two_sided = ttest_ind(x, y, equal_var=True)

        # 단측 p-value로 변환 (엑셀 TTEST(type=1)과 동일)
        if t_stat > 0:
            p_one_sided = p_two_sided / 2
        else:
            p_one_sided = 1 - (p_two_sided / 2)

        # 위: T값, 아래: 단측 p-value
        if i < j:
            result.iloc[i, j] = f"{t_stat:.3f}"
        else:
            result.iloc[i, j] = f"{p_one_sided:.3f}"

# ==============================
# 4. 엑셀로 저장
# ==============================
output_path = r"C:\Users\HUFS_MATH\Desktop\ttest_table.xlsx"
result.to_excel(output_path)

print("저장 완료:", output_path)
print(result)