import pandas as pd
from scipy.stats import ttest_ind

file_path = r"C:\Users\HUFS_MATH\Desktop\LG.xlsx"

df = pd.read_excel(file_path)

cols = df.columns.tolist()
n = len(cols)

result = pd.DataFrame("", index=cols, columns=cols)

for i in range(n):
    for j in range(n):

        if i == j:
            result.iloc[i, j] = "-"
            continue

        # 결측 제거 (독립표본이므로 따로 dropna OK)
        x = df[cols[i]].dropna()
        y = df[cols[j]].dropna()

        # ✅ 독립표본 + 등분산
        t, p_two = ttest_ind(x, y, equal_var=True)

        # ✅ 단측 p-value (Excel type=1)
        p_one = p_two / 2


        if i < j:
            result.iloc[i, j] = f"T={t:.3f}"
        else:
            result.iloc[i, j] = f"{p_one:.3f}"


result.to_excel("pairwise_ttest_excel.xlsx")

print("완료!")
print(result)