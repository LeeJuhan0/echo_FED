import pandas as pd
from scipy.stats import ttest_rel

# 파일 경로
file_path = r"C:\Users\HUFS_MATH\Desktop\LGvsGPT_pol.xlsx"
df = pd.read_excel(file_path)

metrics = ["llm", "ST", "Policy"]

results = []

for m in metrics:

    lg_col = f"LG_{m}"
    gpt_col = f"wo_{m}"

    # 컬럼 존재 확인
    if lg_col not in df.columns or gpt_col not in df.columns:
        print(f"⚠️ 컬럼 없음: {lg_col} or {gpt_col}")
        continue


    # ✅ paired 기준으로 결측 제거
    pair = df[[lg_col, gpt_col]].dropna()

    x = pair[lg_col]
    y = pair[gpt_col]


    # 표본 너무 적으면 skip
    if len(pair) < 3:
        print(f"⚠️ 표본 부족: {m}")
        continue


    # paired t-test
    t, p = ttest_rel(x, y)


    results.append({
        "Metric": m,
        "LG_column": lg_col,
        "GPT_column": gpt_col,
        "N": len(pair),
        "T_value": t,
        "P_value": p,
        "Result": f"T={t:.3f} (p={p:.3f})"
    })


result_df = pd.DataFrame(results)

# 저장
out_file = "ttest_LG_vs_GPT_by_metric_fixed.xlsx"
result_df.to_excel(out_file, index=False)

print(result_df)
print(f"\n✅ 저장 완료: {out_file}")