import pandas as pd
import scipy.stats as stats

# =====================
# 1. 파일 경로
# =====================
file_path = r"C:\Users\HUFS_MATH\Desktop\LGvsGPT_pol.xlsx"
out_file = "normality_results.xlsx"

# =====================
# 2. 데이터 불러오기
# =====================
df = pd.read_excel(file_path)

# =====================
# 3. 별표 함수
# =====================
def star(p):
    if p <= 0.05:
        return "**"
    elif p <= 0.10:
        return "*"
    else:
        return ""

results = []

# =====================
# 4. 컬럼별 검정
# =====================
for col in df.columns:

    data = df[col].dropna()

    # Shapiro-Wilk
    sh_stat, sh_p = stats.shapiro(data)

    # Jarque-Bera
    jb_stat, jb_p = stats.jarque_bera(data)

    # 별표
    sh_star = star(sh_p)
    jb_star = star(jb_p)

    # Overall 판정 (둘 중 하나라도 reject면 Non-normal)
    if sh_p > 0.05 and jb_p > 0.05:
        overall = "Normal"
    else:
        overall = "Non-normal"

    results.append({
        "Variable": col,

        "Shapiro-Stat": f"{sh_stat:.3f}{sh_star} ({sh_p:.3f})",
        "JB-Stat": f"{jb_stat:.3f}{jb_star} ({jb_p:.3f})",

        "Overall": overall
    })

# =====================
# 5. 저장
# =====================
result_df = pd.DataFrame(results)

result_df.to_excel(out_file, index=False)

print("완료!")
print(result_df)
print(f"\n저장 위치: {out_file}")