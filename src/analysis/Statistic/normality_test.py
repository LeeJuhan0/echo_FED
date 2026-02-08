import pandas as pd
from scipy.stats import shapiro
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# 1. 엑셀 파일 불러오기
df = pd.read_excel(r"C:\Users\HUFS_MATH\Desktop\13+19파일\sentiment_13+19_llm.xlsx")

# 2. 결과를 저장할 리스트 생성
results = []

# 3. QQ 플롯 저장 폴더 생성
save_dir = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\정규성가정\qqplots"
os.makedirs(save_dir, exist_ok=True)

# 4. 앞 5개 열을 대상으로 Shapiro-Wilk 검사 + Q-Q Plot 생성
for col in df.columns[:6]:
    values = df[col].dropna().tolist()

    # Shapiro-Wilk Test
    stat, p_value = shapiro(values)
    normality = "정규성 만족" if p_value > 0.05 else "정규성 불만족"

    # 결과 저장
    results.append({
        "Column": col,
        "Shapiro_Statistic": stat,
        "p_value": p_value,
        "Normality": normality
    })

    # ============================
    #      Q-Q Plot 생성 부분
    # ============================
    plt.figure(figsize=(6, 6))
    stats.probplot(values, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot for {col}")

    # 파일명: "qqplot_ColumnName.png"
    plot_path = os.path.join(save_dir, f"qqplot_{col}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

# 5. 결과를 데이터프레임으로 변환
result_df = pd.DataFrame(results)

# 6. 엑셀 파일로 저장
output_path = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\정규성가정\shapiro_results.xlsx"
result_df.to_excel(output_path, index=False)

print(f"정규성 검정 결과 저장 완료 → {output_path}")
print(f"Q-Q plots saved at → {save_dir}")