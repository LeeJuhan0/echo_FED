import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# 파일 경로
file_path = r"C:\Users\HUFS_MATH\Desktop\LGvsGPT_pol.xlsx"
df = pd.read_excel(file_path)

# 👉 행 순서: LLM → ST → Policy
metric_order = ["llm", "ST", "Policy"]

# 제목용 이름 (원하면 수정 가능)
title_map = {
    "LG_llm": "LG_LLM",
    "wo_llm": "ChatGPT_LLM",

    "LG_ST": "LG_GRAG[St]",
    "wo_ST": "ChatGPT_GRAG[St]",

    "LG_Policy": "LG_GRAG[St+Policy]",
    "wo_Policy": "ChatGPT_GRAG[St+Policy]"
}

# 3x2 고정
fig, axes = plt.subplots(2, 3, figsize=(15, 8))


for i, m in enumerate(metric_order):

    lg_col = f"LG_{m}"
    wo_col = f"wo_{m}"

    # --------- LG (위) ---------
    if lg_col in df.columns:

        data = df[lg_col].dropna()
        stat, p = stats.shapiro(data)

        stats.probplot(data, plot=axes[0, i])

        name = title_map.get(lg_col, lg_col)

        axes[0, i].set_title(f"{name}\n(p = {p:.3f})", fontweight="bold")


    # --------- GPT (아래) ---------
    if wo_col in df.columns:

        data = df[wo_col].dropna()
        stat, p = stats.shapiro(data)

        stats.probplot(data, plot=axes[1, i])

        name = title_map.get(wo_col, wo_col)

        axes[1, i].set_title(f"{name}\n(p = {p:.3f})", fontweight="bold")


# 전체 제목 (선택)
fig.suptitle("Normality Test (QQ-Plot): LG vs GPT", fontsize=14, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()