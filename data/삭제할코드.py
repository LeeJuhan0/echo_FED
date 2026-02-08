import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
df = pd.read_excel("nfci_anfci_monthly.xlsx")
df["year"] = pd.to_datetime(df["date"]).dt.year

# 2. 요약 통계 계산
summary = df.groupby("year").agg(
    nfci_mean=("nfci", "mean"),
    nfci_se=("nfci", lambda x: x.std() / np.sqrt(len(x))),
    anfci_mean=("anfci", "mean"),
    anfci_se=("anfci", lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

# 3. 그림 생성
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)

# ------------------
# NFCI
# ------------------
axes[0].scatter(df["year"], df["nfci"], color="black", s=10, alpha=0.7)
axes[0].plot(summary["year"], summary["nfci_mean"], color="royalblue")
axes[0].fill_between(
    summary["year"],
    summary["nfci_mean"] - summary["nfci_se"],
    summary["nfci_mean"] + summary["nfci_se"],
    color="royalblue", alpha=0.25
)
axes[0].set_title("NFCI Over Time")
axes[0].set_ylabel("NFCI")

# ------------------
# ANFCI
# ------------------
axes[1].scatter(df["year"], df["anfci"], color="black", s=10, alpha=0.7)
axes[1].plot(summary["year"], summary["anfci_mean"], color="royalblue")
axes[1].fill_between(
    summary["year"],
    summary["anfci_mean"] - summary["anfci_se"],
    summary["anfci_mean"] + summary["anfci_se"],
    color="royalblue", alpha=0.25
)
axes[1].set_title("ANFCI Over Time")
axes[1].set_ylabel("ANFCI")

plt.tight_layout()
plt.show()
