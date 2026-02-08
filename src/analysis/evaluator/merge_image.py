import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path

# -------------------------------------------------
# 이미지 경로
# -------------------------------------------------
FIG1_PATH = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data_plot\histogram2x2.png")
FIG2_PATH = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data_plot\year_anfci_cpidiff_policy.png")

OUT_FIG = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data_plot\replace_22_with_figure2.png")

# -------------------------------------------------
# 이미지 로드
# -------------------------------------------------
img1 = mpimg.imread(FIG1_PATH)
img2 = mpimg.imread(FIG2_PATH)

# -------------------------------------------------
# histogram(그림1) 분할 (2x2 기준)
# -------------------------------------------------
H, W = img1.shape[:2]
h2, w2 = H // 2, W // 2

img11 = img1[0:h2, 0:w2]     # (1,1)
img12 = img1[0:h2, w2:W]     # (1,2)
img21 = img1[h2:H, 0:w2]     # (2,1)
# img22 = img1[h2:H, w2:W]   # (2,2) → 사용 안 함

# -------------------------------------------------
# Figure 구성
# -------------------------------------------------
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2)

# (1,1) histogram - ANFCI
ax11 = fig.add_subplot(gs[0, 0])
ax11.imshow(img11)
ax11.axis("off")

# (1,2) histogram - CPI diff
ax12 = fig.add_subplot(gs[0, 1])
ax12.imshow(img12)
ax12.axis("off")

# (2,1) histogram - sentiment
ax21 = fig.add_subplot(gs[1, 0])
ax21.imshow(img21)
ax21.axis("off")

# (2,2) ← panel figure로 교체
ax22 = fig.add_subplot(gs[1, 1])
ax22.imshow(img2)
ax22.axis("off")

# -------------------------------------------------
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved combined figure to: {OUT_FIG}")
