from scipy.stats import qmc
import numpy as np

d = 2
N = 2**(7)  # 필요 갯수
sampler = qmc.Sobol(d=d, scramble=True, seed=42)

# 방법 1: 임의 N개 (임의 N 허용)
X = sampler.random(N)               # shape: (N, 2), in [0,1)

# 방법 2: 2^m개가 필요할 때 (그레이코드 네트)
#X_pow2 = sampler.random_base2(m=8) # 2^10 = 1024개

# 처음 몇 점 건너뛰기
sampler.fast_forward(100)           # 다음 호출부터 100개 지나친 지점에서 시작
Y = sampler.random(N)
x = Y.copy()
x[:, 0] *= 69
x[:, 1] *= 10
print(np.round(x).astype(int))
mask_round = (np.round(x[:, 0]).astype(int) == 5)
idx_round = np.where(mask_round)[0]
rows_round = x[mask_round]
print(idx_round)
print(rows_round)
print(np.round(x[mask_round,0]))
print(np.round(x[1]))
import numpy as np

# x: shape (128, 2)
# 두 번째 열을 정수로 반올림해서 그룹 키 생성
y_round = np.rint(x[:, 1]).astype(int)

# 값별 개수 집계
vals, counts = np.unique(y_round, return_counts=True)

# 출력
print("y(rounded) : count")
for v, c in zip(vals, counts):
    print(f"{v:>3} : {c}")





import matplotlib.pyplot as plt

rounded = np.round(x).astype(int)
for i in range(len(x)):
    plt.text(x[i, 0] + 0.15, x[i, 1] + 0.15, f'({rounded[i,0]},{rounded[i,1]})', fontsize=8)

# x: shape (n, 2) 라고 가정
plt.figure(figsize=(5, 7))
plt.scatter(x[:, 1], x[:, 0], c='tab:blue', s=50, edgecolor='k', alpha=0.85)
plt.xlabel('x[:, 0] (×11)')
plt.ylabel('x[:, 1] (×70)')
plt.title('Scatter of x (Sobol samples scaled)')
plt.xlim(0, 10)
plt.ylim(0, 69)
plt.grid(True, ls=':', alpha=0.5)
plt.tight_layout()
plt.show()