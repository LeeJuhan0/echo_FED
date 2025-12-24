import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

# 파일 경로
sentiment_path = r'C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\sentiment_13000.csv'
llm_path = r'C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\sentiment_19000.csv'

# sentiment: date, col2, col3, col4 (예시: sentiment1, sentiment2, sentiment3)
sentiment = pd.read_csv(sentiment_path)
llm = pd.read_csv(llm_path)
print(llm)
print(sentiment)
# sentiment 변수명 지정 (2~4번째 열)
sentiment_vars = sentiment.columns[1:6]
llm_var = llm.columns[1]  # 첫번째 변수, date가 0번째일 경우

# date로 병합
merged = sentiment[['date'] + list(sentiment_vars)].merge(llm[['date', llm_var]], on='date')
print(merged)

# 변수 배열 정의 (llm 먼저, sentiment 3개)
variables = [llm_var] + list(sentiment_vars)

# 결과 저장용 (4x4 테이블)
results = np.empty((6, 6), dtype=object)

for i in range(6):
    for j in range(6):
        x = merged[variables[i]].values
        y = merged[variables[j]].values
        # t-test (독립표본, 비등분산)
        tval, pval = ttest_ind(x, y, equal_var=False)
        results[i, j] = f"T={tval:.3f}, P={pval:.3f}"

# DataFrame으로 변환
result_df = pd.DataFrame(results, columns=variables, index=variables)
print(result_df)

# 결과를 같은 경로에 저장
result_df.to_csv(r'C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\data\ttest_table13,19비교.csv')