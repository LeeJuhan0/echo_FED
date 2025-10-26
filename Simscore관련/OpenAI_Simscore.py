import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


# OpenAI API 키를 OS 환경 변수에서 가져오기
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수 "OPENAI_API_KEY"에서 가져오기

if api_key is None:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)
# 폴더 경로 설정
folder_paths = [
    r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Theory\alldata"
]
# 기준 파일 경로
reference_file_path = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Statement\FOMC_ex\monetary20250618a1.csv"

# 기준 파일 로드
try:
    reference_data = pd.read_csv(reference_file_path, on_bad_lines='skip')
    reference_text = reference_data.astype(str).apply(lambda x: ' '.join(x), axis=1).str.cat(sep=' ').lower()
except Exception as e:
    print(f"Error reading reference file: {e}")
    exit()

# OpenAI API를 사용해 텍스트 임베딩 생성 함수
def get_embedding(client, text, model="text-embedding-ada-002"):
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# 기준 텍스트의 임베딩 생성
reference_embedding = get_embedding(client, reference_text)
if reference_embedding is None:
    print("Failed to generate embedding for the reference file. Exiting...")
    exit()

# 결과 저장 리스트
similarity_results = []

# 각 폴더 순회
for folder_path in folder_paths:
    folder_name = os.path.basename(folder_path)  # 폴더 이름 추출
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            try:
                # 파일 로드
                data = pd.read_csv(file_path, on_bad_lines='skip')
                text_data = data.astype(str).apply(lambda x: ' '.join(x), axis=1).str.cat(sep=' ').lower()

                # 비교 파일의 임베딩 생성
                file_embedding = get_embedding(client, text_data)
                if file_embedding is None:
                    print(f"Failed to generate embedding for {file_name}. Skipping...")
                    continue

                # 코사인 유사도 계산
                similarity = cosine_similarity([reference_embedding], [file_embedding])[0][0]

                # 결과 저장
                similarity_results.append((f"{folder_name}_{file_name}", similarity))
            except Exception as e:
                print(f"Error reading {file_name} in folder {folder_name}: {e}")

# 유사도 기준으로 정렬
sorted_results = sorted(similarity_results, key=lambda x: x[1], reverse=True)

# 결과 출력
print("Similarity results:")
for file_name, similarity in sorted_results:
    print(f"{file_name}: {similarity:.4f}")

# 결과를 DataFrame으로 변환
df = pd.DataFrame(sorted_results, columns=["File Name", "Similarity Score"])

# 엑셀 파일로 저장
output_excel_path = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\wwwww1.xlsx"
df.to_excel(output_excel_path, index=False)

print(f"Similarity results have been saved to {output_excel_path}")