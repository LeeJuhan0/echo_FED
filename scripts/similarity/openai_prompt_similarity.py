import os
import pandas as pd
import json
from openai import OpenAI

# ===============================
# 1. OpenAI client 세팅
# ===============================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===============================
# 2. Statement CSV → 하나의 문자열로 병합
# ===============================
def load_statement_as_one(file_path):
    df = pd.read_csv(file_path)
    merged_text = " ".join(df.iloc[:, 0].dropna().astype(str).tolist())
    return [merged_text]   # 리스트 형태로 반환

# ===============================
# 3. Paragraph Excel 로드
# ===============================
def load_paragraphs(file_path, sheet_name=0, col_idx=0):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    paragraphs = df.iloc[:, col_idx].dropna().astype(str).tolist()
    return paragraphs

# ===============================
# 4. 프롬프트 템플릿
# ===============================
PROMPT_TEMPLATE = """
You are comparing a Federal Open Market Committee (FOMC) statement with an academic/economic paragraph.  
Your task is to compute a **compound similarity score** that integrates three aspects:  

1. **Topical Similarity** – Do they discuss the same economic concepts (inflation, employment, credit, interest rates, financial stability, etc.)?  
2. **Explanatory Relevance** – Does the paragraph explain mechanisms, causes, or effects of the issues in the statement?  
3. **Contextual Alignment** – Does the paragraph provide theoretical or empirical support that aligns with the intent of the statement?  

Scoring rules:
- The score must be a decimal between 0.0 and 1.0 (with 5 decimal places).  
- Avoid giving only 0.0 scores. Instead, use the full scale based on strength of connection:  
  - 0.0 = completely unrelated, no overlap at all.  
  - 0.1 ~ 0.3 = weak similarity (only keywords overlap, no causal link).  
  - 0.4 ~ 0.6 = moderate similarity (partially relevant, some explanatory connection).  
  - 0.7 ~ 0.9 = strong similarity (clear explanation or strong theoretical support).  
  - 1.0 = perfect similarity (direct and comprehensive explanation of the statement).  

Return ONLY a JSON object in the following format:
{{
  "similarity_score": <decimal between 0.0 and 1.0 with 5 decimal places>
}}

FOMC Statement:
"{statement}"

Paper Paragraph:
"{paragraph}"
"""

# ===============================
# 5. 유사도 계산 함수
# ===============================
def get_similarity(statement, paragraph):
    prompt = PROMPT_TEMPLATE.format(statement=statement, paragraph=paragraph)

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # 또는 gpt-5
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        content = response.choices[0].message.content.strip()

        # 코드블록 제거
        if content.startswith("```"):
            content = content.strip("`").replace("json", "").strip()

        data = json.loads(content)
        score = round(float(data["similarity_score"]), 5)
    except Exception as e:
        print("⚠️ Parsing error:", e, "\nResponse:", content)
        score = 0.0

    return score

# ===============================
# 6. 메인 실행 함수
# ===============================
def build_similarity_matrix(fomc_folder, paragraph_file, output_file="similarity_matrix.xlsx"):
    # 문단 로드
    paper_paragraphs = load_paragraphs(paragraph_file)
    print(f"📌 Loaded {len(paper_paragraphs)} Paper Paragraphs")

    df_result = pd.DataFrame({"Paragraph": paper_paragraphs})

    # 폴더 내 모든 Statement CSV 파일 탐색
    fomc_files = [os.path.join(fomc_folder, f) for f in os.listdir(fomc_folder) if f.endswith(".csv")]
    print(f"📌 Found {len(fomc_files)} FOMC Statement CSV files")

    for file_path in fomc_files:
        statements = load_statement_as_one(file_path)  # 각 CSV → 1개 Statement
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        for idx, stmt in enumerate(statements, start=1):
            col_name = f"{base_name}_S{idx}"
            print(f"   → Processing {col_name}: {stmt[:60]}...")

            scores = []
            for i, para in enumerate(paper_paragraphs, start=1):
                score = get_similarity(stmt, para)
                scores.append(score)

                # 🔹 바로바로 결과 출력
                print(f"[{col_name}] Paragraph {i}/{len(paper_paragraphs)} → Score: {score}")

            df_result[col_name] = scores

    # 최종 저장
    df_result.to_excel(output_file, index=False)
    print(f"✅ 최종 저장 완료: {output_file}")
    # 최종 저장
    df_result.to_excel(output_file, index=False)
    print(f"✅ 저장 완료: {output_file}")


# ===============================
# 5. 실행 예시
# ===============================
if __name__ == "__main__":
    fomc_file = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Statement\FOMC_ex"
    paragraph_file = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simscore관련\combined_book_paragraphs.xlsx"  # 문단 파일
    output_file = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\wwwwww.xlsx"

    build_similarity_matrix(fomc_file, paragraph_file, output_file)