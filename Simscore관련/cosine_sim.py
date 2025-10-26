import os
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openpyxl import Workbook

# Input과 Output 폴더 경로 설정
#pdf_input_path = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\excel_File\pdf"
csv_output_path = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Theory\alldata"
report_folder = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Statement\report_example"
result_excel_path = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\TF_IDF_SimsCore.xlsx"
"""
# Output 폴더가 없으면 생성
os.makedirs(csv_output_path, exist_ok=True)

# Step 1: PDF 파일을 페이지별로 나누어 CSV로 저장
for file_name in os.listdir(pdf_input_path):
    if file_name.endswith(".pdf"):  # PDF 파일만 처리
        file_path = os.path.join(pdf_input_path, file_name)
        pdf_reader = PdfReader(file_path)

        # 각 페이지를 읽어 CSV로 저장
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            try:
                # 페이지 텍스트 추출
                page_text = page.extract_text()

                # 텍스트를 pandas DataFrame으로 변환
                data = {"Content": [line for line in page_text.split("\n")]}
                df = pd.DataFrame(data)

                # CSV 파일 이름 생성
                output_file_name = f"{os.path.splitext(file_name)[0]}_page{page_num}.csv"
                output_file_path = os.path.join(csv_output_path, output_file_name)

                # CSV 파일 저장
                df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
                print(f"Saved: {output_file_path}")
            except Exception as e:
                print(f"Error processing page {page_num} in {file_name}: {e}")
"""
# Step 2: Report 파일과 각 페이지 CSV 비교 및 유사도 계산
# 결과를 Excel로 저장하기 위한 Workbook 생성

wb = Workbook()
wb.remove(wb.active)  # 기본 시트 제거

# Report 파일 순회
for report_file_name in os.listdir(report_folder):
    if report_file_name.endswith(".csv"):
        report_file_path = os.path.join(report_folder, report_file_name)

        # 기준 파일 로드
        try:
            report_data = pd.read_csv(report_file_path, on_bad_lines='skip')
            report_text = report_data.astype(str).apply(lambda x: ' '.join(x), axis=1).str.cat(sep=' ').lower()
            print(report_text)
        except Exception as e:
            print(f"Error reading report file {report_file_name}: {e}")
            continue

        # 결과 저장 리스트
        similarity_results = []

        # Dataset 폴더의 모든 CSV 파일과 비교
        for dataset_file_name in os.listdir(csv_output_path):
            if dataset_file_name.endswith('.csv'):
                dataset_file_path = os.path.join(csv_output_path, dataset_file_name)
                try:
                    # 파일 로드
                    dataset_data = pd.read_csv(dataset_file_path, on_bad_lines='skip')
                    dataset_text = dataset_data.astype(str).apply(lambda x: ' '.join(x), axis=1).str.cat(sep=' ').lower()
                    print(dataset_text)

                    # TF-IDF 벡터화
                    vectorizer = TfidfVectorizer()
                    vectors = vectorizer.fit_transform([report_text, dataset_text])

                    # 코사인 유사도 계산
                    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

                    # 결과 저장
                    similarity_results.append((dataset_file_name, similarity))
                except Exception as e:
                    print(f"Error reading {dataset_file_name}: {e}")

        # 유사도 기준으로 정렬
        sorted_results = sorted(similarity_results, key=lambda x: x[1], reverse=True)

        # DataFrame으로 변환
        df_results = pd.DataFrame(sorted_results, columns=["File Name", "Similarity Score"])

        # 결과를 새로운 시트에 추가
        sheet_name = os.path.splitext(report_file_name)[0]  # Report 파일 이름을 시트 이름으로 사용
        ws = wb.create_sheet(title=sheet_name)

        for r_idx, row in enumerate(df_results.itertuples(index=False), start=1):
            if r_idx == 1:
                ws.append(["File Name", "Similarity Score"])  # 헤더 추가
            ws.append(row)

        print(f"Similarity results for {report_file_name} added to sheet: {sheet_name}")

# Step 3: Excel 파일로 저장
wb.save(result_excel_path)
print(f"Final results have been saved to {result_excel_path}")
