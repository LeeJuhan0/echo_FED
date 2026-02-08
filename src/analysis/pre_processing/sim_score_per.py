from openpyxl import load_workbook
import numpy as np

def get_column_top_percent_values(file_path, percentage):
    """
    모든 시트를 순회하면서 B열부터 J열의 각 열별 상위 퍼센티지 값을 리스트에 저장.

    :param file_path: 엑셀 파일 경로
    :param percentage: 상위 퍼센티지 (default=10)
    :return: 각 시트별 열별 상위 퍼센티지 값 딕셔너리
    """
    workbook = load_workbook(file_path)
    result = {}
    thresholdlist = []
    # 모든 시트를 순회
    for sheet_name in workbook.sheetnames[:2]:
        sheet = workbook[sheet_name]
        # B열부터 J열 순회 (열: 2부터 10)

        for col in range(2, 72):  # B열(2)부터 J열(10)까지
            column_data = []

            # 해당 열의 데이터 수집 (행: 2부터 1372)
            for row in sheet.iter_rows(min_row=3, max_row=2043, min_col=col, max_col=col, values_only=True):
                value = row[0]
                if isinstance(value, float):  # float형 데이터만 추가
                    column_data.append(value)

            # 상위 퍼센티지 값 계산
            if column_data:
                column_data = np.array(column_data)
                threshold = float(np.percentile(column_data, 100 - percentage))  # 상위 퍼센티지 기준값
                thresholdlist.append(threshold)

    workbook.close()
    return thresholdlist


# 사용 예제
file_path = "C:/Users/HUFS_MATH/IdeaProjects/FOMC_Graphrag/excel_File/similarity_merge_results_merged_statics_10_.xlsx"
percentage = 8  # 상위 n% 값

#print(get_column_top_percent_values("C:/Users/HUFS_MATH/IdeaProjects/FOMC_Graphrag/excel_File/similarity_merge_results_merged_statics_10_.xlsx", 8))
