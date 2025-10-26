import pandas as pd
import os
from glob import glob

def process_all_csv(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)


    csv_files = glob(os.path.join(input_folder, "*.csv"))

    for file_path in csv_files:
        # CSV 파일 읽기
        df = pd.read_csv(file_path)
        df.columns = [col.strip() for col in df.columns]  # 열 이름 공백 제거

        # G열: Sum = NodeCount + RelCount
        df["Sum"] = df["NodeCount"] + df["RelCount"]

        # H열: Change_rate = max(Gn - G(n-1), 0)
        df["Change_rate"] = df["Sum"].diff().apply(lambda x: max(x, 0))
        df.loc[df.index[0], "Change_rate"] = 0  # 첫 번째 행은 0

        # 새로운 파일명 생성 후 output 폴더에 저장
        base_name = os.path.basename(file_path)
        new_file = os.path.join(output_folder, base_name.replace(".csv", "updated.csv"))

        df.to_csv(new_file, index=False, encoding="utf-8-sig")
        print(f"✅ 처리 완료: {new_file}")

# === 실행 예시 ===
input_folder = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simscore\results_1"
output_folder = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simscore\results_2"

process_all_csv(input_folder, output_folder)