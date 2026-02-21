import os
import sys
import glob
import argparse
import pandas as pd
from pathlib import Path
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import Config
from src.external.camel.storages import Neo4jGraph

# Preprocess & Ingestion
from src.ingestion import generate_para

# Graph Construction
from src.graph.builder import (
    build_theory_graph_from_files,
    build_statement_graph_from_files
)

# Graph Linking
from src.graph.link import run_linking_process


sys.path.append(os.path.join(project_root, "src", "utils"))
from src.utils.common import get_response, extract_pdf_to_dataframe # 예시 경로
from src.analysis.pre_processing.sim_score_per import get_column_top_percent_values

# --- Helper Functions (유지) ---
def find_monetary_statement_csv(year_month: str, base_dir: str) -> str:
    """
    Find the statement CSV file containing an 8-digit number that starts with year_month
    inside the corresponding YYYY folder.
    """
    # year_month(YYYYMM)에서 앞 4자리(YYYY) 추출하여 연도 폴더 경로 생성
    yyyy = year_month[:4]
    year_dir = os.path.join(base_dir, yyyy)

    if not os.path.exists(year_dir):
        raise FileNotFoundError(f"Year directory not found: {year_dir}")

    all_csvs = glob.glob(os.path.join(year_dir, "*.csv"))

    for filepath in all_csvs:
        filename = os.path.basename(filepath)
        # 첫 번째로 등장하는 연속된 8자리 숫자 찾기
        match = re.search(r'\d{8}', filename)
        if match:
            extracted_yyyymm = match.group(0)[:6]
            if extracted_yyyymm == year_month:
                return filepath

    raise FileNotFoundError(f"No statement CSV found for year_month: {year_month} in {year_dir}")
# --- PDF → CSV Extraction Step ---

def run_statement_extraction():

    print(" Starting Statement PDF Extraction...")

    base_dir = Path(Config.RAW_STATEMENT_DIR)   # .../FOMC_Statement
    output_dir = Path(Config.DIR_statement_prompt)

    output_dir.mkdir(parents=True, exist_ok=True)

    year_dirs = sorted([
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    ])


    if not year_dirs:
        print("️ No year directories found.")
        return


    for year_dir in year_dirs:

        print(f"\n Year: {year_dir.name}")

        pdf_files = list(year_dir.glob("*.pdf"))

        if not pdf_files:
            print("  No PDF files in this folder.")
            continue


        for pdf_path in pdf_files:

            try:
                print(f"   Processing: {pdf_path.name}")

                df = extract_pdf_to_dataframe(pdf_path)

                if df.empty:
                    print(f" No text: {pdf_path.name}")
                    continue


                #  연도별로 출력 폴더도 나눌 경우
                year_out = output_dir / year_dir.name
                year_out.mkdir(exist_ok=True)


                out_file = year_out / f"{pdf_path.stem}.csv"


                df.to_csv(
                    out_file,
                    index=False,
                    encoding="utf-8-sig"
                )

                print(f" Saved: {out_file}")


            except Exception as e:

                print(f" Error {pdf_path.name}: {e}")


def load_high(datapath):
    """
    Read the entire file at datapath and return its contents as a single string.
    """
    all_content = ""
    with open(datapath, 'r', encoding='utf-8') as file:
        for line in file:
            all_content += line.strip() + "\n"
    return all_content

# --- Step 4: Inference Simulation Logic (기존 Main 로직 이식) ---

def run_inference_simulation(args):
    print("Starting Inference Simulation")

    input_path = args.sim_excel_path
    simscore_csv_path = args.sim_list_csv
    base_dir = args.output_dir
    report_sum_dir = args.report_sum_dir

    print(f"Connecting to Neo4j ({Config.NEO4J_URL})")
    n4j = Neo4jGraph(
        url=Config.NEO4J_URL,
        username=Config.NEO4J_USERNAME,
        password=Config.NEO4J_PASSWORD
    )

    tasks = []

    # 문자열 비교를 위해 None일 경우를 대비한 기본값 설정
    start_val = args.start_date if args.start_date else "000000"
    end_val = args.end_date if args.end_date else "999999"

    # Theory 단계(파라미터 적용) 활성화 여부에 따라 작업 목록(tasks) 구성
    if args.use_theory:
        print("Theory mode is ON. Loading parameters from CSV")
        if not os.path.exists(simscore_csv_path):
            print(f"Error: Simulation CSV not found at {simscore_csv_path}")
            return

        df_params = pd.read_csv(simscore_csv_path, dtype={"gid": str})
        print(f"Loaded {len(df_params)} potential tasks from CSV.")

        for _, row in df_params.iterrows():
            gid = str(row["gid"])
            if not gid.startswith("FOMC"): continue

            yyyymm = gid.replace("FOMC", "")

            # 지정된 범위(start_date ~ end_date) 내의 데이터만 필터링
            if not (start_val <= yyyymm <= end_val):
                continue

            gid_index = int(row["gid_index"])
            simscore_percentage = float(row["simscore_percentage"])
            round_adj_token = int(row["round_adj_token"])

            tasks.append({
                "gid": gid, "yyyymm": yyyymm,
                "gid_index": gid_index, "simscore_percentage": simscore_percentage,
                "round_adj_token": round_adj_token, "is_theory": True
            })

    else:
        print("⚡ Theory mode is OFF. Scanning year directories for available months...")
        if not args.start_date or not args.end_date:
            print("Error: --start_date and --end_date are required when --use_theory is omitted.")
            return

        # start_date와 end_date에서 연도(YYYY)만 정수로 추출
        start_year = int(start_val[:4])
        end_year = int(end_val[:4])

        valid_yyyymm_set = set()

        # start_year부터 end_year까지 연도별 폴더를 순회
        for year in range(start_year, end_year + 1):
            year_dir = os.path.join(report_sum_dir, str(year))

            # 해당 연도 폴더가 없으면 건너뜀
            if not os.path.exists(year_dir):
                continue

            # 연도 폴더 안의 모든 csv 파일 스캔
            search_pattern = os.path.join(year_dir, "*.csv")
            found_files = glob.glob(search_pattern)

            for filepath in found_files:
                filename = os.path.basename(filepath)

                # 첫 번째로 등장하는 연속된 8자리 숫자 찾기 (예: FOMC_20160127.csv -> 20160127)
                match = re.search(r'\d{8}', filename)
                if match:
                    # 8자리 중 앞 6자리를 YYYYMM으로 사용
                    yyyymm = match.group(0)[:6]

                    # 시작/종료 연월 범위 내에 있는지 확인
                    if start_val <= yyyymm <= end_val:
                        valid_yyyymm_set.add(yyyymm)

        if not valid_yyyymm_set:
            print(f"No matching statement files found in '{report_sum_dir}' for the range {start_val} ~ {end_val}.")
            return

        # 추출된 실제 존재하는 YYYYMM만 task로 추가
        for yyyymm in sorted(valid_yyyymm_set):
            tasks.append({
                "gid": f"FOMC{yyyymm}", "yyyymm": yyyymm,
                "gid_index": 0, "simscore_percentage": 0.0,
                "round_adj_token": 0, "is_theory": False
            })

    if not tasks:
        print("No tasks to process. Check your date ranges or CSV data.")
        return

    # 2. 구성된 작업 목록 순차 실행
    print(f"\n Total tasks to process: {len(tasks)}")
    for idx, task in enumerate(tasks):
        try:
            gid = task["gid"]
            yyyymm = task["yyyymm"]
            yyyy = yyyymm[:4]
            mm = yyyymm[4:6]
            year_numeric = int(yyyymm)

            # Theory가 켜져있을 경우에만 sim_score_median 계산
            sim_score_median = 0.0
            if task["is_theory"]:
                sim_score_median_list = get_column_top_percent_values(input_path, task["simscore_percentage"])
                if 0 <= task["gid_index"] < len(sim_score_median_list):
                    sim_score_median = sim_score_median_list[task["gid_index"]]

            print(f"\n[{idx+1}/{len(tasks)}] Processing {gid} (Theory={'ON' if task['is_theory'] else 'OFF'})")

            # 질문 로드 (Monetary Statement)
            try:
                question_csv_path = find_monetary_statement_csv(yyyymm, base_dir=report_sum_dir)
                question = load_high(question_csv_path)
            except FileNotFoundError as e:
                print(f"{e}. Skipping this month.")
                continue

            # 결과 데이터 구조
            data = [["Statement", "Papers", "FSR", "SLOOS", "beigebook"]]

            # 5회 반복 추론
            for i in range(5):
                print(f" Run {i+1}/5")
                response = get_response(n4j, gid, question, task["gid_index"], sim_score_median, year_numeric, meeting_no=1)
                data.append(response)

            # 결과 저장
            out_df = pd.DataFrame(data[1:], columns=data[0])

            # 저장 경로 생성
            year_dir = os.path.join(base_dir, yyyy)
            month_dir = os.path.join(year_dir, mm)
            os.makedirs(month_dir, exist_ok=True)

            # 파일명 분기 처리 (Theory 적용 유무에 따라 다름)
            if task["is_theory"]:
                out_filename = f"{gid}_({task['gid_index']},{task['round_adj_token']},{task['simscore_percentage']}%).csv"
            else:
                out_filename = f"{gid}_basic_inference.csv"

            out_path = os.path.join(month_dir, out_filename)
            out_df.to_csv(out_path, index=False, encoding="utf-8")

            print(f"Saved to: {out_path}")

        except Exception as e:
            print(f"Error processing task {idx+1} ({task['gid']}): {e}")

# --- Main Pipeline Controller ---

def main():
    parser = argparse.ArgumentParser(description="FOMC GraphRAG Integrated Pipeline")

    # [Step Selection]
    parser.add_argument('--step', type=str, required=True,
                        choices=['preprocess', 'construct_theory','construct_statement', 'link', 'inference', 'extract_statement'],
                        help="Select the pipeline step to run.")
    # Construct Step Args
    parser.add_argument('--grained_chunk', default= False, action='store_true',
                        help="Enable Agentic Chunking (finer granularity) during graph construction.")

    # --- Preprocess Args ---
    parser.add_argument('--input_file', type=str, help="Input text file for preprocessing")

    # --- Link Args ---
    parser.add_argument('--fomc_gid', type=str, help="Target FOMC GID for linking")
    parser.add_argument('--check_fsr', type=str, help="FSR GID for linking")
    parser.add_argument('--check_sloos', type=str, help="SLOOS GID for linking")
    parser.add_argument('--check_beigebook', type=str, help="BeigeBook GID for linking")

    # --- Inference Args ---
    parser.add_argument('--sim_excel_path', type=str,
                        default=Config.DATA_SIMSCORE_EXCEL,
                        help="Path to Similarity Score Excel")

    parser.add_argument('--sim_list_csv', type=str,
                        default=Config.DATA_SIM_LIST_CSV,
                        help="Path to Simulation List CSV")

    parser.add_argument('--output_dir', type=str,
                        default=Config.DIR_OUTPUT,
                        help="Base output directory for inference results")

    parser.add_argument('--report_sum_dir', type=str,
                        default=Config.DIR_statement_prompt,
                        help="Directory containing monetary statement CSVs")

    parser.add_argument('--use_theory', action='store_true',
                        help="[Inference] Theory(CSV 파라미터) 단계를 적용하려면 추가하세요.")
    parser.add_argument('--start_date', type=str,
                        help="[Inference] 시작 연월 (형식: YYYYMM, 예: 202101)")
    parser.add_argument('--end_date', type=str,
                        help="[Inference] 종료 연월 (형식: YYYYMM, 예: 202312)")

    args = parser.parse_args()

    # --- Router ---
    if args.step == 'preprocess':
        if not args.input_file:
            print(" Error: --input_file(Theory) is required for preprocessing.")
            return
        generate_para(input_path=args.input_file)

    elif args.step == 'construct_theory':

        print(" Constructing THEORY Graph...")

        build_theory_graph_from_files(

            data_path=Config.DATA_THEORY_PATH,

            excel_path=Config.DATA_SIMSCORE_EXCEL,

            grained_chunk=args.grained_chunk
        )


    elif args.step == 'construct_statement':

        print(" Constructing STATEMENT Graph...")

        build_statement_graph_from_files(

            base_path= Config.DIR_statement_prompt,

            grained_chunk=args.grained_chunk
        )


    elif args.step == 'link':
            if not args.fomc_gid:
                print(" Error: --fomc_gid is required for linking.")
                return
            run_linking_process(
                fomc_gid=args.fomc_gid,
                check_fsr=args.check_fsr,
                check_sloos=args.check_sloos,
                check_beigebook=args.check_beigebook
            )

    elif args.step == 'inference':
        run_inference_simulation(args)

    elif args.step == 'extract_statement':
        run_statement_extraction()

if __name__ == "__main__":
    main()