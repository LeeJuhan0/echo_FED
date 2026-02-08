import os
import sys
import glob
import argparse
import pandas as pd
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import Config
from src.external.camel.storages import Neo4jGraph

# Preprocess & Ingestion
from src.ingestion import generate_para

# Graph Construction
from src.graph.builder import build_graph_from_files

# Graph Linking
from src.graph.link import run_linking_process


sys.path.append(os.path.join(project_root, "src", "utils"))
from src.utils.common import get_response # 예시 경로
from src.analysis.pre_processing.sim_score_per import get_column_top_percent_values

# --- Helper Functions (유지) ---

def find_monetary_statement_csv(year_month: str, base_dir: str) -> str:
    """
    Find the monetary statement CSV file (monetary{YYYYMM}*.csv).
    """
    # 경로는 Config 또는 인자에서 받도록 수정
    pattern = os.path.join(base_dir, f"monetary{year_month}*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No monetary statement CSV found for pattern: {pattern}")
    return matches[0]

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
    print("🚀 Starting Inference Simulation...")

    # 1. 경로 설정 (Args 또는 Config 사용)
    # 기존 코드의 하드코딩된 경로를 args default로 설정해두었으나, 여기서 오버라이딩 가능
    input_path = args.sim_excel_path
    simscore_csv_path = args.sim_list_csv
    base_dir = args.output_dir
    report_sum_dir = args.report_sum_dir

    # 2. Neo4j 연결
    print(f"🔌 Connecting to Neo4j... ({Config.NEO4J_URL})")
    n4j = Neo4jGraph(
        url=Config.NEO4J_URL,
        username=Config.NEO4J_USERNAME,
        password=Config.NEO4J_PASSWORD
    )

    # 3. 파라미터 CSV 읽기
    if not os.path.exists(simscore_csv_path):
        print(f"❌ Error: Simulation CSV not found at {simscore_csv_path}")
        return

    df_params = pd.read_csv(simscore_csv_path, dtype={"gid": str})
    print(f"📋 Loaded {len(df_params)} tasks from CSV.")

    # 4. 반복 실행 (기존 로직)
    for idx, row in df_params.iterrows():
        try:
            gid_index = int(row["gid_index"])
            gid = str(row["gid"])
            simscore_percentage = float(row["simscore_percentage"])
            round_adj_token = int(row["round_adj_token"])

            # GID 파싱
            if not gid.startswith("FOMC"):
                print(f"[WARN] Invalid GID format: {gid}. Skipping.")
                continue

            yyyymm = gid.replace("FOMC", "")
            if len(yyyymm) < 6 or not yyyymm.isdigit():
                print(f"[WARN] Invalid Date in GID: {gid}. Skipping.")
                continue

            yyyy = yyyymm[:4]
            mm = yyyymm[4:6]
            year_numeric = int(yyyymm)

            # Similarity Score Median 계산
            sim_score_median_list = get_column_top_percent_values(input_path, simscore_percentage)

            if gid_index < 0 or gid_index >= len(sim_score_median_list):
                print(f"[WARN] Index {gid_index} out of range. Skipping.")
                continue

            sim_score_median = sim_score_median_list[gid_index]
            print(f"\n[{idx+1}/{len(df_params)}] Processing {gid} (Idx={gid_index}, %={simscore_percentage}, Median={sim_score_median:.4f})")

            # 질문 로드 (Monetary Statement)
            try:
                question_csv_path = find_monetary_statement_csv(yyyymm, base_dir=report_sum_dir)
                question = load_high(question_csv_path)
            except FileNotFoundError as e:
                print(f"   ⚠️ {e}. Skipping.")
                continue

            # 결과 데이터 구조
            data = [["Statement", "Papers", "FSR", "SLOOS", "beigebook"]]

            # 5회 반복 추론
            for i in range(5):
                print(f"   > Run {i+1}/5...")
                response = get_response(n4j, gid, question, gid_index, sim_score_median, year_numeric, meeting_no=1)
                data.append(response)

            # 결과 저장
            out_df = pd.DataFrame(data[1:], columns=data[0])

            # 저장 경로 생성
            year_dir = os.path.join(base_dir, yyyy)
            month_dir = os.path.join(year_dir, mm)
            os.makedirs(month_dir, exist_ok=True)

            out_filename = f"{gid}_({gid_index},{round_adj_token},{simscore_percentage}%).csv"
            out_path = os.path.join(month_dir, out_filename)
            out_df.to_csv(out_path, index=False, encoding="utf-8")

            print(f"   Saved to: {out_path}")

        except Exception as e:
            print(f"   Error processing row {idx}: {e}")

# --- Main Pipeline Controller ---

def main():
    parser = argparse.ArgumentParser(description="FOMC GraphRAG Integrated Pipeline")

    # [Step Selection]
    parser.add_argument('--step', type=str, required=True,
                        choices=['preprocess', 'construct', 'link', 'inference'],
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
                        default=Config.DIR_REPORT_SUM,
                        help="Directory containing monetary statement CSVs")
    args = parser.parse_args()

    # --- Router ---
    if args.step == 'preprocess':
        if not args.input_file:
            print(" Error: --input_file(Theory) is required for preprocessing.")
            return
        generate_para(input_path=args.input_file)

    elif args.step == 'construct':
        print(" Constructing Graph...")
        build_graph_from_files(
            data_path=Config.DATA_THEORY_PATH,
            excel_path=Config.DATA_SIMSCORE_EXCEL,
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

if __name__ == "__main__":
    main()