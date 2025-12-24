import os
import csv
import argparse
from glob import glob
import pandas as pd

from getpass import getpass
from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO

from dataloader import load_high as _original_load_high  # keep reference if needed
from dataloader import load_high  # will be overridden by the local definition below if desired
from data_chunk import run_chunk
from create_graph import creat_metagraph
from summerize import process_chunks
from retrieve import seq_ret
from utils import *
# from nano_graphrag import GraphRAG, QueryParam
import openpyxl
from sim_score_per import get_column_top_percent_values


parser = argparse.ArgumentParser()
parser.add_argument('-simple', action='store_true')
parser.add_argument('-construct_graph', action='store_true')
parser.add_argument('-inference', action='store_true')
parser.add_argument('-grained_chunk', action='store_true')
parser.add_argument('-trinity', action='store_true')
parser.add_argument('-trinity_gid1', type=str, default="485a49f9-1f22-4817-b490-3f6a09759e41")
parser.add_argument('-trinity_gid2', type=str, default="ff8727c8-c3f8-4a60-8f17-981ffd653609")
parser.add_argument('-ingraphmerge', action='store_true')
parser.add_argument('-crossgraphmerge', action='store_true')
parser.add_argument('-dataset', type=str, default='mimic_ex')
args = parser.parse_args()


def find_monetary_statement_csv(year_month: str) -> str:
    """
    Find the monetary statement CSV file whose name starts with monetary{YYYYMM} in
    the Statement/report_sum directory.

    Example match: monetary20231213a1.csv for year_month='202312'

    Returns the first matching file (sorted) or raises FileNotFoundError if none found.
    """
    base_dir = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Statement\report_sum"
    pattern = os.path.join(base_dir, f"monetary{year_month}*.csv")
    matches = sorted(glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No monetary statement CSV found for pattern: {pattern}")
    return matches[0]


# Override load_high as requested
def load_high(datapath):
    """
    Read the entire file at datapath and return its contents as a single string.
    """
    all_content = ""
    with open(datapath, 'r', encoding='utf-8') as file:
        for line in file:
            all_content += line.strip() + "\n"
    return all_content


def main():
    # Paths
    input_path = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simscore관련\openai_paragraph_vs_statements_only_economic.xlsx"
    simscore_csv_path = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\19000_simpct_list.csv"

    # Neo4j credentials
    url = os.getenv("NEO4J_URL", "neo4j://127.0.0.1:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "12345678")

    # Set Neo4j instance
    n4j = Neo4jGraph(
        url=url,
        username=username,
        password=password
    )

    # Read the simscore parameters CSV
    # Expected columns: gid_index,gid,token_total,simscore_percentage,adj_token,round_adj_token
    df_params = pd.read_csv(simscore_csv_path, dtype={"gid": str})

    # Output base directory
    base_dir = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simulation_19000_score"

    # Iterate each row and run the pipeline
    for _, row in df_params.iterrows():
        try:
            # Extract parameters from row
            gid_index = int(row["gid_index"])
            gid = str(row["gid"])
            simscore_percentage = float(row["simscore_percentage"])
            round_adj_token = int(row["round_adj_token"])

            # Parse year/month from gid like 'FOMCYYYYMM'
            if not gid.startswith("FOMC"):
                print(f"[WARN] gid does not start with 'FOMC': {gid}. Skipping.")
                continue

            yyyymm = gid.replace("FOMC", "")
            if len(yyyymm) < 6 or not yyyymm.isdigit():
                print(f"[WARN] gid does not contain valid YYYYMM: {gid}. Skipping.")
                continue

            yyyy = yyyymm[:4]
            mm = yyyymm[4:6]
            year_numeric = int(yyyymm)  # e.g., 202312

            # Determine sim_score_median from Excel per row's simscore_percentage and gid_index
            sim_score_median_list = get_column_top_percent_values(input_path, simscore_percentage)
            if gid_index < 0 or gid_index >= len(sim_score_median_list):
                print(f"[WARN] gid_index {gid_index} out of range for percentage {simscore_percentage}. Skipping.")
                continue
            sim_score_median = sim_score_median_list[gid_index]
            print(f"[INFO] gid={gid}, index={gid_index}, percentage={simscore_percentage}, sim_score_median={sim_score_median}")

            # Load the question content from the matching monetary statement CSV
            try:
                question_csv_path = find_monetary_statement_csv(yyyymm)
            except FileNotFoundError as e:
                print(f"[WARN] {e}. Skipping gid={gid}.")
                continue
            question = load_high(question_csv_path)
            data = [["Statment", "Papers","FSR","SLOOS","beigebook"]]

            # Run retrieval/generation for the meeting (single iteration as before)
            for _ in range(5):
                meeting_no = 1
                response = get_response(n4j, gid, question, gid_index, sim_score_median, year_numeric, meeting_no)
                data.append(response)
            out_df = pd.DataFrame(data[1:], columns=data[0])

            # Prepare output directories
            year_dir = os.path.join(base_dir, yyyy)
            month_dir = os.path.join(year_dir, mm)
            os.makedirs(month_dir, exist_ok=True)

            # Output file name: "{gid}_({gid_index},{round_adj_token},{simscore_percentage}%).csv"
            out_filename = f"{gid}_({gid_index},{round_adj_token},{simscore_percentage}%).csv"
            out_path = os.path.join(month_dir, out_filename)
            out_df.to_csv(out_path, index=False, encoding="utf-8")

            print(f"[OK] Saved result for gid={gid} to {out_path}")

        except Exception as e:
            print(f"[ERROR] Failed processing row with gid={row.get('gid')}: {e}")


if __name__ == "__main__":
    main()