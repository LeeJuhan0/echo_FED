import os
from getpass import getpass
from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
from dataloader import load_high
import argparse
from data_chunk import run_chunk
from create_graph import creat_metagraph
from summerize import process_chunks
from retrieve import seq_ret
from utils import *
import pandas as pd
# from nano_graphrag import GraphRAG, QueryParam
import openpyxl
from  sim_score_per import get_column_top_percent_values


parser = argparse.ArgumentParser()
parser.add_argument('-simple', action='store_true')
parser.add_argument('-construct_graph', action='store_true')
parser.add_argument('-inference',  action='store_true')
parser.add_argument('-grained_chunk',  action='store_true')
parser.add_argument('-trinity', action='store_true')
parser.add_argument('-trinity_gid1', type=str, default="485a49f9-1f22-4817-b490-3f6a09759e41")
parser.add_argument('-trinity_gid2', type=str, default="ff8727c8-c3f8-4a60-8f17-981ffd653609")
parser.add_argument('-ingraphmerge',  action='store_true')
parser.add_argument('-crossgraphmerge', action='store_true')
parser.add_argument('-dataset', type=str, default='mimic_ex')
args = parser.parse_args()

def save_responses(year, responses):
    """
    Save the responses to CSV files in a structured folder.

    Args:
    - year (int): The year for the folder naming convention.
    - responses (list of str): List of responses to save.
    """
    base_path = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simulation_STD"

    # Determine the number of folders per year
    num_folders = 9 if year == 2020 else 8

    for folder_index in range(1, num_folders + 1):
        # Create folder name (e.g., "2017-1", "2017-2", ..., "2020-9")
        folder_name = f"{year}-{folder_index}"
        folder_path = os.path.join(base_path, folder_name)

        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save each response to a separate CSV file named "1.csv", "2.csv", ...
        for i, response in enumerate(responses):
            file_name = f"{i + 1}.csv"
            file_path = os.path.join(folder_path, file_name)

            # Write the response to the CSV file
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Response"])
                writer.writerow([response])


def main():
    input_path = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simscore관련\openai_paragraph_vs_statements_only_economic.xlsx"
    percentage = 14.1 #상위 퍼센티지 파라미터
    index = 4 #변경필요
    sim_score_median_list = get_column_top_percent_values(input_path,percentage)
    sim_score_median = sim_score_median_list[index]
    print(sim_score_median)
    url=os.getenv("NEO4J_URL","neo4j://127.0.0.1:7687")
    username=os.getenv("NEO4J_USERNAME","neo4j")
    password=os.getenv("NEO4J_PASSWORD","12345678")

    # Set Neo4j instance
    n4j = Neo4jGraph(
            url=url,
            username=username,             # Default username
            password=password    # Replace 'yourpassword' with your actual password
    )

    if False:
        if False:
                # Read and print the contents of each file
            file_path = args.data_path
            content = load_high(file_path)
            gid = str_uuid()
            n4j = creat_metagraph(args, content, gid, n4j)

            if True:
                link_context(n4j, args.trinity_gid1)
            if True:
                merge_similar_nodes(n4j, None)
    if True:
        data = [["Statment", "Papers"]]
        question = load_high(r"C:/Users/HUFS_MATH/IdeaProjects/FOMC_Graphrag/Statement/report_sum/monetary20170726a1.csv") #변경필요
        gid = "FOMC201707" #변경필요
        meeting_no = 0
        year = 201707_1 #변경필요
        for _ in range(1):
            meeting_no += 1
            response = get_response(n4j, gid, question, index, sim_score_median, year, meeting_no)
            data.append(response)
        df = pd.DataFrame(data[1:], columns=data[0])
        base_dir = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\STD,CV"
        year_dir = os.path.join(base_dir, '2017') #변경필요
        month_dir = os.path.join(year_dir, '07') #변경필요
        # 폴더가 없으면 생성
        os.makedirs(year_dir, exist_ok=True)
        os.makedirs(month_dir, exist_ok=True)
        # 파일 경로 지정
        file_path = os.path.join(month_dir, "201707_ony_theory(1,gpt-5,90.5%).csv") #변경필요
        df.to_csv(file_path, index=False, encoding="utf-8")

if __name__ == "__main__":
    main()