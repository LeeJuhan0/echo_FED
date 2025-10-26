import os
from getpass import getpass

import args

from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
import dataloader
import argparse
from data_chunk import run_chunk
from utils import *

import os
import pandas as pd

import os
import pandas as pd

def get_sim_scores(excel_path, data_path):
    """
    단일 시트 엑셀:
    - A열: 파일명
    - B~BU열: simscore들(끝 열은 동적으로 전체 열 기준으로 처리)
    dataset 폴더(data_path)의 파일명과 A열이 일치하면, 해당 행의 B열부터 끝 열까지를 리스트로 반환.
    {file_name: [score1, score2, ..., scoreN]} 형태
    """
    # 데이터셋 폴더의 파일 목록
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    # 엑셀 첫 번째 시트만 읽음
    df = pd.read_excel(excel_path, sheet_name=0)

    # 이름 정규화 함수(공백 제거 및 문자열화)
    def norm(x):
        return str(x).strip()

    # A열(파일명) → 행 인덱스 매핑
    excel_rows = {}
    for idx, fname in enumerate(df.iloc[:, 0]):
        key = norm(fname)
        if key:  # 빈 문자열/NaN 방지
            excel_rows[key] = idx

    sim_scores_flat = {}
    for file_name in files:
        key = norm(file_name)
        if key in excel_rows:
            row_idx = excel_rows[key]
            # B열부터 끝 열까지 모두 점수로 사용
            scores = df.iloc[row_idx, 1:].tolist()
            sim_scores_flat[file_name] = scores
            print(sim_scores_flat[file_name])
        else:
            print(f"파일명 불일치: {file_name} (엑셀에 없음)")

    return sim_scores_flat

def creat_metagraph(args, content, sim_score,gid, n4j):

    # Set instance
    uio = UnstructuredIO()
    kg_agent = KnowledgeGraphAgent()
    whole_chunk = content

    if False:
        content = run_chunk(content)
    else:
        content = [content]
    for cont in content:
        element_example = uio.create_element_from_text(text=cont)

        ans_str = kg_agent.run(element_example, parse_graph_elements=False)
        print(ans_str)

        graph_elements = kg_agent.run(element_example, parse_graph_elements=True)
        graph_elements = add_text_property(graph_elements, element_example.text, field="source_text")

        graph_elements = add_ge_emb(graph_elements)
        graph_elements = add_gid(graph_elements, gid)
        graph_elements = add_sim_score(graph_elements, sim_score)

        n4j.add_graph_elements(graph_elements=[graph_elements])
    return n4j

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-construct_graph', action='store_true')
    parser.add_argument('-inference',  action='store_true')
    parser.add_argument('-grained_chunk',  action='store_true' )
    parser.add_argument('-trinity', action='store_true')
    parser.add_argument('-trinity_gid1', type=str)
    parser.add_argument('-trinity_gid2', type=str)
    parser.add_argument('-ingraphmerge',  action='store_true')
    parser.add_argument('-crossgraphmerge', action='store_true')
    parser.add_argument('-data_path', type=str, default=(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Theory\paragraph")) #변경필요
    args = parser.parse_args()

    url=os.getenv("NEO4J_URL", "neo4j://127.0.0.1:7687")
    username=os.getenv("NEO4J_USERNAME", "neo4j")
    password=os.getenv("NEO4J_PASSWORD", "12345678")

    n4j = Neo4jGraph(
        url=url,
        username=username,             # Default username
        password=password              # Replace 'yourpassword' with your actual password
    )
    files = [file for file in os.listdir(args.data_path) if os.path.isfile(os.path.join(args.data_path, file))]
    excel_path = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simscore관련\openai_paragraph_vs_statements_only_economic.xlsx" #변경필요
    sim_scores_dict = get_sim_scores(excel_path, args.data_path)
    print(sim_scores_dict)

    for file_name in files:
        if not file_name.lower().endswith(".csv"):
            continue
        #date_part = file_name[10:16]
        #ym = date_part[:6]
        gid = f"paragraph"
        sim_score = sim_scores_dict.get(file_name, [])
        print(sim_score)
        file_path = os.path.join(args.data_path, file_name)
        content = dataloader.load_high(file_path)
        n4j = creat_metagraph(args, content, sim_score,gid, n4j)

if __name__ == "__main__":
    main()
