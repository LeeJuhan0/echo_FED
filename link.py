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



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-simple', action='store_true')
    parser.add_argument('-construct_graph', action='store_true')
    parser.add_argument('-inference',  action='store_true')
    parser.add_argument('-grained_chunk',  action='store_true')
    parser.add_argument('-trinity', action='store_true')
    parser.add_argument('-trinity_gid1', type=str, default="8b573e68-9598-47d3-9c7d-17b0dbfb6e70")
    parser.add_argument('-trinity_gid2', type=str, default="485a49f9-1f22-4817-b490-3f6a09759e41")
    parser.add_argument('-ingraphmerge',  action='store_true')
    parser.add_argument('-crossgraphmerge', action='store_true')
    parser.add_argument('-dataset', type=str, default='koreabank')
    parser.add_argument('-data_path', type=str, default=("C:/Users/wngks/ideaprojects/_Graphrag/output_24_1.csv"))
    parser.add_argument('-test_data_path', type=str, default='./dataset_ex/report_0.txt')
    args = parser.parse_args()

    url=os.getenv("NEO4J_URL", "neo4j://127.0.0.1:7687")
    username=os.getenv("NEO4J_USERNAME", "neo4j")
    password=os.getenv("NEO4J_PASSWORD", "12345678")

    n4j = Neo4jGraph(
        url=url,
        username=username,             # Default username
        password=password     # Replace 'yourpassword' with your actual password
    )

    papergid = "paragraph" #고정
    FOMCgid = "FOMC202003"
    FSRgid = "FSR202504"
    SLOOSgid = "SLOOS202507"
    beigebookgid = "beigebook202507"
    ref_link(n4j, papergid, FOMCgid)
    print(link_context(n4j, FOMCgid,23, 0))
    print()
    #ref_link_FSR(n4j, FSRgid, FOMCgid)
    #print(link_context_FSR(n4j, FOMCgid))
    print()
    #ref_link_SLOOS(n4j, SLOOSgid, FOMCgid)
    #print(link_context_SLOOS(n4j, FOMCgid))
    print()
    #ref_link_beigebook(n4j,beigebookgid,FOMCgid)
    #print(link_context_beigebook(n4j,FOMCgid))

if __name__ == "__main__":
    main()