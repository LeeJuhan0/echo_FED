import os
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from src.external.camel.storages import Neo4jGraph
from src.external.camel.agents import KnowledgeGraphAgent
from src.external.camel.loaders import UnstructuredIO
from src.external.camel.models import OpenAIModel
from src.external.camel.types import ModelType
from src.ingestion import dataloader
from src.utils.common import add_text_property, add_ge_emb, add_gid, add_sim_score
from src.utils.config import Config
from src.utils.common import add_sum

# [NEW] 청킹 모듈 가져오기
from src.ingestion.chunk_processor import run_chunk

load_dotenv()

import re


def extract_fomc_gid(filename: str) -> str:
    """
    Extract YYYYMM from filename and return FOMCYYYYMM
    Example: FOMC_20170201.csv → FOMC201702
    """

    match = re.search(r"(20\d{2})(0[1-9]|1[0-2])", filename)

    if not match:
        raise ValueError(f"Cannot extract YYYYMM from: {filename}")

    yyyymm = match.group(0)

    return f"FOMC{yyyymm}"

def normalize_chunks(chunks) -> list[str]:

    if chunks is None:
        return []

    # string → list
    if isinstance(chunks, str):
        return [chunks]

    # list → flatten + filter
    if isinstance(chunks, list):

        result = []

        for x in chunks:

            if isinstance(x, str):
                result.append(x)

            elif isinstance(x, list):
                # flatten
                for y in x:
                    if isinstance(y, str):
                        result.append(y)

            else:
                result.append(str(x))

        return result

    # fallback
    return [str(chunks)]

def get_sim_scores(excel_path, data_path):
    if not os.path.exists(data_path):
        print(f" Warning: Data path not found: {data_path}")
        return {}

    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    try:
        df = pd.read_excel(excel_path, sheet_name=0)
    except FileNotFoundError:
        print(f" Error: Excel file not found at {excel_path}")
        return {}

    def norm(x):
        return str(x).strip()

    excel_rows = {}
    for idx, fname in enumerate(df.iloc[:, 0]):
        key = norm(fname)
        if key:
            excel_rows[key] = idx

    sim_scores_flat = {}
    for file_name in files:
        key = norm(file_name)
        if key in excel_rows:
            row_idx = excel_rows[key]
            scores = df.iloc[row_idx, 1:].tolist()
            sim_scores_flat[file_name] = scores
        else:
            pass

    return sim_scores_flat

def create_theory_metagraph(
        content,
        sim_score,
        gid,
        n4j_instance,
        grained_chunk: bool = False  # [New] 인자 추가
):
    uio = UnstructuredIO()
    model_name_str = Config.MODEL_CONSTRUCTION

    try:
        enum_key = model_name_str.upper().replace("-", "_")
        if hasattr(ModelType, enum_key):
            target_model_type = getattr(ModelType, enum_key)
        else:
            target_model_type = ModelType.GPT_4
    except:
        target_model_type = ModelType.GPT_4

    construction_model = OpenAIModel(
        model_type=target_model_type,
        model_config_dict={}
    )

    kg_agent = KnowledgeGraphAgent(
        model=construction_model
    )
    # -------------------------------------

    whole_chunk = content
    if grained_chunk:
        print(f"   Running Agentic Chunking for {gid}")
        try:
            # run_chunk는 텍스트 리스트를 반환함
            chunks = run_chunk(content)
            print(f"   Split into {len(chunks)} chunks.")
            chunks = normalize_chunks(chunks)
        except Exception as e:
            print(f"   Chunking failed: {e}. Fallback to single chunk.")
            chunks = [content]
    else:
        # 기존 로직: 리스트가 아니면 리스트로 변환하여 그대로 사용
        print("   Using single chunk processing.")
        if not isinstance(content, list):
            chunks = [content]
        else:
            chunks = content

    # 각 청크별 처리
    for i, cont in enumerate(chunks):
        try:
            # Camel: 텍스트 -> Element 변환
            element_example = uio.create_element_from_text(text=cont)

            # 그래프 요소 추출 (Node, Relationship)
            graph_elements = kg_agent.run(element_example, parse_graph_elements=True)

            # 속성 추가 (Utils 활용 - Sim Score 포함)
            graph_elements = add_text_property(graph_elements, element_example.text, field="source_text")
            graph_elements = add_ge_emb(graph_elements)
            graph_elements = add_gid(graph_elements, gid)

            graph_elements = add_sim_score(graph_elements, sim_score)

            # Neo4j 저장
            n4j_instance.add_graph_elements(graph_elements=[graph_elements])

            if len(chunks) > 1 and (i+1) % 5 == 0:
                print(f"      Processed chunk {i+1}/{len(chunks)}")

        except Exception as e:
            print(f"     Error processing chunk {i}: {e}")

    return n4j_instance

def build_theory_graph_from_files(
        data_path,
        excel_path,
        grained_chunk: bool = False  # [New] 인자 받아서 전달
):

    # Neo4j 연결
    n4j = Neo4jGraph(
        url=Config.NEO4J_URL,
        username=Config.NEO4J_USERNAME, # .env 변수명 확인 필요 (USER vs USERNAME)
        password=Config.NEO4J_PASSWORD
    )

    # 유사도 점수 로드 (기존 로직 유지)
    sim_scores_dict = get_sim_scores(excel_path, data_path)
    print(f" Loaded similarity scores for {len(sim_scores_dict)} files.")

    files = [f for f in os.listdir(data_path) if f.lower().endswith(".csv")]

    print(f" Starting graph construction for {len(files)} files (Grained: {grained_chunk})...")

    for file_name in files:
        gid = "paragraph" # (필요 시 로직 변경 가능)

        # 파일명에 해당하는 sim_score 가져오기
        sim_score = sim_scores_dict.get(file_name, [])

        file_path = os.path.join(data_path, file_name)

        # 데이터 로드
        content = dataloader.load_high(file_path)

        # 그래프 생성 및 저장 (인자 전달)
        create_theory_metagraph(content, sim_score, gid, n4j, grained_chunk=grained_chunk)
        print(f"  - Processed: {file_name}")

    print(" Graph construction complete!")

def create_statement_metagraph(

        content,
        gid,
        n4j_instance,
        grained_chunk: bool = False

):

    uio = UnstructuredIO()

    model_name_str = Config.MODEL_CONSTRUCTION


    # 모델 설정 (기존 유지)
    try:

        enum_key = model_name_str.upper().replace("-", "_")

        if hasattr(ModelType, enum_key):
            target_model_type = getattr(ModelType, enum_key)
        else:
            target_model_type = ModelType.GPT_4

    except:
        target_model_type = ModelType.GPT_4


    construction_model = OpenAIModel(
        model_type=target_model_type,
        model_config_dict={}
    )


    kg_agent = KnowledgeGraphAgent(model=construction_model)


    # -------------------------------------

    whole_chunk = content


    # Chunking 로직 유지
    if grained_chunk:

        print(f"   Running Agentic Chunking for {gid}")

        try:
            chunks = run_chunk(content)
            print(f"   Split into {len(chunks)} chunks.")

        except Exception as e:
            print(f"   Chunking failed: {e}")
            chunks = [content]

    else:

        print("   Using single chunk processing.")

        if not isinstance(content, list):
            chunks = [content]
        else:
            chunks = content


    # -------------------------------------

    for i, cont in enumerate(chunks):

        try:
            element_example = uio.create_element_from_text(text=cont)
            graph_elements = kg_agent.run(
                element_example,
                parse_graph_elements=True
            )
            # ===== property 처리 (simscore 제거) =====
            graph_elements = add_text_property(
                graph_elements,
                element_example.text,
                field="source_text"
            )
            graph_elements = add_ge_emb(graph_elements)
            graph_elements = add_gid(graph_elements, gid)
            #  add_sim_score 제거됨
            # 저장
            n4j_instance.add_graph_elements(
                graph_elements=[graph_elements]
            )
            if len(chunks) > 1 and (i+1) % 5 == 0:
                print(f"      Processed chunk {i+1}/{len(chunks)}")
        except Exception as e:
            print(f"     Error processing chunk {i}: {e}")

    add_sum(n4j_instance, whole_chunk, gid)
    return n4j_instance

def build_statement_graph_from_files(

        base_path,
        grained_chunk: bool = False,

        start_year: int = 2000,
        end_year: int = 2016
):

    """
    Build Statement Knowledge Graph by iterating yearly folders.

    base_path/
        2000/*.csv
        2001/*.csv
        ...
        2016/*.csv
    """


    # ---------- Neo4j 연결 ----------

    n4j = Neo4jGraph(

        url=Config.NEO4J_URL,
        username=Config.NEO4J_USERNAME,
        password=Config.NEO4J_PASSWORD
    )


    base_dir = Path(base_path)

    if not base_dir.exists():
        raise FileNotFoundError(f"Statement base path not found: {base_dir}")


    # ---------- 연도 폴더 탐색 ----------

    year_dirs = sorted([

        d for d in base_dir.iterdir()

        if (
                d.is_dir()
                and d.name.isdigit()
                and start_year <= int(d.name) <= end_year
        )

    ])


    if not year_dirs:
        print("⚠ No valid year directories found.")
        return


    print(f" Processing years: {start_year} ~ {end_year}")
    print(f" Found {len(year_dirs)} year folders.")


    # ---------- 연도별 처리 ----------

    for year_dir in year_dirs:
        year = year_dir.name
        print(f"\n Year {year}")


        csv_files = list(year_dir.glob("*.csv"))

        if not csv_files:
            print("    No CSV files.")
            continue


        print(f"   {len(csv_files)} files found.")


        for file_path in csv_files:

            try:

                file_name = file_path.name


                # -------- GID 생성 --------
                gid = extract_fomc_gid(file_name)


                # -------- Load content --------
                content = dataloader.load_high(str(file_path))


                # -------- Build graph --------
                create_statement_metagraph(

                    content=content,

                    gid=gid,

                    n4j_instance=n4j,

                    grained_chunk=grained_chunk
                )
                print(f"   {file_name} → {gid}")
            except Exception as e:
                print(f"  Failed {file_name}: {e}")

    print("\n Statement Graph Construction Complete!")


