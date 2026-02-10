import os
import pandas as pd
from dotenv import load_dotenv

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

def get_sim_scores(excel_path, data_path):
    """
    엑셀 파일에서 파일명별 Similarity Score를 읽어옵니다. (기존 로직 유지)
    """
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

def create_metagraph(
        content,
        sim_score,
        gid,
        n4j_instance,
        grained_chunk: bool = False  # [New] 인자 추가
):
    """
    Camel Agent를 사용하여 텍스트에서 그래프 요소를 추출하고 Neo4j에 저장합니다.
    grained_chunk가 True이면 run_chunk를 통해 세분화된 청킹을 수행합니다.
    """
    uio = UnstructuredIO()
    model_name_str = Config.MODEL_CONSTRUCTION

    # --- [기존] ModelType 설정 로직 유지 ---
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
        model_config_dict={"model": model_name_str}
    )

    kg_agent = KnowledgeGraphAgent(
        model=construction_model
    )
    # -------------------------------------

    # [NEW] 분기 로직: Grained Chunking 여부 확인
    chunks = []
    if grained_chunk:
        print(f"   ✂️ Running Agentic Chunking for {gid}...")
        try:
            # run_chunk는 텍스트 리스트를 반환함
            chunks = run_chunk(content)
            print(f"   ✅ Split into {len(chunks)} chunks.")
        except Exception as e:
            print(f"   ⚠️ Chunking failed: {e}. Fallback to single chunk.")
            chunks = [content]
    else:
        # 기존 로직: 리스트가 아니면 리스트로 변환하여 그대로 사용
        print("   📄 Using single chunk processing.")
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

            # [기존 유지] Sim Score 추가
            graph_elements = add_sim_score(graph_elements, sim_score)

            # Neo4j 저장
            n4j_instance.add_graph_elements(graph_elements=[graph_elements])

            if len(chunks) > 1 and (i+1) % 5 == 0:
                print(f"      > Processed chunk {i+1}/{len(chunks)}")

        except Exception as e:
            print(f"      ❌ Error processing chunk {i}: {e}")

    return n4j_instance

def build_graph_from_files(
        data_path,
        excel_path,
        grained_chunk: bool = False  # [New] 인자 받아서 전달
):
    """
    메인 로직: 파일들을 읽어서 그래프를 구축하는 함수
    """
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
        create_metagraph(content, sim_score, gid, n4j, grained_chunk=grained_chunk)
        print(f"  - Processed: {file_name}")

    print(" Graph construction complete!")