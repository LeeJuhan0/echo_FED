import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 데이터 파일 경로 (프로젝트 루트 기준 상대 경로로 설정)
    DATA_SIMSCORE_EXCEL = os.getenv(
        "DATA_SIMSCORE_EXCEL",
        os.path.join(
            PROJECT_ROOT,
            "data", "processed", "experiment",
            "openai_paragraph_vs_statements_only_economic.xlsx"
        )
    )

    # Simulation List CSV 경로
    DATA_SIM_LIST_CSV = os.getenv(
        "DATA_SIM_LIST_CSV",
        os.path.join(PROJECT_ROOT, "data", "processed", "token", "19000_simpct_list.csv")
    )

    # 결과 저장소 (Output Dir)
    DIR_OUTPUT = os.getenv(
        "DIR_OUTPUT",
        os.path.join(PROJECT_ROOT, "data", "results","experiment","simulation")
    )

    # 통화정책 보고서 폴더
    DIR_REPORT_SUM = os.getenv(
        "DIR_REPORT_SUM",
        os.path.join(PROJECT_ROOT, "data", "processed", "statement_prompt")
    )

    # API Keys & Secrets
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    NEO4J_URL = os.getenv("NEO4J_URL", "neo4j://127.0.0.1:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")


    # 2. Models (구축용 vs 추론용 분리)
    # [중요] 여기서 모델명을 관리합니다.
    MODEL_CONSTRUCTION = os.getenv("LLM_MODEL_CONSTRUCTION", "gpt-4o-mini")
    MODEL_INFERENCE = os.getenv("LLM_MODEL_INFERENCE", "gpt-4o")

    # 4. Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_THEORY_PATH = os.getenv("DATA_THEORY_PATH", os.path.join(BASE_DIR, "data/raw/theory/paragraph"))
    DATA_SIMSCORE_EXCEL = os.getenv("DATA_SIMSCORE_EXCEL", os.path.join(BASE_DIR, "data/processed/experiment/openai_paragraph_vs_statements_only_economic.xlsx"))