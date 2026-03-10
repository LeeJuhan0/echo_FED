import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class Config:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # -----------------------------------------------------------------------
    # API Keys & external service credentials
    # -----------------------------------------------------------------------
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    NEO4J_URL = os.getenv("NEO4J_URL", "neo4j://127.0.0.1:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
    FRIENDLI_TOKEN = os.getenv("FRIENDLI_TOKEN", "")

    # -----------------------------------------------------------------------
    # LLM model selection (construction vs. inference)
    # -----------------------------------------------------------------------
    MODEL_CONSTRUCTION = os.getenv("LLM_MODEL_CONSTRUCTION", "gpt-4o-mini")
    MODEL_INFERENCE = os.getenv("LLM_MODEL_INFERENCE", "gpt-4o")

    # -----------------------------------------------------------------------
    # Data paths (project-root relative defaults, overridable via .env)
    # -----------------------------------------------------------------------

    # Similarity score Excel (paragraph vs. statement)
    DATA_SIMSCORE_EXCEL = os.getenv(
        "DATA_SIMSCORE_EXCEL",
        os.path.join(
            PROJECT_ROOT,
            "data", "processed", "experiment",
            "openai_paragraph_vs_statements_only_economic.xlsx",
        ),
    )

    # Simulation list CSV (GID / parameter grid)
    DATA_SIM_LIST_CSV = os.getenv(
        "DATA_SIM_LIST_CSV",
        os.path.join(PROJECT_ROOT, "data", "processed", "token", "19000_simpct_list.csv"),
    )

    # Theory raw dataset (papers / theory documents)
    DATA_THEORY_PATH = os.getenv(
        "DATA_THEORY_PATH",
        os.path.join(PROJECT_ROOT, "data", "raw", "Theory", "dataset_paper"),
    )

    # Raw FOMC statement PDFs
    RAW_STATEMENT_DIR = os.getenv(
        "RAW_STATEMENT_DIR",
        os.path.join(PROJECT_ROOT, "data", "raw", "Policy", "FOMC_Statement"),
    )

    # Processed monetary-policy statement prompts (CSV)
    DIR_statement_prompt = os.getenv(
        "DIR_statement_prompt",
        os.path.join(PROJECT_ROOT, "data", "processed", "statement_prompt"),
    )

    # Inference output directory
    DIR_OUTPUT = os.getenv(
        "DIR_OUTPUT",
        os.path.join(PROJECT_ROOT, "data", "results", "experiment", "simul30times"),
    )
