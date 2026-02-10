import os
import sys

# 프로젝트 루트 경로 설정 (단독 실행 시 필요)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import Config
from src.external.camel.storages import Neo4jGraph
from src.utils.common import (
    ref_link, link_context,
    ref_link_FSR, link_context_FSR,
    ref_link_SLOOS, link_context_SLOOS,
    ref_link_beigebook, link_context_beigebook
)

def run_linking_process(
        fomc_gid: str,
        paper_gid: str = "paragraph",
        check_fsr: str = None,
        check_sloos: str = None,
        check_beigebook: str = None,
        threshold: float = 0.7,
        ctx_idx: int = 0
):
    """
    Pipeline에서 호출하는 메인 함수입니다.
    """
    # Neo4j 연결
    print(f" Connecting to Neo4j... ({Config.NEO4J_URL})")
    try:
        n4j = Neo4jGraph(
            url=Config.NEO4J_URL,
            username=Config.NEO4J_USERNAME,
            password=Config.NEO4J_PASSWORD
        )
    except Exception as e:
        print(f" Connection Failed: {e}")
        return

    # Paper(Theory) -> FOMC 링크
    print(f"\n [Link] Paper({paper_gid}) -> FOMC({fomc_gid})")
    try:
        ref_link(n4j, paper_gid, fomc_gid)
        context = link_context(n4j, fomc_gid, ctx_idx, threshold)
        print(f"   > Context Sample: {str(context)[:100]}..." if context else "   > Context: Empty")
    except Exception as e:
        print(f"   Error: {e}")

    # FSR 링크
    if check_fsr:
        print(f"\n [Link] FSR({check_fsr}) -> FOMC({fomc_gid})")
        try:
            ref_link_FSR(n4j, check_fsr, fomc_gid)
            link_context_FSR(n4j, fomc_gid)
        except Exception as e:
            print(f"   ️ Error: {e}")

    # SLOOS 링크
    if check_sloos:
        print(f"\n [Link] SLOOS({check_sloos}) -> FOMC({fomc_gid})")
        try:
            ref_link_SLOOS(n4j, check_sloos, fomc_gid)
            link_context_SLOOS(n4j, fomc_gid)
        except Exception as e:
            print(f"   Error: {e}")

    # BeigeBook 링크
    if check_beigebook:
        print(f"\n [Link] BeigeBook({check_beigebook}) -> FOMC({fomc_gid})")
        try:
            ref_link_beigebook(n4j, check_beigebook, fomc_gid)
            link_context_beigebook(n4j, fomc_gid)
        except Exception as e:
            print(f"   Error: {e}")

    print("\n Linking Process Completed.")

# (선택 사항) 단독 테스트용 코드
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fomc_gid', required=True)
    args = parser.parse_args()
    run_linking_process(args.fomc_gid)