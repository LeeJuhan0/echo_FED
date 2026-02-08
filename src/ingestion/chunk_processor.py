import os
from typing import List
from pydantic import BaseModel, Field
from langchainhub import Client as HubClient
hub = HubClient()
from langchain_openai import ChatOpenAI
from src.ingestion.agentic_chunker import AgenticChunker

# 1. Pydantic 모델 정의
class Sentences(BaseModel):
    sentences: List[str] = Field(description="List of distinct propositions extracted from the text")

def get_propositions(text, runnable, structured_llm):
    # (hub에서 가져온 프롬프트로 먼저 문장을 분해합니다)
    runnable_output = runnable.invoke({
        "input": text
    }).content

    extracted_object = structured_llm.invoke(runnable_output)

    return extracted_object.sentences

def run_chunk(essay):
    # 설정
    obj = hub.pull("wfh/proposal-indexing")
    llm = ChatOpenAI(model='gpt-4o', openai_api_key=os.getenv("OPENAI_API_KEY"))

    #  생성 체인 (Proposition Generation)
    runnable = obj | llm

    #  추출 체인 (Structured Output)
    # [핵심] create_extraction_chain_pydantic 대체
    structured_llm = llm.with_structured_output(Sentences)

    # 문단 분리
    paragraphs = essay.split("\n\n")
    essay_propositions = []

    print(f" Processing {len(paragraphs)} paragraphs...")

    for i, para in enumerate(paragraphs):
        if not para.strip():
            continue

        propositions = get_propositions(para, runnable, structured_llm)

        essay_propositions.extend(propositions)
        print(f"  Paragraph {i} done: {len(propositions)} propositions extracted.")
        # print(propositions) # 필요시 주석 해제

    # Agentic Chunking 실행
    print("\n Starting Agentic Chunking...")
    ac = AgenticChunker()
    ac.add_propositions(essay_propositions)
    ac.pretty_print_chunks()

    chunks = ac.get_chunks(get_type='list_of_strings')

    return chunks

# 테스트용 실행 코드
if __name__ == "__main__":
    test_essay = """
    Greg likes to eat pizza. He also likes to eat hamburgers.
    The sky is blue today. It is a beautiful day.
    """
    result = run_chunk(test_essay)
    print("\nFinal Chunks:", result)