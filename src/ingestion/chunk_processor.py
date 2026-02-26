from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from typing import Optional, List
#from langchain_core.pydantic_v1 import BaseModel
from pydantic import BaseModel, Field
from langchainhub import Client
hub = Client()

import os
from src.ingestion.dataloader import load_high
from src.ingestion.agentic_chunker import AgenticChunker
from src.utils.config import Config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

construction_model = Config.MODEL_CONSTRUCTION

# -----------------------------
# Schema
# -----------------------------
class Sentences(BaseModel):
    sentences: list[str] = Field(
        description="Atomic propositions"
    )


def build_extraction_chain():

    system_prompt = """
    Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.
    
    1. Split compound sentence into simple sentences. Maintain the original phrasing whenever possible.
    2. Separate descriptive information into its own proposition.
    3. Replace pronouns with full entity names.
    4. Present results as JSON list of strings.
    
    Return ONLY valid JSON.
    """

    human_prompt = """
    Decompose the following:
    
    {input}
    
    {format}
    """

    parser = PydanticOutputParser(
        pydantic_object=Sentences
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ]).partial(
        format=parser.get_format_instructions()
    )

    llm = ChatOpenAI(
        model = construction_model ,
        temperature=0
    )

    chain = prompt | llm | parser

    return chain


# -----------------------------
# Runner
# -----------------------------
def extract_sentences(chain, text: str) -> list[str]:

    result = chain.invoke({
        "input": text
    })

    if hasattr(result, "sentences"):
        return result.sentences

    return []


# Pydantic data class
class Sentences(BaseModel):
    sentences: List[str]


def get_propositions(text, runnable, extraction_chain):
    runnable_output = runnable.invoke({
        "input": text
    }).content

    propositions = extraction_chain.run(runnable_output)[0].sentences
    return propositions

def run_chunk(essay: str):

    chain = build_extraction_chain()

    paragraphs = [
        p.strip()
        for p in essay.split("\n\n")
        if p.strip()
    ]

    propositions = []

    for i, para in enumerate(paragraphs):

        try:
            props = extract_sentences(chain, para)
            propositions.extend(props)

            print(f"Paragraph {i}: {len(props)} props")

        except Exception as e:

            print(f"Fail {i}: {e}")

            # fallback
            from re import split
            fallback = split(r'(?<=[.!?])\s+', para)
            propositions.extend(fallback)


    # Agentic Chunking
    ac = AgenticChunker()
    ac.add_propositions(propositions)

    chunks = ac.get_chunks("list_of_strings")

    return chunks
    print(chunks)

"""
import os
import re
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 
from src.ingestion.dataloader import load_high
from src.ingestion.agentic_chunker import AgenticChunker

# -----------------------------
# 1. Schema Definition
# -----------------------------
class Sentences(BaseModel):
    """'''List of atomic propositions extracted from the text.'''"""
    sentences: List[str] = Field(
        ..., 
        description="A list of clear, simple assertions/propositions derived from the input text."
    )

# -----------------------------
# 2. Chain Builder (Modern Approach)
# -----------------------------
def build_extraction_chain():
    # OpenAI의 Function Calling 기능을 활용하므로 포맷 지침(format_instructions)이 필요 없습니다.
    system_prompt = """'''
You are an expert at information extraction and text simplification.
Decompose the "Content" into clear and simple propositions (atomic facts), ensuring they are interpretable out of context.

Guidelines:
1. Split compound sentences into simple sentences.
2. Maintain the original phrasing whenever possible.
3. Separate descriptive information into its own proposition.
4. Resolve pronouns (he, she, it, they) to their full entity names based on context.
'''
"""

human_prompt = "Content to decompose:\n{input}"

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

llm = ChatOpenAI(
    model="gpt-4o-mini", # 비용 효율적인 모델 선택
    temperature=0
)

# 핵심 변경: with_structured_output 사용
# PydanticOutputParser보다 훨씬 강력하고 오류가 적습니다.
structured_llm = llm.with_structured_output(Sentences)

chain = prompt | structured_llm

return chain

# -----------------------------
# 3. Execution Logic
# -----------------------------
def run_chunk(essay: str):

chain = build_extraction_chain()

# 문단 나누기 (빈 문단 제거)
paragraphs = [p.strip() for p in essay.split("\n\n") if p.strip()]

all_propositions = []

# LangChain의 batch 기능을 사용하여 병렬 처리 가능 (속도 향상)
# 입력 리스트 생성
inputs = [{"input": p} for p in paragraphs]

# 설정: 동시 요청 수 제한 (Rate Limit 방지) - 필요시 config 조정
# batch_results = chain.batch(inputs, config={"max_concurrency": 5}) 

# 여기서는 디버깅/로깅을 위해 순차처리 구조를 유지하되 로직을 깔끔하게 정리합니다.
for i, para in enumerate(paragraphs):
    try:
        # invoke 호출
        result: Sentences = chain.invoke({"input": para})
        
        # None 체크 (with_structured_output은 실패시 None을 줄 수도 있음)
        if result and result.sentences:
            props = result.sentences
            all_propositions.extend(props)
            print(f"Paragraph {i}: Extracted {len(props)} propositions")
        else:
            raise ValueError("Empty result from LLM")

    except Exception as e:
        print(f"Error in Paragraph {i}: {e}")
        
        # Fallback: 정규식으로 문장 단순 분리
        fallback_props = re.split(r'(?<=[.!?])\s+', para)
        fallback_props = [s.strip() for s in fallback_props if s.strip()]
        all_propositions.extend(fallback_props)
        print(f" -> Fallback used: {len(fallback_props)} sentences added.")

# -----------------------------
# 4. Agentic Chunking
# -----------------------------
ac = AgenticChunker()
ac.add_propositions(all_propositions)

# chunks = ac.get_chunks(get_type='list_of_strings') # 기존 코드 스타일
chunks = ac.get_chunks("list_of_strings") # 리팩토링 코드 스타일

# return 뒤에 print는 실행되지 않으므로 제거하거나 순서 변경
# print(chunks) 

return chunks
"""

