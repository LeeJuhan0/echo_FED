"""
Backward-compatible re-exports for ``src.utils.common``.

The implementation has been split into focused sub-modules:

* :mod:`src.utils.llm_client`  – LLM call wrappers and system prompts
* :mod:`src.utils.graph_utils` – node/relationship property helpers and Neo4j linking
* :mod:`src.utils.retrieval`   – context-retrieval Cypher queries
* :mod:`src.utils.inference`   – sentiment scoring pipeline (``get_response``)

All names that were previously importable from this module are still available
here so that existing ``from src.utils.common import X`` statements continue to
work without modification.
"""

import os
import re
import uuid
import csv
from pathlib import Path

import pdfplumber
import pandas as pd

# ---------------------------------------------------------------------------
# Sub-module re-exports
# ---------------------------------------------------------------------------

from src.utils.llm_client import (          # noqa: F401
    sys_prompt_one,
    sys_prompt_two,
    sys_prompt_three,
    sys_prompt_four,
    sys_prompt_five,
    call_llm,
)

from src.utils.graph_utils import (         # noqa: F401
    get_embedding,
    add_text_property,
    add_sim_score,
    add_gid,
    add_ge_emb,
    fetch_texts,
    add_embeddings,
    add_nodes_emb,
    merge_similar_nodes,
    ref_link,
    ref_link_FSR,
    ref_link_SLOOS,
    ref_link_beigebook,
)

from src.utils.retrieval import (           # noqa: F401
    ret_context,
    link_context,
    link_context_FSR,
    link_context_SLOOS,
    link_context_beigebook,
)

from src.utils.inference import (           # noqa: F401
    parsing_score,
    get_response,
    save_responses_per_meeting,
)

# ---------------------------------------------------------------------------
# Utilities kept here (not yet extracted to a dedicated module)
# ---------------------------------------------------------------------------

def find_index_of_largest(nums):
    """Return the original index of the largest element in *nums*."""
    sorted_with_index = sorted((num, index) for index, num in enumerate(nums))
    return sorted_with_index[-1][1]


def str_uuid() -> str:
    """Return a random UUID as a string."""
    return str(uuid.uuid4())


def extract_pdf_to_dataframe(pdf_path: Path) -> pd.DataFrame:
    """Extract text from *pdf_path* and return a DataFrame with columns
    ``file_name``, ``page``, and ``text``.
    """
    records = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                records.append({
                    "file_name": pdf_path.stem,
                    "page": i,
                    "text": text.strip(),
                })
    return pd.DataFrame(records)


def add_sum(n4j, content: str, gid: str):
    """Create a Summary node summarising *content* and link it to graph *gid*."""
    from src.llm.summarizer import process_chunks  # local import to avoid circular deps
    summary = process_chunks(content)
    create_query = """
        CREATE (s:Summary {content: $sum, gid: $gid})
        RETURN s
    """
    n4j.query(create_query, {"sum": summary, "gid": gid})
    link_query = """
        MATCH (s:Summary {gid: $gid}), (n)
        WHERE n.gid = s.gid AND NOT n:Summary
        CREATE (s)-[:SUMMARIZES]->(n)
        RETURN s, n
    """
    return n4j.query(link_query, {"gid": gid})
