"""
Inference engine for FOMC GraphRAG sentiment scoring.

Provides ``get_response`` (the core 5-step scoring pipeline) and
``parsing_score`` (robust regex-based score extraction).
"""

import os
import re
import csv

from src.utils.config import Config
from src.utils.llm_client import (
    call_llm,
    sys_prompt_one,
    sys_prompt_two,
    sys_prompt_three,
    sys_prompt_four,
    sys_prompt_five,
)
from src.utils.retrieval import (
    ret_context,
    link_context,
    link_context_FSR,
    link_context_SLOOS,
    link_context_beigebook,
)

# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

_SCORE_PATTERN = re.compile(
    r"sentiment\s+score\s*=\s*([+-]?\s*[0-9]*\.[0-9]+)", re.IGNORECASE
)


def parsing_score(res: str | None) -> float | None:
    """Extract and validate the sentiment score from an LLM response.

    * Returns a float in ``[-1.0, 1.0]`` on success.
    * Clamps out-of-range values to the valid boundary.
    * Returns ``None`` and prints a warning when parsing fails.
    """
    if not res:
        print("Warning: received empty LLM response; cannot parse score.")
        return None

    match = _SCORE_PATTERN.search(res)
    if match:
        raw = match.group(1).replace(" ", "")
        score = float(raw)
        clamped = max(-1.0, min(1.0, score))
        if clamped != score:
            print(f"Warning: score {score} out of [-1, 1]; clamped to {clamped}.")
        return clamped

    print(f"Warning: could not parse score from response snippet: {res[:120]!r}")
    return None


# ---------------------------------------------------------------------------
# Response persistence
# ---------------------------------------------------------------------------

def save_responses_per_meeting(year: int, meeting_no: int, responses: list[str]):
    """Persist raw LLM responses for a single meeting to numbered CSV files.

    Files are written to::

        <Config.DIR_OUTPUT>/<year>-<meeting_no>/<1..N>.csv
    """
    folder_path = os.path.join(Config.DIR_OUTPUT, f"{year}-{meeting_no}")
    os.makedirs(folder_path, exist_ok=True)

    for idx, response in enumerate(responses, start=1):
        file_path = os.path.join(folder_path, f"{idx}.csv")
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Response"])
            writer.writerow([response])


# ---------------------------------------------------------------------------
# Core inference pipeline
# ---------------------------------------------------------------------------

def get_response(
    n4j,
    gid: str,
    query: str,
    i: int,
    sim_score_median: float,
    year: int,
    meeting_no: int,
) -> list[float | None]:
    """Run the 5-step sentiment scoring pipeline for a single FOMC statement.

    Steps
    -----
    1. Statement-only baseline (sys_prompt_one).
    2. Refine with theory-paper context (sys_prompt_two).
    3. Refine with FSR context (sys_prompt_three).
    4. Refine with SLOOS context (sys_prompt_four).
    5. Refine with Beige Book context (sys_prompt_five).

    Returns
    -------
    list of five sentiment scores (float or None if parsing failed):
    ``[Statement, Papers, FSR, SLOOS, BeigeBook]``
    """
    # --- Retrieve all context up-front ---
    selfcont = ret_context(n4j, gid)
    linkcont = link_context(n4j, gid, i, sim_score_median)
    linkcontFSR = link_context_FSR(n4j, gid)
    linkcontSLOOS = link_context_SLOOS(n4j, gid)
    linkcontbeigebook = link_context_beigebook(n4j, gid)

    responses: list[str] = []
    score_list: list[float | None] = []

    # Step 1: Statement baseline
    user_one = f"the question is: {query} the references are: {''.join(selfcont)}"
    res = call_llm(sys_prompt_one, user_one)
    responses.append(res or "")
    score_list.append(parsing_score(res))
    print(f"[Step 1 - Statement] score={score_list[-1]}")

    # Step 2: Refine with theory papers
    user_two = (
        f"the question is: {query} "
        f"the last response is: {res or ''} "
        f"the references are: {''.join(linkcont)}"
    )
    res = call_llm(sys_prompt_two, user_two)
    responses.append(res or "")
    score_list.append(parsing_score(res))
    print(f"[Step 2 - Papers] score={score_list[-1]}")

    # Step 3: Refine with FSR
    user_three = (
        f"the question is: {query} "
        f"the provided information is: {res or ''} "
        f"the references are: {''.join(linkcontFSR)}"
    )
    res = call_llm(sys_prompt_three, user_three)
    responses.append(res or "")
    score_list.append(parsing_score(res))
    print(f"[Step 3 - FSR] score={score_list[-1]}")

    # Step 4: Refine with SLOOS
    user_four = (
        f"the question is: {query} "
        f"the provided information is: {res or ''} "
        f"the references are: {''.join(linkcontSLOOS)}"
    )
    res = call_llm(sys_prompt_four, user_four)
    responses.append(res or "")
    score_list.append(parsing_score(res))
    print(f"[Step 4 - SLOOS] score={score_list[-1]}")

    # Step 5: Refine with Beige Book
    user_five = (
        f"the question is: {query} "
        f"the provided information is: {res or ''} "
        f"the references are: {''.join(linkcontbeigebook)}"
    )
    res = call_llm(sys_prompt_five, user_five)
    responses.append(res or "")
    score_list.append(parsing_score(res))
    print(f"[Step 5 - BeigeBook] score={score_list[-1]}")

    save_responses_per_meeting(year, meeting_no, responses)
    return score_list
