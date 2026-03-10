"""
LLM client utilities for FOMC GraphRAG inference.

Provides prompt templates and a unified ``call_llm`` function that supports
both OpenAI GPT and FriendliAI (LG EXAONE) backends with exponential-backoff
retry logic.
"""

import time
import random
import os

from openai import (
    OpenAI,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)
import openai

from src.utils.config import Config

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

sys_prompt_one = """
Rate the economic sentiment of the following text on a scale from -1.00 (very negative) to 1.00 (very positive). 
Provide the score rounded to two decimal places.
Then, identify and quote the single most important sentence that supports your rating. If no such sentence exists, write: "There is no basis."

Finally, present the result with the sentiment score in this exact format at the end of the text: "sentiment score = %float"
"""

sys_prompt_two = """
Modify the response to the question by incorporating insights from provided financial books or academic papers. 
Should adjust the sentiment score accordingly, ensuring that the assessment reflects provided literature in economics and finance.

The output must end with the sentiment score in the following exact format: "sentiment score = %float"
"""

sys_prompt_three = """
Modify the response to the question using the provided Financial Stability Report (FSR) as a reference. 
Should adjust the sentiment score based on insights from the FSR.

The final output must end with the sentiment score in the exact format: "sentiment score = %float"
"""

sys_prompt_four = """
Modify the response to the question using the provided Senior Loan Officer Opinion Survey (SLOOS) as a reference. 
Should adjust the sentiment score based on insights from the SLOOS.

The final output must end with the sentiment score in the exact format: "sentiment score = %float"
"""

sys_prompt_five = """
Modify the response to the question using the provided Beige Book as a reference. 
Should adjust the sentiment score based on insights from the Beige Book.

The final output must end with the sentiment score in the exact format: "sentiment score = %float"
"""

# ---------------------------------------------------------------------------
# Helper: retryable exception predicate
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS_CODES = {"429", "500", "502", "503"}
_RETRYABLE_KEYWORDS = {"rate limit", "timeout"}


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (APITimeoutError, APIConnectionError, InternalServerError, RateLimitError)):
        return True
    msg = str(exc).lower()
    if any(code in msg for code in _RETRYABLE_STATUS_CODES):
        return True
    if any(kw in msg for kw in _RETRYABLE_KEYWORDS):
        return True
    return False


# ---------------------------------------------------------------------------
# Main LLM caller
# ---------------------------------------------------------------------------

_MAX_ATTEMPTS = 6
_BASE_DELAY = 2.0


def call_llm(sys: str, user: str) -> str | None:
    """Call the configured LLM backend with retry on transient errors.

    Supports:
    * Any OpenAI-compatible model (``gpt-4o``, ``gpt-4o-mini``, ``gpt-5``, …)
      via the default OpenAI endpoint.
    * ``LGAI-EXAONE/K-EXAONE-236B-A23B`` via the FriendliAI serverless endpoint.

    Returns the response text on success, or ``None`` if all retries are exhausted.
    Raises immediately on non-retryable errors.
    """
    model = Config.MODEL_INFERENCE

    if model == "LGAI-EXAONE/K-EXAONE-236B-A23B":
        return _call_exaone(sys, user)

    # Default: treat any other model name as an OpenAI-compatible model.
    return _call_openai_gpt(sys, user)


def _call_openai_gpt(sys: str, user: str) -> str | None:
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = openai.chat.completions.create(
                model=Config.MODEL_INFERENCE,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": f" {user}"},
                ],
                max_completion_tokens=3000,
                temperature=1,
                n=1,
                stop=None,
            )
            return response.choices[0].message.content

        except Exception as e:
            if _is_retryable(e):
                delay = _BASE_DELAY * (2 ** attempt) + random.uniform(0.1, 0.5)
                print(f"[GPT Attempt {attempt + 1}] {str(e)[:60]}... retrying in {delay:.2f}s")
                time.sleep(delay)
                continue
            print(f"Non-retryable GPT error: {e}")
            raise

    print("GPT call exhausted all retries.")
    return None


def _call_exaone(sys: str, user: str) -> str | None:
    client = OpenAI(
        api_key=Config.FRIENDLI_TOKEN,
        base_url="https://api.friendli.ai/serverless/v1",
        timeout=90.0,
    )

    for attempt in range(_MAX_ATTEMPTS):
        try:
            completion = client.chat.completions.create(
                model="LGAI-EXAONE/K-EXAONE-236B-A23B",
                extra_body={
                    "parse_reasoning": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ],
            )
            return completion.choices[0].message.content

        except Exception as e:
            if _is_retryable(e):
                delay = _BASE_DELAY * (2 ** attempt) + random.uniform(0.1, 0.5)
                print(f"[EXAONE Attempt {attempt + 1}] {str(e)[:60]}... retrying in {delay:.2f}s")
                time.sleep(delay)
                continue
            print(f"Non-retryable EXAONE error: {e}")
            raise

    print("EXAONE call exhausted all retries.")
    return None
