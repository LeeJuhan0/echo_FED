#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
문서 폴더를 돌면서 문서당 N번 OpenAI에 감정점수를 물어보고,
각 문서의 N개 점수를 한 행(row)으로 CSV에 저장합니다 (값은 쉼표로 구분, 문서마다 줄바꿈).

Usage:
  - 기본 입력 폴더를 사용하려면 인자 없이 실행:
      python compute_sentiments_and_save_csv.py
  - 다른 폴더를 지정하려면:
      python compute_sentiments_and_save_csv.py "C:\path\to\folder"

출력:
 - scores_5perrow.csv      : 각 행이 한 문서, 열은 score_1 ... score_N (기본 N=5)
 - scores_by_file.csv      : filename,score_1,...,score_N

변경 요지:
 - OpenAI 최신 클라이언트 방식으로 우선 시도해서 client = OpenAI(...) 형태로 호출합니다.
 - (하위호환) 이전 openai 패키지 인터페이스가 있을 경우에도 동작하도록 폴백 처리합니다.
 - 모델 응답에서 우선 "sentiment score = <float>" 패턴을 찾고, 없으면 일반 숫자 추출으로 폴백합니다.
 - 점수는 소수점 2자리로 포맷되어 CSV에 저장됩니다.
"""

import os
import sys
import time
import re
import csv
import argparse
from glob import glob
from typing import List, Optional

from tqdm import tqdm

# Try to import new OpenAI client (migration-aware). If not available, fall back to legacy module.
NEW_OPENAI_CLIENT = False
OpenAIClientClass = None
try:
    # New-style SDK: from openai import OpenAI
    from openai import OpenAI as OpenAIClientClass  # type: ignore
    NEW_OPENAI_CLIENT = True
except Exception:
    try:
        import openai  # legacy
        OpenAIClientClass = None
        NEW_OPENAI_CLIENT = False
    except Exception as e:
        raise SystemExit("패키지 'openai'가 필요합니다. 'pip install openai'로 설치하십시오.") from e

# ----------------- 사용자 설정 -----------------
DEFAULT_MODEL = "gpt-5"          # 기본 모델: 필요시 변경하세요 (원하시면 gpt-5로 바꿔도 됨)
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0            # seconds
TRIM_CHARS = 3500
DEFAULT_ASKS_PER_DOC = 5         # 문서당 질문 수 (요청대로 기본 5)
DELAY_BETWEEN_ASKS = 0.3         # 각 질의 사이 대기(초)

# 기본 입력 폴더 (스크립트를 인자 없이 실행할 때 사용)
DEFAULT_INPUT_FOLDER = r"C:/Users/HUFS_MATH/IdeaProjects/FOMC_Graphrag/report_sum"
# ------------------------------------------------

def get_api_key_from_env() -> str:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
    if not key:
        raise SystemExit("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return key

def list_documents(folder: str, patterns: List[str] = None) -> List[str]:
    if patterns is None:
        patterns = ["*.txt", "*.md", "*.csv"]
    files = []
    for p in patterns:
        files.extend(glob(os.path.join(folder, p)))
    files = sorted(files)
    return files

def read_file_text(path: str, encoding: str = "utf-8") -> str:
    try:
        with open(path, "r", encoding=encoding, errors="replace") as f:
            return f.read()
    except Exception as e:
        print(f"[WARN] 파일을 읽는 중 에러({path}): {e}")
        return ""

def _make_client(api_key: str):
    """
    새 OpenAI 클라이언트가 가능하면 인스턴스화하여 반환하고,
    아니면 기존 openai 모듈 (legacy) 를 사용하도록 None을 반환합니다.
    """
    if NEW_OPENAI_CLIENT and OpenAIClientClass is not None:
        # instantiate new client (api_key can be omitted if env var is set)
        try:
            client = OpenAIClientClass(api_key=api_key)
            return client
        except Exception:
            # fallback to legacy module if instantiation fails
            return None
    else:
        return None

def _call_chat_completion(client, model: str, messages: List[dict], temperature: float = 1.0, max_tokens: int = 3000):
    """
    client가 new-style이면 client.chat.completions.create(...) 호출,
    아니면 openai.ChatCompletion.create(...) (legacy) 호출.
    반환: 응답 텍스트(스트링). 예외는 상위에서 처리.
    """
    if client is not None:
        # new client
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
            max_completion_tokens=3000,
        )
        # new client returns pydantic model-like object; choices[0].message.content 접근
        try:
            return resp.choices[0].message.content
        except Exception:
            # 시리얼라이즈 가능한 경우 dict 로 변환 후 접근 시도
            try:
                d = dict(resp)
                return d.get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception:
                return ""
    else:
        # legacy openai module path
        import openai as legacy_openai  # type: ignore
        resp = legacy_openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # legacy response structure: resp['choices'][0]['message']['content']
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            try:
                return resp["choices"][0]["text"]
            except Exception:
                return ""

def ask_openai_for_score_once(text: str, client, model: str, trim_chars: Optional[int]) -> Optional[float]:
    """
    한 번의 OpenAI 호출로부터 점수를 얻으려 시도.
    우선 'sentiment score = X' 패턴을 찾고, 없으면 일반 숫자 추출으로 폴백.
    """
    if trim_chars is not None and len(text) > trim_chars:
        send_text = text[:trim_chars] + "\n\n[TRUNCATED]"
    else:
        send_text = text

    system_prompt = (
        "Rate the economic sentiment of the following text on a scale from -1.00 (very negative) to 1.00 (very positive). "
        "Provide the score rounded to two decimal places. Then, identify and quote the single most important sentence that supports your rating. "
        "If no such sentence exists, write: \"There is no basis.\" "
        "Finally, present the result with the sentiment score in this exact format at the end of the text: \"sentiment score = %float\""
    )
    user_prompt = (
        "Document (start):\n"
        f"{send_text}\n"
        "Document (end).\n\n"
        "Answer according to the system instruction. Ensure you include the exact token sequence 'sentiment score = X' (with the numeric X) somewhere in the response."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content = _call_chat_completion(client, model=model, messages=messages, temperature=0.0, max_tokens=300)
    if content is None:
        content = ""
    content = str(content).strip()
    print(content)

    # 1) 우선적으로 정확한 패턴 'sentiment score = <number>' 찾기 (대소문자 무시)
    m = re.search(r"sentiment\s*score\s*=\s*(-?\d+(?:\.\d+)?)", content, flags=re.IGNORECASE)
    print(m)
    if m:
        try:
            val = float(m.group(1))
            val = max(-1.0, min(1.0, val))
            return val
        except Exception:
            pass

    # 2) 폴백: 어떤 숫자라도 추출
    m2 = re.search(r"(-?\d+\.\d+|-?\d+)", content)
    if m2:
        try:
            val = float(m2.group(0))
            val = max(-1.0, min(1.0, val))
            return val
        except Exception:
            pass

    print(f"[WARN] 응답에서 점수 패턴을 찾지 못함. 응답 내용: {content!r}")
    return None

def ask_openai_for_score(text: str, client, asks_per_doc: int, model: str, trim_chars: Optional[int]) -> List[Optional[float]]:
    scores: List[Optional[float]] = []
    for ask_i in range(asks_per_doc):
        val = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                val = ask_openai_for_score_once(text, client=client, model=model, trim_chars=trim_chars)
                if val is not None:
                    break
                else:
                    print(f"[WARN] 파싱 실패(문서 질문 {ask_i+1}/{asks_per_doc}), 시도 {attempt}/{MAX_RETRIES}")
            except Exception as e:
                print(f"[WARN] OpenAI 요청 중 예외: {e} (문서 질문 {ask_i+1}/{asks_per_doc}, 시도 {attempt}/{MAX_RETRIES})")
            backoff = INITIAL_BACKOFF * (2 ** (attempt - 1))
            time.sleep(backoff)
        scores.append(val)
        time.sleep(DELAY_BETWEEN_ASKS)
    return scores

def save_scores_5perrow_csv(rows: List[List[Optional[float]]], out_path: str):
    max_cols = max((len(r) for r in rows), default=0)
    header = [f"score_{i+1}" for i in range(max_cols)]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            row_out: List[str] = []
            for v in r:
                if v is None:
                    row_out.append("")
                else:
                    row_out.append(f"{v:.2f}")
            if len(row_out) < max_cols:
                row_out.extend([""] * (max_cols - len(row_out)))
            writer.writerow(row_out)
    print(f"Saved per-row CSV: {out_path}")

def save_scores_by_file_csv(filenames: List[str], rows: List[List[Optional[float]]], out_path: str):
    max_cols = max((len(r) for r in rows), default=0)
    header = ["filename"] + [f"score_{i+1}" for i in range(max_cols)]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for fn, r in zip(filenames, rows):
            row_out = [os.path.basename(fn)]
            for v in r:
                if v is None:
                    row_out.append("")
                else:
                    row_out.append(f"{v:.2f}")
            if len(row_out) - 1 < max_cols:
                row_out.extend([""] * (max_cols - (len(row_out) - 1)))
            writer.writerow(row_out)
    print(f"Saved filename->scores CSV: {out_path}")

def main(folder: str, output_dir: Optional[str] = None, asks_per_doc: int = DEFAULT_ASKS_PER_DOC, model: str = DEFAULT_MODEL, trim_chars: Optional[int] = TRIM_CHARS):
    api_key = get_api_key_from_env()
    # try to create a new-style client; if not possible, we'll use legacy openai module
    client = _make_client(api_key)

    # if no new-style client, set legacy API key
    if client is None:
        try:
            import openai as legacy_openai  # type: ignore
            legacy_openai.api_key = api_key
        except Exception:
            pass

    if not os.path.isdir(folder):
        raise SystemExit(f"입력 폴더가 존재하지 않습니다: {folder}")

    files = list_documents(folder, patterns=["*.txt", "*.md", "*.csv"])
    if not files:
        raise SystemExit(f"문서(예: .txt/.md/.csv)가 '{folder}'에 없습니다. 패턴을 변경하려면 스크립트를 수정하세요.")

    if output_dir is None:
        output_dir = folder
    os.makedirs(output_dir, exist_ok=True)

    all_rows: List[List[Optional[float]]] = []
    processed_files: List[str] = []

    print(f"총 {len(files)}개 파일을 처리합니다. 모델={model}, asks_per_doc={asks_per_doc}, trim_chars={trim_chars}")
    for filepath in tqdm(files, desc="files"):
        text = read_file_text(filepath)
        if not text.strip():
            print(f"[WARN] 파일 비어있음: {filepath} -> 빈 점수(빈칸)으로 기록")
            row = [None] * asks_per_doc
            all_rows.append(row)
            processed_files.append(filepath)
            continue
        scores_row = ask_openai_for_score(text, client=client, asks_per_doc=asks_per_doc, model=model, trim_chars=trim_chars)
        all_rows.append(scores_row)
        processed_files.append(filepath)

    out_rows = os.path.join(output_dir, "scores_5perrow.csv")
    out_byfile = os.path.join(output_dir, "scores_by_file.csv")
    save_scores_5perrow_csv(all_rows, out_rows)
    save_scores_by_file_csv(processed_files, all_rows, out_byfile)

    print("완료. 출력 파일:")
    print(" -", out_rows)
    print(" -", out_byfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="폴더 내 문서들에 대해 OpenAI로 감정점수(문서당 N회) 요청 후 CSV로 저장")
    parser.add_argument("folder", nargs="?", help="문서들이 들어있는 폴더 경로", default=DEFAULT_INPUT_FOLDER)
    parser.add_argument("--out", help="출력 디렉토리 (기본: 입력 폴더)", default=None)
    parser.add_argument("--asks", help=f"문서당 질문 수 (기본: {DEFAULT_ASKS_PER_DOC})", default=str(DEFAULT_ASKS_PER_DOC))
    parser.add_argument("--model", help=f"OpenAI 모델 (기본: {DEFAULT_MODEL})", default=DEFAULT_MODEL)
    parser.add_argument("--trim", help="문서가 길 경우 전송할 최대 문자수 (None이면 전체 전송)", default=str(TRIM_CHARS))
    args = parser.parse_args()

    asks_val = int(args.asks) if args.asks.isdigit() else DEFAULT_ASKS_PER_DOC
    trim_val = None if args.trim.lower() in ("none", "null", "0", "") else int(args.trim)
    main(args.folder, output_dir=args.out, asks_per_doc=asks_val, model=args.model, trim_chars=trim_val)