import os
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    # OpenAI Python SDK >= 1.0.0
    from openai import OpenAI
    _CLIENT = OpenAI()
    _USE_NEW_SDK = True
except Exception:
    # Fallback to legacy openai
    import openai
    _CLIENT = None
    _USE_NEW_SDK = False


# ====== 경로 설정 ======
SRC_DIR = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Statement\report_sum")
DST_DIR = Path(r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Statement\report_only_economic")

MODEL = "gpt-5"
TEMPERATURE = 1
MAX_RETRIES = 3
RETRY_SLEEP = 5  # seconds


SYSTEM_PROMPT = (
    "Return an exact excerpt from the original text. Never create a new sentence or guess the intent "
    "Just export the sentences in the input text as they are"
)

USER_PROMPT_TEMPLATE = """ Please share the source text you want
Please share the source text you want me to work from (paste the content or send a link). I’ll summarize only the current economic situation in English and exclude anything related to the FOMC or the Federal Reserve without adding or removing other material.

[text start]
{content}
[End of text]
"""


def read_csv_as_text(csv_path: Path) -> str:
    """
    CSV 파일을 읽어 모든 텍스트 셀을 하나의 큰 문자열로 합칩니다.
    인코딩은 utf-8-sig 우선, 실패 시 cp949로 재시도합니다.
    """
    encodings = ["utf-8-sig", "cp949"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc, dtype=str, keep_default_na=False)
            # 모든 셀을 문자열로 결합
            text = "\n".join(
                str(cell).strip()
                for _, row in df.iterrows()
                for cell in row.tolist()
                if str(cell).strip()
            )
            return text
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CSV 읽기 실패: {csv_path} | 마지막 오류: {last_err}")


def call_openai_extract(content: str) -> str:
    """
    OpenAI에 추출 요청을 보내고, 순수 텍스트 응답을 반환합니다.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(content=content)

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if _USE_NEW_SDK:
                resp = _CLIENT.chat.completions.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return resp.choices[0].message.content or ""
            else:
                # Legacy openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
                openai.api_key = api_key
                resp = openai.ChatCompletion.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return resp["choices"][0]["message"]["content"] or ""
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)
            else:
                raise

    # 일반적으로 도달하지 않음
    raise RuntimeError(f"OpenAI 호출 실패: {last_err}")


def normalize_lines(output_text: str) -> List[str]:
    """
    모델 응답을 줄 단위 문장 리스트로 정리:
    - 공백 트림
    - 빈 줄 제거
    - 중복 제거(순서 유지)
    """
    raw_lines = [ln.strip() for ln in output_text.splitlines()]
    raw_lines = [ln for ln in raw_lines if ln]  # non-empty

    seen = set()
    ordered_unique = []
    for ln in raw_lines:
        if ln not in seen:
            seen.add(ln)
            ordered_unique.append(ln)
    return ordered_unique


def write_sentences_to_csv(sentences: List[str], out_path: Path) -> None:
    """
    추출된 문장 리스트를 CSV로 저장합니다.
    - 컬럼: sentence
    - 인코딩: utf-8-sig (엑셀 호환)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame({"sentence": sentences})
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")


def process_one_file(csv_path: Path, dst_dir: Path) -> None:
    print(f"[처리 시작] {csv_path}")
    text = read_csv_as_text(csv_path)

    if not text.strip():
        print(f"  - 비어있는 파일(텍스트 없음). 빈 CSV로 저장.")
        write_sentences_to_csv([], dst_dir / csv_path.name)
        return

    # 단일 호출로 처리 (gpt-4o-mini는 큰 컨텍스트 지원).
    # 파일이 매우 클 경우 필요 시 여기에서 청크 분할 로직을 추가할 수 있습니다.
    resp_text = call_openai_extract(text)
    sentences = normalize_lines(resp_text)

    out_path = dst_dir / csv_path.name
    write_sentences_to_csv(sentences, out_path)
    print(f"[완료] {out_path} (문장 수: {len(sentences)})")


def main():
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"입력 폴더가 존재하지 않습니다: {SRC_DIR}")
    DST_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted([p for p in SRC_DIR.glob("*.csv") if p.is_file()])
    if not csv_files:
        print(f"입력 폴더에 CSV가 없습니다: {SRC_DIR}")
        return

    for csv_path in csv_files:
        try:
            process_one_file(csv_path, DST_DIR)
        except Exception as e:
            print(f"[오류] {csv_path} 처리 실패: {e}")


if __name__ == "__main__":
    main()