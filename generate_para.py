import os
import re
import csv
import json
import time
import argparse
import hashlib
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# pip install openai (>=1.0)
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("openai 패키지가 필요합니다. `pip install openai` 로 설치하세요.")

# --------- 텍스트 전처리 / 유틸 ---------

def normalize_text(text: str) -> str:
    """
    중복 제거를 위한 정규화:
    - 유니코드 정규화(NFKC)
    - 개행/공백 정리
    - 앞뒤 공백 제거
    """
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", text)
    # 줄바꿈은 유지하되, 다중 공백을 하나로
    t = re.sub(r"[ \t]+", " ", t)
    # 줄 끝 공백 제거
    t = "\n".join([line.strip() for line in t.splitlines()])
    # 양끝 공백 제거
    t = t.strip()
    return t

def fingerprint(text: str) -> str:
    """정규화된 텍스트의 SHA256 해시."""
    n = normalize_text(text)
    return hashlib.sha256(n.encode("utf-8")).hexdigest()

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def chunk_text_by_chars(text: str, chunk_chars: int) -> List[str]:
    """
    텍스트를 chunk_chars(문자수) 단위로 분할.
    """
    text = text or ""
    n = len(text)
    return [text[i:i + chunk_chars] for i in range(0, n, chunk_chars)]

# --------- 페이지 분할 ---------

def split_into_pages(
        full_text: str,
        method: str = "auto",
        chars_per_page: int = 3000,
        page_break_pattern: Optional[str] = None
) -> List[str]:
    """
    full_text를 페이지 리스트로 반환.
    - method == "pattern": page_break_pattern(정규식) 기준 분리
    - method == "formfeed": \f 기준 분리
    - method == "chars": 고정 글자 수 기준 분리
    - method == "auto": \f 또는 패턴이 있으면 그것을 사용, 없으면 chars 사용
    """
    text = full_text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if method == "pattern" and page_break_pattern:
        parts = re.split(page_break_pattern, text)
        pages = [p.strip() for p in parts if p.strip()]
        return pages

    if method == "formfeed" or (method == "auto" and "\f" in text):
        parts = text.split("\f")
        pages = [p.strip() for p in parts if p.strip()]
        return pages

    # fallback: chars
    pages = []
    i = 0
    n = len(text)
    while i < n:
        pages.append(text[i:i + chars_per_page].strip())
        i += chars_per_page
    # 빈 페이지 제거
    pages = [p for p in pages if p]
    return pages

def make_overlap_windows(pages: List[str], window_size: int = 4, stride: Optional[int] = None) -> List[Tuple[int, int, str]]:
    """
    겹치는 윈도우 생성.
    예: window_size=4, stride=3이면 1-4, 4-7, 7-10 ...
    반환: (start_index, end_index_inclusive, joined_with_markers)
    페이지 표시는 1-based로 안내하지만 내부 인덱싱은 0-based.
    """
    if not pages:
        return []
    if stride is None:
        stride = max(1, window_size - 1)
    windows = []
    start = 0
    while start + window_size <= len(pages):
        end = start + window_size - 1
        parts = []
        for i in range(start, end + 1):
            parts.append(f"<<<<PAGE {i+1} START>>>>\n{pages[i]}\n<<<<PAGE {i+1} END>>>>")
        joined = "\n\n".join(parts)
        windows.append((start, end, joined))
        start += stride
    return windows

# --------- OpenAI 호출 ---------

class GPTClient:
    def __init__(self, model: str = "gpt-4o-mini", max_retries: int = 4, timeout: int = 120):
        self.client = OpenAI()
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

    def extract_paragraphs_json(self, chunk_text: str, language_hint: str = "ko") -> Dict[str, Any]:
        """
        chunk_text(여러 페이지 묶음)에 대해 문단 추출을 요청.
        JSON 객체로만 반환하도록 강제.
        반환 예시:
        {
          "paragraphs": [
            {"text": "...", "is_complete": true},
            ...
          ]
        }
        """
        sys_prompt = (
            "You are a precise text segmentation assistant. "
            "You receive consecutive pages of a book with clear page boundary markers like <<<<PAGE N START>>>> and <<<<PAGE N END>>>>. "
            "Your task:\n"
            "1) Segment the text into natural paragraphs (respect original paragraph boundaries; do not arbitrarily merge).\n"
            "2) For each paragraph, determine if it is complete. If the first paragraph seems to begin mid-sentence (missing start) "
            "   or the last paragraph ends abruptly (missing end), mark it as incomplete.\n"
            "3) Output ONLY a strict JSON object with the following schema:\n"
            '{\n  "paragraphs": [\n    {"text": "paragraph text", "is_complete": true|false}\n  ]\n}\n'
            "4) Do NOT include any extra commentary, code fences, or explanations.\n"
            "5) Keep the original language and punctuation of the text. Do not translate.\n"
            "6) Omit excessively short fragments that are clearly not paragraphs.\n"
        )

        user_prompt = (
            f"Language hint: {language_hint}\n"
            "Analyze the following multi-page chunk and return the JSON object:\n\n"
            f"{chunk_text}"
        )

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.1,
                    timeout=self.timeout,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = resp.choices[0].message.content
                return parse_json_strict(content)
            except Exception as e:
                last_err = e
                # 지수 백오프
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"OpenAI 호출 실패: {last_err}")

def parse_json_strict(s: str) -> Dict[str, Any]:
    """
    모델 응답에서 JSON만 안전하게 파싱.
    response_format=json_object 를 사용하지만, 안전망으로 괄호 추출도 시도.
    """
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # 코드펜스/텍스트 섞일 경우를 대비해 중괄호 블록 추출
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))

# --------- 파이프라인 (Superchunk 기반) ---------

def process_book_text_chunked(
        full_text: str,
        book_name: str,
        model: str = "gpt-4o-mini",
        superchunk_chars: int = 21000,
        page_method: str = "chars",
        chars_per_page: int = 3000,
        page_break_pattern: Optional[str] = None,
        window_size: int = 4,
        window_stride: Optional[int] = None,
        csv_rows_per_file: int = 2000,
        output_dir: Path = Path("책_문단"),
        language_hint: str = "ko",
        fuzzy_dedup: bool = False,
        fuzzy_threshold: float = 0.9
) -> List[str]:
    """
    전체 파이프라인 실행 (Superchunk → Pages → Overlap Windows):
    - full_text를 superchunk_chars(기본: 21000)씩 분할
    - 각 superchunk를 3,000자 페이지로 분할
    - 윈도우 크기 window_size(기본:4), 보폭 stride(window_size-1)로 1-4, 4-7, 7-10 ... 윈도우 생성
    - GPT 문단 추출 → 끊긴 문단 제거 → 중복 제거
    - 각 superchunk 단위로 "문단묶음{index}" 파일로 CSV 저장(파일당 rows_per_file 개씩 분할 저장)
    반환: 전체(superchunk 전체) 최종 채택된 문단 리스트
    """
    ensure_dir(output_dir)
    gpt = GPTClient(model=model)

    all_accepted_paragraphs: List[str] = []
    seen_hashes: set = set()

    # 선택적 근사 중복 제거를 위한 최근 문단 버퍼 (전 구간 공용)
    recent_buffer: List[str] = []

    # 1) Superchunk 분할
    superchunks = chunk_text_by_chars(full_text, superchunk_chars)
    if not superchunks:
        print("경고: 입력 텍스트가 비어있습니다.")
        return []

    for sc_idx, sc_text in enumerate(superchunks, start=1):
        print(f"\n[Superchunk {sc_idx}/{len(superchunks)}] 길이={len(sc_text)} 문자")

        # 2) Superchunk 내 페이지 분할
        pages = split_into_pages(
            full_text=sc_text,
            method=page_method,
            chars_per_page=chars_per_page,
            page_break_pattern=page_break_pattern
        )
        if not pages:
            print(f"[Superchunk {sc_idx}] 경고: 페이지가 비어있습니다.")
            continue

        # 3) Overlap 윈도우 생성 (예: 1-4, 4-7 ...)
        windows = make_overlap_windows(pages, window_size=window_size, stride=window_stride)
        if not windows:
            print(f"[Superchunk {sc_idx}] 경고: 윈도우가 생성되지 않았습니다. 페이지 수가 충분하지 않을 수 있습니다.")
            continue

        accepted_in_chunk: List[str] = []

        for w_idx, (start, end, chunk_text) in enumerate(windows, start=1):
            try:
                data = gpt.extract_paragraphs_json(chunk_text, language_hint=language_hint)
            except Exception as e:
                print(f"[Superchunk {sc_idx}] [윈도우 {start+1}-{end+1}] OpenAI 오류로 건너뜀: {e}")
                continue

            paragraphs = data.get("paragraphs", [])
            if not isinstance(paragraphs, list):
                print(f"[Superchunk {sc_idx}] [윈도우 {start+1}-{end+1}] 비정상 JSON(문단 리스트 아님) 건너뜀")
                continue

            for p in paragraphs:
                if not isinstance(p, dict):
                    continue
                text = p.get("text", "")
                is_complete = p.get("is_complete", False)
                if not text or not isinstance(text, str):
                    continue
                if not is_complete:
                    # AI가 끊겼다고 판단 -> 버림
                    continue

                norm = normalize_text(text)
                if not norm:
                    continue

                h = fingerprint(norm)
                if h in seen_hashes:
                    continue

                if fuzzy_dedup and is_near_duplicate(norm, recent_buffer, threshold=fuzzy_threshold):
                    continue

                seen_hashes.add(h)
                accepted_in_chunk.append(norm)
                all_accepted_paragraphs.append(norm)
                recent_buffer.append(norm)
                # 메모리 과다 방지: 근사 비교 버퍼 제한
                if len(recent_buffer) > 2000:
                    recent_buffer = recent_buffer[-1000:]

            # 간단한 진행 로그
            print(f"[Superchunk {sc_idx}] 진행: 윈도우 {w_idx}/{len(windows)} 처리, 누적 문단(현재 묶음)={len(accepted_in_chunk)}")

        # 4) Superchunk 단위 저장 ("문단묶음{sc_idx}")
        save_paragraphs_grouped_to_csvs(
            paragraphs=accepted_in_chunk,
            book_name=book_name,
            output_dir=output_dir,
            group_index=sc_idx,
            rows_per_file=csv_rows_per_file
        )

    return all_accepted_paragraphs

def ratio(a: str, b: str) -> float:
    # 간단하고 빠른 유사도: 공백 제거 후 LCS 근사. 정확도보다 속도 우선.
    # 더 정교하게 하려면 python-Levenshtein 또는 rapidfuzz 사용 권장.
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()

def is_near_duplicate(candidate: str, recent: List[str], threshold: float = 0.9) -> bool:
    # 최근 항목 위주로 유사도 비교하여 근사 중복 제거
    check_pool = recent[-500:] if len(recent) > 500 else recent
    for t in check_pool:
        if ratio(candidate, t) >= threshold:
            return True
    return False

def save_paragraphs_grouped_to_csvs(
        paragraphs: List[str],
        book_name: str,
        output_dir: Path,
        group_index: int,
        rows_per_file: int = 2000
):
    """
    각 superchunk(묶음) 단위로 CSV 저장.
    파일명: {book_name}_문단묶음{group_index}_{part}.csv
    (part는 파일 분할 시 1부터 증가)
    """
    if not paragraphs:
        print(f"[Superchunk {group_index}] 저장할 문단이 없습니다.")
        return

    ensure_dir(output_dir)

    chunks = chunk_list(paragraphs, rows_per_file)
    for i, chunk in enumerate(chunks, start=1):
        file_path = output_dir / f"{book_name}_문단묶음{group_index}_{i}.csv"
        with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "paragraph"])
            # id는 묶음 내부에서 연속 번호를 매김
            for j, para in enumerate(chunk, start=1 + (i - 1) * rows_per_file):
                writer.writerow([j, para])
        print(f"[Superchunk {group_index}] 저장 완료: {file_path} (문단 {len(chunk)}개)")

# --------- CLI ---------

def main():
    parser = argparse.ArgumentParser(description="OpenAI(gpt-4o-mini)로 책 문단 분류/정제 파이프라인 (Superchunk 처리)")
    parser.add_argument("--input", default=r"C:/Users/HUFS_MATH/IdeaProjects/FOMC_Graphrag/Theory/pdf/demo.csv", help="책 전문 텍스트 파일 경로(.txt/.csv 등 텍스트로 읽음)")
    parser.add_argument("--book-name", default="N. Gregory Mankiw. Macroeconomics  9th edition", help="책 이름(파일명 기본값 사용 가능)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI 모델명 (기본: gpt-4o-mini)")

    # Superchunk/페이지/윈도우 설정
    parser.add_argument("--superchunk-chars", type=int, default=21000, help="LLM 입력 제한을 위한 superchunk 크기(문자수, 기본: 21000)")
    parser.add_argument("--page-method", default="chars", choices=["auto", "formfeed", "chars", "pattern"], help="페이지 분할 방식 (기본: chars)")
    parser.add_argument("--chars-per-page", type=int, default=3000, help="고정 글자수 기준 페이지 크기(기본: 3000)")
    parser.add_argument("--page-break-pattern", default=None, help="정규식 패턴으로 페이지 분리 시 사용")
    parser.add_argument("--window-size", type=int, default=4, help="슬라이딩 윈도우 크기(예: 4면 1-4, 4-7, 7-10)")
    parser.add_argument("--window-stride", type=int, default=None, help="윈도우 보폭(기본: window_size-1). 4면 3이 되어 1-4, 4-7 형태")

    parser.add_argument("--csv-rows-per-file", type=int, default=2000, help="CSV 한 파일당 문단 수(기본: 2000)")
    parser.add_argument("--output-dir", default="책_문단_2", help="CSV 저장 폴더명(기본: 책_문단)")
    parser.add_argument("--language-hint", default="ko", help="언어 힌트(기본: ko)")
    parser.add_argument("--fuzzy-dedup", action="store_true", help="근사 중복 제거 활성화")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.9, help="근사 중복 판정 임계값(기본: 0.9)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"입력 파일이 존재하지 않습니다: {input_path}")

    book_name = args.book_name or input_path.stem

    with open(input_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("환경변수 OPENAI_API_KEY를 설정하세요.")

    process_book_text_chunked(
        full_text=full_text,
        book_name=book_name,
        model=args.model,
        superchunk_chars=args.superchunk_chars,
        page_method=args.page_method,
        chars_per_page=args.chars_per_page,
        page_break_pattern=args.page_break_pattern,
        window_size=args.window_size,
        window_stride=args.window_stride,
        csv_rows_per_file=args.csv_rows_per_file,
        output_dir=Path(args.output_dir),
        language_hint=args.language_hint,
        fuzzy_dedup=args.fuzzy_dedup,
        fuzzy_threshold=args.fuzzy_threshold
    )

if __name__ == "__main__":
    main()