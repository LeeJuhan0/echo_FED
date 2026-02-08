import os
import sys
import re
import csv
import json
import time
import argparse
import hashlib
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# 프로젝트 루트 경로 설정 (단독 실행 시 필요)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import Config  # [NEW] Config 사용

# pip install openai (>=1.0)
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("openai 패키지가 필요합니다. `pip install openai` 로 설치하세요.")

# --------- 텍스트 전처리 / 유틸 ---------

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"[ \t]+", " ", t)
    t = "\n".join([line.strip() for line in t.splitlines()])
    t = t.strip()
    return t

def fingerprint(text: str) -> str:
    n = normalize_text(text)
    return hashlib.sha256(n.encode("utf-8")).hexdigest()

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def chunk_text_by_chars(text: str, chunk_chars: int) -> List[str]:
    text = text or ""
    n = len(text)
    return [text[i:i + chunk_chars] for i in range(0, n, chunk_chars)]

# --------- 페이지 분할 ---------

def split_into_pages(full_text: str, method: str = "auto", chars_per_page: int = 3000, page_break_pattern: Optional[str] = None) -> List[str]:
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

    pages = []
    i = 0
    n = len(text)
    while i < n:
        pages.append(text[i:i + chars_per_page].strip())
        i += chars_per_page
    return [p for p in pages if p]

def make_overlap_windows(pages: List[str], window_size: int = 4, stride: Optional[int] = None) -> List[Tuple[int, int, str]]:
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
        # [수정] Config에서 API KEY 가져오기
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

    def extract_paragraphs_json(self, chunk_text: str, language_hint: str = "ko") -> Dict[str, Any]:
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
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"OpenAI 호출 실패: {last_err}")

def parse_json_strict(s: str) -> Dict[str, Any]:
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
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
        rows_per_file: int = 2000,
        output_dir: Path = Path("책_문단"),
        language_hint: str = "ko",
        fuzzy_dedup: bool = False,
        fuzzy_threshold: float = 0.9
) -> List[str]:

    ensure_dir(output_dir)
    gpt = GPTClient(model=model)

    all_accepted_paragraphs: List[str] = []
    seen_hashes: set = set()
    recent_buffer: List[str] = []

    superchunks = chunk_text_by_chars(full_text, superchunk_chars)
    if not superchunks:
        print("️ Warning: Input text is empty.")
        return []

    print(f" Processing '{book_name}' in {len(superchunks)} superchunks...")

    for sc_idx, sc_text in enumerate(superchunks, start=1):
        print(f"\n[Superchunk {sc_idx}/{len(superchunks)}] Length={len(sc_text)} chars")

        pages = split_into_pages(sc_text, method=page_method, chars_per_page=chars_per_page, page_break_pattern=page_break_pattern)
        if not pages:
            continue

        windows = make_overlap_windows(pages, window_size=window_size, stride=window_stride)
        if not windows:
            continue

        accepted_in_chunk: List[str] = []

        for w_idx, (start, end, chunk_text) in enumerate(windows, start=1):
            try:
                data = gpt.extract_paragraphs_json(chunk_text, language_hint=language_hint)
            except Exception as e:
                print(f" Window {w_idx} Error: {e}")
                continue

            paragraphs = data.get("paragraphs", [])
            if not isinstance(paragraphs, list): continue

            for p in paragraphs:
                if not isinstance(p, dict): continue
                text = p.get("text", "")
                is_complete = p.get("is_complete", False)

                if not text or not isinstance(text, str) or not is_complete:
                    continue

                norm = normalize_text(text)
                if not norm: continue

                h = fingerprint(norm)
                if h in seen_hashes: continue

                if fuzzy_dedup and is_near_duplicate(norm, recent_buffer, threshold=fuzzy_threshold):
                    continue

                seen_hashes.add(h)
                accepted_in_chunk.append(norm)
                all_accepted_paragraphs.append(norm)
                recent_buffer.append(norm)
                if len(recent_buffer) > 2000:
                    recent_buffer = recent_buffer[-1000:]

            print(f"  > Window {w_idx}/{len(windows)} done. Accumulated: {len(accepted_in_chunk)}")

        save_paragraphs_grouped_to_csvs(accepted_in_chunk, book_name, output_dir, sc_idx, rows_per_file)

    return all_accepted_paragraphs

def ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()

def is_near_duplicate(candidate: str, recent: List[str], threshold: float = 0.9) -> bool:
    check_pool = recent[-500:] if len(recent) > 500 else recent
    for t in check_pool:
        if ratio(candidate, t) >= threshold:
            return True
    return False

def save_paragraphs_grouped_to_csvs(paragraphs: List[str], book_name: str, output_dir: Path, group_index: int, rows_per_file: int = 2000):
    if not paragraphs: return
    ensure_dir(output_dir)
    chunks = chunk_list(paragraphs, rows_per_file)
    for i, chunk in enumerate(chunks, start=1):
        file_path = output_dir / f"{book_name}_part{group_index}_{i}.csv"
        with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "paragraph"])
            for j, para in enumerate(chunk, start=1 + (i - 1) * rows_per_file):
                writer.writerow([j, para])
        print(f"  Saved: {file_path}")

# ========= [핵심] 외부 호출용 함수 =========
def run_preprocess(
        input_path: str,
        output_dir: str = None,
        book_name: str = None,
        model: str = "gpt-4o-mini",
        **kwargs
):
    """
    Pipeline에서 호출하는 진입점 함수입니다.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        print(f" Error: Input file not found: {input_path}")
        return

    # 출력 경로가 없으면 Config 설정 사용
    if output_dir is None:
        # 임시로 data/processed 폴더 사용 (Config에 추가해도 좋음)
        output_dir = input_file.parent.parent / "processed" / "paragraphs"

    out_path = Path(output_dir)
    b_name = book_name or input_file.stem

    print(f" Starting Preprocessing for: {b_name}")

    # 텍스트 읽기
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            full_text = f.read()
    except Exception as e:
        print(f" Error reading file: {e}")
        return

    # 처리 실행
    process_book_text_chunked(
        full_text=full_text,
        book_name=b_name,
        output_dir=out_path,
        model=model,
        **kwargs # 나머지 옵션(superchunk_chars 등)은 kwargs로 전달
    )
    print("✅ Preprocessing Completed.")

# ========= CLI 테스트용 =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Book Preprocessing Tool")
    parser.add_argument("--input", required=True, help="Input text file path")
    parser.add_argument("--output-dir", help="Output directory")
    args = parser.parse_args()

    run_preprocess(input_path=args.input, output_dir=args.output_dir)