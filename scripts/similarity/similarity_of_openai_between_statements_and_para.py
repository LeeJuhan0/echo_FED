# -*- coding: utf-8 -*-
import os
import re
import io
import glob
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd

# OpenAI SDK v1+
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# tiktoken (선택)
try:
    import tiktoken
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False


# 기본 경로 (요청하신 폴더/파일)
DEFAULT_PARAGRAPHS_XLSX = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simscore관련\combined_book_paragraphs.xlsx"
DEFAULT_STATEMENTS_DIR  = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Statement\report_only_economic"
DEFAULT_OUTPUT_XLSX     = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simscore관련\openai_paragraph_vs_statements_only_economic1.xlsx"

# OpenAI 임베딩 설정
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
_MAX_EMBED_TOKENS = 8000     # 입력 토큰 상한(모델 한도보다 약간 낮게)
_EMBED_BATCH_SIZE = 100      # 임베딩 API 배치 크기


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).split())


def natural_key(s: str):
    # 숫자 포함 파일명의 자연 정렬
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def read_csv_infer_delim(path: str) -> pd.DataFrame:
    """
    CSV/TSV를 다양한 인코딩으로 시도하여 읽습니다.
    - 구분자 자동 추론(sep=None, engine='python')
    - 인코딩 후보 다중 시도
    - chardet 감지(있으면)
    - 최종 폴백: 'replace'로 손실 복구 후 파싱
    - 일부 깨진 줄은 on_bad_lines='skip'로 건너뜀
    """
    enc_candidates = [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "latin1",
        "iso-8859-1",
    ]
    last_err = None

    # 1) 고정 후보 시도
    for enc in enc_candidates:
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=enc,
                on_bad_lines="skip",   # 손상된 행 건너뛰기
            )
        except Exception as e:
            last_err = e
            continue

    # 2) chardet로 인코딩 감지 시도(설치된 경우)
    try:
        import chardet  # pip install chardet
        with open(path, "rb") as f:
            raw = f.read(2_000_000)  # 처음 2MB로 감지
        guess = chardet.detect(raw).get("encoding")
        if guess:
            try:
                return pd.read_csv(
                    path,
                    sep=None,
                    engine="python",
                    encoding=guess,
                    on_bad_lines="skip",
                )
            except Exception as e:
                last_err = e
    except Exception as e:
        # chardet 미설치 또는 실패는 무시하고 폴백 진행
        last_err = e

    # 3) 최종 폴백: 바이너리를 강제로 UTF-8로 'replace' 디코드하여 파싱
    try:
        with open(path, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        return pd.read_csv(
            io.StringIO(text),
            sep=None,
            engine="python",
            on_bad_lines="skip",
        )
    except Exception as e:
        raise RuntimeError(
            f"CSV/TSV 읽기 실패: {path}\n마지막 오류: {last_err}\n폴백 오류: {e}"
        )


def read_excel_first_sheet(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    errs = []
    sheet = 0 if sheet_name is None else sheet_name
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception as e1:
        errs.append(e1)
        try:
            return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        except Exception as e2:
            errs.append(e2)
            try:
                return pd.read_excel(path, sheet_name=sheet, engine="xlrd")
            except Exception as e3:
                errs.append(e3)
                raise RuntimeError(f"엑셀 읽기 실패: {path}\n" + "\n".join(map(str, errs)))


def ensure_paragraphs(df: pd.DataFrame) -> pd.DataFrame:
    """
    combined_book_paragraphs.xlsx에서
    - 식별자: book_name_index (없으면 id/idx/index/번호 중 하나 사용)
    - 텍스트: paragraph
    를 확보하여 반환.
    """
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    key_col = lower.get("book_name_index")
    para_col = lower.get("paragraph")

    if key_col is None:
        for cand in ["id", "idx", "index", "번호"]:
            if cand in lower:
                key_col = lower[cand]
                break

    if para_col is None:
        for cand in ["paragraph", "text", "content", "body", "문단", "para"]:
            if cand in lower:
                para_col = lower[cand]
                break

    if key_col is None or para_col is None:
        raise ValueError(f"필수 컬럼 미존재(book_name_index/id, paragraph). 실제 컬럼: {cols}")

    out = df[[key_col, para_col]].rename(columns={key_col: "book_name_index", para_col: "paragraph"}).copy()
    out["book_name_index"] = out["book_name_index"].astype(str).map(normalize_text)
    out["paragraph"] = out["paragraph"].astype(str).map(normalize_text)
    out = out[out["paragraph"].str.len() > 0].reset_index(drop=True)
    return out


def choose_statement_text_column(df: pd.DataFrame) -> str:
    # 후보 우선
    for c in ["text", "paragraph", "content", "body", "statement"]:
        if c in df.columns:
            return c
    # 문자열형 컬럼 중 가장 총 글자 수가 큰 컬럼 선택
    text_like = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("string")]
    if not text_like:
        # 전부 문자열로 본 다음 첫 컬럼 사용 유도
        return df.columns[0]
    def col_len_sum(c):
        return df[c].dropna().astype(str).str.len().sum()
    return max(text_like, key=col_len_sum)


def aggregate_statement_csv(csv_path: str) -> str:
    df = read_csv_infer_delim(csv_path)
    col = choose_statement_text_column(df)
    s = df[col].dropna().astype(str).map(normalize_text)
    s = s[s.str.len() > 0]
    return " ".join(s.tolist())


def extract_col_label_from_filename(path: str) -> str:
    """
    monetary20170201a1.csv -> '201702~'
    'monetary' 뒤의 6자리(YYYYMM)를 추출.
    """
    base = os.path.basename(path)
    m = re.search(r"monetary(\d{6})", base)
    if m:
        return f"{m.group(1)}~"
    # 폴백: 파일명에서 연속 6자리 숫자 첫 매칭
    m2 = re.search(r"(\d{6})", base)
    return f"{m2.group(1)}~" if m2 else base


def gather_statement_docs(statements_dir: str) -> List[str]:
    if not os.path.isdir(statements_dir):
        raise FileNotFoundError(f"디렉토리가 아닙니다: {statements_dir}")
    files = glob.glob(os.path.join(statements_dir, "*.csv"))
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=natural_key)
    if not files:
        raise FileNotFoundError(f"CSV가 없습니다: {statements_dir}")
    return files


# ---------------- OpenAI 임베딩 유틸 ----------------
def _get_encoder(model: str):
    if not _HAS_TIKTOKEN:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _chunk_text_by_tokens(text: str, enc, max_tokens: int) -> List[str]:
    if enc is None:
        # 폴백: 문자 기준
        return [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]
    toks = enc.encode(text)
    return [enc.decode(toks[i:i+max_tokens]) for i in range(0, len(toks), max_tokens)]


def _maybe_truncate_by_tokens(texts: List[str], enc, max_tokens: int) -> List[str]:
    if enc is None:
        return [t[:max_tokens] if len(t) > max_tokens else t for t in texts]
    out = []
    for t in texts:
        tok = enc.encode(t)
        out.append(enc.decode(tok[:max_tokens]) if len(tok) > max_tokens else t)
    return out


def _embed_texts_in_batches(client, model: str, texts: List[str], batch_size: int) -> List[np.ndarray]:
    embs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        embs.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
    return embs


def _l2_normalize(M: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(M, axis=axis, keepdims=True) + eps
    return M / denom


def build_openai_similarity_matrix(
        paragraphs: List[str],
        docs: List[str],
        model: str = DEFAULT_OPENAI_MODEL,
        max_embed_tokens: int = _MAX_EMBED_TOKENS,
        batch_size: int = _EMBED_BATCH_SIZE,
) -> np.ndarray:
    """
    OpenAI 임베딩으로 각 paragraph(행) vs 각 doc(열) 코사인 유사도 행렬 계산.
    - paragraphs: 길면 자르기(토큰 기준)
    - docs: 길면 청크 분할 후 임베딩 평균
    - 임베딩 호출은 배치 처리
    """
    if not _HAS_OPENAI:
        raise RuntimeError("openai 패키지가 없습니다. pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("환경 변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")

    client = OpenAI(api_key=api_key)
    enc = _get_encoder(model)

    # 1) paragraphs 임베딩
    paras_proc = _maybe_truncate_by_tokens(paragraphs, enc, max_embed_tokens)
    A_list = _embed_texts_in_batches(client, model, paras_proc, batch_size=batch_size)
    A = np.stack(A_list, axis=0)  # (n_para, d)

    # 2) docs 임베딩: 청크 분할 → 임베딩 평균
    D_list: List[np.ndarray] = []
    for doc in docs:
        chunks = _chunk_text_by_tokens(doc, enc, max_embed_tokens)
        if not chunks:
            # 빈 문서는 영벡터로 대체(유사도 0)
            D_list.append(np.zeros_like(A[0]))
            continue
        emb_chunks = _embed_texts_in_batches(client, model, chunks, batch_size=batch_size)
        D_list.append(np.stack(emb_chunks, axis=0).mean(axis=0))
    D = np.stack(D_list, axis=0)  # (n_docs, d)

    # 3) 코사인 유사도 행렬: (n_para, n_docs)
    A_n = _l2_normalize(A, axis=1)
    D_n = _l2_normalize(D, axis=1)
    S = A_n @ D_n.T
    return S


def main():
    ap = argparse.ArgumentParser(description="paragraphs(엑셀) vs report_sum 폴더의 각 monetary*.csv (OpenAI 임베딩 코사인 유사도) 매트릭스를 엑셀로 저장")
    ap.add_argument("--paragraphs-xlsx", type=str, default=DEFAULT_PARAGRAPHS_XLSX, help="combined_book_paragraphs.xlsx 경로")
    ap.add_argument("--paragraphs-sheet", type=str, default=None, help="엑셀 시트명(미지정 시 첫 시트)")
    ap.add_argument("--statements-dir", type=str, default=DEFAULT_STATEMENTS_DIR, help="monetary*.csv 들이 있는 폴더 경로")
    ap.add_argument("--output-xlsx", type=str, default=DEFAULT_OUTPUT_XLSX, help="출력 엑셀 경로")
    # 선택적으로 모델/배치/토큰 상한도 바꿀 수 있게 하지만 기본값 유지
    ap.add_argument("--openai-model", type=str, default=DEFAULT_OPENAI_MODEL, help="OpenAI 임베딩 모델 (기본: text-embedding-3-small)")
    ap.add_argument("--batch-size", type=int, default=_EMBED_BATCH_SIZE, help="임베딩 API 호출 배치 크기")
    ap.add_argument("--max-embed-tokens", type=int, default=_MAX_EMBED_TOKENS, help="입력 당 최대 토큰(청크/자르기 기준)")
    args = ap.parse_args()

    # 1) 문단 읽기
    paras_df_raw = read_excel_first_sheet(args.paragraphs_xlsx, sheet_name=args.paragraphs_sheet)
    paras_df = ensure_paragraphs(paras_df_raw)  # ['book_name_index','paragraph']
    paragraphs = paras_df["paragraph"].tolist()
    keys = paras_df["book_name_index"].tolist()

    # 2) 스테이트먼트 CSV 수집 및 전처리
    csv_files = gather_statement_docs(args.statements_dir)

    col_labels: List[str] = []
    docs: List[str] = []
    for fp in csv_files:
        try:
            doc = aggregate_statement_csv(fp)
            if not doc.strip():
                # 비어 있으면 스킵
                continue
            label = extract_col_label_from_filename(fp)
            col_labels.append(label)
            docs.append(doc)
            print(f"[INFO] {os.path.basename(fp)} -> label={label}, chars={len(doc)}")
        except Exception as e:
            print(f"[WARN] '{fp}' 읽기/전처리 중 오류: {e}")

    if not docs:
        raise RuntimeError("유효한 statement 문서가 없습니다(모두 빈 텍스트이거나 읽기 실패).")

    # 3) OpenAI 임베딩 코사인 유사도 행렬 (paragraphs x docs)
    S = build_openai_similarity_matrix(
        paragraphs,
        docs,
        model=args.openai_model,
        max_embed_tokens=args.max_embed_tokens,
        batch_size=args.batch_size,
    )  # shape: (n_para, n_docs)

    # 4) 결과 DataFrame 구성
    out_df = pd.DataFrame(S, columns=col_labels)
    out_df.insert(0, "book_name_index", keys)

    # 5) 저장
    try:
        out_df.to_excel(args.output_xlsx, index=False)
        print(f"[DONE] Saved OpenAI similarity matrix: {os.path.abspath(args.output_xlsx)}")
        print(f"[INFO] shape: {out_df.shape}, columns: {len(out_df.columns)-1} statements, rows: {len(out_df)} paragraphs")
    except Exception as e:
        print(f"[ERROR] 엑셀 저장 실패: {e}")


if __name__ == "__main__":
    main()