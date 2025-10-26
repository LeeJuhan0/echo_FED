# -*- coding: utf-8 -*-
import os
import argparse
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


# 고정 기본 경로 (요청하신 statement 하나 + paragraphs 엑셀 파일)
DEFAULT_PARAGRAPHS_CSV = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Simscore관련\combined_book_paragraphs.xlsx"
DEFAULT_STATEMENT_CSV  = r"C:\Users\HUFS_MATH\IdeaProjects\FOMC_Graphrag\Statement\report_sum\monetary20170503a1.csv"
DEFAULT_OUTPUT_XLSX    = "similarity_results202003.xlsx"
DEFAULT_OPENAI_MODEL   = "text-embedding-3-small"

# OpenAI 임베딩 안전 파라미터
_MAX_EMBED_TOKENS = 8000   # 모델 한도(예: 8192)보다 약간 낮게
_EMBED_BATCH_SIZE = 100    # 요청 배치 크기


def read_csv_infer_delim(path: str) -> pd.DataFrame:
    """
    CSV/TSV 구분자 자동 추론.
    """
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CSV/TSV 파일을 읽는 중 오류가 발생했습니다: {path}\n{last_err}")


def _coerce_excel_result_to_df(ret) -> pd.DataFrame:
    """
    read_excel이 dict(복수 시트)면 첫 시트를 DataFrame으로 강제.
    """
    if isinstance(ret, dict):
        try:
            return next(iter(ret.values()))
        except StopIteration:
            raise RuntimeError("엑셀 파일에서 읽을 시트를 찾지 못했습니다(빈 통합 문서).")
    return ret


def read_excel_safely(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    엑셀(.xlsx/.xls/.xlsm) 안전 읽기. sheet_name 미지정 시 첫 시트.
    """
    errs = []
    sheet = 0 if sheet_name is None else sheet_name
    try:
        ret = pd.read_excel(path, sheet_name=sheet)
        return _coerce_excel_result_to_df(ret)
    except Exception as e1:
        errs.append(e1)
        try:
            ret = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
            return _coerce_excel_result_to_df(ret)
        except Exception as e2:
            errs.append(e2)
            try:
                ret = pd.read_excel(path, sheet_name=sheet, engine="xlrd")
                return _coerce_excel_result_to_df(ret)
            except Exception as e3:
                errs.append(e3)
                msg = "\n".join([str(e) for e in errs])
                raise RuntimeError(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {path}\n{msg}")


def read_table_auto(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    확장자에 따라 CSV/TSV 또는 Excel 자동 판별.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    base = os.path.basename(path)
    if base.startswith("~$"):
        warnings.warn(f"'{base}'는 엑셀 잠금 임시 파일일 수 있습니다. 원본 파일을 지정하세요.", RuntimeWarning)

    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv", ".txt"):
        return read_csv_infer_delim(path)
    elif ext in (".xlsx", ".xls", ".xlsm"):
        return read_excel_safely(path, sheet_name=sheet_name)
    else:
        # 모호하면 CSV → Excel 순서로 시도
        try:
            return read_csv_infer_delim(path)
        except Exception as e_csv:
            try:
                return read_excel_safely(path, sheet_name=sheet_name)
            except Exception as e_xlsx:
                raise RuntimeError(
                    f"지원하지 않는 파일 형식이거나 읽기 실패: {path}\nCSV 오류: {e_csv}\nExcel 오류: {e_xlsx}"
                )


def normalize_text(s: str) -> str:
    """
    간단 전처리: 문자열화 + 공백 정리.
    """
    if s is None:
        return ""
    return " ".join(str(s).split())


def choose_statement_text_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """
    statement 테이블에서 텍스트 컬럼 선택.
    """
    if preferred:
        if preferred in df.columns:
            return preferred
        else:
            raise ValueError(f"--statement-text-column 으로 지정한 컬럼이 존재하지 않습니다: {preferred}")

    candidates = ["text", "paragraph", "content", "body", "statement"]
    for c in candidates:
        if c in df.columns:
            return c

    text_like_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("string")]
    if not text_like_cols:
        raise ValueError("문자열형 텍스트 컬럼을 찾을 수 없습니다. --statement-text-column 옵션으로 지정해주세요.")

    def col_len_sum(col):
        return df[col].dropna().astype(str).str.len().sum()

    best_col = max(text_like_cols, key=col_len_sum)
    return best_col


def aggregate_statement_text(df: pd.DataFrame, text_col: str) -> str:
    """
    statement 텍스트를 전행 결합 → 하나의 문서(doc).
    """
    series = df[text_col].dropna().astype(str).map(normalize_text)
    series = series[series.str.len() > 0]
    return " ".join(series.tolist())


def ensure_paragraphs_key_and_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    문단 테이블에서 book_name_index(또는 id 계열)와 paragraph를 보장하여 반환.
    반환 컬럼: ['book_name_index', 'paragraph']
    """
    if isinstance(df, dict):
        df = _coerce_excel_result_to_df(df)

    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    # 기본적으로 book_name_index 사용, 없으면 id/idx/index/번호 중 하나를 식별자로 사용
    key_col = lower_map.get("book_name_index")
    para_col = lower_map.get("paragraph")

    if key_col is None:
        for cand in ["id", "idx", "index", "번호"]:
            if cand in lower_map:
                key_col = lower_map[cand]
                break

    if para_col is None:
        for cand in ["paragraph", "text", "content", "body", "문단", "para"]:
            if cand in lower_map:
                para_col = lower_map[cand]
                break

    if key_col is None or para_col is None:
        raise ValueError(f"문단 테이블에서 식별자/paragraph 컬럼을 찾을 수 없습니다. 실제 컬럼: {cols}")

    out = df[[key_col, para_col]].rename(columns={key_col: "book_name_index", para_col: "paragraph"}).copy()
    out["book_name_index"] = out["book_name_index"].astype(str).map(normalize_text)
    out["paragraph"] = out["paragraph"].astype(str).map(normalize_text)
    out = out[out["paragraph"].str.len() > 0]
    return out.reset_index(drop=True)


# ---------------- TF-IDF (각 문단 vs 단일 statement 문서) ----------------
def compute_tfidf_sim(paragraphs: List[str], doc: str) -> np.ndarray:
    """
    TF-IDF로 각 paragraph와 단일 doc 사이의 코사인 유사도.
    """
    docs = paragraphs + [doc]
    vectorizer = TfidfVectorizer(lowercase=True)
    X = vectorizer.fit_transform(docs)
    X_paras = X[:-1]
    X_doc = X[-1]
    sims = cosine_similarity(X_paras, X_doc).ravel()
    return sims


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


def _cosine_sim_from_embeddings(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return A_norm @ b_norm


def compute_openai_embedding_sim(
        paragraphs: List[str],
        doc: str,
        model: str,
        max_embed_tokens: int = _MAX_EMBED_TOKENS,
        batch_size: int = _EMBED_BATCH_SIZE,
        truncate_long_paragraphs: bool = True,
) -> Optional[np.ndarray]:
    """
    OpenAI 임베딩으로 각 paragraph와 단일 statement(doc) 간 코사인 유사도.
    - statement(doc)를 토큰 기준 청크 → 임베딩 평균
    - 문단이 길면 토큰 기준 자르기(옵션)
    - 임베딩 호출은 배치 처리
    """
    if not _HAS_OPENAI:
        warnings.warn("openai 패키지가 없어 openai 유사도를 계산하지 않습니다. pip install openai 필요.")
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        warnings.warn("환경 변수 OPENAI_API_KEY 가 설정되지 않아 openai 유사도를 건너뜁니다.")
        return None

    client = OpenAI(api_key=api_key)
    enc = _get_encoder(model)

    # statement → 청크 임베딩 평균
    doc_chunks = _chunk_text_by_tokens(doc, enc, max_embed_tokens)
    if not doc_chunks:
        warnings.warn("statement 텍스트가 비어 있어 openai 유사도를 건너뜁니다.")
        return None
    B_list = _embed_texts_in_batches(client, model, doc_chunks, batch_size=batch_size)
    B = np.stack(B_list, axis=0)
    b = B.mean(axis=0)

    # paragraphs → 필요 시 자르기 → 배치 임베딩
    paras_proc = _maybe_truncate_by_tokens(paragraphs, enc, max_embed_tokens) if truncate_long_paragraphs else paragraphs
    A_list = _embed_texts_in_batches(client, model, paras_proc, batch_size=batch_size)
    A = np.stack(A_list, axis=0)

    sims = _cosine_sim_from_embeddings(A, b)
    return sims


def main():
    parser = argparse.ArgumentParser(description="Statement(단일) vs combined_book_paragraphs.xlsx 각 문단 유사도 계산(OpenAI, TF-IDF) → 엑셀 저장")
    parser.add_argument("--paragraphs-csv", type=str, default=DEFAULT_PARAGRAPHS_CSV,
                        help="문단 테이블 경로 (CSV/TSV/Excel, book_name_index/paragraph 포함 권장)")
    parser.add_argument("--paragraphs-sheet", type=str, default=None,
                        help="문단이 엑셀 파일일 경우 사용할 시트명(미지정 시 첫 시트)")
    parser.add_argument("--statement-csv", type=str, default=DEFAULT_STATEMENT_CSV,
                        help="Statement 테이블 경로 (CSV/TSV/Excel) - 단일 문서로 합쳐 사용")
    parser.add_argument("--statement-sheet", type=str, default=None,
                        help="Statement가 엑셀 파일일 경우 사용할 시트명(미지정 시 첫 시트)")
    parser.add_argument("--statement-text-column", type=str, default=None,
                        help="Statement 테이블에서 텍스트가 들어있는 컬럼명(지정 시 자동 탐색 대신 사용)")
    parser.add_argument("--openai-model", type=str, default=DEFAULT_OPENAI_MODEL,
                        help="OpenAI 임베딩 모델명 (예: text-embedding-3-small, text-embedding-3-large)")
    parser.add_argument("--output-xlsx", type=str, default=DEFAULT_OUTPUT_XLSX,
                        help="결과 저장 엑셀 경로(.xlsx)")
    args = parser.parse_args()

    # 1) 문단 읽기
    para_df_raw = read_table_auto(args.paragraphs_csv, sheet_name=args.paragraphs_sheet)
    para_df = ensure_paragraphs_key_and_text(para_df_raw)  # ['book_name_index','paragraph']
    paragraphs = para_df["paragraph"].tolist()
    keys = para_df["book_name_index"].tolist()

    # 2) statement 읽기 → 단일 문자열로 합치기
    stmt_df = read_table_auto(args.statement_csv, sheet_name=args.statement_sheet)
    stmt_text_col = choose_statement_text_column(stmt_df, preferred=args.statement_text_column)
    statement_text = aggregate_statement_text(stmt_df, stmt_text_col)

    # 3) TF-IDF 유사도 (각 문단 vs 단일 statement)
    tfidf_sims = compute_tfidf_sim(paragraphs, statement_text)

    # 4) OpenAI 유사도 (옵션)
    openai_sims = compute_openai_embedding_sim(paragraphs, statement_text, model=args.openai_model)

    # 5) 결과 DataFrame 구성: book_name_index, openai, tfidf
    out = pd.DataFrame({
        "book_name_index": keys,
        "tfidf": tfidf_sims
    })
    if openai_sims is not None:
        out["openai"] = openai_sims
    else:
        out["openai"] = np.nan

    # 열 순서 및 반올림
    out = out[["book_name_index", "openai", "tfidf"]]
    out["openai"] = out["openai"].astype(float).round(6)
    out["tfidf"] = out["tfidf"].astype(float).round(6)

    # 6) 엑셀 저장
    try:
        out.to_excel(args.output_xlsx, index=False)
        print(f"[DONE] Saved Excel to: {os.path.abspath(args.output_xlsx)}")
        print(f"[INFO] rows: {len(out)}, columns: {list(out.columns)}")
        print(f"(statement 텍스트 컬럼: {stmt_text_col})")
        if openai_sims is None:
            print("참고: OPENAI_API_KEY 미설정 또는 openai 패키지 미설치/길이 문제로 openai 유사도는 NaN 입니다.")
    except Exception as e:
        print(f"[ERROR] Failed to save Excel: {e}")


if __name__ == "__main__":
    main()