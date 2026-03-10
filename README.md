# FOMC GraphRAG – 금융 텍스트 감성 분석

GraphRAG 기반의 연방준비제도(Fed) 통화정책 텍스트 감성 분석 시스템입니다.  
FOMC 성명서, 학술 논문, FSR, SLOOS, BeigeBook을 지식 그래프(Neo4j)로 연결하고,  
LLM을 통해 5단계 감성 점수를 산출합니다.

📄 관련 논문: [SSRN 6163827](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6163827)

---

## 프로젝트 구조

```
echo_FED/
├── api/                        # FastAPI REST 서비스
│   └── app.py                  # /health, /inference 엔드포인트
├── src/
│   ├── analysis/               # 평가 · 통계 (ARIMA, T-test, 정규성 검사)
│   │   ├── evaluator/          # CPI·ANFCI·NFCI 상관 분석
│   │   ├── Statistic/          # 정규성 검정, Q-Q 플롯
│   │   └── pre_processing/     # NFCI 계산, 유사도 백분위
│   ├── external/camel/         # 멀티 에이전트 오케스트레이션 (CAMEL)
│   ├── graph/
│   │   ├── builder.py          # Neo4j 지식 그래프 구축
│   │   └── link.py             # 그래프 간 문서 연결
│   ├── ingestion/
│   │   ├── agentic_chunker.py  # LLM 기반 청킹
│   │   ├── chunk_processor.py  # 명제 단위 분해
│   │   ├── dataloader.py       # 파일 로더
│   │   └── generate_para.py    # 단락 생성
│   ├── llm/
│   │   └── summarizer.py       # 문서 요약기
│   └── utils/
│       ├── config.py           # 환경변수 설정 (Config 클래스)
│       ├── llm_client.py       # LLM 호출 + 재시도 로직
│       ├── graph_utils.py      # 그래프 유틸리티 (임베딩, 속성 추가)
│       ├── retrieval.py        # Neo4j 컨텍스트 조회
│       ├── inference.py        # 5단계 감성 점수 파이프라인
│       └── common.py           # 하위 호환 재내보내기 (legacy imports)
├── scripts/
│   ├── run_pipeline.py         # 통합 CLI 파이프라인 진입점
│   ├── experiment/             # 실험 스크립트
│   ├── similarity/             # 유사도 계산
│   └── optimization/           # Sobol 시퀀스 최적화
├── data/
│   ├── raw/                    # 원시 데이터 (PDF, 논문)
│   ├── processed/              # 전처리 결과
│   ├── external/               # 참조 데이터 (CPI, ANFCI, UNRATE)
│   └── results/                # 출력 결과
├── .env.example                # 환경변수 템플릿
├── requirements.txt            # Python 의존성 (버전 고정)
└── README.md
```

---

## 빠른 시작

### 1. 사전 요구사항

| 항목 | 버전 |
|------|------|
| Python | 3.11 이상 |
| Neo4j | 5.x (APOC + GDS 플러그인 필요) |
| OpenAI API 키 | — |

### 2. 설치

```bash
# 저장소 클론
git clone https://github.com/LeeJuhan0/echo_FED.git
cd echo_FED

# 가상환경 생성
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY, NEO4J_PASSWORD 등 실제 값 입력
```

---

## CLI 파이프라인 사용법

```
python scripts/run_pipeline.py --step <STEP> [OPTIONS]
```

| Step | 설명 |
|------|------|
| `extract_statement` | FOMC PDF → CSV 변환 |
| `preprocess` | Theory 문서 → 단락 생성 |
| `construct_theory` | Theory 지식 그래프 구축 |
| `construct_statement` | FOMC 성명서 지식 그래프 구축 |
| `link` | 문서 간 그래프 연결 |
| `inference` | 감성 점수 추론 (5단계) |

### 예시 명령어

```bash
# 1. PDF에서 성명서 추출
python scripts/run_pipeline.py --step extract_statement

# 2. Theory 그래프 구축 (세밀한 청킹 적용)
python scripts/run_pipeline.py --step construct_theory --grained_chunk

# 3. FOMC 성명서 그래프 구축
python scripts/run_pipeline.py --step construct_statement

# 4. 그래프 연결
python scripts/run_pipeline.py --step link --fomc_gid FOMC202301

# 5. 추론 (Theory 파라미터 적용, 2021-01 ~ 2023-12)
python scripts/run_pipeline.py --step inference \
    --use_theory \
    --start_date 202101 \
    --end_date 202312

# 6. 추론 (기본 모드)
python scripts/run_pipeline.py --step inference \
    --start_date 202101 \
    --end_date 202312
```

---

## FastAPI 서비스

CLI 대신 REST API로 추론을 실행할 수 있습니다.

```bash
uvicorn api.app:app --reload
```

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/health` | GET | 서비스 상태 확인 |
| `/inference` | POST | 5단계 감성 점수 추론 |
| `/docs` | GET | Swagger UI |

### 요청 예시

```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "gid": "FOMC202301",
    "question": "<FOMC statement text here>",
    "year": 2023,
    "meeting_no": 1
  }'
```

### 응답 예시

```json
{
  "gid": "FOMC202301",
  "scores": {
    "Statement":  0.45,
    "Papers":     0.42,
    "FSR":        0.38,
    "SLOOS":      0.30,
    "BeigeBook":  0.35
  }
}
```

---

## 감성 점수 로직

```
FOMC 성명서 텍스트
   │
   ▼
[Step 1] Statement 단독 → 기준 점수  (-1.0 ~ +1.0)
   │
   ▼
[Step 2] + 학술 논문 컨텍스트 → 점수 수정
   │
   ▼
[Step 3] + FSR 컨텍스트 → 점수 수정
   │
   ▼
[Step 4] + SLOOS 컨텍스트 → 점수 수정
   │
   ▼
[Step 5] + BeigeBook 컨텍스트 → 최종 점수
```

각 단계에서 LLM이 이전 단계의 응답을 참조하여 점수를 재조정합니다.  
점수는 정규식으로 파싱되며 `[-1.0, 1.0]` 범위로 클리핑됩니다.  
파싱 실패 시 `None`을 반환하고 경고 메시지를 출력합니다.

---

## 이론적 배경

<img width="813" alt="architecture" src="https://github.com/user-attachments/assets/94ea4c28-b215-4458-a3c8-8b5918483c43" />

[![논문 보기](https://github.com/user-attachments/assets/cae6c5f2-c2f6-4938-a4a2-02773443952a)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6163827)
