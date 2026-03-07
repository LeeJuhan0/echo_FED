echo_FED/

├── src/                  # 핵심 로직 모듈 (Core Logic)

│   ├── analysis/         # 평가, 통계적 예측 및 평가(arima, ttest, 정규성 검사), 평가 데이터 전처리 (Evaluation, Statistic, Pp)

│   ├── external/         # 외부 라이브러리 (camel) 

│   ├── graph/            # 그래프 구축 및 연결 (Builder, Linker)

│   ├── ingestion/        # 데이터 전처리 및 로딩 (agentic-chunker, chunk-processor, dataloader, generate_para)

│   ├── llm/              # 문서 요약기 Summarizer

│   └── utils/            # 공통 유틸리티 및 설정 (Config, Common: 추론 엔진 (Inference, Retriever) 포함(분리 예정))

├── scripts/              # 실행 스크립트 (CLI Entry Points)

│   ├── experiment/       # 실험 과정

│   ├── optimization/     # Sobol Sequence 기반 최적화 스크립트

│   ├── similarity/       # 문단 단위 유사도 검

│   └── run_pipeline.py   # 통합 실행 파이프라인

├── data/                 # 데이터셋 (Processed Data)

│   ├── external/         # 평가 참조 데이터 (cpi, anfci, unrate...)

│   ├── processed         # 전처리 후 결과데이터 (tokenlist, similarity_pct)

│   ├── raw               # raw 데이터 (Statemnet, Book, Paper, FSR, SLOOS, Beigebook)

│   └── results           # 성능 평가 plot, model선정, sentiment score output

├── .env                  # 환경 변수 관리 (API Keys, git-ignore 처리)

└── README.md

📚 Theoretical Background

이 프로젝트는 **GraphRAG 기반의 금융텍스트 감성 분석**에 관한 연구를 기반으로 구현되었습니다.


<img width="813" height="811" alt="image" src="https://github.com/user-attachments/assets/94ea4c28-b215-4458-a3c8-8b5918483c43" />
