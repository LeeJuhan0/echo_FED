"""
FOMC GraphRAG – FastAPI service
================================
Exposes the sentiment-scoring pipeline as a REST API so downstream
applications can query it without running the CLI pipeline.

Run locally::

    uvicorn api.app:app --reload

Endpoints
---------
GET  /health              Liveness probe.
POST /inference           Run the 5-step sentiment-scoring pipeline.
GET  /inference/{gid}     Convenience alias that re-runs the pipeline with
                          default parameters for the given GID.
"""

from __future__ import annotations

import logging
import os
import sys

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when the module is executed directly.
# ---------------------------------------------------------------------------
_api_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_api_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.config import Config  # noqa: E402
from src.utils.inference import get_response  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FOMC GraphRAG Sentiment API",
    description=(
        "REST wrapper around the GraphRAG-based FOMC sentiment-scoring pipeline. "
        "Returns a sentiment score in [-1.0, 1.0] for each of five reference "
        "sources: Statement, Papers, FSR, SLOOS, and BeigeBook."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Shared Neo4j connection (lazy-initialised on first request)
# ---------------------------------------------------------------------------

_n4j = None


def _get_db():
    global _n4j
    if _n4j is None:
        try:
            from src.external.camel.storages import Neo4jGraph

            _n4j = Neo4jGraph(
                url=Config.NEO4J_URL,
                username=Config.NEO4J_USERNAME,
                password=Config.NEO4J_PASSWORD,
            )
            logger.info("Connected to Neo4j at %s", Config.NEO4J_URL)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Cannot connect to Neo4j: {exc}",
            ) from exc
    return _n4j


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

_SCORE_COLUMNS = ["Statement", "Papers", "FSR", "SLOOS", "BeigeBook"]


class InferenceRequest(BaseModel):
    """Input parameters for the sentiment-scoring pipeline."""

    gid: str = Field(
        ...,
        description="Graph ID of the target FOMC statement, e.g. 'FOMC202301'.",
        examples=["FOMC202301"],
    )
    question: str = Field(
        ...,
        description="The FOMC statement text (or a distilled question about it).",
    )
    gid_index: int = Field(
        default=0,
        ge=0,
        description="Row index into the similarity-score matrix (theory mode).",
    )
    sim_score_median: float = Field(
        default=0.0,
        ge=0.0,
        description="Similarity-score threshold for theory-paper retrieval.",
    )
    year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Numeric year of the statement (used for output file naming).",
        examples=[2023],
    )
    meeting_no: int = Field(
        default=1,
        ge=1,
        description="Meeting sequence number within the year (used for output file naming).",
    )

    @field_validator("gid")
    @classmethod
    def gid_must_start_with_fomc(cls, v: str) -> str:
        if not v.startswith("FOMC"):
            raise ValueError("gid must start with 'FOMC', e.g. 'FOMC202301'.")
        return v


class ScoreResult(BaseModel):
    """Sentiment scores for a single FOMC statement."""

    Statement: float | None = Field(None, description="Score from the statement alone.")
    Papers: float | None = Field(None, description="Score refined with theory papers.")
    FSR: float | None = Field(None, description="Score refined with FSR.")
    SLOOS: float | None = Field(None, description="Score refined with SLOOS.")
    BeigeBook: float | None = Field(None, description="Score refined with Beige Book.")


class InferenceResponse(BaseModel):
    gid: str
    scores: ScoreResult


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Liveness probe", tags=["ops"])
def health_check():
    """Return ``{"status": "ok"}`` when the service is up."""
    return {"status": "ok"}


@app.post(
    "/inference",
    response_model=InferenceResponse,
    summary="Run 5-step sentiment scoring",
    tags=["inference"],
)
def run_inference(request: InferenceRequest) -> InferenceResponse:
    """Execute the full GraphRAG sentiment-scoring pipeline.

    The pipeline calls the LLM up to **five times** (one per source document
    type), progressively refining the sentiment score.  Each score is a float
    in ``[-1.0, 1.0]``.  A ``null`` value means the LLM response could not be
    parsed for that step.
    """
    n4j = _get_db()

    try:
        scores = get_response(
            n4j=n4j,
            gid=request.gid,
            query=request.question,
            i=request.gid_index,
            sim_score_median=request.sim_score_median,
            year=request.year,
            meeting_no=request.meeting_no,
        )
    except Exception as exc:
        logger.exception("Inference failed for gid=%s", request.gid)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    score_dict = dict(zip(_SCORE_COLUMNS, scores))
    return InferenceResponse(gid=request.gid, scores=ScoreResult(**score_dict))


# ---------------------------------------------------------------------------
# Entry-point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
