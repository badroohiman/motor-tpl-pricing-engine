from __future__ import annotations

"""
FastAPI application exposing a /quote endpoint backed by QuoteService.

Design:
- Pydantic input schema for strict policy validation (types + ranges).
- Uses src.pricing.quote_service.QuoteService under the hood.
- Structured logging with request_id, model_version, config_version, warnings_count.
"""

import json
import logging
from uuid import uuid4
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.pricing.quote_service import QuoteService


logger = logging.getLogger("pricing_api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class PolicyInput(BaseModel):
    Area: str = Field(..., min_length=1, max_length=10)
    VehPower: float = Field(..., ge=1, le=400)
    VehAge: float = Field(..., ge=0, le=50)
    DrivAge: float = Field(..., ge=16, le=100)
    BonusMalus: float = Field(..., ge=0, le=400)
    VehBrand: str = Field(..., min_length=1, max_length=50)
    VehGas: str = Field(..., min_length=1, max_length=20)
    Density: float = Field(..., ge=0)
    Region: str = Field(..., min_length=1, max_length=10)
    Exposure: float = Field(..., gt=0, le=2)


def _default_service() -> QuoteService:
    """
    Build a default QuoteService using standard artifact/config paths.
    Assumes the working directory is the repo root when running uvicorn.
    """
    root = Path(__file__).resolve().parents[2]
    freq_model = root / "artifacts" / "models" / "frequency" / "freq_glm_nb.joblib"
    sev_model = root / "artifacts" / "models" / "severity" / "sev_glm_gamma.joblib"
    sev_cap = root / "artifacts" / "models" / "severity" / "sev_cap.json"
    pricing_cfg = root / "configs" / "pricing" / "pricing_config.yaml"

    return QuoteService(
        freq_model_path=freq_model,
        sev_model_path=sev_model,
        sev_cap_path=sev_cap if sev_cap.exists() else None,
        pricing_config_path=pricing_cfg,
    )


app = FastAPI(title="Motor TPL Pricing API", version="1.0.0")

# Allow portfolio/demo frontends from other origins to call /quote
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

service = _default_service()

# Optional: serve the insurance-style demo at /demo/ (same origin as API)
_api_root = Path(__file__).resolve().parents[2]
_demo_dir = _api_root / "web-demo"
if _demo_dir.is_dir():
    app.mount("/demo", StaticFiles(directory=str(_demo_dir), html=True), name="demo")


@app.post("/quote")
async def quote(policy: PolicyInput) -> Dict[str, Any]:
    """
    Compute a quote (pure + gross premium) for a single policy.
    """
    request_id = str(uuid4())

    try:
        quote_res = service.quote(policy.dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Structured log
    log_payload: Dict[str, Any] = {
        "event": "quote",
        "request_id": request_id,
        "decision": quote_res.decision,
        "model_version": quote_res.model_version,
        "config_version": quote_res.config_version,
        "warnings_count": len(quote_res.warnings),
    }
    logger.info(json.dumps(log_payload, ensure_ascii=False))

    from dataclasses import asdict

    return {
        "request_id": request_id,
        "quote": asdict(quote_res),
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

