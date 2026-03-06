from __future__ import annotations

"""
FastAPI application exposing a /quote endpoint backed by QuoteService.

Design:
- Pydantic input schema for strict policy validation (types + ranges).
- Uses src.pricing.quote_service.QuoteService under the hood.
- Structured logging with request_id, model_version, config_version, warnings_count.
- Lambda-friendly lazy service initialization using environment variables.
"""

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.pricing.quote_service import QuoteService


logger = logging.getLogger("pricing_api")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


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


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_service() -> QuoteService:
    """
    Build QuoteService using environment variables when provided,
    otherwise fall back to repo-relative defaults.
    """
    root = _resolve_repo_root()

    freq_model = Path(
        os.getenv(
            "FREQ_MODEL_PATH",
            str(root / "artifacts" / "models" / "frequency" / "freq_glm_nb.joblib"),
        )
    )
    sev_model = Path(
        os.getenv(
            "SEV_MODEL_PATH",
            str(root / "artifacts" / "models" / "severity" / "sev_glm_gamma.joblib"),
        )
    )
    sev_cap_raw = os.getenv(
        "SEV_CAP_PATH",
        str(root / "artifacts" / "models" / "severity" / "sev_cap.json"),
    )
    pricing_cfg = Path(
        os.getenv(
            "PRICING_CONFIG_PATH",
            str(root / "configs" / "pricing" / "pricing_config.yaml"),
        )
    )

    sev_cap = Path(sev_cap_raw) if sev_cap_raw else None
    if sev_cap is not None and not sev_cap.exists():
        sev_cap = None

    return QuoteService(
        freq_model_path=freq_model,
        sev_model_path=sev_model,
        sev_cap_path=sev_cap,
        pricing_config_path=pricing_cfg,
    )


_SERVICE: Optional[QuoteService] = None


def get_service() -> QuoteService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = _build_service()
    return _SERVICE


APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")
CONFIG_VERSION = os.getenv("CONFIG_VERSION", "unknown")

app = FastAPI(title="Motor TPL Pricing API", version=APP_VERSION)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Optional: serve demo locally or from same origin deployment
_api_root = _resolve_repo_root()
_demo_dir = _api_root / "web-demo"
if _demo_dir.is_dir():
    app.mount("/demo", StaticFiles(directory=str(_demo_dir), html=True), name="demo")


@app.post("/quote")
async def quote(policy: PolicyInput) -> Dict[str, Any]:
    request_id = str(uuid4())

    try:
        service = get_service()
        quote_res = service.quote(policy.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unhandled error during quote request")
        raise HTTPException(status_code=500, detail="Internal server error")

    log_payload: Dict[str, Any] = {
        "event": "quote",
        "request_id": request_id,
        "decision": quote_res.decision,
        "model_version": getattr(quote_res, "model_version", MODEL_VERSION),
        "config_version": getattr(quote_res, "config_version", CONFIG_VERSION),
        "warnings_count": len(quote_res.warnings),
    }
    logger.info(json.dumps(log_payload, ensure_ascii=False))

    return {
        "request_id": request_id,
        "quote": asdict(quote_res),
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    try:
        service = get_service()

        return {
            "status": "ok",
            "app_version": APP_VERSION,
            "model_version": getattr(service, "model_version", MODEL_VERSION),
            "config_version": getattr(service, "config_version", CONFIG_VERSION),
        }
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(e),
                "app_version": APP_VERSION,
            },
        )