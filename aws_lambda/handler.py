from __future__ import annotations

"""
AWS Lambda handler for the Motor TPL Pricing Engine.

Wraps `QuoteService.quote` so it can be invoked via API Gateway or direct
Lambda invocations.
"""

import base64
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from src.pricing.quote_service import QuoteService


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_SERVICE: QuoteService | None = None


def _init_service() -> QuoteService:
    """
    Build a QuoteService instance using environment variables for paths.

    Environment variables:
    - FREQ_MODEL_PATH: path to frequency model artifact
    - SEV_MODEL_PATH: path to severity model artifact
    - SEV_CAP_PATH: path to severity cap JSON (optional)
    - PRICING_CONFIG_PATH: path to pricing_config.yaml
    - SEV_GUARDRAIL_CAP: optional float cap for predicted severity at serving time
    """
    # NOTE: Use POSIX-style separators in defaults so paths work on Linux
    # (Lambda container) and Windows. Environment variables can always
    # override these defaults.
    freq_model = Path(
        os.getenv(
            "FREQ_MODEL_PATH",
            "artifacts/models/frequency/freq_glm_nb.joblib",
        )
    )
    sev_model = Path(
        os.getenv(
            "SEV_MODEL_PATH",
            "artifacts/models/severity/sev_glm_gamma.joblib",
        )
    )
    sev_cap_raw = os.getenv(
        "SEV_CAP_PATH",
        "artifacts/models/severity/sev_cap.json",
    )
    sev_cap_path = Path(sev_cap_raw) if sev_cap_raw else None

    pricing_config = Path(
        os.getenv(
            "PRICING_CONFIG_PATH",
            "configs/pricing/pricing_config.yaml",
        )
    )

    sev_guardrail_cap: float | None
    sev_guardrail_env = os.getenv("SEV_GUARDRAIL_CAP")
    if sev_guardrail_env:
        try:
            sev_guardrail_cap = float(sev_guardrail_env)
        except ValueError:
            logger.warning(
                "Invalid SEV_GUARDRAIL_CAP value %r; ignoring.", sev_guardrail_env
            )
            sev_guardrail_cap = None
    else:
        sev_guardrail_cap = None

    logger.info(
        "Initializing QuoteService with freq_model=%s, sev_model=%s, sev_cap=%s, pricing_config=%s, sev_guardrail_cap=%s",
        str(freq_model),
        str(sev_model),
        str(sev_cap_path) if sev_cap_path else None,
        str(pricing_config),
        sev_guardrail_cap,
    )

    return QuoteService(
        freq_model_path=freq_model,
        sev_model_path=sev_model,
        sev_cap_path=sev_cap_path,
        pricing_config_path=pricing_config,
        sev_guardrail=sev_guardrail_cap,
    )


def _get_service() -> QuoteService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = _init_service()
    return _SERVICE


def _parse_policy_from_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract policy JSON from a Lambda event.

    Supports:
    - API Gateway proxy events (body as JSON string, possibly base64)
    - Direct invocation with {'policy': {...}}
    """
    if "body" in event:
        body = event["body"]
        if body is None:
            raise ValueError("Request body is empty.")

        if event.get("isBase64Encoded"):
            if not isinstance(body, str):
                raise ValueError("Expected base64 body to be a string.")
            try:
                body_bytes = base64.b64decode(body)
                body_str = body_bytes.decode("utf-8")
            except Exception as exc:  # noqa: BLE001
                raise ValueError("Failed to decode base64 body.") from exc
        else:
            body_str = body if isinstance(body, str) else json.dumps(body)

        try:
            payload = json.loads(body_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc

        # Allow a top-level {"policy": {...}} or the policy dict itself.
        if isinstance(payload, dict) and "policy" in payload:
            policy = payload["policy"]
        else:
            policy = payload

        if not isinstance(policy, dict):
            raise ValueError("Policy payload must be a JSON object.")
        return policy

    # Fallback: direct invocation with event containing 'policy'
    if "policy" in event:
        policy = event["policy"]
        if not isinstance(policy, dict):
            raise ValueError("event['policy'] must be a JSON object.")
        return policy

    raise ValueError("No policy found in event (missing 'body' or 'policy').")


def _build_response(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": int(status_code),
        "headers": {
            "Content-Type": "application/json",
        },
        "body": json.dumps(payload, ensure_ascii=False),
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:  # noqa: ANN401
    """
    AWS Lambda entrypoint.

    Expects a policy JSON via API Gateway proxy event and returns FinalQuote.
    """
    try:
        policy = _parse_policy_from_event(event)
    except ValueError as exc:
        logger.warning("Bad request: %s", exc)
        return _build_response(
            400,
            {
                "error": "BadRequest",
                "message": str(exc),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while parsing request.")
        return _build_response(
            500,
            {
                "error": "InternalError",
                "message": "Failed to parse request.",
            },
        )

    service = _get_service()

    try:
        quote = service.quote(policy)
        quote_dict = asdict(quote)
        logger.info(
            "Quote generated: decision=%s reasons=%s",
            quote_dict.get("decision"),
            quote_dict.get("decision_reasons"),
        )
        return _build_response(200, quote_dict)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error while generating quote.")
        return _build_response(
            500,
            {
                "error": "InternalError",
                "message": "Failed to generate quote.",
            },
        )

