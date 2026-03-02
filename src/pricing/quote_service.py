from __future__ import annotations

"""
Quote orchestrator for Motor TPL Pricing Engine.

Combines:
- Pure Premium layer (frequency x severity) -> λ, μ, expected_loss
- Gross pricing layer -> gross premium + business breakdown

Design:
- Single entrypoint for API / CLI (`QuoteService.quote`)
- Training-serving parity delegated to `pure_premium` + `pricing_engine`
- Warnings from both layers merged into a single list (no duplication)
- Manifest-based auditability for models + config + git commit
"""

import json
import math
import subprocess
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .pure_premium import PurePremiumEngine
from .pricing_engine import PricingEngine, GrossQuote


@dataclass(frozen=True)
class FinalQuote:
    """
    API-ready quote payload.

    - policy: raw input policy as received (after JSON parsing)
    - pure: pure premium layer outputs (λ, rate, severity mean, expected_loss)
    - gross: gross premium layer outputs (may be None if pure premium failed)
    - warnings: merged warnings from pure + gross layers
    - model_version: hashes + paths + git commit for frequency/severity models
    - config_version: pricing config version (also present inside gross.breakdown)
    """

    policy: Dict[str, Any]
    pure: Dict[str, Any]
    gross: Optional[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    explanation: Dict[str, Any]
    decision: str  # "BIND" | "REFER"
    decision_reasons: List[str]
    model_version: Dict[str, Any]
    config_version: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _repo_root() -> Path:
    # src/pricing/quote_service.py -> repo root is 2 levels up
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _build_manifest(
    *,
    freq_model_path: Path,
    sev_model_path: Path,
    pricing_config_path: Path,
    pricing_config_version: str,
) -> Dict[str, Any]:
    """
    Build and persist a simple manifest for auditability:
    - model artifacts (paths + sha256)
    - pricing config (path + sha256 + version)
    - git commit + timestamp
    """
    root = _repo_root()
    manifest_dir = root / "artifacts"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(root))
        except Exception:
            return str(p)

    manifest: Dict[str, Any] = {
        "created_at_utc": _utc_now_iso(),
        "freq_model_path": _rel(freq_model_path),
        "sev_model_path": _rel(sev_model_path),
        "pricing_config_path": _rel(pricing_config_path),
        "pricing_config_version": pricing_config_version,
        "git_commit": _git_commit_hash(),
    }

    try:
        manifest["freq_model_sha256"] = _sha256_file(freq_model_path)
    except Exception:
        manifest["freq_model_sha256"] = None

    try:
        manifest["sev_model_sha256"] = _sha256_file(sev_model_path)
    except Exception:
        manifest["sev_model_sha256"] = None

    try:
        manifest["pricing_config_sha256"] = _sha256_file(pricing_config_path)
    except Exception:
        manifest["pricing_config_sha256"] = None

    try:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        # Manifest write failure must not block quoting
        pass

    return manifest


class QuoteService:
    """
    Orchestrates pure premium and gross pricing into a single quote.
    """

    def __init__(
        self,
        *,
        freq_model_path: Path,
        sev_model_path: Path,
        sev_cap_path: Optional[Path],
        pricing_config_path: Path,
        sev_guardrail: Optional[float] = None,
    ):
        self.pricing_engine = PricingEngine(config_path=pricing_config_path)

        # Derive severity guardrail cap from config.guardrails if not explicitly provided
        cfg_guardrails = self.pricing_engine.cfg.get("guardrails", {}) or {}
        sev_guardrail_cfg = cfg_guardrails.get("severity_guardrail_cap")
        effective_sev_guardrail: Optional[float] = sev_guardrail if sev_guardrail is not None else sev_guardrail_cfg

        self.freq_model_path = freq_model_path
        self.sev_model_path = sev_model_path
        self.pricing_config_path = pricing_config_path

        self.pure_engine = PurePremiumEngine(
            freq_model_path=freq_model_path,
            sev_model_path=sev_model_path,
            sev_cap_path=sev_cap_path if (sev_cap_path and sev_cap_path.exists()) else None,
            guardrail_pred_sev_cap=effective_sev_guardrail,
        )

        # Build manifest once at service construction time
        cfg_version = str(self.pricing_engine.cfg.get("version", "dev"))
        self.manifest: Dict[str, Any] = _build_manifest(
            freq_model_path=freq_model_path,
            sev_model_path=sev_model_path,
            pricing_config_path=pricing_config_path,
            pricing_config_version=cfg_version,
        )

    def quote(self, policy: Dict[str, Any]) -> FinalQuote:
        """
        Full quote:
        - Compute λ, μ, expected_loss via pure premium engine
        - If successful, compute gross premium via pricing engine
        - Merge warnings from both layers into a single list (no duplication)
        """
        pure_res = self.pure_engine.quote_pure_premium(policy)

        merged_warnings: List[Dict[str, Any]] = list(pure_res.warnings)

        gross_payload: Optional[Dict[str, Any]] = None

        # Only call gross pricing if expected_loss is finite
        if math.isfinite(float(pure_res.expected_loss)):
            gross: GrossQuote = self.pricing_engine.quote_gross(pure_premium=pure_res.expected_loss)

            # IMPORTANT: do not duplicate warnings:
            # - We'll keep a single top-level warnings list.
            # - Remove warnings from gross payload for a clean API contract.
            gross_payload = asdict(gross)
            gross_payload.pop("warnings", None)

            merged_warnings.extend(gross.warnings)

        pure_payload: Dict[str, Any] = {
            "lambda_freq": pure_res.lambda_freq,
            "rate_annual": pure_res.rate_annual,
            "sev_mean": pure_res.sev_mean,
            "expected_loss": pure_res.expected_loss,
        }

        # Decision + reasons: REFER if any ERROR-level or explicit REFER_TO_UNDERWRITER code
        decision = "BIND"
        decision_reasons: List[str] = []
        for w in merged_warnings:
            code = w.get("code")
            level = w.get("level")
            if code:
                decision_reasons.append(code)
            if level == "ERROR" or code == "REFER_TO_UNDERWRITER":
                decision = "REFER"

        cfg_version = str(self.pricing_engine.cfg.get("version", "dev"))
        model_version: Dict[str, Any] = {
            "git_commit": self.manifest.get("git_commit", "unknown"),
            "freq_model_path": self.manifest.get("freq_model_path"),
            "freq_model_sha256": self.manifest.get("freq_model_sha256"),
            "sev_model_path": self.manifest.get("sev_model_path"),
            "sev_model_sha256": self.manifest.get("sev_model_sha256"),
            "pricing_config_path": self.manifest.get("pricing_config_path"),
            "pricing_config_sha256": self.manifest.get("pricing_config_sha256"),
        }

        return FinalQuote(
            policy=policy,
            pure=pure_payload,
            gross=gross_payload,
            warnings=merged_warnings,
            explanation=pure_res.explanation,
            decision=decision,
            decision_reasons=decision_reasons,
            model_version=model_version,
            config_version=cfg_version,
        )


# -----------------------------
# CLI: convenience entrypoint
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="End-to-end quote: policy -> pure + gross premium")
    parser.add_argument(
        "--freq-model",
        type=str,
        default=r"artifacts\models\frequency\freq_glm_nb.joblib",
        help="Path to trained frequency model artifact",
    )
    parser.add_argument(
        "--sev-model",
        type=str,
        default=r"artifacts\models\severity\sev_glm_gamma.joblib",
        help="Path to trained severity model artifact",
    )
    parser.add_argument(
        "--sev-cap",
        type=str,
        default=r"artifacts\models\severity\sev_cap.json",
        help="Path to severity cap json (documentation / tail policy)",
    )
    parser.add_argument(
        "--pricing-config",
        type=str,
        default=r"configs\pricing\pricing_config.yaml",
        help="Path to pricing_config.yaml (gross pricing rules)",
    )
    parser.add_argument(
        "--policy-json",
        type=str,
        required=True,
        help="Path to a JSON file containing a policy input",
    )
    parser.add_argument(
        "--sev-guardrail",
        type=float,
        default=None,
        help="Optional cap for predicted severity at serving time",
    )
    args = parser.parse_args()

    service = QuoteService(
        freq_model_path=Path(args.freq_model),
        sev_model_path=Path(args.sev_model),
        sev_cap_path=Path(args.sev_cap) if args.sev_cap else None,
        pricing_config_path=Path(args.pricing_config),
        sev_guardrail=args.sev_guardrail,
    )

    policy = json.loads(Path(args.policy_json).read_text(encoding="utf-8-sig"))
    quote = service.quote(policy)
    print(json.dumps(asdict(quote), indent=2, ensure_ascii=False))