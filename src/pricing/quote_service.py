from __future__ import annotations

"""
Quote orchestrator for Motor TPL Pricing Engine.

Combines:
- Pure Premium layer (frequency x severity) -> λ, μ, expected_loss
- Gross pricing layer -> gross premium + business breakdown

Design:
- Single entrypoint for API / CLI (`QuoteService.quote`)
- Training-serving parity delegated to `pure_premium` + `pricing_engine`
- Warnings from both layers merged into a single list
"""

import json
from dataclasses import dataclass, asdict
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
    """

    policy: Dict[str, Any]
    pure: Dict[str, Any]
    gross: Optional[Dict[str, Any]]
    warnings: List[Dict[str, Any]]


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
        self.pure_engine = PurePremiumEngine(
            freq_model_path=freq_model_path,
            sev_model_path=sev_model_path,
            sev_cap_path=sev_cap_path,
            guardrail_pred_sev_cap=sev_guardrail,
        )
        self.pricing_engine = PricingEngine(config_path=pricing_config_path)

    def quote(self, policy: Dict[str, Any]) -> FinalQuote:
        """
        Full quote:
        - Compute λ, μ, expected_loss via pure premium engine
        - If successful, compute gross premium via pricing engine
        - Merge warnings from both layers
        """
        pure_res = self.pure_engine.quote_pure_premium(policy)

        # Start with pure layer warnings
        merged_warnings: List[Dict[str, Any]] = list(pure_res.warnings)

        gross_payload: Optional[Dict[str, Any]] = None

        # Only call gross pricing if expected_loss is a real number
        if not (pure_res.expected_loss != pure_res.expected_loss):  # NaN check
            gross: GrossQuote = self.pricing_engine.quote_gross(
                pure_premium=pure_res.expected_loss
            )
            gross_payload = asdict(gross)
            merged_warnings.extend(gross.warnings)

        pure_payload: Dict[str, Any] = {
            "lambda_freq": pure_res.lambda_freq,
            "rate_annual": pure_res.rate_annual,
            "sev_mean": pure_res.sev_mean,
            "expected_loss": pure_res.expected_loss,
        }

        return FinalQuote(
            policy=policy,
            pure=pure_payload,
            gross=gross_payload,
            warnings=merged_warnings,
        )


# -----------------------------
# CLI: convenience entrypoint
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-end quote: policy -> pure + gross premium"
    )
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
        sev_cap_path=Path(args.sev_cap),
        pricing_config_path=Path(args.pricing_config),
        sev_guardrail=args.sev_guardrail,
    )

    policy = json.loads(Path(args.policy_json).read_text(encoding="utf-8-sig"))
    quote = service.quote(policy)
    print(json.dumps(asdict(quote), indent=2, ensure_ascii=False))

