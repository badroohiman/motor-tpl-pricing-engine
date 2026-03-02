# src/pricing/pricing_engine.py
"""
Gross pricing layer for Motor TPL Pricing Engine.

This module converts Pure Premium (Expected Loss) into a sellable Gross Premium
using business/config rules:

- Expense loading
- Profit/risk margin
- (Optional) tax
- Min/Max premium caps
- (Optional) tiering / bands
- Structured breakdown + pricing config version for auditability

Design principles:
- Config-driven (no hard-coded business numbers)
- Deterministic and auditable
- Safe defaults + warnings when inputs are unusual
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class WarningItem:
    code: str
    message: str
    level: str = "WARN"
    field: Optional[str] = None
    value: Optional[Any] = None


@dataclass(frozen=True)
class PricingBreakdown:
    pure_premium: float
    expense_loading: float
    margin_loading: float
    tax_loading: float
    gross_before_caps: float
    gross_after_caps: float
    min_premium_applied: bool
    max_premium_applied: bool


@dataclass(frozen=True)
class GrossQuote:
    gross_premium: float
    breakdown: Dict[str, Any]
    warnings: List[Dict[str, Any]]
    pricing_config_version: str
    created_at_utc: str


# -----------------------------
# Helpers
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _validate_rates(expense: float, margin: float, tax: float) -> List[WarningItem]:
    w: List[WarningItem] = []
    for name, val in [("expense_ratio", expense), ("margin_ratio", margin), ("tax_ratio", tax)]:
        if val < 0:
            w.append(WarningItem(code="NEGATIVE_RATE", message=f"{name} is negative.", field=name, value=val, level="ERROR"))
        if val > 1:
            w.append(WarningItem(code="RATE_GT_1", message=f"{name} > 1 is invalid.", field=name, value=val, level="ERROR"))
    if expense + margin >= 1:
        w.append(
            WarningItem(
                code="INVALID_COMBINED_LOADINGS",
                message="expense_ratio + margin_ratio must be < 1 when using division-based grossing.",
                field="expense_ratio+margin_ratio",
                value=float(expense + margin),
                level="ERROR",
            )
        )
    return w


def _parse_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Config contract (minimal):

    version: "2026-02-26_v1"
    method: "division" | "multiplicative"

    expense_ratio: 0.25
    margin_ratio: 0.10
    tax_ratio: 0.00

    min_premium: 100.0
    max_premium: 5000.0

    # optional:
    tiering:
      enabled: false
      bands:
        - { name: "LOW",  max_pure: 200,  multiplier: 0.95 }
        - { name: "MID",  max_pure: 600,  multiplier: 1.00 }
        - { name: "HIGH", max_pure: 2000, multiplier: 1.10 }
        - { name: "VHIGH", max_pure: 1.0e18, multiplier: 1.20 }
    """
    out = dict(cfg)

    out.setdefault("version", "dev")
    out.setdefault("method", "division")
    out.setdefault("expense_ratio", 0.25)
    out.setdefault("margin_ratio", 0.10)
    out.setdefault("tax_ratio", 0.00)
    out.setdefault("min_premium", 100.0)
    out.setdefault("max_premium", 5000.0)

    out.setdefault("tiering", {"enabled": False, "bands": []})
    return out


def load_pricing_config(path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _parse_config(cfg)


def _apply_tiering(pure: float, cfg: Dict[str, Any]) -> (float, Optional[str], List[WarningItem]):
    """
    Optional tiering: apply multiplier based on pure premium bands.
    """
    w: List[WarningItem] = []
    tier_cfg = cfg.get("tiering", {}) or {}
    if not tier_cfg.get("enabled", False):
        return pure, None, w

    bands = tier_cfg.get("bands", []) or []
    if not bands:
        w.append(WarningItem(code="TIERING_ENABLED_NO_BANDS", message="Tiering enabled but no bands configured."))
        return pure, None, w

    # Expect bands sorted by max_pure ascending
    chosen = None
    for b in bands:
        max_pure = _to_float(b.get("max_pure"), float("inf"))
        if pure <= max_pure:
            chosen = b
            break

    if chosen is None:
        return pure, None, w

    mult = _to_float(chosen.get("multiplier"), 1.0)
    name = str(chosen.get("name", "UNKNOWN"))
    if mult <= 0:
        w.append(WarningItem(code="INVALID_TIER_MULTIPLIER", message="Tier multiplier must be > 0.", field="multiplier", value=mult, level="ERROR"))
        return pure, name, w

    return pure * mult, name, w


# -----------------------------
# Main pricing engine
# -----------------------------
class PricingEngine:
    def __init__(self, *, config_path: Path):
        self.config_path = config_path
        self.cfg = load_pricing_config(config_path)

    def quote_gross(self, *, pure_premium: float) -> GrossQuote:
        warnings: List[WarningItem] = []

        pure = float(pure_premium)
        if pure < 0:
            warnings.append(WarningItem(code="PURE_PREMIUM_NEGATIVE", message="Pure premium is negative; setting to 0.", field="pure_premium", value=pure))
            pure = 0.0

        expense = float(self.cfg["expense_ratio"])
        margin = float(self.cfg["margin_ratio"])
        tax = float(self.cfg["tax_ratio"])
        min_p = float(self.cfg["min_premium"])
        max_p = float(self.cfg["max_premium"])
        method = str(self.cfg["method"])

        warnings.extend(_validate_rates(expense, margin, tax))
        if any(w.level == "ERROR" for w in warnings):
            return GrossQuote(
                gross_premium=float("nan"),
                breakdown={"error": "Invalid pricing config", "details": [asdict(w) for w in warnings]},
                warnings=[asdict(w) for w in warnings],
                pricing_config_version=str(self.cfg.get("version", "dev")),
                created_at_utc=utc_now_iso(),
            )

        # Optional tiering first (business choice: apply on pure premium)
        pure_tiered, tier_name, tier_w = _apply_tiering(pure, self.cfg)
        warnings.extend(tier_w)
        if tier_name:
            warnings.append(WarningItem(code="TIER_APPLIED", message=f"Tier '{tier_name}' applied on pure premium.", field="tier", value=tier_name))

        # Grossing-up
        if method == "division":
            # Actuarial-style: premium = pure / (1 - expense - margin)
            gross_before_tax = pure_tiered / max(1.0 - expense - margin, 1e-12)
        elif method == "multiplicative":
            # Simple approximation
            gross_before_tax = pure_tiered * (1.0 + expense + margin)
        else:
            warnings.append(WarningItem(code="UNKNOWN_METHOD", message="Unknown pricing method; using division.", field="method", value=method))
            gross_before_tax = pure_tiered / max(1.0 - expense - margin, 1e-12)

        # Tax applied on top (if any)
        gross_before_caps = gross_before_tax * (1.0 + tax)

        # Caps
        gross_after = gross_before_caps
        min_applied = False
        max_applied = False

        if gross_after < min_p:
            gross_after = min_p
            min_applied = True
            warnings.append(WarningItem(code="MIN_PREMIUM_APPLIED", message="Minimum premium applied.", field="min_premium", value=min_p))
        if gross_after > max_p:
            gross_after = max_p
            max_applied = True
            warnings.append(WarningItem(code="MAX_PREMIUM_APPLIED", message="Maximum premium applied.", field="max_premium", value=max_p))

        breakdown = PricingBreakdown(
            pure_premium=pure,
            expense_loading=expense,
            margin_loading=margin,
            tax_loading=tax,
            gross_before_caps=float(gross_before_caps),
            gross_after_caps=float(gross_after),
            min_premium_applied=min_applied,
            max_premium_applied=max_applied,
        )

        return GrossQuote(
            gross_premium=float(gross_after),
            breakdown=asdict(breakdown),
            warnings=[asdict(w) for w in warnings],
            pricing_config_version=str(self.cfg.get("version", "dev")),
            created_at_utc=utc_now_iso(),
        )


# -----------------------------
# CLI: quick test
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gross pricing layer (pure -> gross)")
    parser.add_argument("--config", type=str, default="configs/pricing/pricing_config.yaml", help="Path to pricing_config.yaml")
    parser.add_argument("--pure", type=float, required=True, help="Pure premium (expected loss)")
    args = parser.parse_args()

    engine = PricingEngine(config_path=Path(args.config))
    quote = engine.quote_gross(pure_premium=args.pure)
    print(json.dumps(asdict(quote), indent=2, ensure_ascii=False))