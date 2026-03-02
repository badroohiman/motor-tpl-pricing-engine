# src/pricing/pure_premium.py
"""
Pure Premium / Expected Loss engine.

Expected Loss = E[ClaimNb | X, Exposure] * E[ClaimAmount | Claim, X]

Design goals:
- Training-serving parity via patsy formula used at training time
- Dependency-light serving: pandas + numpy + patsy + joblib + statsmodels (for predict)
- Structured warnings for governance and underwriting referrals
- Ready to be wrapped by FastAPI (/quote)

Artifacts expected:
- artifacts/models/frequency/freq_glm_nb.joblib
- artifacts/models/severity/sev_glm_gamma.joblib
- artifacts/models/severity/sev_cap.json (for documentation)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import patsy
from patsy import bs  # for formula bs(DrivAge, df=5), bs(VehAge, df=5) at inference
import joblib


# -----------------------------
# Warning structure
# -----------------------------
@dataclass(frozen=True)
class WarningItem:
    code: str
    message: str
    level: str = "WARN"  # WARN / ERROR
    field: Optional[str] = None
    value: Optional[Any] = None


# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def _ensure_cols(d: Dict[str, Any], cols: List[str]) -> None:
    missing = [c for c in cols if c not in d]
    if missing:
        raise ValueError(f"Missing required policy fields: {missing}")


def _normalize_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal normalization consistent with staging:
    - trim strings
    - Area/Region uppercase, VehGas lowercase
    """
    out = dict(policy)

    for c in ("Area", "VehBrand", "VehGas", "Region"):
        if c in out and out[c] is not None:
            s = str(out[c]).strip()
            if c in ("Area", "Region"):
                s = s.upper()
            if c == "VehGas":
                s = s.lower()
            out[c] = s

    # Numeric coercions
    for c in ("VehPower", "VehAge", "DrivAge", "BonusMalus", "Density", "Exposure"):
        if c in out:
            out[c] = _safe_float(out[c])

    return out


def _range_warnings(p: Dict[str, Any]) -> List[WarningItem]:
    w: List[WarningItem] = []

    exp = p.get("Exposure")
    if exp is None:
        w.append(WarningItem(code="MISSING_EXPOSURE", message="Exposure is missing.", level="ERROR", field="Exposure"))
    else:
        if exp <= 0:
            w.append(WarningItem(code="EXPOSURE_LE_ZERO", message="Exposure must be > 0.", level="ERROR", field="Exposure", value=exp))
        if exp < 0.01:
            w.append(WarningItem(code="LOW_EXPOSURE", message="Very low exposure may lead to unstable rates.", field="Exposure", value=exp))
        if exp > 1.0:
            # In staging we capped; in serving we warn but do NOT auto-cap silently
            w.append(WarningItem(code="EXPOSURE_GT_1", message="Exposure > 1.0 is unusual for policy-year fraction.", field="Exposure", value=exp))

    da = p.get("DrivAge")
    if da is not None and da < 18:
        w.append(WarningItem(code="DRIVAGE_UNDER_18", message="Driver age < 18 is out of expected range.", field="DrivAge", value=da))

    va = p.get("VehAge")
    if va is not None and va < 0:
        w.append(WarningItem(code="VEHAGE_NEGATIVE", message="Vehicle age < 0 is invalid.", level="ERROR", field="VehAge", value=va))

    bm = p.get("BonusMalus")
    if bm is not None and (bm < 50 or bm > 350):
        w.append(WarningItem(code="BONUSMALUS_OUT_OF_RANGE", message="BonusMalus outside typical [50,350].", field="BonusMalus", value=bm))

    den = p.get("Density")
    if den is not None and den < 0:
        w.append(WarningItem(code="DENSITY_NEGATIVE", message="Density < 0 is invalid.", level="ERROR", field="Density", value=den))

    return w


def _errors_present(warnings: List[WarningItem]) -> bool:
    return any(w.level == "ERROR" for w in warnings)


# -----------------------------
# Model wrappers
# -----------------------------
def _ensure_formula_lhs_in_df(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    """Ensure the LHS variable of a patsy formula exists in df (dummy value) so dmatrices can evaluate."""
    if "~" not in formula:
        return df
    lhs_var = formula.split("~")[0].strip()
    if lhs_var in df.columns:
        return df
    df = df.copy()
    # Use a dummy value: 0 for counts (ClaimNb), 1.0 for amounts (ClaimAmount)
    df[lhs_var] = 1.0 if "Amount" in lhs_var or "amount" in lhs_var else 0
    return df


def _expand_df_for_patsy(df: pd.DataFrame, factor_levels: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Build a dataframe that includes all categorical levels so patsy.dmatrices
    produces the same columns as at training (avoids (1,6) vs (1,43) shape mismatch).
    First row is the actual policy; extra rows are copies with levels cycled so every level appears.
    """
    if not factor_levels:
        return df
    out = [df]
    for col, levels in factor_levels.items():
        if col not in df.columns or not levels:
            continue
        for level in levels:
            row = df.copy()
            row[col] = level
            out.append(row)
    return pd.concat(out, ignore_index=True)


def _append_spline_anchor_rows(df: pd.DataFrame, spline_anchor: Dict[str, Any]) -> pd.DataFrame:
    """
    Append rows with training (DrivAge, VehAge) values so patsy bs(., df=5) computes
    the same knot placement as at training (training-serving parity).
    """
    if not spline_anchor or "DrivAge" not in spline_anchor or "VehAge" not in spline_anchor:
        return df
    driv = np.asarray(spline_anchor["DrivAge"]).ravel()
    veh = np.asarray(spline_anchor["VehAge"]).ravel()
    n = min(len(driv), len(veh))
    if n == 0:
        return df
    policy = df.iloc[[0]].copy()
    anchor_rows = []
    for i in range(n):
        row = policy.copy()
        row["DrivAge"] = float(driv[i])
        row["VehAge"] = float(veh[i])
        anchor_rows.append(row)
    return pd.concat([df] + anchor_rows, ignore_index=True)


class PatsyGLMModel:
    """GLM wrapper: uses formula at predict time (design_info is not picklable)."""

    def __init__(self, artifact: Any):
        self.res = artifact.fitted_result
        self.formula = artifact.formula
        self.factor_levels: Dict[str, List[str]] = getattr(
            artifact, "factor_levels", None
        ) or {}
        self.spline_anchor: Dict[str, Any] = getattr(
            artifact, "spline_anchor", None
        ) or {}

    def predict(self, df: pd.DataFrame, *, offset: Optional[np.ndarray] = None) -> np.ndarray:
        df_eval = _ensure_formula_lhs_in_df(df, self.formula)
        if self.factor_levels:
            df_eval = _expand_df_for_patsy(df_eval, self.factor_levels)
        if self.spline_anchor:
            df_eval = _append_spline_anchor_rows(df_eval, self.spline_anchor)
        _, X = patsy.dmatrices(self.formula, df_eval, return_type="dataframe")
        # Use only the first row (the actual policy); rest were for consistent design columns / knots
        X = X.iloc[[0]]
        if offset is None:
            pred = self.res.predict(X)
        else:
            pred = self.res.predict(X, offset=offset)
        return np.asarray(pred).reshape(-1)
@dataclass(frozen=True)
class PurePremiumResult:
    lambda_freq: float          # expected claim count for the given exposure period
    rate_annual: float          # annualized frequency (lambda / exposure)
    sev_mean: float             # expected claim amount conditional on claim
    expected_loss: float        # pure premium for the given exposure period
    warnings: List[Dict[str, Any]]


class PurePremiumEngine:
    def __init__(
        self,
        *,
        freq_model_path: Path,
        sev_model_path: Path,
        sev_cap_path: Optional[Path] = None,
        guardrail_pred_sev_cap: Optional[float] = None,
    ):
        self.freq_art = joblib.load(freq_model_path)
        self.sev_art = joblib.load(sev_model_path)

        self.freq = PatsyGLMModel(self.freq_art)
        self.sev = PatsyGLMModel(self.sev_art)

        self.sev_cap_info: Optional[Dict[str, Any]] = None
        if sev_cap_path is not None and sev_cap_path.exists():
            self.sev_cap_info = json.loads(sev_cap_path.read_text(encoding="utf-8"))

        # Optional serving-time guardrail for predicted severity (pricing stability)
        self.guardrail_pred_sev_cap = guardrail_pred_sev_cap

    def quote_pure_premium(self, policy: Dict[str, Any]) -> PurePremiumResult:
        required = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region", "Exposure"]
        _ensure_cols(policy, required)

        p = _normalize_policy(policy)
        warnings = _range_warnings(p)

        if _errors_present(warnings):
            return PurePremiumResult(
                lambda_freq=float("nan"),
                rate_annual=float("nan"),
                sev_mean=float("nan"),
                expected_loss=float("nan"),
                warnings=[w.__dict__ for w in warnings],
            )

        # Build single-row dataframe (log1p_Density for training-serving parity with freq/sev models)
        df = pd.DataFrame([p])
        df["log1p_Density"] = np.log1p(np.maximum(float(p.get("Density", 0) or 0), 0))

        # Frequency: predict expected count for given exposure period using offset(log(Exposure))
        exp = float(df.loc[0, "Exposure"])
        offset = np.log(np.maximum(np.array([exp], dtype=float), 1e-12))
        pred_count = float(self.freq.predict(df, offset=offset)[0])
        rate_annual = pred_count / max(exp, 1e-12)

        # Severity: predict conditional mean claim amount
        pred_sev = float(self.sev.predict(df)[0])

        # Optional guardrail on predicted severity (NOT training cap)
        if self.guardrail_pred_sev_cap is not None and pred_sev > self.guardrail_pred_sev_cap:
            warnings.append(
                WarningItem(
                    code="SEV_PRED_GUARDRAIL_CAPPED",
                    message="Predicted severity exceeded guardrail cap; capped for pricing stability.",
                    field="sev_mean",
                    value=pred_sev,
                )
            )
            pred_sev = float(min(pred_sev, self.guardrail_pred_sev_cap))

        expected_loss = pred_count * pred_sev

        return PurePremiumResult(
            lambda_freq=pred_count,
            rate_annual=rate_annual,
            sev_mean=pred_sev,
            expected_loss=expected_loss,
            warnings=[w.__dict__ for w in warnings],
        )


# -----------------------------
# Convenience CLI for quick tests
# -----------------------------
if __name__ == "__main__":
    # Example:
    # python -m src.pricing.pure_premium --policy-json configs/sample_policy.json
    import argparse

    parser = argparse.ArgumentParser(description="Pure premium quote (freq x sev)")
    parser.add_argument("--freq-model", type=str, default=r"artifacts\models\frequency\freq_glm_nb.joblib")
    parser.add_argument("--sev-model", type=str, default=r"artifacts\models\severity\sev_glm_gamma.joblib")
    parser.add_argument("--sev-cap", type=str, default=r"artifacts\models\severity\sev_cap.json")
    parser.add_argument("--policy-json", type=str, required=True, help="Path to a JSON file containing a policy input")
    parser.add_argument("--sev-guardrail", type=float, default=None, help="Optional cap for predicted severity at serving time")
    args = parser.parse_args()

    engine = PurePremiumEngine(
        freq_model_path=Path(args.freq_model),
        sev_model_path=Path(args.sev_model),
        sev_cap_path=Path(args.sev_cap),
        guardrail_pred_sev_cap=args.sev_guardrail,
    )

    policy = json.loads(Path(args.policy_json).read_text(encoding="utf-8-sig"))
    res = engine.quote_pure_premium(policy)
    print(json.dumps(res.__dict__, indent=2, ensure_ascii=False))