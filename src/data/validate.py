# src/data/validate.py
"""
Validation for freMTPL2freq and freMTPL2sev.

This module enforces:
1) Schema: required columns + target dtypes + nullability
2) Constraints: ranges and business rules (insurance-grade)
3) Diagnostics: counts, rates, and a structured report for governance

Keep dependency-light: pandas + stdlib (numpy optional but not required here).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.data.schemas import ColumnSpec, DatasetSchema, FREQ_SCHEMA, SEV_SCHEMA
from src.data.utils import ensure_dir, rate


# -----------------------------
# Report structures
# -----------------------------
@dataclass(frozen=True)
class Finding:
    level: str  # "ERROR" or "WARN"
    code: str
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    dataset: str
    ok: bool
    findings: List[Finding]
    summary: Dict[str, Any]


# -----------------------------
# Helpers
# -----------------------------
def _rate(count: int, denom: int) -> float:
    return rate(count, denom)


def _dtype_matches(series: pd.Series, target: str) -> bool:
    """
    Soft dtype check. We validate 'int64', 'float64', and 'string' in a practical way.
    """
    if target == "string":
        return pd.api.types.is_string_dtype(series.dtype)
    if target == "int64":
        return pd.api.types.is_integer_dtype(series.dtype)
    if target == "float64":
        return pd.api.types.is_float_dtype(series.dtype) or pd.api.types.is_integer_dtype(series.dtype)
    # fallback: exact match
    return str(series.dtype) == target


def _coerce_to_target(df: pd.DataFrame, schema: DatasetSchema) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Coerce columns to target dtypes where safe and capture coercion metrics.
    This is validation-friendly: it helps avoid false failures due to pandas reading quirks.
    """
    out = df.copy()
    metrics: Dict[str, Any] = {"coercions": {}}

    for col, spec in schema.columns.items():
        if col not in out.columns:
            continue

        before_dtype = str(out[col].dtype)
        before_na = int(out[col].isna().sum())

        try:
            if spec.dtype == "string":
                out[col] = out[col].astype("string")
            elif spec.dtype == "int64":
                # Use numeric coercion; reject non-integer-like later via checks
                out[col] = pd.to_numeric(out[col], errors="coerce")
                # If nullable ints are present, keep as Int64 first then validate nullability
                out[col] = out[col].round(0).astype("Int64")
                # If we later require non-nullable, we will fail if any NA exists.
                # Convert to int64 only when safe (no NA).
                if int(out[col].isna().sum()) == 0:
                    out[col] = out[col].astype("int64")
            elif spec.dtype == "float64":
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        except Exception as e:
            metrics["coercions"][col] = {"error": str(e), "dtype_before": before_dtype}
            continue

        after_dtype = str(out[col].dtype)
        after_na = int(out[col].isna().sum())

        metrics["coercions"][col] = {
            "dtype_before": before_dtype,
            "dtype_after": after_dtype,
            "na_before": before_na,
            "na_after": after_na,
            "new_nas": max(0, after_na - before_na),
        }

    return out, metrics


def _check_integer_like(series: pd.Series) -> int:
    """
    Return count of non-integer-like finite values.
    Works for numeric series; ignores NA.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if s.empty:
        return 0
    frac = (s % 1 != 0)
    return int(frac.sum())


# -----------------------------
# Schema validation
# -----------------------------
def validate_schema(
    df: pd.DataFrame,
    schema: DatasetSchema,
    *,
    coerce: bool = True,
) -> Tuple[pd.DataFrame, List[Finding], Dict[str, Any]]:
    """
    Validate required columns, dtypes, and nullability.
    Optionally coerce dtypes to reduce brittleness.
    Returns (possibly coerced) df, findings, and summary metrics.
    """
    findings: List[Finding] = []

    # Required columns
    missing = [c for c in schema.columns.keys() if c not in df.columns]
    if missing:
        findings.append(
            Finding(
                level="ERROR",
                code="MISSING_COLUMNS",
                message=f"Missing required columns: {missing}",
                metrics={"missing": missing},
            )
        )
        # If columns are missing, coercion and further checks may not be meaningful.
        return df, findings, {"rows": int(len(df)), "cols": int(df.shape[1])}

    # Coerce (optional)
    work = df
    coercion_metrics: Dict[str, Any] = {}
    if coerce:
        work, coercion_metrics = _coerce_to_target(df, schema)

    # Dtype + null checks per column
    for col, spec in schema.columns.items():
        s = work[col]

        # dtype check (soft)
        if not _dtype_matches(s, spec.dtype):
            findings.append(
                Finding(
                    level="ERROR",
                    code="DTYPE_MISMATCH",
                    message=f"Column '{col}' dtype '{s.dtype}' does not match target '{spec.dtype}'",
                    metrics={"column": col, "dtype": str(s.dtype), "target": spec.dtype},
                )
            )

        # nullability check
        nulls = int(s.isna().sum())
        if not spec.nullable and nulls > 0:
            findings.append(
                Finding(
                    level="ERROR",
                    code="NULLS_NOT_ALLOWED",
                    message=f"Column '{col}' has {nulls} nulls but is non-nullable",
                    metrics={"column": col, "nulls": nulls, "null_rate": _rate(nulls, len(work))},
                )
            )

        # integer-like check (if requested)
        if spec.integer_like:
            bad = _check_integer_like(s)
            if bad > 0:
                findings.append(
                    Finding(
                        level="ERROR",
                        code="NOT_INTEGER_LIKE",
                        message=f"Column '{col}' has {bad} non-integer-like values",
                        metrics={"column": col, "non_integer_like": bad},
                    )
                )

    summary = {
        "rows": int(len(work)),
        "cols": int(work.shape[1]),
        "columns": [str(c) for c in work.columns],
        "coercion": coercion_metrics,
    }
    return work, findings, summary


# -----------------------------
# Business/constraint validation
# -----------------------------
def validate_constraints_freq(
    df: pd.DataFrame,
    *,
    exposure_cap_policy: Optional[float] = 1.0,
    warn_low_exposure_below: float = 0.01,
) -> List[Finding]:
    """
    Insurance-grade constraints for freq table.
    Use WARN for informational policies; ERROR for invalid values.
    """
    findings: List[Finding] = []
    n = int(len(df))

    # Exposure must be > 0 for log-offset
    le0 = int((df["Exposure"] <= 0).sum(skipna=True))
    if le0 > 0:
        findings.append(
            Finding(
                level="ERROR",
                code="EXPOSURE_LE_ZERO",
                message=f"Exposure has {le0} rows <= 0 (invalid for log-offset modeling).",
                metrics={"count": le0, "rate": _rate(le0, n)},
            )
        )

    # Policy-year assumption: Exposure <= 1 (if policy is enabled)
    if exposure_cap_policy is not None:
        gt1 = int((df["Exposure"] > exposure_cap_policy).sum(skipna=True))
        if gt1 > 0:
            findings.append(
                Finding(
                    level="WARN",
                    code="EXPOSURE_GT_1",
                    message=f"Exposure has {gt1} rows > {exposure_cap_policy}. "
                            "If assuming policy-year exposure, cap/drop in staging with documentation.",
                    metrics={"count": gt1, "rate": _rate(gt1, n), "cap": exposure_cap_policy},
                )
            )

    # Low exposure warning (instability risk)
    low = int((df["Exposure"] < warn_low_exposure_below).sum(skipna=True))
    if low > 0:
        findings.append(
            Finding(
                level="WARN",
                code="LOW_EXPOSURE",
                message=f"Exposure has {low} rows < {warn_low_exposure_below}. "
                        "These can increase noise/instability in frequency calibration.",
                metrics={"count": low, "rate": _rate(low, n), "threshold": warn_low_exposure_below},
            )
        )

    # ClaimNb non-negative
    neg_claim = int((df["ClaimNb"] < 0).sum(skipna=True))
    if neg_claim > 0:
        findings.append(
            Finding(
                level="ERROR",
                code="CLAIMNB_NEGATIVE",
                message=f"ClaimNb has {neg_claim} negative rows.",
                metrics={"count": neg_claim, "rate": _rate(neg_claim, n)},
            )
        )

    # Domain checks (WARN by default; can be tightened)
    # DrivAge >= 18
    if "DrivAge" in df.columns:
        under18 = int((df["DrivAge"] < 18).sum(skipna=True))
        if under18 > 0:
            findings.append(
                Finding(
                    level="WARN",
                    code="DRIVAGE_UNDER_18",
                    message=f"DrivAge has {under18} rows < 18.",
                    metrics={"count": under18, "rate": _rate(under18, n)},
                )
            )

    # VehAge >= 0
    if "VehAge" in df.columns:
        neg_veh = int((df["VehAge"] < 0).sum(skipna=True))
        if neg_veh > 0:
            findings.append(
                Finding(
                    level="WARN",
                    code="VEHAGE_NEGATIVE",
                    message=f"VehAge has {neg_veh} rows < 0.",
                    metrics={"count": neg_veh, "rate": _rate(neg_veh, n)},
                )
            )

    # BonusMalus typical range [50, 350]
    if "BonusMalus" in df.columns:
        bm_lo = int((df["BonusMalus"] < 50).sum(skipna=True))
        bm_hi = int((df["BonusMalus"] > 350).sum(skipna=True))
        if bm_lo + bm_hi > 0:
            findings.append(
                Finding(
                    level="WARN",
                    code="BONUSMALUS_OUT_OF_RANGE",
                    message=f"BonusMalus out of typical range [50,350]: below={bm_lo}, above={bm_hi}.",
                    metrics={"below_50": bm_lo, "above_350": bm_hi, "rate": _rate(bm_lo + bm_hi, n)},
                )
            )

    # Key uniqueness (freq should be unique per IDpol in this dataset)
    dup = int(df.duplicated(subset=["IDpol"]).sum())
    if dup > 0:
        findings.append(
            Finding(
                level="WARN",
                code="DUPLICATE_IDPOL_FREQ",
                message=f"Found {dup} duplicate IDpol rows in freq (expected unique for clean joins).",
                metrics={"count": dup, "rate": _rate(dup, n)},
            )
        )

    return findings


def validate_constraints_sev(df: pd.DataFrame) -> List[Finding]:
    """
    Insurance-grade constraints for sev table.
    """
    findings: List[Finding] = []
    n = int(len(df))

    # ClaimAmount must be > 0 for severity modeling
    le0 = int((df["ClaimAmount"] <= 0).sum(skipna=True))
    if le0 > 0:
        findings.append(
            Finding(
                level="ERROR",
                code="CLAIMAMOUNT_NON_POSITIVE",
                message=f"ClaimAmount has {le0} rows <= 0 (invalid for severity modeling).",
                metrics={"count": le0, "rate": _rate(le0, n)},
            )
        )

    # IDpol nulls handled by schema; here we can report claim frequency per policy distribution if desired
    return findings


# -----------------------------
# High-level API
# -----------------------------
def validate_dataset(
    df: pd.DataFrame,
    schema: DatasetSchema,
    *,
    coerce: bool = True,
    fail_on_warn: bool = False,
    constraints_kwargs: Optional[Dict[str, Any]] = None,
) -> ValidationReport:
    """
    Validate a dataset against schema + constraints.
    Returns a structured report suitable for CI gates.
    """
    work, schema_findings, summary = validate_schema(df, schema, coerce=coerce)

    constraints_kwargs = constraints_kwargs or {}
    constraint_findings: List[Finding] = []
    if schema.name == "freMTPL2freq":
        constraint_findings = validate_constraints_freq(work, **constraints_kwargs)
    elif schema.name == "freMTPL2sev":
        constraint_findings = validate_constraints_sev(work, **constraints_kwargs)

    findings = schema_findings + constraint_findings

    has_error = any(f.level == "ERROR" for f in findings)
    has_warn = any(f.level == "WARN" for f in findings)
    ok = (not has_error) and (not has_warn or not fail_on_warn)

    return ValidationReport(dataset=schema.name, ok=ok, findings=findings, summary=summary)


def save_report(report: ValidationReport, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    payload = {
        "dataset": report.dataset,
        "ok": report.ok,
        "summary": report.summary,
        "findings": [asdict(f) for f in report.findings],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# -----------------------------
# CLI (optional)
# -----------------------------
if __name__ == "__main__":
    # Example:
    # python -m src.data.validate --freq data/staging/freq_staged.parquet --sev data/staging/sev_staged.parquet --out artifacts/reports
    import argparse

    parser = argparse.ArgumentParser(description="Validate freMTPL2 datasets (schema + constraints)")
    parser.add_argument("--freq", type=str, required=False, help="Path to freq parquet (raw or staged)")
    parser.add_argument("--sev", type=str, required=False, help="Path to sev parquet (raw or staged)")
    parser.add_argument("--out", type=str, required=True, help="Output directory for validation reports")
    parser.add_argument("--fail-on-warn", action="store_true", help="Treat warnings as failures")
    args = parser.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    if args.freq:
        df_freq = pd.read_parquet(args.freq)
        rep = validate_dataset(
            df_freq,
            FREQ_SCHEMA,
            coerce=True,
            fail_on_warn=args.fail_on_warn,
            constraints_kwargs={"exposure_cap_policy": 1.0},
        )
        save_report(rep, out_dir / "freq_validation_report.json")
        print(f"freq ok={rep.ok} findings={len(rep.findings)}")

    if args.sev:
        df_sev = pd.read_parquet(args.sev)
        rep = validate_dataset(df_sev, SEV_SCHEMA, coerce=True, fail_on_warn=args.fail_on_warn)
        save_report(rep, out_dir / "sev_validation_report.json")
        print(f"sev ok={rep.ok} findings={len(rep.findings)}")