# src/data/staging.py
"""
Staging layer for freMTPL2freq / freMTPL2sev.

Purpose (insurance-grade):
- Canonicalize keys (IDpol) for safe joins
- Apply controlled, documented data policies (e.g., Exposure cap)
- Normalize categorical strings (trim/case)
- Write staged parquet + a staging report (JSON) for governance/audit

IMPORTANT:
- Never modify raw snapshots. Staging produces new artifacts.
- Any transformation that can affect modeling must be documented in the report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_str(s: Any) -> str:
    return "" if s is None else str(s)


# -----------------------------
# Canonicalization policies
# -----------------------------
def canonicalize_idpol_to_int64(df: pd.DataFrame, *, dataset: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Canonicalize IDpol to int64 for stable joins.

    Why:
    - Raw freq sometimes reads IDpol as float (e.g., 1198145.0)
    - Raw sev often reads IDpol as int
    - Joins on float/int can silently fail or degrade reproducibility

    Policy:
    - Require IDpol present and non-null
    - Require values are integer-like (no fractional part)
    - Convert to pandas Int64 (nullable) then to int64 (non-null enforced)
    """
    if "IDpol" not in df.columns:
        raise ValueError(f"[{dataset}] Missing required key column: IDpol")

    out = df.copy()

    # Ensure non-null
    nulls = int(out["IDpol"].isna().sum())
    if nulls > 0:
        raise ValueError(f"[{dataset}] IDpol contains nulls: {nulls}")

    # Convert to numeric (if string/object), then check integer-like
    id_num = pd.to_numeric(out["IDpol"], errors="coerce")
    bad_parse = int(id_num.isna().sum())
    if bad_parse > 0:
        raise ValueError(f"[{dataset}] IDpol has {bad_parse} non-numeric values (cannot parse)")

    # Integer-like check (e.g., 123.0 OK, 123.4 NOT OK)
    frac = (id_num % 1 != 0)
    non_integer_like = int(frac.sum())
    if non_integer_like > 0:
        sample_bad = id_num[frac].head(5).tolist()
        raise ValueError(
            f"[{dataset}] IDpol has non-integer-like values: {non_integer_like}. Examples: {sample_bad}"
        )

    # Cast to int64
    out["IDpol"] = id_num.astype("int64")

    meta = {
        "policy": "canonicalize_idpol_to_int64",
        "nulls": nulls,
        "bad_parse": bad_parse,
        "non_integer_like": non_integer_like,
        "dtype_after": str(out["IDpol"].dtype),
    }
    return out, meta


def normalize_categories(df: pd.DataFrame, *, dataset: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalize common categorical columns:
    - trim whitespace
    - unify casing where appropriate
    - keep as string dtype

    NOTE:
    - We keep normalization minimal here (no bucketing, no encodings).
    - Bucketing rare categories is a feature-layer responsibility.
    """
    out = df.copy()
    cols = [c for c in ["Area", "VehBrand", "VehGas", "Region"] if c in out.columns]

    changed = {}
    for c in cols:
        before_na = int(out[c].isna().sum())
        # Convert to string safely and normalize
        s = out[c].map(_safe_str).astype("string")
        s = s.str.strip()

        # Common conventions
        if c in ("VehGas",):
            s = s.str.lower()  # diesel / regular
        if c in ("Area", "Region"):
            s = s.str.upper()  # A / B / ... ; region codes

        out[c] = s
        after_na = int(out[c].isna().sum())

        changed[c] = {
            "na_before": before_na,
            "na_after": after_na,
            "dtype_after": str(out[c].dtype),
        }

    meta = {"policy": "normalize_categories", "columns": changed}
    return out, meta


def apply_exposure_policy_cap_1(df: pd.DataFrame, *, dataset: str, cap: float = 1.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply exposure governance policy:
    - Assume Exposure is a policy-year fraction in (0, 1]
    - Cap rare values above 1.0 to 1.0

    IMPORTANT:
    - This is a controlled transformation in STAGING (raw remains immutable).
    - All changes are recorded in staging report for auditability.
    """
    if "Exposure" not in df.columns:
        raise ValueError(f"[{dataset}] Missing required column: Exposure")

    out = df.copy()
    exp = pd.to_numeric(out["Exposure"], errors="coerce")

    # Basic sanity: exposure must be > 0 for log-offset modeling
    nulls = int(exp.isna().sum())
    le0 = int((exp <= 0).sum(skipna=True))
    if nulls > 0:
        raise ValueError(f"[{dataset}] Exposure has NaNs after numeric coercion: {nulls}")
    if le0 > 0:
        raise ValueError(f"[{dataset}] Exposure has <=0 values (invalid for log-offset): {le0}")

    gt_cap = int((exp > cap).sum())
    max_before = float(exp.max())

    # Cap and write back
    exp_capped = exp.clip(upper=cap)
    out["Exposure"] = exp_capped

    meta = {
        "policy": "cap_exposure",
        "cap_value": cap,
        "rows_capped": gt_cap,
        "max_before": max_before,
        "max_after": float(exp_capped.max()),
    }
    return out, meta


# -----------------------------
# Orchestrator (freq + sev)
# -----------------------------
def stage_freq_and_sev(
    *,
    freq_snapshot_path: Path,
    sev_snapshot_path: Path,
    out_dir: Path,
    report_path: Path,
    parquet_engine: str = "pyarrow",
    compression: str = "snappy",
    exposure_cap: float = 1.0,
) -> Dict[str, Any]:
    """
    Read raw snapshots -> apply staging policies -> write staged parquet + report.

    Output:
      - out_dir/freq_staged.parquet
      - out_dir/sev_staged.parquet
      - report_path JSON with policies + counts + hashes
    """
    _ensure_dir(out_dir)
    _ensure_dir(report_path.parent)

    # Load raw snapshots (immutable inputs)
    df_freq = pd.read_parquet(freq_snapshot_path)
    df_sev = pd.read_parquet(sev_snapshot_path)

    # Apply policies (freq)
    freq_meta: Dict[str, Any] = {"dataset": "freq", "policies": []}

    df_freq, m = canonicalize_idpol_to_int64(df_freq, dataset="freq")
    freq_meta["policies"].append(m)

    df_freq, m = normalize_categories(df_freq, dataset="freq")
    freq_meta["policies"].append(m)

    df_freq, m = apply_exposure_policy_cap_1(df_freq, dataset="freq", cap=exposure_cap)
    freq_meta["policies"].append(m)

    # Apply policies (sev)
    sev_meta: Dict[str, Any] = {"dataset": "sev", "policies": []}

    df_sev, m = canonicalize_idpol_to_int64(df_sev, dataset="sev")
    sev_meta["policies"].append(m)

    # Staged outputs
    freq_out = out_dir / "freq_staged.parquet"
    sev_out = out_dir / "sev_staged.parquet"

    df_freq.to_parquet(freq_out, index=False, engine=parquet_engine, compression=compression)
    df_sev.to_parquet(sev_out, index=False, engine=parquet_engine, compression=compression)

    # Hashes for auditability
    report = {
        "created_at_utc": _utc_now_iso(),
        "inputs": {
            "freq_snapshot_path": str(freq_snapshot_path),
            "sev_snapshot_path": str(sev_snapshot_path),
            "freq_snapshot_sha256": _sha256_file(freq_snapshot_path),
            "sev_snapshot_sha256": _sha256_file(sev_snapshot_path),
        },
        "outputs": {
            "freq_staged_path": str(freq_out),
            "sev_staged_path": str(sev_out),
            "freq_staged_sha256": _sha256_file(freq_out),
            "sev_staged_sha256": _sha256_file(sev_out),
        },
        "row_counts": {
            "freq_rows": int(df_freq.shape[0]),
            "sev_rows": int(df_sev.shape[0]),
        },
        "metadata": {
            "parquet_engine": parquet_engine,
            "compression": compression,
        },
        "policies": [freq_meta, sev_meta],
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "freq_staged": str(freq_out),
        "sev_staged": str(sev_out),
        "report": str(report_path),
        "freq_rows": int(df_freq.shape[0]),
        "sev_rows": int(df_sev.shape[0]),
    }


if __name__ == "__main__":
    # Example:
    # python -m src.data.staging \
    #   --freq-snapshot data/raw_snapshots/freMTPL2freq__<RUN>.parquet \
    #   --sev-snapshot  data/raw_snapshots/freMTPL2sev__<RUN>.parquet \
    #   --out           data/staging \
    #   --report        artifacts/reports/staging_report.json
    import argparse

    parser = argparse.ArgumentParser(description="Stage freMTPL2freq & freMTPL2sev (governed transformations)")
    parser.add_argument("--freq-snapshot", type=str, required=True, help="Path to freq raw snapshot parquet")
    parser.add_argument("--sev-snapshot", type=str, required=True, help="Path to sev raw snapshot parquet")
    parser.add_argument("--out", type=str, required=True, help="Output directory for staged parquet files")
    parser.add_argument("--report", type=str, required=True, help="Path to write staging report JSON")
    parser.add_argument("--exposure-cap", type=float, default=1.0, help="Exposure cap value (default: 1.0)")
    args = parser.parse_args()

    summary = stage_freq_and_sev(
        freq_snapshot_path=Path(args.freq_snapshot),
        sev_snapshot_path=Path(args.sev_snapshot),
        out_dir=Path(args.out),
        report_path=Path(args.report),
        exposure_cap=float(args.exposure_cap),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))