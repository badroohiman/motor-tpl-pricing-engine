# src/data/ingest.py
"""
Data ingest for freMTPL2freq and freMTPL2sev.

Level-up goals:
- Read raw CSVs robustly with explicit dtypes where possible
- Stronger schema + basic data quality checks (nulls, ranges, types)
- Save immutable raw snapshots (Parquet) with stable metadata (hashes, bytes, rows, cols)
- Manifest includes: input CSV hashes + snapshot hashes + dtypes + validation summary
- Deterministic and auditable

Notes:
- Keep this module dependency-light (pandas + stdlib) — numpy used only for isfinite speed/stability
- For very large files: consider pyarrow.csv / polars / chunked reads later
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.utils import ensure_dir, rate


# -----------------------------
# Data classes for auditability
# -----------------------------
@dataclass(frozen=True)
class DatasetQuality:
    dataset: str
    checks: Dict[str, Any]
    ok: bool


@dataclass(frozen=True)
class IngestSource:
    name: str
    kind: str  # "csv" or "parquet"
    path: str
    sha256: str
    bytes: int
    rows: int
    cols: int
    columns: List[str]
    dtypes: Dict[str, str]


@dataclass(frozen=True)
class IngestManifest:
    run_id: str
    created_at_utc: str
    pandas_version: str
    parquet_engine: str
    sources: List[IngestSource]
    quality: List[DatasetQuality]
    notes: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Low-level utilities
# -----------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run_id(prefix: str = "ingest") -> str:
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    ts = ts.replace(":", "-").replace("+00:00", "Z")
    return f"{prefix}_{ts}"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _df_dtypes_map(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c): str(df[c].dtype) for c in df.columns}


def _read_csv_robust(
    csv_path: Path,
    *,
    dtype: Optional[Dict[str, Any]] = None,
    parse_dates: Optional[List[str]] = None,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    enc = encoding or "utf-8"
    try:
        return pd.read_csv(
            csv_path,
            dtype=dtype,
            parse_dates=parse_dates,
            encoding=enc,
            low_memory=False,
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            csv_path,
            dtype=dtype,
            parse_dates=parse_dates,
            encoding="latin1",
            low_memory=False,
        )


def _require_columns(df: pd.DataFrame, required: Iterable[str], dataset_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{dataset_name}] Missing required columns: {missing}")


def _fail_on_duplicate_columns(df: pd.DataFrame, dataset_name: str) -> None:
    cols = list(df.columns)
    dupes = sorted({c for c in cols if cols.count(c) > 1})
    if dupes:
        raise ValueError(f"[{dataset_name}] Duplicate column names found: {dupes}")


def _coerce_numeric(
    df: pd.DataFrame,
    cols: List[str],
    dataset_name: str,
    *,
    errors: str = "raise",
) -> Dict[str, Any]:
    """
    Coerce selected columns to numeric if they exist.
    Returns metrics about coercion (how many values became NaN).

    errors:
      - "raise": raise if conversion fails
      - "coerce": invalid parses become NaN
    """
    metrics: Dict[str, Any] = {"columns": {}}
    for c in cols:
        if c not in df.columns:
            continue

        before_na = int(df[c].isna().sum())
        try:
            df[c] = pd.to_numeric(df[c], errors=errors)
        except Exception as e:
            raise ValueError(f"[{dataset_name}] Failed numeric coercion for column '{c}': {e}") from e
        after_na = int(df[c].isna().sum())

        metrics["columns"][c] = {
            "na_before": before_na,
            "na_after": after_na,
            "new_nas_from_coercion": max(0, after_na - before_na),
        }
    return metrics


def _rate(count: int, denom: int) -> float:
    return rate(count, denom)


def _check_not_null(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    if col not in df.columns:
        return {"present": False}
    nulls = int(df[col].isna().sum())
    return {"present": True, "nulls": nulls, "null_rate": _rate(nulls, len(df))}


def _check_non_negative(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    if col not in df.columns:
        return {"present": False}
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s):
        return {"present": True, "numeric": False}
    bad = int((s < 0).sum(skipna=True))
    return {"present": True, "numeric": True, "negatives": bad, "negative_rate": _rate(bad, len(df))}


def _check_strictly_positive(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    if col not in df.columns:
        return {"present": False}
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s):
        return {"present": True, "numeric": False}
    le0 = int((s <= 0).sum(skipna=True))
    zeros = int((s == 0).sum(skipna=True))
    return {
        "present": True,
        "numeric": True,
        "non_positive": le0,
        "non_positive_rate": _rate(le0, len(df)),
        "zeros": zeros,
        "zero_rate": _rate(zeros, len(df)),
    }


def _check_range(df: pd.DataFrame, col: str, *, lo: Optional[float] = None, hi: Optional[float] = None) -> Dict[str, Any]:
    if col not in df.columns:
        return {"present": False}
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s):
        return {"present": True, "numeric": False}

    bad_lo = 0
    bad_hi = 0
    if lo is not None:
        bad_lo = int((s < lo).sum(skipna=True))
    if hi is not None:
        bad_hi = int((s > hi).sum(skipna=True))

    total_bad = bad_lo + bad_hi
    return {
        "present": True,
        "numeric": True,
        "lo": lo,
        "hi": hi,
        "below_lo": bad_lo,
        "above_hi": bad_hi,
        "out_of_range": total_bad,
        "out_of_range_rate": _rate(total_bad, len(df)),
    }


def _check_integer_like(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    if col not in df.columns:
        return {"present": False}
    s = df[col]
    if pd.api.types.is_integer_dtype(s):
        return {"present": True, "integer_dtype": True, "non_integer_like": 0, "non_integer_like_rate": 0.0}
    if not pd.api.types.is_numeric_dtype(s):
        return {"present": True, "integer_dtype": False, "numeric": False}

    v = s.to_numpy(dtype="float64", copy=False)
    mask = np.isfinite(v)
    v = v[mask]
    if v.size == 0:
        return {"present": True, "integer_dtype": False, "numeric": True, "non_integer_like": 0, "non_integer_like_rate": 0.0}

    non_int = int(np.sum(v != np.floor(v)))
    denom = int(mask.sum())
    return {
        "present": True,
        "integer_dtype": False,
        "numeric": True,
        "non_integer_like": non_int,
        "non_integer_like_rate": _rate(non_int, denom),
        "finite_count": denom,
    }


def _check_unique_key(df: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
    for c in cols:
        if c not in df.columns:
            return {"present": False, "missing": c}
    dup_rows = int(df.duplicated(subset=cols).sum())
    return {"present": True, "dupe_rows": dup_rows, "dupe_rate": _rate(dup_rows, len(df)), "key_cols": cols}


# -----------------------------
# Public ingest functions
# -----------------------------
def ingest_freq(
    freq_csv: Path,
    *,
    expected_columns: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
    encoding: Optional[str] = None,
    enforce_basic_quality: bool = True,
) -> Tuple[pd.DataFrame, DatasetQuality]:
    df = _read_csv_robust(freq_csv, dtype=dtype, encoding=encoding)
    _fail_on_duplicate_columns(df, "freq")
    if expected_columns:
        _require_columns(df, expected_columns, "freq")

    # Coerce numerics and capture coercion metrics
    coercion = _coerce_numeric(df, ["ClaimNb", "Exposure"], "freq", errors="coerce")

    checks: Dict[str, Any] = {"coercion": coercion}
    ok = True

    if enforce_basic_quality:
        checks["IDpol_not_null"] = _check_not_null(df, "IDpol")
        checks["Exposure_positive"] = _check_range(df, "Exposure", lo=0.0)  # will catch negatives
        checks["Exposure_reasonable_max_1"] = _check_range(df, "Exposure", lo=0.0, hi=1.0)
        checks["ClaimNb_non_negative"] = _check_non_negative(df, "ClaimNb")
        checks["ClaimNb_integer_like"] = _check_integer_like(df, "ClaimNb")
        checks["IDpol_duplicates"] = _check_unique_key(df, ["IDpol"])

        # Additional gate: new NaNs due to coercion should be 0 (or you decide threshold)
        new_na_claim = coercion["columns"].get("ClaimNb", {}).get("new_nas_from_coercion", 0)
        new_na_exp = coercion["columns"].get("Exposure", {}).get("new_nas_from_coercion", 0)

        idpol_nulls = checks["IDpol_not_null"].get("nulls", 0)
        exp_out_lo = checks["Exposure_positive"].get("below_lo", 0)
        claim_negs = checks["ClaimNb_non_negative"].get("negatives", 0)

        # Hard fails (conservative)
        if idpol_nulls > 0:
            ok = False
        if exp_out_lo > 0:
            ok = False
        if claim_negs > 0:
            ok = False
        if new_na_claim > 0 or new_na_exp > 0:
            ok = False

    return df, DatasetQuality(dataset="freq", checks=checks, ok=ok)


def ingest_sev(
    sev_csv: Path,
    *,
    expected_columns: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
    encoding: Optional[str] = None,
    enforce_basic_quality: bool = True,
) -> Tuple[pd.DataFrame, DatasetQuality]:
    df = _read_csv_robust(sev_csv, dtype=dtype, encoding=encoding)
    _fail_on_duplicate_columns(df, "sev")
    if expected_columns:
        _require_columns(df, expected_columns, "sev")

    coercion = _coerce_numeric(df, ["ClaimAmount"], "sev", errors="coerce")

    checks: Dict[str, Any] = {"coercion": coercion}
    ok = True

    if enforce_basic_quality:
        checks["IDpol_not_null"] = _check_not_null(df, "IDpol")
        checks["ClaimAmount_strictly_positive"] = _check_strictly_positive(df, "ClaimAmount")

        new_na_amt = coercion["columns"].get("ClaimAmount", {}).get("new_nas_from_coercion", 0)
        idpol_nulls = checks["IDpol_not_null"].get("nulls", 0)
        non_pos = checks["ClaimAmount_strictly_positive"].get("non_positive", 0)

        if idpol_nulls > 0:
            ok = False
        if new_na_amt > 0:
            ok = False
        if non_pos > 0:
            ok = False

    return df, DatasetQuality(dataset="sev", checks=checks, ok=ok)


def write_raw_snapshots(
    df_freq: pd.DataFrame,
    df_sev: pd.DataFrame,
    *,
    raw_dir: Path,
    run_id: Optional[str] = None,
    freq_name: str = "freMTPL2freq",
    sev_name: str = "freMTPL2sev",
    parquet_engine: str = "pyarrow",
    compression: str = "snappy",
) -> Tuple[Path, Path]:
    ensure_dir(raw_dir)
    rid = run_id or _run_id("raw")

    out_freq = raw_dir / f"{freq_name}__{rid}.parquet"
    out_sev = raw_dir / f"{sev_name}__{rid}.parquet"

    df_freq.to_parquet(out_freq, index=False, engine=parquet_engine, compression=compression)
    df_sev.to_parquet(out_sev, index=False, engine=parquet_engine, compression=compression)

    return out_freq, out_sev


def build_manifest(
    *,
    run_id: str,
    freq_csv: Path,
    sev_csv: Path,
    freq_snapshot: Path,
    sev_snapshot: Path,
    df_freq: pd.DataFrame,
    df_sev: pd.DataFrame,
    q_freq: DatasetQuality,
    q_sev: DatasetQuality,
    parquet_engine: str,
) -> IngestManifest:
    def _src(name: str, kind: str, path: Path, df: pd.DataFrame) -> IngestSource:
        return IngestSource(
            name=name,
            kind=kind,
            path=str(path),
            sha256=_sha256_file(path),
            bytes=path.stat().st_size,
            rows=int(df.shape[0]),
            cols=int(df.shape[1]),
            columns=[str(c) for c in df.columns],
            dtypes=_df_dtypes_map(df),
        )

    sources = [
        _src("freq", "csv", freq_csv, df_freq),
        _src("sev", "csv", sev_csv, df_sev),
        _src("freq_snapshot", "parquet", freq_snapshot, df_freq),
        _src("sev_snapshot", "parquet", sev_snapshot, df_sev),
    ]

    return IngestManifest(
        run_id=run_id,
        created_at_utc=_utc_now_iso(),
        pandas_version=pd.__version__,
        parquet_engine=parquet_engine,
        sources=sources,
        quality=[q_freq, q_sev],
        notes={"freq_csv_name": freq_csv.name, "sev_csv_name": sev_csv.name},
    )


def save_manifest(manifest: IngestManifest, *, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(asdict(manifest), indent=2, ensure_ascii=False), encoding="utf-8")


def ingest_all(
    *,
    freq_csv: Path,
    sev_csv: Path,
    raw_dir: Path,
    manifest_path: Path,
    expected_freq_cols: Optional[List[str]] = None,
    expected_sev_cols: Optional[List[str]] = None,
    dtype_freq: Optional[Dict[str, Any]] = None,
    dtype_sev: Optional[Dict[str, Any]] = None,
    encoding: Optional[str] = None,
    parquet_engine: str = "pyarrow",
    compression: str = "snappy",
    enforce_basic_quality: bool = True,
    fail_on_quality: bool = True,
) -> Dict[str, Any]:
    run_id = _run_id("ingest")

    df_freq, q_freq = ingest_freq(
        freq_csv,
        expected_columns=expected_freq_cols,
        dtype=dtype_freq,
        encoding=encoding,
        enforce_basic_quality=enforce_basic_quality,
    )
    df_sev, q_sev = ingest_sev(
        sev_csv,
        expected_columns=expected_sev_cols,
        dtype=dtype_sev,
        encoding=encoding,
        enforce_basic_quality=enforce_basic_quality,
    )

    if fail_on_quality and enforce_basic_quality and (not q_freq.ok or not q_sev.ok):
        details = {"freq": q_freq.checks, "sev": q_sev.checks}
        raise ValueError(f"Ingest quality checks failed:\n{json.dumps(details, indent=2, ensure_ascii=False)}")

    snap_freq, snap_sev = write_raw_snapshots(
        df_freq,
        df_sev,
        raw_dir=raw_dir,
        run_id=run_id,
        parquet_engine=parquet_engine,
        compression=compression,
    )

    manifest = build_manifest(
        run_id=run_id,
        freq_csv=freq_csv,
        sev_csv=sev_csv,
        freq_snapshot=snap_freq,
        sev_snapshot=snap_sev,
        df_freq=df_freq,
        df_sev=df_sev,
        q_freq=q_freq,
        q_sev=q_sev,
        parquet_engine=parquet_engine,
    )
    save_manifest(manifest, out_path=manifest_path)

    return {
        "run_id": run_id,
        "freq_rows": int(df_freq.shape[0]),
        "sev_rows": int(df_sev.shape[0]),
        "freq_snapshot": str(snap_freq),
        "sev_snapshot": str(snap_sev),
        "manifest": str(manifest_path),
        "quality_ok": bool(q_freq.ok and q_sev.ok),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest freMTPL2freq & freMTPL2sev")
    parser.add_argument("--freq", type=str, required=True, help="Path to freMTPL2freq CSV")
    parser.add_argument("--sev", type=str, required=True, help="Path to freMTPL2sev CSV")
    parser.add_argument("--out", type=str, required=True, help="Output directory for raw snapshots")
    parser.add_argument("--manifest", type=str, required=True, help="Path to write ingest manifest JSON")
    parser.add_argument("--no-fail", action="store_true", help="Do not fail on quality checks")
    args = parser.parse_args()

    expected_freq = ["IDpol", "ClaimNb", "Exposure"]
    expected_sev = ["IDpol", "ClaimAmount"]

    summary = ingest_all(
        freq_csv=Path(args.freq),
        sev_csv=Path(args.sev),
        raw_dir=Path(args.out),
        manifest_path=Path(args.manifest),
        expected_freq_cols=expected_freq,
        expected_sev_cols=expected_sev,
        enforce_basic_quality=True,
        fail_on_quality=not args.no_fail,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))