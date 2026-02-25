# src/data/joins.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


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


def _require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}")


def build_severity_training_dataset(
    *,
    freq_staged_path: Path,
    sev_staged_path: Path,
    out_path: Path,
    report_path: Optional[Path] = None,
    policy_feature_cols: Optional[List[str]] = None,
    validate_unique_policy_key: bool = True,
) -> Dict[str, Any]:
    """
    Build claim-level severity dataset with audit-friendly handling of unmatched claims.

    Outputs:
      - out_path: matched (trainable) claim rows with policy features
      - out_path sibling: sev_unmatched_claims.parquet (claims missing policy features)
      - report_path (optional): full diagnostics + hashes
    """
    _ensure_dir(out_path.parent)
    if report_path is not None:
        _ensure_dir(report_path.parent)

    freq = pd.read_parquet(freq_staged_path)
    sev = pd.read_parquet(sev_staged_path)

    _require_cols(freq, ["IDpol"], "freq_staged")
    _require_cols(sev, ["IDpol", "ClaimAmount"], "sev_staged")

    # Enforce unique policy key (recommended)
    dup_policies = int(freq.duplicated(subset=["IDpol"]).sum())
    if validate_unique_policy_key and dup_policies > 0:
        raise ValueError(
            f"[freq_staged] Expected unique IDpol, but found {dup_policies} duplicate IDpol rows."
        )

    # Choose policy columns to bring in
    if policy_feature_cols is None:
        # Exclude ClaimNb to avoid leakage into severity
        drop_cols = {"ClaimNb"}
        policy_feature_cols = [c for c in freq.columns if c not in drop_cols]

    if "IDpol" not in policy_feature_cols:
        policy_feature_cols = ["IDpol"] + policy_feature_cols

    freq_small = freq[policy_feature_cols].copy()

    # Ensure ClaimAmount is strictly positive (severity is conditional on claim)
    amt = pd.to_numeric(sev["ClaimAmount"], errors="coerce")
    na_amt = int(amt.isna().sum())
    non_pos = int((amt <= 0).sum(skipna=True))
    if na_amt > 0:
        raise ValueError(f"[sev_staged] ClaimAmount contains NaNs after coercion: {na_amt}")
    if non_pos > 0:
        raise ValueError(f"[sev_staged] ClaimAmount has non-positive values: {non_pos}")

    sev = sev.copy()
    sev["ClaimAmount"] = amt

    # LEFT join to explicitly detect unmatched claims
    joined = sev.merge(freq_small, on="IDpol", how="left", indicator=True, validate="many_to_one")

    unmatched = joined[joined["_merge"] == "left_only"].copy()
    matched = joined[joined["_merge"] == "both"].copy()

    unmatched.drop(columns=["_merge"], inplace=True)
    matched.drop(columns=["_merge"], inplace=True)

    # Persist outputs
    matched.to_parquet(out_path, index=False)

    unmatched_path = out_path.with_name("sev_unmatched_claims.parquet")
    unmatched.to_parquet(unmatched_path, index=False)

    # Diagnostics
    n_claims = int(len(sev))
    n_matched = int(len(matched))
    n_unmatched = int(len(unmatched))
    match_rate = float(n_matched / n_claims) if n_claims else 0.0

    missing_unique = int(unmatched["IDpol"].nunique()) if n_unmatched else 0
    top_missing = (
        unmatched["IDpol"].value_counts().head(50).to_dict() if n_unmatched else {}
    )

    result = {
        "created_at_utc": _utc_now_iso(),
        "inputs": {
            "freq_staged_path": str(freq_staged_path),
            "sev_staged_path": str(sev_staged_path),
            "freq_staged_sha256": _sha256_file(freq_staged_path),
            "sev_staged_sha256": _sha256_file(sev_staged_path),
        },
        "outputs": {
            "sev_train_path": str(out_path),
            "sev_train_sha256": _sha256_file(out_path),
            "sev_train_rows": n_matched,
            "sev_unmatched_path": str(unmatched_path),
            "sev_unmatched_sha256": _sha256_file(unmatched_path),
            "sev_unmatched_rows": n_unmatched,
        },
        "diagnostics": {
            "claims_rows_input": n_claims,
            "rows_matched": n_matched,
            "rows_unmatched": n_unmatched,
            "match_rate": match_rate,
            "dup_policies_in_freq": dup_policies,
            "missing_unique_idpol": missing_unique,
            "top_missing_idpol_counts": top_missing,
            "merge_validate": "many_to_one",
        },
        "policy": {
            "severity_trained_on": "claim rows only (ClaimAmount > 0) with policy features present",
            "unmatched_handling": "unmatched claims quarantined to sev_unmatched_claims.parquet (excluded from training)",
            "excluded_policy_cols": ["ClaimNb"],
        },
    }

    if report_path is not None:
        report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build severity training dataset (matched + unmatched quarantine)")
    parser.add_argument("--freq", type=str, required=True, help="Path to staged freq parquet")
    parser.add_argument("--sev", type=str, required=True, help="Path to staged sev parquet")
    parser.add_argument("--out", type=str, required=True, help="Output path for matched severity training parquet")
    parser.add_argument("--report", type=str, required=False, help="Optional path for join report JSON")
    parser.add_argument("--no-unique-check", action="store_true", help="Disable unique IDpol check on freq")
    args = parser.parse_args()

    summary = build_severity_training_dataset(
        freq_staged_path=Path(args.freq),
        sev_staged_path=Path(args.sev),
        out_path=Path(args.out),
        report_path=Path(args.report) if args.report else None,
        validate_unique_policy_key=not args.no_unique_check,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))