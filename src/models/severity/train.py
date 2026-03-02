# src/models/severity/train.py
"""
Train Severity model for Motor TPL (freMTPL2) using claim-level data.

Baseline (insurance-grade):
- Gamma GLM with log link
- Tail stabilization via winsorization at P99.9 (configurable)
- Decile calibration report (Observed vs Predicted)
- Auditability: input hash + training config + artifacts

Expected input:
- data/staging/sev_train.parquet (claim-level, matched with policy features)
  Columns expected:
    IDpol, ClaimAmount,
    Exposure, Area, VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Density, Region
"""

from __future__ import annotations

from src.models.artifacts import SeverityModelArtifact
import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Statsmodels for GLM
import statsmodels.api as sm
import patsy
from patsy import bs  # for formula bs(DrivAge, df=5), bs(VehAge, df=5)

import joblib


# -----------------------------
# Utilities
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_log(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return np.log(np.maximum(x, eps))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def decile_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    # qcut may fail if too many ties; add rank noise fallback
    try:
        df["decile"] = pd.qcut(df["y_pred"], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        r = df["y_pred"].rank(method="average")
        df["decile"] = pd.qcut(r, q=n_bins, labels=False, duplicates="drop")

    agg = (
        df.groupby("decile", dropna=True)
        .agg(
            rows=("y_true", "size"),
            true_mean=("y_true", "mean"),
            true_median=("y_true", "median"),
            pred_mean=("y_pred", "mean"),
            pred_median=("y_pred", "median"),
        )
        .reset_index()
        .sort_values("decile")
    )
    agg["ratio_true_over_pred_mean"] = agg["true_mean"] / np.maximum(agg["pred_mean"], 1e-9)
    return agg


# -----------------------------
# Config / model packaging
# -----------------------------
@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    valid_frac: float = 0.2
    cap_quantile: float = 0.999  # P99.9
    glm_family: str = "Gamma"
    link: str = "log"
    # Density as log1p. Age as bs(., df=5) for nonlinear risk.
    formula: str = (
        "ClaimAmount_capped ~ "
        "VehPower + bs(DrivAge, df=5) + bs(VehAge, df=5) + BonusMalus + log1p_Density + Exposure + "
        "C(Area) + C(VehBrand) + C(VehGas) + C(Region)"
    )




# -----------------------------
# Core training
# -----------------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = [
        "IDpol",
        "ClaimAmount",
        "Exposure",
        "Area",
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "VehBrand",
        "VehGas",
        "Density",
        "Region",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in sev_train: {missing}")

    # Enforce numeric for key measure
    df["ClaimAmount"] = pd.to_numeric(df["ClaimAmount"], errors="raise")
    if (df["ClaimAmount"] <= 0).any():
        n_bad = int((df["ClaimAmount"] <= 0).sum())
        raise ValueError(f"ClaimAmount must be strictly positive for severity modeling. Bad rows: {n_bad}")

    # Density is heavy-tailed; log1p gives a more realistic, stable effect (training-serving parity)
    df["log1p_Density"] = np.log1p(np.maximum(pd.to_numeric(df["Density"], errors="coerce").fillna(0), 0))

    return df


def compute_cap(df: pd.DataFrame, q: float) -> float:
    cap = float(df["ClaimAmount"].quantile(q))
    return cap


def apply_cap(df: pd.DataFrame, cap_value: float) -> pd.DataFrame:
    out = df.copy()
    out["ClaimAmount_capped"] = out["ClaimAmount"].clip(upper=cap_value)
    return out


def train_valid_split(df: pd.DataFrame, *, seed: int, valid_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * (1 - valid_frac))
    tr_idx, va_idx = idx[:cut], idx[cut:]
    return df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()


def fit_gamma_glm(train_df: pd.DataFrame, formula: str):
    y, X = patsy.dmatrices(formula, train_df, return_type="dataframe")
    design_info = X.design_info

    model = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.log()))
    res = model.fit()
    return res, design_info

def predict_glm(res: Any, df: pd.DataFrame, formula: str) -> np.ndarray:
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    # For prediction, only need X (but using dmatrices ensures same encoding)
    mu = res.predict(X)
    return np.asarray(mu).reshape(-1)


def build_model_card_md(
    *,
    run_id: str,
    created_at_utc: str,
    input_path: str,
    input_sha256: str,
    rows: int,
    cap_quantile: float,
    cap_value: float,
    formula: str,
    metrics: Dict[str, Any],
    notes: Dict[str, Any],
) -> str:
    # Keep it concise but "audit-friendly"
    return f"""# Severity Model Card — Motor TPL (freMTPL2)

## Run
- run_id: `{run_id}`
- created_at_utc: `{created_at_utc}`

## Data
- input: `{input_path}`
- input_sha256: `{input_sha256}`
- rows: `{rows}` (claim-level; policy features joined)
- target: `ClaimAmount` (strictly positive)

## Modeling Approach
- Model: **Gamma GLM**
- Link: **log**
- Formula:
  - `{formula}`

## Tail Handling Policy
- Winsorization (cap) applied to training target:
  - cap_quantile: `{cap_quantile}`
  - cap_value: `{cap_value:.6f}`
- Rationale:
  - Extreme outliers can dominate GLM coefficient estimates and degrade calibration for the bulk of claims.
  - Catastrophic / very large claims are typically handled via large-loss controls or separate modeling.

## Evaluation (validation split)
- log(MAE): `{metrics.get("mae_log", "n/a")}`
- log(RMSE): `{metrics.get("rmse_log", "n/a")}`
- mean(actual): `{metrics.get("actual_mean", "n/a")}`
- mean(pred): `{metrics.get("pred_mean", "n/a")}`
- median(actual): `{metrics.get("actual_median", "n/a")}`
- median(pred): `{metrics.get("pred_median", "n/a")}`

## Known Limitations
- Validation uses a random split (not time-based; dataset is not time-indexed).
- Tail policy is global; regional/segment-specific tail modelling is out of scope for baseline.
- Serving-time feature parity requires the same categorical normalization and levels; unseen categories must be handled upstream (feature layer policy).

## Governance / Audit Notes
{json.dumps(notes, indent=2, ensure_ascii=False)}
"""


def train(
    *,
    data_path: Path,
    out_model_path: Path,
    out_cap_path: Path,
    out_metrics_path: Path,
    out_deciles_path: Path,
    out_model_card_path: Path,
    config: TrainConfig,
) -> Dict[str, Any]:
    ensure_dir(out_model_path.parent)
    ensure_dir(out_cap_path.parent)
    ensure_dir(out_metrics_path.parent)
    ensure_dir(out_deciles_path.parent)
    ensure_dir(out_model_card_path.parent)

    run_id = f"sev_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}"
    created_at = utc_now_iso()

    df = load_data(data_path)
    input_hash = sha256_file(data_path)

    cap_value = compute_cap(df, config.cap_quantile)
    df2 = apply_cap(df, cap_value)

    tr, va = train_valid_split(df2, seed=config.seed, valid_frac=config.valid_frac)

    # Fit
    res, _ = fit_gamma_glm(tr, config.formula)

    # Predict on validation
    pred_va = predict_glm(res, va, config.formula)

    # Evaluate in log space (more robust for skew)
    y_va = va["ClaimAmount"].to_numpy()
    y_va_c = va["ClaimAmount_capped"].to_numpy()

    mae_log = mae(safe_log(y_va), safe_log(pred_va))
    rmse_log = rmse(safe_log(y_va), safe_log(pred_va))

    # Decile calibration on original actuals (business view) + capped actuals (model view)
    dec_orig = decile_table(y_va, pred_va, n_bins=10)
    dec_cap = decile_table(y_va_c, pred_va, n_bins=10)
    dec_orig = dec_orig.add_prefix("orig_")
    dec_cap = dec_cap.add_prefix("cap_")
    deciles = pd.concat([dec_orig, dec_cap.drop(columns=["cap_decile"])], axis=1)

    deciles.to_csv(out_deciles_path, index=False)

    metrics = {
        "run_id": run_id,
        "created_at_utc": created_at,
        "input_path": str(data_path),
        "input_sha256": input_hash,
        "rows": int(len(df)),
        "valid_frac": config.valid_frac,
        "cap_quantile": config.cap_quantile,
        "cap_value": cap_value,
        "mae_log": mae_log,
        "rmse_log": rmse_log,
        "actual_mean": float(np.mean(y_va)),
        "pred_mean": float(np.mean(pred_va)),
        "actual_median": float(np.median(y_va)),
        "pred_median": float(np.median(pred_va)),
        "glm_aic": float(res.aic) if hasattr(res, "aic") else None,
        "glm_deviance": float(res.deviance) if hasattr(res, "deviance") else None,
        "glm_scale": float(res.scale) if hasattr(res, "scale") else None,
    }
    out_metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save cap policy (frozen)
    cap_payload = {
        "run_id": run_id,
        "created_at_utc": created_at,
        "cap_quantile": config.cap_quantile,
        "cap_value": cap_value,
        "target": "ClaimAmount",
        "applies_to": "training_target_winsorization",
        "notes": "Cap is applied for training stability; catastrophic claims quarantined separately for audit.",
    }
    out_cap_path.write_text(json.dumps(cap_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    factor_cols = [c for c in ["Area", "VehBrand", "VehGas", "Region"] if c in tr.columns]
    factor_levels = {
        c: sorted(tr[c].dropna().astype(str).unique().tolist()) for c in factor_cols
    }

    n_anchor = min(2000, len(tr))
    rng = np.random.default_rng(config.seed)
    anchor_idx = rng.choice(len(tr), size=n_anchor, replace=False)
    spline_anchor = {
        "DrivAge": tr["DrivAge"].iloc[anchor_idx].to_numpy(),
        "VehAge": tr["VehAge"].iloc[anchor_idx].to_numpy(),
    }

    # Save model artifact
    artifact = SeverityModelArtifact(
        fitted_result=res,
        formula=config.formula,
        cap_value=cap_value,
        config=asdict(config),
        factor_levels=factor_levels,
        spline_anchor=spline_anchor,
    )
    joblib.dump(artifact, out_model_path)

    # Model card
    notes = {
        "tail_observation": "Extreme claims concentrated in specific regions (e.g., R24/R82); documented for monitoring.",
        "unmatched_claims_handling": "Claims without matching policy features excluded from training and stored separately (sev_unmatched_claims.parquet).",
    }
    md = build_model_card_md(
        run_id=run_id,
        created_at_utc=created_at,
        input_path=str(data_path),
        input_sha256=input_hash,
        rows=int(len(df)),
        cap_quantile=config.cap_quantile,
        cap_value=cap_value,
        formula=config.formula,
        metrics=metrics,
        notes=notes,
    )
    out_model_card_path.write_text(md, encoding="utf-8")

    return {
        "run_id": run_id,
        "model_path": str(out_model_path),
        "cap_path": str(out_cap_path),
        "metrics_path": str(out_metrics_path),
        "deciles_path": str(out_deciles_path),
        "model_card_path": str(out_model_card_path),
        "cap_value": cap_value,
        "rows": int(len(df)),
    }


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train Severity model (Gamma GLM, log link) with P99.9 cap")
    parser.add_argument(
        "--data",
        type=str,
        default=r"data\staging\sev_train.parquet",
        help="Path to sev_train.parquet",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=r"artifacts\models\severity",
        help="Output directory for model artifacts",
    )
    parser.add_argument(
        "--reportdir",
        type=str,
        default=r"artifacts\reports\severity",
        help="Output directory for reports",
    )
    parser.add_argument("--cap-quantile", type=float, default=0.999, help="Cap quantile (default: 0.999)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--valid-frac", type=float, default=0.2, help="Validation fraction (default: 0.2)")
    args = parser.parse_args()

    cfg = TrainConfig(seed=args.seed, valid_frac=args.valid_frac, cap_quantile=args.cap_quantile)

    outdir = Path(args.outdir)
    reportdir = Path(args.reportdir)

    summary = train(
        data_path=Path(args.data),
        out_model_path=outdir / "sev_glm_gamma.joblib",
        out_cap_path=outdir / "sev_cap.json",
        out_metrics_path=reportdir / "sev_metrics.json",
        out_deciles_path=reportdir / "sev_deciles.csv",
        out_model_card_path=reportdir / "sev_model_card.md",
        config=cfg,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()