# src/models/frequency/train.py
"""
Train Frequency model for Motor TPL (freMTPL2) using policy-level data.

Baseline (insurance-grade):
- Negative Binomial GLM with log link
- Exposure handled as an OFFSET: log(Exposure)
- Calibration by deciles of predicted annual frequency (rate)
- Auditability: input hash + training config + artifacts

Expected input:
- data/staging/freq_staged.parquet
  Columns expected (at least):
    IDpol, ClaimNb, Exposure,
    Area, VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Density, Region

Outputs:
- artifacts/models/frequency/freq_glm_nb.joblib
- artifacts/reports/frequency/freq_metrics.json
- artifacts/reports/frequency/freq_deciles.csv
- artifacts/reports/frequency/freq_model_card.md
"""

from __future__ import annotations
from patsy import dmatrices
from src.models.artifacts import FrequencyModelArtifact
import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import patsy
import statsmodels.api as sm
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


def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(x, eps))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def exposure_weighted_rate(claims: np.ndarray, exposure: np.ndarray) -> float:
    denom = float(np.sum(exposure))
    return float(np.sum(claims) / denom) if denom > 0 else 0.0


def decile_table_rate(
    *,
    y_claims: np.ndarray,
    exposure: np.ndarray,
    rate_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Calibration table by deciles of predicted annual frequency (rate).

    Observed rate is computed exposure-weighted:
      obs_rate = sum(ClaimNb) / sum(Exposure)
    """
    df = pd.DataFrame(
        {"ClaimNb": y_claims, "Exposure": exposure, "rate_pred": rate_pred}
    )

    # qcut may fail if too many ties; rank fallback
    try:
        df["decile"] = pd.qcut(df["rate_pred"], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        r = df["rate_pred"].rank(method="average")
        df["decile"] = pd.qcut(r, q=n_bins, labels=False, duplicates="drop")

    g = df.groupby("decile", dropna=True)
    out = g.agg(
        rows=("ClaimNb", "size"),
        exposure_sum=("Exposure", "sum"),
        claims_sum=("ClaimNb", "sum"),
        pred_rate_mean=("rate_pred", "mean"),
        pred_rate_median=("rate_pred", "median"),
    ).reset_index().sort_values("decile")

    out["obs_rate"] = out["claims_sum"] / np.maximum(out["exposure_sum"], 1e-12)
    out["obs_over_pred"] = out["obs_rate"] / np.maximum(out["pred_rate_mean"], 1e-12)
    return out


# -----------------------------
# Config / model packaging
# -----------------------------
@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    valid_frac: float = 0.2
    # Frequency formula excludes ClaimNb/Exposure obviously; Exposure is an offset
    formula: str = (
        "ClaimNb ~ "
        "VehPower + VehAge + DrivAge + BonusMalus + Density + "
        "C(Area) + C(VehBrand) + C(VehGas) + C(Region)"
    )
    model: str = "NegativeBinomialGLM"
    link: str = "log"




# -----------------------------
# Core training
# -----------------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    required = [
        "IDpol", "ClaimNb", "Exposure",
        "Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in freq_staged: {missing}")

    df["ClaimNb"] = pd.to_numeric(df["ClaimNb"], errors="raise")
    df["Exposure"] = pd.to_numeric(df["Exposure"], errors="raise")

    if (df["ClaimNb"] < 0).any():
        n_bad = int((df["ClaimNb"] < 0).sum())
        raise ValueError(f"ClaimNb must be non-negative. Bad rows: {n_bad}")

    if (df["Exposure"] <= 0).any():
        n_bad = int((df["Exposure"] <= 0).sum())
        raise ValueError(f"Exposure must be > 0 for log-offset. Bad rows: {n_bad}")

    return df


def train_valid_split(df: pd.DataFrame, *, seed: int, valid_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * (1 - valid_frac))
    tr_idx, va_idx = idx[:cut], idx[cut:]
    return df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()


def fit_nb_glm(train_df: pd.DataFrame, formula: str):
    y, X = patsy.dmatrices(formula, train_df, return_type="dataframe")
    design_info = X.design_info  # <-- IMPORTANT

    offset = safe_log(train_df["Exposure"].to_numpy())
    fam = sm.families.NegativeBinomial(link=sm.families.links.log())
    model = sm.GLM(y, X, family=fam, offset=offset)
    res = model.fit()
    return res, design_info

def predict_counts_and_rates(res: Any, df: pd.DataFrame, formula: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict:
    - pred_count: expected claims for the given exposure period
    - pred_rate: annualized frequency = pred_count / Exposure
    """
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    offset = safe_log(df["Exposure"].to_numpy())
    pred_count = np.asarray(res.predict(X, offset=offset)).reshape(-1)
    pred_rate = pred_count / np.maximum(df["Exposure"].to_numpy(), 1e-12)
    return pred_count, pred_rate


def build_model_card_md(
    *,
    run_id: str,
    created_at_utc: str,
    input_path: str,
    input_sha256: str,
    rows: int,
    formula: str,
    metrics: Dict[str, Any],
    notes: Dict[str, Any],
) -> str:
    return f"""# Frequency Model Card — Motor TPL (freMTPL2)

## Run
- run_id: `{run_id}`
- created_at_utc: `{created_at_utc}`

## Data
- input: `{input_path}`
- input_sha256: `{input_sha256}`
- rows: `{rows}` (policy-level)
- target: `ClaimNb` (count)
- exposure handling: **offset(log(Exposure))**

## Modeling Approach
- Model: **Negative Binomial GLM**
- Link: **log**
- Formula:
  - `{formula}`

## Evaluation (validation split)
- Zero rate (val): `{metrics.get("zero_rate_val", "n/a")}`
- Obs annual rate (val): `{metrics.get("obs_rate_val", "n/a")}`
- Pred annual rate mean (val): `{metrics.get("pred_rate_mean_val", "n/a")}`
- Exposure-weighted rate error (abs): `{metrics.get("abs_rate_error_val", "n/a")}`
- MAE(count): `{metrics.get("mae_count", "n/a")}`
- RMSE(count): `{metrics.get("rmse_count", "n/a")}`
- MAE(log1p(count)): `{metrics.get("mae_log1p", "n/a")}`
- RMSE(log1p(count)): `{metrics.get("rmse_log1p", "n/a")}`
- AIC: `{metrics.get("glm_aic", "n/a")}`
- Deviance: `{metrics.get("glm_deviance", "n/a")}`
- Scale: `{metrics.get("glm_scale", "n/a")}`

## Known Limitations
- Random split (dataset not time-indexed).
- Categorical handling relies on stable levels; unseen categories must be handled upstream (feature layer policy).
- Zero-inflation not explicitly modeled in this baseline (NB often suffices in practice).

## Governance / Audit Notes
{json.dumps(notes, indent=2, ensure_ascii=False)}
"""


def train(
    *,
    data_path: Path,
    out_model_path: Path,
    out_metrics_path: Path,
    out_deciles_path: Path,
    out_model_card_path: Path,
    config: TrainConfig,
) -> Dict[str, Any]:
    ensure_dir(out_model_path.parent)
    ensure_dir(out_metrics_path.parent)
    ensure_dir(out_deciles_path.parent)
    ensure_dir(out_model_card_path.parent)

    run_id = f"freq_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}"
    created_at = utc_now_iso()

    df = load_data(data_path)
    input_hash = sha256_file(data_path)

    tr, va = train_valid_split(df, seed=config.seed, valid_frac=config.valid_frac)

    # Fit NB GLM with exposure offset
    res, _ = fit_nb_glm(tr, config.formula)

    # Predict
    pred_count_va, pred_rate_va = predict_counts_and_rates(res, va, config.formula)

    y_va = va["ClaimNb"].to_numpy()
    exp_va = va["Exposure"].to_numpy()

    # Core validation stats
    obs_rate_val = exposure_weighted_rate(y_va, exp_va)
    pred_rate_mean_val = float(np.mean(pred_rate_va))
    abs_rate_error_val = float(abs(obs_rate_val - pred_rate_mean_val))

    zero_rate_val = float(np.mean(y_va == 0))

    # Count-space + log1p(count) errors (counts are sparse; log1p is more stable)
    mae_count = mae(y_va, pred_count_va)
    rmse_count = rmse(y_va, pred_count_va)

    mae_log1p = mae(np.log1p(y_va), np.log1p(np.maximum(pred_count_va, 0)))
    rmse_log1p = rmse(np.log1p(y_va), np.log1p(np.maximum(pred_count_va, 0)))

    # Decile calibration on annual rate
    dec = decile_table_rate(
        y_claims=y_va,
        exposure=exp_va,
        rate_pred=pred_rate_va,
        n_bins=10,
    )
    dec.to_csv(out_deciles_path, index=False)

    metrics = {
        "run_id": run_id,
        "created_at_utc": created_at,
        "input_path": str(data_path),
        "input_sha256": input_hash,
        "rows": int(len(df)),
        "valid_frac": config.valid_frac,
        "zero_rate_val": zero_rate_val,
        "obs_rate_val": obs_rate_val,
        "pred_rate_mean_val": pred_rate_mean_val,
        "abs_rate_error_val": abs_rate_error_val,
        "mae_count": mae_count,
        "rmse_count": rmse_count,
        "mae_log1p": float(mae_log1p),
        "rmse_log1p": float(rmse_log1p),
        "glm_aic": float(res.aic) if hasattr(res, "aic") else None,
        "glm_deviance": float(res.deviance) if hasattr(res, "deviance") else None,
        "glm_scale": float(res.scale) if hasattr(res, "scale") else None,
    }
    out_metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    factor_cols = [c for c in ["Area", "VehBrand", "VehGas", "Region"] if c in tr.columns]
    factor_levels = {
        c: sorted(tr[c].dropna().astype(str).unique().tolist()) for c in factor_cols
    }

    artifact = FrequencyModelArtifact(
        fitted_result=res,
        formula=config.formula,
        config=asdict(config),
        factor_levels=factor_levels,
    )
    joblib.dump(artifact, out_model_path)

    notes = {
        "offset_policy": "Exposure used as offset(log(Exposure)) to model claim counts proportional to time-at-risk.",
        "data_contract": "Input is staged and schema-validated prior to training.",
    }
    md = build_model_card_md(
        run_id=run_id,
        created_at_utc=created_at,
        input_path=str(data_path),
        input_sha256=input_hash,
        rows=int(len(df)),
        formula=config.formula,
        metrics=metrics,
        notes=notes,
    )
    out_model_card_path.write_text(md, encoding="utf-8")

    return {
        "run_id": run_id,
        "model_path": str(out_model_path),
        "metrics_path": str(out_metrics_path),
        "deciles_path": str(out_deciles_path),
        "model_card_path": str(out_model_card_path),
        "rows": int(len(df)),
    }


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train Frequency model (NB GLM, exposure offset)")
    parser.add_argument(
        "--data",
        type=str,
        default=r"data\staging\freq_staged.parquet",
        help="Path to freq_staged.parquet",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=r"artifacts\models\frequency",
        help="Output directory for model artifact",
    )
    parser.add_argument(
        "--reportdir",
        type=str,
        default=r"artifacts\reports\frequency",
        help="Output directory for reports",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--valid-frac", type=float, default=0.2, help="Validation fraction (default: 0.2)")
    args = parser.parse_args()

    cfg = TrainConfig(seed=args.seed, valid_frac=args.valid_frac)

    outdir = Path(args.outdir)
    reportdir = Path(args.reportdir)

    summary = train(
        data_path=Path(args.data),
        out_model_path=outdir / "freq_glm_nb.joblib",
        out_metrics_path=reportdir / "freq_metrics.json",
        out_deciles_path=reportdir / "freq_deciles.csv",
        out_model_card_path=reportdir / "freq_model_card.md",
        config=cfg,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()