"""
Batch rating and pricing adequacy report.

Scores a portfolio (e.g. freq_staged) with the pure premium engine, builds deciles
by predicted pure premium, and compares to observed loss (claim count and total amount).
Produces tables, charts, and pricing_adequacy_report.md as the main production proof.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .pure_premium import PurePremiumEngine


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_portfolio(path: Path) -> pd.DataFrame:
    """Load policy-level portfolio (e.g. freq_staged.parquet)."""
    df = pd.read_parquet(path)
    required = [
        "IDpol", "ClaimNb", "Exposure",
        "Area", "VehPower", "VehAge", "DrivAge", "BonusMalus",
        "VehBrand", "VehGas", "Density", "Region",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Portfolio missing columns: {missing}")
    return df


def load_observed_amounts(path: Path) -> pd.DataFrame:
    """
    Aggregate claim-level data (e.g. sev_train.parquet) to policy level:
    IDpol -> observed_claim_count, observed_total_amount.
    """
    sev = pd.read_parquet(path)
    if "IDpol" not in sev.columns or "ClaimAmount" not in sev.columns:
        raise ValueError("Observed data must have IDpol and ClaimAmount")
    agg = (
        sev.groupby("IDpol", as_index=False)
        .agg(
            observed_claim_count=("ClaimAmount", "count"),
            observed_total_amount=("ClaimAmount", "sum"),
        )
    )
    return agg


def merge_observed(portfolio: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Left-join observed counts and amounts onto portfolio. Fill missing with 0."""
    out = portfolio.merge(
        observed,
        on="IDpol",
        how="left",
    )
    out["observed_claim_count"] = out["observed_claim_count"].fillna(0).astype(int)
    out["observed_total_amount"] = out["observed_total_amount"].fillna(0.0)
    return out


def decile_table(
    df: pd.DataFrame,
    *,
    pred_col: str = "pred_pure_premium",
    exposure_col: str = "Exposure",
    observed_amount_col: Optional[str] = "observed_total_amount",
    observed_count_col: str = "ClaimNb",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Bin policies by decile of predicted pure premium; aggregate exposure,
    predicted pure, observed claims, observed amount (if present).
    """
    d = df.copy()
    try:
        d["decile"] = pd.qcut(d[pred_col], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        r = d[pred_col].rank(method="average")
        d["decile"] = pd.qcut(r, q=n_bins, labels=False, duplicates="drop")

    agg: Dict[str, Any] = {
        "policies": (pred_col, "count"),
        "exposure_sum": (exposure_col, "sum"),
        "pred_pure_sum": (pred_col, "sum"),
        "observed_claims_sum": (observed_count_col, "sum"),
    }
    if observed_amount_col and observed_amount_col in d.columns:
        agg["observed_amount_sum"] = (observed_amount_col, "sum")

    dec = d.groupby("decile", dropna=True).agg(**agg).reset_index()
    dec = dec.sort_values("decile")

    dec["obs_over_pred_ratio"] = np.nan
    if "observed_amount_sum" in dec.columns:
        dec["obs_over_pred_ratio"] = dec["observed_amount_sum"] / np.maximum(dec["pred_pure_sum"], 1e-12)
    return dec


def build_adequacy_report_md(
    *,
    run_at: str,
    portfolio_path: str,
    portfolio_rows: int,
    deciles: pd.DataFrame,
    has_observed_amounts: bool,
    model_paths: Dict[str, str],
) -> str:
    """Generate pricing_adequacy_report.md content."""
    lines = [
        "# Pricing Adequacy Report",
        "",
        "Production proof: batch rating of portfolio vs observed loss by decile of predicted pure premium.",
        "",
        f"- **Run at (UTC):** {run_at}",
        f"- **Portfolio:** `{portfolio_path}`",
        f"- **Policies scored:** {portfolio_rows:,}",
        f"- **Observed amounts:** " + ("included (from claim-level merge)" if has_observed_amounts else "not included (portfolio only)"),
        "",
        "## Model artifacts",
        "",
    ]
    for k, v in model_paths.items():
        lines.append(f"- **{k}:** `{v}`")
    lines.extend(["", "## Deciles by predicted pure premium", ""])

    # Markdown table from deciles
    cols = ["decile", "policies", "exposure_sum", "pred_pure_sum", "observed_claims_sum"]
    if "observed_amount_sum" in deciles.columns:
        cols.extend(["observed_amount_sum", "obs_over_pred_ratio"])
    sub = deciles[[c for c in cols if c in deciles.columns]].copy()
    sub = sub.round(4)
    header = "| " + " | ".join(sub.columns.astype(str)) + " |"
    sep = "| " + " | ".join("---" for _ in sub.columns) + " |"
    lines.append(header)
    lines.append(sep)
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in sub.columns) + " |")
    lines.append("")

    # Summary
    total_pred = deciles["pred_pure_sum"].sum()
    total_exp = deciles["exposure_sum"].sum()
    total_claims = deciles["observed_claims_sum"].sum()
    lines.extend([
        "## Portfolio totals",
        "",
        f"- **Total predicted pure premium:** {total_pred:,.2f}",
        f"- **Total exposure (policy-years):** {total_exp:,.2f}",
        f"- **Total observed claims:** {total_claims:,.0f}",
        "",
    ])
    if "observed_amount_sum" in deciles.columns:
        total_obs = deciles["observed_amount_sum"].sum()
        overall_ratio = total_obs / max(total_pred, 1e-12)
        lines.extend([
            f"- **Total observed loss (amount):** {total_obs:,.2f}",
            f"- **Overall observed / predicted ratio:** {overall_ratio:.4f}",
            "",
        ])
    lines.append("---")
    lines.append("*Generated by batch_rating (pricing validation).*")
    return "\n".join(lines)


def run(
    *,
    portfolio_path: Path,
    observed_path: Optional[Path] = None,
    freq_model_path: Path,
    sev_model_path: Path,
    sev_cap_path: Optional[Path] = None,
    out_dir: Path,
    n_deciles: int = 10,
    save_scored_portfolio: bool = True,
) -> Dict[str, Any]:
    """
    Run batch rating: score portfolio, build deciles, write tables/chart/report.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    portfolio = load_portfolio(portfolio_path)
    if observed_path is not None and observed_path.exists():
        observed = load_observed_amounts(observed_path)
        portfolio = merge_observed(portfolio, observed)
        has_observed = True
    else:
        portfolio["observed_claim_count"] = portfolio["ClaimNb"]
        portfolio["observed_total_amount"] = np.nan
        has_observed = False

    engine = PurePremiumEngine(
        freq_model_path=freq_model_path,
        sev_model_path=sev_model_path,
        sev_cap_path=sev_cap_path,
    )
    scored = engine.batch_quote_pure_premium(portfolio)
    # scored retains portfolio columns (observed_* from merge_observed if present)

    deciles = decile_table(
        scored,
        pred_col="pred_pure_premium",
        exposure_col="Exposure",
        observed_amount_col="observed_total_amount" if has_observed else None,
        observed_count_col="ClaimNb",
        n_bins=n_deciles,
    )

    run_at = _utc_now_iso()

    # Save outputs
    deciles.to_csv(out_dir / "adequacy_deciles.csv", index=False)
    if save_scored_portfolio:
        out_parquet = out_dir / "scored_portfolio.parquet"
        scored.to_parquet(out_parquet, index=False)

    model_paths = {
        "freq_model": str(freq_model_path),
        "sev_model": str(sev_model_path),
    }
    report_md = build_adequacy_report_md(
        run_at=run_at,
        portfolio_path=str(portfolio_path),
        portfolio_rows=len(scored),
        deciles=deciles,
        has_observed_amounts=has_observed,
        model_paths=model_paths,
    )
    report_path = out_dir / "pricing_adequacy_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    # Chart: observed vs predicted by decile (if we have observed amounts)
    chart_path = out_dir / "adequacy_deciles.png"
    if has_observed and "observed_amount_sum" in deciles.columns:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            x = deciles["decile"].astype(int)
            ax.bar(x - 0.2, deciles["pred_pure_sum"], width=0.35, label="Predicted pure", color="steelblue", alpha=0.9)
            ax.bar(x + 0.2, deciles["observed_amount_sum"], width=0.35, label="Observed loss", color="coral", alpha=0.9)
            ax.set_xlabel("Decile of predicted pure premium")
            ax.set_ylabel("Sum (currency)")
            ax.set_title("Pricing adequacy: predicted vs observed loss by decile")
            ax.legend()
            ax.set_xticks(x)
            fig.tight_layout()
            fig.savefig(chart_path, dpi=120)
            plt.close(fig)
        except Exception:
            chart_path = None
    else:
        chart_path = None

    return {
        "run_at": run_at,
        "portfolio_path": str(portfolio_path),
        "portfolio_rows": int(len(scored)),
        "deciles_path": str(out_dir / "adequacy_deciles.csv"),
        "report_path": str(report_path),
        "chart_path": str(chart_path) if chart_path else None,
        "scored_portfolio_path": str(out_dir / "scored_portfolio.parquet") if save_scored_portfolio else None,
    }


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Batch rating + pricing adequacy: score portfolio, deciles, report"
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        default=str(root / "data" / "staging" / "freq_staged.parquet"),
        help="Path to policy-level portfolio (e.g. freq_staged.parquet)",
    )
    parser.add_argument(
        "--observed",
        type=str,
        default=None,
        help="Path to claim-level data (e.g. sev_train.parquet) to aggregate observed amounts by policy",
    )
    parser.add_argument(
        "--freq-model",
        type=str,
        default=str(root / "artifacts" / "models" / "frequency" / "freq_glm_nb.joblib"),
        help="Frequency model path",
    )
    parser.add_argument(
        "--sev-model",
        type=str,
        default=str(root / "artifacts" / "models" / "severity" / "sev_glm_gamma.joblib"),
        help="Severity model path",
    )
    parser.add_argument(
        "--sev-cap",
        type=str,
        default=str(root / "artifacts" / "models" / "severity" / "sev_cap.json"),
        help="Severity cap JSON path",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(root / "artifacts" / "reports" / "pricing_adequacy"),
        help="Output directory for deciles CSV, report MD, chart, scored portfolio",
    )
    parser.add_argument(
        "--deciles",
        type=int,
        default=10,
        help="Number of decile bins",
    )
    parser.add_argument(
        "--no-scored-portfolio",
        action="store_true",
        help="Do not save scored_portfolio.parquet",
    )
    args = parser.parse_args()

    summary = run(
        portfolio_path=Path(args.portfolio),
        observed_path=Path(args.observed) if args.observed else None,
        freq_model_path=Path(args.freq_model),
        sev_model_path=Path(args.sev_model),
        sev_cap_path=Path(args.sev_cap) if Path(args.sev_cap).exists() else None,
        out_dir=Path(args.out_dir),
        n_deciles=args.deciles,
        save_scored_portfolio=not args.no_scored_portfolio,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
