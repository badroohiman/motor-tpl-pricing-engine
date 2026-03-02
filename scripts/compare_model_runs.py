"""
Compare frequency (and optionally severity) model runs: before vs after feature engineering.

Usage:
  1. Save baseline outputs before changing the model (e.g. copy artifacts/reports to artifacts/reports_baseline).
  2. Train the new model (e.g. with log1p_Density); it will overwrite artifacts/reports/frequency/.
  3. Run:
     python scripts/compare_model_runs.py --before artifacts/reports_baseline --after artifacts/reports
     (Or use --before-dir and --after-dir for the report root containing frequency/ and optionally severity/.)

Output: Prints a comparison of metrics and decile calibration (obs_over_pred by decile).
Lower abs_rate_error_val, mae_log1p, AIC/deviance = better. obs_over_pred closer to 1.0 = better calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_deciles(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def compare_frequency(before_dir: Path, after_dir: Path) -> None:
    before_metrics = load_metrics(before_dir / "frequency" / "freq_metrics.json")
    after_metrics = load_metrics(after_dir / "frequency" / "freq_metrics.json")
    before_dec = load_deciles(before_dir / "frequency" / "freq_deciles.csv")
    after_dec = load_deciles(after_dir / "frequency" / "freq_deciles.csv")

    print("=" * 60)
    print("FREQUENCY MODEL — Before vs After")
    print("=" * 60)

    if before_metrics and after_metrics:
        keys = [
            "abs_rate_error_val",
            "mae_log1p",
            "rmse_log1p",
            "glm_aic",
            "glm_deviance",
            "obs_rate_val",
            "pred_rate_mean_val",
        ]
        print("\nMetrics (lower is better for error/AIC/deviance):")
        print(f"{'metric':<28} {'before':>14} {'after':>14} {'diff (after-before)':>18}")
        print("-" * 74)
        for k in keys:
            b = before_metrics.get(k)
            a = after_metrics.get(k)
            if b is not None and a is not None:
                try:
                    diff = float(a) - float(b)
                    print(f"{k:<28} {float(b):>14.6g} {float(a):>14.6g} {diff:>+18.6g}")
                except (TypeError, ValueError):
                    print(f"{k:<28} {str(b):>14} {str(a):>14}")
        print()

    if before_dec is not None and after_dec is not None and "obs_over_pred" in before_dec.columns and "obs_over_pred" in after_dec.columns:
        print("Decile calibration (obs_over_pred; target 1.0):")
        print(f"{'decile':<8} {'before':>10} {'after':>10} {'change':>10}")
        print("-" * 40)
        for i in range(min(len(before_dec), len(after_dec))):
            b = float(before_dec["obs_over_pred"].iloc[i])
            a = float(after_dec["obs_over_pred"].iloc[i])
            print(f"{i:<8} {b:>10.4f} {a:>10.4f} {a - b:>+10.4f}")
        # Summary: mean absolute deviation from 1.0 (lower = better calibration)
        mad_before = (before_dec["obs_over_pred"] - 1.0).abs().mean()
        mad_after = (after_dec["obs_over_pred"] - 1.0).abs().mean()
        print("-" * 40)
        print(f"{'MAD from 1.0':<8} {mad_before:>10.4f} {mad_after:>10.4f} {mad_after - mad_before:>+10.4f}")
    print()


def compare_severity(before_dir: Path, after_dir: Path) -> None:
    before_metrics = load_metrics(before_dir / "severity" / "sev_metrics.json")
    after_metrics = load_metrics(after_dir / "severity" / "sev_metrics.json")
    before_dec = load_deciles(before_dir / "severity" / "sev_deciles.csv")
    after_dec = load_deciles(after_dir / "severity" / "sev_deciles.csv")

    print("=" * 60)
    print("SEVERITY MODEL — Before vs After")
    print("=" * 60)

    if before_metrics and after_metrics:
        keys = ["mae_log", "rmse_log", "glm_aic", "glm_deviance", "actual_mean", "pred_mean"]
        print("\nMetrics (lower is better for error/AIC/deviance):")
        print(f"{'metric':<16} {'before':>14} {'after':>14} {'diff':>14}")
        print("-" * 56)
        for k in keys:
            b = before_metrics.get(k)
            a = after_metrics.get(k)
            if b is not None and a is not None:
                try:
                    diff = float(a) - float(b)
                    print(f"{k:<16} {float(b):>14.6g} {float(a):>14.6g} {diff:>+14.6g}")
                except (TypeError, ValueError):
                    print(f"{k:<16} {str(b):>14} {str(a):>14}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model runs (before vs after feature engineering)")
    parser.add_argument("--before", "--before-dir", dest="before_dir", type=str, required=True, help="Path to 'before' report directory (e.g. artifacts/reports_baseline)")
    parser.add_argument("--after", "--after-dir", dest="after_dir", type=str, required=True, help="Path to 'after' report directory (e.g. artifacts/reports)")
    parser.add_argument("--no-severity", action="store_true", help="Skip severity comparison")
    args = parser.parse_args()

    before = Path(args.before_dir)
    after = Path(args.after_dir)

    if not before.exists():
        print(f"Before dir not found: {before}")
        return
    if not after.exists():
        print(f"After dir not found: {after}")
        return

    compare_frequency(before, after)
    if not args.no_severity:
        compare_severity(before, after)


if __name__ == "__main__":
    main()
