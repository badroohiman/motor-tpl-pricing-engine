# Testing and comparison guide

Instructions to **test** the pipeline (train + quote) and **compare** model performance before vs after feature engineering (e.g. splines, log1p(Density)).

---

## Prerequisites

- From the **repo root** (`motor-tpl-pricing-engine`).
- Activate the virtual environment (optional but recommended):
  - **PowerShell:** `.venv\Scripts\Activate.ps1`
  - **Cmd:** `.venv\Scripts\activate.bat`
- Ensure staged data exists: `data/staging/freq_staged.parquet`, `data/staging/sev_train.parquet` (run data pipeline first if not).

---

## Part A: Test the pipeline (train + quote)

### 1. Train frequency model

```powershell
python -m src.models.frequency.train --data data/staging/freq_staged.parquet
```

**Outputs:** `artifacts/models/frequency/freq_glm_nb.joblib`, `artifacts/reports/frequency/freq_metrics.json`, `freq_deciles.csv`, `freq_model_card.md`.

### 2. Train severity model

```powershell
python -m src.models.severity.train --data data/staging/sev_train.parquet --cap-quantile 0.999
```

**Outputs:** `artifacts/models/severity/sev_glm_gamma.joblib`, `sev_cap.json`, `artifacts/reports/severity/sev_metrics.json`, `sev_deciles.csv`, `sev_model_card.md`.

### 3. Quote pure premium (single policy)

```powershell
python -m src.pricing.pure_premium --policy-json configs/sample_policy.json
```

You should see JSON with `lambda_freq`, `rate_annual`, `sev_mean`, `expected_loss`, and `warnings`.

### 4. (Optional) Gross pricing from pure premium

```powershell
python -m src.pricing.pricing_engine --config configs/pricing/pricing_config.yaml --pure 353.26
```

Replace `353.26` with the `expected_loss` from step 3 if you want to test the full flow.

---

## Part B: Compare before vs after (e.g. splines)

Use this when you want to compare **two model runs** (e.g. linear age vs spline age, or Density vs log1p_Density).

### Option 1: You already have the “before” run

If you **saved** the reports from a previous run (e.g. in `artifacts/reports_baseline/`):

1. Train the **new** model (steps A.1 and A.2 above). This overwrites `artifacts/reports/frequency/` and `artifacts/reports/severity/`.
2. Run the comparison:

```powershell
python scripts/compare_model_runs.py --before artifacts/reports_baseline --after artifacts/reports
```

### Option 2: You do not have a baseline yet

Do this **once** before changing the model (e.g. before adding splines), so you keep a copy of the current run.

**Step 1 — Save current run as baseline (PowerShell):**

```powershell
New-Item -ItemType Directory -Force -Path artifacts/reports_baseline
Copy-Item -Recurse artifacts/reports/frequency artifacts/reports_baseline/
Copy-Item -Recurse artifacts/reports/severity artifacts/reports_baseline/
```

**Step 2 — Change the model** (e.g. add splines in the formula) and **retrain** (Part A.1 and A.2).

**Step 3 — Compare:**

```powershell
python scripts/compare_model_runs.py --before artifacts/reports_baseline --after artifacts/reports
```

### What the comparison script prints

- **Frequency:** `abs_rate_error_val`, `mae_log1p`, `rmse_log1p`, AIC, deviance; decile table with `obs_over_pred` and MAD from 1.0.
- **Severity:** `mae_log`, `rmse_log`, AIC, deviance (and related metrics).

**Interpretation:** Lower error metrics and lower AIC/deviance = better fit. `obs_over_pred` near 1.0 and lower MAD = better calibration by decile.

### Skip severity in the comparison

```powershell
python scripts/compare_model_runs.py --before artifacts/reports_baseline --after artifacts/reports --no-severity
```

---

## Quick reference

| Goal                    | Command |
|-------------------------|--------|
| Train frequency         | `python -m src.models.frequency.train --data data/staging/freq_staged.parquet` |
| Train severity          | `python -m src.models.severity.train --data data/staging/sev_train.parquet --cap-quantile 0.999` |
| Quote pure premium      | `python -m src.pricing.pure_premium --policy-json configs/sample_policy.json` |
| Compare two runs        | `python scripts/compare_model_runs.py --before <before_dir> --after <after_dir>` |
| Save baseline (PowerShell) | `Copy-Item -Recurse artifacts/reports/frequency artifacts/reports_baseline/` (and same for `severity`) |
