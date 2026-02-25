# Motor TPL Pricing Engine (France Dataset)

Production-oriented implementation of a **Motor Third-Party Liability (TPL) pricing engine**
based on the freMTPL2freq and freMTPL2sev datasets.

The system is designed following insurance-grade principles:

- Frequency–Severity decomposition
- Pure Premium modeling (Expected Loss)
- Governance and auditability
- Deterministic data pipeline
- Training–serving parity (planned)

---

## Project Goal

Build a reproducible and auditable pipeline that:

1. Ingests raw motor insurance data
2. Applies governed staging policies
3. Models:
   - Claim frequency (Poisson / GLM)
   - Claim severity (Gamma / Lognormal)
4. Produces:
   - Expected Loss (Pure Premium)
   - Risk-based Gross Premium (with pricing rules)

---

## Data Pipeline (Current Status)

### Stage 1 — Ingest (CSV → Immutable Snapshots)

Reads raw CSV files and writes immutable Parquet snapshots.

Also generates an ingest manifest containing:
- Input CSV hashes
- Snapshot hashes
- Row/column counts
- Dtypes
- Basic quality checks

Run:

```powershell
python -m src.data.ingest `
  --freq data/raw/freMTPL2freq.csv `
  --sev  data/raw/freMTPL2sev.csv `
  --out  data/raw_snapshots `
  --manifest artifacts/reports/ingest_manifest.json