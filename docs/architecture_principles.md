# Motor TPL Pricing Engine

## Architecture & Governance Principles (v0.1)

---

## 1) System Objective

The system produces the following outputs for each input policy:

* `lambda_freq`: Expected claim frequency adjusted for Exposure
* `sev_mean`: Expected claim severity conditional on claim
* `expected_loss` (Pure Premium):

  ```
  expected_loss = lambda_freq × sev_mean
  ```
* `gross_premium`: Final premium after applying pricing rules (expenses, margin, caps, tiers)
* `warnings`: Structured data-quality and guardrail alerts
* `versions`: Model, feature, and pricing configuration versions for reproducibility

The system must be **production-oriented, auditable, and governance-ready** from inception.

---

## 2) Scope & Non-Scope

### In Scope

* Separate Frequency and Severity modeling
* Offset handling using `log(Exposure)`
* Pure Premium computation
* Config-driven pricing layer
* API + batch scoring capability
* Monitoring for drift and pricing stability

### Out of Scope (for initial version)

* Automated underwriting workflows
* Pricing optimization experiments
* Full regulatory automation framework

---

## 3) Core Architectural Decisions

---

### D1 — Separate Frequency and Severity Models

* Frequency model trained on all policies
* Severity model trained only on claim-level data
* Pure premium derived as:

  ```
  PurePremium(X) = E[ClaimNb | X] × E[ClaimAmount | Claim, X]
  ```

**Rationale:**
Avoids bias, aligns with actuarial standards, and supports pricing adequacy validation.

---

### D2 — Single Source of Truth for Feature Engineering

Feature transformations must be identical between training and inference.

Two modes:

* **Training mode** → fit encoders + transform
* **Runtime mode** → transform only using stored state

No re-fitting at inference time.

**Rationale:**
Prevents training–serving skew and ensures reproducibility.

---

### D3 — Mandatory Versioning & Auditability

Each generated quote must include:

* `model_version_freq`
* `model_version_sev`
* `feature_version`
* `pricing_config_version`
* `timestamp`
* `trace_id`

Versions are generated using artifact/config hashes.

**Rationale:**
Ensures reproducibility, traceability, and audit compliance.

---

### D4 — Pricing Rules Must Be Config-Driven

* All pricing parameters stored in `pricing_config.yaml`
* No hard-coded margins, caps, or loadings
* Every config change must be logged in `pricing_change_log.md`

**Rationale:**
Separates model risk from pricing policy and enables governance control.

---

### D5 — Structured Guardrails & Warnings

The system must explicitly handle risky inputs:

* `OUT_OF_RANGE`
* `UNKNOWN_CATEGORY`
* `LOW_EXPOSURE`
* `REFER_TO_UNDERWRITER`

Warnings must be structured and logged.

**Rationale:**
Prevents silent model misuse and supports safe deployment.

---

## 4) Data Quality & Validation Gates

### Schema Validation

* Column presence
* Data types
* Nullability

### Constraint Enforcement

* `Exposure > 0`
* `DrivAge ≥ 18`
* `VehAge ≥ 0`
* `BonusMalus` within allowed range
* `Density ≥ 0`

Validation failures above threshold must fail training.

**Rationale:**
Bad data directly translates into pricing risk.

---

## 5) Minimum Evaluation Standards

The following must be produced and archived:

### Frequency

* Poisson/NB deviance
* Calibration by decile (Observed vs Predicted)

### Severity

* Log-scale error metrics
* Decile calibration
* Tail stability metrics (e.g., P90 error)

### Pure Premium

* Decile-based pricing adequacy
* Observed vs Expected loss ratio proxy

**Rationale:**
Pricing defensibility requires calibration, not just accuracy.

---

## 6) CI/CD Strategy (Progressive Implementation)

### CI — From Early Development

* Linting
* Unit tests
* Coverage threshold
* Data validation on sample data

### CD — Once API Exists

* Docker image build
* Staging deployment
* Smoke tests
* Manual approval before production

**Rationale:**
Avoid early over-engineering while maintaining production readiness.

---

## 7) Governance Roles

* **Model Owner** → Responsible for performance, monitoring, and model cards
* **Pricing Owner** → Responsible for pricing configuration and rule changes

All pricing changes require traceable review and documentation.

---

## 8) Required Project Artifacts

* `data_validation_report.md`
* `freq_model_card.md`
* `sev_model_card.md`
* `pricing_change_log.md`
* `monitoring_report.md`

---

## 9) Definition of Done (v1)

The system is considered v1 production-ready when:

* Training–serving parity is tested
* Quotes include full version metadata
* Pricing config is externalized and versioned
* Decile adequacy report is generated
* CI checks run on every pull request
