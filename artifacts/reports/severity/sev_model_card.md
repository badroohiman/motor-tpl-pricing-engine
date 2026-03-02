# Severity Model Card — Motor TPL (freMTPL2)

## Run
- run_id: `sev_2026-02-26T15-54-04Z`
- created_at_utc: `2026-02-26T15:54:04+00:00`

## Data
- input: `data\staging\sev_train.parquet`
- input_sha256: `a5716b1a8f8f7443292f2be3733ba41f65f5eec9b024a974d29670e37c50bc38`
- rows: `26444` (claim-level; policy features joined)
- target: `ClaimAmount` (strictly positive)

## Modeling Approach
- Model: **Gamma GLM**
- Link: **log**
- Formula:
  - `ClaimAmount_capped ~ VehPower + VehAge + DrivAge + BonusMalus + Density + Exposure + C(Area) + C(VehBrand) + C(VehGas) + C(Region)`

## Tail Handling Policy
- Winsorization (cap) applied to training target:
  - cap_quantile: `0.999`
  - cap_value: `152223.240820`
- Rationale:
  - Extreme outliers can dominate GLM coefficient estimates and degrade calibration for the bulk of claims.
  - Catastrophic / very large claims are typically handled via large-loss controls or separate modeling.

## Evaluation (validation split)
- log(MAE): `0.9304355965992276`
- log(RMSE): `1.3196998531386352`
- mean(actual): `1998.5159160521837`
- mean(pred): `1931.1631605591167`
- median(actual): `1172.0`
- median(pred): `1793.198731160378`

## Known Limitations
- Validation uses a random split (not time-based; dataset is not time-indexed).
- Tail policy is global; regional/segment-specific tail modelling is out of scope for baseline.
- Serving-time feature parity requires the same categorical normalization and levels; unseen categories must be handled upstream (feature layer policy).

## Governance / Audit Notes
{
  "tail_observation": "Extreme claims concentrated in specific regions (e.g., R24/R82); documented for monitoring.",
  "unmatched_claims_handling": "Claims without matching policy features excluded from training and stored separately (sev_unmatched_claims.parquet)."
}
