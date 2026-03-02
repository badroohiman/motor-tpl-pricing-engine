# Frequency Model Card — Motor TPL (freMTPL2)

## Run
- run_id: `freq_2026-02-26T15-53-35Z`
- created_at_utc: `2026-02-26T15:53:35+00:00`

## Data
- input: `data\staging\freq_staged.parquet`
- input_sha256: `a88923b42b6c8e2994335d88cb48df0b555e6d2369b3192a57c3d8062fad016c`
- rows: `678013` (policy-level)
- target: `ClaimNb` (count)
- exposure handling: **offset(log(Exposure))**

## Modeling Approach
- Model: **Negative Binomial GLM**
- Link: **log**
- Formula:
  - `ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus + Density + C(Area) + C(VehBrand) + C(VehGas) + C(Region)`

## Evaluation (validation split)
- Zero rate (val): `0.9493963997846655`
- Obs annual rate (val): `0.10106632898668375`
- Pred annual rate mean (val): `0.10958567585633315`
- Exposure-weighted rate error (abs): `0.008519346869649405`
- MAE(count): `0.0997176906019333`
- RMSE(count): `0.23733795480670566`
- MAE(log1p(count)): `0.08061469728933048`
- RMSE(log1p(count)): `0.15967131315005176`
- AIC: `228503.7808079413`
- Deviance: `151359.2014106695`
- Scale: `1.0`

## Known Limitations
- Random split (dataset not time-indexed).
- Categorical handling relies on stable levels; unseen categories must be handled upstream (feature layer policy).
- Zero-inflation not explicitly modeled in this baseline (NB often suffices in practice).

## Governance / Audit Notes
{
  "offset_policy": "Exposure used as offset(log(Exposure)) to model claim counts proportional to time-at-risk.",
  "data_contract": "Input is staged and schema-validated prior to training."
}
