from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd


# -----------------------------
# Local feature spec (kept simple to avoid cross-module import issues)
# -----------------------------


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    build: Callable[[pd.DataFrame], pd.DataFrame]
    formula: str


# -----------------------------
# Frequency feature sets
# -----------------------------

# 1) Baseline feature set: no engineered transformations (raw numeric features)
FREQ_BASE_FORMULA = (
    "ClaimNb ~ "
    "VehPower + DrivAge + VehAge + BonusMalus + Density + "
    "C(Area) + C(VehBrand) + C(VehGas) + C(Region)"
)


def build_freq_base(df: pd.DataFrame) -> pd.DataFrame:
    # Keep as close as possible to staged schema (no extra columns)
    return df


# 2) Engineered feature set: mirrors the current production-style setup
FREQ_ENGINEERED_FORMULA = (
    "ClaimNb ~ "
    "VehPower + bs(DrivAge, df=5) + bs(VehAge, df=5) + BonusMalus + log1p_Density + "
    "C(Area) + C(VehBrand) + C(VehGas) + C(Region)"
)


def _add_log1p_density(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log1p_Density"] = np.log1p(
        np.maximum(pd.to_numeric(out["Density"], errors="coerce").fillna(0), 0)
    )
    return out


def build_freq_engineered(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    out = _add_log1p_density(out)
    return out


FREQ_FEATURE_SETS: Dict[str, FeatureSpec] = {
    "base": FeatureSpec(
        name="base",
        build=build_freq_base,
        formula=FREQ_BASE_FORMULA,
    ),
    "engineered": FeatureSpec(
        name="engineered",
        build=build_freq_engineered,
        formula=FREQ_ENGINEERED_FORMULA,
    ),
}

