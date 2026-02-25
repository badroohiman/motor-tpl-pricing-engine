# src/data/schemas.py
"""
Dataset schemas ("data contracts") for freMTPL2freq and freMTPL2sev.

This module is intentionally declarative:
- No heavy logic
- Used by src/data/validate.py to enforce structure + business constraints

Design notes:
- We separate "required columns + dtypes" (schema) from "range/business rules" (constraints).
- Dtypes here are "target dtypes" after staging canonicalization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ColumnSpec:
    dtype: str
    nullable: bool = False
    # Basic constraints (optional; validation enforces if provided)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed: Optional[List[Any]] = None
    # If True, numeric values must be integer-like (e.g., 1.0 OK, 1.2 NOT OK)
    integer_like: bool = False


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    key_cols: List[str]
    columns: Dict[str, ColumnSpec]
    # Additional dataset-level notes (not enforced)
    notes: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# freMTPL2freq (policy table)
# -----------------------------
FREQ_SCHEMA = DatasetSchema(
    name="freMTPL2freq",
    key_cols=["IDpol"],
    columns={
        "IDpol": ColumnSpec(dtype="int64", nullable=False, integer_like=True),
        "ClaimNb": ColumnSpec(dtype="int64", nullable=False, min_value=0, integer_like=True),
        "Exposure": ColumnSpec(dtype="float64", nullable=False, min_value=0.0),
        # The rest are core risk features (types can vary; we enforce "string" post-staging for categoricals)
        "Area": ColumnSpec(dtype="string", nullable=False),
        "VehPower": ColumnSpec(dtype="int64", nullable=False, min_value=0, integer_like=True),
        "VehAge": ColumnSpec(dtype="float64", nullable=False, min_value=0.0),
        "DrivAge": ColumnSpec(dtype="float64", nullable=False, min_value=0.0),
        "BonusMalus": ColumnSpec(dtype="int64", nullable=False, integer_like=True),
        "VehBrand": ColumnSpec(dtype="string", nullable=False),
        "VehGas": ColumnSpec(dtype="string", nullable=False),
        "Density": ColumnSpec(dtype="int64", nullable=False, min_value=0, integer_like=True),
        "Region": ColumnSpec(dtype="string", nullable=False),
    },
    notes={
        "exposure_definition": "policy-year fraction typically in (0, 1]; governed in validate constraints",
        "categoricals_post_staging": ["Area", "VehBrand", "VehGas", "Region"],
    },
)


# -----------------------------
# freMTPL2sev (claims table)
# -----------------------------
SEV_SCHEMA = DatasetSchema(
    name="freMTPL2sev",
    key_cols=["IDpol"],  # not unique (many claims per policy)
    columns={
        "IDpol": ColumnSpec(dtype="int64", nullable=False, integer_like=True),
        "ClaimAmount": ColumnSpec(dtype="float64", nullable=False, min_value=0.0),
    },
    notes={
        "claimamount_definition": "claim-level severity (positive for modeling); strict positivity enforced in validate",
    },
)