# src/models/artifacts.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class FrequencyModelArtifact:
    fitted_result: Any
    formula: str
    config: Dict[str, Any]
    # Categorical levels seen at training (for building same design matrix at inference)
    factor_levels: Dict[str, List[str]] = field(default_factory=dict)
    # Training values for spline vars (DrivAge, VehAge) so inference can match bs() knot placement
    spline_anchor: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeverityModelArtifact:
    fitted_result: Any
    formula: str
    cap_value: float
    config: Dict[str, Any]
    factor_levels: Dict[str, List[str]] = field(default_factory=dict)
    spline_anchor: Dict[str, Any] = field(default_factory=dict)