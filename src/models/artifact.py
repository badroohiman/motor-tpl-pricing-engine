# src/models/artifacts.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class SeverityModelArtifact:
    fitted_result: Any
    formula: str
    design_info: Any          # <-- add this
    cap_value: float
    config: Dict[str, Any]

@dataclass
class FrequencyModelArtifact:
    fitted_result: Any
    formula: str
    design_info: Any          # <-- add this
    config: Dict[str, Any]
