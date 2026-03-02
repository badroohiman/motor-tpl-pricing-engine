from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
from typing import Any, Dict, List


class GLMExplainer:
    """
    Exact feature contribution explainer for log-link GLM models (statsmodels).

    Works on the linear predictor:
        eta = X beta
        mu  = exp(eta)

    Returns per-term contributions on:
    - log scale (additive on eta)
    - multiplicative scale (exp(log_contribution))
    """

    def __init__(self, fitted_result: Any, formula: str, top_n: int = 5):
        self.res = fitted_result
        self.params = fitted_result.params  # pandas Series indexed by term name
        self.top_n = top_n

        # Keep RHS only for design matrix construction
        if "~" in formula:
            self.rhs_formula = formula.split("~", 1)[1].strip()
        else:
            self.rhs_formula = formula.strip()

    def explain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain a single policy row given as a one-row DataFrame.
        """
        if len(df) != 1:
            raise ValueError("GLMExplainer.explain expects a single-row DataFrame")

        # Build RHS design matrix
        X = patsy.dmatrix(self.rhs_formula, df, return_type="dataframe")
        row = X.iloc[0]

        # Align to full parameter index; missing columns imply 0 contribution for this policy
        row_full = row.reindex(self.params.index, fill_value=0.0)

        linear_terms = row_full * self.params  # Series: term -> log contribution

        intercept = float(linear_terms.get("Intercept", 0.0))
        feature_terms = linear_terms.drop(index="Intercept", errors="ignore")

        total_linear = float(linear_terms.sum())
        predicted = float(np.exp(total_linear))

        # Sort by absolute impact on log scale
        order = feature_terms.abs().sort_values(ascending=False)
        top_idx = order.head(self.top_n).index

        top_features: List[Dict[str, Any]] = []
        for term in top_idx:
            val = float(feature_terms[term])
            top_features.append(
                {
                    "term": term,
                    "log_contribution": val,
                    "multiplicative_effect": float(np.exp(val)),
                }
            )

        return {
            "linear_predictor": total_linear,
            "predicted_value": predicted,
            "intercept": intercept,
            "terms": {k: float(v) for k, v in feature_terms.items()},
            "top_features": top_features,
        }

