"""
Replicates the training-time feature engineering chain for a single
incoming inference request.

IMPORTANT: this must stay in lockstep with pipelines/training_pipeline.py.
If you change the feature engineering steps there (add/remove a strategy,
change the ordinal mapping, change which features get log-transformed),
update this module to match, or predictions will silently be wrong
because the trained Pipeline expects a specific column shape/encoding
that this function is responsible for reproducing at inference time.

Training-time order (see pipelines/training_pipeline.py):
  1. handle_missing_values_step(strategy="mean")  -> numeric NaNs filled
     with the TRAINING set's mean. At inference time we do not have
     access to that mean (and a single request has no "mean" of its
     own), so numeric fields are required in the API schema instead
     (HouseFeatures requires all numeric fields). This sidesteps the
     issue entirely: there should be no numeric NaNs to fill for a
     single, fully-specified request.
  2. feature_engineering_step(strategy="column_dropping", features=["Order","PID"])
     -> N/A here: Order/PID are not part of the API schema at all,
     since they are row identifiers with no meaning for a single
     inference request.
  3. feature_engineering_step(strategy="ordinal_encoding")
     -> uses OrdinalEncoding.DEFAULT_MAPPING from src/feature_engineering.py
  4. feature_engineering_step(strategy="log", features=["Gr Liv Area","SalePrice"])
     -> only "Gr Liv Area" applies at inference time (SalePrice is the
     target, not a feature, and is not present in the request).
"""

import numpy as np
import pandas as pd

from src.feature_engineering import OrdinalEncoding


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_transformed = df.copy()

    # Step 3 equivalent: ordinal encoding of quality/condition columns.
    # OrdinalEncoding.apply_transformation already handles NaN -> "None"
    # -> 0 internally, which is exactly what we want for fields like
    # "Bsmt Qual" or "Garage Qual" being None when the house has no
    # basement/garage.
    ordinal_encoder = OrdinalEncoding()  # uses DEFAULT_MAPPING
    df_transformed = ordinal_encoder.apply_transformation(df_transformed)

    # Step 4 equivalent: log1p transform on Gr Liv Area, matching
    # LogTransformation in src/feature_engineering.py. We don't import
    # LogTransformation here since it operates on an arbitrary feature
    # list and pulls in logging noise we don't need per-request; np.log1p
    # is the exact same operation it performs.
    if "Gr Liv Area" in df_transformed.columns:
        df_transformed["Gr Liv Area"] = np.log1p(df_transformed["Gr Liv Area"])

    return df_transformed