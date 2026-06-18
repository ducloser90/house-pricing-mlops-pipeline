from typing import Optional
import pandas as pd
from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
    ColumnDropping,
    OrdinalEncoding,
)
from zenml import step


@step(enable_cache=False)
def feature_engineering_step(
    df: pd.DataFrame,
    strategy: str = "log",
    features: Optional[list] = None,
    mapping: Optional[dict] = None,
) -> pd.DataFrame:

    if features is None:
        features = []

    if strategy == "log":
        engineer = FeatureEngineer(LogTransformation(features))
    elif strategy == "standard_scaling":
        engineer = FeatureEngineer(StandardScaling(features))
    elif strategy == "minmax_scaling":
        engineer = FeatureEngineer(MinMaxScaling(features))
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineer(OneHotEncoding(features))
    elif strategy == "column_dropping":
        engineer = FeatureEngineer(ColumnDropping(features))
    elif strategy == "ordinal_encoding":
        engineer = FeatureEngineer(OrdinalEncoding(mapping))
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

    transformed_df = engineer.apply_feature_engineering(df)
    return transformed_df