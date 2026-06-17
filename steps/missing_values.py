import pandas as pd
from src.missing_values_handling import DropMissingValuesStrategy,FillMissingValuesStrategy,MissingValueHandler
from zenml import step


@step(enable_cache=False)
def handle_missing_values_step(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    if strategy == "drop":
        handler = MissingValueHandler(DropMissingValuesStrategy(axis=0))
    elif strategy in ["mean", "median", "mode", "constant"]:
        handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy))
    else:
        raise ValueError(f"Unsupported missing value handling strategy: {strategy}")

    cleaned_df = handler.handle_missing_values(df)
    return cleaned_df
