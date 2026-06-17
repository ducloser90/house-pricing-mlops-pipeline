from typing import Tuple

import pandas as pd
from src.data_splitting import DataSplitter, SimpleTrainTestSplitStrategy
from zenml import step


@step(enable_cache=False)
def data_splitter_step(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    X_train, X_test, y_train, y_test = splitter.split(df, target_column)
    return X_train, X_test, y_train, y_test
