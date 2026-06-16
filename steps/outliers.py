import logging
import pandas as pd
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection
from zenml import step


@step
def outlier_detection_step(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        raise ValueError("Input df must be a non-null pandas DataFrame.")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")

    logging.info(f"Starting outlier detection step with shape: {df.shape}")

    df_numeric = df.select_dtypes(include="number")

    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))

    df_cleaned = outlier_detector.handle_outliers(
        df_numeric,
        method="remove",
    )

    return df_cleaned