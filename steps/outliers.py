import logging
import pandas as pd
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection
from zenml import step


@step(enable_cache=False)
def outlier_detection_step(
    df: pd.DataFrame,
    column_name: str,
) -> pd.DataFrame:
    if df is None:
        raise ValueError("Input df must be a non-null pandas DataFrame.")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    logging.info(
        f"Starting outlier detection step for column '{column_name}' "
        f"with shape: {df.shape}"
    )

    outlier_detector = OutlierDetector(
        ZScoreOutlierDetection(threshold=3)
    )

    df_cleaned = outlier_detector.handle_outliers(
        df,
        columns=[column_name],
        method="remove",
    )

    logging.info(
        f"Outlier detection step completed. Shape after removal: {df_cleaned.shape}"
    )

    return df_cleaned


if __name__ == "__main__":
    pass