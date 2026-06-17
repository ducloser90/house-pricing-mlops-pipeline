import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


# Idea:
# This code implements a flexible feature engineering system using the Strategy design pattern.
# Different transformation techniques such as log transformation, standard scaling,
# min-max scaling, one-hot encoding, ordinal encoding, and column dropping are encapsulated
# as interchangeable strategies. The transformation logic is separated from the execution
# logic, allowing feature engineering methods to be switched or extended without modifying
# the core workflow.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(
            f"Applying log transformation to features: {self.features}"
        )

        df_transformed = df.copy()

        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])

        logging.info("Log transformation completed.")
        return df_transformed


class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(
            f"Applying standard scaling to features: {self.features}"
        )

        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(
            df[self.features]
        )

        logging.info("Standard scaling completed.")
        return df_transformed


class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(
            f"Applying Min-Max scaling to features: {self.features} "
            f"with range {self.scaler.feature_range}"
        )

        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(
            df[self.features]
        )

        logging.info("Min-Max scaling completed.")
        return df_transformed


class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.encoder = OneHotEncoder(
            sparse=False,
            drop="first",
        )

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(
            f"Applying one-hot encoding to features: {self.features}"
        )

        df_transformed = df.copy()

        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )

        df_transformed = (
            df_transformed
            .drop(columns=self.features)
            .reset_index(drop=True)
        )

        df_transformed = pd.concat(
            [df_transformed, encoded_df],
            axis=1,
        )

        logging.info("One-hot encoding completed.")
        return df_transformed


class ColumnDropping(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        existing = [f for f in self.features if f in df.columns]
        missing = [f for f in self.features if f not in df.columns]

        if missing:
            logging.warning(
                f"ColumnDropping: the following columns were not found "
                f"and will be skipped: {missing}"
            )

        logging.info(f"Dropping columns: {existing}")
        df_transformed = df.drop(columns=existing)
        logging.info("Column dropping completed.")
        return df_transformed


class OrdinalEncoding(FeatureEngineeringStrategy):
    """
    Maps categorical quality/condition-style columns to ordered integers
    based on an explicit mapping, instead of one-hot encoding them.

    Many Ames-housing-style columns (Exter Qual, Kitchen Qual, Bsmt Qual,
    Heating QC, Garage Qual, Fireplace Qu, Pool QC, ...) follow a natural
    order such as Ex > Gd > TA > Fa > Po, plus sometimes a 'None' category
    for "doesn't apply" (no basement, no fireplace, etc). One-hot encoding
    throws away that ordering, forcing a linear model to learn each level
    independently. Encoding ordinally preserves the monotonic relationship
    with price and typically improves linear model fit.

    mapping: a dict of {column_name: {category_value: ordinal_int}}.
    Any value in a column that isn't present in its mapping is left as NaN
    (so it can be caught by an imputer downstream) and a warning is logged.
    """

    # A ready-to-use default mapping for the common Ames Housing quality
    # columns. Pass your own `mapping` to override or extend this.
    DEFAULT_QUALITY_MAP = {
        "Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0,
    }

    DEFAULT_MAPPING = {
        "Exter Qual": DEFAULT_QUALITY_MAP,
        "Exter Cond": DEFAULT_QUALITY_MAP,
        "Bsmt Qual": DEFAULT_QUALITY_MAP,
        "Bsmt Cond": DEFAULT_QUALITY_MAP,
        "Heating QC": DEFAULT_QUALITY_MAP,
        "Kitchen Qual": DEFAULT_QUALITY_MAP,
        "Fireplace Qu": DEFAULT_QUALITY_MAP,
        "Garage Qual": DEFAULT_QUALITY_MAP,
        "Garage Cond": DEFAULT_QUALITY_MAP,
        "Pool QC": DEFAULT_QUALITY_MAP,
        "Bsmt Exposure": {
            "Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0,
        },
        "BsmtFin Type 1": {
            "GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0,
        },
        "BsmtFin Type 2": {
            "GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0,
        },
        "Garage Finish": {
            "Fin": 3, "RFn": 2, "Unf": 1, "None": 0,
        },
        "Functional": {
            "Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4,
            "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0,
        },
    }

    def __init__(self, mapping: dict = None):
        self.mapping = mapping if mapping is not None else self.DEFAULT_MAPPING

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        df_transformed = df.copy()

        for feature, value_map in self.mapping.items():
            if feature not in df_transformed.columns:
                logging.warning(
                    f"OrdinalEncoding: column '{feature}' not found, skipping."
                )
                continue

            logging.info(f"Ordinally encoding feature: {feature}")

            # Fill actual NaNs with the 'None' label first so they map to 0
            # (meaning "doesn't apply"), consistent with how this dataset
            # encodes missing basement/garage/pool/fireplace features.
            series = df_transformed[feature].fillna("None")
            mapped = series.map(value_map)

            unmapped_mask = mapped.isna() & series.notna()
            if unmapped_mask.any():
                unseen = series[unmapped_mask].unique().tolist()
                logging.warning(
                    f"OrdinalEncoding: feature '{feature}' has values not "
                    f"present in the mapping and will be left as NaN: {unseen}"
                )

            df_transformed[feature] = mapped

        logging.info("Ordinal encoding completed.")
        return df_transformed


class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy

    def apply_feature_engineering(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        logging.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df)


if __name__ == "__main__":
    pass