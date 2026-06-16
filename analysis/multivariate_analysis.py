from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        pass


class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame):
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("/Users/m1/Desktop/ house-pricing-mlops-pipeline/extracted_data/AmesHousing.csv")
    mlt_analysis = SimpleMultivariateAnalysis()
    selected_features = df[['SalePrice', 'Gr Liv Area', 'Overall Qual', 'Total Bsmt SF', 'Year Built']]
    mlt_analysis.analyze(selected_features)