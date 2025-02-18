import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class FraudEDA:
    def __init__(self, file_path):
        """
        Initializes the FraudEDA class by loading the dataset.
        :param file_path: Path to the CSV file containing the dataset.
        """
        self.df = pd.read_csv(file_path, parse_dates=["signup_time", "purchase_time"])
        print("Dataset loaded successfully. Shape:", self.df.shape)

    def correlation_analysis(self):
        """
        Computes and visualizes the correlation matrix to identify relationships between numerical variables.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number])
        corr_matrix = numeric_cols.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

    def univariate_analysis(self):
        """
        Performs univariate analysis on relevant numerical and categorical features.
        """
        # Numerical features
        num_cols = ["purchase_value", "age"]
        fig, axes = plt.subplots(1, len(num_cols), figsize=(12, 4))
        for i, col in enumerate(num_cols):
            sns.histplot(self.df[col], bins=30, kde=True, ax=axes[i])
            axes[i].set_title(f"Distribution of {col}")
        plt.show()

        # Categorical features
        cat_cols = ["source", "browser", "sex"]
        for col in cat_cols:
            plt.figure(figsize=(6, 3.5))
            sns.countplot(y=self.df[col], order=self.df[col].value_counts().index)
            plt.title(f"Distribution of {col}")
            plt.show()

    def purchase_value_vs_fraud(self):
        """
        Analyzes purchase value distribution across fraud classes.
        """
        print(self.df.groupby("class")["purchase_value"].describe())
        plt.figure(figsize=(6, 3.5))
        sns.boxplot(x="class", y="purchase_value", data=self.df)
        plt.title("Purchase Value vs Fraud")
        plt.show()

    def age_vs_fraud(self):
        """
        Analyzes age distribution across fraud classes.
        """
        print(self.df.groupby("class")["age"].describe())
        plt.figure(figsize=(6, 3.5))
        sns.boxplot(x="class", y="age", data=self.df)
        plt.title("Age vs Fraud")
        plt.show()

    def source_vs_fraud(self):
        """
        Analyzes source distribution across fraud classes.
        """
        print(self.df.groupby("source")["class"].value_counts(normalize=True))
        plt.figure(figsize=(6, 3.5))
        sns.countplot(x="source", hue="class", data=self.df)
        plt.title("Source vs Fraud")
        plt.show()

    def browser_vs_fraud(self):
        """
        Analyzes browser distribution across fraud classes.
        """
        print(self.df.groupby("browser")["class"].value_counts(normalize=True))
        plt.figure(figsize=(6, 3.5))
        sns.countplot(x="browser", hue="class", data=self.df)
        plt.title("Browser vs Fraud")
        plt.show()

    def country_vs_fraud(self):
        """
        Analyzes country distribution across fraud classes.
        """
        top_countries = self.df["country"].value_counts().nlargest(10).index
        country_data = self.df[self.df["country"].isin(top_countries)]
        print(country_data.groupby("country")["class"].value_counts(normalize=True))
        plt.figure(figsize=(10, 5))
        sns.countplot(y="country", hue="class", data=country_data, order=top_countries)
        plt.title("Top 10 Countries vs Fraud")
        plt.show()
