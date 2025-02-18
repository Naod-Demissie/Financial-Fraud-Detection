# Financial Fraud Detection Modules

This project contains several Python modules that are essential for the financial fraud detection pipeline. Each module is responsible for a specific part of the workflow, from data preprocessing to model training and interpretation.

## Modules Overview

## `preprocess_data.py`

This module handles the initial data preprocessing steps, including:
- **Missing Values Proportions**: Calculates the proportion of missing values in each column of the DataFrame.
- **Handle Outliers**: Detects and handles outliers in specified columns by either replacing them with boundaries or the mean. Optionally, it can plot box plots to visualize the outliers.

## `feature_engineer.py`

This module is responsible for feature engineering, which involves creating new features from the existing data:
- **`FraudFeatureEngineer` Class**: 
  - **Initialization**: Initializes the class with paths to the fraud and credit card data, a random seed, a MinMaxScaler, and a dictionary for LabelEncoders.
  - `load_data`: Loads data from a specified file path.
  - `process_fraud_data`: Processes the fraud data by converting timestamps, extracting time-based features, calculating transaction frequency and velocity, encoding categorical features, normalizing numerical features, and splitting the data into training, validation, and test sets.
  - `process_creditcard_data`: Processes the credit card data by normalizing features and splitting the data into training, validation, and test sets.
  - `split_and_save`: Splits the data into training, validation, and test sets and saves them as .npy files.


## `explore_data.py`

This module is dedicated to exploratory data analysis (EDA) to understand the data better:
- **`FraudEDA` Class**:
  - **Initialization**: Initializes the class by loading the dataset.
  - `correlation_analysis`: Computes and visualizes the correlation matrix to identify relationships between numerical variables.
  - `univariate_analysis`: Performs univariate analysis on relevant numerical and categorical features, including visualizations like histograms and count plots.
  - `purchase_value_vs_fraud`: Analyzes the distribution of purchase values across fraud classes and visualizes it using box plots.
  - `age_vs_fraud`: Analyzes the distribution of age across fraud classes and visualizes it using box plots.
  - `source_vs_fraud`: Analyzes the distribution of sources across fraud classes and visualizes it using count plots.
  - `browser_vs_fraud`: Analyzes the distribution of browsers across fraud classes and visualizes it using count plots.
  - `country_vs_fraud`: Analyzes the distribution of countries across fraud classes and visualizes it using count plots for the top 10 countries.
