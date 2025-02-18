# Financial Fraud Detection Notebooks

This folder contains a series of Jupyter Notebooks that guide you through the process of detecting financial fraud using machine learning models. Each notebook focuses on a specific aspect of the data science workflow, from data preprocessing to model interpretation.

## Notebooks Overview

### `1.0-Data-Preprocessing.ipynb`

This notebook handles the initial data preprocessing steps, including:
- **Importing Libraries**: Loading necessary Python libraries for data manipulation and visualization.
- **Data Loading**: Reading raw data files into Pandas DataFrames.
- **Data Inspection**: Inspecting the datasets for general information, uniqueness, missing values, and duplication.
- **Data Preprocessing**: Handling duplicates, converting data types, handling outliers, and mapping IP addresses to countries.

### `2.0-Data-Exploration.ipynb`

This notebook focuses on exploratory data analysis (EDA) to understand the data better:
- **Import Libraries**: Loading necessary libraries for EDA.
- **Data Loading**: Loading the processed data from the previous notebook.
- **Exploratory Data Analysis**: Performing univariate and bivariate analysis to explore relationships between features and the target variable. This includes visualizations like histograms, box plots, and correlation matrices.

### `3.0-Feature-Engineering.ipynb`

This notebook is dedicated to feature engineering, which involves creating new features from the existing data:
- **Import Libraries**: Loading necessary libraries for feature engineering.
- **Data Loading**: Loading the processed data.
- **Feature Engineering**: Creating new features such as time-based features, transaction frequency, and velocity. Encoding categorical features and normalizing numerical features. Splitting the data into training and testing sets and saving them for model training.