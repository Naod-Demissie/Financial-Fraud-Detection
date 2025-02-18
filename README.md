# Financial Fraud Detection

This project enhances fraud detection for e-commerce and banking transactions at Adey Innovations Inc. using machine learning. It involves data preprocessing, feature engineering, model training, and explainability with SHAP and LIME. The models will be deployed via a Flask API, containerized with Docker, and integrated with a Dash dashboard for real-time monitoring, improving security and fraud prevention.

## Project Structure


```
├── app
│   ├── __init__.py
│   ├── dashboard.py
│   └── serve_model.py
├── assets
│   └── images
│       ├── image 1.png
│       └── image 2.png
├── data
│   ├── processed
│   └── raw
├── logs
│   ├── app.log
│   └── mlruns
├── notebooks
│   ├── 1.0-data-preprocessing.ipynb
│   ├── 2.0-data-exploration.ipynb
│   ├── 3.0-feature-engineering.ipynb
│   ├── 4.0-model-training.ipynb
│   ├── 5.0-model-Interpretation.ipynb
│   └── README.md
├── scripts
│   ├── __init__.py
│   ├── explore_data.py
│   ├── feature_engineer.py
│   ├── interprate_models.py
│   ├── preprocess_data.py
│   ├── README.md
│   └── train_models.py
├── src
│   ├── __init__.py
│   └── README.md
├── tests
│  └── __init__.py
├── checkpoints
│   ├── best_CNN_creditcard_data.h5
│   ├── best_CNN_fraud_data.h5
│   ├── best_Decision_Tree_creditcard_data.joblib
│   ├── best_Decision_Tree_fraud_data.joblib
│   ├── best_Gradient_Boosting_creditcard_data.joblib
│   ├── best_Gradient_Boosting_fraud_data.joblib
│   ├── best_Logistic_Regression_creditcard_data.joblib
│   ├── best_Logistic_Regression_fraud_data.joblib
│   ├── best_LSTM_creditcard_data.h5
│   ├── best_LSTM_fraud_data.h5
│   ├── best_Random_Forest_creditcard_data.joblib
│   ├── best_Random_Forest_fraud_data.joblib
│   ├── best_RNN_creditcard_data.h5
│   └── best_RNN_fraud_data.h5
├── README.md
├── requirements.txt
├── Dockerfile
```


## Endpoints

- `GET /api/summary`: Returns summary statistics of fraudulent transactions.
- `GET /api/fraud_trend`: Returns fraud trends over time.
- `GET /api/fraud_location`: Returns fraud cases by location.
- `GET /api/device_analysis`: Returns fraud cases by device ID.
- `GET /api/correlation_analysis`: Performs correlation analysis on fraud-related features.
- `GET /api/univariate_analysis`: Performs univariate analysis on fraud-related features.
- `GET /dashboard/`: Displays a Dash dashboard with fraud analysis charts.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Naod-Demissie/Financial-Fraud-Detection.git
   cd Financial-Fraud-Detection
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv/Scripts/activate`
   pip install -r requirements.txt
   ```

3. **Run the application**:
    ```sh
    python app/serve_model.py
    ```

4. **Access the API**:
    Open your browser and navigate to `http://127.0.0.1:5000`.


## Dashboard

The Dash dashboard is accessible at `http://127.0.0.1:5000/dashboard/` and provides visualizations for fraud trends and fraud cases by location.
