# Financial Fraud Detection API

A Flask-based API for analyzing financial fraud data includes several endpoints for retrieving fraud-related statistics and visualizations.

## Endpoints

- `GET /api/summary`: Returns summary statistics of fraudulent transactions.
- `GET /api/fraud_trend`: Returns fraud trends over time.
- `GET /api/fraud_location`: Returns fraud cases by location.
- `GET /api/device_analysis`: Returns fraud cases by device ID.
- `GET /api/correlation_analysis`: Performs correlation analysis on fraud-related features.
- `GET /api/univariate_analysis`: Performs univariate analysis on fraud-related features.

## How to Run

1. **Clone the repository**:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the application**:
    ```sh
    python app/serve_model.py
    ```

4. **Access the API**:
    Open your browser and navigate to `http://127.0.0.1:5000`.
