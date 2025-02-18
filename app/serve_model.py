import os
import sys
import logging
import pandas as pd
from flask import Flask, jsonify

# Add project path to sys.path
sys.path.append("/home/naod/Projects/tenx/W8/Financial-Fraud-Detection/")
from scripts.explore_data import FraudEDA
from dashboard import create_dashboard

# Configure logging
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)

# Initialize Flask app
app = Flask(__name__)

# Load dataset
DATA_PATH = "../data/processed/merged_fraud_df.csv"
df = pd.read_csv(DATA_PATH)
fraud_eda = FraudEDA(DATA_PATH)


def get_summary():
    """Returns summary statistics for fraud detection."""
    total_transactions = len(df)
    fraud_cases = df[df["class"] == 1].shape[0]
    fraud_percentage = (fraud_cases / total_transactions) * 100
    return {
        "total_transactions": total_transactions,
        "fraud_cases": fraud_cases,
        "fraud_percentage": round(fraud_percentage, 2),
    }


@app.route("/api/summary", methods=["GET"])
def summary():
    """API endpoint to get summary statistics of fraudulent transactions."""
    logging.info("API Call: /api/summary")
    return jsonify(get_summary())


@app.route("/api/fraud_trend", methods=["GET"])
def fraud_trend():
    """API endpoint to get fraud trends over time."""
    logging.info("API Call: /api/fraud_trend")
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    fraud_over_time = (
        df[df["class"] == 1]
        .groupby(df["purchase_time"].dt.date)
        .size()
        .reset_index(name="fraud_cases")
    )
    return jsonify(fraud_over_time.to_dict(orient="records"))


@app.route("/api/fraud_location", methods=["GET"])
def fraud_location():
    """API endpoint to get fraud cases by location."""
    logging.info("API Call: /api/fraud_location")
    fraud_by_location = df[df["class"] == 1]["country"].value_counts().reset_index()
    fraud_by_location.columns = ["country", "fraud_cases"]
    return jsonify(fraud_by_location.to_dict(orient="records"))


@app.route("/api/device_analysis", methods=["GET"])
def device_analysis():
    """API endpoint to get fraud cases by device ID."""
    logging.info("API Call: /api/device_analysis")
    fraud_by_device = df[df["class"] == 1]["device_id"].value_counts().reset_index()
    fraud_by_device.columns = ["device_id", "fraud_cases"]
    return jsonify(fraud_by_device.to_dict(orient="records"))


@app.route("/api/correlation_analysis", methods=["GET"])
def correlation_analysis():
    """API endpoint to perform correlation analysis on fraud-related features."""
    logging.info("API Call: /api/correlation_analysis")
    fraud_eda.correlation_analysis()
    return jsonify({"message": "Correlation analysis plotted."})


@app.route("/api/univariate_analysis", methods=["GET"])
def univariate_analysis():
    """API endpoint to perform univariate analysis on fraud-related features."""
    logging.info("API Call: /api/univariate_analysis")
    fraud_eda.univariate_analysis()
    return jsonify({"message": "Univariate analysis plotted."})


# Attach Dash dashboard
create_dashboard(app)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
