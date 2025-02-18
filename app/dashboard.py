import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd


def create_dashboard(flask_app):
    """Creates a 2x2 grid dashboard with fraud analysis charts."""

    app = dash.Dash(server=flask_app, routes_pathname_prefix="/dashboard/")

    # Load Data
    df = pd.read_csv("../data/processed/merged_fraud_df.csv")

    # Ensure necessary columns exist
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")

    # Fraud Trend Over Time
    fraud_trend_df = (
        df[df["class"] == 1]
        .groupby(df["purchase_time"].dt.date)
        .size()
        .reset_index(name="fraud_cases")
    )
    fraud_trend_fig = px.line(
        fraud_trend_df,
        x="purchase_time",
        y="fraud_cases",
        title="Fraud Trend Over Time",
    )

    # Fraud by Location (Fix: Ensure correct column names)
    fraud_by_location = df[df["class"] == 1]["country"].value_counts().reset_index()
    fraud_by_location.columns = ["country", "count"]
    fraud_location_fig = px.bar(
        fraud_by_location, x="country", y="count", title="Fraud Cases by Location"
    )

    # Fraud by Device
    fraud_by_device = df[df["class"] == 1]["device_id"].value_counts().reset_index()
    fraud_by_device.columns = ["device_id", "count"]
    fraud_device_fig = px.bar(
        fraud_by_device, x="device_id", y="count", title="Fraud Cases by Device"
    )

    # Feature Correlation (Fix: Select numeric columns only)
    correlation_matrix = df.select_dtypes(include=["number"]).corr()
    correlation_fig = px.imshow(correlation_matrix, title="Feature Correlation Heatmap")

    # Dashboard Layout (2x2 Grid)
    app.layout = html.Div(
        [
            html.H1("Fraud Detection Dashboard", style={"textAlign": "center"}),
            html.Div(
                [
                    html.Div(
                        [dcc.Graph(figure=fraud_trend_fig)], className="six columns"
                    ),
                    html.Div(
                        [dcc.Graph(figure=fraud_location_fig)], className="six columns"
                    ),
                ],
                className="row",
            ),
            html.Div(
                [
                    html.Div(
                        [dcc.Graph(figure=fraud_device_fig)], className="six columns"
                    ),
                    html.Div(
                        [dcc.Graph(figure=correlation_fig)], className="six columns"
                    ),
                ],
                className="row",
            ),
        ]
    )

    return app
