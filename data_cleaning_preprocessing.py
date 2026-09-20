"""
Data Collection, Cleaning, and Preprocessing for Logistics Route Analysis
Logistics Data Analyst Internship - Week 2 Task
Yuva Intern (NSDC)

This file contains the data cleaning and preprocessing pipeline used
to prepare raw delivery data for the route optimization project
started in Week 1. See the accompanying DOCX report for full details.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------
# 1. Load Raw Data and Initial Inspection
# ---------------------------------------------------------------------
def load_and_inspect(path="raw_delivery_data.csv"):
    df = pd.read_csv(path)
    print(df.info())
    print(df.isnull().sum())
    print(df.describe())
    return df


# ---------------------------------------------------------------------
# 2. Handle Missing Values
# ---------------------------------------------------------------------
def handle_missing_values(df):
    # Drop rows with missing GPS coordinates (cannot be safely imputed)
    df = df.dropna(subset=["lat", "lon"])

    # Median imputation for numeric fields prone to skew
    df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())
    df["traffic_index"] = df["traffic_index"].fillna(df["traffic_index"].median())
    return df


# ---------------------------------------------------------------------
# 3. Remove Duplicates and Standardize Formats
# ---------------------------------------------------------------------
def clean_and_standardize(df):
    # Remove duplicate delivery logs
    df = df.drop_duplicates(subset=["order_id", "pickup_time"])

    # Standardize timestamps to a consistent datetime format
    df["pickup_time"] = pd.to_datetime(df["pickup_time"], errors="coerce")
    df["delivery_time"] = pd.to_datetime(df["delivery_time"], errors="coerce")

    # Compute delivery duration in minutes
    df["duration_min"] = (
        df["delivery_time"] - df["pickup_time"]
    ).dt.total_seconds() / 60
    return df


# ---------------------------------------------------------------------
# 4. Outlier Detection (IQR Method)
# ---------------------------------------------------------------------
def remove_outliers(df):
    Q1 = df["duration_min"].quantile(0.25)
    Q3 = df["duration_min"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df["is_outlier"] = ~df["duration_min"].between(lower_bound, upper_bound)
    df_clean = df[~df["is_outlier"]].copy()
    return df_clean


# ---------------------------------------------------------------------
# 5. Normalization (Min-Max Scaling)
# ---------------------------------------------------------------------
def normalize_features(df_clean):
    scaler = MinMaxScaler()
    df_clean[["distance_km_norm", "traffic_index_norm"]] = scaler.fit_transform(
        df_clean[["distance_km", "traffic_index"]]
    )
    return df_clean


# ---------------------------------------------------------------------
# Main pipeline (illustrative)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    data = load_and_inspect()
    data = handle_missing_values(data)
    data = clean_and_standardize(data)
    data = remove_outliers(data)
    data = normalize_features(data)
    print("Preprocessing complete. Sample rows:")
    print(data.head())
