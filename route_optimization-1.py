"""
Strategic Planning for Delivery Route Optimization
Logistics Data Analyst Internship - Week 1 Task
Yuva Intern (NSDC)

This file contains pseudocode/illustrative Python snippets showing the
proposed approach for reducing late deliveries through data-driven
route optimization. See the accompanying DOCX report for full details.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------
# 1. Data Loading and Cleaning
# ---------------------------------------------------------------------
def load_and_clean_data(path="delivery_data.csv"):
    df = pd.read_csv(path)

    # Drop rows with missing coordinates or timestamps
    df = df.dropna(subset=["pickup_time", "delivery_time", "lat", "lon"])

    # Convert to datetime and compute actual delivery duration
    df["pickup_time"] = pd.to_datetime(df["pickup_time"])
    df["delivery_time"] = pd.to_datetime(df["delivery_time"])
    df["duration_min"] = (
        df["delivery_time"] - df["pickup_time"]
    ).dt.total_seconds() / 60

    # Remove impossible/negative durations
    df = df[df["duration_min"] > 0]
    return df


# ---------------------------------------------------------------------
# 2. Clustering Delivery Points into Zones
# ---------------------------------------------------------------------
def cluster_into_zones(df, n_clusters=8):
    coords = df[["lat", "lon"]]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["zone"] = kmeans.fit_predict(coords)
    # Each 'zone' groups nearby delivery points for route assignment
    return df


# ---------------------------------------------------------------------
# 3. Predicting Delivery Time
# ---------------------------------------------------------------------
def train_delivery_time_model(df):
    features = df[["distance_km", "hour_of_day", "traffic_index"]]
    target = df["duration_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    predicted_time = model.predict(X_test)
    return model, predicted_time


# ---------------------------------------------------------------------
# 4. Route Sequencing Within a Zone (Nearest-Neighbor Heuristic)
# ---------------------------------------------------------------------
def distance(a, b):
    """Placeholder distance function (e.g., haversine or Euclidean)."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def nearest_neighbor_route(start_point, stops):
    route = [start_point]
    remaining = stops.copy()
    current = start_point
    while remaining:
        next_stop = min(remaining, key=lambda s: distance(current, s))
        route.append(next_stop)
        remaining.remove(next_stop)
        current = next_stop
    return route


# ---------------------------------------------------------------------
# Main pipeline (illustrative)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    data = load_and_clean_data()
    data = cluster_into_zones(data)
    model, predictions = train_delivery_time_model(data)
    print("Pipeline complete. Sample predictions:", predictions[:5])
