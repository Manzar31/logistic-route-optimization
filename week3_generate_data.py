"""
Generates a hypothetical logistics dataset for Week 3:
Advanced Data Analysis and Visualization in Logistics.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

n = 500
zones = ["North", "South", "East", "West", "Central"]
vehicle_types = ["Van", "Mini-Truck", "Bike", "Truck"]

zone = np.random.choice(zones, n, p=[0.22, 0.2, 0.2, 0.18, 0.2])
vehicle_type = np.random.choice(vehicle_types, n, p=[0.35, 0.25, 0.25, 0.15])

# Base distance depends loosely on zone (Central = shorter avg trips)
zone_distance_base = {"Central": 6, "North": 12, "South": 14, "East": 11, "West": 13}
distance_km = np.array([
    max(1, np.random.normal(zone_distance_base[z], 4)) for z in zone
])

# Shipment volume (kg)
shipment_volume_kg = np.random.gamma(shape=3, scale=15, size=n)

# Traffic index 0-10 (higher = more congestion), Central tends higher
traffic_base = {"Central": 6.5, "North": 5, "South": 5.5, "East": 4.5, "West": 5}
traffic_index = np.clip(
    [np.random.normal(traffic_base[z], 1.5) for z in zone], 0, 10
)

# Delivery time (minutes): depends on distance + traffic, with noise
delivery_time_min = (
    8
    + distance_km * 3.2
    + traffic_index * 4.5
    + np.random.normal(0, 8, n)
)
delivery_time_min = np.clip(delivery_time_min, 5, None)

# Transportation cost (INR): depends on distance, volume, vehicle type
vehicle_rate = {"Bike": 8, "Van": 14, "Mini-Truck": 18, "Truck": 24}
transportation_cost = (
    np.array([vehicle_rate[v] for v in vehicle_type]) * distance_km
    + shipment_volume_kg * 2.1
    + np.random.normal(0, 40, n)
)
transportation_cost = np.clip(transportation_cost, 50, None)

# On-time flag: less likely when traffic high / delivery time high
late_prob = 1 / (1 + np.exp(-(delivery_time_min - 75) / 15))
on_time = np.where(np.random.rand(n) > late_prob, 1, 0)

# Day of week
day_of_week = np.random.choice(
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], n
)

df = pd.DataFrame({
    "order_id": [f"ORD{1000+i}" for i in range(n)],
    "zone": zone,
    "vehicle_type": vehicle_type,
    "day_of_week": day_of_week,
    "distance_km": np.round(distance_km, 2),
    "shipment_volume_kg": np.round(shipment_volume_kg, 2),
    "traffic_index": np.round(traffic_index, 2),
    "delivery_time_min": np.round(delivery_time_min, 2),
    "transportation_cost_inr": np.round(transportation_cost, 2),
    "on_time": on_time,
})

df.to_csv("logistics_dataset.csv", index=False)
print(df.head())
print(df.shape)
print(df.describe())
