import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11

df = pd.read_csv("logistics_dataset.csv")

# ---------------------------------------------------------------
# Chart 1: Distribution of delivery times (histogram + KDE)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
sns.histplot(df["delivery_time_min"], bins=30, kde=True, color="#2E5090", ax=ax)
ax.set_title("Distribution of Delivery Times")
ax.set_xlabel("Delivery Time (minutes)")
ax.set_ylabel("Number of Deliveries")
plt.tight_layout()
plt.savefig("chart1_delivery_time_distribution.png")
plt.close()

# ---------------------------------------------------------------
# Chart 2: Average transportation cost by zone (bar chart)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
zone_cost = df.groupby("zone")["transportation_cost_inr"].mean().sort_values(ascending=False)
sns.barplot(x=zone_cost.index, y=zone_cost.values, hue=zone_cost.index,
            palette="Blues_d", legend=False, ax=ax)
ax.set_title("Average Transportation Cost by Zone")
ax.set_xlabel("Zone")
ax.set_ylabel("Avg. Transportation Cost (INR)")
plt.tight_layout()
plt.savefig("chart2_avg_cost_by_zone.png")
plt.close()

# ---------------------------------------------------------------
# Chart 3: Distance vs Delivery Time (scatter with regression line)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
sns.regplot(data=df, x="distance_km", y="delivery_time_min",
            scatter_kws={"alpha": 0.4, "s": 20, "color": "#2E5090"},
            line_kws={"color": "#D9534F"}, ax=ax)
ax.set_title("Relationship Between Distance and Delivery Time")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Delivery Time (minutes)")
plt.tight_layout()
plt.savefig("chart3_distance_vs_time.png")
plt.close()

# ---------------------------------------------------------------
# Chart 4: Correlation heatmap of key numeric variables
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5))
numeric_cols = ["distance_km", "shipment_volume_kg", "traffic_index",
                 "delivery_time_min", "transportation_cost_inr"]
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Correlation Between Key Logistics Variables")
plt.tight_layout()
plt.savefig("chart4_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------------
# Chart 5: On-time delivery rate by vehicle type (bar chart)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
ontime_rate = df.groupby("vehicle_type")["on_time"].mean().sort_values(ascending=False) * 100
sns.barplot(x=ontime_rate.index, y=ontime_rate.values, hue=ontime_rate.index,
            palette="Greens_d", legend=False, ax=ax)
ax.set_title("On-Time Delivery Rate by Vehicle Type")
ax.set_xlabel("Vehicle Type")
ax.set_ylabel("On-Time Delivery Rate (%)")
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig("chart5_ontime_by_vehicle.png")
plt.close()

print("All charts generated.")

# Print summary stats used in the report narrative
print("\n--- Summary stats ---")
print("Mean delivery time:", df["delivery_time_min"].mean())
print("Median delivery time:", df["delivery_time_min"].median())
print("Std delivery time:", df["delivery_time_min"].std())
print("\nZone avg cost:\n", zone_cost)
print("\nCorrelation matrix:\n", corr)
print("\nOn-time rate by vehicle:\n", ontime_rate)
print("\nOverall on-time rate:", df["on_time"].mean() * 100)
