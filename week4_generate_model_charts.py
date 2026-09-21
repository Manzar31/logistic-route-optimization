import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11

df = pd.read_csv("logistics_dataset.csv")

numeric_features = ["distance_km", "shipment_volume_kg", "traffic_index"]
categorical_features = ["zone", "vehicle_type", "day_of_week"]
target = "delivery_time_min"

X = df[numeric_features + categorical_features]
y = df[target]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop="first"), categorical_features)],
    remainder="passthrough",
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Best model: Linear Regression
lr_pipe = Pipeline([("prep", preprocessor), ("model", LinearRegression())])
lr_pipe.fit(X_train, y_train)
lr_preds = lr_pipe.predict(X_test)

# ---------------------------------------------------------------
# Chart 1: Actual vs Predicted (Linear Regression)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.scatter(y_test, lr_preds, alpha=0.5, s=25, color="#2E5090")
lims = [min(y_test.min(), lr_preds.min()), max(y_test.max(), lr_preds.max())]
ax.plot(lims, lims, color="#D9534F", linestyle="--", label="Perfect prediction")
ax.set_xlabel("Actual Delivery Time (minutes)")
ax.set_ylabel("Predicted Delivery Time (minutes)")
ax.set_title("Actual vs Predicted Delivery Time (Linear Regression)")
ax.legend()
plt.tight_layout()
plt.savefig("chart_actual_vs_predicted.png")
plt.close()

# ---------------------------------------------------------------
# Chart 2: Model comparison bar chart (RMSE/MAE)
# ---------------------------------------------------------------
results_df = pd.read_csv("model_results.csv")
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(results_df))
width = 0.35
ax.bar(x - width/2, results_df["RMSE"], width, label="RMSE", color="#2E5090")
ax.bar(x + width/2, results_df["MAE"], width, label="MAE", color="#7FA8D9")
ax.set_xticks(x)
ax.set_xticklabels(results_df["model"])
ax.set_ylabel("Error (minutes)")
ax.set_title("Model Comparison: RMSE and MAE")
ax.legend()
plt.tight_layout()
plt.savefig("chart_model_comparison.png")
plt.close()

# ---------------------------------------------------------------
# Chart 3: Feature importance (Random Forest)
# ---------------------------------------------------------------
fi = pd.read_csv("feature_importances.csv", index_col=0).iloc[:, 0]
fi_top = fi.sort_values(ascending=True).tail(8)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(fi_top.index, fi_top.values, color="#2E5090")
ax.set_xlabel("Importance")
ax.set_title("Top Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("chart_feature_importance.png")
plt.close()

print("Charts generated.")
