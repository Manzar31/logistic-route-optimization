import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("logistics_dataset.csv")

# ---------------------------------------------------------------
# Features / target
# ---------------------------------------------------------------
numeric_features = ["distance_km", "shipment_volume_kg", "traffic_index"]
categorical_features = ["zone", "vehicle_type", "day_of_week"]
target = "delivery_time_min"

X = df[numeric_features + categorical_features]
y = df[target]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
    ],
    remainder="passthrough",
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
}

results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")

    results.append({
        "model": name,
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 3),
        "CV_R2_mean": round(cv_scores.mean(), 3),
        "CV_R2_std": round(cv_scores.std(), 3),
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv("model_results.csv", index=False)

# ---------------------------------------------------------------
# Feature importance from the best model (Random Forest)
# ---------------------------------------------------------------
best_pipe = Pipeline([("prep", preprocessor), ("model", RandomForestRegressor(
    n_estimators=200, max_depth=8, random_state=42))])
best_pipe.fit(X_train, y_train)

feature_names = (
    list(best_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features))
    + numeric_features
)
importances = best_pipe.named_steps["model"].feature_importances_
fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("\nFeature importances (Random Forest):")
print(fi)
fi.to_csv("feature_importances.csv")
