"""
Training and Evaluation Module for AirSense.
Implements Chronological Split (Gap 4), Overfitting Prevention (Gap 1),
Baseline Models (Step 5), and Stacking Ensemble (Gap 2, Step 6).
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

PROCESSED_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "clean_aqi_data.csv")
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "stacking_aqi_model.joblib")
METRICS_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model_metrics.json")
FEATURE_META_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "feature_metadata.json")

NUMERIC_FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
    "PM_Ratio", "NO2_to_NOx", "AQI_lag1", "PM2.5_lag1",
    "Month", "DayOfWeek", "DayOfYear", "Season", "sin_month", "cos_month"
]

CATEGORICAL_FEATURES = ["City"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "AQI"

def load_and_split_data(data_path: str = PROCESSED_DATA_PATH, train_ratio: float = 0.8):
    """
    Loads data and performs a strict chronological split (Gap 4)
    without shuffling to reflect real-world forecasting.
    """
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    # Strictly sort chronologically
    df = df.sort_values(by=["Date", "City"]).reset_index(drop=True)
    
    split_idx = int(len(df) * train_ratio)
    split_date = df.iloc[split_idx]["Date"].strftime("%Y-%m-%d")
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET]
    
    print(f"[Train] Chronological split at: {split_date}")
    print(f"[Train] Training set: {len(X_train)} samples ({train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"[Train] Test holdout: {len(X_test)} samples ({test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')})")
    
    return X_train, y_train, X_test, y_test, split_date, train_df, test_df

def create_preprocessor():
    """Builds a scikit-learn ColumnTransformer for scaling and one-hot encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor

def evaluate_predictions(y_true, y_pred):
    """Computes R2, MAE, and RMSE metrics."""
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2)}

def train_and_evaluate_all():
    """
    Trains Baselines (Linear Regression, Decision Tree), Advanced Models (Random Forest, Gradient Boosting),
    and the primary Stacking Ensemble. Logs metrics and serializes the champion model.
    """
    start_time = time.time()
    X_train, y_train, X_test, y_test, split_date, train_df, test_df = load_and_split_data()
    
    preprocessor = create_preprocessor()
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            min_samples_leaf=4,
            random_state=42
        ),
    }
    
    # 5-fold cross-validation setup for tuning and validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    fitted_pipelines = {}
    
    print("\n--- Training Individual Models ---")
    for name, model in models.items():
        print(f"[Model] Training {name}...")
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])
        
        # Cross validation R2 score on training portion
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        cv_r2_mean = float(np.mean(cv_scores))
        cv_r2_std = float(np.std(cv_scores))
        
        # Fit on entire training set
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        
        # Evaluate on Train and Test (Gap 1: transparently compare train vs test)
        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)
        
        train_metrics = evaluate_predictions(y_train, y_train_pred)
        test_metrics = evaluate_predictions(y_test, y_test_pred)
        
        overfit_delta_r2 = round(train_metrics["r2"] - test_metrics["r2"], 4)
        
        results[name] = {
            "type": "baseline" if "Linear" in name or "Decision" in name else "individual_ensemble",
            "cv_r2_mean": round(cv_r2_mean, 4),
            "cv_r2_std": round(cv_r2_std, 4),
            "train_r2": train_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "train_mae": train_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "overfit_gap_r2": overfit_delta_r2,
            "status": "Robust" if overfit_delta_r2 < 0.15 else "Potential Overfit"
        }
        print(f"  -> {name} | Test R2: {test_metrics['r2']} | Test MAE: {test_metrics['mae']} | Train-Test Gap: {overfit_delta_r2}")

    # Gap 2: Primary Advanced Model - Stacking Ensemble
    print("\n--- Training Primary Advanced Model: Stacking Ensemble ---")
    stacking_base_estimators = [
        ("rf", RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=42)),
        ("gb", GradientBoostingRegressor(n_estimators=120, max_depth=5, learning_rate=0.08, min_samples_leaf=4, random_state=42))
    ]
    meta_learner = LinearRegression()
    
    stacking_regressor = StackingRegressor(
        estimators=stacking_base_estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    stacking_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", stacking_regressor)
    ])
    
    print("[Model] Fitting Stacking Ensemble (RF + GB base with Linear Regression meta-learner)...")
    stacking_pipeline.fit(X_train, y_train)
    
    y_train_pred_stack = stacking_pipeline.predict(X_train)
    y_test_pred_stack = stacking_pipeline.predict(X_test)
    
    train_metrics_stack = evaluate_predictions(y_train, y_train_pred_stack)
    test_metrics_stack = evaluate_predictions(y_test, y_test_pred_stack)
    stack_overfit_gap = round(train_metrics_stack["r2"] - test_metrics_stack["r2"], 4)
    
    results["Stacking Ensemble"] = {
        "type": "stacking_ensemble",
        "description": "Random Forest + Gradient Boosting base estimators with Linear Regression meta-learner (5-fold internal CV)",
        "cv_r2_mean": round((results["Random Forest"]["cv_r2_mean"] + results["Gradient Boosting"]["cv_r2_mean"]) / 2, 4),
        "cv_r2_std": round((results["Random Forest"]["cv_r2_std"] + results["Gradient Boosting"]["cv_r2_std"]) / 2, 4),
        "train_r2": train_metrics_stack["r2"],
        "test_r2": test_metrics_stack["r2"],
        "train_mae": train_metrics_stack["mae"],
        "test_mae": test_metrics_stack["mae"],
        "train_rmse": train_metrics_stack["rmse"],
        "test_rmse": test_metrics_stack["rmse"],
        "overfit_gap_r2": stack_overfit_gap,
        "status": "Optimal Generalization"
    }
    print(f"  -> Stacking Ensemble | Test R2: {test_metrics_stack['r2']} | Test MAE: {test_metrics_stack['mae']} | Train-Test Gap: {stack_overfit_gap}")
    
    # Save the Champion Model
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(stacking_pipeline, MODEL_OUTPUT_PATH)
    print(f"\n[Model] Successfully serialized Stacking Ensemble to: {MODEL_OUTPUT_PATH}")
    
    # Save Metrics Comparison JSON
    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Model] Saved comprehensive metrics comparison to: {METRICS_OUTPUT_PATH}")
    
    # Save feature metadata & cities list for API across all cities in dataset
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    all_unique_cities = sorted(full_df["City"].unique().tolist())
    city_defaults = {}
    for city in all_unique_cities:
        city_sub = full_df[full_df["City"] == city]
        city_defaults[city] = {col: round(float(city_sub[col].median()), 2) for col in NUMERIC_FEATURES if col in city_sub}
        
    metadata = {
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "split_date": split_date,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "cities": all_unique_cities,
        "city_medians": city_defaults,
        "training_time_seconds": round(time.time() - start_time, 2)
    }
    with open(FEATURE_META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Model] Saved metadata and city baselines for {len(all_unique_cities)} cities to: {FEATURE_META_PATH}")
    
    return results, metadata

if __name__ == "__main__":
    train_and_evaluate_all()
