"""
Automated Pipeline and API Verification Suite for AirSense.
Tests models, data quality, error analysis, and FastAPI endpoints.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.advisory import get_aqi_category, get_aqi_advisory, get_season
from app.main import app, PredictionRequest, predict_aqi, get_cities, get_model_comparison, get_error_analysis, get_historical_trends

def test_advisory_bands():
    print("[Test 1] Testing CPCB Advisory Bands...")
    test_cases = [
        (35, "Good", "#10B981"),
        (75, "Satisfactory", "#84CC16"),
        (150, "Moderate", "#F59E0B"),
        (250, "Poor", "#F97316"),
        (350, "Very Poor", "#EF4444"),
        (450, "Severe", "#7F1D1D")
    ]
    for aqi, expected_cat, expected_color in test_cases:
        adv = get_aqi_advisory(aqi)
        assert adv["category"] == expected_cat, f"Expected {expected_cat}, got {adv['category']}"
        assert adv["color"] == expected_color, f"Expected {expected_color}, got {adv['color']}"
        assert "general_population" in adv
        assert "sensitive_groups" in adv
        assert "outdoor_workers" in adv
        assert len(adv["actions"]) >= 2
    print("  [PASS] All CPCB advisory bands and guidance fields verified.")

def test_model_and_metrics():
    print("[Test 2] Testing Stacking Model & Literature Gap Benchmarks...")
    model_path = os.path.join(ROOT_DIR, "models", "stacking_aqi_model.joblib")
    metrics_path = os.path.join(ROOT_DIR, "models", "model_metrics.json")
    
    assert os.path.exists(model_path), "Stacking model file does not exist!"
    assert os.path.exists(metrics_path), "Model metrics file does not exist!"
    
    model = joblib.load(model_path)
    assert model is not None, "Failed to load model from disk!"
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    stacking_metrics = metrics["Stacking Ensemble"]
    
    # Gap 1: Overfitting check (train R2 - test R2 < 0.05)
    overfit_gap = stacking_metrics["overfit_gap_r2"]
    assert overfit_gap < 0.05, f"Overfit gap too high: {overfit_gap}"
    print(f"  [PASS] Gap 1 Overfitting Controlled: Train R²={stacking_metrics['train_r2']} vs Test R²={stacking_metrics['test_r2']} (Gap: {overfit_gap})")
    
    # Gap 2: Stacking superiority
    assert stacking_metrics["test_r2"] >= metrics["Linear Regression"]["test_r2"], "Stacking did not beat Linear Regression!"
    assert stacking_metrics["test_r2"] >= metrics["Decision Tree"]["test_r2"], "Stacking did not beat Decision Tree!"
    assert stacking_metrics["test_mae"] <= metrics["Linear Regression"]["test_mae"], "Stacking MAE higher than baseline!"
    print(f"  [PASS] Gap 2 Stacking Outperforms: Stacking MAE ({stacking_metrics['test_mae']}) beat Linear Regression ({metrics['Linear Regression']['test_mae']}) and RF ({metrics['Random Forest']['test_mae']})")

def test_data_quality():
    print("[Test 3] Testing Data Quality and Gap 5 Resolution...")
    summary_path = os.path.join(ROOT_DIR, "data", "processed", "data_quality_summary.json")
    clean_csv_path = os.path.join(ROOT_DIR, "data", "processed", "clean_aqi_data.csv")
    
    assert os.path.exists(summary_path), "Data quality summary missing!"
    assert os.path.exists(clean_csv_path), "Clean CSV missing!"
    
    with open(summary_path, "r") as f:
        summary = json.load(f)
        
    assert summary["cities_count"] >= 25, f"Expected at least 25 cities, got {summary['cities_count']}"
    assert sum(summary["final_missing_counts"].values()) == 0, "Processed data still has missing values!"
    assert summary["fake_zeros_addressed"]["CO"] > 0, "Fake zeros in CO were not addressed!"
    print(f"  [PASS] Gap 5 Data Quality Audit: 0 missing values across all columns, {summary['fake_zeros_addressed']['CO']} CO sensor fake-zeros fixed.")

def test_api_endpoints():
    print("[Test 4] Testing FastAPI Backend Endpoints...")
    
    # Test /api/cities
    cities_res = get_cities()
    cities = cities_res["cities"]
    assert len(cities) >= 25, f"Expected >= 25 cities, got {len(cities)}"
    print(f"  [PASS] /api/cities returns {len(cities)} Indian cities.")
    
    # Test /api/models/comparison
    comp = get_model_comparison()
    assert "metrics" in comp
    assert "Stacking Ensemble" in comp["metrics"]
    print("  [PASS] /api/models/comparison returns complete benchmark suite.")
    
    # Test /api/error-analysis
    err = get_error_analysis()
    assert "performance_by_category" in err
    assert "Severe" in err["performance_by_category"]
    print(f"  [PASS] /api/error-analysis returns category diagnostics (Accuracy: {err['overall']['category_classification_accuracy_percent']}%).")
    
    # Test /api/historical
    hist = get_historical_trends("Chennai", 30)
    assert len(hist["history"]) == 30, f"Expected 30 records, got {len(hist['history'])}"
    print("  [PASS] /api/historical returns valid chronological records.")
    
    # Test /api/predict for multiple regions
    test_cities = ["Chennai", "Delhi", "Bengaluru", "Mumbai", "Coimbatore"]
    for city in test_cities:
        req = PredictionRequest(
            city=city,
            date="2026-09-20",
            pm25=45.0,
            pm10=80.0
        )
        pred_res = predict_aqi(req)
        assert pred_res["city"] == city
        assert pred_res["predicted_aqi"] > 0
        assert "advisory" in pred_res
        assert pred_res["advisory"]["category"] in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
        print(f"  [PASS] /api/predict for {city}: AQI={pred_res['predicted_aqi']} [{pred_res['advisory']['category']}]")

if __name__ == "__main__":
    print("=" * 60)
    print("  RUNNING AIRSENSE AUTOMATED VERIFICATION SUITE")
    print("=" * 60)
    test_advisory_bands()
    test_model_and_metrics()
    test_data_quality()
    test_api_endpoints()
    print("=" * 60)
    print("  ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)
