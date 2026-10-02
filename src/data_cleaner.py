"""
Data Cleaner and Preprocessing Module for AirSense.
Addresses Gap 5 (Data Quality & Sensor Dropout Handling) and Feature Engineering.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "city_day.csv")
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "clean_aqi_data.csv")
QUALITY_SUMMARY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "data_quality_summary.json")

POLLUTANT_COLS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]

def get_season(month: int) -> int:
    """
    Returns Indian meteorological season:
    1: Winter (Dec, Jan, Feb)
    2: Summer / Pre-monsoon (Mar, Apr, May)
    3: Monsoon (Jun, Jul, Aug, Sep)
    4: Post-monsoon / Autumn (Oct, Nov)
    """
    if month in [12, 1, 2]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8, 9]:
        return 3
    else:
        return 4

def clean_and_prepare_data(raw_path: str = RAW_DATA_PATH, save_processed: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw CPCB data, fixes fake zeros / sensor dropout, imputes missing values,
    creates temporal and lag features, and exports processed dataset.
    """
    print(f"[DataCleaner] Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    
    # 1. Capture initial data quality stats (before audit)
    initial_missing = df[POLLUTANT_COLS + ["AQI"]].isnull().sum().to_dict()
    # Detect 'fake zeros' (physically implausible zeroes representing sensor dropouts)
    fake_zeros_detected = {
        col: int((df[col] == 0).sum()) for col in POLLUTANT_COLS if (df[col] == 0).sum() > 0
    }
    
    # 2. Fix fake zero sensor dropouts by converting to NaN for systematic imputation
    for col, count in fake_zeros_detected.items():
        print(f"[DataCleaner] Detected {count} fake zero readings in {col} (sensor dropouts) -> replacing with NaN for imputation")
        df.loc[df[col] == 0, col] = np.nan
        
    # 3. Filter to rows with ground-truth AQI for supervised learning
    df = df.dropna(subset=["AQI"]).copy()
    valid_aqi_rows = len(df)
    print(f"[DataCleaner] Retained {valid_aqi_rows} rows with ground-truth AQI (dropped {initial_rows - valid_aqi_rows} ungrounded rows)")
    
    # 4. Chronological sorting
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["City", "Date"]).reset_index(drop=True)
    
    # 5. Extract temporal features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Season"] = df["Month"].apply(get_season)
    df["sin_month"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["Month"] / 12)
    
    # 6. Imputation pipeline (Gap 5):
    # Step A: City-level forward fill (past readings carry over) within reasonable time bounds
    df[POLLUTANT_COLS] = df.groupby("City")[POLLUTANT_COLS].ffill()
    
    # Step B: City + Month seasonal median imputation
    for col in POLLUTANT_COLS:
        city_month_medians = df.groupby(["City", "Month"])[col].transform("median")
        df[col] = df[col].fillna(city_month_medians)
        
    # Step C: Overall City median imputation fallback
    for col in POLLUTANT_COLS:
        city_medians = df.groupby("City")[col].transform("median")
        df[col] = df[col].fillna(city_medians)
        
    # Step D: Nationwide monthly median imputation fallback
    for col in POLLUTANT_COLS:
        nation_monthly_medians = df.groupby("Month")[col].transform("median")
        df[col] = df[col].fillna(nation_monthly_medians)
        # Any residual nan filled with global median
        df[col] = df[col].fillna(df[col].median())
        
    # 7. Domain Feature Engineering
    # Particulate matter ratio (PM2.5 to PM10)
    df["PM_Ratio"] = (df["PM2.5"] / (df["PM10"] + 1e-5)).clip(0.0, 1.0)
    
    # Nitrogen oxides balance (NO2 to NOx)
    df["NO2_to_NOx"] = (df["NO2"] / (df["NOx"] + 1e-5)).clip(0.0, 2.0)
    
    # Operational 1-day lag features (Yesterday's AQI and PM2.5)
    df["AQI_lag1"] = df.groupby("City")["AQI"].shift(1)
    df["PM2.5_lag1"] = df.groupby("City")["PM2.5"].shift(1)
    
    # Fill initial lag nulls per city with current day values
    df["AQI_lag1"] = df["AQI_lag1"].fillna(df["AQI"])
    df["PM2.5_lag1"] = df["PM2.5_lag1"].fillna(df["PM2.5"])
    
    # Re-sort strictly by Date then City for chronological modeling
    df = df.sort_values(by=["Date", "City"]).reset_index(drop=True)
    
    # 8. Post-cleaning quality audit
    final_missing = df[POLLUTANT_COLS + ["AQI", "PM_Ratio", "AQI_lag1"]].isnull().sum().to_dict()
    
    quality_summary = {
        "initial_total_records": initial_rows,
        "records_with_ground_truth_aqi": valid_aqi_rows,
        "cities_count": int(df["City"].nunique()),
        "cities_list": sorted(df["City"].unique().tolist()),
        "date_range": {
            "start": df["Date"].min().strftime("%Y-%m-%d"),
            "end": df["Date"].max().strftime("%Y-%m-%d")
        },
        "fake_zeros_addressed": fake_zeros_detected,
        "initial_missing_counts": {k: int(v) for k, v in initial_missing.items()},
        "final_missing_counts": {k: int(v) for k, v in final_missing.items()},
        "imputation_methods": [
            "1. Zero-inflated sensor dropout replacement with NaN",
            "2. City-level temporal forward-fill",
            "3. City x Month seasonal median imputation",
            "4. City median and nationwide monthly median hierarchical fallback"
        ]
    }
    
    if save_processed:
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        with open(QUALITY_SUMMARY_PATH, "w") as f:
            json.dump(quality_summary, f, indent=2)
        print(f"[DataCleaner] Processed dataset saved -> {PROCESSED_DATA_PATH}")
        print(f"[DataCleaner] Data quality audit saved -> {QUALITY_SUMMARY_PATH}")
        
    return df, quality_summary

if __name__ == "__main__":
    clean_and_prepare_data()
