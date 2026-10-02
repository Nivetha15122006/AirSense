"""
Data Loader module for AirSense.
Downloads and caches CPCB Indian air quality daily dataset across all Indian cities.
"""

import os
import urllib.request
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/learning-monk/datasets/master/Indian_cities_daily_pollution_2015-2020.csv"
RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "city_day.csv")

def download_dataset(force: bool = False) -> str:
    """
    Downloads the CPCB air quality dataset covering all Indian cities if not already present.
    """
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    if os.path.exists(RAW_DATA_PATH) and not force and os.path.getsize(RAW_DATA_PATH) > 1000000:
        print(f"[DataLoader] Dataset already exists at: {RAW_DATA_PATH} ({os.path.getsize(RAW_DATA_PATH)/1024/1024:.2f} MB)")
        return RAW_DATA_PATH

    print(f"[DataLoader] Downloading nationwide dataset from {DATA_URL} ...")
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "AirSense/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response, open(RAW_DATA_PATH, "wb") as out_file:
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"[DataLoader] Downloaded {downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB ({percent:.1f}%)")
            else:
                print(f"[DataLoader] Downloaded {downloaded / 1024 / 1024:.1f}MB...")
                
    print(f"[DataLoader] Download completed successfully -> {RAW_DATA_PATH}")
    return RAW_DATA_PATH

def load_raw_data() -> pd.DataFrame:
    """Loads raw nationwide CSV dataset into a pandas DataFrame."""
    path = download_dataset()
    df = pd.read_csv(path)
    print(f"[DataLoader] Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"[DataLoader] Cities covered ({df['City'].nunique()} total): {sorted(df['City'].dropna().unique().tolist())}")
    return df

if __name__ == "__main__":
    load_raw_data()
