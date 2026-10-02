"""
AirSense Execution Script.
Checks model availability and launches the FastAPI Web Dashboard.
"""

import os
import sys
import uvicorn

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "models", "stacking_aqi_model.joblib")

def main():
    if not os.path.exists(MODEL_PATH):
        print("[AirSense] Model artifact not found. Triggering automated data pipeline & training...")
        from src.data_loader import download_dataset
        from src.data_cleaner import clean_and_prepare_data
        from src.train import train_and_evaluate_all
        from src.error_analysis import perform_error_analysis
        
        download_dataset()
        clean_and_prepare_data()
        train_and_evaluate_all()
        perform_error_analysis()
        print("[AirSense] Model training completed successfully!")

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print("\n" + "=" * 60)
    print("  [AirSense] LOCALIZED AQI & HEALTH ADVISORY SYSTEM")
    print("=" * 60)
    print(f"  * Local Web Dashboard: http://localhost:{port}")
    print(f"  * Interactive Swagger API: http://localhost:{port}/docs")
    print(f"  * Health Endpoint: http://localhost:{port}/api/health")
    print("=" * 60 + "\n")

    uvicorn.run("app.main:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    main()
