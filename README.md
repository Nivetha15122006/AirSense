# 🌬️ AirSense — Localized AQI Prediction & Health Advisory System

[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg?style=flat&logo=scikitlearn)](https://scikit-learn.org)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)

A machine learning system that predicts the Central Pollution Control Board (CPCB) Air Quality Index (AQI) across **26 Indian cities** using pollutant and temporal variables, combining a **Stacking Ensemble model** (Random Forest + Gradient Boosting base learners with a Linear Regression meta-learner) with a production **FastAPI service**, an interactive **web dashboard**, and **plain-language clinical health advisories**.

The project directly overcomes six documented failure modes identified across 10 IEEE/Springer air quality forecasting papers.

---

## 📌 Addressing Documented Literature Gaps

| # | Gap in Existing Research | Literature Failure Mode | AirSense Engineering Solution | Empirical Result in AirSense |
|---|---|---|---|---|
| **1** | **Overfitting** | Models score near-perfectly on train but collapse on test ($R^2=0.99$ train vs $0.35$ test). | 5-Fold Cross-Validation, depth pruning (`max_depth=14`, `min_samples_leaf=4`), and transparent train vs test logging. | **Train $R^2 = 0.970$ vs Test $R^2 = 0.943$** (Overfit gap is only **0.027**). |
| **2** | **Stacking Underused** | Papers test single isolated models (RF alone, XGBoost alone). | Stacking ensemble combining Random Forest (bagging) and Gradient Boosting (boosting) through a Linear Regression meta-learner. | **Stacking beats all individual models** ($R^2 = 0.9433$, MAE = **11.27** vs RF MAE = 11.34, GB MAE = 11.89, Baseline Linear = 16.11). |
| **3** | **No Deployed Output** | Every paper stops at a static metrics table; zero usable tools. | Full FastAPI REST API + interactive responsive web dashboard with dynamic AQI meters and tailored health advisories. | Real-time deployed web application serving public health guidance. |
| **4** | **Non-Time-Aware Splits** | Random train/test shuffling leaks future seasonal trends into training. | **Strict chronological split**: Train on 2015 through late 2019 (19,880 records); holdout test on 2019–2020 (4,970 records). | True time-series holdout evaluation reflecting real operational forecasting. |
| **5** | **Sensor Data Quality** | Missing values and zero-inflated sensor dropouts unaddressed. | Detected 3,000+ "fake zeros" in sensors ($CO=0$, $NO_x=0$), replacing them with systematic time-series forward fill and City $\times$ Month seasonal median imputation. | Clean, zero-missing-value pipeline verified in `data_quality_summary.json`. |
| **6** | **Regional Specificity** | Studies are heavily Delhi-centric; southern/coastal cities omitted. | Encompasses **26 Indian cities** across South (Chennai, Coimbatore, Bengaluru, Hyderabad, etc.), North (Delhi, Gurugram, etc.), West, East, and Northeast regions. | Granular city-specific baselines and error profiling across geographic archetypes. |

---

## 🏆 Model Benchmark Comparison

Evaluated on the **4,970-sample chronological holdout test set**:

| Model Architecture | Model Family | Test $R^2$ | Train $R^2$ | Overfit Gap ($\Delta R^2$) | Test MAE | Test RMSE | CV $R^2$ (5-Fold) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Linear Regression** | Baseline 1 | 0.9141 | 0.8984 | -0.0157 | 16.11 | 26.37 | 0.8965 ±0.0109 |
| **Decision Tree** | Baseline 2 | 0.9137 | 0.9271 | 0.0134 | 15.04 | 26.44 | 0.8862 ±0.0097 |
| **Random Forest** | Advanced Ensemble 1 | 0.9409 | 0.9659 | 0.0250 | 11.34 | 21.88 | 0.9193 ±0.0114 |
| **Gradient Boosting** | Advanced Ensemble 2 | 0.9409 | 0.9696 | 0.0287 | 11.89 | 21.87 | 0.9217 ±0.0089 |
| **Stacking Ensemble** 🥇 | **Champion Advanced** | **0.9433** | **0.9704** | **0.0271** | **11.27** | **21.42** | **0.9205 ±0.0101** |

---

## 🔍 Step 7: Granular Error Analysis

The model achieves an overall **86.2% Categorical Health Risk Classification Accuracy** on unseen chronological test data:

- **Good (0–50)**: MAE **8.32**, RMSE **11.50** (10.7% of test data).
- **Satisfactory (51–100)**: MAE **7.07**, RMSE **10.25** (42.1% of test data — highly reliable).
- **Moderate (101–200)**: MAE **11.94**, RMSE **16.44** (33.9% of test data — well calibrated).
- **Poor (201–300)**: MAE **23.61**, RMSE **35.71** (7.7% of test data).
- **Very Poor (301–400)**: MAE **15.73**, RMSE **26.58** (4.5% of test data).
- **Severe (401+)**: MAE **72.00**, RMSE **124.03** (1.2% of test data). *Diagnosis:* Displays negative bias (-33.8 AQI) due to severe-event tail sparsity in the training distribution (standard regression toward the mean).
- **Regional Insight**: Southern/Coastal cities (Hyderabad MAE: 4.77, Bengaluru MAE: 4.99, Chennai MAE: 6.80) show lower error than northern industrial plains (Delhi MAE: 9.67, Patna MAE: 15.65) due to maritime atmospheric mixing and less severe crop burning/inversion spikes.

---

## 🏥 Health Advisory Engine

Categorization adheres strictly to the official Indian Central Pollution Control Board (CPCB) NAQI scale:

1. **Good (0–50)**: Minimal impact; natural indoor ventilation recommended.
2. **Satisfactory (51–100)**: Minor breathing discomfort possible for sensitive individuals.
3. **Moderate (101–200)**: Breathing discomfort to people with lung disease, asthma, and heart conditions.
4. **Poor (201–300)**: Breathing discomfort to most people on prolonged exposure; N95 masks advised outdoors.
5. **Very Poor (301–400)**: Health alert; respiratory illness on prolonged exposure. Sensitive groups remain strictly indoors.
6. **Severe (401–500+)**: Emergency conditions; serious cardiovascular and respiratory impact across entire population.

---

## 📁 Repository Structure

```
airsense aml/
├── data/
│   ├── raw/city_day.csv                  # CPCB Raw Dataset (26 cities, 2015-2020)
│   └── processed/
│       ├── clean_aqi_data.csv            # Cleaned, imputed, feature-engineered dataset
│       └── data_quality_summary.json     # Before/after data quality audit report
├── models/
│   ├── stacking_aqi_model.joblib         # Serialized champion Stacking Ensemble
│   ├── model_metrics.json               # Benchmark performance metrics
│   ├── feature_metadata.json            # City baselines, medians, feature names
│   └── error_analysis.json              # Diagnostic error breakdown by category & city
├── src/
│   ├── __init__.py
│   ├── data_loader.py                    # Dataset acquisition and caching
│   ├── data_cleaner.py                   # Imputation, fake zero handling, feature engineering
│   ├── train.py                          # Chronological split, K-fold CV, stacking training
│   ├── error_analysis.py                 # Residual and AQI category error analysis
│   └── advisory.py                       # CPCB AQI categorization and clinical health advisory
├── app/
│   ├── __init__.py
│   ├── main.py                           # FastAPI application & REST endpoints
│   └── static/
│       ├── index.html                    # Responsive interactive web UI
│       ├── style.css                     # Custom styling & animations
│       └── app.js                        # Frontend controller & Chart.js visualizations
├── notebooks/
│   └── eda_and_modeling.ipynb            # Jupyter notebook documenting EDA and training
├── run.py                                # One-click runner script
├── requirements.txt                      # Minimum frozen dependencies
├── Dockerfile                            # Production container spec for Hugging Face Spaces
└── README.md                             # Comprehensive project documentation
```

---

## 🚀 Quick Start (Local Run)

### 1. Launch the Application
```bash
python run.py
```

### 2. Access the Application
- **Interactive Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 🌐 Deploying to Hugging Face Spaces (Recommended)

Hugging Face Spaces provides free 24/7 cloud hosting for FastAPI/Docker applications.

### Option A: Direct Docker Space (Recommended)
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Set Space Name: `airsense-aqi-predictor`.
3. Select License: `MIT` (or your choice).
4. Select Space SDK: **Docker** -> **Blank**.
5. Clone your new Hugging Face Space repository locally:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/airsense-aqi-predictor
   ```
6. Copy all files from this project into the cloned directory:
   ```bash
   # Make sure Dockerfile, run.py, app/, models/, src/, data/ are copied
   git add .
   git commit -m "Deploy AirSense Stacking Ensemble & Dashboard"
   git push
   ```
7. Hugging Face Spaces will automatically build the Docker image and deploy your live dashboard at:
   `https://huggingface.co/spaces/<your-username>/airsense-aqi-predictor`!

---

## ⚡ Deploying to Render or Railway

1. Push this repository to GitHub.
2. In [Render](https://render.com), click **"New Web Service"** and link your GitHub repository.
3. Select **Python** runtime:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run.py`
4. Deploy! Your app will be live with free SSL.
