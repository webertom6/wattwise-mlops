
# Wattwise Vertex AI ML Pipeline

This directory contains the full implementation of the **machine learning pipeline** that runs on **Google Cloud Vertex AI**, covering data ingestion, preprocessing, training, and prediction. The pipeline automatically stores results in **Google Bigtable**.

---

## What It Does

- Ingests energy and meteorological data
- Preprocesses & scales features
- Trains two separate models:
  - `training_energy.py`: Forecasting energy-related variables
  - `training_meteo.py`: Forecasting meteorological features
- Predicts values and pushes them to Bigtable
- Runs containerized via Docker on Vertex AI Pipelines

---

## Project Structure

```
vertex_pipeline/
│
├── Dockerfile                         # Pipeline container config
├── cmd_setup_dockerimg_windows.bat   # Windows setup script
├── requirements.txt                   # Python dependencies
├── run_pipeline.py                    # Vertex pipeline runner
│
└── src/
    ├── __init__.py
    ├── data_ingestion.py             # API/data fetch logic
    ├── data_preprocessing.py         # Cleaning, scaling, feature engineering
    ├── training_energy.py            # Train model on energy data
    ├── training_meteo.py             # Train model on weather data
    ├── prediction.py                 # Generate and save predictions
    ├── data_upload.py                # Upload results to Bigtable
    └── setup.py                      # Pipeline component definitions
```

---

## How to Use

### 1. Build and Push Docker Image

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/vertex-pipeline
```

### 2. Run the Pipeline

```bash
python run_pipeline.py
```

Ensure your `GOOGLE_APPLICATION_CREDENTIALS` are properly set before running.

---

## Dependencies

Install requirements:

```bash
pip install -r requirements.txt
```

---

## GCP Services Used

- Vertex AI Pipelines
- Artifact Registry
- Google Bigtable
- IAM Service Accounts

---
