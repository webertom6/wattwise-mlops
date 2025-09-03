
# Wattwise Energy Forecast Web Application

This repository contains a web application built with **Streamlit** to visualize energy forecasts and weather data, connected to a **Google Cloud Bigtable** backend. The app is containerized with Docker and deployed on **Google Cloud Run**.

---

## Features

- Interactive geographical map of European countries.
- Time-series plots of energy demand, renewables, and weather features.
- Data fed from a Vertex AI ML pipeline into Bigtable.
- Live data reads directly from Bigtable on every selection.
- Containerized & deployed with Docker and Cloud Run.

---

## Project Structure

```
web_application/
│
├── streamlit_app.py              # Main Streamlit web app
├── Dockerfile                    # Container config
├── requirements.txt              # Python dependencies
├── .dockerignore                 # Docker exclusions
│
├── db/
│   ├── init_bigtable.py          # Bigtable table creation
│   ├── write_to_bigtable.py      # Script to push data to Bigtable
│   ├── delete_tables.py          # Optional clean-up script
│   ├── *.csv                     # Sample data (energy & predictions)
│
└── __pycache__/                  # Python bytecode cache (ignored)
```

---

## Deployment

To build and deploy the app using Google Cloud:

### 1. Build Docker image

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/streamlit-app
```

### 2. Deploy to Cloud Run

```bash
gcloud run deploy streamlit-app \
  --image gcr.io/YOUR_PROJECT_ID/streamlit-app \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --service-account=YOUR_SERVICE_ACCOUNT
```

---

## Requirements

See `requirements.txt` for full dependency list.

Install locally:

```bash
pip install -r requirements.txt
```

---

## Cloud Dependencies

- Google Bigtable API
- Google Cloud IAM (service accounts)
- Cloud Build & Cloud Run

---
