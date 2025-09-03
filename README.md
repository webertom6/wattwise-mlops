
# Wattwise MLOps Project

A full-stack MLOps solution to forecast energy consumption and weather-related variables across European countries using machine learning. This project integrates automated model training and deployment using Google Cloud Platform.

---

## Project Overview

- **Machine Learning Pipeline**: Built on **Vertex AI**, it automates data ingestion, preprocessing, model training, and prediction.
- **Web Application**: Built with **Streamlit**, deployed with **Cloud Run**, it visualizes forecasts and allows user interaction.
- **Cloud Native**: Data is stored in **Google Bigtable**, and infrastructure is orchestrated through **Docker** and **gcloud**.

---

## Web application
The online web application is accessible at : https://streamlit-app-231204006378.europe-west1.run.app/

## Directory Structure

```
wattwise-mlops-project/
│
├── vertex_pipeline/            # Vertex AI ML training pipeline
│   └── README.md
│
├── web_application/            # Streamlit frontend and Docker setup
│   └── README.md
│
├── notebooks/                  # Exploratory notebooks (optional)
├── data/                       # Local CSV files (optional dev use)
├── README.md                   # ← You are here
└── requirements.txt            # Root dependencies
```

---

## Cloud Services Used

| Component         | GCP Service        |
|------------------|--------------------|
| ML Pipelines      | Vertex AI          |
| Model Serving     | Cloud Run          |
| Containerization  | Cloud Build + Docker |
| Storage           | Google Bigtable    |
| Auth              | IAM + Service Accounts |

---

## Contributors

This project was developed by a team of 3 for academic purposes. Contributions welcome!

