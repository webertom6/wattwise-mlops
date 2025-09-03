# MLOps API (Flask + MySQL + Cloud Run)

## What it does
- Serves predictions via a Flask API
- Stores data in Cloud SQL (MySQL)
- Ready for Cloud Run deployment

## 🧪 Local Development

1. Create a MySQL instance on GCP
    - Use the `mlops` name for the instance
    - Use the `mluser` name for the user
    - Use the `mlpassword` password for the user
    - Use the `mlops` name for the database
    - Use the `3306` port for the database
    - Use the `us-central1` region for the instance
    - Use the `mysql` database engine for the instance
    - Add your public IP to the "authorized networks" list in the Cloud SQL instance settings

2. Use Docker to create a MySQL container locally:

```bash
    docker run --name mysql-dev -e MYSQL_DATABASE=mlops -e MYSQL_USER=mluser -e MYSQL_PASSWORD=mlpassword -e MYSQL_ROOT_PASSWORD=rootpass -p 3306:3306 -d mysql:8

```

3. Create `.env` file:

```env
DB_HOST=YOUR_CLOUDSQL_PUBLIC_IP/localhost
DB_USER=mluser
DB_PASSWORD=mlpassword
DB_NAME=mlops
DB_PORT=3306
```

4. Run
    - python db/init_db.py
    - python db/insert_mock.py
    - python app.py


## Deploy on Cloud Run

gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/mlops-api

gcloud run deploy mlops-api --image gcr.io/YOUR_PROJECT_ID/mlops-api --platform managed --region us-central1 --allow-unauthenticated --set-env-vars DB_HOST=PUBLIC_IP,DB_USER=mluser,DB_PASSWORD=mlpassword,DB_NAME=mlops
