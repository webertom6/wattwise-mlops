@REM  running deamon docker desktop

set PROJECT_ID="wattwise-451908"
set REGION="europe-west1"
set REPOSITORY="vertex-ai-pipeline-wattwise-5"
set IMAGE_NAME="training"
set IMAGE_TAG="5"

gcloud beta artifacts repositories create %REPOSITORY% --repository-format=docker --location=%REGION% --description="Repository for Vertex AI pipeline components"

gcloud auth configure-docker %REGION%-docker.pkg.dev

@REM got to the directory where the Dockerfile is located

docker build -t %IMAGE_NAME%:%IMAGE_TAG% .

docker tag %IMAGE_NAME%:%IMAGE_TAG% %REGION%-docker.pkg.dev/%PROJECT_ID%/%REPOSITORY%/%IMAGE_NAME%:%IMAGE_TAG%

docker push %REGION%-docker.pkg.dev/%PROJECT_ID%/%REPOSITORY%/%IMAGE_NAME%:%IMAGE_TAG%

@REM & C:/Users/Student11/AppData/Local/miniconda3/envs/mlops/python.exe c:/Users/Student11/Documents/git/MLOps/wattwise-mlops-project/wattwise/vertex_pipeline/run_pipeline.p
