__all__ = [
    "config",
]

from . import config


# from kfp.v2 import dsl
# from kfp.v2.dsl import (
#     Artifact,    # For handling ML artifacts
#     Dataset,     # For handling datasets
#     Input,       # For component inputs
#     Model,       # For handling ML models
#     Output,      # For component outputs
#     Metrics,     # For tracking metrics
#     HTML,        # For visualization
#     component,   # For creating pipeline components
#     pipeline     # For defining the pipeline
# )
# from kfp.v2 import compiler
# from google.cloud.aiplatform import pipeline_jobs

# BUCKET_NAME = "bucket_wattwise_ml_1"
# PIPELINE_ROOT_FOLDER = "wattise"

# PIPELINE_ROOT = f"{BUCKET_NAME}/{PIPELINE_ROOT_FOLDER}"
