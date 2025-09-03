from kfp.v2.dsl import pipeline  # For defining the pipeline
from kfp.v2 import compiler
import google.cloud.aiplatform as aiplatform

import src.config as config

from src.data_upload import data_upload
from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.training_energy import training_energy
from src.training_meteo import training_meteo
from src.prediction import prediction

PIPELINE_ROOT = f"gs://{config.BUCKET_NAME}"


@pipeline(name="wattwise_pipeline", pipeline_root=PIPELINE_ROOT)
def wattwise_pipeline():
    cfg = {
        "cds_url": config.CDS_URL,
        "cds_key": config.CDS_KEY,
        "wandb_api_key": config.WANDB_API_KEY,
        "wandb_project_name": config.WANDB_PROJECT_NAME,
        "bucket_name": config.BUCKET_NAME,
        "project_id": config.PROJECT_ID,
        "region": config.REGION,
        "repository": config.REPOSITORY,
        "image_name": config.IMAGE_NAME,
        "image_tag": config.IMAGE_TAG,
        "seed": config.SEED,
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.EPOCHS,
        "early_stopping": config.EARLY_STOPPING,
        "batch_size": config.BATCH_SIZE,
        "train_temp_size": config.TRAIN_TEMP_SIZE,
        "test_val_size": config.TEST_VAL_SIZE,
        "hidden_size": config.HIDDEN_SIZE,
        "num_layers": config.NUM_LAYERS,
        "dropout": config.DROPOUT,
        "seq_length": config.SEQ_LENGTH,
        "prediction_horizon": config.PREDICTION_HORIZON,
    }

    # data_upload_task = data_upload(cfg=cfg)
    data_upload(cfg=cfg)

    ingestion_task = data_ingestion(cfg=cfg)

    preprocessing_task = data_preprocessing(
        input_dataset=ingestion_task.outputs["dataset"]
    )

    training_energy_task = training_energy(
        cfg=cfg,
        preprocessed_dataset=preprocessing_task.outputs["preprocessed_dataset"],
        preprocessed_target_energy=preprocessing_task.outputs[
            "preprocessed_target_energy"
        ],
    )

    training_meteo_task = training_meteo(
        cfg=cfg,
        preprocessed_dataset=preprocessing_task.outputs["preprocessed_dataset"],
        preprocessed_target_meteo=preprocessing_task.outputs[
            "preprocessed_target_meteo"
        ],
    )

    # prediction_task = prediction(
    prediction(
        cfg=cfg,
        dataset=ingestion_task.outputs["dataset"],
        preprocessed_dataset=preprocessing_task.outputs["preprocessed_dataset"],
        preprocessed_target_energy=preprocessing_task.outputs[
            "preprocessed_target_energy"
        ],
        preprocessed_target_meteo=preprocessing_task.outputs[
            "preprocessed_target_meteo"
        ],
        model_energy=training_energy_task.outputs["model_energy"],
        model_meteo=training_meteo_task.outputs["model_meteo"],
        scaler_y_energy=preprocessing_task.outputs["scaler_y_energy"],
        scaler_y_meteo=preprocessing_task.outputs["scaler_y_meteo"],
    )


compiler.Compiler().compile(
    pipeline_func=wattwise_pipeline, package_path="wattwise_pipeline.json"
)

aiplatform.init(project=config.PROJECT_ID, location=config.REGION)

pipeline_job = aiplatform.PipelineJob(
    display_name="wattwise_pipeline_job",
    template_path="wattwise_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
)

pipeline_job_schedule = pipeline_job.create_schedule(
    display_name="SCHEDULE_NAME",
    cron=config.CRON_FREQUENCY,
    max_concurrent_run_count=config.MAX_CONCURRENT_RUN_COUNT,
    max_run_count=config.MAX_RUN_COUNT,
)

pipeline_job.run(
    service_account=config.SA,
)
