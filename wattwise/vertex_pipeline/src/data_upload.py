import src.config as config

from kfp.v2.dsl import (
    component,  # For creating pipeline components
)


@component(
    base_image=f"{config.REGION}-docker.pkg.dev/{config.PROJECT_ID}/{config.REPOSITORY}/{config.IMAGE_NAME}:{config.IMAGE_TAG}",
    output_component_file="data_upload.yaml",
)
def data_upload(
    cfg: dict,
):
    """ """
    import fsspec
    import logging

    import zipfile
    import cdsapi
    import tempfile
    import os

    # from google.cloud.storage import Client, transfer_manager

    bucket_name = cfg["bucket_name"]

    fs = fsspec.filesystem("gs")

    REQUEST_METEO = {
        "variable": [
            "wind_speed_at_100m",
            "wind_speed_at_10m",
            "surface_downwelling_shortwave_radiation",
            "pressure_at_sea_level",
            "2m_air_temperature",
            "total_precipitation",
        ],
        "spatial_aggregation": ["country_level"],
        "temporal_aggregation": ["daily"],
    }

    REQUEST_ENERGY = {
        "variable": [
            "electricity_demand",
            "hydro_power_generation_reservoirs",
            "hydro_power_generation_rivers",
            "solar_photovoltaic_power_generation",
            "wind_power_generation_onshore",
        ],
        "spatial_aggregation": ["country_level"],
        "energy_product_type": ["power"],
        "temporal_aggregation": ["daily"],
    }

    DATASET_COPERNICUS = "sis-energy-derived-reanalysis"

    for key, request in zip(["meteo", "energy"], [REQUEST_METEO, REQUEST_ENERGY]):
        logging.info(f"Requesting {key} data from CDS...")
        client = cdsapi.Client(
            url=cfg["cds_url"],
            key=cfg["cds_key"],
        )

        # Create a temporary file for the zip
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
            temp_zip_path = temp_zip.name
            client.retrieve(DATASET_COPERNICUS, request).download(target=temp_zip_path)

        logging.info(
            f"Downloaded {key} data from CDS to temporary file {temp_zip_path}."
        )

        # Extract the zip file to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            logging.info(f"Unzipping {key} data to temporary directory {temp_dir}...")
            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            logging.info(f"Uploading extracted {key} data to GCS bucket...")
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    local_file_path = os.path.join(root, file)

                    relative_path = os.path.relpath(local_file_path, temp_dir)

                    gcs_path = f"gs://{bucket_name}/data/{key}2/{relative_path}"

                    fs.put(local_file_path, gcs_path)

                    logging.info(f"Uploaded {local_file_path} to {gcs_path}.")

        # Clean up the temporary zip file
        os.remove(temp_zip_path)

        logging.info(f"Cleaned up temporary file {temp_zip_path}.")

        logging.info(f"Finished processing {key} data.")
