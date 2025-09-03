import src.config as config

# from src.config import BUCKET_NAME, PROJECT_ID, REGION, REPOSITORY, IMAGE_NAME, IMAGE_TAG

from kfp.v2.dsl import (
    Dataset,  # For handling datasets
    Output,  # For component outputs
    component,  # For creating pipeline components
)


@component(
    base_image=f"{config.REGION}-docker.pkg.dev/{config.PROJECT_ID}/{config.REPOSITORY}/{config.IMAGE_NAME}:{config.IMAGE_TAG}",
    output_component_file="data_ingestion.yaml",
)
def data_ingestion(
    dataset: Output[Dataset],
    cfg: dict,
):
    """
    Loads and prepares the house price dataset.

    Args:
        dataset: Output artifact to store the prepared dataset
    """
    import pandas as pd
    import sys
    import traceback
    import fsspec
    import logging
    from google.cloud import storage

    def open_df_title_unit_csv(file_path):
        # import pandas as pd
        # import os

        # Read the file and store comments
        logging.info(f"Reading file: {file_path}")
        fs = fsspec.filesystem("gs")
        with fs.open(file_path, "r") as file:
            lines = file.readlines()
        logging.info("File read successfully.")

        comments = [line for line in lines if line.startswith("#")]
        data = [line for line in lines if not line.startswith("#")]

        # # Extract the title and unit from comments
        title_index = comments.index(
            next(
                line
                for line in comments
                if line.startswith("## Title") or line.startswith("## File content")
            )
        )
        title = comments[title_index + 1].strip()
        title = title.replace("#", "").strip()
        # Extract the unit from comments if there is formated like this:
        """
        ## Unit
        ### MWh
        """
        unit_index = comments.index(
            next(line for line in comments if line.startswith("## Unit"))
        )
        unit = comments[unit_index + 1].strip()
        unit = unit.replace("#", "").strip()

        logging.info(f"Title: {title}")
        logging.info(f"Unit: {unit}")

        logging.info("Writing data to temporary file...")

        # Write the data back to a temporary file in Google Cloud Storage
        temp_file_path = file_path + ".tmp"
        with fs.open(temp_file_path, "w") as temp_file:
            temp_file.writelines(data)

        logging.info(f"Temporary file written: {temp_file_path}")

        logging.info(
            f"Reading the data with pandas from temporary file {temp_file_path}..."
        )

        # Read the data with pandas
        df = pd.read_csv(
            temp_file_path, index_col="Date", storage_options={"token": None}
        )

        logging.info("Data read successfully.")

        logging.info(f"Removing temporary file {temp_file_path}...")

        # Optionally, remove the temporary file
        fs.rm(temp_file_path)

        logging.info("Temporary file removed.")

        return df, title, unit

    try:
        logging.info("Starting data ingestion...")
        # 1. Load the dataset from the GCS bucket.
        # 2. Save the dataset to the output artifact.

        data_dir = "data"
        bucket_name = cfg["bucket_name"]

        logging.info(f"Loading dataset from {bucket_name}/{data_dir}...")

        storage_client = storage.Client()

        # Note: Client.list_blobs requires at least package version 1.17.0.
        blobs = storage_client.list_blobs(bucket_name)

        ###################################################################################

        df_es = pd.DataFrame()

        # Note: os.listdir() is not used here as it does not work with GCS paths.
        # Note: The call returns a response only when the iterator is consumed.
        for blob in blobs:
            logging.info(blob.name)

            filename = f"gs://{bucket_name}/{blob.name}"
            logging.info(f"Processing file: {filename}")

            # XOR such that read only if this a .csv or 
            # is either in directory meteo2 XOR energy2
            if (
                filename.endswith(".csv")
                and (("meteo2" in filename) ^ ("energy2" in filename))
            ):
                try:
                    file_path = filename
                    logging.info(f"Reading file: {file_path}")

                    df, title, unit = open_df_title_unit_csv(file_path)
                    title_parts = title.split()
                    if len(title_parts) > 5:
                        title = " ".join(title_parts[:2])
                    logging.info(f"Title after processing: {title}")
                    logging.info(f"Unit: {unit}")

                    # Ensure the dataset contains an 'ES' column to filter for Spain
                    if "ES" in df.columns:
                        # Keep only the 'Date' and 'ES' columns
                        df = df[["ES"]].copy()
                        df.reset_index(inplace=True)
                    else:
                        print(f"Skipping {filename} as no 'ES' column found.")
                        continue

                    # Convert 'Date' column to datetime format before merging
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

                    # Drop rows where 'Date' could not be converted (if any)
                    df = df.dropna(subset=["Date"])

                    # Rename the 'ES' column with the title name
                    df.rename(columns={"ES": title}, inplace=True)

                    # Merge into main DataFrame (on 'Date')
                    if df_es.empty:
                        df_es = df  # First dataset initializes df_es
                    else:
                        df_es = pd.merge(
                            df_es, df, on="Date", how="outer"
                        )  # Merge on 'Date'

                except Exception as e:
                    logging.info(f"Error processing {filename}: {e}")

        # Filter data to keep only records from 2020 onwards
        df_es = df_es[df_es["Date"] >= "1980-01-01"]

        # Display rows where date conversion failed (if any remain after dropping NaNs)
        invalid_dates = df_es[df_es["Date"].isna()]
        if not invalid_dates.empty:
            logging.info("Warning: Some date values could not be converted:")
            logging.info(invalid_dates)

        # Save the final DataFrame to a CSV file
        logging.info(f"Saving dataset to {dataset.path}...")
        df_es.to_csv(dataset.path, index=False)
        logging.info(f"Dataset saved to: {dataset.path}")

        ###################################################################################

    except Exception as e:
        logging.error(f"Error during data ingestion: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)
