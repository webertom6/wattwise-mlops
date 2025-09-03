import src.config as config

from kfp.v2.dsl import (
    Artifact,  # For handling ML artifacts
    Dataset,  # For handling datasets
    Input,  # For component inputs
    Output,  # For component outputs
    component,  # For creating pipeline components
)


@component(
    base_image=f"{config.REGION}-docker.pkg.dev/{config.PROJECT_ID}/{config.REPOSITORY}/{config.IMAGE_NAME}:{config.IMAGE_TAG}",
    output_component_file="data_preprocessing.yaml",
)
def data_preprocessing(
    input_dataset: Input[Dataset],
    preprocessed_dataset: Output[Dataset],
    preprocessed_target_energy: Output[Dataset],
    preprocessed_target_meteo: Output[Dataset],
    scaler_y_energy: Output[Artifact],
    scaler_y_meteo: Output[Artifact],
):
    """
    Preprocesses the dataset for training.

    Args:
        input_dataset: Input dataset from the data ingestion step
        preprocessed_dataset: Output artifact for the preprocessed dataset
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import logging
    import joblib

    # Load the dataset
    df = pd.read_csv(input_dataset.path, parse_dates=["Date"], index_col="Date")

    # Handle missing values
    data = df.fillna(method="ffill")

    # Preprocess the dataset energy
    data_y_energy = data[
        [
            "Electricity Demand",
            "Hydropower reservoir",
            "Hydropower run-of-river",
            "Solar PV Power",
            "Wind Power Onshore",
        ]
    ]

    targets_energy = data_y_energy.columns.tolist()

    logging.info(f"Targets energy: {targets_energy}")

    # Preprocess the dataset meteo
    data_y_meteo = data[
        [
            "Global Horizontal Irradiance",
            "Mean Sea Level Pressure",
            "Air Temperature",
            "Total Precipitation",
            "Wind Speed_x",
            "Wind Speed_y",
        ]
    ]

    targets_meteo = data_y_meteo.columns.tolist()

    logging.info(f"Targets meteo: {targets_meteo}")

    # Normalize data
    scalerX = StandardScaler()
    dataX_scaled = scalerX.fit_transform(data)

    scalerY_energy = StandardScaler()
    data_y_energy_scaled = scalerY_energy.fit_transform(data_y_energy)

    scalerY_meteo = StandardScaler()
    data_y_meteo_scaled = scalerY_meteo.fit_transform(data_y_meteo)

    joblib.dump(scalerY_energy, scaler_y_energy.path)
    logging.info(f"Scaler saved to: {scaler_y_energy.path}")

    joblib.dump(scalerY_meteo, scaler_y_meteo.path)
    logging.info(f"Scaler saved to: {scaler_y_meteo.path}")

    # Save preprocessed dataset
    df_processed = pd.DataFrame(data=dataX_scaled, columns=df.columns)
    df_processed.to_csv(preprocessed_dataset.path, index=False)
    logging.info(f"Preprocessed dataset saved to: {preprocessed_dataset.path}")

    # Save preprocessed target energy
    df_y_energy_processed = pd.DataFrame(
        data=data_y_energy_scaled, columns=targets_energy
    )
    df_y_energy_processed.to_csv(preprocessed_target_energy.path, index=False)
    logging.info(
        f"Preprocessed target energy saved to: {preprocessed_target_energy.path}"
    )

    # Save preprocessed target meteo
    df_y_meteo_processed = pd.DataFrame(data=data_y_meteo_scaled, columns=targets_meteo)
    df_y_meteo_processed.to_csv(preprocessed_target_meteo.path, index=False)
    logging.info(
        f"Preprocessed target meteo saved to: {preprocessed_target_meteo.path}"
    )
