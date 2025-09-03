import src.config as config

from kfp.v2.dsl import (
    Artifact,  # For handling ML artifacts
    Dataset,  # For handling datasets
    Input,  # For component inputs
    Model,  # For handling ML models
    Output,  # For component outputs
    HTML,  # For visualization
    component,  # For creating pipeline components
)


@component(
    base_image=f"{config.REGION}-docker.pkg.dev/{config.PROJECT_ID}/{config.REPOSITORY}/{config.IMAGE_NAME}:{config.IMAGE_TAG}",
    output_component_file="prediction.yaml",
)
def prediction(
    cfg: dict,
    dataset: Input[Dataset],
    preprocessed_dataset: Input[Dataset],
    preprocessed_target_energy: Input[Dataset],
    preprocessed_target_meteo: Input[Dataset],
    model_energy: Input[Artifact],
    model_meteo: Input[Model],
    scaler_y_energy: Input[Dataset],
    scaler_y_meteo: Input[Dataset],
    prediction_energy: Output[Dataset],
    prediction_meteo: Output[Dataset],
    html_report: Output[HTML],
):
    """
    Trains the model on the preprocessed dataset.

    Args:
        preprocessed_dataset: Input preprocessed dataset

    """
    import joblib
    import logging
    import base64

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn

    from google.cloud import bigtable

    # Load preprocessed dataset
    dataX = pd.read_csv(dataset.path)
    dataX_scaled = pd.read_csv(preprocessed_dataset.path)
    data_y_energy_scaled = pd.read_csv(preprocessed_target_energy.path)
    data_y_meteo_scaled = pd.read_csv(preprocessed_target_meteo.path)

    all_features = dataX_scaled.columns.tolist()
    targets_energy = data_y_energy_scaled.columns.tolist()
    targets_meteo = data_y_meteo_scaled.columns.tolist()

    # Convert dataX_scaled from pandas DataFrame to numpy.ndarray, ignoring the 'Date' column
    if "Date" in dataX_scaled.columns:
        dataX_scaled = dataX_scaled.drop(columns=["Date"])
    dataX_scaled = dataX_scaled.to_numpy()

    if "Date" in data_y_energy_scaled.columns:
        data_y_energy_scaled = data_y_energy_scaled.drop(columns=["Date"])
    data_y_energy_scaled = data_y_energy_scaled.to_numpy()

    if "Date" in data_y_meteo_scaled.columns:
        data_y_meteo_scaled = data_y_meteo_scaled.drop(columns=["Date"])
    data_y_meteo_scaled = data_y_meteo_scaled.to_numpy()

    # 1. Split features and target
    def create_sequences(dataX, data_y, seq_length, prediction_horizon):
        X, y = [], []
        for i in range(len(dataX) - seq_length - prediction_horizon + 1):
            X.append(dataX[i : i + seq_length])
            y.append(data_y[i + seq_length + prediction_horizon - 1])
        return np.array(X), np.array(y)

    seq_length = cfg["seq_length"]  # Use past 30 days to predict
    prediction_horizon = cfg["prediction_horizon"]  # Predict 30 days ahead

    X, y1 = create_sequences(
        dataX_scaled, data_y_energy_scaled, seq_length, prediction_horizon
    )
    X, y2 = create_sequences(
        dataX_scaled, data_y_meteo_scaled, seq_length, prediction_horizon
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # PyTorch v0.4.0
    logging.info(f"Using device: {device}")

    # 3. Initialize and train the model
    class GRUModelEnergy(nn.Module):
        def __init__(
            self, input_size, hidden_size, num_layers, output_size, dropout=0.3
        ):
            super(GRUModelEnergy, self).__init__()
            self.gru = nn.GRU(
                input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
            )
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size // 2, output_size)

            # Initialize GRU weights
            for name, param in self.gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)

            # Initialize FC weights
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.constant_(self.fc2.bias, 0)

        def forward(self, x, h0=None):
            # If hidden state is not provided, initialize it as zeros
            if h0 is None:
                h0 = torch.zeros(
                    self.gru.num_layers, x.size(0), self.gru.hidden_size
                ).to(x.device)

            gru_out, hn = self.gru(x, h0)
            x = self.dropout(gru_out[:, -1, :])
            x = self.fc1(x)  # Use the last output of the GRU
            x = self.relu(x)
            return self.fc2(x), hn

    input_size = len(all_features)
    hidden_size = cfg[
        "hidden_size"
    ]  # Increased hidden size for better feature learning
    num_layers = cfg[
        "num_layers"
    ]  # Increased number of layers for deeper representation
    output_size = len(targets_energy)
    dropout = cfg["dropout"]  # Added dropout for regularization

    gru_model_energy = GRUModelEnergy(
        input_size, hidden_size, num_layers, output_size, dropout
    ).to(device)

    logging.info(
        f"Nb of hyperparamters in {gru_model_energy.__class__.__name__}: {sum(p.numel() for p in gru_model_energy.parameters() if p.requires_grad):,}"
    )

    logging.info(f"Model: {gru_model_energy}")

    # load the model weights if available

    ######################################################################

    # Load the model weights for GRUModelEnergy if available
    logging.info(f"Loading GRUModelEnergy weights from: {model_energy.path}")
    try:
        gru_model_energy.load_state_dict(
            torch.load(model_energy.path, map_location=device)
        )
        logging.info("GRUModelEnergy weights loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load GRUModelEnergy weights from GCS: {e}")

    #######################################################################

    class GRUModelMeteo(nn.Module):
        def __init__(
            self, input_size, hidden_size, num_layers, output_size, dropout=0.3
        ):
            super(GRUModelMeteo, self).__init__()
            self.gru = nn.GRU(
                input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
            )
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size // 2, output_size)

            # Initialize GRU weights
            for name, param in self.gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)

            # Initialize FC weights
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.constant_(self.fc2.bias, 0)

        def forward(self, x, h0=None):
            # If hidden state is not provided, initialize it as zeros
            if h0 is None:
                h0 = torch.zeros(
                    self.gru.num_layers, x.size(0), self.gru.hidden_size
                ).to(x.device)

            gru_out, hn = self.gru(x, h0)
            x = self.dropout(gru_out[:, -1, :])
            x = self.fc1(x)  # Use the last output of the GRU
            x = self.relu(x)
            return self.fc2(x), hn

    input_size = len(all_features)
    hidden_size = cfg[
        "hidden_size"
    ]  # Increased hidden size for better feature learning
    num_layers = cfg[
        "num_layers"
    ]  # Increased number of layers for deeper representation
    output_size = len(targets_meteo)
    dropout = cfg["dropout"]  # Added dropout for regularization

    gru_model_meteo = GRUModelMeteo(
        input_size, hidden_size, num_layers, output_size, dropout
    ).to(device)

    logging.info(
        f"Nb of hyperparamters in {gru_model_meteo.__class__.__name__}: {sum(p.numel() for p in gru_model_meteo.parameters() if p.requires_grad):,}"
    )

    logging.info(f"Model: {gru_model_meteo}")

    # load the model weights if available

    ######################################################################
    # Load the model weights for GRUModelMeteo if available
    logging.info(f"Loading GRUModelMeteo weights from: {model_meteo.path}")
    try:
        gru_model_meteo.load_state_dict(
            torch.load(model_meteo.path, map_location=device)
        )
        logging.info("GRUModelMeteo weights loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load GRUModelMeteo weights from GCS: {e}")

    #######################################################################

    # Take last sequence and predict next month
    last_sequence = X[-2]

    def predict_next_month_uncertinaty(
        model_energy, model_meteo, last_data, steps=30, nb_iter=100
    ):
        model_energy.train()
        model_meteo.train()

        predictions_energy = []
        predictions_meteo = []

        input = torch.tensor(last_data, dtype=torch.float32).unsqueeze(0)
        logging.info(input.shape)
        logging.info(input.device)

        for _ in range(steps):
            logging.info(f"step : {_} / {steps}")
            with torch.no_grad():
                predict_monte_carlo_energy = []
                predict_monte_carlo_meteo = []
                for _ in range(nb_iter):
                    # logging.info(f"iter : {_} / {nb_iter}")
                    input = input.to(device)

                    h0 = None  # Initialize hidden and cell states

                    # outputs1 : (1, 5)
                    outputs1, h0 = model_energy(input, h0)

                    # outputs2 : (1, 6)
                    outputs2, h0 = model_meteo(input, h0)

                    # Detach hidden and cell states to prevent backpropagation through the entire sequence
                    h0 = h0.detach()

                    predict_monte_carlo_energy.append(outputs1.cpu().numpy())
                    predict_monte_carlo_meteo.append(outputs2.cpu().numpy())

                predictions_energy.append(predict_monte_carlo_energy)
                predictions_meteo.append(predict_monte_carlo_meteo)

            # Combine outputs1 and outputs2 to form the 11 features
            combined_outputs = torch.cat((outputs2, outputs1), dim=1)
            # logging.info(combined_outputs.shape) # (1, 11)

            # Add the combined outputs to the input sequence
            # logging.info(input[:, 1:, :].shape) # (1, 299, 11)
            input = torch.cat((input[:, 1:, :], combined_outputs.unsqueeze(0)), dim=1)

        logging.info(f"nb of pedicctions : {len(predictions_energy)}")  # 30
        logging.info(f"nb of pedicctions : {len(predictions_meteo)}")

        return predictions_energy, predictions_meteo

    preds_mc_energy, preds_mc_meteo = predict_next_month_uncertinaty(
        gru_model_energy,
        gru_model_meteo,
        last_sequence,
        steps=prediction_horizon * 2,
        nb_iter=20,
    )

    preds_mc_energy = np.array(preds_mc_energy)
    logging.info(preds_mc_energy.shape)  # (30, 10, 1, 5)
    preds_mc_meteo = np.array(preds_mc_meteo)
    logging.info(preds_mc_meteo.shape)  # (30, 10, 1, 6)

    scaler_y_enrg = joblib.load(scaler_y_energy.path)
    scaler_y_mto = joblib.load(scaler_y_meteo.path)

    # Inverse transform the predictions on preds_mc
    preds_mc_energy_inv = []
    preds_mc_meteo_inv = []
    for i in range(preds_mc_energy.shape[1]):
        logging.info(f"preds_mc_energy.shape : {preds_mc_energy[:, i].shape}")
        preds_mc_energy_inv.append(
            scaler_y_enrg.inverse_transform(preds_mc_energy[:, i].squeeze(axis=1))
        )
        preds_mc_meteo_inv.append(
            scaler_y_mto.inverse_transform(preds_mc_meteo[:, i].squeeze(axis=1))
        )

    preds_mc_energy_inv = np.array(preds_mc_energy_inv)
    preds_mc_energy_inv = np.transpose(preds_mc_energy_inv, (1, 0, 2))
    preds_mc_energy_inv = np.expand_dims(preds_mc_energy_inv, axis=2)
    logging.info(preds_mc_energy_inv.shape)  # (60, 10, 1, 5)

    preds_mc_meteo_inv = np.array(preds_mc_meteo_inv)
    preds_mc_meteo_inv = np.transpose(preds_mc_meteo_inv, (1, 0, 2))
    preds_mc_meteo_inv = np.expand_dims(preds_mc_meteo_inv, axis=2)
    logging.info(preds_mc_meteo_inv.shape)  # (60, 10, 1, 6)

    num_days = len(data_y_energy_scaled)
    last_date = pd.Timestamp("1980-01-01") + pd.Timedelta(days=num_days - 1)
    logging.info(f"Last date in dataset: {last_date}")
    # Generate future dates for the x-axis
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=prediction_horizon * 2, freq="D"
    )

    start_date = pd.Timestamp("1980-01-01") + pd.Timedelta(days=num_days - seq_length)
    logging.info(f"Start date for actual data: {start_date}")

    # Generate the actual data using inverse transform
    actual_data_energy = pd.DataFrame(
        scaler_y_enrg.inverse_transform(data_y_energy_scaled[-seq_length:]),
        columns=targets_energy,
        index=pd.date_range(start=start_date, periods=seq_length, freq="D"),
    )

    actual_data_meteo = pd.DataFrame(
        scaler_y_mto.inverse_transform(data_y_meteo_scaled[-seq_length:]),
        columns=targets_meteo,
        index=pd.date_range(start=start_date, periods=seq_length, freq="D"),
    )

    # Generate dates for the actual data
    actual_dates = actual_data_energy.index

    def save_preds_df(predictions, targets):
        # Calculate mean, lower bound, and upper bound for the interval of confidence
        mean_pred = predictions.mean(axis=1).squeeze(axis=1)  # Mean across iterations
        std_pred = predictions.std(axis=1).squeeze(
            axis=1
        )  # Standard deviation across iterations

        lower_bound = mean_pred - 1.96 * std_pred  # 95% confidence interval lower bound
        upper_bound = mean_pred + 1.96 * std_pred  # 95% confidence interval upper bound

        # Create a DataFrame for predictions with mean, lower bound, and upper bound
        predictions_df = pd.DataFrame(mean_pred, columns=targets)
        predictions_df.insert(0, "Date", future_dates)

        # Add lower and upper bounds for each energy feature
        for i, feature in enumerate(targets):
            predictions_df[f"{feature}_lower_bound"] = lower_bound[:, i]
            predictions_df[f"{feature}_upper_bound"] = upper_bound[:, i]

        return predictions_df

    predictions_energy_df = save_preds_df(preds_mc_energy_inv, targets_energy)
    predictions_meteo_df = save_preds_df(preds_mc_meteo_inv, targets_meteo)

    # Save predictions to CSV
    logging.info(f"Saving predictions to {prediction_energy.path}...")
    predictions_energy_df.to_csv(prediction_energy.path, index=False)
    logging.info(f"Predictions saved to: {prediction_energy.path}")

    logging.info(f"Saving predictions to {prediction_meteo.path}...")
    predictions_meteo_df.to_csv(prediction_meteo.path, index=False)
    logging.info(f"Predictions saved to: {prediction_meteo.path}")

    client = bigtable.Client(project="wattwise-459502", admin=True)
    instance = client.instance("wattwise-bigtable")

    def write_df_to_bigtable(df, table_id, country_code="ES"):
        table = instance.table(table_id)
        if not table.exists():
            logging.info(f"Table '{table_id}' does not exist.")
            return

        for _, row in df.iterrows():
            date_str = str(row["Date"])[:10]  # assume format: YYYY-MM-DD
            row_key = f"{country_code}#{date_str}"
            bt_row = table.direct_row(row_key.encode())

            for col in df.columns:
                if col == "Date":
                    continue
                value = str(row[col])
                bt_row.set_cell("cf1", col, value)

            bt_row.commit()

    write_df_to_bigtable(
        predictions_energy_df, table_id="predictions", country_code="ES"
    )
    write_df_to_bigtable(
        predictions_meteo_df, table_id="predictions", country_code="ES"
    )
    write_df_to_bigtable(dataX, table_id="input_data", country_code="ES")

    def plot_actual_preds_uncertainty(predictions, data_y, targets, name):
        # Calculate mean, lower bound, and upper bound for the interval of confidence
        mean_pred = predictions.mean(axis=1).squeeze(axis=1)  # Mean across iterations
        std_pred = predictions.std(axis=1).squeeze(
            axis=1
        )  # Standard deviation across iterations

        lower_bound = mean_pred - 1.96 * std_pred  # 95% confidence interval lower bound
        upper_bound = mean_pred + 1.96 * std_pred  # 95% confidence interval upper bound

        # Plot each feature in a subplot
        num_features = len(targets)
        plt.figure(figsize=(12, num_features * 3))
        plt.suptitle("Actual and Predicted Energy Features for Continuity", fontsize=16)

        for i, feature in enumerate(targets):
            plt.subplot(num_features, 1, i + 1)
            plt.plot(actual_dates, data_y[feature], label="Actual", color="blue")
            plt.plot(
                future_dates, mean_pred[:, i], label="Mean Prediction", color="orange"
            )
            plt.fill_between(
                future_dates,
                lower_bound[:, i],
                upper_bound[:, i],
                color="orange",
                alpha=0.2,
                label="95% Confidence Interval",
            )
            plt.title(feature)
            plt.xlabel("Date")
            plt.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        time_series_plot = f"{html_report.path}_plot_{name}.png"
        plt.savefig(time_series_plot)

        return time_series_plot

    plot_energy = plot_actual_preds_uncertainty(
        preds_mc_energy_inv, actual_data_energy, targets_energy, name="energy"
    )
    plot_meteo = plot_actual_preds_uncertainty(
        preds_mc_meteo_inv, actual_data_meteo, targets_meteo, name="meteo"
    )

    # OPTIONAL: Save the HTML report
    with open(plot_energy, "rb") as image_file:
        encoded_img_energy = base64.b64encode(image_file.read()).decode("utf-8")
    with open(plot_meteo, "rb") as image_file:
        encoded_img_meteo = base64.b64encode(image_file.read()).decode("utf-8")

    # Embed the image directly into the HTML
    html_content = f"""
    <html>
        <head><title>Model Evaluation Report</title></head>
        <body>
            <h1>Evaluation Metrics</h1>
            <h1>Visualizations</h1>
            <img src="data:image/png;base64,{encoded_img_energy}" alt="Scatter Plot">
            <img src="data:image/png;base64,{encoded_img_meteo}" alt="Scatter Plot">
        </body>
    </html>
    """
    with open(html_report.path, "w") as f:
        f.write(html_content)

    logging.info(f"Evaluation report saved to: {html_report.path}")
