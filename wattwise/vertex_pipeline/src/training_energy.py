import src.config as config

from kfp.v2.dsl import (
    Artifact,  # For handling ML artifacts
    Dataset,  # For handling datasets
    Input,  # For component inputs
    Output,  # For component outputs
    Metrics,  # For tracking metrics
    component,  # For creating pipeline components
)


@component(
    base_image=f"{config.REGION}-docker.pkg.dev/{config.PROJECT_ID}/{config.REPOSITORY}/{config.IMAGE_NAME}:{config.IMAGE_TAG}",
    output_component_file="training_energy.yaml",
)
def training_energy(
    cfg: dict,
    preprocessed_dataset: Input[Dataset],
    preprocessed_target_energy: Input[Dataset],
    model_energy: Output[Artifact],
    metrics_energy: Output[Metrics],
):
    """
    Trains the model on the preprocessed dataset.

    Args:
        preprocessed_dataset: Input preprocessed dataset
        model: Output artifact for the trained model
        metrics: Output artifact for training metrics
        hyperparameters: Dictionary of hyperparameters
    """
    import logging
    from google.cloud import storage
    import tempfile

    import numpy as np
    import pandas as pd

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from datetime import datetime

    import wandb

    connect = wandb.login(key=cfg["wandb_api_key"])
    if connect:
        logging.info("Wandb login successful")
    else:
        logging.error("Wandb login failed")
        raise Exception("Wandb login failed")

    # Load preprocessed dataset
    dataX_scaled = pd.read_csv(preprocessed_dataset.path)
    data_y_energy_scaled = pd.read_csv(preprocessed_target_energy.path)

    all_features = dataX_scaled.columns.tolist()
    targets_energy = data_y_energy_scaled.columns.tolist()

    # Convert dataX_scaled from pandas DataFrame to numpy.ndarray, ignoring the 'Date' column
    if "Date" in dataX_scaled.columns:
        dataX_scaled = dataX_scaled.drop(columns=["Date"])
    dataX_scaled = dataX_scaled.to_numpy()

    if "Date" in data_y_energy_scaled.columns:
        data_y_energy_scaled = data_y_energy_scaled.drop(columns=["Date"])
    data_y_energy_scaled = data_y_energy_scaled.to_numpy()

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

    # Split data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y1, test_size=cfg["train_temp_size"], random_state=cfg["seed"], shuffle=False
    )
    X_val, _, y_val, _ = train_test_split(
        X_temp,
        y_temp,
        test_size=cfg["test_val_size"],
        random_state=cfg["seed"],
        shuffle=False,
    )

    # Convert data to PyTorch tensors
    X_train, X_val, y_train, y_val = (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    batch_size = cfg["batch_size"]

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Train:", len(train_dataset))
    print("Val:", len(val_dataset))

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
            x = self.fc1(gru_out[:, -1, :])  # Use the last output of the GRU
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
    bucket_name = cfg["bucket_name"]

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)

    for blob in blobs:
        logging.info(f"Blob name: {blob.name}")

        filename = f"gs://{bucket_name}/{blob.name}"
        logging.info(f"Processing file: {filename}")
        if filename.endswith(".pth"):
            # if filename containes word "energy" whatver the case
            if "energy" in filename.lower():
                logging.info(f"Loading model weights from: {filename}")
                # Load the model weights
                # Download the model file from GCS to a temporary file
                with tempfile.NamedTemporaryFile() as temp_file:
                    blob = storage.Blob.from_string(filename, client=storage_client)
                    blob.download_to_filename(temp_file.name)
                    gru_model_energy.load_state_dict(
                        torch.load(temp_file.name, map_location=device)
                    )
                logging.info(f"Model weights loaded from: {filename}")
            else:
                logging.info(
                    f"Skipping file: {filename} as it does not contain 'energy'"
                )

    #######################################################################

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(gru_model_energy.parameters(), lr=cfg["learning_rate"])
    logging.info(f"Loss function: {criterion}")
    logging.info(f"Optimizer: {optimizer}")

    model_name = gru_model_energy.__class__.__name__
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{current_time}"

    logging.info(f"Run name: {run_name}")

    wandb.init(
        project=cfg["wandb_project_name"],
        name=run_name,
        config={
            "model_name": model_name,
            "learning_rate": cfg["learning_rate"],
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "hidden_size": cfg["hidden_size"],
            "num_layers": cfg["num_layers"],
            "dropout": cfg["dropout"],
            "seq_length": cfg["seq_length"],
            "prediction_horizon": cfg["prediction_horizon"],
            "loss": criterion.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
        },
    )

    # 4. Make predictions

    epochs = cfg["epochs"]

    train_loss_arr = []
    val_loss_arr = []

    train_mae_arr = []
    val_mae_arr = []
    train_mse_arr = []
    val_mse_arr = []
    train_rmse_arr = []
    val_rmse_arr = []

    # for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        gru_model_energy.train()
        train_loss = 0.0
        acc_train = 0.0
        mae_train = 0.0
        mse_train = 0.0
        rmse_train = 0.0

        h0 = None  # Initialize hidden and cell states

        for idx, (X_batch, y_batch) in enumerate(train_loader):
            logging.info(f"Training progress : {idx}/{len(train_loader)}")
            if X_batch.size(0) != batch_size:
                continue
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            outputs, h0 = gru_model_energy(X_batch, h0)

            # Detach hidden and cell states to prevent backpropagation through the entire sequence
            h0 = h0.detach()

            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            acc_train += (outputs == y_batch).type(torch.float).sum().item()

            # Compute training accuracy
            with torch.no_grad():
                y_pred_train_np = outputs.cpu().numpy()
                y_train_np = y_batch.cpu().numpy()

                # Inverse transform predictions and actual values
                # y_pred_train_inv = scalerY.inverse_transform(y_pred_train_np)
                # y_train_inv = scalerY.inverse_transform(y_train_np)

                # Compute evaluation metrics
                mae_train += mean_absolute_error(y_train_np, y_pred_train_np)
                mse_train += mean_squared_error(y_train_np, y_pred_train_np)
                rmse_train += np.sqrt(mse_train)

            if idx == cfg["early_stopping"]:
                break

        train_loss /= len(train_loader.dataset)
        acc_train /= len(train_loader.dataset)
        mae_train /= len(train_loader.dataset)
        mse_train /= len(train_loader.dataset)
        rmse_train /= len(train_loader.dataset)

        train_loss_arr.append(train_loss)
        train_mae_arr.append(mae_train)
        train_mse_arr.append(mse_train)
        train_rmse_arr.append(rmse_train)

        # Validation phase
        gru_model_energy.eval()
        val_loss = 0.0
        mae_val = 0.0
        mse_val = 0.0
        rmse_val = 0.0
        acc_val = 0.0

        h0 = None  # Initialize hidden and cell states

        with torch.no_grad():
            for idx, (X_batch, y_batch) in enumerate(val_loader):
                logging.info(f"Validation progress : {idx}/{len(val_loader)}")
                if X_batch.size(0) != batch_size:
                    continue
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs, h0 = gru_model_energy(X_batch, h0)

                # Detach hidden and cell states to prevent backpropagation through the entire sequence
                h0 = h0.detach()

                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                acc_val += (outputs == y_batch).type(torch.float).sum().item()

                # Compute validation accuracy
                with torch.no_grad():
                    y_pred_val = outputs
                    y_pred_val_np = y_pred_val.cpu().numpy()
                    y_val_np = y_batch.cpu().numpy()

                    # Inverse transform predictions and actual values
                    # y_pred_val_inv = scalerY.inverse_transform(y_pred_val_np)
                    # y_val_inv = scalerY.inverse_transform(y_val_np)

                    # Compute evaluation metrics
                    mae_val += mean_absolute_error(y_val_np, y_pred_val_np)
                    mse_val += mean_squared_error(y_val_np, y_pred_val_np)
                    rmse_val += np.sqrt(mse_val)

                if idx == cfg["early_stopping"]:
                    break

        val_loss /= len(val_loader.dataset)
        acc_val /= len(val_loader.dataset)
        mae_val /= len(val_loader.dataset)
        mse_val /= len(val_loader.dataset)
        rmse_val /= len(val_loader.dataset)

        val_loss_arr.append(val_loss)
        val_mae_arr.append(mae_val)
        val_mse_arr.append(mse_val)
        val_rmse_arr.append(rmse_val)

        logging.info(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f} \n \
            Training MAE: {mae_train:.4f}, Validation MAE: {mae_val:.4f}, Training RMSE: {rmse_train:.4f}, Validation RMSE: {rmse_val:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_mae": mae_train,
                "val_mae": mae_val,
                "train_rmse": rmse_train,
                "val_rmse": rmse_val,
            }
        )

    # 5. Calculate metrics

    # Save metrics to the output artifact
    with open(metrics_energy.path, "w") as f:
        f.write(f"Train Loss: {np.mean(train_loss_arr):.4f}\n")
        f.write(f"Validation Loss: {np.mean(val_loss_arr):.4f}\n")
        f.write(f"Train MSE: {np.mean(train_mse_arr):.4f}\n")
        f.write(f"Validation MSE: {np.mean(val_mse_arr):.4f}\n")
        f.write(f"Train MAE: {np.mean(train_mae_arr):.4f}\n")
        f.write(f"Validation MAE: {np.mean(val_mae_arr):.4f}\n")
        f.write(f"Train RMSE: {np.mean(train_rmse_arr):.4f}\n")
        f.write(f"Validation RMSE: {np.mean(val_rmse_arr):.4f}\n")
    logging.info(f"Metrics saved to: {metrics_energy.path}")

    # 6. Save the model
    # Save the PyTorch model weights to a .pth file
    torch.save(gru_model_energy.state_dict(), model_energy.path)
    logging.info(f"Model weights saved to: {model_energy.path}")
    logging.info(f"Validation Loss: {np.mean(val_loss_arr):.4f}")
    logging.info(f"Validation MAE: {np.mean(val_mae_arr):.4f}")
    logging.info(f"Validation MSE: {np.mean(val_mse_arr):.4f}")
    logging.info(f"Validation RMSE: {np.mean(val_rmse_arr):.4f}")

    wandb.finish()
    logging.info("Training completed successfully.")
