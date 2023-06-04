import argparse
import os
import json
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision.transforms import Normalize

# used for monitoring during prediction time
TRAINING_DATASET_INFO = "training_dataset.json"
# numeric/categorical features in Chicago trips dataset to be preprocessed
NUM_COLS = ["start_station_id", "end_station_id"]
OHE_COLS = ["is_weekend"]

DEFAULT_HPARAMS = dict(
    batch_size=100,
    epochs=1,
    loss_fn="MSELoss",
    optimizer="Adam",
    learning_rate=0.001,
    metrics=[
        "RMSELoss",
        "L1Loss",
        "L1Loss",
        "MSELoss",
    ],
    hidden_units=[(10, nn.ReLU())],
    distribute_strategy="single",
    early_stopping_epochs=5,
    label="tripduration",
)

logging.getLogger().setLevel(logging.INFO)


class CustomDataset(Dataset):
    def __init__(self, input_data, label_name):
        self.input_data = input_data
        self.label_name = label_name

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx]

    def get_labels(self):
        return self.input_data[self.label_name].values


def create_dataset(input_data, label_name, model_params):
    """Create a PyTorch Dataset from input csv files.
    Args:
        input_data (pd.DataFrame): Train/Valid data in DataFrame format
        label_name (str): Name of column containing the labels
        model_params (dict): model parameters
    Returns:
        dataset (CustomDataset): PyTorch dataset
    """
    logging.info(f"Creating dataset from DataFrame...")
    dataset = CustomDataset(input_data, label_name)
    return dataset


def get_distribution_strategy(distribute_strategy):
    """Set distribute strategy based on input string.
    Args:
        distribute_strategy (str): single, mirror or multi
    Returns:
        strategy (None): No distribution strategy implemented in PyTorch
    """
    logging.info(f"Distribution strategy: {distribute_strategy}")

    if distribute_strategy == "single":
        strategy = None
    elif distribute_strategy == "mirror" or distribute_strategy == "multi":
        raise RuntimeError(f"Distribute strategy: {distribute_strategy} not supported")
    else:
        raise RuntimeError(f"Distribute strategy: {distribute_strategy} not supported")
    return strategy


def normalization(name, dataset):
    """Create a normalization layer for a feature.
    Args:
        name (str): name of feature to be normalized
        dataset (CustomDataset): dataset to adapt layer
    Returns:
        normalization layer (Normalize): adapted normalization layer
    """
    logging.info(f"Normalizing numerical input '{name}'...")
    x = Normalize(dataset.input_data[name].mean(), dataset.input_data[name].std())
    return x


def build_and_compile_model(dataset, model_params):
    """Build and compile model.
    Args:
        dataset (CustomDataset): training dataset
        model_params (dict): model parameters
    Returns:
        model (nn.Module): built and compiled model
    """
    input_size = len(NUM_COLS) + len(OHE_COLS)
    hidden_units = model_params["hidden_units"]

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.normalizations = nn.ModuleList([normalization(name, dataset) for name in NUM_COLS])
            self.embeddings = nn.ModuleList()

            for name in OHE_COLS:
                num_classes = len(dataset.input_data[name].unique())
                embedding = nn.Embedding(num_classes, hidden_units[0][0])
                self.embeddings.append(embedding)

            self.linear1 = nn.Linear(input_size * hidden_units[0][0], hidden_units[0][0])
            self.relu1 = hidden_units[0][1]
            self.linear2 = nn.Linear(hidden_units[0][0], 1)

        def forward(self, x):
            num_inputs = [self.normalizations[i](x[:, i]) for i in range(len(NUM_COLS))]
            ohe_inputs = [self.embeddings[i](x[:, len(NUM_COLS) + i].long()) for i in range(len(OHE_COLS))]

            num_inputs = torch.cat(num_inputs, dim=1)
            ohe_inputs = torch.cat(ohe_inputs, dim=1)

            x = torch.cat((num_inputs, ohe_inputs), dim=1)
            x = self.linear1(x)
            x = self.relu1(x)
            x = self.linear2(x)

            return x

    model = Model()
    logging.info(f"Model:\n{model}")
    logging.info(f"Use optimizer {model_params['optimizer']}")
    optimizer = getattr(optim, model_params["optimizer"])
    optimizer = optimizer(model.parameters(), lr=model_params["learning_rate"])

    criterion = getattr(nn, model_params["loss_fn"])
    metrics = model_params["metrics"]

    return model, optimizer, criterion, metrics


def _is_chief(strategy):
    """Determine whether current worker is the chief (master).
    Args:
        strategy (None): No distributed training in PyTorch
    Returns:
        is_chief (bool): True if worker is chief, otherwise False
    """
    return True


def train(model, optimizer, criterion, dataloader, metrics):
    """Train the model.
    Args:
        model (nn.Module): PyTorch model
        optimizer (optim.Optimizer): PyTorch optimizer
        criterion (nn.Module): PyTorch loss function
        dataloader (DataLoader): PyTorch data loader
        metrics (list): List of metrics to evaluate
    Returns:
        history (dict): Training history
    """
    model.train()
    history = {}

    for epoch in range(hparams["epochs"]):
        running_loss = 0.0

        for batch in dataloader:
            inputs = batch.float()
            labels = inputs[:, -1]  # Assuming label is the last column

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        logging.info(f"Epoch [{epoch+1}/{hparams['epochs']}], Loss: {epoch_loss:.4f}")

        history["loss"] = history.get("loss", []) + [epoch_loss]

    return history


parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type=str, required=True)
parser.add_argument("--valid_data", type=str, required=True)
parser.add_argument("--test_data", type=str, required=True)
parser.add_argument("--model", default=os.getenv("AIP_MODEL_DIR"), type=str, help="")
parser.add_argument("--metrics", type=str, required=True)
parser.add_argument("--hparams", default={}, type=json.loads)
args = parser.parse_args()

hparams = {**DEFAULT_HPARAMS, **args.hparams}
logging.info(f"Using model hyper-parameters: {hparams}")
label = hparams["label"]

strategy = get_distribution_strategy(hparams["distribute_strategy"])

train_data = pd.read_csv(args.train_data)
valid_data = pd.read_csv(args.valid_data)
test_data = pd.read_csv(args.test_data)

train_dataset = create_dataset(train_data, label, hparams)
valid_dataset = create_dataset(valid_data, label, hparams)
test_dataset = create_dataset(test_data, label, hparams)

train_dataloader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=hparams["batch_size"], shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False)

train_features = train_data.columns.tolist()
valid_features = valid_data.columns.tolist()
logging.info(f"Training feature names: {train_features}")
logging.info(f"Validation feature names: {valid_features}")

if len(train_features) != len(valid_features):
    raise RuntimeError(f"No. of training features != # validation features")

torch_model, optimizer, criterion, metrics = build_and_compile_model(train_dataset, hparams)

logging.info("Fit model...")
history = train(torch_model, optimizer, criterion, train_dataloader, metrics)

# only persist output files if current worker is chief
if not _is_chief(strategy):
    logging.info("not chief node, exiting now")
    sys.exit()

os.makedirs(args.model, exist_ok=True)
logging.info(f"Save model to: {args.model}")
torch.save(torch_model.state_dict(), os.path.join(args.model, "model.pt"))

logging.info(f"Save metrics to: {args.metrics}")
eval_metrics = {}

with torch.no_grad():
    for batch in test_dataloader:
        inputs = batch.float()
        labels = inputs[:, -1]  # Assuming label is the last column

        outputs = torch_model(inputs)
        loss = criterion(outputs.squeeze(), labels)

        for metric in metrics:
            if metric == "RMSELoss":
                eval_metrics[metric] = torch.sqrt(loss).item()
            else:
                eval_metrics[metric] = loss.item()

metrics = {
    "problemType": "regression",
    "rootMeanSquaredError": eval_metrics["RMSELoss"],
    "meanAbsoluteError": eval_metrics["L1Loss"],
    "meanAbsolutePercentageError": eval_metrics["L1Loss"],
    "rSquared": None,
    "rootMeanSquaredLogError": eval_metrics["MSELoss"],
}

with open(args.metrics, "w") as fp:
    json.dump(metrics, fp)

# Persist URIs of training file(s) for model monitoring in batch predictions
path = Path(args.model) / TRAINING_DATASET_INFO
training_dataset_for_monitoring = {
    "gcsSource": {"uris": [args.train_data]},
    "dataFormat": "csv",
    "targetField": hparams["label"],
}
logging.info(f"Save training dataset info for model monitoring: {path}")
logging.info(f"Training dataset: {training_dataset_for_monitoring}")

with open(path, "w") as fp:
    json.dump(training_dataset_for_monitoring, fp)
