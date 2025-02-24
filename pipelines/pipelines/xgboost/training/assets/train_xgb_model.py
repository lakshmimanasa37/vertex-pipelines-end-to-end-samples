import argparse
import joblib
import json
import os
import logging

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from xgboost import XGBRegressor

logging.basicConfig(level=logging.DEBUG)


NUM_COLS = ["start_station_id", "end_station_id"]
OHE_COLS = ["is_weekend"]


def split_xy(df: pd.DataFrame, label: str) -> (pd.DataFrame, pd.Series):
    """Split dataframe into X and y."""
    return df.drop(columns=[label]), df[label]


def indices_in_list(elements: list, base_list: list) -> list:
    """Get indices of specific elements in a base list"""
    return [idx for idx, elem in enumerate(base_list) if elem in elements]


parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type=str, required=True)
parser.add_argument("--valid_data", type=str, required=True)
parser.add_argument("--test_data", type=str, required=True)
parser.add_argument("--model", default=os.getenv("AIP_MODEL_DIR"), type=str, help="")
parser.add_argument("--metrics", type=str, required=True)
parser.add_argument("--hparams", default={}, type=json.loads)
args = parser.parse_args()

logging.info("Read csv files into dataframes")
df_train = pd.read_csv(args.train_data)
df_valid = pd.read_csv(args.valid_data)
df_test = pd.read_csv(args.test_data)

logging.info("Split dataframes")
label = args.hparams["label"]
X_train, y_train = split_xy(df_train, label)
X_valid, y_valid = split_xy(df_valid, label)
X_test, y_test = split_xy(df_test, label)


logging.info("Get indices of columns in base data")
col_list = X_train.columns.tolist()
num_indices = indices_in_list(NUM_COLS, col_list)
cat_indices_onehot = indices_in_list(OHE_COLS, col_list)

all_transformers = [
    (
        "one_hot_encoding",
        OneHotEncoder(handle_unknown="ignore"),
        cat_indices_onehot,
    ),
] 

logging.info("Build sklearn preprocessing steps")
preprocesser = ColumnTransformer(transformers=all_transformers)
logging.info("Build sklearn pipeline with XGBoost model")
xgb_model = XGBRegressor(**args.hparams)

pipeline = Pipeline(
    steps=[("feature_engineering", preprocesser), ("train_model", xgb_model)]
)

logging.info("Transform validation data")
valid_preprocesser = preprocesser.fit(X_train)
X_valid_transformed = valid_preprocesser.transform(X_valid)

logging.info("Fit model")
pipeline.fit(X_train, y_train, train_model__eval_set=[(X_valid_transformed, y_valid)])

logging.info("Predict test data")
y_pred = pipeline.predict(X_test)
y_pred = y_pred.clip(0)

metrics = {
    "problemType": "regression",
    "rootMeanSquaredError": np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    "meanAbsoluteError": metrics.mean_absolute_error(y_test, y_pred),
    "meanAbsolutePercentageError": metrics.mean_absolute_percentage_error(
        y_test, y_pred
    ),
    "rSquared": metrics.r2_score(y_test, y_pred),
    "rootMeanSquaredLogError": np.sqrt(metrics.mean_squared_log_error(y_test, y_pred)),
}

try:
    model_path = args.model.replace("gs://", "/gcs/")
    logging.info(f"Save model to: {model_path}")
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(pipeline, model_path + "model.joblib")
except Exception as e:
    print(e)
    raise e

logging.info(f"Metrics: {metrics}")
with open(args.metrics, "w") as fp:
    json.dump(metrics, fp)
