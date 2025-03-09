import ast
import math
import os
import shutil
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Union

import boto3
import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Get the current datetime string for experiment names
current_datetime = datetime.now()
datetime_string = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")


def read_yaml_file(path, file):
    # reading credentials files
    with open(f"{os.path.join(path, file)}") as f:
        try:
            content = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise e

    return content


CONFIG_PATH = os.path.join("src", "config")

general_settings = read_yaml_file(path=CONFIG_PATH, file="settings.yaml")

mlflow.set_tracking_uri(f"http://mlflow:5000")

print(f"Tracking Server URI: '{mlflow.get_tracking_uri()}'")

SEED = 42
ARTIFACTS_OUTPUT_PATH = general_settings["ARTIFACTS_PATH"]
FEATURES_OUTPUT_PATH = general_settings["FEATURES_PATH"]
RAW_FILE_PATH = os.path.join(
    general_settings["DATA_PATH"], general_settings["RAW_FILE_NAME"]
)
PROCESSED_RAW_FILE = "Preprocessed_" + general_settings["RAW_FILE_NAME"]
PROCESSED_RAW_FILE_PATH = os.path.join(
    general_settings["DATA_PATH"], PROCESSED_RAW_FILE
)
FEATURE_SELECTION_EXPERIMENT_NAME = f"feature-selection-experimentation"
HYPERPARAMETER_TUNING_EXPERIMENT_NAME = f"hyperparameters-tuning-experimentation"

# loading features
X_train = joblib.load(os.path.join(FEATURES_OUTPUT_PATH, "X_train.pkl"))
y_train = joblib.load(os.path.join(FEATURES_OUTPUT_PATH, "y_train.pkl"))

X_valid = joblib.load(os.path.join(FEATURES_OUTPUT_PATH, "X_valid.pkl"))
y_valid = joblib.load(os.path.join(FEATURES_OUTPUT_PATH, "y_valid.pkl"))

# loading artifacts
sc = joblib.load(os.path.join(ARTIFACTS_OUTPUT_PATH, "features_sc.pkl"))
ohe = joblib.load(os.path.join(ARTIFACTS_OUTPUT_PATH, "features_ohe.pkl"))
ohe_label = joblib.load(os.path.join(ARTIFACTS_OUTPUT_PATH, "label_ohe.pkl"))

# loading feature columns
temp_df = pd.read_csv(PROCESSED_RAW_FILE_PATH, sep=",")
FEATURES_NAME = temp_df.columns.tolist()
del temp_df

X_train = X_train[:1000, :10]
y_train = y_train[:1000]
X_valid = X_valid[:200, :10]
y_valid = y_valid[:200]


class Objective:
    def __init__(
        self,
        run_group_name: str,
        experiment_id: str,
        X_train: np.ndarray,
        y_train: np.array,
        X_valid: np.ndarray,
        y_valid: np.array,
        indexes: List,
    ) -> None:
        self.run_group_name = run_group_name
        self.experiment_id = experiment_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.indexes_name = indexes
        self.indexes = [FEATURES_NAME.index(i) for i in indexes]

        if self.run_group_name in ["decision_tree", "lightgbm", "catboost"]:
            self.y_train = np.argmax(self.y_train, axis=1)
            self.y_valid = np.argmax(self.y_valid, axis=1)

        self.X_train = self.X_train[:, self.indexes]
        self.X_valid = self.X_valid[:, self.indexes]

    def __call__(self, trial: optuna.trial.Trial) -> float:
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=f"{self.run_group_name}_trial_{trial.number}",
            nested=True,
        ) as run:
            trial.set_user_attr("run_id", run.info.run_id)
            trial.set_user_attr("run_name", run.info.run_name)
            if self.run_group_name == "decision_tree":
                params = {
                    "max_depth": trial.suggest_int("max_depth", 2, 32, step=2),
                    "min_samples_split": trial.suggest_int(
                        "min_samples_split", 2, 8, step=1
                    ),
                    "min_samples_leaf": trial.suggest_int(
                        "min_samples_leaf", 1, 6, step=1
                    ),
                    "min_weight_fraction_leaf": trial.suggest_float(
                        "min_weight_fraction_leaf", 0, 0.5, step=0.1
                    ),
                    "max_leaf_nodes": trial.suggest_int(
                        "max_leaf_nodes", 2, 16, step=2
                    ),
                    "random_state": SEED,
                }
                model = DecisionTreeClassifier(**params)

            if self.run_group_name == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "max_depth": trial.suggest_int("max_depth", 10, 50),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 32),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 32),
                    "random_state": SEED,
                    "n_jobs": -1,
                }
                model = RandomForestClassifier(**params)

            if self.run_group_name == "xgboost":
                params = {
                    "booster": trial.suggest_categorical(
                        "booster", ["gbtree", "gblinear", "dart"]
                    ),
                    "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                    "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                    "random_state": SEED,
                    "n_jobs": -1,
                }
                model = XGBClassifier(**params)

            if self.run_group_name == "lightgbm":
                params = {
                    "objective": "multiclass",
                    "verbosity": -1,
                    "random_state": SEED,
                    "n_jobs": -1,
                    "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                    "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                    "feature_fraction": trial.suggest_float(
                        "feature_fraction", 0.4, 1.0
                    ),
                    "bagging_fraction": trial.suggest_float(
                        "bagging_fraction", 0.4, 1.0
                    ),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                }
                model = LGBMClassifier(**params)

            if self.run_group_name == "catboost":
                params = {
                    "random_seed": SEED,
                    "verbose": 0,
                    "allow_writing_files": False,
                    "colsample_bylevel": trial.suggest_float(
                        "colsample_bylevel", 0.01, 0.1
                    ),
                    "depth": trial.suggest_int("depth", 1, 12),
                    "boosting_type": trial.suggest_categorical(
                        "boosting_type", ["Ordered", "Plain"]
                    ),
                    "bootstrap_type": trial.suggest_categorical(
                        "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
                    ),
                }
                model = CatBoostClassifier(**params)

            model.fit(X=self.X_train, y=self.y_train)

            # calculating the training f1 score
            train_prediction = model.predict(self.X_train)
            train_f1 = f1_score(
                y_true=self.y_train, y_pred=train_prediction, average="weighted"
            )

            # calculating the validation f1 score
            valid_prediction = model.predict(self.X_valid)
            valid_f1 = f1_score(
                y_true=self.y_valid, y_pred=valid_prediction, average="weighted"
            )

            # logging the training and validation scores
            mlflow.log_metric("train_f1", train_f1)
            mlflow.log_metric("valid_f1", valid_f1)

            # inferring the signature of the trained model
            signature = infer_signature(
                model_input=self.X_train, model_output=train_prediction
            )

            # saving the trained model
            if self.run_group_name in ["decision_tree", "random_forest"]:
                # sklearn flavor
                mlflow.sklearn.log_model(
                    model, self.run_group_name, signature=signature
                )
                # logging the model"s default parameters
                mlflow.log_params(model.get_params(deep=True))
            elif self.run_group_name == "xgboost":
                mlflow.xgboost.log_model(
                    model, self.run_group_name, signature=signature
                )
                # logging the model's default parameters
                mlflow.log_params(model.get_params())
            elif self.run_group_name == "lightgbm":
                mlflow.lightgbm.log_model(
                    model, self.run_group_name, signature=signature
                )
                # logging the model's default parameters
                mlflow.log_params(model.get_params())
            elif self.run_group_name == "catboost":
                mlflow.catboost.log_model(
                    model, self.run_group_name, signature=signature
                )
                # logging the model's default parameters
                mlflow.log_params(model.get_all_params())

        return valid_f1


# creating a new mlflow's experiment
hpt_experiment_id = mlflow.create_experiment(
    name=HYPERPARAMETER_TUNING_EXPERIMENT_NAME + "_" + datetime_string,
)


def get_latest_model_params(model_name):
    client = mlflow.tracking.MlflowClient()
    latest_version_info = client.get_latest_versions(model_name)[0]
    run_id = latest_version_info.run_id
    run = client.get_run(run_id)

    # Get the parameters logged for the run
    parameters = run.data.params
    return parameters


def run_mlflow_experiment(run_group_name, experiment_id, direction="maximize"):
    dt_run_params = get_latest_model_params(
        FEATURE_SELECTION_EXPERIMENT_NAME + "_" + run_group_name
    )
    dt_features_indexes = ast.literal_eval(dt_run_params["features"])

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_group_name):
        objective = Objective(
            run_group_name=run_group_name,
            experiment_id=experiment_id,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            indexes=dt_features_indexes,
        )

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=10)

        return study.trials_dataframe()


### Analyze all trial results, and register best model

all_trials_dfs = []
run_group_names = ["decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]
for run_group_name in run_group_names:
    trials_df = run_mlflow_experiment(
        run_group_name, hpt_experiment_id, direction="maximize"
    )
    trials_df["run_group_name"] = run_group_name
    all_trials_dfs.append(trials_df)

df_trials = pd.concat(all_trials_dfs, axis=0)
best_trial_id = df_trials["value"].idxmax()
best_trial = df_trials.iloc[best_trial_id]

results = mlflow.register_model(
    model_uri=f"runs:/{best_trial['user_attrs_run_id']}/{best_trial['user_attrs_run_name']}",
    name=f"experimentation-best-model",
)

print(results)

df_trials.to_csv(
    os.path.join(ARTIFACTS_OUTPUT_PATH, "hyperparameters_tuning_results.csv")
)
# Update config/model.yaml with best_trial['user_attrs_run_id'] and hpt_experiment_id
model_config_path = os.path.join(CONFIG_PATH, "model.yaml")
with open(model_config_path, "r") as file:
    model_config = yaml.safe_load(file)

model_config["RUN_ID"] = best_trial["user_attrs_run_id"]
model_config["EXPERIMENT_ID"] = hpt_experiment_id
model_config["MODEL_NAME"] = best_trial["run_group_name"]
model_config["MODEL_FLAVOR"] = best_trial["run_group_name"]
model_config["VERSION"] = results.version


with open(model_config_path, "w") as file:
    yaml.safe_dump(model_config, file)
