import math
import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Union

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from loguru import logger
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

# if credentials_config["EC2"] != "YOUR_EC2_INSTANCE_URL":
#     mlflow.set_tracking_uri(f"http://{credentials_config['EC2']}:5000")
# else:
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

# Creating subsets of data to make the feature selection process faster
X_train = X_train[:1000, :10]
y_train = y_train[:1000]
X_valid = X_valid[:200, :10]
y_valid = y_valid[:200]

# creating the baseline models
dt = DecisionTreeClassifier(random_state=SEED)
rf = RandomForestClassifier(random_state=SEED, verbose=0, n_jobs=-1)
xg = XGBClassifier(random_state=SEED, n_jobs=-1)
lg = LGBMClassifier(random_state=SEED, verbose=-1, objective="multiclass")
cb = CatBoostClassifier(random_seed=SEED, verbose=0, allow_writing_files=False)


def apply_feature_selection(
    model: Union[
        DecisionTreeClassifier,
        RandomForestClassifier,
        XGBClassifier,
        LGBMClassifier,
        CatBoostClassifier,
    ],
    number_features: int,
    X_train: np.ndarray,
    y_train: np.array,
    X_valid: np.ndarray,
    y_valid: np.array,
) -> Dict:
    # initializing and fitting the sfs class
    sfs = SequentialFeatureSelector(model, n_features_to_select=number_features, cv=3)
    sfs.fit(X=X_train, y=y_train)

    # getting the indexes of the best features
    selected_features_indexes = np.argwhere(sfs.get_support()).reshape(-1)

    reduced_X_train = sfs.transform(X_train)
    reduced_X_valid = sfs.transform(X_valid)

    # training the model
    model.fit(reduced_X_train, y_train)

    # calculating the training f1 score
    predicted_y_train = model.predict(reduced_X_train)
    train_f1 = f1_score(y_true=y_train, y_pred=predicted_y_train, average="weighted")

    # calculating the validation f1 score
    predicted_y_valid = model.predict(reduced_X_valid)
    valid_f1 = f1_score(y_true=y_valid, y_pred=predicted_y_valid, average="weighted")

    # inferring the signature of the trained model
    signature = infer_signature(
        model_input=reduced_X_train, model_output=predicted_y_train
    )

    # saving the metrics and artifacts that we want to log in mlflow
    selected_features_names = list(
        map(lambda i: FEATURES_NAME[i], selected_features_indexes.tolist())
    )

    results = {
        "train_f1": train_f1,
        "valid_f1": valid_f1,
        "features": selected_features_names,
        "model": model,
        "model_signature": signature,
    }

    return results


def set_configurations_mlflow(
    model: Union[
        DecisionTreeClassifier,
        RandomForestClassifier,
        XGBClassifier,
        LGBMClassifier,
        CatBoostClassifier,
    ],
    y_train: np.array,
    y_valid: np.array,
) -> Tuple[np.array, np.array, str, str]:
    # reshaping the target values (if needed) and setting the run name and which
    # flavor is being used for each machine learning model
    if isinstance(model, DecisionTreeClassifier):
        y_train = np.argmax(y_train, axis=1)
        y_valid = np.argmax(y_valid, axis=1)
        run_name = "decision_tree"
        flavor = "sklearn"

    if isinstance(model, RandomForestClassifier):
        run_name = "random_forest"
        flavor = "sklearn"

    if isinstance(model, XGBClassifier):
        run_name = "xgboost"
        flavor = "xgboost"

    if isinstance(model, LGBMClassifier):
        y_train = np.argmax(y_train, axis=1)
        y_valid = np.argmax(y_valid, axis=1)
        run_name = "lightgbm"
        flavor = "lightgbm"

    if isinstance(model, CatBoostClassifier):
        y_train = np.argmax(y_train, axis=1)
        y_valid = np.argmax(y_valid, axis=1)
        run_name = "catboost"
        flavor = "catboost"

    # disabling some options of the current flavor's autolog
    if flavor == "sklearn":
        mlflow.sklearn.autolog(
            log_models=False,
            log_post_training_metrics=False,
            log_model_signatures=False,
            log_input_examples=True,
            log_datasets=False,
            silent=True,
            disable=True,
        )
    elif flavor == "xgboost":
        mlflow.xgboost.autolog(
            log_models=False,
            log_model_signatures=False,
            log_input_examples=True,
            log_datasets=False,
            silent=True,
            disable=True,
        )
    elif flavor == "lightgbm":
        mlflow.lightgbm.autolog(
            log_models=False,
            log_model_signatures=False,
            log_input_examples=True,
            log_datasets=False,
            silent=True,
            disable=True,
        )
    elif flavor == "catboost":
        # there is no autolog implemented for catboost
        pass

    return y_train, y_valid, run_name, flavor


def run_feature_selection_experiment(
    models: List,
    min_features: int,
    max_features: int,
    experiment_id: str,
    metric_to_optimize: str = "valid_f1",
    direction: str = "max",
) -> None:
    for model in models:
        # reshaping the target values (if needed) and setting some mlflow's configuration
        new_y_train, new_y_valid, run_name, flavor = set_configurations_mlflow(
            model=model, y_train=y_train, y_valid=y_valid
        )
        model_type_results = []

        # starting a new run for the current model
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            logger.info(f"Starting the run for the {run_name} model!\n")

            for i, n_features in enumerate(range(min_features, max_features + 1)):
                # creating a nested run inside the model's main run
                with mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name=f"{run_name}_experiment_{i}",
                    nested=True,
                ) as run:
                    # running the feature selection main function
                    results = apply_feature_selection(
                        model=model,
                        number_features=n_features,
                        X_train=X_train,
                        y_train=new_y_train,
                        X_valid=X_valid,
                        y_valid=new_y_valid,
                    )

                    # logging the trained model
                    if flavor == "sklearn":
                        mlflow.sklearn.log_model(
                            results["model"],
                            run_name,
                            signature=results["model_signature"],
                        )
                        # logging the model"s default parameters
                        mlflow.log_params(results["model"].get_params(deep=True))
                    elif flavor == "xgboost":
                        mlflow.xgboost.log_model(
                            results["model"],
                            run_name,
                            signature=results["model_signature"],
                        )
                        # logging the model's default parameters
                        mlflow.log_params(results["model"].get_params(deep=True))
                    elif flavor == "lightgbm":
                        mlflow.lightgbm.log_model(
                            results["model"],
                            run_name,
                            signature=results["model_signature"],
                        )
                        # logging the model's default parameters
                        mlflow.log_params(results["model"].get_params())
                    elif flavor == "catboost":
                        mlflow.catboost.log_model(
                            results["model"],
                            run_name,
                            signature=results["model_signature"],
                        )
                        # logging the model's default parameters
                        mlflow.log_params(results["model"].get_all_params())

                    # logging the training and validation scores
                    mlflow.log_metric("train_f1", results["train_f1"])
                    mlflow.log_metric("valid_f1", results["valid_f1"])

                    # logging the artifacts (original dataset, features, and encoders objects)
                    mlflow.log_artifact(PROCESSED_RAW_FILE_PATH)
                    mlflow.log_artifact(ARTIFACTS_OUTPUT_PATH)
                    mlflow.log_artifact(FEATURES_OUTPUT_PATH)

                    # logging the indexes of the best features
                    mlflow.log_param("features", results["features"])

                    # add mlflow ids to results object and append to list
                    results["experiment_id"] = experiment_id
                    results["run_name"] = f"{run_name}_experiment_{i}"
                    results["flavor"] = flavor
                    results["run_id"] = run.info.run_id
                    model_type_results.append(results)

        # register the best feature selection in mlflow
        df = pd.DataFrame.from_records(model_type_results)
        if direction == "min":
            best_result = df.loc[df[metric_to_optimize].idxmin()]
        elif direction == "max":
            best_result = df.loc[df[metric_to_optimize].idxmax()]
        else:
            raise NotImplementedError("")
        tags = {"type": "baseline", "model": run_name}
        result = mlflow.register_model(
            model_uri=f"runs:/{best_result['run_id']}/{best_result['run_name']}",
            name=f"{FEATURE_SELECTION_EXPERIMENT_NAME}_{run_name}",
            tags=tags,
        )


models = [dt, rf, xg, lg, cb]
min_features = math.floor(X_train.shape[1] * 0.2)
max_features = math.floor(X_train.shape[1] * 0.5)

# creating a new mlflow's experiment
experiment_id = mlflow.create_experiment(
    name=FEATURE_SELECTION_EXPERIMENT_NAME + "_" + datetime_string,
)

# running the feature selection experiments
run_feature_selection_experiment(
    models=models,
    min_features=min_features,
    max_features=max_features,
    experiment_id=experiment_id,
)
