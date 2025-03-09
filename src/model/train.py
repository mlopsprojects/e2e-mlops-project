import os
import warnings
from datetime import datetime

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

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


CONFIG_PATH = os.path.join("/workspace", "src", "config")

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

TRAINING_EXPERIMENT_NAME = f"training-experimentation"

# loading features
X_train = joblib.load(os.path.join(FEATURES_OUTPUT_PATH, "X_train.pkl"))
y_train = joblib.load(os.path.join(FEATURES_OUTPUT_PATH, "y_train.pkl"))

X_valid = joblib.load(os.path.join(FEATURES_OUTPUT_PATH, "X_valid.pkl"))
y_valid = joblib.load(os.path.join(FEATURES_OUTPUT_PATH, "y_valid.pkl"))

# loading artifacts #TODO check how these are generated
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

# Check if the experiment name already exists
experiment_names = [exp.name for exp in mlflow.search_experiments()]
experiment = mlflow.get_experiment_by_name(TRAINING_EXPERIMENT_NAME)
if experiment is not None:
    experiment_id = experiment.experiment_id
else:
    experiment_id = mlflow.create_experiment(TRAINING_EXPERIMENT_NAME)

# Train DecisionTree classifier model using X_train, y_train with all features and evaluate it using X_valid, y_valid
with mlflow.start_run(experiment_id=experiment_id):
    params = {
        "criterion": "gini",
        "splitter": "best",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": None,
        "random_state": SEED,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "class_weight": None,
    }
    mlflow.log_params(params)
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    f1 = f1_score(y_valid, y_pred, average="weighted")
    print(f"DecisionTreeClassifier f1_score: {f1}")
    signature = infer_signature(X_train, y_pred)
    mlflow.sklearn.log_model(model, "decision_tree_model", signature=signature)
    mlflow.log_metric("f1_score", f1)
    mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/decision_tree_model",
        name="obesity-pred-model",
    )
    # mlflow.register_model(
    #         model_uri=f"runs:/{best_result['run_id']}/{best_result['run_name']}",
    #         name=f"{FEATURE_SELECTION_EXPERIMENT_NAME}_{run_name}",
    #         tags=tags,
    #     )
