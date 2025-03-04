import os

import joblib
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler


def read_yaml_file(path, file):
    # reading credentials files
    with open(f"{os.path.join(path, file)}") as f:
        try:
            content = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise e

    return content


CONFIG_PATH = os.path.join("src", "config")

# credentials_config = read_yaml_file(path=CONFIG_PATH, file="credentials.yaml")

general_settings = read_yaml_file(path=CONFIG_PATH, file="settings.yaml")

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

df = pd.read_csv(RAW_FILE_PATH, sep=",")
df = df.drop(columns=["id"])
logger.info(f"Dataset shape: {df.shape}")

df = df.drop_duplicates(keep="first")
logger.info(f"Dataset shape: {df.shape}")

df["Height"] *= 100

# calculating the upper and lower limits
Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)
threshold = 3.5
IQR = Q3 - Q1

logger.info(f"Dataset shape before removing the outliers: {df.shape}")

# removing the data samples that exceeds the upper or lower limits
df = df[
    ~((df["Age"] >= (Q3 + threshold * IQR)) | (df["Age"] <= (Q1 - threshold * IQR)))
]
logger.info(f"Dataset shape after removing the outliers: {df.shape}")

df["BMI"] = df["Weight"] / (df["Height"] ** 2)
df["PAL"] = df["FAF"] - df["TUE"]


def calculate_bsa(gender: str, height: float, weight: float) -> float:
    # Schlich formula
    if gender == "Female":
        return 0.000975482 * (weight**0.46) * (height**1.08)

    return 0.000579479 * (weight**0.38) * (height**1.24)


df["BSA"] = df.apply(
    lambda x: calculate_bsa(x["Gender"], x["Height"], x["Weight"]), axis=1
)


def calculate_ibw(gender: str, height: float) -> float:
    # B. J. Devine formula
    if gender == "Female":
        return 45.5 + 0.9 * (height - 152)

    return 50 + 0.9 * (height - 152)


df["IBW"] = df.apply(lambda x: calculate_ibw(x["Gender"], x["Height"]), axis=1)
df["diff_W_IBW"] = df["Weight"] - df["IBW"]


def calculate_bmr(age: int, gender: str, height: float, weight: float) -> float:
    s = -161 if gender == "Female" else 5
    return (10 * weight) + (6.25 * height) - (5 * age) + s


df["BMR"] = df.apply(
    lambda x: calculate_bmr(x["Age"], x["Gender"], x["Height"], x["Weight"]), axis=1
)


def calculate_tdee(bmr: float, activity: float) -> float:
    if activity == 0:
        return bmr * 1.2
    elif activity < 1:
        return bmr * 1.55
    elif activity > 1 and activity <= 2:
        return bmr * 1.725
    else:
        return bmr * 1.9


df["TDEE"] = df.apply(lambda x: calculate_tdee(x["BMR"], x["FAF"]), axis=1)

df["SWC"] = df["CH2O"] > ((df["Weight"] / 2) * 0.0295735)  # converting onces to liters
df["SWC"] = df["SWC"].astype(int)

df["IS"] = df["FAF"] <= 1
df["IS"] = df["IS"].astype(int)


def calculate_healthy_habits(row: pd.DataFrame) -> float:

    eat_healthy = -1 if (row["FCVC"] * row["NCP"]) < 3 else 1
    is_sedentary = -1 if row["FAF"] <= 1 else 1
    is_smoker = -1 if row["SMOKE"] == "yes" else 1
    sufficient_water_consumption = (
        -1 if (row["CH2O"] < ((row["Weight"] / 2) * 0.0295735)) else 1
    )
    drink_frequently = (
        -1 if (row["CALC"] == "Always" or row["CALC"] == "Frequently") else 1
    )
    active_person = -1 if (row["TUE"] - row["FAF"]) > 0 else 1
    is_overweight = (
        -1 if (row["Height"] - calculate_ibw(row["Age"], row["Height"])) > 0 else 1
    )

    return (
        eat_healthy
        + is_sedentary
        + is_smoker
        + sufficient_water_consumption
        + drink_frequently
        + active_person
        + is_overweight
    )


df["HH"] = df.apply(lambda x: calculate_healthy_habits(x), axis=1)

df["INMM"] = df["NCP"] == 3
df["INMM"] = df["INMM"].astype(int)

df["EVEMM"] = df["FCVC"] >= df["NCP"]
df["EVEMM"] = df["EVEMM"].astype(int)


values, bins = pd.qcut(x=df["Age"], q=4, retbins=True, labels=["q1", "q2", "q3", "q4"])
bins = np.concatenate(([-np.inf], bins[1:-1], [np.inf]))

# Transforming `Age` Column Into a Categorical Column
df["Age"] = values
df["Age"] = df["Age"].astype("object")

# Transforming `IS`, `SWC`, `EVEMM`, `INMM` into Categorical Columns
df["SWC"] = df["SWC"].astype("object")
df["IS"] = df["IS"].astype("object")
df["EVEMM"] = df["EVEMM"].astype("object")
df["INMM"] = df["INMM"].astype("object")

# Transforming `HH` Column Into a Categorical Column
df["HH"] = df["HH"].astype(int)
df["HH"] = pd.qcut(x=df["HH"], q=3, labels=["bad", "ok", "good"])
df["HH"] = df["HH"].astype("object")

# Splitting the Data into Training and Validation Sets
X = df.drop(columns=["NObeyesdad"])
y = df["NObeyesdad"].values

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED
)

X_train = X_train.reset_index(drop=True)
X_valid = X_valid.reset_index(drop=True)

logger.info(f"Train set shape: {X_train.shape} and {y_train.shape}")
logger.info(f"Validation set shape: {X_valid.shape} and {y_valid.shape}")

# Transforming the Numerical Columns (Log Transformation)
numerical_columns = df.select_dtypes(exclude="object").columns.tolist()
epsilon = 1e-10

for nc in numerical_columns:
    if not nc in ["diff_W_IBW", "PAL"]:
        X_train[nc] = np.log(X_train[nc].values + epsilon)
        X_valid[nc] = np.log(X_valid[nc].values + epsilon)

logger.info("Training set skewness before scaling:")
logger.info(X_train[numerical_columns].skew())
logger.info("Validation set skewness before scaling:")
logger.info(X_valid[numerical_columns].skew())

scalers = {}

for nc in numerical_columns:
    sc = StandardScaler()
    X_train[nc] = sc.fit_transform(X_train[nc].values.reshape(-1, 1))
    X_valid[nc] = sc.transform(X_valid[nc].values.reshape(-1, 1))
    scalers[nc] = sc

logger.info("Training set skewness after scaling:")
logger.info(X_train[numerical_columns].skew())
logger.info("Validation set skewness after scaling:")
logger.info(X_valid[numerical_columns].skew())

# Encoding the Categorical Columns
# plotting categorical columns distributions
categorical_columns = df.select_dtypes(include="object").columns.tolist()
target_column = "NObeyesdad"
categorical_columns.remove(target_column)

new_train_df = pd.DataFrame()
new_valid_df = pd.DataFrame()

encoders = {}

for cc in categorical_columns:
    ohe = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="infrequent_if_exist",
        min_frequency=20,
    )

    train_categorical_features = pd.DataFrame(
        ohe.fit_transform(X_train[cc].values.reshape(-1, 1)),
        columns=ohe.get_feature_names_out(),
    )
    train_categorical_features = train_categorical_features.add_prefix(cc + "_")
    new_train_df = pd.concat([new_train_df, train_categorical_features], axis=1)

    valid_categorical_features = pd.DataFrame(
        ohe.transform(X_valid[cc].values.reshape(-1, 1)),
        columns=ohe.get_feature_names_out(),
    )
    valid_categorical_features = valid_categorical_features.add_prefix(cc + "_")
    new_valid_df = pd.concat([new_valid_df, valid_categorical_features], axis=1)

    encoders[cc] = ohe

new_train_df = pd.concat(
    [new_train_df, X_train.drop(columns=categorical_columns)], axis=1
)
new_valid_df = pd.concat(
    [new_valid_df, X_valid.drop(columns=categorical_columns)], axis=1
)

X_train = new_train_df.values.copy()
X_valid = new_valid_df.values.copy()

# Encoding the labels
ohe_label = LabelBinarizer(sparse_output=False)

original_y_train = y_train.copy()
original_y_valid = y_valid.copy()

y_train = ohe_label.fit_transform(y_train.reshape(-1, 1))
y_valid = ohe_label.transform(y_valid.reshape(-1, 1))

logger.info(f"Train set shape: {X_train.shape} and {y_train.shape}")
logger.info(f"Validation set shape: {X_valid.shape} and {y_valid.shape}")

# Saving the artifacts
# saving the artifacts locally
os.makedirs(ARTIFACTS_OUTPUT_PATH, exist_ok=True)
os.makedirs(FEATURES_OUTPUT_PATH, exist_ok=True)

joblib.dump(scalers, os.path.join(ARTIFACTS_OUTPUT_PATH, "features_sc.pkl"))
joblib.dump(encoders, os.path.join(ARTIFACTS_OUTPUT_PATH, "features_ohe.pkl"))
joblib.dump(ohe_label, os.path.join(ARTIFACTS_OUTPUT_PATH, "label_ohe.pkl"))
joblib.dump(bins, os.path.join(ARTIFACTS_OUTPUT_PATH, "qcut_bins.pkl"))

joblib.dump(X_train, os.path.join(FEATURES_OUTPUT_PATH, "X_train.pkl"))
joblib.dump(y_train, os.path.join(FEATURES_OUTPUT_PATH, "y_train.pkl"))
joblib.dump(X_valid, os.path.join(FEATURES_OUTPUT_PATH, "X_valid.pkl"))
joblib.dump(y_valid, os.path.join(FEATURES_OUTPUT_PATH, "y_valid.pkl"))

# saving the preprocessed dataset locally
new_train_df[target_column] = original_y_train
new_valid_df[target_column] = original_y_valid

preprocessed_data = pd.concat([new_train_df, new_valid_df])
preprocessed_data.to_csv(PROCESSED_RAW_FILE_PATH, index=False, sep=",")
