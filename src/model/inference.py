"""
Stores a model serve class that will be used to make predictions with
the trained model.
"""

import os

import joblib
import mlflow
import numpy as np
from loguru import logger

from ..config.aws import aws_credentials
from ..config.model import model_settings
from ..config.settings import general_settings
from ..data.utils import load_feature

label_encoder = load_feature(
    path=general_settings.ARTIFACTS_PATH, feature_name="label_ohe"
)

# if aws_credentials.EC2 != "YOUR_EC2_INSTANCE_URL":
#     mlflow.set_tracking_uri(f"http://{aws_credentials.EC2}:5000")
# else:
#     mlflow.set_tracking_uri("http://mlflow:5000")


class ModelServe:
    """The trained model's class."""

    def __init__(
        self,
        model_name: str,
        model_flavor: str,
        model_version: str,
    ) -> None:
        """Model's instance initializer.

        Args:
            model_name (str): the model's name.
            model_flavor (str): the model's MLflow flavor.
            model_version (str): the model's version.
        """
        self.model_name = model_name
        self.model_flavor = model_flavor
        self.model_version = model_version
        self.model = None

    @logger.catch
    def load(self) -> None:
        """Loads the trained model.

        Raises:
            NotImplementedError: raises NotImplementedError if the model's
                flavor value is not 'lightbm'.
        """
        # logger.info(
        #     f"Loading the model {model_settings.MODEL_NAME} from run ID {model_settings.RUN_ID}."
        # )

        # if self.model_flavor == "lightgbm":
        # model_uri = f"runs:/{model_settings.RUN_ID}/{model_settings.MODEL_NAME}"
        # # model_uri = "models:/experimentation-best-model/None"
        # self.model = mlflow.pyfunc.load_model(model_uri)
        self.model = joblib.load(
            os.path.join(general_settings.ARTIFACTS_PATH, "obesity-pred-model.pkl")
        )
        # else:
        #     logger.critical(
        #         f"Couldn't load the model using the flavor {model_settings.MODEL_FLAVOR}."
        #     )
        #     raise NotImplementedError()

    def predict(
        self, features: np.ndarray, transform_to_str: bool = True
    ) -> np.ndarray:
        """Uses the trained model to make a prediction on a given feature array."""
        prediction = self.model.predict(features)

        if transform_to_str:
            # For decision tree models that return class indices directly
            if isinstance(prediction, np.ndarray) and prediction.dtype in [
                np.int32,
                np.int64,
            ]:
                prediction = label_encoder.inverse_transform(prediction)
            else:
                # Convert one-hot or probability predictions to class indices
                n_classes = len(label_encoder.classes_)
                one_hot = np.zeros((len(prediction), n_classes))
                one_hot[np.arange(len(prediction)), prediction.astype(int)] = 1
                prediction = label_encoder.inverse_transform(one_hot)

        logger.info(f"Prediction: {prediction}.")
        return prediction
