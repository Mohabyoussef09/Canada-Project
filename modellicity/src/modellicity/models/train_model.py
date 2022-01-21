"""@copyright Copyright © 2020, Modellicity Inc., All Rights Reserved."""
from sklearn.linear_model import LogisticRegression

import pandas as pd


class TrainModel:
    """Given data and a target, train the model."""

    def __init__(
            self,
            data_x: pd.DataFrame,
            target_y: pd.Series,
            model_type: str = "logistic_regression",
    ) -> None:
        """
        Train a user-specified model on input data.

        :param data_x: The data to be processed.
        :param target_y: The target to predict.
        :param model_type: A string specifying the model to train on.
        """
        self._data_x = data_x
        self._target_y = target_y
        self._model_type = model_type

    def train_model(self):
        """
        Specify the type of model we want to train on.

        :return: The trained model of `model_type`.
        """
        if self._model_type == "logistic_regression":
            return LogisticRegression(penalty="none", random_state=0).fit(
                self._data_x, self._target_y
            )
        else:
            raise ValueError(f"Model type {self._model_type} not supported.")
