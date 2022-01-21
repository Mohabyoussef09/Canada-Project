"""@copyright Copyright © 2020, Modellicity Inc., All Rights Reserved."""
import matplotlib.pyplot as plt
from modellicity.models.train_model import TrainModel
from modellicity.stats.kpi import KPI

import pandas as pd


class Performance:
    """Plot performance of trained model."""

    def __init__(
            self,
            data_x,
            target_y,
            model_type
    ) -> None:
        """
        Utilities for plotting performance.

        :param data_x: The data to be processed.
        :param target_y: The target to predict.
        :param model_type: A string specifying the model to train on.
        """
        self._data_x = data_x
        self._target_y = target_y
        self._model_type = model_type

        self._trained_model = TrainModel(self._data_x,
                                         self._target_y,
                                         self._model_type).train_model()

        self._kpi = KPI(self._data_x, self._target_y, self._trained_model)

        # Obtain model-predicted values.
        self._y_hat_prob = self._trained_model.predict_proba(self._data_x)[:, 1]
        self._y_hat_value = self._trained_model.predict(self._data_x)

    def plot_performance(self) -> None:
        """
        Plot performance of trained model.

        :return: None.
        """
        predictions_df = pd.DataFrame()
        predictions_df["y"] = self._target_y
        predictions_df["y_hat_prob"] = self._y_hat_prob
        predictions_df["y_hat_value"] = self._y_hat_value

        bins = pd.qcut(self._y_hat_prob, 5)
        trend_df = predictions_df.groupby(bins)["y"].agg(["count", "mean"])
        trend_df.reset_index(inplace=True)
        trend_df["index"] = trend_df["index"].astype(str)
        trend_df.rename(columns={"mean": "event_rate"}, inplace=True)
        plt.bar("index", "event_rate", data=trend_df, color="blue")
        plt.xlabel("Probability buckets")
        plt.xticks(rotation=90)
        plt.ylabel("Event rate")
        plt.show()
