"""@copyright Copyright © 2020, Modellicity Inc., All Rights Reserved."""
import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.metrics import f1_score, roc_auc_score
from typing import Dict


class KPI:
    """Calculate KPIs (key performance indicators) for a given dataset."""

    def __init__(
        self,
        data_x: pd.DataFrame,
        target_y: pd.Series,
        trained_model,
    ) -> None:
        """
        KPI metrics.

        :param data_x: The data to be processed.
        :param target_y: The target to predict.
        :param trained_model: The model on which the data was trained.
        :return: None.
        """
        self._data_x = data_x
        self._target_y = target_y
        self._trained_model = trained_model

        # Obtain model-predicted values.
        self._y_hat_prob = self._trained_model.predict_proba(self._data_x)[:, 1]
        self._y_hat_value = self._trained_model.predict(self._data_x)

    def f1(self) -> float:
        """
        Calculate the f1-score KPI on the trained data.

        :return: f1-score of target variable.
        """
        return f1_score(self._target_y, self._y_hat_value)

    def ks(self) -> float:
        """
        Calculate the Kolmogorov-Smirnov score on the trained data.

        :return: KS score of target variable.
        """
        tmp = pd.DataFrame()
        tmp["default"], tmp["pd"] = self._target_y, self._y_hat_prob
        tmp_def = tmp[tmp["default"] == 1]
        tmp_nondef = tmp[tmp["default"] == 0]
        ks = stats.ks_2samp(tmp_def["pd"], tmp_nondef["pd"])[0]
        return ks

    def auroc(self) -> float:
        """
        Calculate the AUROC score on the trained data.

        :return: AUROC score of target variable.
        """
        return roc_auc_score(self._target_y, self._y_hat_prob)

    def calib(self) -> float:
        """
        Calculate the calibration on the trained data.

        :return: Calibration of target variable.
        """
        return abs(np.mean(self._target_y) - np.mean(self._y_hat_prob)) / np.mean(
            self._target_y
        )

    def get_kpis(self) -> Dict[str, float]:
        """:return: KPIs of trained model."""
        return {
            "f1": self.f1(),
            "ks": self.ks(),
            "auroc": self.auroc(),
            "calib": self.calib()
        }
