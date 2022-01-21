from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import scipy
import sklearn
import sklearn_json as skljson

from abc import ABC, abstractmethod
from typing import Any

from train_settings import ModelType


class Model(ABC):
    """Given data and a target, construct the model."""

    def __init__(self,
                 data_x: pd.DataFrame,
                 target_y: pd.Series,
                 model_type=None,
                 **kwargs) -> None:
        self._data_x = data_x
        self._target_y = target_y
        self._model_type = model_type
        self._trained_model = None

    @abstractmethod
    def __copy__(self) -> Model:
        pass

    @property
    def data_x(self) -> pd.DataFrame:
        return self._data_x

    @property
    def target_y(self) -> pd.Series:
        return self._target_y

    @property
    def model_type(self) -> ModelType:
        return self._model_type

    @property
    def trained_model(self):
        return self._trained_model

    @property
    def variable_list(self) -> list[str]:
        """Get list of variable features."""
        return list(self._data_x.columns)

    @property
    def prob_y(self):
        return pd.Series(self.trained_model.predict_proba(self._data_x)[:, 1])

    def get_predicted_values(self, data_x):
        """Predict probabilities using the trained model."""
        return pd.Series(self.trained_model.predict_proba(data_x)[:, 1])

    def get_average_probability(self, data_x):
        """Get average probability values using the trained model."""
        return self.get_predicted_values(data_x).mean()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def serialize(self):
        pass

    @abstractmethod
    def deserialize(self, model_serialized_dict):
        pass


class LogisticRegression(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._model_type = ModelType.LOGISTIC_REGRESSION
        self._penalty = kwargs.get("penalty", "none")
        self._solver = kwargs.get("solver", "newton-cg")
        self._max_iter_input = kwargs.get("max_iter_input", 1000)
        self._tol_input = kwargs.get("tol_input", 1e-7)
        self._fit_intercept = kwargs.get("fit_intercept", True)

    def __copy__(self):
        new_instance = LogisticRegression(data_x=self.data_x, target_y=self.target_y)
        new_instance._penalty = self._penalty
        new_instance._solver = self._solver
        new_instance._max_iter_input = self._max_iter_input
        new_instance._to_input = self._tol_input
        new_instance._fit_intercept = self._fit_intercept
        new_instance._trained_model = copy.copy(self._trained_model)
        return new_instance

    @property
    def num_parameters(self):
        """Calculate number of features"""
        # TODO: To incorporate potential penalty terms if penalty != "none"
        return len(self.variable_list) + 1

    @property
    def coefficients(self):
        return pd.DataFrame.from_dict(
            {"variable": ["intercept"] + list(self._data_x.columns),
             "coefficient": self.trained_model.intercept_.tolist() + self.trained_model.coef_.tolist()[0]}
        )

    @property
    def p_values(self):
        x_temp = pd.DataFrame({"intercept": np.ones(self._data_x.shape[0])}).join(pd.DataFrame(self._data_x))
        x = np.asarray(x_temp)
        coefs = np.array(self.coefficients["coefficient"])
        prob_y = np.array(self.prob_y)

        hessian = np.dot(prob_y * (1 - prob_y) * x.T, x)
        k = hessian.shape[0]

        # Check if full rank: distort slightly if not.
        if np.linalg.matrix_rank(hessian) < k:
            hessian += np.diag(np.random.rand(k) / 10 ** 5)

        cov_params = np.linalg.inv(hessian)
        standard_errors = np.sqrt(np.diagonal(cov_params))
        z_statistics = coefs / standard_errors
        p_values = 2 * scipy.stats.norm.sf(np.abs(z_statistics))
        return pd.DataFrame.from_dict({"variable": list(x_temp.columns), "p_value": list(p_values)})

    def train(self):
        """Train vanilla logistic regression model."""
        self._trained_model = \
            sklearn.linear_model.LogisticRegression(penalty=self._penalty,
                                                    solver=self._solver,
                                                    max_iter=self._max_iter_input,
                                                    tol=self._tol_input,
                                                    fit_intercept=self._fit_intercept,
                                                    random_state=0).fit(self.data_x, self.target_y)

    def serialize(self):
        logistic_regression_dict: dict[str, Any] = {"data_x": self._data_x.to_json(),
                                                    "target_y": self._target_y.tolist(),
                                                    "penalty": self._penalty,
                                                    "solver": self._solver,
                                                    "max_iter_input": self._max_iter_input,
                                                    "tol_input": self._tol_input,
                                                    "fit_intercept": self._fit_intercept,
                                                    "trained_model": ""
                                                    }

        if self.trained_model is not None:
            logistic_regression_dict["trained_model"] = skljson.to_json(self.trained_model,
                                                                        "train_model")

        return logistic_regression_dict

    def deserialize(self, logistic_regression_serialized_dict):
        self._data_x = pd.read_json(logistic_regression_serialized_dict["data_x"])
        self._target_y = pd.Series(logistic_regression_serialized_dict["target_y"])
        self._penalty = logistic_regression_serialized_dict["penalty"]
        self._solver = logistic_regression_serialized_dict["solver"]
        self._max_iter_input = logistic_regression_serialized_dict["max_iter_input"]
        self._tol_input = logistic_regression_serialized_dict["tol_input"]
        self._fit_intercept = logistic_regression_serialized_dict["fit_intercept"]

        self._trained_model = None
        if logistic_regression_serialized_dict["trained_model"] != "":
            self._trained_model = skljson.from_json("train_model")


class RandomForest(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._model_type = ModelType.RANDOM_FOREST
        self._n_estimators = kwargs.get("n_estimators", 10)
        self._max_depth = kwargs.get("max_depth", None)
        self._min_samples_split = kwargs.get("min_samples_split", 2)
        self._min_samples_leaf = kwargs.get("min_samples_leaf", 1)

    def __copy__(self):
        new_instance = RandomForest(data_x=self.data_x, target_y=self.target_y)
        new_instance._n_estimators = self._n_estimators
        new_instance._max_depth = self._max_depth
        new_instance._min_samples_split = self._min_samples_split
        new_instance._min_samples_leaf = self._min_samples_leaf
        new_instance._trained_model = copy.copy(self._trained_model)
        return new_instance

    @property
    def feature_importances(self):
        """Get feature importance values."""
        return pd.DataFrame({"variable": self._data_x.columns,
                             "importance": self._trained_model.feature_importances_}
                            ).sort_values(by=["importance", "variable"],
                                          ascending=[False, True]).reset_index(drop=True)

    @property
    def params(self):
        """Random forest parameters."""
        params_df = pd.DataFrame([self._trained_model.get_params()]).T.reset_index()
        params_df.columns = ["param", "value"]
        return params_df

    def train(self) -> sklearn.ensemble:
        """Train random forest model."""
        self._trained_model = sklearn.ensemble.RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            min_samples_split=self._min_samples_split,
            min_samples_leaf=self._min_samples_leaf,
            random_state=0).fit(self.data_x, self.target_y)

    def serialize(self):
        random_forest_dict: dict[str, Any] = {"data_x": self._data_x.to_json(),
                                              "target_y": self._target_y.tolist(),
                                              "n_estimators": self._n_estimators,
                                              "max_depth": self._max_depth,
                                              "min_samples_split": self._min_samples_split,
                                              "min_samples_leaf": self._min_samples_leaf,
                                              "trained_model": ""
                                              }

        if self.trained_model is not None:
            random_forest_dict["trained_model"] = \
                skljson.to_json(self.trained_model, "train_model")

        return random_forest_dict

    def deserialize(self, random_forest_serialized_dict):
        self._data_x = pd.read_json(random_forest_serialized_dict["data_x"])
        self._target_y = pd.Series(random_forest_serialized_dict["target_y"])
        self._n_estimators = random_forest_serialized_dict["n_estimators"]
        self._max_depth = random_forest_serialized_dict["max_depth"]
        self._min_samples_split = random_forest_serialized_dict["min_samples_split"]
        self._min_samples_leaf = random_forest_serialized_dict["min_samples_leaf"]

        self._trained_model = None
        if random_forest_serialized_dict["trained_model"] != "":
            self._trained_model = skljson.from_json("train_model")


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression as LR

    df = pd.read_csv("C:/Users/Basil/Desktop/data/dummy_train_nice.csv")
    data_x = df.drop(columns=["default"])
    target_y = df["default"]
    train_logistic_regression = LogisticRegression(data_x=data_x, target_y=target_y)
    train_logistic_regression.train()
    print(train_logistic_regression.prob_y.mean())

    train_random_forest = RandomForest(data_x=data_x, target_y=target_y)
    train_random_forest.train()
    print(train_random_forest.prob_y.mean())
