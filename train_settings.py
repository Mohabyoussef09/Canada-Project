from __future__ import annotations
from enum import Enum
import copy
import pandas as pd


class ModelType(Enum):
    LOGISTIC_REGRESSION = 0
    RANDOM_FOREST = 1

    def __int__(self):
        return self.value

    def __str__(self):
        to_str = {
            self.LOGISTIC_REGRESSION: "Logistic regression",
            self.RANDOM_FOREST: "Random forest",
        }
        return to_str.get(self, "")

    @staticmethod
    def from_str(str_value: str) -> ModelType:
        enum = ModelType(0)
        """Mapping output strings representations to enums."""
        to_enum = {
            "Logistic regression": enum.LOGISTIC_REGRESSION,
            "Random forest": enum.RANDOM_FOREST,
        }
        return to_enum.get(str_value, None)

    @staticmethod
    def get_formula(enum):
        to_formula = {
            enum.LOGISTIC_REGRESSION: "1/(1-exp^(-(b0+b1x1+...+bkxk))",
            enum.RANDOM_FOREST: "N/A",
        }
        return to_formula.get(enum, "")


class IncludeExclude(Enum):
    INCLUDE = 0
    EXCLUDE = 1

    def __int__(self):
        return self.value

    def __str__(self):
        to_str = {
            self.INCLUDE: "Include",
            self.EXCLUDE: "Exclude",
        }
        return to_str.get(self, "")

    @staticmethod
    def from_str(str_val: str) -> IncludeExclude:
        enum = IncludeExclude(0)
        to_enum = {
            "Include": enum.INCLUDE,
            "Exclude": enum.EXCLUDE,
        }
        return to_enum.get(str_val, None)


class TrainModelSettings:

    def __init__(self):
        super().__init__()
        self.model_frameworks = [ModelType.LOGISTIC_REGRESSION,
                                 ModelType.RANDOM_FOREST]
        self.model_framework = ModelType.LOGISTIC_REGRESSION

        self.n_estimators_default = 10
        self.n_estimators_min = 1
        self.n_estimators = self.n_estimators_default

        self.max_depth_default = None
        self.max_depth_min = 1
        self.max_depth = self.max_depth_default

        self.min_samples_split_default = 2
        self.min_samples_split_min = 1
        self.min_samples_split = self.min_samples_split_default

        self.min_samples_leaf_default = 1
        self.min_samples_leaf_min = 1
        self.min_samples_leaf = self.min_samples_leaf_default

    def reset(self):
        self.model_framework = ModelType.LOGISTIC_REGRESSION
        self.changed.emit()


    def __copy__(self):
        new_instance = TrainModelSettings()
        new_instance.model_framework = self.model_framework
        new_instance.n_estimators = self.n_estimators
        new_instance.max_depth = self.max_depth
        new_instance.min_samples_split = self.min_samples_split
        new_instance.min_samples_leaf = self.min_samples_leaf
        return new_instance

    def serialize(self):
        return {
            "model_framework": str(self.model_framework),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
        }

    def deserialize(self, train_model_settings_serialized_dict):
        self.model_framework = \
            ModelType.from_str(train_model_settings_serialized_dict["model_framework"])

        self.n_estimators = train_model_settings_serialized_dict["n_estimators"]
        self.max_depth = train_model_settings_serialized_dict["max_depth"]
        self.min_samples_split = train_model_settings_serialized_dict["min_samples_split"]
        self.min_samples_leaf = train_model_settings_serialized_dict["min_samples_leaf"]
        self.changed.emit()


class TrainSettings:
    def __init__(self):
        super().__init__()
        self.is_model_trained = False
        self.is_model_trained_default = self.is_model_trained

        self.model_settings = TrainModelSettings()

        self.settings_frozen = None

    def reset(self):
        self.is_model_trained = False
        self.model_settings.reset()
        self.settings_frozen = None
        self.changed.emit()

    def update(self):
        self.changed.emit()

    def __copy__(self):
        new_instance = TrainSettings()
        new_instance.is_model_trained = copy.copy(self.is_model_trained)
        new_instance.model_settings = copy.copy(self.model_settings)
        return new_instance

    def freeze(self):
        self.settings_frozen = copy.copy(self)

    @property
    def model_settings_df(self):
        if self.model_settings.model_framework == ModelType.LOGISTIC_REGRESSION:
            return pd.DataFrame({"model_setting": ["model_framework"],
                                 "value": [str(self.model_settings.model_framework)]})

        labels = \
            ["model_framework", "num_trees", "max_depth", "min_samples_split", "min_samples_leaf"]
        values = \
            [str(self.model_settings.model_framework),
             self.model_settings.n_estimators,
             self.model_settings.max_depth,
             self.model_settings.min_samples_split,
             self.model_settings.min_samples_leaf,
             ]
        return pd.DataFrame({"model_setting": labels, "value": values})


    def serialize(self):
        return {
            "is_model_trained": self.is_model_trained,
            "model_settings": self.model_settings.serialize(),
        }

    def deserialize(self, train_settings_serialized_dict):
        self.is_model_trained = train_settings_serialized_dict["is_model_trained"]
        self.model_settings.deserialize(train_settings_serialized_dict["model_settings"])
        self.update()
