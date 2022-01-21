"""Retain model artifacts during the life of the application"""

from analyze_artifacts import AnalyzeArtifacts
from binning_artifacts import BinningArtifacts
from train_model import LogisticRegression, Model, ModelType, RandomForest
from train_settings import IncludeExclude, TrainSettings
import copy
import pandas as pd


class TrainArtifacts:
    def __init__(self):
        super().__init__()
        self.train_data_profile_df = None
        self.model_type = None
        self.model = None
        self.target_label = None
        self.probability_label = None
        self.variable_selection_df = None
        self.variable_property_df = None
        self.calibration_df = None
        self.kpis_df = None
        self.model_variable_dynamics_df = None
        self.ranking_df = None
        self.distribution_df = None
        self.train_settings = None
        self.binning_artifacts_at_training = None
        self.analyze_artifacts_at_training = None

    def reset(self):
        self.train_data_profile_df = None
        self.model_type = None
        self.model = None
        self.target_label = None
        self.probability_label = None
        self.variable_selection_df = None
        self.variable_property_df = None
        self.calibration_df = None
        self.kpis_df = None
        self.model_variable_dynamics_df = None
        self.ranking_df = None
        self.distribution_df = None
        self.train_settings = None
        self.binning_artifacts_at_training = None
        self.analyze_artifacts_at_training = None
        self.reset_signal.emit()

    def update_all(self, train_data_profile_df, model, target_label, probability_label,
                   variable_selection_df, variable_property_df, calibration_df, kpis_df,
                   model_variable_dynamics_df, ranking_df, distribution_df, train_settings,
                   binning_artifacts_at_training, analyze_artifacts_at_training):
        self.train_data_profile_df = train_data_profile_df.copy()
        self.model_type = model.model_type
        self.model = copy.copy(model)
        self.target_label = target_label
        self.probability_label = probability_label
        self.variable_selection_df = variable_selection_df.copy()
        self.variable_property_df = variable_property_df.copy()
        self.calibration_df = calibration_df.copy()
        self.kpis_df = kpis_df.copy()
        self.model_variable_dynamics_df = model_variable_dynamics_df.copy()
        self.ranking_df = ranking_df.copy()
        self.distribution_df = distribution_df.copy()
        self.train_settings = copy.copy(train_settings)
        self.binning_artifacts_at_training = copy.copy(binning_artifacts_at_training)
        self.analyze_artifacts_at_training = copy.copy(analyze_artifacts_at_training)
        self.update()

    def update(self):
        self.changed_signal.emit()

    @property
    def all_artifacts_available(self):
        if self.train_data_profile_df is not None and \
                self.model_type is not None and \
                self.model is not None and \
                self.target_label is not None and \
                self.probability_label is not None and \
                self.variable_selection_df is not None and \
                self.variable_property_df is not None and \
                self.calibration_df is not None and \
                self.kpis_df is not None and \
                self.model_variable_dynamics_df is not None and \
                self.ranking_df is not None and \
                self.distribution_df is not None and \
                self.train_settings is not None and \
                self.binning_artifacts_at_training is not None\
                and self.analyze_artifacts_at_training is not None:
            return True
        return False

    @property
    def train_df(self):
        probability_label = self.probability_label
        target_label = self.target_label
        ret = pd.DataFrame()
        ret[probability_label] = self.model.prob_y
        ret[target_label] = self.model.target_y.copy()
        ret = pd.concat([ret, self.model.data_x], axis=1)
        return ret

    def serialize(self):
        artifacts_dict = {
            "all_available": False,
            "train_data_profile": "",
            "model_type": "",
            "model": "",
            "target_label": "",
            "probability_label": "",
            "variable_selection": "",
            "variable_property": "",
            "calibration": "",
            "kpis": "",
            "model_variable_dynamics": "",
            "ranking": "",
            "distribution": "",
            "train_settings": "",
            "binning_artifacts_at_training": "",
            "analyze_artifacts_at_training": "",
        }

        if self.all_artifacts_available:
            artifacts_dict["all_available"] = True

            artifacts_dict["train_data_profile"] = self.train_data_profile_df.to_json()

            artifacts_dict["model_type"] = str(self.model.model_type)
            artifacts_dict["model"] = self.model.serialize()

            artifacts_dict["target_label"] = self.target_label
            artifacts_dict["probability_label"] = self.probability_label

            variable_selection_df = self.variable_selection_df.copy()
            variable_selection_df["include_exclude_panel"] = \
                [str(x) for x in list(variable_selection_df["include_exclude_panel"])]
            variable_selection_df["include_exclude"] = \
                [str(x) for x in list(variable_selection_df["include_exclude"])]
            artifacts_dict["variable_selection"] = variable_selection_df.to_json()

            variable_property_df = self.variable_property_df.copy()
            artifacts_dict["variable_property"] = variable_property_df.to_json()

            calibration_df = self.calibration_df.copy()
            artifacts_dict["calibration"] = calibration_df.to_json()

            kpis_df = self.kpis_df.copy()
            artifacts_dict["kpis"] = kpis_df.to_json()

            model_variable_dynamics_df = self.model_variable_dynamics_df.copy()
            artifacts_dict["model_variable_dynamics"] = model_variable_dynamics_df.to_json()

            ranking_df = self.ranking_df.copy()
            artifacts_dict["ranking"] = ranking_df.to_json()

            distribution_df = self.distribution_df.copy()
            artifacts_dict["distribution"] = distribution_df.to_json()

            artifacts_dict["train_settings"] = self.train_settings.serialize()

            artifacts_dict["binning_artifacts_at_training"] = \
                self.binning_artifacts_at_training.serialize()

            artifacts_dict["analyze_artifacts_at_training"] = \
                self.analyze_artifacts_at_training.serialize()

        return artifacts_dict

    def deserialize(self, artifacts_serialized):
        self.reset()

        if artifacts_serialized["all_available"]:
            self.train_data_profile_df = pd.read_json(artifacts_serialized["train_data_profile"])
            self.model_type = ModelType.from_str(artifacts_serialized["model_type"])
            if self.model_type == ModelType.LOGISTIC_REGRESSION:
                self.model = LogisticRegression(data_x=pd.DataFrame(), target_y=pd.Series())
            elif self.model_type == ModelType.RANDOM_FOREST:
                self.model = RandomForest(data_x=pd.DataFrame(), target_y=pd.Series())
            self.model.deserialize(artifacts_serialized["model"])

            self.target_label = artifacts_serialized["target_label"]

            self.probability_label = artifacts_serialized["probability_label"]

            self.variable_selection_df = pd.read_json(artifacts_serialized["variable_selection"])
            self.variable_selection_df["include_exclude_panel"] = \
                [IncludeExclude.from_str(x) for x in
                 list(self.variable_selection_df["include_exclude_panel"])]
            self.variable_selection_df["include_exclude"] = \
                [IncludeExclude.from_str(x) for x in
                 list(self.variable_selection_df["include_exclude"])]

            self.variable_property_df = pd.read_json(artifacts_serialized["variable_property"])

            self.calibration_df = pd.read_json(artifacts_serialized["calibration"])

            self.kpis_df = pd.read_json(artifacts_serialized["kpis"])

            self.model_variable_dynamics_df = \
                pd.read_json(artifacts_serialized["model_variable_dynamics"])

            self.ranking_df = pd.read_json(artifacts_serialized["ranking"])

            self.distribution_df = pd.read_json(artifacts_serialized["distribution"])

            self.train_settings = TrainSettings()
            self.train_settings.deserialize(artifacts_serialized["train_settings"])

            self.binning_artifacts_at_training = BinningArtifacts()
            self.binning_artifacts_at_training.deserialize(
                artifacts_serialized["binning_artifacts_at_training"])

            self.analyze_artifacts_at_training = AnalyzeArtifacts()
            self.analyze_artifacts_at_training.deserialize(
                artifacts_serialized["analyze_artifacts_at_training"])

            self.update()
