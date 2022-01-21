"""Retain analyze artifacts during the life of the application"""
from analyze_settings import AnalyzeSettings, IncludeExclude
import copy
import pandas as pd


class AnalyzeArtifacts:

    def __init__(self):
        super().__init__()
        self.distribution_analysis_df = None
        self.volatility_analysis_df = None
        self.correlation_analysis_df = None
        self.disabled_var_labels_list = None
        self.analyze_settings = None

    def __copy__(self):
        new_instance = AnalyzeArtifacts()
        if self.distribution_analysis_df is not None:
            new_instance.distribution_analysis_df = self.distribution_analysis_df.copy()
        if self.volatility_analysis_df is not None:
            new_instance.volatility_analysis_df = self.volatility_analysis_df.copy()
        if self.correlation_analysis_df is not None:
            new_instance.correlation_analysis_df = self.correlation_analysis_df.copy()
        if self.disabled_var_labels_list is not None:
            new_instance.disabled_var_labels_list = self.disabled_var_labels_list.copy()
        if self.analyze_settings is not None:
            new_instance.analyze_settings = copy.copy(self.analyze_settings)
        return new_instance

    def reset(self):
        self.distribution_analysis_df = None
        self.volatility_analysis_df = None
        self.correlation_analysis_df = None
        self.disabled_var_labels_list = None
        self.analyze_settings = None
        self.reset_signal.emit()

    def update_all(self,
                   distribution_analysis_df,
                   volatility_analysis_df,
                   correlation_analysis_df,
                   disabled_var_labels_list,
                   analyze_settings):
        self.distribution_analysis_df = distribution_analysis_df.copy()
        self.volatility_analysis_df = volatility_analysis_df.copy()
        self.correlation_analysis_df = correlation_analysis_df.copy()
        self.disabled_var_labels_list = disabled_var_labels_list.copy()
        self.analyze_settings = copy.copy(analyze_settings)
        self.update()

    def update(self):
        self.changed_signal.emit()

    @property
    def all_artifacts_available(self):
        if self.distribution_analysis_df is not None and \
                self.volatility_analysis_df is not None and \
                self.correlation_analysis_df is not None and \
                self.disabled_var_labels_list is not None and \
                self.analyze_settings is not None:
            return True
        return False

    def serialize(self):
        artifacts_dict = {
            "all_available": False,
            "distribution_analysis": "",
            "volatility_analysis": "",
            "correlation_analysis": "",
            "disabled_var_labels": "",
            "analyze_settings": "",
        }
        if self.all_artifacts_available:
            artifacts_dict["all_available"] = True

            distribution_analysis_df = self.distribution_analysis_df.copy()
            distribution_analysis_df["include_exclude_panel"] = \
                [str(x) for x in list(distribution_analysis_df["include_exclude_panel"])]
            distribution_analysis_df["include_exclude"] = \
                [str(x) for x in list(distribution_analysis_df["include_exclude"])]
            artifacts_dict["distribution_analysis"] = distribution_analysis_df.to_json()

            volatility_analysis_df = self.volatility_analysis_df.copy()
            volatility_analysis_df["include_exclude_panel"] = \
                [str(x) for x in list(volatility_analysis_df["include_exclude_panel"])]
            volatility_analysis_df["include_exclude"] = \
                [str(x) for x in list(volatility_analysis_df["include_exclude"])]
            artifacts_dict["volatility_analysis"] = volatility_analysis_df.to_json()

            correlation_analysis_df = self.correlation_analysis_df.copy()
            correlation_analysis_df["include_exclude_panel"] = \
                [str(x) for x in list(correlation_analysis_df["include_exclude_panel"])]
            correlation_analysis_df["include_exclude"] = \
                [str(x) for x in list(correlation_analysis_df["include_exclude"])]
            artifacts_dict["correlation_analysis"] = correlation_analysis_df.to_json()

            disabled_var_labels_list = self.disabled_var_labels_list.copy()
            artifacts_dict["disabled_var_labels"] = disabled_var_labels_list

            artifacts_dict["analyze_settings"] = self.analyze_settings.serialize()

        return artifacts_dict

    def deserialize(self, artifacts_serialized):
        self.reset()

        if artifacts_serialized["all_available"]:
            self.distribution_analysis_df = \
                pd.read_json(artifacts_serialized["distribution_analysis"])

            self.distribution_analysis_df["include_exclude_panel"] = \
                [IncludeExclude.from_str(x) for x in
                 list(self.distribution_analysis_df["include_exclude_panel"])]
            self.distribution_analysis_df["include_exclude"] = \
                [IncludeExclude.from_str(x) for x in
                 list(self.distribution_analysis_df["include_exclude"])]

            self.volatility_analysis_df = pd.read_json(artifacts_serialized["volatility_analysis"])
            self.volatility_analysis_df["include_exclude_panel"] = \
                [IncludeExclude.from_str(x) for x in
                 list(self.volatility_analysis_df["include_exclude_panel"])]
            self.volatility_analysis_df["include_exclude"] = \
                [IncludeExclude.from_str(x) for x in
                 list(self.volatility_analysis_df["include_exclude"])]
            self.correlation_analysis_df = \
                pd.read_json(artifacts_serialized["correlation_analysis"])
            self.correlation_analysis_df["include_exclude_panel"] = \
                [IncludeExclude.from_str(x) for x in
                 list(self.correlation_analysis_df["include_exclude_panel"])]

            self.correlation_analysis_df["include_exclude"] = \
                [IncludeExclude.from_str(x) for x in
                 list(self.correlation_analysis_df["include_exclude"])]

            self.disabled_var_labels_list = artifacts_serialized["disabled_var_labels"]

            self.analyze_settings = AnalyzeSettings()
            self.analyze_settings.deserialize(artifacts_serialized["analyze_settings"])

            self.update()
