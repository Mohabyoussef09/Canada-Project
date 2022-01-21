from __future__ import annotations
from enum import Enum
import copy
import pandas as pd


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
    def from_str(str_value: str) -> IncludeExclude:
        enum = IncludeExclude(0)
        to_enum = {
            "Include": enum.INCLUDE,
            "Exclude": enum.EXCLUDE,
        }
        return to_enum.get(str_value, None)

class AnalyzeInclusions:

    def __init__(self):
        super().__init__()
        self.perc_most_common_lt_label = "% most common value <"
        self.perc_most_common_lt_min = 0
        self.perc_most_common_lt_max = 100
        self.perc_most_common_lt = 99
        self.perc_most_common_lt_default = self.perc_most_common_lt

        self.information_value_gt_label = "Information value ≥"
        self.information_value_gt_min = 0
        self.information_value_gt_max = 100
        self.information_value_gt = 10
        self.information_value_gt_default = self.information_value_gt

        self.abs_correlation_lt_label = "abs(correlation) <"
        self.abs_correlation_lt_min = 0
        self.abs_correlation_lt_max = 100
        self.abs_correlation_lt = 50
        self.abs_correlation_lt_default = self.abs_correlation_lt

        self.binned_vars_only_label = "Binned variables only"
        self.binned_vars_only = False
        self.binned_vars_only_default = self.binned_vars_only

    def __copy__(self):
        new_instance = AnalyzeInclusions()
        new_instance.perc_most_common_lt = self.perc_most_common_lt
        new_instance.information_value_gt = self.information_value_gt
        new_instance.abs_correlation_lt = self.abs_correlation_lt
        new_instance.binned_vars_only = self.binned_vars_only
        return new_instance


    def serialize(self):
        return {
            "perc_most_common":  self.perc_most_common_lt,
            "information_value": self.information_value_gt,
            "abs_correlation": self.abs_correlation_lt,
            "binned_vars": self.binned_vars_only,
        }

    def deserialize(self, include_serialized_dict):
        self.perc_most_common_lt = include_serialized_dict["perc_most_common"]
        self.information_value_gt = include_serialized_dict["information_value"]
        self.abs_correlation_lt = include_serialized_dict["abs_correlation"]
        self.binned_vars_only = include_serialized_dict["binned_vars"]
        self.changed.emit()


class AnalyzeSettings:

    def __init__(self):
        super().__init__()
        self.include = AnalyzeInclusions()

    def __copy__(self):
        new_instance = AnalyzeSettings()
        new_instance.include = copy.copy(self.include)
        return new_instance

    def update(self):
        self.changed.emit()

    def get_include_settings_df(self):
        ret_dict = {
            "perc_most_common_less_than": [self.include.perc_most_common_lt / 100],
            "perc_information_value_greater_than_equal":
                [self.include.information_value_gt / 100.0],
            "abs_correlation_less_than": [self.include.abs_correlation_lt / 100],
            "binned_vars_only": [self.include.binned_vars_only],
        }
        return pd.DataFrame.from_dict(ret_dict, orient="index").reset_index().rename(
            columns={"index": "include_criterion", 0: "value"}
        )

    def serialize(self):
        return {
            "include": self.include.serialize()
        }

    def deserialize(self, analyze_settings_serialized_dict):
        self.include.deserialize(analyze_settings_serialized_dict["include"])
        self.update()
