from __future__ import annotations
from enum import Enum
import pandas as pd
import copy

class BinAsMode(Enum):
    NUM_MIN_MAX = 0
    NUM_MINUS_INF_MAX = 1
    NUM_MIN_PLUS_INF = 2
    NUM_MINUS_INF_PLUS_INF = 3
    CATEGORICAL = 4
    NONE = 5

    def __int__(self):
        return self.value

    def __str__(self):
        to_str = {
            self.NUM_MIN_MAX: "Numeric (LB=min, UB=max)",
            self.NUM_MINUS_INF_MAX: "Numeric (LB=-inf, UB=max)",
            self.NUM_MIN_PLUS_INF: "Numeric (LB=min, UB=+inf)",
            self.NUM_MINUS_INF_PLUS_INF: "Numeric (LB=-inf, UB=+inf)",
            self.CATEGORICAL: "Categorical",
            self.NONE: "None",
        }
        return to_str.get(self, "")

    @staticmethod
    def from_str(str_value: str) -> BinAsMode:
        enum = BinAsMode(0)
        to_enum = {
            "Numeric (LB=min, UB=max)": enum.NUM_MIN_MAX,
            "Numeric (LB=-inf, UB=max)": enum.NUM_MINUS_INF_MAX,
            "Numeric (LB=min, UB=+inf)": enum.NUM_MIN_PLUS_INF,
            "Numeric (LB=-inf, UB=+inf)": enum.NUM_MINUS_INF_PLUS_INF,
            "Categorical": enum.CATEGORICAL,
            "None": enum.NONE,
        }
        return to_enum.get(str_value, None)

    @staticmethod
    def non_numeric_modes() -> list[BinAsMode]:
        return [BinAsMode.CATEGORICAL]

    @staticmethod
    def numeric_modes() -> list[BinAsMode]:
        return [
            BinAsMode.NUM_MIN_MAX,
            BinAsMode.NUM_MINUS_INF_MAX,
            BinAsMode.NUM_MIN_PLUS_INF,
            BinAsMode.NUM_MINUS_INF_PLUS_INF,
        ]


class BinningNumericMode(Enum):
    NONE = 0
    ALL = 1
    ONLY_WITH_MISSING = 2

    def __int__(self):
        return self.value

    def __str__(self):
        to_str = {
            self.NONE: "None",
            self.ALL: "All",
            self.ONLY_WITH_MISSING: "Only with missing",
        }
        return to_str.get(self, "")

    @staticmethod
    def from_str(str_value: str) -> BinningNumericMode:
        enum = BinningNumericMode(0)
        to_enum = {
            "None": enum.NONE,
            "All": enum.ALL,
            "Only with missing": enum.ONLY_WITH_MISSING,
        }
        return to_enum.get(str_value, None)


class BinningNumericSettings():
    def __init__(self):
        super().__init__()
        self.bin_mode = BinningNumericMode.ONLY_WITH_MISSING
        self.bin_all_radio_default = False
        self.bin_only_missing_radio_default = True

        self.max_num_bins_min = 2
        self.max_num_bins_max = 10
        self.max_num_bins = 5
        self.max_num_bins_default = 10

        self.decimal_place_min = 0
        self.decimal_place_max = 5
        self.decimal_place = 2
        self.decimal_place_default = 0

        self.lower_bound_is_infinity = False
        self.lower_bound_is_infinity_default = 0

        self.upper_bound_is_infinity = False
        self.upper_bound_is_infinity_default = 0

    def __copy__(self):
        new_instance = BinningNumericSettings()
        new_instance.bin_mode = self.bin_mode
        new_instance.max_num_bins = self.max_num_bins
        new_instance.decimal_place = self.decimal_place
        new_instance.lower_bound_is_infinity = self.lower_bound_is_infinity
        new_instance.upper_bound_is_infinity = self.upper_bound_is_infinity
        return new_instance

    def serialize(self):
        return {
            "bin_mode": str(self.bin_mode),
            "max_num_bins": self.max_num_bins,
            "decimal_place": self.decimal_place,
            "lower_bound": self.lower_bound_is_infinity,
            "upper_bound": self.upper_bound_is_infinity,
        }

    def deserialize(self, numeric_serialized_dict):
        self.bin_mode = BinningNumericMode.from_str(numeric_serialized_dict["bin_mode"])
        self.max_num_bins = numeric_serialized_dict["max_num_bins"]
        self.decimal_place = numeric_serialized_dict["decimal_place"]
        self.lower_bound_is_infinity = numeric_serialized_dict["lower_bound"]
        self.upper_bound_is_infinity = numeric_serialized_dict["upper_bound"]
        self.changed.emit()


class BinningCategoricalSettings():

    def __init__(self):
        super().__init__()
        self.bin_all_checkbox = True
        self.bin_all_checkbox_default = True

        self.max_num_bins_min = 1
        self.max_num_bins_max = 50
        self.max_num_bins = 20
        self.max_num_bins_default = 1


    def __copy__(self):
        new_instance = BinningCategoricalSettings()
        new_instance.bin_all_checkbox = self.bin_all_checkbox
        new_instance.max_num_bins = self.max_num_bins
        return new_instance

    def serialize(self):
        return {
            "bin_all": self.bin_all_checkbox,
            "max_num_bins": self.max_num_bins,
        }

    def deserialize(self, categorical_serialized_dict):
        self.bin_all_checkbox = categorical_serialized_dict["bin_all"]
        self.max_num_bins = categorical_serialized_dict["max_num_bins"]
        self.changed.emit()


class BinningExclusions():

    def __init__(self):
        super().__init__()
        self.datetime_vars = True
        self.datetime_vars_default = True

        self.perc_missing_vals_gt_min = 0
        self.perc_missing_vals_gt_max = 100
        self.perc_missing_vals_gt = 60
        self.perc_missing_vals_gt_default = 60

        self.num_unique_vals_lt_min = 2
        self.num_unique_vals_lt = 2
        self.num_unique_vals_lt_default = 2

    def __copy__(self):
        new_instance = BinningExclusions()
        new_instance.datetime_vars = self.datetime_vars
        new_instance.perc_missing_vals_gt = self.perc_missing_vals_gt
        new_instance.num_unique_vals_lt = self.num_unique_vals_lt
        return new_instance

    def serialize(self):
        return {
            "datetime_vars": self.datetime_vars,
            "perc_missing_vals": self.perc_missing_vals_gt,
            "num_unique_vals": self.num_unique_vals_lt,
        }

    def deserialize(self, exclude_serialized_dict):
        self.datetime_vars = exclude_serialized_dict["datetime_vars"]
        self.perc_missing_vals_gt = exclude_serialized_dict["perc_missing_vals"]
        self.num_unique_vals_lt = exclude_serialized_dict["num_unique_vals"]
        self.changed.emit()


class BinningSettings():

    def __init__(self):
        super().__init__()
        self.numeric = BinningNumericSettings()
        self.categorical = BinningCategoricalSettings()
        self.exclude = BinningExclusions()
        self.settings_frozen = None


    def __copy__(self):
        new_instance = BinningSettings()
        new_instance.numeric = copy.copy(self.numeric)
        new_instance.categorical = copy.copy(self.categorical)
        new_instance.exclude = copy.copy(self.exclude)
        return new_instance

    def freeze(self):
        self.settings_frozen = copy.copy(self)

    def get_numeric_settings_df(self):
        bin_all = False
        bin_only_missing = True
        if self.numeric.bin_mode == self.numeric.bin_mode.ALL:
            bin_all = True
            bin_only_missing = False

        ret_dict = {
            "bin_all": [bin_all],
            "bin_only_missing": [bin_only_missing],
            "max_num_bins": [self.numeric.max_num_bins],
            "decimal_place": [self.numeric.decimal_place],
            "lower_bound_all_minus_inf": [self.numeric.lower_bound_is_infinity],
            "upper_bound_all_plus_inf": [self.numeric.upper_bound_is_infinity]
        }
        return pd.DataFrame.from_dict(ret_dict, orient="index").reset_index().rename(
            columns={"index": "numeric_criterion", 0: "value"}
        )

    def get_categorical_settings_df(self):
        ret_dict = {
            "bin_all": [self.categorical.bin_all_checkbox],
            "max_num_bins": [self.categorical.max_num_bins],
        }
        return pd.DataFrame.from_dict(ret_dict, orient="index").reset_index().rename(
            columns={"index": "categorical_criterion", 0: "value"}
        )

    def get_exclude_settings_df(self):
        ret_dict = {
            "datetime_variables": [self.exclude.datetime_vars],
            "perc_missing_values_greater_equal": [self.exclude.perc_missing_vals_gt / 100.0],
            "num_unique_values_less_than": [self.exclude.num_unique_vals_lt]
        }
        return pd.DataFrame.from_dict(ret_dict, orient="index").reset_index().rename(
            columns={"index": "binning_exclusion_criterion", 0: "value"}
        )

    def serialize(self):
        return {
            "numeric": self.numeric.serialize(),
            "categorical": self.categorical.serialize(),
            "exclude": self.exclude.serialize(),
        }

    def deserialize(self, binning_settings_serialized_dict):
        self.numeric.deserialize(binning_settings_serialized_dict["numeric"])
        self.categorical.deserialize(binning_settings_serialized_dict["categorical"])
        self.exclude.deserialize(binning_settings_serialized_dict["exclude"])
        self.update()
