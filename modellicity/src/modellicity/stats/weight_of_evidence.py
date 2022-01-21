"""Weight of Evidence class."""

from modellicity.src.modellicity.extended_pandas import ExtendedSeries
from modellicity.src.modellicity.stats.weight_of_evidence_utils import (
    apply_numeric_binning_left_open_right_closed,
    is_monotonic,
    largest_unique_quantiles,
    most_balanced_two_bin_split_left_open_right_closed,
    round_bin_edges_left_open_right_closed,
)
from typing import Any, Dict, Union
import logging
import numpy as np
import pandas as pd
import pandas.core.algorithms as algos

logger = logging.getLogger(__name__)


class WeightOfEvidence:
    """Class to calculate map and calculate the weight of evidence (WOE)."""

    @staticmethod
    def bin_woe_numeric_monotonic(variable: Any, target: Any, n_max: int = 10,
                                  round_limit: int = 4, lower_limit_bounded: bool = True,
                                  upper_limit_bounded: bool = True) -> pd.DataFrame:
        """Bin numeric variable such that the weight of evidence (WOE) is monotonic."""
        n_unique = algos.unique(variable).size
        n = min(n_unique, n_max)
        bin_edges_list_rounded = np.array([])
        sum_y = np.array([])
        count_y = np.array([])
        bin_is_monotonic_inc, bin_is_monotonic_dec = False, False
        while not (bin_is_monotonic_inc or bin_is_monotonic_dec):
            bin_edges_list_rounded = \
                round_bin_edges_left_open_right_closed(largest_unique_quantiles(variable, q=n),
                                                       round_limit=round_limit)
            if bin_edges_list_rounded.size <= 3:  # Check if one or two bins were created
                bin_edges_list_rounded = \
                    most_balanced_two_bin_split_left_open_right_closed(x=variable,
                                                                       round_limit=round_limit)
            n = bin_edges_list_rounded.size - 1
            if not lower_limit_bounded:
                bin_edges_list_rounded[0] = -float("inf")
            if not upper_limit_bounded:
                bin_edges_list_rounded[n] = float("inf")
            x_bin_list, ids = \
                apply_numeric_binning_left_open_right_closed(variable, bin_edges_list_rounded)
            count_y = np.bincount(ids, minlength=n)[1:]
            sum_y = np.bincount(ids, weights=target, minlength=n)[1:]
            if n == 1:
                break
            mean_y = sum_y / count_y
            bin_is_monotonic_dec, bin_is_monotonic_inc = is_monotonic(mean_y)
            if n == 2:
                break
            n -= 1
        n = bin_edges_list_rounded.size - 1
        monotonic_str = "none"
        if bin_is_monotonic_inc:
            monotonic_str = "increasing"
        elif bin_is_monotonic_dec:
            monotonic_str = "decreasing"

        bin_array = []
        bin_edges_list_rounded_str = bin_edges_list_rounded.astype(str)
        for i in range(len(bin_edges_list_rounded_str) - 1):
            start = bin_edges_list_rounded_str[i]
            end = bin_edges_list_rounded_str[i + 1]
            bin_array.append(f"[{start}, {end})")

        return pd.DataFrame.from_dict({"variable": np.repeat(["x"], n),
                                       "bin_number": np.arange(1, n+1),
                                       "lower_limit": bin_edges_list_rounded[:-1],
                                       "upper_limit": bin_edges_list_rounded[1:],
                                       "bin": bin_array,
                                       "num_obs": count_y,
                                       "num_events": sum_y,
                                       "num_non_events": count_y - sum_y,
                                       "trend": np.repeat([monotonic_str], n)
                                       })

    @staticmethod
    def bin_woe_char_direct(
        variable: np.ndarray, target: np.ndarray, n_max: int = 20
    ) -> pd.DataFrame:
        """Create a weight of evidence (WOE) frequency table for categorical variables."""
        df_x_bin = pd.DataFrame({"bin": variable, "y": target})
        df_x_bin_group = df_x_bin.groupby("bin").agg({"y": ["count", "sum"]})
        df_x_bin_group.columns = df_x_bin_group.columns.droplevel(0)
        df_x_bin_group["bin_str"] = df_x_bin_group.index.astype(str)
        df_x_bin_group.sort_values(by=["count", "bin_str"], ascending=[False, True], inplace=True)
        df_x_bin_group.reset_index(drop=True, inplace=True)
        df_x_bin_group["bin_str"] = \
            np.select([df_x_bin_group.index >= n_max - 1], ["everything_else"],
                      default=df_x_bin_group["bin_str"])
        n = df_x_bin_group.shape[0]
        if n >= n_max:
            count_everything_else = df_x_bin_group["count"].iloc[n_max-1:].sum()
            sum_everything_else = df_x_bin_group["sum"].iloc[n_max-1:].sum()
            df_x_bin_group = df_x_bin_group[0:n_max]
            df_x_bin_group["count"].iloc[-1] = count_everything_else
            df_x_bin_group["sum"].iloc[-1] = sum_everything_else
            n = n_max
        return pd.DataFrame.from_dict({"variable": np.repeat(["x"], n),
                                       "bin_number": np.arange(1, n+1),
                                       "lower_limit": np.repeat([np.nan], n),
                                       "upper_limit": np.repeat([np.nan], n),
                                       "bin": df_x_bin_group["bin_str"],
                                       "num_obs": df_x_bin_group["count"],
                                       "num_events": df_x_bin_group["sum"],
                                       "num_non_events":
                                           df_x_bin_group["count"] -
                                           df_x_bin_group["sum"],
                                       "trend": np.repeat(["none"], n)
                                       })

    @staticmethod
    def bin_woe_missing(target: np.ndarray) -> pd.DataFrame:
        """Objective: Create a weight of evidence (WOE) frequency table for missing values."""
        return pd.DataFrame.from_dict({
                "variable": ["x"],
                "bin_number": [1],
                "lower_limit": [np.nan],
                "upper_limit": [np.nan],
                "bin": ["missing"],
                "num_obs": [target.size],
                "num_events": [np.sum(target)],
                "num_non_events": [target.size - np.sum(target)],
                "trend": "none",
            })

    @staticmethod
    def calculate_woe_iv_from_freq_table(df_freq: pd.DataFrame):
        """Calculate weight of evidence (WOE) and information value (IV) from a bin table."""
        df_woe = df_freq.copy()
        sum_events = df_woe["num_events"].sum()
        sum_non_events = df_woe["num_non_events"].sum()
        np.seterr(divide="ignore")
        correction_factor = 0.5  # Used to address division by zero and ln zero cases
        df_woe["event_dist"] = \
            np.select([sum_events == 0], [0], default=df_woe["num_events"] / sum_events)
        df_woe["non_event_dist"] = \
            np.select([sum_non_events == 0], [0], default=df_woe["num_non_events"] / sum_non_events)
        df_woe["woe"] = \
            np.select([(df_woe["event_dist"] == 0) | (df_woe["non_event_dist"] == 0)],
                      [np.log((df_woe["event_dist"] + correction_factor) /
                              (df_woe["non_event_dist"] + correction_factor))],
                      default=np.log(df_woe["event_dist"] / df_woe["non_event_dist"]))
        iv = ((df_woe["event_dist"] - df_woe["non_event_dist"]) * df_woe["woe"]).sum()
        df_woe["iv"] = np.repeat([iv], df_woe.shape[0])
        np.seterr(divide="warn")  # Turn warning back on
        return df_woe

    def calculate_woe_iv_from_df(self, df: pd.DataFrame, var_list: list, target_label: str,
                                 n_max_num: int = 10, n_max_cat: int = 20, round_limit: int = 4,
                                 lower_limit_bounded: Dict[str, bool] = None,
                                 upper_limit_bounded: Dict[str, bool] = None) -> pd.DataFrame:
        """Calculate weight of evidence (WOE) and information value (IV) of variables."""
        df_freq_missing = pd.DataFrame()
        df_freq_non_missing = pd.DataFrame()
        df_woe = pd.DataFrame()
        first_variable = True
        for var in var_list:
            df_feature_missing = df[[target_label]][df[var].isnull()]
            df_feature_non_missing = \
                df[[var, target_label]][df[var].notnull()]
            missing_values_exist = df_feature_missing.shape[0] > 0
            if missing_values_exist:
                df_freq_missing = \
                    self.bin_woe_missing(target=df_feature_missing[target_label].to_numpy())
            var_is_numeric = ExtendedSeries(df_feature_non_missing[var]).is_all_numeric_object()
            non_missing_values_exist = df_feature_non_missing.shape[0] > 0
            if non_missing_values_exist:  # To avoid the case that the entire row is missing.
                if var_is_numeric:
                    if lower_limit_bounded and var in lower_limit_bounded:
                        lower_limit_bounded_value = lower_limit_bounded[var]
                    else:
                        lower_limit_bounded_value = True
                    if upper_limit_bounded and var in upper_limit_bounded:
                        upper_limit_bounded_value = upper_limit_bounded[var]
                    else:
                        upper_limit_bounded_value = True

                    df_freq_non_missing = \
                        self.bin_woe_numeric_monotonic(
                            variable=df_feature_non_missing[var].to_numpy(),
                            target=df_feature_non_missing[target_label].to_numpy(),
                            n_max=n_max_num, round_limit=round_limit,
                            lower_limit_bounded=lower_limit_bounded_value,
                            upper_limit_bounded=upper_limit_bounded_value)
                else:  # If variable isn't numeric.
                    df_freq_non_missing = \
                        self.bin_woe_char_direct(
                            variable=df_feature_non_missing[var].to_numpy(),
                            target=df_feature_non_missing[target_label].to_numpy(),
                            n_max=n_max_cat)
            if missing_values_exist:
                df_freq = df_freq_missing
                if non_missing_values_exist:
                    df_freq = df_freq.append(df_freq_non_missing, ignore_index=True)
                df_freq["bin_number"] = np.arange(1, df_freq.shape[0]+1)  # Reset bin numbers
            else:
                df_freq = df_freq_non_missing
            df_woe_var = self.calculate_woe_iv_from_freq_table(df_freq)
            if not var_is_numeric:
                df_woe_var["woe_temp"] = \
                    np.select([df_woe_var["bin"].isin(["missing"])], [df_woe_var["woe"].min() - 1],
                              default=df_woe_var["woe"])
                df_woe_var.sort_values(by=["woe_temp", "bin"], ascending=[True, True], inplace=True)
                df_woe_var.drop(columns=["woe_temp"], inplace=True)
                df_woe_var["bin_number"] = np.arange(1, df_woe_var.shape[0]+1)
            df_woe_var["variable"] = np.repeat([var], df_woe_var.shape[0])
            if first_variable:
                first_variable = False
                df_woe = df_woe_var
            else:
                df_woe = df_woe.append(df_woe_var, ignore_index=True)
        df_woe.sort_values(by=["iv", "variable", "bin_number"], ascending=[False, True, True],
                           na_position="first", inplace=True)
        df_woe.reset_index(drop=True, inplace=True)
        return df_woe

    @staticmethod
    def map_to_woe(variable: pd.Series, df_woe_map: pd.DataFrame) -> Union[pd.Series, np.ndarray]:
        """Map a variable to its weight of evidence (WOE) counterpart."""
        missing_values_exist = ExtendedSeries(variable).is_any_missing()
        missing_value_category_exists = "missing" in df_woe_map["bin"].tolist()
        variable_is_numeric = ExtendedSeries(variable).is_all_numeric_object()
        lower_limits = df_woe_map[df_woe_map["lower_limit"].notnull()]["lower_limit"].sort_values()
        upper_limits = df_woe_map[df_woe_map["upper_limit"].notnull()]["upper_limit"].sort_values()
        if variable_is_numeric and lower_limits.size > 0 and lower_limits.size == upper_limits.size:
            df_woe_map_copy = df_woe_map[df_woe_map["lower_limit"].notnull()].copy()
            df_woe_map_copy.sort_values(by=["lower_limit"], inplace=True)
            df_woe_map_copy["temp_bin_label"] = np.arange(df_woe_map_copy.shape[0])
            bin_edges_list = np.sort(algos.unique(np.concatenate((lower_limits, upper_limits))))
            x_bin_label = bin_edges_list.searchsorted(variable, side="right") - 1
            df_x_mapped = \
                pd.DataFrame({"order": np.arange(variable.size), "x": variable,
                              "temp_bin_label": x_bin_label})
            df_x_merged = pd.merge(df_x_mapped, df_woe_map_copy, how="left", on="temp_bin_label")
            df_x_merged.sort_values(by=["order"], inplace=True)
            x_woe = df_x_merged["woe"]
            if missing_values_exist and missing_value_category_exists:
                woe_null = df_woe_map[df_woe_map["bin"] == "missing"]["woe"].iloc[0]
                x_woe = \
                    np.select([df_x_merged["x"].isnull(), ~df_x_merged["x"].isnull()],
                              [woe_null, x_woe])
        else:  # Non-numeric is checking if a catch-all bin exists and directly mapping variables
            # that are found.
            catch_all_category_exists = "everything_else" in df_woe_map["bin"].tolist()
            variable = np.select([variable.isnull()], [np.nan], default=variable.astype(str))
            variable = pd.Series(variable)
            woe_map_bin = df_woe_map["bin"].astype(str)
            x_bin = np.select(
                [
                    (variable.isnull() & missing_value_category_exists),
                    (variable.isnull() & ~missing_value_category_exists),
                    (~variable.isnull() & ~variable.isin(woe_map_bin) & catch_all_category_exists),
                    (~variable.isnull() & ~variable.isin(woe_map_bin) & ~catch_all_category_exists),
                    (~variable.isnull() & variable.isin(woe_map_bin))
                ],
                ["missing", np.nan, "everything_else", np.nan, variable]
            )
            # Include "order" variable to enable sorting in same order as original after merge.
            df_x_mapped = \
                pd.DataFrame({"order": np.arange(variable.size), "x": variable,
                              "bin": pd.Series(x_bin).astype(str)})
            df_woe_map_str = df_woe_map.copy()
            df_woe_map_str["bin"] = woe_map_bin  # Ensure value is string
            df_x_merged = pd.merge(df_x_mapped, df_woe_map_str, how="left", on="bin")
            df_x_merged.sort_values(by=["order"], inplace=True)
            x_woe = df_x_merged["woe"]
        return x_woe

    def map_df_to_woe(self, df: pd.DataFrame, df_woe_map: pd.DataFrame) -> pd.DataFrame:
        """Map variables in a dataframe to their weight of evidence (WOE) counterpart."""
        var_list = [x for x in df.columns if x in df_woe_map["variable"].tolist()]
        df_woe = df.copy()
        for var in var_list:
            df_woe[var] = \
                self.map_to_woe(variable=df[var],
                                df_woe_map=df_woe_map[df_woe_map["variable"] == var])
        return df_woe
