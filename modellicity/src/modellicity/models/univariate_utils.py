"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import logging
import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from typing import List, Tuple
from modellicity.stats.stats_utils import StatsUtils
from modellicity.extended_pandas.extended_properties.dataframe_properties import (
    DataFrameProperties,
)

logger = logging.getLogger(__name__)


class UnivariateUtils:
    """Univariate analysis functions."""

    @staticmethod
    def generate_event_non_event_dataframe(
        categorical_variable: pd.Series, target: pd.Series
    ) -> pd.DataFrame:
        """
        Generate event/non event dataframe.

        :param categorical_variable:
        :param target:
        :return:
        """
        var_label = categorical_variable.name
        target_label = target.name

        event_dataframe = pd.DataFrame()
        event_dataframe[var_label] = categorical_variable
        event_dataframe[target_label] = target
        event_dataframe.rename(columns={target_label: "num_event"}, inplace=True)

        non_event_dataframe = pd.DataFrame()
        non_event_dataframe[var_label] = categorical_variable
        non_event_dataframe[target_label] = 1 - target
        non_event_dataframe.rename(
            columns={target_label: "num_non_event"}, inplace=True
        )

        event_dataframe.reset_index(inplace=True)
        non_event_dataframe.reset_index(inplace=True)
        event_non_event_dataframe = pd.merge(
            event_dataframe, non_event_dataframe, how="inner", on=["index", var_label]
        )
        event_non_event_dataframe.drop(columns=["index"], inplace=True)

        event_non_event_dataframe = event_non_event_dataframe.groupby([var_label]).sum()
        event_non_event_dataframe.reset_index(inplace=True)
        event_non_event_dataframe.sort_values(by=[var_label])
        return event_non_event_dataframe

    @staticmethod
    def convert_event_non_event_dataframe_to_woe_dataframe(
        event_non_event_dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate event/non event dataframe to weight-of-evidence dataframe.

        :param event_non_event_dataframe:
        :return:
        """
        woe_and_iv_dataframe = event_non_event_dataframe.copy()
        woe_list: List[int] = []
        iv_list: List[int] = []
        total_num_event = woe_and_iv_dataframe["num_event"].sum()
        total_num_non_event = woe_and_iv_dataframe["num_non_event"].sum()

        if total_num_event == 0 or total_num_non_event == 0:
            woe_list = [0] * woe_and_iv_dataframe.shape[0]
            iv_list = [0] * woe_and_iv_dataframe.shape[0]
        else:
            for i in range(woe_and_iv_dataframe.shape[0]):
                numerator = woe_and_iv_dataframe["num_event"].iloc[i] / total_num_event
                denominator = (
                    woe_and_iv_dataframe["num_non_event"].iloc[i] / total_num_non_event
                )
                if numerator == 0 or denominator == 0:
                    woe = 0
                    iv = 0
                else:
                    woe = np.log(numerator / denominator)
                    iv = (numerator - denominator) * woe
                woe_list.append(woe)
                iv_list.append(iv)
        woe_and_iv_dataframe["woe"] = woe_list
        woe_and_iv_dataframe["iv"] = iv_list
        return woe_and_iv_dataframe

    def convert_categorical_series_to_woe_series(
        self, categorical_series: pd.Series, target: pd.Series
    ) -> Tuple[pd.Series, pd.DataFrame, float]:
        """
        Convert categorical series to weight-of-evidence series.

        Takes a series that is categorical and converts each value in that series_actions
        to its WOE counterpart. WOE is calculated first in this function
        :param categorical_series:
        :param target:
        :return:
        """
        if categorical_series.isnull().values.any():
            categorical_series.fillna("[Missing]", inplace=True)

        event_non_event_dataframe = self.generate_event_non_event_dataframe(
            categorical_series, target
        )
        woe_and_iv_dataframe = self.convert_event_non_event_dataframe_to_woe_dataframe(
            event_non_event_dataframe
        )
        woe_dataframe = woe_and_iv_dataframe.copy()
        woe_dataframe.drop(columns=["num_event", "num_non_event", "iv"], inplace=True)

        categorical_series_dataframe = pd.DataFrame(categorical_series)
        categorical_series_dataframe.reset_index(inplace=True)
        var_label = categorical_series.name
        woe_dataframe = pd.merge(
            categorical_series_dataframe, woe_dataframe, on=var_label, how="inner"
        )
        woe_dataframe.sort_values(by=["index"], inplace=True)
        iv_series = woe_and_iv_dataframe["iv"]
        woe_series = pd.Series(list(woe_dataframe["woe"]))
        woe_series.name = var_label
        iv = iv_series.sum()

        category_to_woe_map = woe_and_iv_dataframe[[var_label, "woe"]]
        if "[Missing]" in list(
            category_to_woe_map[var_label]
        ):  # bring NULL to first position
            category_to_woe_map.reset_index(inplace=True)
            temp = category_to_woe_map.copy()  # To supress copy warning message
            temp.loc[temp[var_label] == "[Missing]", "index"] = -1
            temp.sort_values(by=["index"], inplace=True)
            temp.drop(columns=["index"], inplace=True)
            temp.reset_index(drop=True, inplace=True)
            category_to_woe_map = temp.copy()

        return woe_series, category_to_woe_map, iv

    def convert_numeric_series_to_woe_series_quantiles_approach(
        self, numeric_series: pd.Series, target: pd.Series, quantiles: int
    ) -> Tuple[pd.Series, pd.DataFrame, float]:
        """
        Convert numeric series to weight-of-evidence series using quantile approach.

        :param numeric_series:
        :param target:
        :param quantiles:
        :return:
        """
        if not is_numeric_dtype(numeric_series):
            logger.exception(
                "Exception: Attempt to partition non-numeric series. "
                "No action taken."
            )
            raise ValueError
        bin_ranges = StatsUtils.get_binned_quantiles(numeric_series, quantiles)
        extra_margin_upper_limit = 1e-10
        x, y = StatsUtils.convert_numeric_series_to_binned_series(
            numeric_series, bin_ranges, extra_margin_upper_limit
        )
        binned_series, numeric_range_to_bin_map = x, y
        (
            woe_series,
            category_to_woe_map,
            iv,
        ) = self.convert_categorical_series_to_woe_series(binned_series, target)
        numeric_to_woe_map = category_to_woe_map.copy()
        var_label = numeric_series.name
        numeric_to_woe_map.rename(columns={var_label: var_label + "_cat"}, inplace=True)
        numeric_to_woe_map = pd.merge(
            numeric_range_to_bin_map,
            numeric_to_woe_map,
            on=var_label + "_cat",
            how="inner",
        )
        return woe_series, numeric_to_woe_map, iv

    def convert_original_dataframe_to_woe_dataframe_from_scratch(
        self, dataframe: pd.DataFrame, target: pd.Series, max_bins: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Convert dataframe to weight-of-evidence dataframe.

        Takes a dataframe with categorical and numeric variables and converts each value
        to a WOE. The function also returns the IV value for each variable in the list.
        This can then be used in univariate analysis.
        If the dataframe has the target variable in it, it is excluded from the calculation
        :param dataframe:
        :param target:
        :param max_bins:
        :return:
        """
        dataframe_copy = dataframe.copy()
        target_label = target.name
        # Check if target variable is in dataframe and remove since WOE isn't
        # needed for this variable
        if target_label in dataframe_copy:
            dataframe_copy.drop(columns=[target_label], inplace=True)
        woe_dataframe = pd.DataFrame()
        iv_dataframe = pd.DataFrame()
        woe_dataframe[target_label] = target
        var_list = []
        iv_list = []

        variable_to_woe_map = pd.DataFrame()
        for series in dataframe_copy:
            var_list.append(series)
            # Numeric
            if is_numeric_dtype(dataframe_copy[series]):
                quantiles = max_bins
                if dataframe_copy[series].isnull().values.any():
                    quantiles -= 1
                x, y, iv = self.convert_numeric_series_to_woe_series_quantiles_approach(
                    dataframe_copy[series], target, quantiles
                )

                woe_dataframe[series] = x
                numeric_to_woe_map = y

                num_bins = numeric_to_woe_map.shape[0]
                temp_dataframe = pd.DataFrame()
                temp_dataframe["variable"] = [series] * num_bins
                temp_dataframe["type"] = ["numeric"] * num_bins
                temp_dataframe["bin"] = range(1, num_bins + 1)
                temp_dataframe["lower_limit"] = numeric_to_woe_map["lower_limit"].copy()
                temp_dataframe["upper_limit"] = numeric_to_woe_map["upper_limit"].copy()
                temp_dataframe["category"] = numeric_to_woe_map[series + "_cat"].copy()
                temp_dataframe["woe"] = numeric_to_woe_map["woe"].copy()

            # Categorical
            else:
                (
                    woe_dataframe[series],
                    category_to_woe_map,
                    iv,
                ) = self.convert_categorical_series_to_woe_series(
                    dataframe_copy[series], target
                )
                num_bins = category_to_woe_map.shape[0]
                temp_dataframe = pd.DataFrame()
                temp_dataframe["variable"] = [series] * num_bins
                temp_dataframe["type"] = ["categorical"] * num_bins
                temp_dataframe["bin"] = range(1, num_bins + 1)
                temp_dataframe["lower_limit"] = [np.nan] * num_bins
                temp_dataframe["upper_limit"] = [np.nan] * num_bins
                temp_dataframe["category"] = category_to_woe_map[series].copy()
                temp_dataframe["woe"] = category_to_woe_map["woe"].copy()

            variable_to_woe_map = pd.concat([variable_to_woe_map, temp_dataframe])
            iv_list.append(iv)
        variable_to_woe_map.reset_index(drop=True, inplace=True)

        iv_dataframe["variable"] = var_list
        iv_dataframe["iv"] = iv_list
        return woe_dataframe, variable_to_woe_map, iv_dataframe

    def convert_original_dataframe_to_woe_dataframe_use_map(
        self, dataframe: pd.DataFrame, variable_to_woe_map: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert numeric series to weight-of-evidence series using map.

        Takes a dataframe with categorical and numeric variables and converts each value
        to a WOE via a mapping table. The function also returns the IV value for each variable
        in the list.

        This can then be used in univariate analysis.
        If the dataframe has variables not in the map, no conversion is done for those variables
        :param dataframe:
        :param variable_to_woe_map:
        :return:
        """
        var_list = list(variable_to_woe_map["variable"].unique())
        do_not_convert_var_list = list(set(dataframe.columns) - set(var_list))
        dataframe_variables_to_convert = dataframe[var_list].copy()
        props = DataFrameProperties()
        numeric_var_list = props.get_all_dataframe_numeric_object(
            dataframe_variables_to_convert
        )

        woe_dataframe = pd.DataFrame()
        for var_label in dataframe.columns:

            if var_label in do_not_convert_var_list:
                woe_dataframe[var_label] = dataframe[var_label].copy()

            else:
                temp_df = pd.DataFrame()
                bin_ranges = variable_to_woe_map[
                    variable_to_woe_map["variable"] == var_label
                ].copy()
                if var_label in numeric_var_list:
                    numeric_series = dataframe[var_label]
                    # Set upper margin to zero since an upper margin is already created in
                    # bin_ranges (avoid duplication of margin)
                    extra_margin_upper_limit = 0
                    (
                        binned_series,
                        _,
                    ) = StatsUtils.convert_numeric_series_to_binned_series(
                        numeric_series, bin_ranges, extra_margin_upper_limit
                    )
                    temp_df["category"] = binned_series
                else:
                    temp_df["category"] = dataframe[var_label].copy()
                    if temp_df[
                        "category"
                    ].isnull().values.any() and "[Missing]" in list(
                        bin_ranges["category"]
                    ):
                        temp_df["category"].fillna("[Missing]", inplace=True)
                temp_df = pd.merge(temp_df, bin_ranges, on="category", how="left")
                woe_dataframe[var_label] = temp_df["woe"]
            # In case some variables failed to map, set WOE to zero
            woe_dataframe[var_label].fillna(0, inplace=True)
        return woe_dataframe
