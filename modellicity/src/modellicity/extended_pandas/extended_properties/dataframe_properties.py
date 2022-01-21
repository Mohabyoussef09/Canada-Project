"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
from typing import Dict, List

import logging
import numpy as np
import pandas as pd

from modellicity.src.modellicity.extended_pandas.extended_properties.series_properties import (
    SeriesProperties,
)
from modellicity.src.modellicity.logging.log import log_function_call, model_logger
from modellicity.src.modellicity.settings import settings
from modellicity.src.modellicity.stats.stats_utils import StatsUtils

log = logging.getLogger(__name__)


class DataFrameProperties(object):
    """General properties for Pandas dataframe objects."""

    @staticmethod
    @log_function_call(model_logger)
    def get_all_dataframe_categorical(df: pd.DataFrame) -> List[str]:
        """
        Return a list of each series name that is considered to be cateogrical.

        :param df: The pandas dataframe to be processed.
        :return: A list of all series in dataframe that contain all categorical-type entries.
        """
        return list(
            df.select_dtypes(include=settings.OPTIONS["categorical_types"]).columns
        )

    @staticmethod
    @log_function_call(model_logger)
    def get_all_dataframe_numeric_object(df: pd.DataFrame) -> List[str]:
        """
        Return a list of each series name that is of object numeric.

        :param df: The pandas dataframe to be processed.
        :return: A list of all series in dataframe that contain all numeric object entries.
        """
        return list(df.select_dtypes(include=settings.OPTIONS["numeric_types"]).columns)

    @staticmethod
    @log_function_call(model_logger)
    def is_any_dataframe_datetime_object(df: pd.DataFrame) -> bool:
        """
        Return True if any of the series in the given dataframe are datetime.

        :param df: The pandas dataframe to be processed.
        :return: Return True if any series in the dataframe are datetime objects and
                 False otherwise.
        """
        for series in df.columns:
            if SeriesProperties.is_all_series_datetime_object(df[series]):
                return True
        return False

    @staticmethod
    @log_function_call(model_logger)
    def get_all_dataframe_numeric_format(
        df: pd.DataFrame, formats: List[str] = settings.OPTIONS["numeric_types"]
    ) -> List[str]:
        """
        Return a list of each series name that is of numeric format.

        :param df: The pandas dataframe to be processed.
        :param formats: List of accepted numeric-type formats.
        :return: A list of all series in dataframe that contain all numeric format entries.
        """
        column_names: List[str] = []
        for series in df.columns:
            if SeriesProperties.is_all_series_numeric_format(df[series], formats):
                column_names.append(series)
        return column_names

    @staticmethod
    @log_function_call(model_logger)
    def get_all_dataframe_datetime_format(
        df: pd.DataFrame, formats: List[str]
    ) -> List[str]:
        """
        Return a list of each series name that is of datetime format.

        :param df: The pandas dataframe to be processed.
        :param formats: All accepted formats to be deemed as datetime formats.
        :return: A list of all series in dataframe that contain all datetime format entries.
        """
        column_names: List[str] = []
        for series in df.columns:
            if SeriesProperties.is_all_series_datetime_format(df[series], formats):
                column_names.append(series)
        return column_names

    @staticmethod
    @log_function_call(model_logger)
    def get_all_dataframe_datetime_object(df: pd.DataFrame) -> List[str]:
        """
        Return a list of each series name that is of object datetime.

        :param df: The pandas dataframe to be processed.
        :return: A list of all series in dataframe that contain all datetime object entries.
        """
        column_names: List[str] = []
        for series in df.columns:
            if SeriesProperties.is_all_series_datetime_object(df[series]):
                column_names.append(series)
        return column_names

    @staticmethod
    @log_function_call(model_logger)
    def get_all_dataframe_missing(df: pd.DataFrame) -> List[str]:
        """
        Return the labels of each series that have missing data in the dataframe.

        :param df: The pandas dataframe to be processed.
        :return: A list of all series in dataframe that contain all missing entries.
        """
        return [
            series
            for series in df.columns
            if df[series].isin(settings.OPTIONS["missing_types"]).any()
        ]

    @staticmethod
    @log_function_call(model_logger)
    def get_all_dataframe_no_missing(df: pd.DataFrame) -> List[str]:
        """
        Return the labels of each series that have no missing data in the dataframe.

        :param df: The pandas dataframe to be processed.
        :return: A list of all series in dataframe that contain no missing entries.
        """
        return [
            series
            for series in df.columns
            if not df[series].isin(settings.OPTIONS["missing_types"]).any()
        ]

    @staticmethod
    @log_function_call(model_logger)
    def get_dataframe_num_missing(df: pd.DataFrame) -> Dict[str, int]:
        """
        Get total number of missing entries in a given dataframe across all series in the dataframe.

        :param df: The pandas dataframe to be processed.
        :return: A dictionary where the keys contain all series in dataframe that contain missing
                 entries and the values contain the number of missing entries from that series.
        """
        missing_value_dict: Dict[str, int] = dict()
        for series in df.columns:
            missing_value_dict[series] = SeriesProperties.get_series_num_missing(
                df[series]
            )
        return missing_value_dict

    @staticmethod
    @log_function_call(model_logger)
    def get_dataframe_percent_missing(df: pd.DataFrame) -> Dict[str, float]:
        """
        Get the percent of missing entries in a given dataframe across all series in the dataframe.

        :param df: The pandas dataframe to be processed.
        :return: A dictionary where the keys contain all series in dataframe that contain missing
                 entries and the values contain the percent missing entries from that series.
        """
        missing_value_dict: Dict[str, float] = dict()
        for series in df.columns:
            missing_value_dict[series] = SeriesProperties.get_series_percent_missing(
                df[series]
            )
        return missing_value_dict

    @staticmethod
    @log_function_call(model_logger)
    def get_dataframe_n_value_variables(df: pd.DataFrame, num_unique: int) -> List[str]:
        """
        Obtain a list of all variables with "num_unique" unique values.

        :param df:
        :param num_unique:
        :return:
        """
        n_value_variables: List[str] = []
        for series_label in df.columns:
            series_num_unique = len(df[series_label].unique())
            if series_num_unique == num_unique:
                n_value_variables.append(series_label)
        return n_value_variables

    @staticmethod
    @log_function_call(model_logger)
    def get_dataframe_high_concentration_variables(
        df: pd.DataFrame, concentration_threshold: float
    ) -> Dict[str, float]:
        """
        Obtain a dictionary of all variables in dataframe deemed to have a high concentration.

        :return: A dictionary object where the keys correspond to the name of the variable found
                 to be above the concentration threshold and the value is the percentage that
                 corresponds to the number of unique elements over the length of the series.
        """
        high_concentration_variables: Dict[str, float] = dict()
        for series_label in df.columns:
            num_unique = len(df[series_label].unique())
            concentration_percent = num_unique / len(df[series_label])

            if concentration_percent <= concentration_threshold:
                high_concentration_variables[series_label] = concentration_percent

        return high_concentration_variables

    @staticmethod
    @log_function_call(model_logger)
    def get_dataframe_outliers(
        df: pd.DataFrame, outlier_threshold: float = 3.29
    ) -> List[str]:
        """
        Obtain a list of all outliers in dataframe according to an outlier threshold value.

        :param df: The pandas dataframe to be processed.
        :param outlier_threshold: The number of standard deviations away from the mean an entry
                                  needs to be in order to be deemed an outlier
        :return: A list of all series in dataframe that are outlier entries.
        """
        df_numeric = df[DataFrameProperties.get_all_dataframe_numeric_object(df)]
        df_normalized = StatsUtils.normalize_dataframe(df_numeric)

        outlier_list: List[str] = []
        for variable in list(df_normalized.columns):
            max_abs_val_norm = max(
                abs(np.nanmin(df_normalized[variable])),
                abs(np.nanmax(df_normalized[variable])),
            )
            if max_abs_val_norm > outlier_threshold:
                outlier_list.append(variable)
        return outlier_list
