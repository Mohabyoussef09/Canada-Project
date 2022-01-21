"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
from typing import List

import logging
import numpy as np
import pandas as pd

from modellicity.src.modellicity.extended_pandas.extended_properties.dataframe_properties import (
    DataFrameProperties,
)
from modellicity.src.modellicity.extended_pandas.extended_actions.series_actions import SeriesActions
from modellicity.src.modellicity.logging.log import log_function_call, model_logger
from modellicity.src.modellicity.settings import settings

log = logging.getLogger(__name__)


class DataFrameActions(object):
    """DataFrameActions class."""

    @staticmethod
    @log_function_call(model_logger)
    def convert_dataframe_to_datetime_object(
        df: pd.DataFrame,
        cols: List[str],
        date_formats: List[str] = settings.OPTIONS["date_formats"],
    ) -> pd.DataFrame:
        """Convert all series entries of the dataframe to datetime objects.

        Iterate through all columns in dataframe and convert the respective entries to
        datetime entry objects.

        :param df: The pandas dataframe to be processed.
        :param cols: A list of column headings in df to be converted to datetime objects.
        :param date_formats: A list of accepted date formats.
        :return: Modified dataframe where the subset of columns labelled by cols are
                 converted to datetime object entries.
        """
        if cols is None:
            cols = df.columns

        df_new = df.copy()
        for series in cols:
            df_new[series] = SeriesActions.convert_series_to_datetime_object(
                df_new[series], date_formats
            )
        return df_new

    @staticmethod
    @log_function_call(model_logger)
    def convert_dataframe_to_numeric_object(
        df: pd.DataFrame, cols: List[str] = None
    ) -> pd.DataFrame:
        """Convert all series entries of the dataframe to numeric objects if possible.

        Iterate through all columns in dataframe and convert the respective entries to
        numeric entry objects.

        :param df: The pandas dataframe to be processed.
        :param cols: A list of column headings in df to be converted to numeric objects.
        :return: Modified dataframe where the subset of columns labelled by cols are
                 converted to numeric object entries.
        """
        if cols is None:
            cols = df.columns

        df_new = df.copy()
        for series in cols:
            df_new[series] = SeriesActions.convert_series_to_numeric_object(
                df_new[series]
            )
        return df_new

    @staticmethod
    @log_function_call(model_logger)
    def remove_empty_cols(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove all empty columns from dataframe.

        :param df: The pandas dataframe to be processed.
        :return: The dataframe with no empty columns.
        """
        empty_cols = [col for col in df.columns if df[col].isnull().all()]
        df.drop(empty_cols, axis=1, inplace=True)

        return df

    @staticmethod
    @log_function_call(model_logger)
    def remove_n_unique_value_variables(
        df: pd.DataFrame, num_unique: int
    ) -> pd.DataFrame:
        """Remove n-unique value variables from dataframe.

        Obtains all variables that have n-unique value variables and removes those
        variables from the dataframe.

        :param df: The pandas dataframe to be processed.
        :param num_unique: Number of the largest number of unique elements
                           that can be present.
        :return: Modified dataframe where the columns with num_unique number of
                 unique elements are removed.
        """
        df_new = df.copy()
        drop_columns = DataFrameProperties.get_dataframe_n_value_variables(
            df, num_unique
        )
        return df_new.drop(columns=drop_columns)

    @staticmethod
    @log_function_call(model_logger)
    def remove_high_concentration_variables(
        df: pd.DataFrame, concentration_threshold: float
    ) -> pd.DataFrame:
        """Remove high-concentration variables.

        Obtains all variables that have only high concentration-value variables and removes those
        variables from the dataframe.

        :param df: The pandas dataframe to be processed.
        :param concentration_threshold: Percentage of the largest number of unique elements
                                        that can be present.
        :return: Modified dataframe where the columns with concentration_threshold number of
                 unique elements are removed.
        """
        df_new = df.copy()
        drop_columns = DataFrameProperties.get_dataframe_high_concentration_variables(
            df, concentration_threshold
        ).keys()
        return df_new.drop(columns=drop_columns)

    @staticmethod
    @log_function_call(model_logger)
    def floor_and_cap_dataframe(
        df: pd.DataFrame,
        lower_percentile_threshold: float = 1,
        upper_percentile_threshold: float = 99,
    ) -> pd.DataFrame:
        """
        Floor-and-cap entries in dataframe.

        Given a dataframe, treat the dataframe by performing a floor-and-cap operation
        to the entries.

        :param df:
        :param lower_percentile_threshold:
        :param upper_percentile_threshold:
        :return:
        """
        df_numeric = df[DataFrameProperties.get_all_dataframe_numeric_object(df)]
        df_final = df_numeric.copy()

        for var in df_numeric:
            min_val = np.nanmin(df_numeric[var])
            max_val = np.nanmax(df_numeric[var])

            # We floor for lower level percentile and ceiling for the upper level percentile
            # to add a slightly higher range to preserve the distribution, as long as the
            # minimum and maximum values are not exceeded
            lower_level_percentile = max(
                min_val,
                np.floor(np.nanpercentile(df_numeric[var], lower_percentile_threshold)),
            )
            upper_level_percentile = min(
                max_val,
                np.ceil(np.nanpercentile(df_numeric[var], upper_percentile_threshold)),
            )

            df_final[var] = np.maximum(
                lower_level_percentile,
                np.minimum(upper_level_percentile, df_numeric[var]),
            )

            log.info(f"Processing outlier variable: {var}")
            log.info(
                f"Percentile {lower_percentile_threshold}: {lower_level_percentile}"
            )
            log.info(
                f"Percentile {upper_percentile_threshold}: {upper_level_percentile}"
            )

        return df_final
