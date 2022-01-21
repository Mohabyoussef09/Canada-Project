"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
from typing import Any, List

import pandas as pd

from pandas.api.types import is_numeric_dtype
from modellicity.src.modellicity.settings import settings
from modellicity.src.modellicity.extended_pandas.extended_properties.entry_properties import (
    EntryProperties,
)


class SeriesProperties(object):
    """General properties of series objects in Pandas."""

    @staticmethod
    def is_all_series_datetime_format(series: pd.Series, formats: List[str]) -> bool:
        """
        Determine if all series entries are datetime format.

        :param series: The series to be processed.
        :param formats: All accepted formats to be deemed as datetime formats.
        :return: True if all entries in series are datetime-format type and False otherwise.
        """
        for entry in series:
            if (
                not EntryProperties.is_entry_datetime_format(entry, formats)
                and entry not in settings.OPTIONS["missing_types"]
            ):
                return False
        return True

    @staticmethod
    def is_all_series_numeric_format(
        series: pd.Series, formats: List[str] = settings.OPTIONS["numeric_types"]
    ) -> bool:
        """
        Determine if all series entries are numeric format.

        :param series: The series to be processed.
        :param formats: List of accepted numeric-type formats.
        :return: True if all entries in series are numeric-format type and False otherwise.
        """
        for entry in series:
            if not EntryProperties.is_entry_numeric_format(entry, formats):
                return False
        return True

    @staticmethod
    def is_all_series_datetime_object(series: pd.Series) -> bool:
        """
        Determine if all series entries are datetime objects.

        :param series: The series to be processed.
        :return: True if all entries in series are datetime-object type and False otherwise.
        """
        for entry in series:
            if (
                not EntryProperties.is_entry_datetime_object(entry)
                and entry not in settings.OPTIONS["missing_types"]
            ):
                return False
        return True

    @staticmethod
    def is_all_series_numeric_object(series: pd.Series) -> bool:
        """
        Determine if all series entries are numeric objects.

        :param series: The series to be processed.
        :return: True if all entries in series are numeric-object type and False otherwise.
        """
        return is_numeric_dtype(series) and not SeriesProperties.is_all_series_missing(
            series
        )

    @staticmethod
    def is_any_series_missing(series: pd.Series) -> bool:
        """
        Determine if there are any missing values in the given series.

        :param series: The series to be processed.
        :return: True if any entries in series missing type and False otherwise.
        """
        return bool(series.isin(settings.OPTIONS["missing_types"]).any())

    @staticmethod
    def is_all_series_missing(series: pd.Series) -> bool:
        """
        Determine if series is composed of all missing values.

        :param series: The series to be processed.
        :return: True if all entries in series are missing type and False otherwise.
        """
        return bool(series.isin(settings.OPTIONS["missing_types"]).all())

    @staticmethod
    def get_series_not_missing(series: pd.Series) -> pd.Series:
        """
        Return a subset of the given series that contains only the non-missing entries.

        :param series: The series to be processed.
        :return: A subset of the series that contains entries of non-missing type.
        """
        return series.where(~series.isin(settings.OPTIONS["missing_types"])).dropna()

    @staticmethod
    def get_series_num_missing(series: pd.Series) -> int:
        """
        Get total number of missing entries in a given series.

        :param series: The series to be processed.
        :return: The number of missing entries of the series.
        """
        # Total value counts for missing values.
        is_missing_counts = series.isin(
            settings.OPTIONS["missing_types"]
        ).value_counts()

        # If there are missing values, return how many there are.
        if True in is_missing_counts.index:
            return is_missing_counts[True]
        # Otherwise, if this index is not present, there are no missing values.
        return 0

    @staticmethod
    def get_series_percent_missing(series: pd.Series) -> float:
        """
        Get percent missing of entries in a given series.

        :param series: The series to be processed.
        :return: The percentage of missing entries of the series.
        """
        return SeriesProperties.get_series_num_missing(series) / len(series)

    @staticmethod
    def get_series_object_types(series: pd.Series) -> List[Any]:
        """
        Get a list of types for each entry in the series.

        :param series: The series to be processed.
        :return: A list of all object types for the given series.
        """
        return [type(entry) for entry in series]
