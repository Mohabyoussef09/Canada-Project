"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
from typing import List
import pandas as pd

from modellicity.src.modellicity.extended_pandas.extended_actions.entry_actions import EntryActions
from modellicity.src.modellicity.settings import settings


class SeriesActions(object):
    """General manipulations of series objects in Pandas."""

    @staticmethod
    def convert_series_to_datetime_object(
        series: pd.Series, date_formats: List[str] = settings.OPTIONS["date_formats"],
    ) -> pd.Series:
        """
        Convert the entries in a series to datetime objects if possible.

        :param series: The pandas series to be processed.
        :param date_formats: A list of accepted date formats.
        :return: Returns the series converted to have datetime object entries if each entry can
                 be converted to datetime. Otherwise, return the original series.
        """
        datetime_list = []
        for _, entry in series.items():
            datetime_list.append(
                EntryActions.convert_entry_to_datetime_object(entry, date_formats)
            )
        return pd.Series(datetime_list)

    @staticmethod
    def convert_series_to_numeric_object(series: pd.Series) -> pd.Series:
        """
        Convert the entries in a series to numeric objects if possible.

        :param series: The pandas series to be processed.
        :return: Returns the series converted to have numeric object entries if each entry can
                 be converted to numeric. Otherwise, return the original series.
        """
        numeric_list = []
        for _, entry in series.items():
            numeric_list.append(EntryActions.convert_entry_to_numeric_object(entry))
        return pd.Series(numeric_list)
