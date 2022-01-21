"""Extended Pandas Series class."""
from modellicity.src.modellicity.extended_pandas.extended_properties.series_properties import (
    SeriesProperties,
)
from modellicity.src.modellicity.extended_pandas.extended_actions.series_actions import SeriesActions
from modellicity.src.modellicity.settings import settings
from typing import Any, List

import pandas as pd


class ExtendedSeries(pd.Series):
    """Extended Pandas Series class."""

    def __init__(self, *args, **kwargs):
        """Construct ExtendedSeries class."""
        super(ExtendedSeries, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        """:return: ExtendedSeries object."""
        return ExtendedSeries

    @staticmethod
    def cast(s: pd.Series):
        """Cast a pandas Series to an ExtendedSeries without copying data."""
        if not isinstance(s, pd.Series):
            raise TypeError("Only a Series can be cast to an ExtendedSeries")

        s.__class__ = ExtendedSeries
        return s

    """
    Series Properties:
    """

    def is_all_datetime_object(self) -> bool:
        """Provide ExtendedSeries wrapper for is_all_series_datetime_object."""
        return SeriesProperties.is_all_series_datetime_object(self)

    def is_all_datetime_format(self, formats=settings.OPTIONS["date_formats"]) -> bool:
        """Provide ExtendedSeries wrapper for is_all_series_datetime_format."""
        return SeriesProperties.is_all_series_datetime_format(self, formats)

    def is_all_numeric_object(self) -> bool:
        """Provide ExtendedSeries wrapper for is_all_series_numeric_object."""
        return SeriesProperties.is_all_series_numeric_object(self)

    def is_all_numeric_format(self) -> bool:
        """Provide ExtendedSeries wrapper for is_all_series_numeric_format."""
        return SeriesProperties.is_all_series_numeric_format(self)

    def is_any_missing(self) -> bool:
        """Provide ExtendedSeries wrapper for is_any_series_missing."""
        return SeriesProperties.is_any_series_missing(self)

    def is_all_missing(self) -> bool:
        """Provide ExtendedSeries wrapper for is_all_series_missing."""
        return SeriesProperties.is_all_series_missing(self)

    def get_not_missing(self) -> pd.Series:
        """Provide ExtendedSeries wrapper for get_series_not_missing."""
        return SeriesProperties.get_series_not_missing(self)

    def get_percent_missing(self) -> float:
        """Provide ExtendedSeries wrapper for get_series_percent_missing."""
        return SeriesProperties.get_series_percent_missing(self)

    def get_num_missing(self) -> int:
        """Provide ExtendedSeries wrapper for get_series_num_missing."""
        return SeriesProperties.get_series_num_missing(self)

    def get_object_types(self) -> List[Any]:
        """Provide ExtendedSeries wrapper for get_series_object_types."""
        return SeriesProperties.get_series_object_types(self)

    """
    Series Actions:
    """

    def convert_to_datetime_object(self) -> pd.Series:
        """Provide ExtendedSeries wrapper for convert_series_to_datetime_object."""
        return SeriesActions.convert_series_to_datetime_object(self)

    def convert_to_numeric_object(self) -> pd.Series:
        """Provide ExtendedSeries wrapper for convert_series_to_numeric_object."""
        return SeriesActions.convert_series_to_numeric_object(self)
