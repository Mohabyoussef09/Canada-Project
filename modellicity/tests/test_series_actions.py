"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import datetime
import unittest
import numpy as np
import pandas as pd

from modellicity.extended_pandas.extended_actions import series_actions
from modellicity.settings import settings


class TestSeriesActions(unittest.TestCase):
    """Unit tests for SeriesActions class."""

    @classmethod
    def setup_class(cls) -> None:
        """:return: None."""
        cls.utils = series_actions.SeriesActions()

    def test_convert_series_to_numeric_object(self) -> None:
        """
        Tests for convert_series_to_numeric_object.

        :return: None
        """
        numeric_series = pd.Series(np.array([1, 2, 3, 4]))
        numeric_series_converted = self.utils.convert_series_to_numeric_object(
            numeric_series)
        assert type(numeric_series_converted[0]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_series_converted[1]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_series_converted[2]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_series_converted[3]) in settings.OPTIONS["numeric_types"]

        numeric_str_int_series = pd.Series(np.array(["1", "2", "3", "4"]))
        numeric_str_int_series_converted = self.utils.convert_series_to_numeric_object(
            numeric_str_int_series)
        assert type(numeric_str_int_series_converted[0]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_str_int_series_converted[1]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_str_int_series_converted[2]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_str_int_series_converted[3]) in settings.OPTIONS["numeric_types"]

        numeric_float_int_series = pd.Series(np.array(["1.1", "2.2", "3.3", "4.4"]))
        numeric_float_int_series_converted = self.utils.convert_series_to_numeric_object(
            numeric_float_int_series)
        assert type(numeric_float_int_series_converted[0]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_float_int_series_converted[1]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_float_int_series_converted[2]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_float_int_series_converted[3]) in settings.OPTIONS["numeric_types"]

        numeric_int_float_str_series = pd.Series(np.array(["1.2", 1.2, "AB", "55"]))
        numeric_int_float_str_series_converted = self.utils.convert_series_to_numeric_object(
            numeric_int_float_str_series
        )
        assert type(numeric_int_float_str_series_converted[0]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_int_float_str_series_converted[1]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_int_float_str_series_converted[2]) not in \
            settings.OPTIONS["numeric_types"]
        assert type(numeric_int_float_str_series_converted[3]) in settings.OPTIONS["numeric_types"]

    def test_convert_series_to_datetime_object(self) -> None:
        """
        Tests for convert_series_to_dataframe_object.

        :return: None.
        """
        non_datetime_data = np.array(["SHATCH", "1999-08-08", "", "2005/01/01"])
        non_datetime_series = pd.Series(non_datetime_data)

        new_non_datetime_series = self.utils.convert_series_to_datetime_object(non_datetime_series)

        assert isinstance(new_non_datetime_series[1], datetime.datetime)
        assert isinstance(new_non_datetime_series[3], datetime.datetime)

        datetime_data = np.array(["2016-01-01", "2015-05-05", "2017-05-28"])
        datetime_series = pd.Series(datetime_data)

        converted_datetime_series = self.utils.convert_series_to_datetime_object(datetime_series)

        assert isinstance(converted_datetime_series[0], datetime.datetime)
        assert isinstance(converted_datetime_series[1], datetime.datetime)
        assert isinstance(converted_datetime_series[2], datetime.datetime)
