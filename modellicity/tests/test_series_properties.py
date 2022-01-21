"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import datetime
import unittest
import numpy as np
import pandas as pd

from modellicity.extended_pandas.extended_properties import series_properties
from modellicity.settings import settings


class TestSeriesProperties(unittest.TestCase):
    """Unit tests for SeriesProperties class."""

    @classmethod
    def setup_class(cls) -> None:
        """:return: None."""
        cls.utils = series_properties.SeriesProperties()

    def test_is_all_series_datetime_object(self) -> None:
        """
        Tests for is_all_series_datetime_object.

        :return: None.
        """
        non_datetime_data = np.array(["1999-08-08", "", "2005/1-1"], dtype=object)
        non_datetime_series = pd.Series(non_datetime_data)
        assert self.utils.is_all_series_datetime_object(non_datetime_series) is False

        some_datetime_data = np.array(
            ["1998-08-08", datetime.date(2015, 5, 4)], dtype=object
        )
        some_datetime_series = pd.Series(some_datetime_data)
        assert self.utils.is_all_series_datetime_object(some_datetime_series) is False

        datetime_data = np.array(
            [datetime.date(2019, 1, 1), datetime.date(1999, 5, 5)], dtype=object
        )
        datetime_series = pd.Series(datetime_data)
        assert self.utils.is_all_series_datetime_object(datetime_series) is True

        datetime_data_exceptions = np.array(
            [datetime.date(2019, 1, 1), datetime.date(1999, 5, 5), np.nan]
        )
        datetime_exceptions_series = pd.Series(datetime_data_exceptions)
        assert (
            self.utils.is_all_series_datetime_object(datetime_exceptions_series) is True
        )

    def test_is_all_series_numeric_format(self) -> None:
        """
        Tests for is_all_series_numeric_format.

        :return: None.
        """
        int_str_series = pd.Series(np.array(["100", "20"], dtype=object))
        assert self.utils.is_all_series_numeric_format(int_str_series) is True

        float_str_series = pd.Series(np.array(["20.56", "45.544"], dtype=object))
        assert self.utils.is_all_series_numeric_format(float_str_series) is True

        float_int_str_series = pd.Series(np.array(["20.56", "4"], dtype=object))
        assert self.utils.is_all_series_numeric_format(float_int_str_series) is True

        non_numeric_str_series = pd.Series(np.array(["A", "B"], dtype=object))
        assert self.utils.is_all_series_numeric_format(non_numeric_str_series) is False

    def test_is_all_series_datetime_format(self) -> None:
        """
        Tests for is_all_series_datetime_format.

        :return: None.
        """
        datetime_data = np.array(["1999-08-08", "2001-01-01", "2005-1-1"], dtype=object)
        datetime_series = pd.Series(datetime_data)
        res = self.utils.is_all_series_datetime_format(
            datetime_series, settings.OPTIONS["date_formats"]
        )
        assert res is True

        some_non_datetime_data = np.array(["TEST", "1999-01-01"], dtype=object)
        some_non_datetime_series = pd.Series(some_non_datetime_data)
        assert (
            self.utils.is_all_series_datetime_format(
                some_non_datetime_series, settings.OPTIONS["date_formats"]
            )
            is False
        )

        all_non_datetime_data = np.array(["TEST1", "TEST2", "TEST3"], dtype=object)
        all_non_datetime_series = pd.Series(all_non_datetime_data)
        assert (
            self.utils.is_all_series_datetime_format(
                all_non_datetime_series, settings.OPTIONS["date_formats"]
            )
            is False
        )

    def test_is_all_series_numeric_object(self) -> None:
        """
        Tests for is_all_series_numeric_object.

        :return: None.
        """
        integer_data = np.array([100, 5039, 39393, 9329, 293])
        integer_series = pd.Series(integer_data)
        assert self.utils.is_all_series_numeric_object(integer_series) is True

        float_data = np.array([103.33, 1232.31, 3113.131, 13.1313])
        float_series = pd.Series(float_data)
        assert self.utils.is_all_series_numeric_object(float_series) is True

        integer_mix_data = np.array([100, 392, "TEST", 23])
        integer_mix_series = pd.Series(integer_mix_data)
        assert self.utils.is_all_series_numeric_object(integer_mix_series) is False

    def test_get_series_num_missing(self) -> None:
        """
        Tests for get_series_num_missing.

        :return: None.
        """
        no_missing_data = np.array(
            ["item_1", "item_2", "item_3", "item_4"], dtype=object
        )
        no_missing_series = pd.Series(no_missing_data)
        assert self.utils.get_series_num_missing(no_missing_series) == 0

        some_missing_data = np.array([np.nan, "item_2", np.nan, "item_4"], dtype=object)
        some_missing_series = pd.Series(some_missing_data)
        assert self.utils.get_series_num_missing(some_missing_series) == 2

        all_missing_data = np.array([np.nan, np.nan, np.nan, np.nan], dtype=object)
        all_missing_series = pd.Series(all_missing_data)
        assert self.utils.get_series_num_missing(all_missing_series) == 4

    def test_is_any_series_missing(self) -> None:
        """
        Tests for is_any_series_missing.

        :return: None.
        """
        no_missing_data = np.array(
            ["item_1", "item_2", "item_3", "item_4"], dtype=object
        )
        no_missing_series = pd.Series(no_missing_data)
        assert self.utils.is_any_series_missing(no_missing_series) is False

        some_missing_data = np.array([np.nan, "item_2", np.nan, "item_4"], dtype=object)
        some_missing_series = pd.Series(some_missing_data)
        assert self.utils.is_any_series_missing(some_missing_series) is True

        all_missing_data = np.array([np.nan, np.nan, np.nan, np.nan], dtype=object)
        all_missing_series = pd.Series(all_missing_data)
        assert self.utils.is_any_series_missing(all_missing_series) is True

    def test_is_all_series_missing(self) -> None:
        """
        Tests for is_any_series_missing.

        :return: None.
        """
        no_missing_data = np.array(
            ["item_1", "item_2", "item_3", "item_4"], dtype=object
        )
        no_missing_series = pd.Series(no_missing_data)
        assert self.utils.is_all_series_missing(no_missing_series) is False

        some_missing_data = np.array([np.nan, "item_2", np.nan, "item_4"], dtype=object)
        some_missing_series = pd.Series(some_missing_data)
        assert self.utils.is_all_series_missing(some_missing_series) is False

        all_missing_data = np.array([np.nan, np.nan, np.nan, np.nan], dtype=object)
        all_missing_series = pd.Series(all_missing_data)
        assert self.utils.is_all_series_missing(all_missing_series) is True

    def test_get_series_not_missing(self) -> None:
        """
        Tests for get_series_not_missing.

        :return: None.
        """
        test_1 = pd.Series(np.array(["2018-01-01", "2017-01-01", "2016-01-01"]))
        test_2 = pd.Series(np.array(["a", "b", "c"]))
        test_3 = pd.Series(np.array(["2018-01-01", "", np.nan]))
        test_4 = pd.Series(np.array([datetime.datetime.now(),
                                     datetime.datetime.now(),
                                     datetime.datetime.now()]))
        test_5 = pd.Series(np.array([np.nan, np.nan, np.nan]))
        test_6 = pd.Series(np.array(["", "", 5]))

        assert len(self.utils.get_series_not_missing(test_1)) == 3
        assert len(self.utils.get_series_not_missing(test_2)) == 3
        assert len(self.utils.get_series_not_missing(test_3)) == 2
        assert len(self.utils.get_series_not_missing(test_4)) == 3
        assert self.utils.get_series_not_missing(test_5).empty is True
        assert len(self.utils.get_series_not_missing(test_6)) == 1

    def test_get_series_percent_missing(self) -> None:
        """
        Tests for get_series_percent_missing.

        :return: None.
        """
        no_missing_data = np.array(
            ["item_1", "item_2", "item_3", "item_4"], dtype=object
        )
        no_missing_series = pd.Series(no_missing_data)
        assert self.utils.get_series_percent_missing(no_missing_series) == 0.0

        some_missing_data = np.array([np.nan, "item_2", np.nan, "item_4"], dtype=object)
        some_missing_series = pd.Series(some_missing_data)
        assert self.utils.get_series_percent_missing(some_missing_series) == 0.5

        all_missing_data = np.array([np.nan, np.nan, np.nan, np.nan], dtype=object)
        all_missing_series = pd.Series(all_missing_data)
        assert self.utils.get_series_percent_missing(all_missing_series) == 1.0

    def test_get_series_object_types(self) -> None:
        """
        Tests for get_series_object_types.

        :return: None.
        """
        int_data = np.array([1, 2, 3, 4, 5], dtype=object)
        int_series = pd.Series(int_data)
        assert self.utils.get_series_object_types(int_series) == [
            int,
            int,
            int,
            int,
            int,
        ]

        numeric_data = np.array([1, 2.0, 3.5, 4, 5.33], dtype=object)
        numeric_series = pd.Series(numeric_data)
        assert self.utils.get_series_object_types(numeric_series) == [
            int,
            float,
            float,
            int,
            float,
        ]

        mixed_data = np.array([1, 2.0, datetime.datetime.now(), "test", np.nan])
        mixed_series = pd.Series(mixed_data)
        assert self.utils.get_series_object_types(mixed_series) == [
            int,
            float,
            datetime.datetime,
            str,
            float,
        ]
