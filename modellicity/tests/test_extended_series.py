"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import datetime
import numpy as np
import unittest

from modellicity.extended_pandas import ExtendedSeries
from modellicity.settings import settings


class TestExtendedSeries(unittest.TestCase):
    """Unit tests for ExtendedSeries class."""

    @classmethod
    def setup_class(cls) -> None:
        """:return: None."""
        cls.utils = ExtendedSeries()

    def test_is_all_datetime_object(self) -> None:
        """
        Tests for is_all_datetime_object.

        :return: None
        """
        non_datetime_data = np.array(["1999-08-08", "", "2005/1-1"], dtype=object)
        non_datetime_series = ExtendedSeries(non_datetime_data)
        assert non_datetime_series.is_all_datetime_object() is False

        some_datetime_data = np.array(
            ["1998-08-08", datetime.date(2015, 5, 4)], dtype=object
        )
        some_datetime_series = ExtendedSeries(some_datetime_data)
        assert some_datetime_series.is_all_datetime_object() is False

        datetime_data = np.array(
            [datetime.date(2019, 1, 1), datetime.date(1999, 5, 5)], dtype=object
        )
        datetime_series = ExtendedSeries(datetime_data)
        assert datetime_series.is_all_datetime_object() is True

        datetime_data_exceptions = np.array(
            [datetime.date(2019, 1, 1), datetime.date(1999, 5, 5), np.nan]
        )
        datetime_exceptions_series = ExtendedSeries(datetime_data_exceptions)
        assert (
            datetime_exceptions_series.is_all_datetime_object() is True
        )

    def test_is_all_datetime_format(self) -> None:
        """
        Tests for is_all_datetime_format.

        :return: None
        """
        datetime_data = np.array(["1999-08-08", "2001-01-01", "2005-1-1"], dtype=object)
        datetime_series = ExtendedSeries(datetime_data)
        res = datetime_series.is_all_datetime_format()
        assert res is True

        some_non_datetime_data = np.array(["TEST", "1999-01-01"], dtype=object)
        some_non_datetime_series = ExtendedSeries(some_non_datetime_data)
        assert (
            some_non_datetime_series.is_all_datetime_format() is False
        )

        all_non_datetime_data = np.array(["TEST1", "TEST2", "TEST3"], dtype=object)
        all_non_datetime_series = ExtendedSeries(all_non_datetime_data)
        assert (
            all_non_datetime_series.is_all_datetime_format() is False
        )

    def test_is_all_numeric_object(self) -> None:
        """
        Tests for is_all_numeric_object.

        :return: None
        """
        integer_data = np.array([100, 5039, 39393, 9329, 293])
        integer_series = ExtendedSeries(integer_data)
        assert integer_series.is_all_numeric_object() is True

        float_data = np.array([103.33, 1232.31, 3113.131, 13.1313])
        float_series = ExtendedSeries(float_data)
        assert float_series.is_all_numeric_object() is True

        integer_mix_data = np.array([100, 392, "TEST", 23])
        integer_mix_series = ExtendedSeries(integer_mix_data)
        assert integer_mix_series.is_all_numeric_object() is False

    def test_is_all_numeric_format(self) -> None:
        """
        Tests for is_all_numeric_format.

        :return: None
        """
        int_str_series = ExtendedSeries(np.array(["100", "20"], dtype=object))
        assert int_str_series.is_all_numeric_format() is True

        float_str_series = ExtendedSeries(np.array(["20.56", "45.544"], dtype=object))
        assert float_str_series.is_all_numeric_format() is True

        float_int_str_series = ExtendedSeries(np.array(["20.56", "4"], dtype=object))
        assert float_int_str_series.is_all_numeric_format() is True

        non_numeric_str_series = ExtendedSeries(np.array(["A", "B"], dtype=object))
        assert non_numeric_str_series.is_all_numeric_format() is False

    def test_is_any_missing(self) -> None:
        """
        Tests for is_any_missing.

        :return: None
        """
        no_missing_data = np.array(
            ["item_1", "item_2", "item_3", "item_4"], dtype=object
        )
        no_missing_series = ExtendedSeries(no_missing_data)
        assert no_missing_series.is_any_missing() is False

        some_missing_data = np.array([np.nan, "item_2", np.nan, "item_4"], dtype=object)
        some_missing_series = ExtendedSeries(some_missing_data)
        assert some_missing_series.is_any_missing() is True

        all_missing_data = np.array([np.nan, np.nan, np.nan, np.nan], dtype=object)
        all_missing_series = ExtendedSeries(all_missing_data)
        assert all_missing_series.is_any_missing() is True

    def test_is_all_missing(self) -> None:
        """
        Tests for is_all_missing.

        :return: None
        """
        no_missing_data = np.array(
            ["item_1", "item_2", "item_3", "item_4"], dtype=object
        )
        no_missing_series = ExtendedSeries(no_missing_data)
        assert no_missing_series.is_all_missing() is False

        some_missing_data = np.array([np.nan, "item_2", np.nan, "item_4"], dtype=object)
        some_missing_series = ExtendedSeries(some_missing_data)
        assert some_missing_series.is_all_missing() is False

        all_missing_data = np.array([np.nan, np.nan, np.nan, np.nan], dtype=object)
        all_missing_series = ExtendedSeries(all_missing_data)
        assert all_missing_series.is_all_missing() is True

    def test_get_not_missing(self) -> None:
        """
        Tests for get_not_missing.

        :return: None
        """
        test_1 = ExtendedSeries(np.array(["2018-01-01", "2017-01-01", "2016-01-01"]))
        test_2 = ExtendedSeries(np.array(["a", "b", "c"]))
        test_3 = ExtendedSeries(np.array(["2018-01-01", "", np.nan]))
        test_4 = ExtendedSeries(np.array([datetime.datetime.now(),
                                          datetime.datetime.now(),
                                          datetime.datetime.now()]))
        test_5 = ExtendedSeries(np.array([np.nan, np.nan, np.nan]))
        test_6 = ExtendedSeries(np.array(["", "", 5]))

        assert len(test_1.get_not_missing()) == 3
        assert len(test_2.get_not_missing()) == 3
        assert len(test_3.get_not_missing()) == 2
        assert len(test_4.get_not_missing()) == 3
        assert test_5.get_not_missing().empty is True
        assert len(test_6.get_not_missing()) == 1

    def test_get_percent_missing(self) -> None:
        """
        Tests for get_percent_missing.

        :return: None
        """
        no_missing_data = np.array(
            ["item_1", "item_2", "item_3", "item_4"], dtype=object
        )
        no_missing_series = ExtendedSeries(no_missing_data)
        assert no_missing_series.get_percent_missing() == 0.0

        some_missing_data = np.array([np.nan, "item_2", np.nan, "item_4"], dtype=object)
        some_missing_series = ExtendedSeries(some_missing_data)
        assert some_missing_series.get_percent_missing() == 0.5

        all_missing_data = np.array([np.nan, np.nan, np.nan, np.nan], dtype=object)
        all_missing_series = ExtendedSeries(all_missing_data)
        assert all_missing_series.get_percent_missing() == 1.0

    def test_get_num_missing(self) -> None:
        """
        Tests for get_num_missing.

        :return: None
        """
        no_missing_data = np.array(
            ["item_1", "item_2", "item_3", "item_4"], dtype=object
        )
        no_missing_series = ExtendedSeries(no_missing_data)
        assert no_missing_series.get_num_missing() == 0

        some_missing_data = np.array([np.nan, "item_2", np.nan, "item_4"], dtype=object)
        some_missing_series = ExtendedSeries(some_missing_data)
        assert some_missing_series.get_num_missing() == 2

        all_missing_data = np.array([np.nan, np.nan, np.nan, np.nan], dtype=object)
        all_missing_series = ExtendedSeries(all_missing_data)
        assert all_missing_series.get_num_missing() == 4

    def test_get_object_types(self) -> None:
        """
        Tests for get_object_types.

        :return: None
        """
        int_data = np.array([1, 2, 3, 4, 5], dtype=object)
        int_series = ExtendedSeries(int_data)
        assert int_series.get_object_types() == [
            int, int, int, int, int
        ]

        mixed_data = np.array([1, 2.0, datetime.datetime.now(), "test", np.nan])
        mixed_series = ExtendedSeries(mixed_data)
        assert mixed_series.get_object_types() == [
            int, float, datetime.datetime, str, float,
        ]

    def test_convert_to_numeric_object(self) -> None:
        """
        Tests for convert_to_numeric_object.

        :return: None
        """
        numeric_series = ExtendedSeries(np.array([1, 2, 3, 4]))
        numeric_series_converted = numeric_series.convert_to_numeric_object()
        assert type(numeric_series_converted[0]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_series_converted[1]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_series_converted[2]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_series_converted[3]) in settings.OPTIONS["numeric_types"]

        numeric_str_int_series = ExtendedSeries(np.array(["1", "2", "3", "4"]))
        numeric_str_int_series_converted = numeric_str_int_series.convert_to_numeric_object()
        assert type(numeric_str_int_series_converted[0]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_str_int_series_converted[1]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_str_int_series_converted[2]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_str_int_series_converted[3]) in settings.OPTIONS["numeric_types"]

        numeric_float_int_series = ExtendedSeries(np.array(["1.1", "2.2", "3.3", "4.4"]))
        numeric_float_int_series_converted = numeric_float_int_series.convert_to_numeric_object()
        assert type(numeric_float_int_series_converted[0]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_float_int_series_converted[1]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_float_int_series_converted[2]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_float_int_series_converted[3]) in settings.OPTIONS["numeric_types"]

        numeric_int_float_str_series = ExtendedSeries(np.array(["1.2", 1.2, "AB", "55"]))
        numeric_int_float_str_series_converted = \
            numeric_int_float_str_series.convert_to_numeric_object()
        assert type(numeric_int_float_str_series_converted[0]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_int_float_str_series_converted[1]) in settings.OPTIONS["numeric_types"]
        assert type(numeric_int_float_str_series_converted[2]) not in \
            settings.OPTIONS["numeric_types"]
        assert type(numeric_int_float_str_series_converted[3]) in settings.OPTIONS["numeric_types"]

    def test_convert_to_datetime_object(self) -> None:
        """
        Test for convert_to_datetime_object.

        :return: None
        """
        date_str_series = ExtendedSeries(np.array(
            ["2018-01-01", "2018-02-01", "2014-03-12", "2013-12-01"]
        ))
        date_str_series_converted = date_str_series.convert_to_datetime_object()
        assert type(date_str_series_converted[0]) in settings.OPTIONS["date_types"]
        assert type(date_str_series_converted[1]) in settings.OPTIONS["date_types"]
        assert type(date_str_series_converted[2]) in settings.OPTIONS["date_types"]
        assert type(date_str_series_converted[3]) in settings.OPTIONS["date_types"]

        date_str_numeric_series = ExtendedSeries(np.array(
            [1, "2", "2018.01.02", "2018/01/02"]
        ))
        date_str_numeric_series_converted = date_str_numeric_series.convert_to_datetime_object()
        assert type(date_str_numeric_series_converted[0]) not in settings.OPTIONS["date_types"]
        assert type(date_str_numeric_series_converted[1]) not in settings.OPTIONS["date_types"]
        assert type(date_str_numeric_series_converted[2]) in settings.OPTIONS["date_types"]
        assert type(date_str_numeric_series_converted[3]) in settings.OPTIONS["date_types"]
