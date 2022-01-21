"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import datetime
import unittest

import numpy as np
import pandas as pd

from modellicity.extended_pandas.extended_properties import dataframe_properties
from modellicity.settings import settings


class TestDataFrameProperties(unittest.TestCase):
    """Unit tests for DataFrameProperties class."""

    @classmethod
    def setup_class(cls) -> None:
        """:return: None."""
        cls.utils = dataframe_properties.DataFrameProperties()

    def test_get_all_dataframe_missing(self) -> None:
        """
        Tests for get_all_dataframe_missing.

        :return: None.
        """
        dataframe_data = {
            "test_1": ["2018-01-01", "2017-01-01", "2016-01-01"],
            "test_2": ["a", "b", "c"],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [
                datetime.datetime.now(),
                datetime.datetime.now(),
                datetime.datetime.now(),
            ],
            "test_5": [np.nan, np.nan, np.nan],
            "test_6": ["", "", 5],
        }
        dataframe = pd.DataFrame(dataframe_data)
        assert self.utils.get_all_dataframe_missing(dataframe) == [
            "test_3",
            "test_5",
            "test_6",
        ]

    def test_get_all_dataframe_no_missing(self) -> None:
        """
        Tests for get_all_dataframe_no_missing.

        :return: None.
        """
        dataframe_data = {
            "test_1": ["2018-01-01", "2017-01-01", "2016-01-01"],
            "test_2": ["a", "b", "c"],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [
                datetime.datetime.now(),
                datetime.datetime.now(),
                datetime.datetime.now(),
            ],
            "test_5": [np.nan, np.nan, np.nan],
            "test_6": ["", "", 5],
        }
        dataframe = pd.DataFrame(dataframe_data)
        assert self.utils.get_all_dataframe_no_missing(dataframe) == [
            "test_1",
            "test_2",
            "test_4",
        ]

    def test_get_all_dataframe_numeric_object(self) -> None:
        """
        Tests for get_all_dataframe_numeric_object.

        :return: None.
        """
        dataframe_data = {
            "int_series": [5, 2, 4, 231, 31, 31],
            "float_series": [4.32, 323.32, 232.13, 131.131, 131.11111, 1.11],
            "int_float_series": [3, 3, 2, 3.4, 432.13, 311.11],
            "string_series": [
                "test_1",
                "test_2",
                "test_3",
                "test_4",
                "test_5",
                "test_6",
            ],
            "int_string_series": ["test_1", "test_2", "test_3", 3, 4, 4],
        }
        dataframe = pd.DataFrame(dataframe_data)
        assert self.utils.get_all_dataframe_numeric_object(dataframe) == [
            "int_series",
            "float_series",
            "int_float_series",
        ]

    def test_is_any_dataframe_datetime_object(self) -> None:
        """
        Tests for is_any_dateframe_datetime_object.

        :return: None.
        """
        datetime_data = {
            "test_1": ["2018-01-01", "2017-01-01", "2016-01-01"],
            "test_2": ["a", "b", "c"],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [
                datetime.datetime.now(),
                datetime.datetime.now(),
                datetime.datetime.now(),
            ],
        }
        datetime_df = pd.DataFrame(datetime_data)
        assert self.utils.is_any_dataframe_datetime_object(datetime_df) is True

        non_datetime_data = {
            "test_1": ["2018-01-01", "2017-01-01", "2016-01-01"],
            "test_2": ["a", "b", "c"],
            "test_3": ["2018-01-01", "", np.nan],
        }
        non_datetime_df = pd.DataFrame(non_datetime_data)
        assert self.utils.is_any_dataframe_datetime_object(non_datetime_df) is False

    def test_get_all_dataframe_numeric_format(self) -> None:
        """
        Tests for get_all_dataframe_numeric_format.

        :return: None.
        """
        df = pd.DataFrame(
            data={
                "X": ["100", "20"],
                "Y": ["A", "B"],
                "Z1": [20.56, 35],
                "Z2": ["20.56", "35"]
            }
        )
        res = self.utils.get_all_dataframe_numeric_format(df)
        self.assertEqual(res, ["X", "Z1", "Z2"])

    def test_get_all_dataframe_datetime_format(self) -> None:
        """
        Tests for get_all_dataframe_datetime_format.

        :return: None.
        """
        some_datetime_data = {
            "test_1": ["2018-01-01", "2017-01-01", "2016-01-01"],
            "test_2": ["a", "b", "c"],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [
                datetime.datetime.now(),
                datetime.datetime.now(),
                datetime.datetime.now(),
            ],
        }
        some_datetime_df = pd.DataFrame(some_datetime_data)
        res = self.utils.get_all_dataframe_datetime_format(
            some_datetime_df, settings.OPTIONS["date_formats"]
        )
        self.assertEqual(res, ["test_1", "test_3"])

        some_datetime_data = {
            "test_1": [20180101, 20170101, 20160101],
            "test_2": ["a", "b", "c"],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [
                datetime.datetime.now(),
                datetime.datetime.now(),
                datetime.datetime.now(),
            ],
        }
        some_datetime_df = pd.DataFrame(some_datetime_data)
        assert self.utils.get_all_dataframe_datetime_format(
            some_datetime_df, settings.OPTIONS["date_formats"]
        ) == ["test_1", "test_3"]

    def test_get_all_dataframe_datetime_object(self) -> None:
        """
        Tests for get_all_dataframe_datetime_object.

        :return: None.
        """
        some_datetime_data = {
            "test_1": ["2018-01-01", "2017-01-01", "2016-01-01"],
            "test_2": ["a", "b", "c"],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [
                datetime.datetime.now(),
                datetime.datetime.now(),
                datetime.datetime.now(),
            ],
        }
        some_datetime_df = pd.DataFrame(some_datetime_data)
        assert self.utils.get_all_dataframe_datetime_object(some_datetime_df) == [
            "test_4"
        ]

    def test_get_all_dataframe_categorical(self) -> None:
        """
        Tests for get_all_dataframe_categorical.

        :return: None.
        """
        numeric_data = {
            "test_1": [1, 2, 3, 4],
            "test_2": [5, 6, 7, 8],
            "test_3": [9, 10, 11, 12],
        }
        numeric_dataframe = pd.DataFrame(numeric_data)

        assert self.utils.get_all_dataframe_categorical(numeric_dataframe) == []

        mixed_dataframe_data = {
            "test_1": ["2018-01-01", "2017-01-01", "2016-01-01"],
            "test_2": ["a", "b", "c"],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [
                datetime.datetime.now(),
                datetime.datetime.now(),
                datetime.datetime.now(),
            ],
            "test_5": [np.nan, np.nan, np.nan],
            "test_6": ["", "", 5],
        }
        mixed_dataframe = pd.DataFrame(mixed_dataframe_data)

        assert self.utils.get_all_dataframe_categorical(mixed_dataframe) == [
            "test_1",
            "test_2",
            "test_3",
            "test_6",
        ]

    def test_get_dataframe_num_missing(self) -> None:
        """
        Tests for get_dataframe_num_missing.

        :return: None.
        """
        missing_data = {
            "test_1": [np.nan, np.nan, np.nan],
            "test_2": ["a", "b", ""],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [
                datetime.datetime.now(),
                datetime.datetime.now(),
                datetime.datetime.now(),
            ],
        }
        missing_df = pd.DataFrame.from_dict(missing_data, dtype=object)
        expected_percent_missing = {"test_1": 3, "test_2": 1, "test_3": 2, "test_4": 0}
        assert (
            self.utils.get_dataframe_num_missing(missing_df) == expected_percent_missing
        )

    def test_get_dataframe_percent_missing(self) -> None:
        """
        Tests for get_dataframe_percent_missing.

        :return: None.
        """
        missing_data = {
            "test_1": [np.nan, np.nan, np.nan],
            "test_2": ["a", "b", ""],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [
                datetime.datetime.now(),
                datetime.datetime.now(),
                datetime.datetime.now(),
            ],
        }
        missing_df = pd.DataFrame.from_dict(missing_data, dtype=object)
        expected_percent_missing = {
            "test_1": 1.0,
            "test_2": 1 / 3,
            "test_3": 2 / 3,
            "test_4": 0.0,
        }
        assert (
            self.utils.get_dataframe_percent_missing(missing_df)
            == expected_percent_missing
        )

    def test_get_dataframe_high_concentration_variables(self) -> None:
        """
        Tests for get_dataframe_high_concentration_variables.

        :return: None.
        """
        df = pd.DataFrame(
            data={
                "X": [1, np.nan, 1, 1, 1],
                "Y": ["X", "X", "X", "X", "X"],
                "Z1": [1, 1, 1, 1, 1],
                "Z2": [1, 2, 3, 4, 5]
            }
        )
        result_dict = self.utils.get_dataframe_high_concentration_variables(df, 0.6)
        assert result_dict == {'X': 0.4, 'Y': 0.2, 'Z1': 0.2}

    def test_get_dataframe_remove_n_value_variables(self) -> None:
        """
        Tests for get_dataframe_remove_n_value_variables.

        :return: None.
        """
        df = pd.DataFrame(
            data={
                "X": [1, np.nan, 1, 1, 1],
                "Y": ["X", "X", "X", "X", "X"],
                "Z1": [1, 1, 1, 1, 1],
                "Z2": [1, 2, 3, 4, 5]
            }
        )
        one_unique_results = self.utils.get_dataframe_n_value_variables(df, 1)
        assert one_unique_results == ["Y", "Z1"]

        two_unique_results = self.utils.get_dataframe_n_value_variables(df, 2)
        assert two_unique_results == ["X"]

    def test_get_dataframe_outliers(self) -> None:
        """
        Tests for get_dataframe_outliers.

        :return: None.
        """
        df = pd.DataFrame(
            data={
                "X": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 100],
                "Y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                "Z": ["X", "Y", "Y", "Y", "Y", "Y", "Y", "Y", "Y",
                      "Z", "X", "Y", "Y", "X", "X", "X", "Z"],
            }
        )
        assert self.utils.get_dataframe_outliers(df, outlier_threshold=3.29) == ["X"]
        assert self.utils.get_dataframe_outliers(df, outlier_threshold=0.5) == ["X", "Y"]
