"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import datetime
import unittest

import numpy as np
import pandas as pd

from modellicity.extended_pandas import ExtendedDataFrame
from modellicity.settings import settings


class TestExtendedDataFrame(unittest.TestCase):
    """Unit tests for ExtendedDataFrame class."""

    @classmethod
    def setup_class(cls) -> None:
        """:return: None."""
        cls.utils = ExtendedDataFrame()

    def test_get_all_categorical(self) -> None:
        """
        Tests for get_all_categorical.

        :return: None.
        """
        numeric_data = {
            "test_1": [1, 2, 3, 4],
            "test_2": [5, 6, 7, 8],
            "test_3": [9, 10, 11, 12],
        }
        df_numeric = ExtendedDataFrame(numeric_data)

        assert df_numeric.get_all_categorical() == []

        mixed_data = {
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
        df_mixed = ExtendedDataFrame(mixed_data)

        assert df_mixed.get_all_categorical() == [
            "test_1",
            "test_2",
            "test_3",
            "test_6",
        ]

    def test_get_all_numeric_object(self) -> None:
        """
        Tests for get_all_numeric_object.

        :return: None.
        """
        raw_data = {
            "test_1": [1, 2, 3],
            "test_2": ["a", "b", "c"],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [datetime.datetime.now(), datetime.datetime.now(), datetime.datetime.now()]
        }
        df = pd.DataFrame(raw_data)
        extended_df = ExtendedDataFrame(df)

        assert extended_df.get_all_numeric_object() == ["test_1"]

    def test_is_any_datetime_object(self) -> None:
        """
        Tests for is_any_datetime_object.

        :return: None.
        """
        raw_data = {
            "test_1": ["2018-01-01", "2017-01-01", "2016-01-01"],
            "test_2": ["a", "b", "c"],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [datetime.datetime.now(), datetime.datetime.now(), datetime.datetime.now()]
        }
        df = pd.DataFrame(raw_data)
        extended_df = ExtendedDataFrame(df)

        assert extended_df.is_any_datetime_object() is True

    def test_get_all_numeric_format(self) -> None:
        """
        Tests for get_all_numeric_format.

        :return: None.
        """
        df = ExtendedDataFrame(
            data={
                "X": ["100", "20"],
                "Y": ["A", "B"],
                "Z1": [20.56, 35],
                "Z2": ["20.56", "35"]
            }
        )
        res = df.get_all_numeric_format()
        self.assertEqual(res, ["X", "Z1", "Z2"])

    def test_get_all_datetime_format(self) -> None:
        """
        Tests for get_all_datetime_format.

        :return: None.
        """
        raw_data = {
            "test_1": ["2018-01-01", "2017-01-01", "2016-01-01"],
            "test_2": ["a", "b", "c"],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [datetime.datetime.now(), datetime.datetime.now(), datetime.datetime.now()]
        }
        df = pd.DataFrame(raw_data)
        extended_df = ExtendedDataFrame(df)

        assert extended_df.get_all_datetime_format() == ["test_1", "test_3"]

    def test_get_all_datetime_object(self) -> None:
        """
        Tests for get_all_datetime_object.

        :return: None.
        """
        raw_data = {
            "test_1": ["2018-01-01", "2017-01-01", "2016-01-01"],
            "test_2": ["a", "b", "c"],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [datetime.datetime.now(), datetime.datetime.now(), datetime.datetime.now()]
        }
        df = pd.DataFrame(raw_data)
        extended_df = ExtendedDataFrame(df)

        assert extended_df.get_all_datetime_object() == ["test_4"]

    def test_get_all_missing(self) -> None:
        """
        Tests for get_all_missing.

        :return: None.
        """
        raw_data = {
            "test_1": ["2018-01-01", "2017-01-01", "2016-01-01"],
            "test_2": ["a", "b", "c"],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [datetime.datetime.now(), datetime.datetime.now(), datetime.datetime.now()]
        }
        df = pd.DataFrame(raw_data)
        extended_df = ExtendedDataFrame(df)

        assert extended_df.get_all_missing() == ["test_3"]

    def test_get_all_no_missing(self) -> None:
        """
        Tests for get_all_no_missing.

        :return: None.
        """
        raw_data = {
            "test_1": ["2018-01-01", "2017-01-01", "2016-01-01"],
            "test_2": ["a", "b", "c"],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [datetime.datetime.now(), datetime.datetime.now(), datetime.datetime.now()]
        }
        df = pd.DataFrame(raw_data)
        extended_df = ExtendedDataFrame(df)

        assert extended_df.get_all_no_missing() == ["test_1", "test_2", "test_4"]

    def test_get_num_missing(self) -> None:
        """
        Tests for get_num_missing.

        :return: None.
        """
        raw_data = {
            "test_1": ["2018-01-01", "2017-01-01", "2016-01-01"],
            "test_2": ["a", "b", "c"],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [datetime.datetime.now(), datetime.datetime.now(), datetime.datetime.now()]
        }
        df = pd.DataFrame(raw_data)
        extended_df = ExtendedDataFrame(df)

        assert extended_df.get_num_missing() == {'test_1': 0, 'test_2': 0, 'test_3': 2, 'test_4': 0}

    def test_get_percent_missing(self) -> None:
        """
        Tests for get_percent_missing.

        :return: None.
        """
        raw_data = {
            "test_1": ["2018-01-01", "2017-01-01", "2016-01-01"],
            "test_2": ["a", "b", "c"],
            "test_3": ["2018-01-01", "", np.nan],
            "test_4": [datetime.datetime.now(), datetime.datetime.now(), datetime.datetime.now()]
        }
        df = pd.DataFrame(raw_data)
        extended_df = ExtendedDataFrame(df)

        assert extended_df.get_percent_missing() == {"test_1": 0.0,
                                                     "test_2": 0.0,
                                                     "test_3": 2/3,
                                                     "test_4": 0.0}

    def test_get_n_value_variables(self) -> None:
        """
        Tests for get_n_value_variables.

        :return: None
        """
        df = ExtendedDataFrame(
            data={
                "X": [1, np.nan, 1, 1, 1],
                "Y": ["X", "X", "X", "X", "X"],
                "Z1": [1, 1, 1, 1, 1],
                "Z2": [1, 2, 3, 4, 5]
            }
        )
        one_unique_results = df.get_n_value_variables(num_unique=1)
        assert one_unique_results == ["Y", "Z1"]

        two_unique_results = df.get_n_value_variables(num_unique=2)
        assert two_unique_results == ["X"]

    def test_get_high_concentration_variables(self) -> None:
        """
        Tests for get_high_concentration_variables.

        :return: None.
        """
        df = ExtendedDataFrame(
            data={
                "X": [1, np.nan, 1, 1, 1],
                "Y": ["X", "X", "X", "X", "X"],
                "Z1": [1, 1, 1, 1, 1],
                "Z2": [1, 2, 3, 4, 5]
            }
        )
        result_dict = df.get_high_concentration_variables(concentration_threshold=0.6)
        assert result_dict == {'X': 0.4, 'Y': 0.2, 'Z1': 0.2}

    def test_floor_and_cap(self) -> None:
        """
        Test for floor_and_cap.

        :return: None.
        """
        df = pd.DataFrame(
            data={
                "X": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 100],
                "Y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                "Z1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 100],
                "Z2": ["X", "Y", "Y", "Y", "Y", "Y", "Y", "Y",
                       "Y", "Z", "X", "Y", "Y", "X", "X", "X", "Z"]
            }
        )
        expected_df = pd.DataFrame(
            data={
                "X": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                      9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 87.0],
                "Y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                "Z1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                       9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 87.0]
            }
        )

        extended_df = ExtendedDataFrame(df)
        expected_extended_df = ExtendedDataFrame(expected_df)
        extended_df = extended_df.floor_and_cap()

        print(extended_df)
        print(expected_extended_df)

        assert extended_df.equals(expected_extended_df)

    def test_get_outliers(self) -> None:
        """
        Tests for get_outliers.

        :return: None.
        """
        df = ExtendedDataFrame(
            data={
                "X": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 100],
                "Y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                "Z": ["X", "Y", "Y", "Y", "Y", "Y", "Y", "Y", "Y",
                      "Z", "X", "Y", "Y", "X", "X", "X", "Z"],
            }
        )
        assert df.get_outliers(outlier_threshold=3.29) == ["X"]
        assert df.get_outliers(outlier_threshold=0.5) == ["X", "Y"]

    def test_convert_to_numeric_object(self) -> None:
        """
        Tests for convert_to_numeric_object.

        :return: None.
        """
        df = ExtendedDataFrame(
            data={
                "X": ["100", "20"],
                "Y": ["A", "B"],
                "Z1": [20.56, 35],
                "Z2": ["20.56", "35"]
            }
        )
        converted_df = df.convert_to_numeric_object()

        self.assertEqual(type(converted_df["X"][0]) in settings.OPTIONS["numeric_types"], True)
        self.assertEqual(type(converted_df["X"][1]) in settings.OPTIONS["numeric_types"], True)

        self.assertEqual(type(converted_df["Y"][0]) in settings.OPTIONS["numeric_types"], False)
        self.assertEqual(type(converted_df["Y"][1]) in settings.OPTIONS["numeric_types"], False)

        self.assertEqual(type(converted_df["Z1"][0]) in settings.OPTIONS["numeric_types"], True)
        self.assertEqual(type(converted_df["Z1"][1]) in settings.OPTIONS["numeric_types"], True)

        self.assertEqual(type(converted_df["Z2"][0]) in settings.OPTIONS["numeric_types"], True)
        self.assertEqual(type(converted_df["Z2"][1]) in settings.OPTIONS["numeric_types"], True)

    def test_convert_to_datetime_object(self) -> None:
        """
        Tests for convert_to_datetime_object.

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
        df = ExtendedDataFrame(dataframe_data)

        converted_df = df.convert_to_datetime_object()

        for col in converted_df.columns:
            if col in ["test_1", "test_3", "test_4", "test_5"]:
                self.assertEqual(converted_df[col].dtype, "datetime64[ns]")
            else:
                self.assertNotEqual(converted_df[col].dtype, "datetime64[ns]")

    def test_remove_n_value_variables(self) -> None:
        """
        Tests for remove_n_value_variables.

        :return: None
        """
        df = pd.DataFrame(
            data={
                "X": [1, np.nan, 1, 1, 1],
                "Y": ["X", "X", "X", "X", "X"],
                "Z1": [1, 1, 1, 1, 1],
                "Z2": [1, 2, 3, 4, 5]
            }
        )
        extended_df = ExtendedDataFrame(df)

        one_unique_results = extended_df.remove_n_unique_value_variables(1)
        assert all(one_unique_results.columns == ["X", "Z2"])

        two_unique_results = extended_df.remove_n_unique_value_variables(2)
        assert all(two_unique_results.columns == ["Y", "Z1", "Z2"])

    def test_remove_high_concentration_variables(self) -> None:
        """
        Tests for remove_high_concentration_variables.

        :return: None.
        """
        df = pd.DataFrame(
            data={
                "test_1": [1, 1, 1, 1, 1],
                "test_2": [2, 3, 4, 5, 6]
            }
        )
        extended_df = ExtendedDataFrame(df)

        assert len(extended_df.columns) == 2

        extended_df_all = extended_df.remove_high_concentration_variables(1.0)
        assert len(extended_df_all.columns) == 0

        extended_df_half = extended_df.remove_high_concentration_variables(0.5)
        assert len(extended_df_half.columns) == 1
