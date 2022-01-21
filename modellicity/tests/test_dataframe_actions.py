"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import datetime
import unittest

import numpy as np
import pandas as pd

from modellicity.extended_pandas.extended_actions import dataframe_actions
from modellicity.settings import settings


class TestDataFrameActions(unittest.TestCase):
    """Unit tests for DataFrameActions class."""

    @classmethod
    def setup_class(cls) -> None:
        """:return: None."""
        cls.utils = dataframe_actions.DataFrameActions()

    def test_convert_dataframe_to_numeric_object(self) -> None:
        """
        Tests for convert_dataframe_to_numeric_object.

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
        converted_df = self.utils.convert_dataframe_to_numeric_object(df)

        self.assertEqual(type(converted_df["X"][0]) in settings.OPTIONS["numeric_types"], True)
        self.assertEqual(type(converted_df["X"][1]) in settings.OPTIONS["numeric_types"], True)

        self.assertEqual(type(converted_df["Y"][0]) in settings.OPTIONS["numeric_types"], False)
        self.assertEqual(type(converted_df["Y"][1]) in settings.OPTIONS["numeric_types"], False)

        self.assertEqual(type(converted_df["Z1"][0]) in settings.OPTIONS["numeric_types"], True)
        self.assertEqual(type(converted_df["Z1"][1]) in settings.OPTIONS["numeric_types"], True)

        self.assertEqual(type(converted_df["Z2"][0]) in settings.OPTIONS["numeric_types"], True)
        self.assertEqual(type(converted_df["Z2"][1]) in settings.OPTIONS["numeric_types"], True)

    def test_convert_dataframe_to_datetime_object(self) -> None:
        """
        Tests for convert_dataframe_to_datetime_object.

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

        converted_df = self.utils.convert_dataframe_to_datetime_object(
            dataframe, dataframe.columns
        )

        assert isinstance(converted_df["test_1"][0], pd.Timestamp)
        assert converted_df["test_2"][0] == "a"
        assert isinstance(converted_df["test_4"][0], pd.Timestamp)

        converted_no_cols_df = self.utils.convert_dataframe_to_datetime_object(
            dataframe, None
        )

        assert isinstance(converted_no_cols_df["test_1"][0], pd.Timestamp)
        assert converted_no_cols_df["test_2"][0] == "a"
        assert isinstance(converted_no_cols_df["test_4"][0], pd.Timestamp)

    def test_remove_n_unique_value_variables(self) -> None:
        """
        Tests for remove_n_unique_value_variables.

        :return:
        """
        df = pd.DataFrame(
            data={
                "X": [1, np.nan, 1, 1, 1],
                "Y": ["X", "X", "X", "X", "X"],
                "Z1": [1, 1, 1, 1, 1],
                "Z2": [1, 2, 3, 4, 5]
            }
        )
        one_unique_results = self.utils.remove_n_unique_value_variables(df, 1)
        assert all(one_unique_results.columns == ["X", "Z2"])

        two_unique_results = self.utils.remove_n_unique_value_variables(df, 2)
        assert all(two_unique_results.columns == ["Y", "Z1", "Z2"])

    def test_remove_high_concentration_variables(self) -> None:
        """
        Tests for remove_high_concentration_variables.

        :return: None.
        """
        numeric_data = {"test_1": [1, 1, 1, 1, 1], "test_2": [2, 3, 4, 5, 6]}
        numeric_dataframe = pd.DataFrame(numeric_data)
        assert len(numeric_dataframe.columns) == 2

        new_dataframe = self.utils.remove_high_concentration_variables(
            df=numeric_dataframe, concentration_threshold=1.0
        )
        assert len(new_dataframe.columns) == 0

        new_dataframe = self.utils.remove_high_concentration_variables(
            df=numeric_dataframe, concentration_threshold=0.5
        )
        assert len(new_dataframe.columns) == 1

    def test_floor_and_cap_dataframe(self) -> None:
        """
        Tests for floor_and_cap_dataframe.

        :return:
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
        df_floor_and_cap = self.utils.floor_and_cap_dataframe(df)

        expected_df = pd.DataFrame(
            data={
                "X": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                      8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 87.0],
                "Y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                "Z1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                       8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 87.0]
            }
        )

        assert df_floor_and_cap.equals(expected_df)
