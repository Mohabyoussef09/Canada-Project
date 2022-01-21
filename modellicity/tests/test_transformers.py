"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import numpy as np
import unittest

from modellicity.data_source import DataSource
from modellicity.extended_pandas import ExtendedDataFrame
from modellicity.transformers import (
    ConvertDatetime,
    ConvertNumeric,
    RemoveColumns,
    RemoveHighMissingness,
    RemoveOneValueVariables,
    RemoveOutliers,
    convert_datetime,
    convert_numeric,
    remove_columns,
    remove_high_missingness,
    remove_one_value_variables,
    remove_outliers,
)
from modellicity.settings import settings


class TestPipeline(unittest.TestCase):
    """Unittests for transformer functions."""

    def test_column_removal(self) -> None:
        """Tests for column_removal."""
        ds = DataSource(
            ExtendedDataFrame(data={"A": [1], "B": [2], "C": [3], "D": [4], "E": [5]})
        )

        self.assertEqual(ds.data.shape, (1, 5))

        datasources = [
            RemoveColumns(["C", "E"]).run(ds),
            remove_columns(ds, ["C", "E"]),
        ]

        for ds in datasources:
            self.assertEqual(ds.data.shape, (1, 3))

            for col in ["A", "B", "D"]:
                self.assertTrue(col in ds.data.columns)

        datasources = [
            RemoveColumns(["F", "G"]).run(ds),
            remove_columns(ds, ["F", "G"]),
        ]

        for ds in datasources:

            for col in ["F", "G"]:
                self.assertTrue(col not in ds.data.columns)

        empty_ds = DataSource(ExtendedDataFrame(data={}))
        datasources = [RemoveColumns([]).run(empty_ds),
                       remove_columns(empty_ds, [])]
        for ds in datasources:
            assert len(ds.data.columns) == 0
            assert all(ds.data.columns == [])

    def test_convert_numeric(self) -> None:
        """Tests for numeric_treatment."""
        ds = DataSource(ExtendedDataFrame(
            data={
                "X": ["100", "20"],
                "Y": ["A", "B"],
                "Z1": [20.56, 35],
                "Z2": ["20.56", "35"]
            }
        ))

        # Check outputs of types prior to conversion.
        for col in ds.data.columns:
            if col == "Z1":
                self.assertEqual(ds.data[col].dtype in settings.OPTIONS["numeric_types"], True)
            else:
                self.assertEqual(ds.data[col].dtype in settings.OPTIONS["numeric_types"], False)

        datasources = [
            ConvertNumeric().run(ds),
            convert_numeric(ds),
        ]

        # Check to ensure types are properly converted post ConvertNumeric.
        for ds in datasources:
            for col in ds.data.columns:
                if col in ["X", "Z1", "Z2"]:
                    self.assertEqual(ds.data[col].dtype in settings.OPTIONS["numeric_types"], True)
                else:
                    self.assertEqual(ds.data[col].dtype in settings.OPTIONS["numeric_types"], False)

        # Ensure graceful return on empty dataframe.
        empty_ds = DataSource(ExtendedDataFrame(data={}))
        datasources = [ConvertNumeric().run(empty_ds),
                       convert_numeric(empty_ds)]
        for ds in datasources:
            assert len(ds.data.columns) == 0
            assert all(ds.data.columns == [])

        # Extra numeric types passed in.
        ds = DataSource(ExtendedDataFrame(
            data={
                "X": ["100", "20"],
                "Y": ["A", "B"],
                "Z1": [20.56, 35],
                "Z2": ["20.56", "35"],
                "Z3": [np.double(5), np.double(53)]
            }
        ))
        datasources = [
            ConvertNumeric(extra_formats=[np.double]).run(ds),
            convert_numeric(ds, extra_formats=[np.double]),
        ]

        for ds in datasources:
            for col in ds.data.columns:
                if col in ["X", "Z1", "Z2"]:
                    self.assertEqual(ds.data[col].dtype in settings.OPTIONS["numeric_types"], True)

    def test_convert_datetime(self) -> None:
        """Tests for datetime_treatment."""
        ds = DataSource(
            ExtendedDataFrame(
                data={
                    "test_1": ["2018-01-01", "2017-01-01", "2016-01-01"],
                    "test_2": ["a", "b", "c"],
                    "test_3": ["2018-01-01", "", np.nan],
                    "test_4": [np.nan, np.nan, np.nan],
                    "test_5": ["", "", 5],
                }
            )
        )

        # Check the types of variables before performing ConvertDatetime.
        for col in ds.data.columns:
            if col == "test_4":
                self.assertEqual(ds.data[col].dtype, "float64")
            else:
                self.assertEqual(ds.data[col].dtype, "O")

        datasources = [
            ConvertDatetime().run(ds),
            convert_datetime(ds),
        ]

        # Convert string formatted dates to datetime objects.
        for ds in datasources:
            for col in ds.data.columns:
                if col in ["test_1", "test_3"]:
                    self.assertEqual(ds.data[col].dtype, "datetime64[ns]")
                else:
                    self.assertNotEqual(ds.data[col].dtype, "datetime64[ns]")

        # Ensure graceful return on empty dataframe.
        empty_ds = DataSource(ExtendedDataFrame(data={}))
        datasources = [ConvertDatetime().run(empty_ds),
                       convert_datetime(empty_ds)]
        for ds in datasources:
            assert len(ds.data.columns) == 0
            assert all(ds.data.columns == [])

        # Check that extra date formats are supported.
        extra_format_ds = DataSource(
            ExtendedDataFrame(
                data={
                    "test_1": ["2018-01-01", "2017-01-01", "2016-01-01"],
                    "test_2": ["a", "b", "c"],
                    "test_3": ["2018-01-01", "", np.nan],
                    "test_4": [np.nan, np.nan, np.nan],
                    "test_5": ["", "", 5],
                    "test_6": ["Dec-2012", "Jan-2013", "Feb-2013"]
                }
            )
        )
        datasources = [ConvertDatetime(extra_formats=["%b-%Y"]).run(extra_format_ds),
                       convert_datetime(extra_format_ds, extra_formats=["%b-%Y"])]
        for ds in datasources:
            assert isinstance(ds.data["test_6"][0], tuple(settings.OPTIONS["date_types"]))

    def test_high_missingness(self) -> None:
        """Tests for high_missingness."""
        ds = DataSource(
            ExtendedDataFrame(
                data={
                    "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "B": [1, 2, 3, 4, 5, 6, 7, 8, None, None],
                    "C": [1, 2, 3, 4, 5, None, None, None, None, None],
                    "D": [None] * 10,
                    "E": [None] * 10,
                }
            )
        )

        # Remove column "D" with invalid threshold:
        self.assertEqual(ds.data.shape, (10, 5))
        invalid_threshold_datasources = [
            RemoveHighMissingness(5.8, ["D"]).run(ds),
            remove_high_missingness(ds, 5.8, ["D"]),
        ]
        for ds in invalid_threshold_datasources:
            self.assertEqual(ds.data.shape, (10, 5))

        # Remove column "D" with valid threshold:
        datasources = [
            RemoveHighMissingness(0.8, ["D"]).run(ds),
            remove_high_missingness(ds, 0.8, ["D"]),
        ]
        for ds in datasources:
            self.assertEqual(ds.data.shape, (10, 4))

        # Remove with no exception list:
        no_exception_datasources = [
            RemoveHighMissingness(0.8).run(ds),
            remove_high_missingness(ds, 0.8),
        ]
        for ds in no_exception_datasources:
            self.assertEqual(ds.data.shape, (10, 3))

        # Remove no columns with empty dataframe:
        empty_ds = DataSource(ExtendedDataFrame(data={}))
        datasources = [RemoveHighMissingness(0.8, []).run(empty_ds),
                       remove_high_missingness(empty_ds, 0.8, [])]
        for ds in datasources:
            assert len(ds.data.columns) == 0
            assert all(ds.data.columns == [])

    def test_remove_one_value_variables(self) -> None:
        """Tests for remove one-value variables."""
        ds = DataSource(ExtendedDataFrame(
                data={
                    "X": [1, np.nan, 1, 1, 1],
                    "Y": ["X", "X", "X", "X", "X"],
                    "Z1": [1, 1, 1, 1, 1],
                    "Z2": [1, 2, 3, 4, 5]
                }))

        datasources = [RemoveOneValueVariables().run(ds), remove_one_value_variables(ds)]

        for ds in datasources:
            assert len(ds.data.columns) == 2
            assert all(ds.data.columns == ["X", "Z2"])

        # Test with empty dataframe.
        empty_ds = DataSource(ExtendedDataFrame(data={}))
        datasources = [RemoveOneValueVariables().run(empty_ds),
                       remove_one_value_variables(empty_ds)]
        for ds in datasources:
            assert len(ds.data.columns) == 0
            assert all(ds.data.columns == [])

    def test_remove_outliers(self) -> None:
        """Tests for remove outliers."""
        ds = DataSource(
            ExtendedDataFrame(
                data={
                    "X": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 100],
                    "Y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                    "Z": ["X", "Y", "Y", "Y", "Y", "Y", "Y", "Y", "Y", "Z", "X", "Y",
                          "Y", "X", "X", "X", "Z"],
                }
            )
        )

        datasources = [RemoveOutliers(exceptions=["does_not_exist"]).run(ds),
                       remove_outliers(ds, exceptions=["does_not_exist"])]
        for ds in datasources:
            assert len(ds.data.columns) == 3
            assert all(ds.data.columns == ["X", "Y", "Z"])

        datasources = [RemoveOutliers(exceptions=["X"]).run(ds),
                       remove_outliers(ds, exceptions=["X"])]
        for ds in datasources:
            assert len(ds.data.columns) == 3
            assert all(ds.data.columns == ["X", "Y", "Z"])

        datasources = [RemoveOutliers().run(ds), remove_outliers(ds)]
        for ds in datasources:
            assert len(ds.data.columns) == 3
            assert all(ds.data.columns == ["X", "Y", "Z"])

        empty_ds = DataSource(
            ExtendedDataFrame(
                data={
                }
            )
        )
        datasources = [RemoveOutliers().run(empty_ds), remove_outliers(empty_ds)]
        for ds in datasources:
            assert len(ds.data.columns) == 0
            assert all(ds.data.columns == [])
