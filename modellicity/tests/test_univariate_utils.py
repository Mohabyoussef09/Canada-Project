"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import numpy as np
import numpy.testing as npt
import pandas as pd
import unittest

from pandas.util.testing import assert_frame_equal
from modellicity.models import univariate_utils


class TestUnivariateUtils(unittest.TestCase):
    """Unit tests for UnivariateUtils class."""

    @classmethod
    def setup_class(cls) -> None:
        """:return: None."""
        cls.utils = univariate_utils.UnivariateUtils()

    def test_generate_event_non_event_dataframe(self) -> None:
        """
        Tests for generate_event_non_event_dataframe.

        :return: None.
        """
        event_non_event_dataframe = univariate_utils.UnivariateUtils()

        categorical_variable = ["A",
                                "A",
                                "B",
                                "B",
                                "A",
                                "C",
                                "C"]
        target = [1, 0, 1, 1, 1, 0, 0]

        categorical_variable = pd.Series(categorical_variable)
        target = pd.Series(target)
        categorical_variable = categorical_variable.rename("var_label")
        target = target.rename("target_label")
        event_non_event_dataframe = event_non_event_dataframe.generate_event_non_event_dataframe(
            categorical_variable, target)

        benchmark = pd.DataFrame()
        benchmark["var_label"] = ["A", "B", "C"]
        benchmark["num_event"] = [2, 2, 0]
        benchmark["num_non_event"] = [1, 0, 2]
        try:
            assert_frame_equal(event_non_event_dataframe, benchmark)
            match = True
        except ValueError:
            match = False
        assert match is True

    def test_convert_event_non_event_dataframe_to_woe_dataframe(self) -> None:
        """
        Tests for convert_event_non_event_dataframe_to_woe_dataframe.

        :return: None.
        """
        categorical_list = ["A", "B", "C", "D"]

        # Normal case
        num_event_list1 = [5, 3, 2, 1]

        # Edge case
        num_event_list2 = [0, 0, 0, 0]
        num_non_event_list = [4, 2, 1, 0]

        # Normal case: non-zero events and non-events
        uu1 = univariate_utils.UnivariateUtils()
        sum_event1 = 11
        sum_non_event = 7

        dataframe1 = pd.DataFrame()
        dataframe1["var_list"] = categorical_list
        dataframe1["num_event"] = num_event_list1
        dataframe1["num_non_event"] = num_non_event_list
        dataframe_woe_and_iv1 = uu1.convert_event_non_event_dataframe_to_woe_dataframe(dataframe1)

        woe_a = np.log((5/sum_event1) / (4/sum_non_event))
        woe_b = np.log((3/sum_event1) / (2/sum_non_event))
        woe_c = np.log((2/sum_event1) / (1/sum_non_event))
        woe_d = 0
        woe_list1 = [woe_a, woe_b, woe_c, woe_d]
        iv_list1 = [(5/sum_event1 - 4/sum_non_event) * woe_a,
                    (3/sum_event1 - 2/sum_non_event) * woe_b,
                    (2/sum_event1 - 1/sum_non_event) * woe_c,
                    0]

        benchmark1 = dataframe1.copy()
        benchmark1["woe"] = woe_list1
        benchmark1["iv"] = iv_list1
        try:
            assert_frame_equal(dataframe_woe_and_iv1, benchmark1)
            match1 = True
        except ValueError:
            match1 = False
        assert match1 is True

        # Edge case: num_event is zeroes all across
        uu2 = univariate_utils.UnivariateUtils()
        dataframe2 = pd.DataFrame()
        dataframe2["var_list"] = categorical_list
        dataframe2["num_event"] = num_event_list2
        dataframe2["num_non_event"] = num_non_event_list
        dataframe_woe_and_iv2 = uu2.convert_event_non_event_dataframe_to_woe_dataframe(dataframe2)

        benchmark2 = dataframe2.copy()
        benchmark2["woe"] = [0] * 4
        benchmark2["iv"] = [0] * 4
        try:
            assert_frame_equal(dataframe_woe_and_iv2, benchmark2)
            match2 = True
        except ValueError:
            match2 = False
        assert match2 is True

    def test_convert_categorical_series_to_woe_series(self) -> None:
        """
        Tests for convert_categorical_series_to_woe_series.

        :return:
        """
        uu = univariate_utils.UnivariateUtils()
        dataframe = pd.DataFrame()
        dataframe["target_label"] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        dataframe["X_c1"] = ["A", "A", "A", "B", "C", "C", "C", "A", "B", "C"]
        dataframe["X_c2"] = [np.nan, "A", "A", "B", "C", "C", "C", "A", "B", "C"]

        woe_series1, _, iv1 = uu.convert_categorical_series_to_woe_series(dataframe["X_c1"],
                                                                          dataframe["target_label"])

        benchmark_woe1 = pd.Series([1.098612289, 1.098612289, 1.098612289, 0,
                                    -1.098612289, -1.098612289, -1.098612289,
                                    1.098612289, 0, -1.098612289])
        benchmark_iv1 = 0.878889831

        print(f"woe_series1 = {woe_series1}")
        print(f"benchmark1 = {benchmark_woe1}")

        for i in range(10):
            npt.assert_almost_equal(woe_series1[i], benchmark_woe1[i])
        npt.assert_almost_equal(iv1, benchmark_iv1)

        woe_series2, _, iv2 = uu.convert_categorical_series_to_woe_series(dataframe["X_c2"],
                                                                          dataframe["target_label"])

        benchmark_woe2 = pd.Series([0, 0.693147181, 0.693147181, 0,
                                    -1.098612289, -1.098612289, -1.098612289,
                                    0.693147181, 0, -1.098612289])
        benchmark_iv2 = 0.578074352

        for i in range(10):
            npt.assert_almost_equal(woe_series2[i], benchmark_woe2[i])
        npt.assert_almost_equal(iv2, benchmark_iv2)

    def test_convert_numeric_series_to_woe_series_quantiles_approach(self) -> None:
        """
        Tests for convert_numeric_series_to_woe_series_quantiles_approach.

        :return: None.
        """
        uu = univariate_utils.UnivariateUtils()
        dataframe = pd.DataFrame()
        dataframe["target_label"] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        dataframe["X_n1"] = [3, 1, 5, 6, 7, 1, 2, 1, 3, 1]
        dataframe["X_n2"] = [np.nan, 1, 5, 6, 7, 1, 2, 1, 3, 1]
        quantiles1 = 5
        woe_series1, _, iv1 = uu.convert_numeric_series_to_woe_series_quantiles_approach(
            dataframe["X_n1"], dataframe["target_label"], quantiles1)

        benchmark_woe1 = pd.Series([0.693147181, -1.098612289, 0.693147181,
                                    0, 0, -1.098612289, 0, -1.098612289,
                                    0.693147181, -1.098612289])
        benchmark_iv1 = 0.578074352

        for i in range(10):
            npt.assert_almost_equal(woe_series1[i], benchmark_woe1[i])
        npt.assert_almost_equal(iv1, benchmark_iv1)

        quantiles2 = 4
        woe_series2, _, iv2 = uu.convert_numeric_series_to_woe_series_quantiles_approach(
            dataframe["X_n2"], dataframe["target_label"], quantiles2)

        benchmark_woe2 = pd.Series([0, -1.098612289, 0, 0, 0, -1.098612289, 0,
                                    -1.098612289, 0, -1.098612289])
        benchmark_iv2 = 0.439444915

        print(f"woe_series1 = {woe_series2}")
        print(f"benchmark1 = {benchmark_woe2}")

        for i in range(10):
            npt.assert_almost_equal(woe_series2[i], benchmark_woe2[i])
        npt.assert_almost_equal(iv2, benchmark_iv2)

    def test_convert_original_dataframe_to_woe_dataframe_from_scratch(self) -> None:
        """
        Tests for convert_original_dataframe_to_woe_dataframe_from_scratch.

        :return: None.
        """
        uu = univariate_utils.UnivariateUtils()
        dataframe = pd.DataFrame()
        dataframe["target_label"] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        dataframe["X_c1"] = ["A", "A", "A", "B", "C", "C", "C", "A", "B", "C"]
        dataframe["X_c2"] = [np.nan, "A", "A", "B", "C", "C", "C", "A", "B", "C"]
        dataframe["X_n1"] = [3, 1, 5, 6, 7, 1, 2, 1, 3, 1]
        dataframe["X_n2"] = [np.nan, 1, 5, 6, 7, 1, 2, 1, 3, 1]

        max_bins = 5
        woe_dataframe, variable_to_woe_map, iv_dataframe = \
            uu.convert_original_dataframe_to_woe_dataframe_from_scratch(
                dataframe,
                dataframe["target_label"],
                max_bins)

        benchmark_woe = pd.DataFrame()
        benchmark_woe["target_label"] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        benchmark_woe["X_c1"] = [1.098612289, 1.098612289, 1.098612289, 0,
                                 -1.098612289, -1.098612289, -1.098612289,
                                 1.098612289, 0, -1.098612289]
        benchmark_woe["X_c2"] = [0, 0.693147181, 0.693147181, 0,
                                 -1.098612289, -1.098612289, -1.098612289,
                                 0.693147181, 0, -1.098612289]
        benchmark_woe["X_n1"] = [0.693147181, -1.098612289, 0.693147181,
                                 0, 0, -1.098612289, 0, -1.098612289,
                                 0.693147181, -1.098612289]
        benchmark_woe["X_n2"] = [0, -1.098612289, 0, 0, 0, -1.098612289, 0,
                                 -1.098612289, 0, -1.098612289]

        benchmark_iv = pd.DataFrame()
        benchmark_iv["variable"] = ["X_c1", "X_c2", "X_n1", "X_n2"]
        benchmark_iv["iv"] = [0.878889831, 0.578074352, 0.578074352, 0.439444915]

        for i in range(10):
            npt.assert_almost_equal(woe_dataframe["target_label"].iloc[i],
                                    benchmark_woe["target_label"].iloc[i])
            npt.assert_almost_equal(woe_dataframe["X_c1"].iloc[i],
                                    benchmark_woe["X_c1"].iloc[i])
            npt.assert_almost_equal(woe_dataframe["X_c2"].iloc[i],
                                    benchmark_woe["X_c2"].iloc[i])
            npt.assert_almost_equal(woe_dataframe["X_n1"].iloc[i],
                                    benchmark_woe["X_n1"].iloc[i])
            npt.assert_almost_equal(woe_dataframe["X_n2"].iloc[i],
                                    benchmark_woe["X_n2"].iloc[i])

        benchmark_map_woe = [1.098612289, 0, -1.098612289, 0, 0.693147181, 0,
                             -1.098612289, -1.098612289, 0, 0.693147181, 0, 0,
                             -1.098612289, 0, 0]

        for i in range(15):
            npt.assert_almost_equal(variable_to_woe_map["woe"].iloc[i],
                                    benchmark_map_woe[i])

        for i in range(4):
            assert iv_dataframe["variable"].iloc[i] == benchmark_iv["variable"].iloc[i]
            npt.assert_almost_equal(iv_dataframe["iv"].iloc[i], benchmark_iv["iv"].iloc[i])

    def test_convert_original_dataframe_to_woe_dataframe_use_map(self) -> None:
        """
        Tests for convert_original_dataframe_to_woe_dataframe_use_map.

        :return: None.
        """
        uu = univariate_utils.UnivariateUtils()
        dataframe = pd.DataFrame()
        dataframe["target_label"] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        dataframe["X_c1"] = ["A", "A", "A", "B", "C", "C", "C", "A", "B", "C"]
        dataframe["X_c2"] = [np.nan, "A", "A", "B", "C", "C", "C", "A", "B", "C"]
        dataframe["X_n1"] = [3, 1, 5, 6, 7, 1, 2, 1, 3, 1]
        dataframe["X_n2"] = [np.nan, 1, 5, 6, 7, 1, 2, 1, 3, 1]

        max_bins = 5
        benchmark_woe, variable_to_woe_map, iv_dataframe = \
            uu.convert_original_dataframe_to_woe_dataframe_from_scratch(
                dataframe, dataframe["target_label"], max_bins)
        woe_dataframe = uu.convert_original_dataframe_to_woe_dataframe_use_map(
            dataframe, variable_to_woe_map)

        for i in range(10):
            npt.assert_almost_equal(woe_dataframe["target_label"].iloc[i],
                                    benchmark_woe["target_label"].iloc[i])
            npt.assert_almost_equal(woe_dataframe["X_c1"].iloc[i],
                                    benchmark_woe["X_c1"].iloc[i])
            npt.assert_almost_equal(woe_dataframe["X_c2"].iloc[i],
                                    benchmark_woe["X_c2"].iloc[i])
            npt.assert_almost_equal(woe_dataframe["X_n1"].iloc[i],
                                    benchmark_woe["X_n1"].iloc[i])
            npt.assert_almost_equal(woe_dataframe["X_n2"].iloc[i],
                                    benchmark_woe["X_n2"].iloc[i])
