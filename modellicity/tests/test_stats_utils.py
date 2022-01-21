"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import numpy as np
import numpy.testing as npt
import pandas as pd
import unittest

from nose.tools import assert_raises
from modellicity.stats import stats_utils


class TestStatsUtils(unittest.TestCase):
    """Unit tests for StatsUtils class."""

    @classmethod
    def setup_class(cls) -> None:
        """:return: None."""
        cls.utils = stats_utils.StatsUtils()

    def test_t_test(self) -> None:
        """
        Tests for t_test.

        :return: None.
        """
        assert self.utils.t_test([1.0, 2.0], [3.0, 4.0]) == 0.10557280900008414

    def test_kolomogorov_smirnov_test(self) -> None:
        """
        Tests for kolomogorov_smirnov_test.

        :return: None.
        """
        assert self.utils.kolmogorov_smirnov_test([1.0, 2.0], [3.0, 4.0]) == 1.0

    def test_normalize_dataframe(self) -> None:
        """
        Tests for normalize_dataframe.

        :return: None.
        """
        numeric_data = {
            "test_1": [1, 2, 3, 4],
            "test_2": [5, 6, 7, 8],
            "test_3": [9, 10, 11, 12]
        }
        numeric_dataframe = pd.DataFrame(numeric_data)
        normalized_numeric_dataframe = self.utils.normalize_dataframe(numeric_dataframe)
        assert normalized_numeric_dataframe["test_1"][0] == -1.161895003862225

        non_numeric_data = {
            "test_1": [1, 2, 3, "4"],
            "test_2": [5, 6, 7, 8],
            "test_3": [9, 10, 11, 12]
        }
        non_numeric_dataframe = pd.DataFrame(non_numeric_data)

        assert_raises(ValueError, self.utils.normalize_dataframe, non_numeric_dataframe)

    def test_get_binned_quantiles(self) -> None:
        """
        Tests for get_binned_quantiles.

        :return: None.
        """
        # Normal case
        numeric_series_1 = pd.Series([1, 2, 3, 4, 5])
        quantiles = 5
        dataframe_1 = self.utils.get_binned_quantiles(numeric_series_1,
                                                      quantiles)

        benchmark_1 = pd.DataFrame()
        benchmark_1["bin"] = [1, 2, 3, 4, 5]
        benchmark_1["lower_limit"] = [1.0, 1.8, 2.6, 3.4, 4.2]
        benchmark_1["upper_limit"] = [1.8, 2.6, 3.4, 4.2, 5.0]

        for i in range(5):
            npt.assert_almost_equal(dataframe_1["bin"].iloc[i],
                                    benchmark_1["bin"].iloc[i])
            npt.assert_almost_equal(dataframe_1["lower_limit"].iloc[i],
                                    benchmark_1["lower_limit"].iloc[i])
            npt.assert_almost_equal(dataframe_1["upper_limit"].iloc[i],
                                    benchmark_1["upper_limit"].iloc[i])

        # Edge case
        numeric_series_2 = pd.Series([np.nan, 1, 1, 1, 1, 1, 1, 1, 2, 2])
        dataframe_2 = self.utils.get_binned_quantiles(numeric_series_2,
                                                      quantiles)
        benchmark_2 = pd.DataFrame()
        benchmark_2["bin"] = [1, 2, 3]
        benchmark_2["lower_limit"] = [np.nan, 1.0, 1.4]
        benchmark_2["upper_limit"] = [np.nan, 1.4, 2.0]

        for i in range(3):
            npt.assert_almost_equal(dataframe_2["bin"].iloc[i],
                                    benchmark_2["bin"].iloc[i])
            npt.assert_almost_equal(dataframe_2["lower_limit"].iloc[i],
                                    benchmark_2["lower_limit"].iloc[i])
            npt.assert_almost_equal(dataframe_2["upper_limit"].iloc[i],
                                    benchmark_2["upper_limit"].iloc[i])

    def test_convert_numeric_series_to_binned_series(self) -> None:
        """
        Tests for get_quantiles_numeric_variables.

        :return: None.
        """
        su = stats_utils.StatsUtils()
        extra_margin_upper_limit = 1e-10
        # Normal case
        numeric_series1 = pd.Series([1, 4, 5, 2, 3])
        numeric_series1.name = "var_label"
        bin_ranges1 = pd.DataFrame()
        bin_ranges1["bin"] = [1, 2, 3, 4, 5]
        bin_ranges1["lower_limit"] = [1.0, 1.8, 2.6, 3.4, 4.2]
        bin_ranges1["upper_limit"] = [1.8, 2.6, 3.4, 4.2, 5.0]
        binned_series1, _ = su.convert_numeric_series_to_binned_series(numeric_series1,
                                                                       bin_ranges1,
                                                                       extra_margin_upper_limit)
        benchmark1 = pd.Series(["[1.0, 1.8)",
                                "[3.4, 4.2)",
                                "[4.2, " + str(5.0 + 1e-10) + ")",
                                "[1.8, 2.6)",
                                "[2.6, 3.4)"])
        for i in range(5):
            assert binned_series1[i] == benchmark1[i]
        # Edge case 1
        numeric_series2 = pd.Series([np.nan, 1, 1, 1, 1, 1, 1, 1, 2, 2])
        numeric_series2.name = "var_label"
        bin_ranges2 = pd.DataFrame()
        bin_ranges2["bin"] = [1, 2, 3]
        bin_ranges2["lower_limit"] = [np.nan, 1.0, 1.4]
        bin_ranges2["upper_limit"] = [np.nan, 1.4, 2.0]
        binned_series2, _ = su.convert_numeric_series_to_binned_series(numeric_series2,
                                                                       bin_ranges2,
                                                                       extra_margin_upper_limit)
        benchmark2 = pd.Series(["[Missing]",
                                "[1.0, 1.4)",
                                "[1.0, 1.4)",
                                "[1.0, 1.4)",
                                "[1.0, 1.4)",
                                "[1.0, 1.4)",
                                "[1.0, 1.4)",
                                "[1.0, 1.4)",
                                "[1.4, " + str(2.0 + 1e-10) + ")",
                                "[1.4, " + str(2.0 + 1e-10) + ")"])
        for i in range(10):
            assert binned_series2[i] == benchmark2[i]

        # Edge case 2
        numeric_series3 = pd.Series([np.nan, 1, 1, 1, 1, 1, 1, 1, 2, 2])
        numeric_series3.name = "var_label"
        bin_ranges3 = pd.DataFrame()
        bin_ranges3["bin"] = [1, 2, 3]
        bin_ranges3["lower_limit"] = [1.0, 1.5, 1.8]
        bin_ranges3["upper_limit"] = [1.5, 1.8, 1.9]
        binned_series3, _ = su.convert_numeric_series_to_binned_series(numeric_series3,
                                                                       bin_ranges3,
                                                                       extra_margin_upper_limit)
        benchmark3 = pd.Series([str(np.nan),
                                "[1.0, 1.5)",
                                "[1.0, 1.5)",
                                "[1.0, 1.5)",
                                "[1.0, 1.5)",
                                "[1.0, 1.5)",
                                "[1.0, 1.5)",
                                "[1.0, 1.5)",
                                str(np.nan),
                                str(np.nan)])
        for i in range(10):
            if pd.isnull(binned_series3[i]):
                npt.assert_almost_equal(binned_series3[i], benchmark3[i])
            else:
                self.assertEqual(binned_series3[i], benchmark3[i])
