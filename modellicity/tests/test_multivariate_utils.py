"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import numpy as np
import pandas as pd
import unittest

from pandas.util.testing import assert_frame_equal
from modellicity.models import multivariate_utils


class TestMultivariateUtils(unittest.TestCase):
    """Unit tests for MultivariateUtils class."""

    @classmethod
    def setup_class(cls) -> None:
        """:return: None."""
        cls.utils = multivariate_utils.MultivariateUtils()

    def test_decorrelate(self) -> None:
        """
        Tests for decorrelate.

        :return:
        """
        mu = multivariate_utils.MultivariateUtils()
        dataframe = pd.DataFrame()
        dataframe["X_c1"] = ["A", "A", "A", "B", "C", "C", "C", "A", "B", "C"]
        dataframe["X_n1"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        dataframe["X_n2"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        dataframe["X_n3"] = [np.nan, 2, 20, -5, 7, 8, -5, 4, -10, 6]
        dataframe["X_n4"] = [np.nan, -2, -20, 5, -7, -8, 5, -4, 10, -6]

        var_list = ["X_n1", "X_n2", "X_n3", "X_n4"]
        kpi_map = pd.DataFrame()
        kpi_map["variable"] = ["X_n1", "X_n2", "X_n3", "X_n4"]
        kpi_map["kpi"] = [0.8, 0.7, 0.6, 0.5]
        high_correlation_threshold = 0.7

        dataframe_decorr = mu.decorrelate(dataframe, kpi_map, var_list, high_correlation_threshold)
        benchmark = dataframe.copy()
        benchmark.drop(columns=["X_n2", "X_n4"], inplace=True)
        assert_frame_equal(dataframe_decorr, benchmark)
