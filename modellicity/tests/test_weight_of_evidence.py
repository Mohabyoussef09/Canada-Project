"""@copyright Copyright © 2020, Modellicity Inc., All Rights Reserved."""
import numpy as np
import numpy.testing as npt
import pandas as pd
import unittest
from modellicity.stats.weight_of_evidence import WeightOfEvidence


class TestWeightOfEvidence(unittest.TestCase):
    """Unit tests for WeightOfEvidence class."""

    @classmethod
    def setup_class(cls) -> None:
        """Set class up."""
        cls.utils = WeightOfEvidence

    def test_bin_woe_numeric_monotonic(self):
        """Test bin_woe_numeric_monotonic."""
        woe = WeightOfEvidence()

        # Case 1: One-valued variable and one-valued target
        input_df = pd.DataFrame({"x": [1, 1], "y": [0, 0]})
        df1 = woe.bin_woe_numeric_monotonic(variable=input_df["x"],
                                            target=input_df["y"],
                                            n_max=20,
                                            round_limit=2,
                                            lower_limit_bounded=True,
                                            upper_limit_bounded=True)
        df2 = pd.DataFrame({"variable": ["x"],
                            "bin_number": [1],
                            "lower_limit": [1],
                            "upper_limit": [1.01],
                            "bin": ["[1.0, 1.01)"],
                            "num_obs": [2],
                            "num_events": [0],
                            "num_non_events": [2],
                            "trend": ["none"]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            if col in ["variable", "bin", "trend"]:  # String variables
                self.assertEqual(list(df1[col]), list(df2[col]))
            else:  # Numeric variables
                npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

        # Case 2: One-valued variable and two-valued target
        input_df = pd.DataFrame({"x": [1, 1], "y": [0, 1]})
        df1 = woe.bin_woe_numeric_monotonic(variable=input_df["x"],
                                            target=input_df["y"],
                                            n_max=20,
                                            round_limit=2,
                                            lower_limit_bounded=True,
                                            upper_limit_bounded=True)
        df2 = pd.DataFrame({"variable": ["x"],
                            "bin_number": [1],
                            "lower_limit": [1],
                            "upper_limit": [1.01],
                            "bin": ["[1.0, 1.01)"],
                            "num_obs": [2],
                            "num_events": [1],
                            "num_non_events": [1],
                            "trend": ["none"]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            if col in ["variable", "bin", "trend"]:  # String variables
                self.assertEqual(list(df1[col]), list(df2[col]))
            else:  # Numeric variables
                npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

        # Case 3: Two-valued variable and two-valued target
        input_df = pd.DataFrame({"x": [1, 2], "y": [1, 0]})
        df1 = woe.bin_woe_numeric_monotonic(variable=input_df["x"],
                                            target=input_df["y"],
                                            n_max=20,
                                            round_limit=2,
                                            lower_limit_bounded=True,
                                            upper_limit_bounded=True)
        df2 = pd.DataFrame({"variable": ["x", "x"],
                            "bin_number": [1, 2],
                            "lower_limit": [1, 1.51],
                            "upper_limit": [1.51, 2.01],
                            "bin": ["[1.0, 1.51)", "[1.51, 2.01)"],
                            "num_obs": [1, 1],
                            "num_events": [1, 0],
                            "num_non_events": [0, 1],
                            "trend": ["decreasing", "decreasing"]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            if col in ["variable", "bin", "trend"]:  # String variables
                self.assertEqual(list(df1[col]), list(df2[col]))
            else:  # Numeric variables
                npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

        # Case 4: Multi-valued decreasing trend
        input_df = pd.DataFrame({"x": [1, 1, 1, 1, 0, 0, 2, 2, 3, 3, 3],
                                 "y": [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1]})
        df1 = woe.bin_woe_numeric_monotonic(variable=input_df["x"],
                                            target=input_df["y"],
                                            n_max=20,
                                            round_limit=2,
                                            lower_limit_bounded=True,
                                            upper_limit_bounded=True)
        df2 = pd.DataFrame({"variable": ["x", "x", "x"],
                            "bin_number": [1, 2, 3],
                            "lower_limit": [0.0, 1.0, 2.0],
                            "upper_limit": [1.0, 2.0, 3.01],
                            "bin": ["[0.0, 1.0)", "[1.0, 2.0)", "[2.0, 3.01)"],
                            "num_obs": [2, 4, 5],
                            "num_events": [2, 2, 1],
                            "num_non_events": [0, 2, 4],
                            "trend": ["decreasing", "decreasing", "decreasing"]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            if col in ["variable", "bin", "trend"]:  # String variables
                self.assertEqual(list(df1[col]), list(df2[col]))
            else:  # Numeric variables
                npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

        # Case 5: Multi-valued increasing trend
        input_df = pd.DataFrame({"x": [1, 1, 1, 1, 0, 0, 2, 2, 3, 3, 3],
                                 "y": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]})
        df1 = woe.bin_woe_numeric_monotonic(variable=input_df["x"],
                                            target=input_df["y"],
                                            n_max=20,
                                            round_limit=2,
                                            lower_limit_bounded=True,
                                            upper_limit_bounded=True)
        df2 = pd.DataFrame({"variable": ["x", "x"],
                            "bin_number": [1, 2],
                            "lower_limit": [0.0, 2.0],
                            "upper_limit": [2.0, 3.01],
                            "bin": ["[0.0, 2.0)", "[2.0, 3.01)"],
                            "num_obs": [6, 5],
                            "num_events": [0, 3],
                            "num_non_events": [6, 2],
                            "trend": ["increasing", "increasing"]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            if col in ["variable", "bin", "trend"]:  # String variables
                self.assertEqual(list(df1[col]), list(df2[col]))
            else:  # Numeric variables
                npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

        # Case 6: Multi-valued increasing trend lower bound unbounded
        input_df = pd.DataFrame({"x": [1, 1, 1, 1, 0, 0, 2, 2, 3, 3, 3],
                                 "y": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]})
        df1 = woe.bin_woe_numeric_monotonic(variable=input_df["x"],
                                            target=input_df["y"],
                                            n_max=20,
                                            round_limit=2,
                                            lower_limit_bounded=False,
                                            upper_limit_bounded=True)
        df2 = pd.DataFrame({"variable": ["x", "x"],
                            "bin_number": [1, 2],
                            "lower_limit": [-float("inf"), 2.0],
                            "upper_limit": [2.0, 3.01],
                            "bin": ["[-inf, 2.0)", "[2.0, 3.01)"],
                            "num_obs": [6, 5],
                            "num_events": [0, 3],
                            "num_non_events": [6, 2],
                            "trend": ["increasing", "increasing"]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            if col in ["variable", "bin", "trend"]:  # String variables
                self.assertEqual(list(df1[col]), list(df2[col]))
            else:  # Numeric variables
                npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

        # Case 7: Multi-valued increasing trend upper bound unbounded
        input_df = pd.DataFrame({"x": [1, 1, 1, 1, 0, 0, 2, 2, 3, 3, 3],
                                 "y": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]})
        df1 = woe.bin_woe_numeric_monotonic(variable=input_df["x"],
                                            target=input_df["y"],
                                            n_max=20,
                                            round_limit=2,
                                            lower_limit_bounded=True,
                                            upper_limit_bounded=False)
        df2 = pd.DataFrame({"variable": ["x", "x"],
                            "bin_number": [1, 2],
                            "lower_limit": [0.0, 2.0],
                            "upper_limit": [2.0, float("inf")],
                            "bin": ["[0.0, 2.0)", "[2.0, inf)"],
                            "num_obs": [6, 5],
                            "num_events": [0, 3],
                            "num_non_events": [6, 2],
                            "trend": ["increasing", "increasing"]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            if col in ["variable", "bin", "trend"]:  # String variables
                self.assertEqual(list(df1[col]), list(df2[col]))
            else:  # Numeric variables
                npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

        # Case 8: Multi-valued increasing trend lower and upper bounds unbounded
        input_df = pd.DataFrame({"x": [1, 1, 1, 1, 0, 0, 2, 2, 3, 3, 3],
                                 "y": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]})
        df1 = woe.bin_woe_numeric_monotonic(variable=input_df["x"],
                                            target=input_df["y"],
                                            n_max=20,
                                            round_limit=2,
                                            lower_limit_bounded=False,
                                            upper_limit_bounded=False)
        df2 = pd.DataFrame({"variable": ["x", "x"],
                            "bin_number": [1, 2],
                            "lower_limit": [-float("inf"), 2.0],
                            "upper_limit": [2.0, float("inf")],
                            "bin": ["[-inf, 2.0)", "[2.0, inf)"],
                            "num_obs": [6, 5],
                            "num_events": [0, 3],
                            "num_non_events": [6, 2],
                            "trend": ["increasing", "increasing"]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            if col in ["variable", "bin", "trend"]:  # String variables
                self.assertEqual(list(df1[col]), list(df2[col]))
            else:  # Numeric variables
                npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

    def test_bin_woe_char_direct(self):
        """Test bin_woe_char_direct."""
        woe = WeightOfEvidence()

        # Case 1: One-valued variable
        input_df = pd.DataFrame({"x": ["A", "A"], "y": [0, 1]})
        df1 = woe.bin_woe_char_direct(variable=input_df["x"], target=input_df["y"], n_max=10000)
        df2 = pd.DataFrame({"variable": ["x"],
                            "bin_number": [1],
                            "lower_limit": [np.nan],
                            "upper_limit": [np.nan],
                            "bin": ["A"],
                            "num_obs": [2],
                            "num_events": [1],
                            "num_non_events": [1],
                            "trend": ["none"]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            if col in ["variable", "bin", "trend"]:  # String variables
                self.assertEqual(list(df1[col]), list(df2[col]))
            else:  # Numeric variables
                npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

        # Case 2: Multiple valued variable
        input_df = pd.DataFrame({"x": ["C", "A", "A", "B", "C"], "y": [0, 1, 0, 1, 0]})
        df1 = woe.bin_woe_char_direct(variable=input_df["x"], target=input_df["y"], n_max=10000)
        df2 = pd.DataFrame({"variable": ["x", "x", "x"],
                            "bin_number": [1, 2, 3],
                            "lower_limit": [np.nan, np.nan, np.nan],
                            "upper_limit": [np.nan, np.nan, np.nan],
                            "bin": ["A", "C", "B"],
                            "num_obs": [2, 2, 1],
                            "num_events": [1, 0, 1],
                            "num_non_events": [1, 2, 0],
                            "trend": ["none", "none", "none"]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            if col in ["variable", "bin", "trend"]:  # String variables
                self.assertEqual(list(df1[col]), list(df2[col]))
            else:  # Numeric variables
                npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

        # Case 3: Multiple valued variable with constrained number of bins
        input_df = pd.DataFrame({"x": ["C", "A", "D", "A", "B", "C", "D", "D", "E", "E", "E"],
                                 "y": [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1]})
        df1 = woe.bin_woe_char_direct(variable=input_df["x"], target=input_df["y"], n_max=3)
        df2 = pd.DataFrame({"variable": ["x", "x", "x"],
                            "bin_number": [1, 2, 3],
                            "lower_limit": [np.nan, np.nan, np.nan],
                            "upper_limit": [np.nan, np.nan, np.nan],
                            "bin": ["D", "E", "everything_else"],
                            "num_obs": [3, 3, 5],
                            "num_events": [2, 2, 3],
                            "num_non_events": [1, 1, 2],
                            "trend": ["none", "none", "none"]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            if col in ["variable", "bin", "trend"]:  # String variables
                self.assertEqual(list(df1[col]), list(df2[col]))
            else:  # Numeric variables
                npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

        # Case 4: Mixed type
        input_df = pd.DataFrame({"x": ["C", "A", "D", "A", "B", "C", "D", "D", 1.15, 1.15, 1.15],
                                 "y": [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1]})
        df1 = woe.bin_woe_char_direct(variable=input_df["x"], target=input_df["y"], n_max=3)
        df2 = pd.DataFrame({"variable": ["x", "x", "x"],
                            "bin_number": [1, 2, 3],
                            "lower_limit": [np.nan, np.nan, np.nan],
                            "upper_limit": [np.nan, np.nan, np.nan],
                            "bin": ["1.15", "D", "everything_else"],
                            "num_obs": [3, 3, 5],
                            "num_events": [2, 2, 3],
                            "num_non_events": [1, 1, 2],
                            "trend": ["none", "none", "none"]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            if col in ["variable", "bin", "trend"]:  # String variables
                self.assertEqual(list(df1[col]), list(df2[col]))
            else:  # Numeric variables
                npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

    def test_calculate_woe_iv_from_freq_table(self):
        """Test calculate_woe_iv_from_freq_table."""
        woe = WeightOfEvidence()

        # Case 1: Typical small example
        input_df = pd.DataFrame({"num_events": [0, 1, 2, 2], "num_non_events": [1, 0, 4, 3]})
        df1 = woe.calculate_woe_iv_from_freq_table(df_freq=input_df)
        df2 = pd.DataFrame({"num_events": [0, 1, 2, 2],
                            "num_non_events": [1, 0, 4, 3],
                            "event_dist": [0, 0.2, 0.4, 0.4],
                            "non_event_dist": [0.125, 0, 0.5, 0.375],
                            "woe": [-0.223144, 0.336472, -0.223144, 0.064539],
                            "iv": [0.119115, 0.119115, 0.119115, 0.119115]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

    def test_bin_woe_missing(self):
        """Test bin_woe_missing."""
        woe = WeightOfEvidence()

        # Case 1: Single-valued result
        df1 = woe.bin_woe_missing(target=np.array([0]))
        df2 = pd.DataFrame({"variable": ["x"],
                            "bin_number": [1],
                            "lower_limit": [np.nan],
                            "upper_limit": [np.nan],
                            "bin": ["missing"],
                            "num_obs": [1],
                            "num_events": [0],
                            "num_non_events": [1],
                            "trend": ["none"]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            if col in ["variable", "bin", "trend"]:  # String variables
                self.assertEqual(list(df1[col]), list(df2[col]))
            else:  # Numeric variables
                npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

        # Case 2: Typical small example
        df1 = woe.bin_woe_missing(target=np.array([0, 0, 0, 1, 0, 0, 1]))
        df2 = pd.DataFrame({"variable": ["x"],
                            "bin_number": [1],
                            "lower_limit": [np.nan],
                            "upper_limit": [np.nan],
                            "bin": ["missing"],
                            "num_obs": [7],
                            "num_events": [2],
                            "num_non_events": [5],
                            "trend": ["none"]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            if col in ["variable", "bin", "trend"]:  # String variables
                self.assertEqual(list(df1[col]), list(df2[col]))
            else:  # Numeric variables
                npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

    def test_calculate_woe_iv_from_df(self):
        """Test calculate_woe_iv_from_df."""
        woe = WeightOfEvidence()

        # Case 1: Tiny case where there is only one NULL
        input_df = pd.DataFrame({"target": [1, 0, 1, 0, 1],
                                 "cat_cat": ["a", "b", "c", "d", np.nan],
                                 "num_cat": [5, 6, 7, 8, np.nan],
                                 "num_num_min_max": [0, 1, 2, 3, np.nan]})
        df1 = woe.calculate_woe_iv_from_df(df=input_df,
                                           var_list=["cat_cat",
                                                     "num_cat",
                                                     "num_num_min_max"],
                                           target_label="target")
        df2 = pd.DataFrame({"variable": ["cat_cat", "cat_cat", "cat_cat", "cat_cat", "cat_cat",
                                         "num_cat", "num_cat",
                                         "num_cat", "num_num_min_max", "num_num_min_max",
                                         "num_num_min_max"],
                            "bin_number": [1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 3],
                            "lower_limit": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5, 7,
                                            np.nan, 0, 2],
                            "upper_limit": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7,
                                            8.0001, np.nan, 2, 3.0001],
                            "bin": ["missing", "b", "d", "a", "c", "missing", "[5.0, 7.0)",
                                    "[7.0, 8.0001)", "missing",
                                    "[0.0, 2.0)", "[2.0, 3.0001)"],
                            "num_obs": [1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2],
                            "num_events": [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                            "num_non_events": [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1],
                            "trend": ["none", "none", "none", "none", "none", "none", "none",
                                      "none", "none", "none",
                                      "none"],
                            "event_dist": [0.333333, 0, 0, 0.333333, 0.333333, 0.333333, 0.333333,
                                           0.333333, 0.333333,
                                           0.333333, 0.333333],
                            "non_event_dist": [0, 0.5, 0.5, 0, 0, 0, 0.5, 0.5, 0, 0.5, 0.5],
                            "woe": [0.510826, -0.693147, -0.693147, 0.510826, 0.510826, 0.510826,
                                    -0.405465, -0.405465,
                                    0.510826, -0.405465, -0.405465],
                            "iv": [1.203973, 1.203973, 1.203973, 1.203973, 1.203973, 0.305430,
                                   0.305430, 0.305430,
                                   0.305430, 0.305430, 0.305430]})

        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            if col in ["variable", "bin", "trend"]:  # String variables
                self.assertEqual(list(df1[col]), list(df2[col]))
            else:  # Numeric variables
                npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

        # Case 2: Typical small example
        input_df = pd.DataFrame({"y": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                 "unary": [7] * 11,
                                 "binary": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                 "numeric_non_missing": [1.01, 1.01, 1.01, 1.01, 0.015, 0.015,
                                                         2.113, 2.113, 3.444, 3.444, 3.444],
                                 "all_missing": [np.nan] * 11,
                                 "numeric_missing": [np.nan, 7, 7, 7, 7, 7, 7, 7, 7, np.nan,
                                                     np.nan],
                                 "string_missing": [np.nan, np.nan, "c", "c", "a", "b", "d", "d",
                                                    "d", "b", np.nan]})
        df1 = woe.calculate_woe_iv_from_df(df=input_df,
                                           var_list=["unary",
                                                     "binary",
                                                     "numeric_non_missing",
                                                     "all_missing",
                                                     "numeric_missing",
                                                     "string_missing"],
                                           target_label="y",
                                           n_max_num=20,
                                           n_max_cat=3,
                                           round_limit=2,
                                           lower_limit_bounded={"numeric_non_missing": False},
                                           upper_limit_bounded={"numeric_missing": False})
        df2 = pd.DataFrame({"variable": ["binary", "binary", "string_missing", "string_missing",
                                         "string_missing", "string_missing", "numeric_non_missing",
                                         "numeric_non_missing", "numeric_missing",
                                         "numeric_missing", "all_missing", "unary"],
                            "bin_number": [1, 2, 1, 2, 3, 4, 1, 2, 1, 2, 1, 1],
                            "lower_limit": [0, 1, np.nan, np.nan, np.nan, np.nan,
                                            -float("inf"), 2.12, np.nan, 7.00, np.nan, 7.00],
                            "upper_limit": [1, 1.01, np.nan, np.nan, np.nan, np.nan,
                                            2.12, 3.45, np.nan, float("inf"), np.nan, 7.01],
                            "bin": ["[0.0, 1.0)", "[1.0, 1.01)", "missing", "everything_else",
                                    "b", "d", "[-inf, 2.12)", "[2.12, 3.45)", "missing",
                                    "[7.0, inf)", "missing", "[7.0, 7.01)"],
                            "num_obs": [6, 5, 3, 3, 2, 3, 8, 3, 3, 8, 11, 11],
                            "num_events": [6, 0, 1, 0, 2, 3, 3, 3, 2, 4, 6, 6],
                            "num_non_events": [0, 5, 2, 3, 0, 0, 5, 0, 1, 4, 5, 5],
                            "trend": ["decreasing", "decreasing", "none", "none", "none",
                                      "none", "increasing", "increasing", "none", "none", "none",
                                      "none"],
                            "event_dist": [1.00000, 0.00000, 0.16667, 0.00000, 0.33333, 0.50000,
                                           0.50000, 0.50000, 0.33333, 0.66667, 1.00000, 1.00000],
                            "non_event_dist": [0.0, 1.0, 0.4, 0.6, 0.0, 0.0, 1.0, 0.0, 0.2, 0.8,
                                               1.0, 1.0],
                            "woe": [1.098612, -1.098612, -0.875469, -0.788457, 0.510826, 0.693147,
                                    -0.693147, 0.693147, 0.510826, -0.182322, 0.00000, 0.00000],
                            "iv": [2.197225, 2.197225, 1.194199, 1.194199, 1.194199, 1.194199,
                                   0.693147, 0.693147, 0.092420, 0.092420, 0.000000, 0.000000]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            if col in ["variable", "bin", "trend"]:  # String variables
                self.assertEqual(list(df1[col]), list(df2[col]))
            else:  # Numeric variables
                npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)

    def test_map_to_woe(self):
        """Test map_to_woe."""
        woe = WeightOfEvidence()

        # Case 1: Numeric, missing value present and missing label present
        input_x = pd.Series([0, 1, 2, 1.1, 3, 4, -1, np.nan])
        input_df = pd.DataFrame({"lower_limit": [np.nan, 0, 1.1, 2],
                                 "upper_limit": [np.nan, 1.1, 2, 3.01],
                                 "bin": ["missing", "[0.0, 1.1)", "[1.1, 2.0)", "[2.0, 3.01)"],
                                 "woe": [0.2, 0.5, -0.1, 0.3]})
        array1 = woe.map_to_woe(variable=input_x, df_woe_map=input_df)
        array2 = np.array([0.5, 0.5, 0.3, -0.1, 0.3, np.nan, np.nan, 0.2])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 2: Numeric, missing value present and missing label absent
        input_x = pd.Series([0, 1, 2, 1.1, 3, 4, -1, np.nan])
        input_df = pd.DataFrame({"lower_limit": [0, 1.1, 2],
                                 "upper_limit": [1.1, 2, 3.01],
                                 "bin": ["[0.0, 1.1)", "[1.1, 2.0)", "[2.0, 3.01)"],
                                 "woe": [0.5, -0.1, 0.3]})
        array1 = woe.map_to_woe(variable=input_x, df_woe_map=input_df)
        array2 = np.array([0.5, 0.5, 0.3, -0.1, 0.3, np.nan, np.nan, np.nan])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 3: Categorical, missing value present and missing label present
        input_x = pd.Series(["A", "B", "C", "D", np.nan])
        input_df = pd.DataFrame({"lower_limit": [np.nan, np.nan, np.nan, np.nan],
                                 "upper_limit": [np.nan, np.nan, np.nan, np.nan],
                                 "bin": ["missing", "A", "B", "C"],
                                 "woe": [0.5, -0.1, 0.3, 0.4]})
        array1 = woe.map_to_woe(variable=input_x, df_woe_map=input_df)
        array2 = np.array([-0.1, 0.3, 0.4, np.nan, 0.5])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 4: Categorical, missing value present and missing label absent
        input_x = pd.Series(["A", "B", "C", "D", np.nan])
        input_df = pd.DataFrame({"lower_limit": [np.nan, np.nan, np.nan],
                                 "upper_limit": [np.nan, np.nan, np.nan],
                                 "bin": ["A", "B", "C"],
                                 "woe": [-0.1, 0.3, 0.4]})
        array1 = woe.map_to_woe(variable=input_x, df_woe_map=input_df)
        array2 = np.array([-0.1, 0.3, 0.4, np.nan, np.nan])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 5: Numeric treated as categorical
        input_x = pd.Series([1.15, 1.25, 1.35, 1.45, np.nan])
        input_df = pd.DataFrame({"lower_limit": [np.nan, np.nan, np.nan],
                                 "upper_limit": [np.nan, np.nan, np.nan],
                                 "bin": ["1.15", "1.25", "1.35"],
                                 "woe": [-0.1, 0.3, 0.4]})
        array1 = woe.map_to_woe(variable=input_x, df_woe_map=input_df)
        array2 = np.array([-0.1, 0.3, 0.4, np.nan, np.nan])
        npt.assert_almost_equal(array1, array2, decimal=4)

    def test_map_df_to_woe(self):
        """Test map_df_to_woe."""
        woe = WeightOfEvidence()

        # Case 1: Small example with various cases
        input_df = pd.DataFrame({"x0": [1, 2, 3, 4],
                                 "x1": [0, 1, 2, np.nan],
                                 "x2": ["A", "B", np.nan, "D"],
                                 "x3": [-100, 2, 3, 400],
                                 "x4": [4, 3, 2, 1],
                                 "x5": ["A", "B", np.nan, "D"],
                                 "x6": ["A", "B", np.nan, "D"]})
        input_df_map = pd.DataFrame(
            {"variable": ["x2", "x2", "x2", "x1", "x5", "x3", "x3", "x5", "x5",
                          "x5", "x6", "x6"],
             "lower_limit": [np.nan, np.nan, np.nan, 0, 0, -float("inf"), 1.5,
                             np.nan, np.nan, np.nan, np.nan, np.nan],
             "upper_limit": [np.nan, np.nan, np.nan, 2.01, 3, 1.5,
                             float("inf"), np.nan, np.nan, np.nan, np.nan,
                             np.nan],
             "bin": ["missing", "A", "B", "[0.0, 2.01)", "[0.0, 3.01)",
                     "[-inf, 1.50)", "[1.50, inf)", "missing", "A",
                     "everything_else", "A", "everything_else"],
             "woe": [0.5, -0.1, 0.3, 0.2, 0.4, 0.1, -0.1, 0.0, 0.1, 0.2, 0.1,
                     0.2]})
        df1 = woe.map_df_to_woe(df=input_df, df_woe_map=input_df_map)
        df2 = pd.DataFrame({"x0": [1, 2, 3, 4],
                            "x1": [0.2, 0.2, 0.2, np.nan],
                            "x2": [-0.1, 0.3, 0.5, np.nan],
                            "x3": [0.1, -0.1, -0.1, -0.1],
                            "x4": [4, 3, 2, 1],
                            "x5": [0.1, 0.2, 0.0, 0.2],
                            "x6": [0.1, 0.2, np.nan, 0.2]})
        assert list(df1.columns) == list(df2.columns)
        for col in list(df1.columns):
            npt.assert_almost_equal(list(df1[col]), list(df2[col]), decimal=4)


if __name__ == "__main__":
    unittest.main()
