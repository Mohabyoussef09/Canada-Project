"""@copyright Copyright © 2020, Modellicity Inc., All Rights Reserved."""
from pandas.core.dtypes.missing import isna
from modellicity.stats.weight_of_evidence_utils import (
    apply_numeric_binning_left_open_right_closed,
    is_monotonic,
    largest_unique_quantiles,
    most_balanced_two_bin_split_left_open_right_closed,
    round_bin_edges_left_open_right_closed,
    round_ceil,
    round_floor,
)
import numpy as np
import numpy.testing as npt
import pandas as pd
import unittest


class TestUtils(unittest.TestCase):
    """Unit tests for Utils Module."""

    def test_apply_numeric_binning_left_open_right_closed(self):
        """Test apply_numeric_binning_left_open_right_closed."""
        x, bins = np.array([4, 4.99, 5, 0.99, 1, 6, np.nan]), np.array([1, 3.01, 4.01, 5])
        array1, ids1 = apply_numeric_binning_left_open_right_closed(x, bins)
        array2, ids2 = \
            np.array([pd.Interval(3.01, 4.01, closed="left"), pd.Interval(4.01, 5.0, closed="left"),
                      np.nan, np.nan, pd.Interval(1.0, 3.01, closed="left"), np.nan, np.nan],
                     dtype=object), np.array([2, 3, 0, 0, 1, 0, 0])
        for i in range(array1.size):
            if not isna(array1[i]) or not isna(array2[i]):
                self.assertEqual(array1[i], array2[i])
                self.assertEqual(ids1[i], ids2[i])

    def test_is_monotonic(self):
        """Test is_monotonic."""
        # Case 1: Single-valued variable with only one element
        tuple1a, tuple1b = is_monotonic(np.array([1]))
        tuple2a, tuple2b = False, False
        self.assertEqual(tuple1a, tuple2a)
        self.assertEqual(tuple1b, tuple2b)

        # Case 2: Single-valued variable with more than one element
        tuple1a, tuple1b = is_monotonic(np.array([1, 1, 1]))
        tuple2a, tuple2b = False, False
        self.assertEqual(tuple1a, tuple2a)
        self.assertEqual(tuple1b, tuple2b)

        # Case 3: Monotonically increasing array
        tuple1a, tuple1b = is_monotonic(np.array([-1, 1.1, 3]))
        tuple2a, tuple2b = False, True
        self.assertEqual(tuple1a, tuple2a)
        self.assertEqual(tuple1b, tuple2b)

        # Case 4: Monotonically decreasing array
        tuple1a, tuple1b = is_monotonic(np.array([3, 1.1, -1]))
        tuple2a, tuple2b = True, False
        self.assertEqual(tuple1a, tuple2a)
        self.assertEqual(tuple1b, tuple2b)

        # Case 5: Non-monotonic array
        tuple1a, tuple1b = is_monotonic(np.array([-3, 1.1, -1]))
        tuple2a, tuple2b = False, False
        self.assertEqual(tuple1a, tuple2a)
        self.assertEqual(tuple1b, tuple2b)

    def test_largest_unique_quantiles(self):
        """Test largest_unique_quantiles."""
        # Case 1: Single-valued variable with only one element
        array1 = largest_unique_quantiles(np.array([1]), q=4)
        array2 = np.array([1])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 2: Single-valued variable with many elements
        array1 = largest_unique_quantiles(np.array([-1, -1, -1, -1, -1]), q=4)
        array2 = np.array([-1])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 3: Two-valued variables with two values very close
        array1 = largest_unique_quantiles(np.array([1.001, 1]), q=4)
        array2 = np.array([1, 1.0005, 1.001])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 4: Two-valued variable with three values
        array1 = largest_unique_quantiles(np.array([2, 1, 2]), q=4)
        array2 = np.array([1, 2])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 5: Multiple-valued variable
        array1 = \
            largest_unique_quantiles(np.array([-1, -2, 3, 4, 5, 5, 6, 7, 8, 9]), q=3)
        array2 = np.array([-2, 4, 6, 9])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 6: Negative values with several decimal places
        array1 = \
            largest_unique_quantiles(np.array([-1.1115, -2.1115, -3.1115, -0.001]), q=2)
        array2 = np.array([-3.1115, -1.6115, -0.001])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 7: Binary split on heavily concentrated variable on minimum
        array1 = largest_unique_quantiles(np.array([1, 1, 1, 2, 3]), q=2)
        array2 = np.array([1, 3])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 8: Binary split on heavily concentrated variable on maximum
        array1 = largest_unique_quantiles(np.array([1, 2, 3, 3, 3]), q=2)
        array2 = np.array([1, 3])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 9: Binary split on heavily concentrated variable on left
        array1 = largest_unique_quantiles(np.array([0, 1, 1, 1, 2, 3]), q=2)
        array2 = np.array([0, 1, 3])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 10: Binary split on heavily concentrated variable on right
        array1 = largest_unique_quantiles(np.array([0, 1, 2, 2, 2, 3]), q=2)
        array2 = np.array([0, 2, 3])
        npt.assert_almost_equal(array1, array2, decimal=4)

    def test_most_balanced_two_bin_split_left_open_right_closed(self):
        """Test most_balanced_two_bin_split_left_open_right_closed."""
        # Case 1: Single-valued variable with only one element
        array1 = most_balanced_two_bin_split_left_open_right_closed(x=np.array([1]), round_limit=2)
        array2 = np.array([1, 1.01])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 2: Three values very close to each other and differ only when looking beyond given
        #         decimal places
        array1 = \
            most_balanced_two_bin_split_left_open_right_closed(x=np.array([-1.0001, -1, -0.999]),
                                                               round_limit=2)
        array2 = np.array([-1.01, -1, -0.99])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 3: Three values very close to each other and with two of them differing only when
        #         looking beyond given
        #         decimal places
        array1 = \
            most_balanced_two_bin_split_left_open_right_closed(x=np.array([-1.0001, -1, -0.99]),
                                                               round_limit=2)
        array2 = np.array([-1.01, -1, -0.98])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 4: Two-valued variable with three values
        array1 = \
            most_balanced_two_bin_split_left_open_right_closed(x=np.array([2, 1, 2]), round_limit=2)
        array2 = np.array([1.0, 2, 2.01])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 5: Multiple-valued variable
        array1 = \
            most_balanced_two_bin_split_left_open_right_closed(x=np.array([-1, -2, 3, 4, 5, 5, 6,
                                                                           7, 8, 9]), round_limit=2)
        array2 = np.array([-2.0, 5.0, 9.01])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 6: Negative values with several decimal places
        array1 = \
            most_balanced_two_bin_split_left_open_right_closed(x=np.array([-1.1115, -2.1115,
                                                                           -3.1115, -0.001]),
                                                               round_limit=3)
        array2 = np.array([-3.112, -1.111, 0])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 7: Binary split on heavily concentrated variable on minimum
        array1 = \
            most_balanced_two_bin_split_left_open_right_closed(x=np.array([1, 1, 1, 2, 3]),
                                                               round_limit=2)
        array2 = np.array([1.0, 2, 3.01])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 8: Binary split on heavily concentrated variable on maximum
        array1 = \
            most_balanced_two_bin_split_left_open_right_closed(x=np.array([1, 2, 3, 3, 3]),
                                                               round_limit=2)
        array2 = np.array([1.0, 3, 3.01])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 9: Binary split on heavily concentrated variable on left
        array1 = \
            most_balanced_two_bin_split_left_open_right_closed(x=np.array([0, 1, 1, 1, 2, 3]),
                                                               round_limit=2)
        array2 = np.array([0.0, 2.0, 3.01])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 10: Binary split on heavily concentrated variable on right
        array1 = \
            most_balanced_two_bin_split_left_open_right_closed(x=np.array([0, 1, 2, 2, 2, 3]),
                                                               round_limit=2)
        array2 = np.array([0.0, 2.0, 3.01])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 11: Two values within tolerance range collapsing to only one bin
        array1 = \
            most_balanced_two_bin_split_left_open_right_closed(x=np.array([2.1, 2.9]),
                                                               round_limit=0)
        array2 = np.array([2, 3])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 12: Two values within tolerance range eligible to be split to two bins
        array1 = \
            most_balanced_two_bin_split_left_open_right_closed(x=np.array([2.1, 3.1]),
                                                               round_limit=0)
        array2 = np.array([2, 3, 4])
        npt.assert_almost_equal(array1, array2, decimal=4)

    def test_round_bin_edges_left_open_right_closed(self):
        """Test round_bin_edges_left_open_right_closed."""
        # Case 1: One-valued integer
        array1 = round_bin_edges_left_open_right_closed(np.array([1]), round_limit=1)
        array2 = np.array([1, 1.1])
        npt.assert_almost_equal(array1, array2)

        # Case 2: One-valued float
        array1 = round_bin_edges_left_open_right_closed(np.array([1.1]), round_limit=0)
        array2 = np.array([1, 2])
        npt.assert_almost_equal(array1, array2)

        # Case 3: Integers with zero decimal places for rounding
        array1 = round_bin_edges_left_open_right_closed(np.array([1, 2, 3]), round_limit=0)
        array2 = np.array([1, 2, 4])
        npt.assert_almost_equal(array1, array2)

        # Case 4: Integers with one decimal place rounding
        array1 = round_bin_edges_left_open_right_closed(np.array([1, 2, 3]), round_limit=1)
        array2 = np.array([1, 2, 3.1])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 5: Floats with zero decimal places for rounding
        array1 = round_bin_edges_left_open_right_closed(np.array([0.99, 2.12, 3.01]), round_limit=0)
        array2 = np.array([0, 3, 4])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 6: Floats with one decimal place rounding
        array1 = round_bin_edges_left_open_right_closed(np.array([0.99, 2.12, 3.01]), round_limit=1)
        array2 = np.array([0.9, 2.2, 3.1])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 7: Negative numbers
        array1 = round_bin_edges_left_open_right_closed(np.array([-1]), round_limit=2)
        array2 = np.array([-1, -0.99])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 8: Very close edges
        array1 = round_bin_edges_left_open_right_closed(np.array([1, 1.0005, 1.001]), round_limit=2)
        array2 = np.array([1, 1.01])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 9: Very close edges with zero rounding
        array1 = round_bin_edges_left_open_right_closed(np.array([2.1, 3.0, 3.1]), round_limit=0)
        array2 = np.array([2, 3, 4])
        npt.assert_almost_equal(array1, array2)

        # Case 10: Edges with tolerance range of each other yet distinct when rounded
        array1 = round_bin_edges_left_open_right_closed(np.array([2.1, 2.9]), round_limit=0)
        array2 = np.array([2, 3])
        npt.assert_almost_equal(array1, array2)

    def test_round_ceil(self):
        """Test round_ceil."""
        # Case 1: Array of integers
        array1 = round_ceil(np.array([1, 2]), decimals=1)
        array2 = np.array([1, 2])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 2: Integer number
        num1 = round_ceil(4, decimals=1)
        num2 = 4
        npt.assert_almost_equal(num1, num2, decimal=4)

        # Case 3: Array with decimals
        array1 = \
            round_ceil(np.array([10.7499, 10.74999, 10.75, 10.7500, 10, 10.7501, 10.75011,
                                 10.75010000001]), decimals=4)
        array2 = np.array([10.7499, 10.75, 10.75, 10.75, 10, 10.7501, 10.7502, 10.7502])
        npt.assert_almost_equal(array1, array2, decimal=5)

    def test_round_floor(self):
        """Test round_floor."""
        # Case 1: Array of integers
        array1 = round_ceil(np.array([1, 2]), decimals=1)
        array2 = np.array([1, 2])
        npt.assert_almost_equal(array1, array2, decimal=4)

        # Case 2: Integer number
        num1 = round_ceil(4, decimals=1)
        num2 = 4
        npt.assert_almost_equal(num1, num2, decimal=4)

        # Case 3: Array with decimals
        array1 = \
            round_floor(np.array([10.7499, 10.74999, 10.75, 10.7500, 10, 10.7501, 10.75011,
                                  10.75010000001]), decimals=4)
        array2 = np.array([10.7499, 10.7499, 10.75, 10.75, 10, 10.7501, 10.7501, 10.7501])
        npt.assert_almost_equal(array1, array2, decimal=5)


if __name__ == "__main__":
    unittest.main()
