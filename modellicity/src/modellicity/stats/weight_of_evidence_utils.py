"""Utility functions."""

import numpy as np
import pandas as pd
import pandas.core.algorithms as algos
from pandas.core.dtypes.missing import isna
from typing import Tuple, Union


def apply_numeric_binning_left_open_right_closed(x: np.ndarray, bins: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    """Apply binning given an array of bin edges."""
    bins = algos.unique(bins)
    ids = bins.searchsorted(x, side="right")  # TODO: Find way to speed up
    labels = pd.IntervalIndex.from_breaks(bins, closed="left")
    na_mask = isna(x) | (ids >= bins.size)
    np.putmask(ids, na_mask, 0)
    labels_cat = pd.Categorical(labels, categories=labels, ordered=True)
    return algos.take_nd(labels_cat, ids - 1), ids


def is_monotonic(x: np.ndarray) -> Tuple[bool, bool]:
    """Check if a numeric array is monotonic."""
    if x.size <= 1:
        return False, False
    is_decreasing, is_increasing = False, False
    if np.all(np.diff(x) < 0):
        is_decreasing = True
    elif np.all(np.diff(x) > 0):
        is_increasing = True
    return is_decreasing, is_increasing


def largest_unique_quantiles(x: np.ndarray, q: int) -> np.ndarray:
    """Find largest quantiles that are unique."""
    x_size = x.size
    bins = np.array([])
    if x_size == 0:
        return bins
    q_final = min(x_size, q)
    is_unique = False
    while not is_unique:
        quantiles = np.linspace(0, 1, q_final + 1)
        bins = np.array(np.quantile(x, quantiles))  # TODO: Find way to speed up
        if bins.size == algos.unique(bins).size:
            is_unique = True
        else:
            q_final -= 1
    return bins


def most_balanced_two_bin_split_left_open_right_closed(x: np.ndarray, round_limit: int):
    """Calculate edges of a vector to bin it into two groups as balanced as can be."""
    x_rounded = np.sort(x.astype(np.float64))
    x_min, x_max = x_rounded[0], x_rounded[-1]
    tol = 1 / 10 ** round_limit
    x_lower_edge, x_upper_edge = round_floor(x_min, round_limit), round_ceil(x_max, round_limit)
    if x_max == x_upper_edge:
        x_upper_edge += tol
    if np.isclose(x_lower_edge, x_upper_edge, atol=tol):
        return np.array([x_lower_edge, x_upper_edge])
    x_rounded[0] = x_lower_edge
    x_rounded[1:] = round_ceil(x_rounded[1:], round_limit)
    if x_max == x_rounded[-1]:
        x_rounded[-1] = x_max + tol
    x_rounded_unique, freq_x = np.unique(x_rounded, return_counts=True)
    perc_x = freq_x / np.sum(freq_x)
    perc_cum_x = np.cumsum(perc_x)
    perc_tail_x = 1 - perc_cum_x
    abs_diff_perc = np.abs(perc_cum_x - perc_tail_x)
    index = np.where(np.isclose(abs_diff_perc, np.min(abs_diff_perc), atol=tol))[0][0]
    max_value_bin1 = x_rounded_unique[index]
    bin2_list_indices = np.where(x_rounded_unique > max_value_bin1)[0]
    min_value_bin2 = x_rounded_unique[bin2_list_indices[0]]
    if bin2_list_indices.size == 1:  # The last edge is the only value in the list
        x_intermediate_edge = \
            round_ceil((max_value_bin1 + min_value_bin2) / 2, round_limit)
    else:
        x_intermediate_edge = min_value_bin2
    return algos.unique(np.array([x_lower_edge, x_intermediate_edge, x_upper_edge]))


def round_bin_edges_left_open_right_closed(bins: np.ndarray, round_limit: int) -> np.ndarray:
    """Apply rounding to bin edges for left open, right closed interval creation."""
    bins_rounded = bins.astype(np.float64)
    bins_min, bins_max = bins_rounded[0], bins_rounded[-1]
    tol = 1 / 10 ** round_limit
    if bins_rounded.size == 1:
        bins_rounded = \
            np.array([round_floor(bins_min, round_limit), round_ceil(bins_max, round_limit)])
        if bins_max == bins_rounded[-1]:
            bins_rounded[-1] = bins_max + tol
        return bins_rounded
    bins_rounded[0] = round_floor(bins_min, round_limit)
    bins_rounded[1:] = round_ceil(bins_rounded[1:], round_limit)
    bins_rounded = algos.unique(bins_rounded)
    if bins_max == bins_rounded[-1]:
        bins_rounded[-1] = bins_max + tol
    return bins_rounded


def round_ceil(x: Union[np.ndarray, np.float64, int], decimals: int = 4) -> \
        Union[np.ndarray, np.float64, int]:
    """Get ceiling rounded to a given number of decimal places."""
    if np.all(np.equal(np.mod(x, 1), 0)):  # Check if integer.
        return x
    return np.ceil(x * 10 ** decimals) / 10 ** decimals


def round_floor(x: Union[np.ndarray, np.float64, int], decimals: int = 4) -> \
        Union[np.ndarray, np.float64, int]:
    """Get floor rounded to a given number of decimal places."""
    if np.all(np.equal(np.mod(x, 1), 0)):  # Check if integer.
        return x
    return np.floor(x * 10 ** decimals) / 10 ** decimals


if __name__ == "__main__":
    pass
