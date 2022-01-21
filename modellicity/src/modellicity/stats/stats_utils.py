"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import logging
import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from scipy import stats
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class StatsUtils:
    """General purpose statistics functions."""

    @staticmethod
    def t_test(sample_1: List[float], sample_2: List[float]) -> float:
        """
        Statistical T-test.

        Test to determine if there is a significant difference between
        the means of two groups.
        :param sample_1:
        :param sample_2:
        :return:
        """
        return stats.f_oneway(sample_1, sample_2).pvalue

    @staticmethod
    def kolmogorov_smirnov_test(sample_1: List[float], sample_2: List[float]) -> float:
        """
        Kolmogorov-Smirnov statistical test.

        Test of the equality of one-dimensional probability distributions that
        can be used to compare a sample with a reference probability
        distribution (one-sample K–S test), or to compare two samples
        (two-sample K–S test).
        :param sample_1:
        :param sample_2:
        :return:
        """
        return stats.ks_2samp(sample_1, sample_2).statistic

    @staticmethod
    def normalize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the elements of a given dataframe.

        :param dataframe:
        :return:
        """
        # If the dataframe is not numeric, we cannot perform normalization on it.
        # Ensure that the types of all series that compose the dataframe are
        # numeric, and if they are not, raise an exception and return the original
        # dataframe provided as input.
        for series in dataframe:
            if is_numeric_dtype(dataframe[series]) is False:
                logger.exception(
                    f"Exception: Attempt to normalize non-numeric series '{series}' "
                    f"in dataframe. No action taken."
                )
                raise ValueError
        return (dataframe - dataframe.mean()) / dataframe.std()

    @staticmethod
    def get_binned_quantiles(numeric_series: pd.Series, quantiles: int) -> pd.DataFrame:
        """
        Get the quantiles as binned entities.

        Given a (numeric) series, and an integer representing the number of
        quantiles, return a dataframe where each entry represent a bin with
        a given range of values.
        :param numeric_series:
        :param quantiles:
        :return:
        """
        if is_numeric_dtype(numeric_series) is False:
            logger.exception(
                f"Exception: Attempt to get quantiles of non-numeric series: '{numeric_series}' "
                f"No action taken."
            )
            raise ValueError

        _, bins = pd.qcut(
            x=numeric_series, q=quantiles, retbins=True, duplicates="drop"
        )

        binned_ranges_dict: Dict[str, list] = {}
        binned_ranges_dict["bin"] = []
        binned_ranges_dict["lower_limit"] = list(bins[0: len(bins) - 1])
        binned_ranges_dict["upper_limit"] = list(bins[1: len(bins)])

        if numeric_series.isnull().values.any():
            binned_ranges_dict["lower_limit"].insert(0, np.nan)
            binned_ranges_dict["upper_limit"].insert(0, np.nan)

        binned_ranges_dict["bin"] = list(
            range(1, len(binned_ranges_dict["lower_limit"]) + 1)
        )

        binned_ranges_dataframe = pd.DataFrame.from_dict(binned_ranges_dict)
        return binned_ranges_dataframe

    @staticmethod
    def convert_numeric_series_to_binned_series(
        numeric_series: pd.Series,
        bin_ranges: pd.DataFrame,
        extra_margin_upper_limit: float,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Convert a series of type numeric to one that is binned.

        :param numeric_series:
        :param bin_ranges:
        :param extra_margin_upper_limit:
        :return:
        """
        # Incorporate the possibility of out-of-bounds values in series relative
        # to bin_ranges. We create bin_ranges_copy that extends the limits
        # however, later on, we will assign the out-of-bounds binds to NULL
        # Also incorporate the possibility that NULLs may exist in the series
        # but aren't captured in bin_ranges
        bin_ranges_copy = pd.DataFrame()
        lower_limit_list = []
        upper_limit_list = []
        missing_not_incorporated = False
        if (
            numeric_series.isnull().values.any()
            and not bin_ranges["lower_limit"].isnull().values.any()
        ):
            missing_not_incorporated = True
            lower_limit_list.append(np.nan)
            upper_limit_list.append(np.nan)
        if numeric_series.min() < bin_ranges["lower_limit"].min():
            lower_limit_min = numeric_series.min()
            upper_limit_min = bin_ranges["lower_limit"].min()
            lower_limit_list.append(lower_limit_min)
            upper_limit_list.append(upper_limit_min)
        lower_limit_list.extend(list(bin_ranges["lower_limit"]))
        upper_limit_list.extend(list(bin_ranges["upper_limit"]))
        if numeric_series.max() > bin_ranges["upper_limit"].max():
            lower_limit_max = numeric_series.max()
            upper_limit_max = lower_limit_max + 100
            lower_limit_list.append(lower_limit_max)
            upper_limit_list.append(upper_limit_max)
        bin_ranges_copy["lower_limit"] = lower_limit_list
        bin_ranges_copy["upper_limit"] = upper_limit_list

        lower_limit_list = []
        upper_limit_list = []
        cat_label_list = []

        for i in range(bin_ranges_copy.shape[0]):
            lower_limit = bin_ranges_copy["lower_limit"].iloc[i]
            upper_limit = bin_ranges_copy["upper_limit"].iloc[i]

            if pd.isnull(lower_limit) and pd.isnull(upper_limit):
                lower_limit_list.append(bin_ranges_copy["lower_limit"].min() - 100)
                upper_limit_list.append(bin_ranges_copy["lower_limit"].min())
                if missing_not_incorporated:
                    cat_label_list.append(str(np.nan))
                else:
                    cat_label_list.append("[Missing]")
            elif pd.isnull(lower_limit) and not pd.isnull(upper_limit):
                lower_limit_list.append(bin_ranges_copy["upper_limit"].min() - 100)
                upper_limit_list.append(upper_limit)
                cat_label_list.append("[Missing, " + str(upper_limit) + ")")
            else:
                # Treat out of bounds cases
                # Note: we use bin_ranges here not bin_ranges_copy because
                # bin_ranges_copy may have an out-of-bounds interval
                if (
                    lower_limit < bin_ranges["lower_limit"].min()
                    or upper_limit > bin_ranges["upper_limit"].max()
                ):
                    lower_limit_list.append(lower_limit)
                    upper_limit_list.append(upper_limit)
                    cat_label_list.append(str(np.nan))
                # Treat normal case
                else:
                    if upper_limit == bin_ranges["upper_limit"].max():
                        upper_limit += extra_margin_upper_limit
                    lower_limit_list.append(lower_limit)
                    upper_limit_list.append(upper_limit)
                    cat_label_list.append(
                        "[" + str(lower_limit) + ", " + str(upper_limit) + ")"
                    )
        bin_mapper = pd.DataFrame()
        bin_mapper["index"] = ["x"] * len(lower_limit_list)
        bin_mapper["lower_limit"] = lower_limit_list
        bin_mapper["upper_limit"] = upper_limit_list
        bin_mapper["cat_label"] = cat_label_list

        # Dealing with NULL edge case
        dataframe_temp = pd.DataFrame()
        dataframe_temp["index"] = ["x"] * len(numeric_series)
        dataframe_temp["var_label"] = numeric_series
        dataframe_temp["var_label"].fillna(
            bin_ranges_copy["lower_limit"].min() - 50, inplace=True
        )
        dataframe_temp = pd.merge(dataframe_temp, bin_mapper, on="index", how="inner")

        dataframe_temp = dataframe_temp[
            dataframe_temp["var_label"] >= dataframe_temp["lower_limit"]
        ]
        dataframe_temp = dataframe_temp[
            dataframe_temp["var_label"] < dataframe_temp["upper_limit"]
        ]
        numeric_range_to_bin_map = dataframe_temp[
            ["lower_limit", "upper_limit", "cat_label"]
        ].copy()
        numeric_range_to_bin_map.sort_values(by=["lower_limit"], inplace=True)
        numeric_range_to_bin_map.drop_duplicates(inplace=True)

        if bin_ranges["lower_limit"].isnull().values.any():
            temp = numeric_range_to_bin_map.copy()  # To supress copy warning message
            temp.loc[0, "lower_limit"] = np.nan
            numeric_range_to_bin_map = temp.copy()
        if bin_ranges["upper_limit"].isnull().values.any():
            temp = numeric_range_to_bin_map.copy()  # To supress copy warning message
            temp.loc[0, "upper_limit"] = np.nan
            numeric_range_to_bin_map = temp.copy()

        numeric_range_to_bin_map.rename(
            columns={"cat_label": numeric_series.name + "_cat"}, inplace=True
        )
        numeric_range_to_bin_map.reset_index(drop=True, inplace=True)

        binned_series = pd.Series(list(dataframe_temp["cat_label"]))
        binned_series.name = numeric_series.name
        return binned_series, numeric_range_to_bin_map
