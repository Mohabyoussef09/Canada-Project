"""Remove outliers transformers."""

import logging
import time
from typing import List

from modellicity.data_source import DataSource
from modellicity.pipeline import PipelineOperation

log = logging.getLogger(__name__)
filelog = logging.getLogger("file")


class RemoveOutliers(PipelineOperation):
    """Remove outliers pipeline operation."""

    def __init__(
        self,
        outlier_threshold: float = 3.29,
        lower_percentile_threshold: float = 1,
        upper_percentile_threshold: float = 99,
        exceptions: List[str] = None,
    ):
        """
        Initialize remove outliers pipeline operation.

        :param outlier_threshold:
        :param lower_percentile_threshold:
        :param upper_percentile_threshold:
        :param exceptions:
        """
        self.outlier_threshold = outlier_threshold
        self.lower_percentile_threshold = lower_percentile_threshold
        self.upper_percentile_threshold = upper_percentile_threshold
        self.exceptions = exceptions

    def run(self, data: DataSource) -> DataSource:
        """
        Remove columns exceeding the provided missingness threshold.

        :param data: Data source to transform.
        :return: New data source with outliers removed.
        """
        return remove_outliers(
            data,
            self.outlier_threshold,
            self.lower_percentile_threshold,
            self.upper_percentile_threshold,
            self.exceptions
        )


def remove_outliers(
    ds: DataSource,
    outlier_threshold: float = 3.29,
    lower_percentile_threshold: float = 1,
    upper_percentile_threshold: float = 99,
    exceptions: List[str] = None
) -> DataSource:
    """
    Given a dataframe, treat the outliers according to the `outlier_threshold`.

    :param ds: Data source to transform.
    :param outlier_threshold:
    :param lower_percentile_threshold:
    :param upper_percentile_threshold:
    :param exceptions:
    :return:
    """
    log.info("Performing outlier treatment.")

    if ds.data.empty:
        log.warning("Received empty dataframe.")
        return ds

    if not exceptions:
        exceptions = []

    start = time.perf_counter()
    log.info(f"Shape before treatment: {ds.data.shape}.")

    df_numeric = ds.data[ds.data.get_all_numeric_object()]
    df_final = ds.data.copy()

    log.info(
        f"Found {len(df_numeric.columns)} numeric variables "
        f"out of {len(ds.data.columns)} total variables"
    )

    # Obtain the outliers based on `outlier_threshold` from the dataframe.
    outlier_list = df_numeric.get_outliers(outlier_threshold)

    # Remove the any variables from list of outliers that are present in list of exceptions.
    outlier_list = list(set(outlier_list) - set(exceptions))

    log.info(f"Found {len(outlier_list)} outlier(s).")

    # Floor-and-cap the variables deemed to be outliers.
    df_final[outlier_list] = df_numeric[outlier_list].floor_and_cap(
        lower_percentile_threshold, upper_percentile_threshold
    )

    log.info(f"Shape after treatment: {df_final.shape}.")

    end = time.perf_counter()
    log.info(f"End outlier treatment. Time taken: {end-start} seconds")

    return DataSource(df_final)
