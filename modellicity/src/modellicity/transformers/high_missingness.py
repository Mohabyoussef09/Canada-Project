"""Remove high missingness transformers."""

import logging
import time
from typing import List

from tabulate import tabulate

from modellicity.data_source import DataSource
from modellicity.pipeline import PipelineOperation

log = logging.getLogger(__name__)
filelog = logging.getLogger("file")


class RemoveHighMissingness(PipelineOperation):
    """Remove high missingness pipeline operation."""

    def __init__(self, threshold: float = 0.8, exceptions: List[str] = None):
        """
        Initialize remove high missingness pipeline operation.

        :param threshold: Missigness threshold for removal (0-1).
        :param exceptions: Exception columns which should not be checked for missingness.
        """
        self.threshold = threshold
        self.exceptions = exceptions

    def run(self, data: DataSource) -> DataSource:
        """
        Remove columns exceeding the provided missingness threshold.

        :param data: Data source to transform.
        :return: New data source with high missingness columns removed.
        """
        return remove_high_missingness(data, self.threshold, self.exceptions)


def remove_high_missingness(
    ds: DataSource, threshold: float = 0.8, exceptions: List[str] = None
) -> DataSource:
    """
    Remove columns exceeding the provided missingness threshold.

    :param ds: Data source to transform.
    :param threshold: Missingness threshold for removal (0-1).
    :param exceptions: Exception columns which should not be checked for missingness.
    :return: New data source with high missingness columns removed.
    """
    log.info("Performing high missingness treatment")
    if ds.data.empty:
        log.warning("Received empty dataframe")
        return ds

    if threshold < 0 or threshold > 1:
        log.error(f"Invalid threshold: {threshold}. Must be in range 0.0 to 1.0")
        return ds

    if not exceptions:
        exceptions = []

    start = time.perf_counter()

    log.info(f"Shape before treatment: {ds.data.shape}")

    drop_list = [k for k, v in ds.data.get_percent_missing().items() if v > threshold]
    drop_list_clean = [x for x in drop_list if x not in exceptions]

    num_excepted = len(drop_list) - len(drop_list_clean)
    num_clean = len(drop_list_clean)

    log.info(
        f"{len(drop_list)} columns have missingness higher than the threshold ({threshold})"
    )
    log.info(f" - {num_excepted} column(s) in exception list")
    log.info(f" - {num_clean} column(s) not in exception list")

    headers = ["Variable Name", "% Missing", "To Remove"]
    table = []
    for col, missingness in sorted(
        ds.data.get_percent_missing().items(),
        key=lambda items: (items[1], items[0]),
        reverse=True,
    ):
        miss_percent = int(missingness * 100)
        to_remove = col in drop_list_clean
        table.append([col, miss_percent, to_remove])
    filelog.info("\n" + tabulate(table, headers=headers, tablefmt="psql"))

    df = ds.data.drop(columns=drop_list_clean)

    log.info(f"Shape after treatment: {df.shape}")
    log.info(f"{len(drop_list)} columns removed with names: {drop_list}")

    end = time.perf_counter()

    log.info(f"End high missing treatment. Time taken: {end-start} seconds")

    return DataSource(df)
