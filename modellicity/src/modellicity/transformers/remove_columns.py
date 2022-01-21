"""Remove columns transformers."""

import logging
import time
from typing import List

from modellicity.data_source import DataSource
from modellicity.pipeline import PipelineOperation

log = logging.getLogger(__name__)
filelog = logging.getLogger("file")


class RemoveColumns(PipelineOperation):
    """Remove columns pipeline operation."""

    def __init__(self, columns: List[str]):
        """
        Initialize remove columns pipeline operation.

        :param columns: List of columns to remove.
        """
        self.columns = columns

    def run(self, data: DataSource) -> DataSource:
        """
        Remove columns from a data source.

        :param data: Data source to transform.
        :return: New data source with columns removed.
        """
        return remove_columns(data, self.columns)


def remove_columns(ds: DataSource, columns: List[str]) -> DataSource:
    """
    Remove columns from a data source.

    :param ds: Data source to transform.
    :param columns: List of columns to remove.
    :return: New data source with columns removed.
    """
    log.info("Performing column removal")
    if ds.data.empty:
        log.warning("Received empty dataframe")
        return ds

    start = time.perf_counter()
    log.info(f"Shape before treatment: {ds.data.shape}")

    found: List[str] = []
    not_found: List[str] = []
    for col in columns:
        if col in ds.data.columns:
            found.append(col)
        else:
            not_found.append(col)

    log.info(f"Columns found: {found}")
    log.info(f"Number of columns found: {len(found)}")

    log.info(f"Columns not found: {not_found}")
    log.info(f"Number of columns not found: {len(not_found)}")

    df = ds.data.drop(columns=found)

    log.info(f"Shape after treatment: {df.shape}")
    log.info(f"Columns removed: {found}")
    log.info(f"Number of columns removed: {len(found)}")

    end = time.perf_counter()
    log.info(f"End column removal. Time taken: {end-start} seconds")

    return DataSource(df)
