"""Remove one-value variable transformers."""

import logging
import time

from modellicity.data_source import DataSource
from modellicity.pipeline import PipelineOperation

log = logging.getLogger(__name__)
filelog = logging.getLogger("file")


class RemoveOneValueVariables(PipelineOperation):
    """Remove one-value variables pipeline operation."""

    def run(self, data: DataSource) -> DataSource:
        """
        Remove one-value variables from a data source.

        :param data: Data source to transform.
        :return: New data source with one-value variables removed.
        """
        return remove_one_value_variables(data)


def remove_one_value_variables(ds: DataSource) -> DataSource:
    """
    Remove one-value variables from a data source.

    :param ds: Data source to transform.
    :return: New data source with columns removed.
    """
    log.info("Performing one-value variable removal.")
    if ds.data.empty:
        log.warning("Received empty dataframe.")
        return ds

    start = time.perf_counter()
    log.info(f"Shape before treatment: {ds.data.shape}")

    one_value_vars = ds.data.get_n_value_variables(num_unique=1)

    log.info(f"One-value variables found: {one_value_vars}")
    log.info(f"Number of one-value variables found: {len(one_value_vars)}")

    df = ds.data.drop(columns=one_value_vars)

    log.info(f"Shape after treatment: {df.shape}")
    log.info(f"Number of columns removed: {len(one_value_vars)}")

    end = time.perf_counter()
    log.info(f"End column removal. Time taken: {end-start} seconds")

    return DataSource(df)
