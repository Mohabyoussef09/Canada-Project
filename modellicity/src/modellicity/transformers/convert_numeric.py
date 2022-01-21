"""Convert numeric transformers."""

import logging
import time
from typing import Any, List

from tabulate import tabulate

from modellicity.data_source import DataSource
from modellicity.pipeline import PipelineOperation

log = logging.getLogger(__name__)
filelog = logging.getLogger("file")


class ConvertNumeric(PipelineOperation):
    """Convert numeric-like columns pipeline operation."""

    def __init__(self, extra_formats: List[Any] = None):
        """
        Initialize convert numeric pipeline operation.

        :param extra_formats: Additional numeric format strings to check for and convert.
        """
        self.extra_formats = extra_formats

    def run(self, data: DataSource) -> DataSource:
        """
        Convert numeric-like columns to native numeric objects.

        :param data: Data source to transform.
        :return: New datasource with columns converted.
        """
        return convert_numeric(data, self.extra_formats)


def convert_numeric(ds: DataSource, extra_formats: List[Any] = None) -> DataSource:
    """
    Convert numeric-like columns to native numeric objects.

    :param ds: Data source to transform.
    :param extra_formats: Additional numeric format strings to check for and convert.
    :return: New data source with columns converted.
    """
    log.info("Performing numeric treatment")
    if ds.data.empty:
        log.warning("Received empty dataframe")
        return ds

    start = time.perf_counter()

    log.info(f"Shape before treatment: {ds.data.shape}")

    all_numeric_format = ds.data.get_all_numeric_format()
    all_numeric_object = ds.data.get_all_numeric_object()

    # We only care about converting the elements of the dataframe that are not already numeric
    # but are identified as potentially numeric.
    default = list(set(all_numeric_format) - set(all_numeric_object))
    extra: List[str] = []
    if extra_formats:
        extra = ds.data.get_all_numeric_format(formats=extra_formats)

    log.info(f"{len(default) + len(extra)} numeric columns found")
    if extra_formats:
        log.info(f" - {len(default)} in default known formats")
        log.info(f" - {len(extra)} in user provided format")

    df = ds.data.convert_to_numeric_object(default)
    df = df.convert_to_numeric_object(extra)

    headers = ["Column", "Default format", "Manual format"]
    table = []
    for col in default:
        table.append([col, True, False])
    for col in extra:
        table.append([col, False, True])

    filelog.info("\n" + tabulate(table, headers=headers, tablefmt="psql"))

    log.info(f"Shape after treatment: {df.shape}")

    end = time.perf_counter()
    log.info(f"End numeric treatment. Time taken: {end-start} seconds")

    return DataSource(df)
