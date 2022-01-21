"""Convert datetime transformers."""

import logging
import time
from typing import List

from tabulate import tabulate

from modellicity.data_source import DataSource
from modellicity.pipeline import PipelineOperation

log = logging.getLogger(__name__)
filelog = logging.getLogger("file")


class ConvertDatetime(PipelineOperation):
    """Convert datetime-like columns pipeline operation."""

    def __init__(self, extra_formats: List[str] = None):
        """
        Initialize convert datatime pipeline operation.

        :param extra_formats: Additional datetime format strings to check for and convert.
        """
        self.extra_formats = extra_formats

    def run(self, data: DataSource) -> DataSource:
        """
        Convert datetime-like columns to native datetime objects.

        :param data: Data source to transform.
        :return: New datasource with columns converted.
        """
        return convert_datetime(data, self.extra_formats)


def convert_datetime(ds: DataSource, extra_formats: List[str] = None) -> DataSource:
    """
    Convert datetime-like columns to native datetime objects.

    :param ds: Data source to transform.
    :param extra_formats: Additional datetime format strings to check for and convert.
    :return: New data source with columns converted.
    """
    log.info("Performing datetime treatment")
    if ds.data.empty:
        log.warning("Received empty dataframe")
        return ds

    start = time.perf_counter()

    log.info(f"Shape before treatment: {ds.data.shape}")

    default = ds.data.get_all_datetime_format()
    extra: List[str] = []
    if extra_formats:
        extra = ds.data.get_all_datetime_format(formats=extra_formats)

    log.info(f"{len(default) + len(extra)} datetime columns found")
    if extra_formats:
        log.info(f" - {len(default)} in default known formats")
        log.info(f" - {len(extra)} in user provided format")

    df = ds.data.convert_to_datetime_object(default)
    df = df.convert_to_datetime_object(extra, extra_formats)

    headers = ["Column", "Default format", "Manual format"]
    table = []
    for col in default:
        table.append([col, True, False])
    for col in extra:
        table.append([col, False, True])

    filelog.info("\n" + tabulate(table, headers=headers, tablefmt="psql"))

    log.info(f"Shape after treatment: {df.shape}")

    end = time.perf_counter()
    log.info(f"End datetime treatment. Time taken: {end-start} seconds")

    return DataSource(df)
