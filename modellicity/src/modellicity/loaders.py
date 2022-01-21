"""Data source loaders."""

from pandas import DataFrame, read_csv
from pathlib import Path

from modellicity.data_source import DataSource
from modellicity.extended_pandas import ExtendedDataFrame, ExtendedSeries


def load_csv(path: Path, **kwargs) -> DataSource:
    """Load a `DataSource` from a CSV."""
    df: DataFrame = read_csv(path, **kwargs)
    for col in df.columns:
        ExtendedSeries.cast(df[col])
    return DataSource(ExtendedDataFrame.cast(df))


def load_dict(data: dict) -> DataSource:
    """Load a `DataSource` from a dictionary."""
    return DataSource(ExtendedDataFrame(data=data))


def load_dataframe(data: DataFrame) -> DataSource:
    """Load a datasource from a pandas `DataFrame`."""
    return DataSource(ExtendedDataFrame(data))
