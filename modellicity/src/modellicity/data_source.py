"""Datasource."""

from modellicity.src.modellicity.extended_pandas import ExtendedDataFrame


class DataSource(object):
    """DataSource."""

    def __init__(self, data: ExtendedDataFrame = ExtendedDataFrame()):
        """Initialize a datasource.

        :param data: Initial data
        """
        self.data = data
