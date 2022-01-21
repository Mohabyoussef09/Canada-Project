"""Modellicity pipeline."""

import logging
from abc import ABC, abstractmethod
from typing import List

from modellicity.data_source import DataSource

log = logging.getLogger(__name__)
filelog = logging.getLogger("file")


class PipelineOperation(ABC):
    """A single pipeline operation to be applied on a data source."""

    @abstractmethod
    def run(self, data: DataSource) -> DataSource:
        """Run the pipeline operation on the data source."""
        pass


class Pipeline(object):
    """A pipeline of operations to be run on a data source."""

    def __init__(self, operations: List[PipelineOperation] = None):
        """
        Initialize a pipeline.

        :param operations: A list of operations for this pipeline.
        """
        self.operations = operations if operations else []

    def add(self, operation: PipelineOperation) -> None:
        """Add an operation to this pipeline."""
        self.operations.append(operation)

    def run(self, data: DataSource) -> DataSource:
        """
        Run this pipeline.

        :param data: Target data source for this pipeline
        :return: Pipeline data source result
        """
        for operation in self.operations:
            data = operation.run(data)
        return data
