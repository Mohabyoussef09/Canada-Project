"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import os
import time

import pandas as pd

from typing import Callable, Dict, List

from modellicity.settings import settings
from modellicity.extended_pandas.extended_properties import (
    series_properties,
    dataframe_properties,
)
from modellicity.extended_pandas.extended_actions import (
    series_actions,
    dataframe_actions,
)


class Benchmark:
    """Benchmark script for running time trials and optimizing bottlenecks."""

    def __init__(self, test_data: pd.DataFrame) -> None:
        """
        Set up the functions to be benchmarked.

        :param test_data:
        :return: None.
        """
        self._test_data = test_data
        self._benchmark_results: Dict[str, float] = dict()

        self._dataframe_actions = dataframe_actions.DataFrameActions()
        self._dataframe_properties = dataframe_properties.DataFrameProperties()

        self._series_actions = series_actions.SeriesActions()
        self._series_properties = series_properties.SeriesProperties()

        self._series_functions: List[Callable] = [
            self._series_properties.is_all_series_datetime_format,
            # self._series_properties.is_all_series_datetime_object,
            # self._series_properties.is_all_series_numeric_object,
            # self._series_properties.get_series_percent_missing,
            # self._series_properties.get_series_num_missing,
            # self._series_actions.convert_series_to_datetime_object
        ]
        self._dataframe_functions: List[Callable] = [
            # self._dataframe_properties.is_any_dataframe_datetime_object,
            # self._dataframe_properties.get_all_dataframe_datetime_object,
            # self._dataframe_properties.get_all_dataframe_datetime_object
        ]

    @staticmethod
    def start_benchmark(benchmark_function_name: str) -> float:
        """
        Start running the time and benchmark.

        :param benchmark_function_name:
        :return:
        """
        print(f"START: benchmark for {benchmark_function_name.upper()}")
        return time.time()

    def end_benchmark(self, start_time: float, benchmark_function_name: str) -> None:
        """
        Stop running the time and benchmark.

        :param start_time:
        :param benchmark_function_name:
        :return:
        """
        elapsed_time = time.time() - start_time
        self._benchmark_results[benchmark_function_name] = elapsed_time
        print(
            f"END: benchmark for {benchmark_function_name.upper()} at {elapsed_time} seconds."
        )

    def series_benchmark_template(
        self, benchmark_function_name: str, benchmark_function: Callable
    ) -> None:
        """
        Template function for benchmarking all series functions.

        :param benchmark_function_name:
        :param benchmark_function:
        :return: None.
        """
        start_time = self.start_benchmark(benchmark_function_name)
        labels = self._test_data.columns
        for label in labels:
            benchmark_function(self._test_data[label])
        self.end_benchmark(start_time, benchmark_function_name)

    def dataframe_benchmark_template(
        self, benchmark_function_name: str, benchmark_function: Callable
    ) -> None:
        """
        Template function for benchmarking all dataframe functions.

        :param benchmark_function_name:
        :param benchmark_function:
        :return: None.
        """
        start_time = self.start_benchmark(benchmark_function_name)
        benchmark_function(self._test_data)
        self.end_benchmark(start_time, benchmark_function_name)

    def run_series_benchmark(self) -> None:
        """
        Run the series benchmarks.

        :return: None.
        """
        for function in self._series_functions:
            self.series_benchmark_template(function.__name__, function)

    def run_dataframe_benchmark(self) -> None:
        """
        Run the dataframe benchmarks.

        :return: None.
        """
        for function in self._dataframe_functions:
            self.dataframe_benchmark_template(function.__name__, function)

    def get_benchmark_results(self) -> Dict[str, float]:
        """
        Obtain the results from running the benchmarks.

        :return:
        """
        return self._benchmark_results

    def run_benchmarks(self) -> None:
        """
        Run all benchmarks.

        :return: None.
        """
        print("***SERIES BENCHMARKS START***")
        series_start_time = time.time()
        self.run_series_benchmark()
        series_end_time = time.time() - series_start_time
        print("***SERIES BENCHMARKS END***")
        print(f"TOTAL SERIES ELAPSED: {series_end_time}")

        print("***DATAFRAME BENCHMARKS START***")
        dataframe_start_time = time.time()
        self.run_dataframe_benchmark()
        dataframe_end_time = time.time() - dataframe_start_time
        print("***DATAFRAME BENCHMARKS END***")
        print(f"TOTAL DATAFRAME ELAPSED: {dataframe_end_time}")


def main() -> None:
    """
    Run benchmarks.

    :return: None
    """
    data_path = os.path.join(settings.PATHS["data"], "Data Request2.txt")
    test_data = pd.read_csv(data_path, sep="|", encoding="ISO-8859-1")
    benchmark = Benchmark(test_data)

    benchmark.run_benchmarks()


if __name__ == "__main__":
    main()
