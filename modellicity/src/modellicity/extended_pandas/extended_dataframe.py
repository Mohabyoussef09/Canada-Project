"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
from modellicity.src.modellicity.extended_pandas.extended_properties.dataframe_properties import (
    DataFrameProperties,
)
from modellicity.src.modellicity.extended_pandas.extended_actions.dataframe_actions import (
    DataFrameActions,
)
from modellicity.src.modellicity.settings import settings
from typing import Dict, List

import logging
import pandas as pd

log = logging.getLogger(__name__)


class ExtendedDataFrame(pd.DataFrame):
    """Extended Pandas DataFrame class."""

    def __init__(self, *args, **kwargs):
        """Construct ExtendedDataFrame class."""
        super(ExtendedDataFrame, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        """:return: ExtendedDataFrame object."""
        return ExtendedDataFrame

    @staticmethod
    def cast(df: pd.DataFrame):
        """Cast a pandas DataFrame to an ExtendedDataFrame without copying data."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Only a DataFrame can be cast to an ExtendedDataFrame")

        df.__class__ = ExtendedDataFrame
        return df

    """
    DataFrame Properties
    """

    def get_all_categorical(self) -> List[str]:
        """Provide ExtendedDataFrame wrapper for get_all_dataframe_categorical."""
        return DataFrameProperties.get_all_dataframe_categorical(self)

    def get_all_numeric_object(self) -> List[str]:
        """Provide ExtendedDataFrame wrapper for get_all_dataframe_numeric_object."""
        return DataFrameProperties.get_all_dataframe_numeric_object(self)

    def is_any_datetime_object(self) -> bool:
        """Provide ExtendedDataFrame wrapper for is_any_dataframe_datetime_object."""
        return DataFrameProperties.is_any_dataframe_datetime_object(self)

    def get_all_datetime_format(
        self, formats=settings.OPTIONS["date_formats"]
    ) -> List[str]:
        """Provide ExtendedDataFrame wrapper for get_all_dataframe_datetime_format."""
        return DataFrameProperties.get_all_dataframe_datetime_format(self, formats)

    def get_all_numeric_format(
        self, formats=settings.OPTIONS["numeric_types"]
    ) -> List[str]:
        """Provide ExtendedDataFrame wrapper for get_all_dataframe_numeric_format."""
        return DataFrameProperties.get_all_dataframe_numeric_format(self, formats)

    def get_all_datetime_object(self) -> List[str]:
        """Provide ExtendedDataFrame wrapper for get_all_dataframe_datetime_object."""
        return DataFrameProperties.get_all_dataframe_datetime_object(self)

    def get_all_missing(self) -> List[str]:
        """Provide ExtendedDataFrame wrapper for get_all_dataframe_missing."""
        return DataFrameProperties.get_all_dataframe_missing(self)

    def get_all_no_missing(self) -> List[str]:
        """Provide ExtendedDataFrame wrapper for get_all_dataframe_no_missing."""
        return DataFrameProperties.get_all_dataframe_no_missing(self)

    def get_num_missing(self) -> Dict[str, int]:
        """Provide ExtendedDataFrame wrapper for get_dataframe_num_missing."""
        return DataFrameProperties.get_dataframe_num_missing(self)

    def get_percent_missing(self) -> Dict[str, float]:
        """Provide ExtendedDataFrame wrapper for get_dataframe_percent_missing."""
        return DataFrameProperties.get_dataframe_percent_missing(self)

    def get_n_value_variables(self, num_unique: int) -> List[str]:
        """Provide ExtendedDataFrame wrapper for get_dataframe_n_value_variables."""
        return DataFrameProperties.get_dataframe_n_value_variables(self, num_unique)

    def get_high_concentration_variables(
        self, concentration_threshold: float
    ) -> Dict[str, float]:
        """Provide ExtendedDataFrame wrapper for get_dataframe_high_concentration_variables."""
        return DataFrameProperties.get_dataframe_high_concentration_variables(
            self, concentration_threshold
        )

    def get_outliers(self, outlier_threshold: float) -> List[str]:
        """Provide ExtendedDataFrame wrapper for get_dataframe_outliers."""
        return DataFrameProperties.get_dataframe_outliers(self, outlier_threshold)

    """
    DataFrame Actions:
    """

    def convert_to_numeric_object(self, cols: List[str] = None) -> pd.DataFrame:
        """Provide ExtendedDataFrame wrapper for convert_dataframe_to_numeric_object."""
        if cols is None:
            cols = self.columns
        return DataFrameActions.convert_dataframe_to_numeric_object(self, cols)

    def convert_to_datetime_object(
        self,
        cols: List[str] = None,
        date_formats: List[str] = settings.OPTIONS["date_formats"],
    ) -> pd.DataFrame:
        """Provide ExtendedDataFrame wrapper for convert_dataframe_to_datetime_object."""
        if cols is None:
            cols = self.columns
        return DataFrameActions.convert_dataframe_to_datetime_object(
            self, cols, date_formats
        )

    def remove_empty_cols(self) -> pd.DataFrame:
        """Provide ExtendedDataFrame wrapper for remove_empty_cols."""
        return DataFrameActions.remove_empty_cols(self)

    def remove_n_unique_value_variables(self, num_unique: int) -> pd.DataFrame:
        """Provide ExtendedDataFrame wrapper for remove_n_unique_value_variables."""
        return DataFrameActions.remove_n_unique_value_variables(self, num_unique)

    def remove_high_concentration_variables(
        self, concentration_threshold: float
    ) -> pd.DataFrame:
        """Provide ExtendedDataFrame wrapper for remove_high_concentration_variables."""
        return DataFrameActions.remove_high_concentration_variables(
            self, concentration_threshold
        )

    def floor_and_cap(
        self,
        lower_percentile_threshold: float = 1,
        upper_percentile_threshold: float = 99,
    ) -> pd.DataFrame:
        """Provide ExtendedDataFrame wrapper for floor_and_cap_dataframe."""
        return DataFrameActions.floor_and_cap_dataframe(
            self, lower_percentile_threshold, upper_percentile_threshold,
        )
