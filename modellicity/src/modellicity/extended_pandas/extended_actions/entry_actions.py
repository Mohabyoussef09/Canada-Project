"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import pandas as pd

from dateutil import parser
from modellicity.src.modellicity.extended_pandas.extended_properties.entry_properties import (
    EntryProperties,
)
from modellicity.src.modellicity.settings import settings
from typing import Any, List


class EntryActions(object):
    """General manipulations of entries in Pandas series object."""

    @staticmethod
    def convert_entry_to_datetime_object(
        entry: Any, date_formats: List[str] = settings.OPTIONS["date_formats"]
    ) -> Any:
        """
        Attempt to convert entry to an object of type datetime.

        If the entry cannot be converted to a datetime object, return the original entry
        provided as input.

        :param entry: The entry to be processed.
        :param date_formats: A list of accepted date formats.
        :return: If entry can be converted to datetime, the converted entry. Otherwise, returns the
                 original entry.
        """
        if entry in settings.OPTIONS["missing_types"] or pd.isnull(entry):
            return pd.NaT

        if not EntryProperties.is_entry_datetime_format(
            entry, date_formats
        ) or EntryProperties.is_entry_datetime_object(entry):
            return entry

        return parser.parse(str(entry))

    @staticmethod
    def convert_entry_to_numeric_object(entry: Any) -> Any:
        """
        Attempt to convert entry to an object of numeric type.

        If the entry cannot be converted to a numeric object, return the original entry
        provided as input.

        :param entry: The entry to be processed.
        :return: If entry can be converted to numeric, the converted entry. Otherwise, returns the
                 original entry.
        """
        if isinstance(entry, str):
            if str.isdigit(entry):
                return int(entry)

        try:
            return float(entry)
        except ValueError:
            return entry
