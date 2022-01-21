"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import datetime
from typing import Any, List

from modellicity.src.modellicity.settings import settings


class EntryProperties(object):
    """General properties of entries in Pandas series object."""

    @staticmethod
    def is_entry_datetime_format(
        entry: Any, formats: List[str] = settings.OPTIONS["date_formats"]
    ) -> bool:
        """
        Determine if entry is formatted as datetime.

        :param entry: The entry to be processed.
        :param formats:
        :return: True if entry is of datetime-format type and False otherwise.
        """
        for fmt in formats:
            try:
                datetime.datetime.strptime(str(entry), fmt)
                return True
            except ValueError:
                pass
        return False

    @staticmethod
    def is_entry_numeric_format(
        entry: Any, formats: List[str] = settings.OPTIONS["numeric_types"]
    ) -> bool:
        """
        Determine if entry is formatted as numeric.

        :param entry: The entry to be processed.
        :param formats: List of accepted numeric formats.
        :return: True if entry is of numeric-format type and False otherwise.
        """
        if EntryProperties.is_entry_numeric_object(entry) and type(entry) in formats:
            return True
        try:
            float(entry)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_entry_datetime_object(entry: Any) -> bool:
        """
        Determine if entry is a datetime object type.

        :param entry: The entry to be processed.
        :return: True if entry is of datetime object-type and False otherwise.
        """
        return isinstance(entry, tuple(settings.OPTIONS["date_types"]))

    @staticmethod
    def is_entry_numeric_object(entry: Any) -> bool:
        """
        Determine if entry is a numeric object type.

        :param entry: The entry to be processed.
        :return: True if entry is of numeric-object type and False otherwise.
        """
        return isinstance(
            entry, tuple(settings.OPTIONS["numeric_types"])
        ) and not isinstance(entry, bool)

    @staticmethod
    def is_entry_missing(entry: Any) -> bool:
        """
        Determine if entry is a missing object type.

        :param entry: The entry to be processed.
        :return: True if entry is of type missing and False otherwise.
        """
        return entry in settings.OPTIONS["missing_types"]
