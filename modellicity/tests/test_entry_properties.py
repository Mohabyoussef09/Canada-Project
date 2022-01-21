"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import datetime
import unittest
import numpy as np
import pandas as pd

from modellicity.extended_pandas.extended_properties import entry_properties


class TestEntryProperties(unittest.TestCase):
    """Unit tests for EntryProperties class."""

    @classmethod
    def setup_class(cls) -> None:
        """:return: None."""
        cls.utils = entry_properties.EntryProperties()

    def test_is_entry_datetime_format(self) -> None:
        """
        Tests for: is_entry_datetime_format.

        :return: None.
        """
        assert self.utils.is_entry_datetime_format("1999-01-15") is True
        assert self.utils.is_entry_datetime_format("1999.01.15") is True
        assert self.utils.is_entry_datetime_format("1999/01/15") is True
        assert self.utils.is_entry_datetime_format("01-01-1999") is True
        assert self.utils.is_entry_datetime_format("01.01.1999") is True
        assert self.utils.is_entry_datetime_format("01/01/1999") is True
        assert self.utils.is_entry_datetime_format("August 28 1999") is True
        assert self.utils.is_entry_datetime_format("August 28, 1999") is True
        assert self.utils.is_entry_datetime_format("Aug 28 1999") is True
        assert self.utils.is_entry_datetime_format("Aug 28, 1999") is True

        assert self.utils.is_entry_datetime_format("SHATCH") is False
        assert self.utils.is_entry_datetime_format(np.nan) is False
        assert self.utils.is_entry_datetime_format("") is False
        assert self.utils.is_entry_datetime_format(961.0) is False

    def test_is_entry_numeric_format(self) -> None:
        """
        Tests for: is_entry_numeric_format.

        :return: None.
        """
        assert self.utils.is_entry_numeric_format(5) is True
        assert self.utils.is_entry_numeric_format(104.435) is True
        assert self.utils.is_entry_numeric_format(-349) is True
        assert self.utils.is_entry_numeric_format(-243.34) is True
        assert self.utils.is_entry_numeric_format("5") is True
        assert self.utils.is_entry_numeric_format("5.3") is True
        assert self.utils.is_entry_numeric_format("-3.33") is True
        assert self.utils.is_entry_numeric_format("-3039") is True

        assert self.utils.is_entry_numeric_format("SHATCH") is False
        assert self.utils.is_entry_numeric_format("Modellicity") is False

    def test_is_entry_datetime_object(self) -> None:
        """
        Tests for: is_entry_datetime_object.

        :return: None.
        """
        assert self.utils.is_entry_datetime_object(datetime.datetime.now()) is True
        assert self.utils.is_entry_datetime_object(datetime.date(2019, 4, 13)) is True

        assert self.utils.is_entry_datetime_object("SHATCH") is False
        assert self.utils.is_entry_datetime_object("Aug 28 1999") is False
        assert self.utils.is_entry_datetime_object(19990828) is False
        assert self.utils.is_entry_datetime_object("1999-08/28") is False
        assert self.utils.is_entry_datetime_object(961.0) is False

    def test_is_entry_numeric_object(self) -> None:
        """
        Tests for: is_entry_numeric_object.

        :return: None.
        """
        assert self.utils.is_entry_numeric_object(5) is True
        assert self.utils.is_entry_numeric_object(104.435) is True
        assert self.utils.is_entry_numeric_object(-349) is True
        assert self.utils.is_entry_numeric_object(-243.34) is True

        assert self.utils.is_entry_numeric_object("5") is False
        assert self.utils.is_entry_numeric_object("5.3") is False
        assert self.utils.is_entry_numeric_object("-3.33") is False
        assert self.utils.is_entry_numeric_object("-3039") is False
        assert self.utils.is_entry_numeric_object("SHATCH") is False
        assert self.utils.is_entry_numeric_object("Modellicity") is False

    def test_is_entry_missing(self) -> None:
        """
        Tests for: is_entry_missing.

        :return: None.
        """
        assert self.utils.is_entry_missing(np.nan) is True
        assert self.utils.is_entry_missing(pd.NaT) is True
        assert self.utils.is_entry_missing("") is True

        assert self.utils.is_entry_missing(5) is False
