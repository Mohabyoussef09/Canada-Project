"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import unittest
import datetime
import numpy as np
import pandas as pd

from modellicity.extended_pandas.extended_actions import entry_actions
from modellicity.settings import settings


class TestEntryActions(unittest.TestCase):
    """Unit tests for EntryActions class."""

    @classmethod
    def setup_class(cls) -> None:
        """:return: None."""
        cls.utils = entry_actions.EntryActions()

    def test_convert_entry_to_numeric_object(self) -> None:
        """
        Tests for convert_entry_to_numeric_object.

        :return: None
        """
        assert type(self.utils.convert_entry_to_numeric_object("50")) \
            in settings.OPTIONS["numeric_types"]
        assert type(self.utils.convert_entry_to_numeric_object("50.53")) \
            in settings.OPTIONS["numeric_types"]

        assert type(self.utils.convert_entry_to_numeric_object("ABC")) \
            not in settings.OPTIONS["numeric_types"]

    def test_convert_entry_to_datetime_object(self) -> None:
        """
        Tests for convert_entry_to_datetime_object.

        :return: None.
        """
        assert isinstance(self.utils.convert_entry_to_datetime_object("1999-01-15"),
                          datetime.datetime)
        assert isinstance(self.utils.convert_entry_to_datetime_object("1999.01.15"),
                          datetime.datetime)
        assert isinstance(self.utils.convert_entry_to_datetime_object("1999/01/15"),
                          datetime.datetime)
        assert isinstance(self.utils.convert_entry_to_datetime_object("01-01-1999"),
                          datetime.datetime)
        assert isinstance(self.utils.convert_entry_to_datetime_object("01.01.1999"),
                          datetime.datetime)
        assert isinstance(self.utils.convert_entry_to_datetime_object("01/01/1999"),
                          datetime.datetime)
        assert isinstance(self.utils.convert_entry_to_datetime_object("August 28 1999"),
                          datetime.datetime)
        assert isinstance(self.utils.convert_entry_to_datetime_object("August 28, 1999"),
                          datetime.datetime)
        assert isinstance(self.utils.convert_entry_to_datetime_object("Aug 28 1999"),
                          datetime.datetime)
        assert isinstance(self.utils.convert_entry_to_datetime_object("Aug 28, 1999"),
                          datetime.datetime)

        assert isinstance(self.utils.convert_entry_to_datetime_object("SHATCH"),
                          str)
        assert isinstance(self.utils.convert_entry_to_datetime_object(np.nan),
                          datetime.datetime)
        assert isinstance(self.utils.convert_entry_to_datetime_object(pd.NaT),
                          datetime.datetime)
        assert isinstance(self.utils.convert_entry_to_datetime_object(None),
                          datetime.datetime)
        assert isinstance(self.utils.convert_entry_to_datetime_object(""),
                          datetime.datetime)
