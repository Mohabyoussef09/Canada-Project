"""
@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved.

Global settings file for entire project.
"""

import datetime
import os
import numpy as np
import pandas as pd

"""
Project logging settings.
"""
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
        },
        "file": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "modellicity.log"
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": False
        },
        "my.packg": {
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": False
        },
        "__main__": {  # if __name__ == "__main__"
            "handlers": ["default", "file"],
            "level": 'DEBUG',
            "propagate": False
        },
    }
}

"""
Optional parameters used throughout project.
"""
OPTIONS = dict()
OPTIONS["missing_types"] = [np.nan, pd.NaT, None, ""]
OPTIONS["numeric_types"] = [int, np.int64, float, np.float64]
OPTIONS["categorical_types"] = [object]
OPTIONS["date_types"] = [datetime.datetime, datetime.date, pd.Timestamp]
OPTIONS["date_formats"] = [
    "%Y-%m-%d",
    "%Y.%m.%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%B %d %Y",
    "%B %d, %Y",
    "%b %d %Y",
    "%b %d, %Y",
    "%Y%m%d",
]

"""
Path parameters used throughout project.
"""
PATHS = dict()
PATHS["root"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
PATHS["data"] = os.path.join(PATHS["root"], "data")
PATHS["logs"] = os.path.join(PATHS["root"], "logs")
PATHS["reports"] = os.path.join(PATHS["root"], "reports")
PATHS["settings"] = os.path.join(PATHS["root"], "settings")

REPORT_OPTIONS = dict()
REPORT_OPTIONS["enabled_reports"] = {
    "generate_csv": True,
    "generate_plots": True,
    "email_alert": True,
}
