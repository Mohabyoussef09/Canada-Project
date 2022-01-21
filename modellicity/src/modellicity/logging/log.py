"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""

import logging.config

from modellicity.src.modellicity.settings import settings


def init_logging():
    """
    Initialize Modellicity logger as specified by the logging settings in settings.py.

    :return: Project logging object.
    """
    return logging.config.dictConfig(settings.LOGGING_CONFIG)


def log_function_call(logger):
    """
    Return decorator function to be used throughout modellicity project.

    :param logger:
    :return:
    """
    return logged(logger)


def logged(logger):
    """
    Wrap decorator in try/except.

    :param logger:
    :return:
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as err:
                logger.exception(err)
            raise

        return wrapper

    return decorator


model_logger = init_logging()
