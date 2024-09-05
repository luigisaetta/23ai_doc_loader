"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-27
Python Version: 3.11
"""

import logging
import os


def debug_bool(b_str):
    """
    to convert the env value to bool
    """
    if b_str == "True":
        return True
    return False


def remove_path_from_ref(ref_pathname):
    """
    remove the path from source (ref)
    """
    ref = ref_pathname
    # check if / or \ is contained
    if len(ref_pathname.split(os.sep)) > 0:
        ref = ref_pathname.split(os.sep)[-1]

    return ref


def get_console_logger():
    """
    To get a logger to print on console
    """
    logger = logging.getLogger("ConsoleLogger")

    # to avoid duplication of logging
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False

    return logger


def check_value_in_list(value, values_list):
    """
    to check that we don't enter a not supported value
    """
    if value not in values_list:
        raise ValueError(
            f"Value {value} is not valid: value must be in list {values_list}"
        )
