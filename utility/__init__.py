"""Import utility functions and define working variables."""

from .handle_data import read_source as read_source
from .handle_exceptions import exception_handler as exception_handler
from .setup_logging import logging as logging

TIME_INTERVAL_COL = "__time_interval"
OVERVIEW_COL = "__overview"
PREFIX_COL = "__"
PREFIX_COL_E = "e__"
