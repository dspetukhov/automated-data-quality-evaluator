"""Decorator that catches, logs, and optionally exits on exceptions raised by a wrapped function."""

import sys
import traceback
from collections.abc import Callable
from functools import wraps

from .setup_logging import logging


def exception_handler(exit_on_error: bool = False):
    """Build a decorator that catches and logs exceptions from the wrapped function.

    On exception, the decorator logs an error message identifying the
    exception type, the source location of the wrapped function's own
    frame, and the exception's string representation. `SystemExit` and
    `KeyboardInterrupt` are not caught, since only `Exception` is handled.

    Args:
        exit_on_error (bool): If True, exit the process with status 1
            after logging the exception. If False, swallow the exception
            and return the wrapped call's first positional argument, or
            None if it was called with no positional arguments.

    Returns:
        Callable: A decorator that wraps a function with exception handling.

    Example:
        @exception_handler(exit_on_error=True)
        def my_function(...):
            ...
    """

    def decorator(func: Callable) -> Callable:
        def make_message(exc_info: tuple) -> str:
            exc_type, exc_obj, tb_obj = exc_info
            extract_tb = traceback.extract_tb(tb_obj)
            if extract_tb:
                summary = extract_tb[min(1, len(extract_tb) - 1)]
                return f"{exc_type.__name__}: {summary.filename}#{summary.lineno}: {summary.line}: {str(exc_obj)}"
            else:
                return f"{exc_type.__name__}: {str(exc_obj)}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                logging.error(make_message(sys.exc_info()))
                if exit_on_error:
                    sys.exit(1)
                return args[0] if args else None

        return wrapper

    return decorator
