import sys
import traceback
from collections.abc import Callable
from functools import wraps

from .setup_logging import logging


def exception_handler(exit_on_error: bool = False):
    """Decorator to handle exceptions in the decorated function.

    This decorator wraps a function to catch and log exceptions
    using a customized error message through sys and traceback modules.
    If specified, it can terminate the program in case of error.

    Args:
        exit_on_error (bool): If True, the program will log the exception
            and exit with status 1. If False, the function returns
            the first argument or None if no arguments.

    Returns:
        Callable: Wrapped function with exception handling.

    Raises:
        SystemExit: If exit_on_error is True and an exception occurs.

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
