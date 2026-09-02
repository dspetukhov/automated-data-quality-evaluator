"""Baseline regression tests for handle_exceptions.py — captures CURRENT behavior as-is.

Suspected bugs are noted inline as comments; none are fixed here.
"""

import logging
import sys

import pytest

from utility.handle_exceptions import exception_handler


class TestNoException:
    def test_returns_function_result_unchanged(self):
        @exception_handler()
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_wraps_preserves_function_metadata(self):
        @exception_handler()
        def documented(x):
            """Docstring."""
            return x

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "Docstring."


class TestExceptionDefaultBehavior:
    def test_returns_first_positional_arg_on_exception(self):
        @exception_handler()
        def boom(x, y):
            raise ValueError("bad")

        result = boom("first_arg", "second_arg")
        assert result == "first_arg"

    def test_returns_none_when_no_positional_args(self):
        @exception_handler()
        def boom():
            raise ValueError("bad")

        assert boom() is None

    def test_returns_none_when_only_kwargs_passed(self):
        # BUG-SUSPECT: the fallback `args[0] if args else None` only inspects
        # positional args. If the caller passes everything as a keyword
        # argument, `args` is empty and None is returned instead of, e.g.,
        # the value of that keyword argument.
        @exception_handler()
        def boom(x=None):
            raise ValueError("bad")

        assert boom(x="kwarg_value") is None

    def test_does_not_raise_when_exit_on_error_false(self):
        @exception_handler(exit_on_error=False)
        def boom():
            raise RuntimeError("bad")

        # Should not raise; exception is swallowed.
        boom()


class TestExitOnError:
    def test_exits_with_status_1_on_exception(self):
        @exception_handler(exit_on_error=True)
        def boom():
            raise RuntimeError("bad")

        with pytest.raises(SystemExit) as exc_info:
            boom()
        assert exc_info.value.code == 1

    def test_default_exit_on_error_is_false(self):
        @exception_handler()
        def boom():
            raise RuntimeError("bad")

        # Should return None (not exit) since exit_on_error defaults to False.
        assert boom() is None


class TestBareExceptScope:
    # BUG-SUSPECT: the wrapper catches `Exception` only, not `BaseException`.
    # This means SystemExit (e.g. from an underlying sys.exit() call inside
    # the wrapped function) propagates unhandled and is NOT logged, even
    # though the decorator's job is to catch/log errors from the function.
    def test_systemexit_from_wrapped_function_is_not_caught(self):
        @exception_handler(exit_on_error=False)
        def boom():
            sys.exit(42)

        with pytest.raises(SystemExit) as exc_info:
            boom()
        assert exc_info.value.code == 42

    def test_keyboardinterrupt_from_wrapped_function_is_not_caught(self):
        @exception_handler(exit_on_error=False)
        def boom():
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            boom()


class TestLogMessageFormat:
    def test_logs_error_with_exception_type_and_message(self, caplog):
        @exception_handler()
        def boom():
            raise ValueError("something broke")

        with caplog.at_level(logging.ERROR):
            boom()

        assert len(caplog.records) == 1
        message = caplog.records[0].message
        assert message.startswith("ValueError:")
        assert "something broke" in message
        assert __file__ in message

    def test_message_references_second_traceback_frame_not_deepest(self, caplog):
        # BUG-SUSPECT: `make_message` always reports
        # extract_tb[min(1, len(extract_tb) - 1)], i.e. the *second* frame
        # from the top (the wrapped function's own frame), rather than the
        # deepest frame where the exception actually originated. For calls
        # nested more than one level deep, the reported filename/line
        # therefore points at the call site inside the wrapped function,
        # not at the line that actually raised.
        def helper():
            raise ValueError("boom from helper")

        @exception_handler()
        def outer():
            return helper()

        with caplog.at_level(logging.ERROR):
            outer()

        message = caplog.records[0].message
        # Reports outer()'s call line...
        assert "return helper()" in message
        # ...not helper()'s actual raise line.
        assert 'raise ValueError("boom from helper")' not in message

    def test_message_format_for_direct_raise_shows_raise_line(self, caplog):
        # When the exception is raised directly in the wrapped function
        # (only one frame beyond the wrapper's own try/except frame), the
        # min(1, len - 1) index correctly resolves to that function's frame.
        @exception_handler()
        def boom():
            raise ValueError("direct")

        with caplog.at_level(logging.ERROR):
            boom()

        message = caplog.records[0].message
        assert 'raise ValueError("direct")' in message

    def test_exception_without_str_representation_still_logs(self, caplog):
        class CustomError(Exception):
            def __str__(self):
                return ""

        @exception_handler()
        def boom():
            raise CustomError()

        with caplog.at_level(logging.ERROR):
            boom()

        assert caplog.records[0].message.startswith("CustomError:")


class TestArgsKwargsPassthrough:
    def test_kwargs_passed_through_to_wrapped_function(self):
        @exception_handler()
        def greet(greeting, name=""):
            return f"{greeting}, {name}"

        assert greet("Hello", name="World") == "Hello, World"

    def test_multiple_positional_args_only_first_returned_on_error(self):
        @exception_handler()
        def boom(a, b, c):
            raise ValueError("bad")

        assert boom(1, 2, 3) == 1
