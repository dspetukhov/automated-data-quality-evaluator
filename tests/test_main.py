"""Baseline regression tests for main.py — captures CURRENT behavior as-is."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_config_file(tmp_path):
    """Minimal valid config with a truthy source dict."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps({"source": {"file_path": "data.csv"}, "date_column": "date"}),
        encoding="utf-8",
    )
    return cfg


@pytest.fixture
def no_source_config_file(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"date_column": "date"}), encoding="utf-8")
    return cfg


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_nonexistent_path_raises_system_exit(self, tmp_path):
        from main import main

        with pytest.raises(SystemExit) as exc:
            main(tmp_path / "nonexistent.json")
        assert "configuration file wasn't found" in str(exc.value)

    def test_directory_path_raises_system_exit(self, tmp_path):
        # tmp_path is a directory, not a file
        from main import main

        with pytest.raises(SystemExit) as exc:
            main(tmp_path)
        assert "configuration file wasn't found" in str(exc.value)

    def test_malformed_json_is_caught_and_returns_path(self, tmp_path):
        """json.JSONDecodeError is an Exception — handler catches it and returns args[0]."""
        from main import main

        bad = tmp_path / "bad.json"
        bad.write_text("{ not valid json }", encoding="utf-8")
        result = main(bad)
        assert result == bad


# ---------------------------------------------------------------------------
# Source key validation
# ---------------------------------------------------------------------------


class TestSourceValidation:
    def test_missing_source_key_raises_system_exit(self, no_source_config_file):
        from main import main

        with pytest.raises(SystemExit) as exc:
            main(no_source_config_file)
        assert "missing required 'source' section" in str(exc.value)

    def test_null_source_raises_system_exit(self, tmp_path):
        # "source": null → config.get("source") is None → SystemExit
        from main import main

        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"source": None}), encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            main(cfg)
        assert "missing required 'source' section" in str(exc.value)

    @pytest.mark.parametrize(
        "source_value",
        ["", {}, 0],
        ids=["empty_string", "empty_dict", "zero"],
    )
    def test_non_none_falsy_source_reaches_read_source(self, tmp_path, source_value):
        # These values are not None so they pass the guard and reach read_source.
        # Previously they raised SystemExit with a misleading message.
        from main import main

        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"source": source_value}), encoding="utf-8")
        with patch("main.read_source", side_effect=RuntimeError("reached")) as mock_rs:
            result = main(cfg)
        mock_rs.assert_called_once_with(source_value)
        assert result == cfg  # exception_handler swallows and returns args[0]


# ---------------------------------------------------------------------------
# Happy path — pipeline called in correct order with correct arguments
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_pipeline_functions_called_in_order(self, valid_config_file):
        from main import main

        mock_lf = MagicMock(name="lazy_frame")
        mock_df = MagicMock(name="df")
        mock_meta = MagicMock(name="metadata")

        with (
            patch("main.read_source", return_value=mock_lf) as mock_rs,
            patch("main.make_preprocessing", return_value=(mock_df, mock_meta)) as mock_pp,
            patch("main.make_report") as mock_mr,
        ):
            result = main(valid_config_file)

        expected_config = {"source": {"file_path": "data.csv"}, "date_column": "date"}
        mock_rs.assert_called_once_with(expected_config["source"])
        mock_pp.assert_called_once_with(mock_lf, expected_config)
        mock_mr.assert_called_once_with(mock_df, mock_meta, expected_config)
        assert result is None

    def test_returns_none_on_success(self, valid_config_file):
        from main import main

        with (
            patch("main.read_source", return_value=MagicMock()),
            patch("main.make_preprocessing", return_value=(MagicMock(), MagicMock())),
            patch("main.make_report"),
        ):
            assert main(valid_config_file) is None


# ---------------------------------------------------------------------------
# Exception handler behavior — Exception subclasses are swallowed; returns args[0]
# ---------------------------------------------------------------------------


class TestExceptionHandlerBehavior:
    def test_read_source_exception_returns_path(self, valid_config_file):
        from main import main

        with patch("main.read_source", side_effect=RuntimeError("source failed")):
            result = main(valid_config_file)
        assert result == valid_config_file

    def test_make_preprocessing_exception_returns_path(self, valid_config_file):
        from main import main

        with (
            patch("main.read_source", return_value=MagicMock()),
            patch("main.make_preprocessing", side_effect=ValueError("bad data")),
        ):
            result = main(valid_config_file)
        assert result == valid_config_file

    def test_make_report_exception_returns_path(self, valid_config_file):
        from main import main

        with (
            patch("main.read_source", return_value=MagicMock()),
            patch("main.make_preprocessing", return_value=(MagicMock(), MagicMock())),
            patch("main.make_report", side_effect=IOError("write failed")),
        ):
            result = main(valid_config_file)
        assert result == valid_config_file

    def test_system_exit_inside_main_is_not_swallowed(self, tmp_path):
        """SystemExit inherits BaseException, not Exception — it escapes the handler."""
        from main import main

        # missing file triggers SystemExit — must propagate, not be swallowed
        with pytest.raises(SystemExit):
            main(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# __main__ CLI entry point — tested via subprocess
# ---------------------------------------------------------------------------


class TestCLIEntryPoint:
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "main.py"), *args],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

    def test_no_args_exits_nonzero_with_usage(self):
        proc = self._run()
        assert proc.returncode != 0
        combined = proc.stdout + proc.stderr
        assert "Usage" in combined

    def test_too_many_args_exits_nonzero_with_usage(self):
        proc = self._run("a", "b")
        assert proc.returncode != 0
        combined = proc.stdout + proc.stderr
        assert "Usage" in combined

    def test_nonexistent_config_exits_nonzero_with_message(self, tmp_path):
        proc = self._run(str(tmp_path / "missing.json"))
        assert proc.returncode != 0
        combined = proc.stdout + proc.stderr
        assert "configuration file wasn't found" in combined
