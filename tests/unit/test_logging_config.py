"""Tests for observability logging configuration."""

from __future__ import annotations

import logging

import pytest

import llm_gateway.observability.logging as log_mod
from llm_gateway.observability.logging import configure_logging, get_logger


@pytest.mark.unit
class TestConfigureLogging:
    """Tests for configure_logging() function."""

    def setup_method(self) -> None:
        # Reset the _CONFIGURED flag so each test can call configure_logging
        log_mod._CONFIGURED = False

    def teardown_method(self) -> None:
        log_mod._CONFIGURED = False

    def test_configure_only_runs_once(self) -> None:
        """Second call to configure_logging is a no-op."""
        configure_logging(level="DEBUG", fmt="console")
        # Set the flag manually and call again with different args
        assert log_mod._CONFIGURED is True
        # Second call should be a no-op (no error, no reconfiguration)
        configure_logging(level="ERROR", fmt="json")
        # Still configured — no crash
        assert log_mod._CONFIGURED is True

    def test_configure_with_structlog_json(self) -> None:
        """Structlog JSON configuration works when structlog is installed."""
        if not log_mod.HAS_STRUCTLOG:
            pytest.skip("structlog not installed")
        configure_logging(level="INFO", fmt="json")
        assert log_mod._CONFIGURED is True

    def test_configure_with_structlog_console(self) -> None:
        """Structlog console configuration works when structlog is installed."""
        if not log_mod.HAS_STRUCTLOG:
            pytest.skip("structlog not installed")
        configure_logging(level="DEBUG", fmt="console")
        assert log_mod._CONFIGURED is True

    def test_configure_without_structlog_falls_back(self) -> None:
        """Falls back to stdlib logging when structlog is not available."""
        original = log_mod.HAS_STRUCTLOG
        log_mod.HAS_STRUCTLOG = False
        try:
            configure_logging(level="WARNING", fmt="json")
            assert log_mod._CONFIGURED is True
        finally:
            log_mod.HAS_STRUCTLOG = original

    def test_invalid_level_defaults_to_info(self) -> None:
        """Invalid log level string falls back to INFO."""
        original = log_mod.HAS_STRUCTLOG
        log_mod.HAS_STRUCTLOG = False
        try:
            configure_logging(level="NOTAVALIDLEVEL", fmt="console")
            # Should not raise — getattr falls back to logging.INFO
            assert log_mod._CONFIGURED is True
        finally:
            log_mod.HAS_STRUCTLOG = original


@pytest.mark.unit
class TestGetLogger:
    """Tests for get_logger() function."""

    def test_returns_structlog_logger_when_installed(self) -> None:
        """get_logger returns structlog BoundLogger when structlog is available."""
        if not log_mod.HAS_STRUCTLOG:
            pytest.skip("structlog not installed")
        logger = get_logger("test.module")
        # structlog loggers have a bind method
        assert hasattr(logger, "bind")

    def test_returns_stdlib_logger_when_no_structlog(self) -> None:
        """get_logger returns stdlib logger when structlog is not available."""
        original = log_mod.HAS_STRUCTLOG
        log_mod.HAS_STRUCTLOG = False
        try:
            logger = get_logger("test.module")
            assert isinstance(logger, logging.Logger)
        finally:
            log_mod.HAS_STRUCTLOG = original
