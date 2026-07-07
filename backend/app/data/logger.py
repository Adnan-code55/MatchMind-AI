"""
Reusable logging utility for MatchMind AI data pipeline.

This module provides a centralized logging utility that formats logs with
timestamp, module name, log level, and message. It ensures consistent
logging across all pipeline modules.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class PipelineFormatter(logging.Formatter):
    """
    Custom formatter for pipeline logs.

    Formats logs with timestamp, module name, level, and message in a
    standardized, human-readable format.
    """

    FORMAT_STRING = (
        "%(asctime)s - %(execution_id)s - %(name)s - %(levelname)s - %(message)s - %(elapsed_time)s"
    )
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        """Initialize the formatter with standard format."""
        super().__init__(fmt=self.FORMAT_STRING, datefmt=self.DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with timestamp, module, level, and message.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: Formatted log message.
        """
        return super().format(record)


class PipelineContextFilter(logging.Filter):
    """Attach pipeline execution context to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add execution-id and elapsed-time fields to each log record."""
        record.execution_id = PipelineLogger._execution_id or "N/A"

        if PipelineLogger._execution_start_time:
            elapsed = datetime.now(timezone.utc) - PipelineLogger._execution_start_time
            record.elapsed_time = f"{elapsed.total_seconds():.3f}s"
        else:
            record.elapsed_time = "0.000s"

        return True


class PipelineLogger:
    """
    Reusable logging utility for the MatchMind AI pipeline.

    Provides a consistent logging interface with both console and optional
    file output. Creates module-specific loggers with standardized formatting.
    """

    _loggers: dict[str, logging.Logger] = {}
    _initialized: bool = False
    _log_file: Optional[Path] = None
    _execution_id: Optional[str] = None
    _execution_start_time: Optional[datetime] = None

    @classmethod
    def initialize(
        cls,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        execution_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the pipeline logger.

        Sets up console output and optional file output for all pipeline loggers.
        Should be called once at application startup.

        Args:
            log_file (Optional[str]): Path to log file. If None, only console
                output is used. Defaults to None.
            level (int): Logging level. Defaults to logging.INFO.
            execution_id (Optional[str]): Unique execution identifier.

        Raises:
            IOError: If log file cannot be created or written to.
        """
        if cls._initialized:
            if execution_id and execution_id == cls._execution_id:
                return
            cls.reset()

        cls._log_file = Path(log_file) if log_file else None
        cls._execution_id = execution_id
        cls._execution_start_time = datetime.now(timezone.utc)

        if cls._log_file:
            cls._log_file.parent.mkdir(parents=True, exist_ok=True)

        cls._initialized = True

    @classmethod
    def get_logger(cls, module_name: str) -> logging.Logger:
        """
        Get or create a logger for a specific module.

        Creates a new logger for the module if it doesn't exist, or returns
        the existing logger. Each logger has a console handler and optional
        file handler.

        Args:
            module_name (str): Name of the module requesting the logger.

        Returns:
            logging.Logger: Configured logger instance for the module.
        """
        if not cls._initialized:
            cls.initialize()

        if module_name in cls._loggers:
            return cls._loggers[module_name]

        logger = logging.getLogger(module_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if logger.handlers:
            return logger

        formatter = PipelineFormatter()
        context_filter = PipelineContextFilter()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(context_filter)
        logger.addHandler(console_handler)

        if cls._log_file:
            try:
                file_handler = logging.FileHandler(
                    cls._log_file, mode="a", encoding="utf-8"
                )
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                file_handler.addFilter(context_filter)
                logger.addHandler(file_handler)
            except (IOError, OSError) as e:
                logger.warning(
                    f"Could not create file handler for {cls._log_file}: {e}"
                )

        cls._loggers[module_name] = logger
        return logger

    @classmethod
    def log_info(cls, module_name: str, message: str) -> None:
        """
        Log an info-level message.

        Args:
            module_name (str): Name of the module logging the message.
            message (str): The message to log.
        """
        logger = cls.get_logger(module_name)
        logger.info(message)

    @classmethod
    def log_warning(cls, module_name: str, message: str) -> None:
        """
        Log a warning-level message.

        Args:
            module_name (str): Name of the module logging the message.
            message (str): The message to log.
        """
        logger = cls.get_logger(module_name)
        logger.warning(message)

    @classmethod
    def log_error(cls, module_name: str, message: str) -> None:
        """
        Log an error-level message.

        Args:
            module_name (str): Name of the module logging the message.
            message (str): The message to log.
        """
        logger = cls.get_logger(module_name)
        logger.error(message)

    @classmethod
    def log_debug(cls, module_name: str, message: str) -> None:
        """
        Log a debug-level message.

        Args:
            module_name (str): Name of the module logging the message.
            message (str): The message to log.
        """
        logger = cls.get_logger(module_name)
        logger.debug(message)

    @classmethod
    def log_critical(cls, module_name: str, message: str) -> None:
        """
        Log a critical-level message.

        Args:
            module_name (str): Name of the module logging the message.
            message (str): The message to log.
        """
        logger = cls.get_logger(module_name)
        logger.critical(message)

    @classmethod
    def reset(cls) -> None:
        """
        Reset all loggers.

        Clears all registered loggers and resets initialization state.
        Useful for testing purposes.
        """
        for logger in cls._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

        cls._loggers.clear()
        cls._initialized = False
        cls._log_file = None
