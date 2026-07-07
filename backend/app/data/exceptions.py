"""
Custom exception classes for MatchMind AI data pipeline.

This module defines custom exceptions used throughout the data processing
pipeline to provide clear, actionable error messages for various failure scenarios.
"""


class MatchMindAIException(Exception):
    """
    Base exception class for all MatchMind AI pipeline errors.

    This is the parent exception for all custom exceptions in the pipeline,
    allowing users to catch all pipeline-specific errors with a single except clause.
    """

    pass


class DataValidationError(MatchMindAIException):
    """
    Raised when data fails validation checks.

    This exception is raised when a dataset does not meet the required validation
    criteria, such as missing required columns, invalid data types, or corrupted records.
    """

    pass


class MissingColumnError(DataValidationError):
    """
    Raised when required columns are missing from dataset.

    This exception is raised when one or more required columns defined in the
    schema are not found in the dataset.
    """

    pass


class DuplicateRowError(DataValidationError):
    """
    Raised when duplicate rows are detected in dataset.

    This exception is raised when the dataset contains duplicate records that
    should not exist according to business rules.
    """

    pass


class NullValueError(DataValidationError):
    """
    Raised when null/missing values are found in dataset.

    This exception is raised when null values are found in columns where they
    are not permitted or would compromise data quality.
    """

    pass


class InvalidDateError(DataValidationError):
    """
    Raised when date columns contain invalid date values.

    This exception is raised when date fields cannot be parsed or contain
    dates outside expected ranges.
    """

    pass


class InvalidScoreError(DataValidationError):
    """
    Raised when match score columns contain invalid values.

    This exception is raised when score fields (goals, shots, etc.) contain
    negative values or non-numeric data.
    """

    pass


class InvalidTeamNameError(DataValidationError):
    """
    Raised when team name columns contain invalid values.

    This exception is raised when team name fields contain empty strings,
    excessive whitespace, or invalid characters.
    """

    pass


class DatasetNotFoundError(MatchMindAIException):
    """
    Raised when required dataset files cannot be found.

    This exception is raised when expected CSV files do not exist in the
    specified data directory or path.
    """

    pass


class InvalidDatasetError(MatchMindAIException):
    """
    Raised when dataset format or structure is invalid.

    This exception is raised when a CSV file cannot be read, has invalid
    encoding, or is corrupted.
    """

    pass


class SchemaMismatchError(DataValidationError):
    """
    Raised when dataset schema does not match expected schema.

    This exception is raised when the columns, data types, or structure
    of the dataset do not conform to the defined schema.
    """

    pass


class DataProcessingError(MatchMindAIException):
    """
    Raised when data processing operations fail.

    This exception is raised when transformations, cleanings, or preprocessing
    steps encounter unexpected errors.
    """

    pass
