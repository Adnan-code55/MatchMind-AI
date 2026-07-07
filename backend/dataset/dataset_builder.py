"""Dataset builder for MatchMind AI.

This module converts processed match records into a machine-learning-ready
pandas DataFrame, validates required fields, and generates target labels.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence, Union

import pandas as pd

from backend.app.data.logger import PipelineLogger
from .exceptions import (
    DuplicateMatchError,
    EmptyDatasetError,
    InvalidMatchError,
    MissingFeatureError,
)

MatchRecord = Union[pd.DataFrame, Sequence[Mapping[str, Any]]]

TARGET_LABEL_COLUMN = "target_label"
REQUIRED_COLUMNS = {"Date", "HomeTeam", "AwayTeam"}
RESULT_COLUMNS = ["FTR", "Result", "MatchResult", "match_result", "result"]
SCORE_COLUMNS = [("FTHG", "FTAG")]
TARGET_LABEL_MAP = {
    "H": "HOME_WIN",
    "D": "DRAW",
    "A": "AWAY_WIN",
    "HOME_WIN": "HOME_WIN",
    "DRAW": "DRAW",
    "AWAY_WIN": "AWAY_WIN",
}


class DatasetBuilder:
    """Build a machine-learning-ready dataset from processed match records."""

    def __init__(self) -> None:
        """Initialize the dataset builder."""
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)

    def build_dataset(self, matches: MatchRecord) -> pd.DataFrame:
        """Build a dataset from processed match records.

        Args:
            matches (MatchRecord): Processed match records as a pandas DataFrame
                or iterable of mappings.

        Returns:
            pd.DataFrame: A cleaned, ordered dataset with a generated target.

        Raises:
            EmptyDatasetError: If the input dataset is empty.
            MissingFeatureError: If required fields are missing.
            DuplicateMatchError: If duplicate matches are detected.
            InvalidMatchError: If a target label cannot be generated.
        """
        self.logger.info("Dataset creation started")
        dataset = self._normalize_input(matches)
        self._validate_non_empty(dataset)
        self._validate_required_columns(dataset)
        dataset = self._prepare_date_order(dataset)
        self._validate_duplicate_matches(dataset)
        dataset = self._generate_target_label(dataset)
        feature_columns = self._discover_features(dataset)

        self.logger.info("Number of matches processed: %d", len(dataset))
        self.logger.info("Features discovered: %s", feature_columns)
        self.logger.info(
            "Dataset size: %d rows x %d columns",
            dataset.shape[0],
            dataset.shape[1],
        )
        self.logger.info("Dataset creation completed")
        return dataset

    def _normalize_input(self, matches: MatchRecord) -> pd.DataFrame:
        if isinstance(matches, pd.DataFrame):
            return matches.copy(deep=True)

        if not isinstance(matches, Iterable):
            raise InvalidMatchError("Input matches must be a DataFrame or iterable of records.")

        dataset = pd.DataFrame(list(matches))
        return dataset

    def _validate_non_empty(self, dataset: pd.DataFrame) -> None:
        if dataset.empty:
            raise EmptyDatasetError("Input dataset must contain at least one match.")

    def _validate_required_columns(self, dataset: pd.DataFrame) -> None:
        missing_columns = sorted(REQUIRED_COLUMNS.difference(dataset.columns))
        if missing_columns:
            raise MissingFeatureError(
                f"Required columns are missing: {missing_columns}"
            )

    def _prepare_date_order(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.copy()
        dataset["Date"] = pd.to_datetime(dataset["Date"], errors="raise")
        return dataset.sort_values("Date", kind="stable").reset_index(drop=True)

    def _validate_duplicate_matches(self, dataset: pd.DataFrame) -> None:
        duplicate_keys = dataset.duplicated(subset=["Date", "HomeTeam", "AwayTeam"], keep=False)
        if duplicate_keys.any():
            duplicates = dataset[duplicate_keys][["Date", "HomeTeam", "AwayTeam"]]
            raise DuplicateMatchError(
                f"Duplicate matches detected: {duplicates.to_dict(orient='records')}"
            )

    def _generate_target_label(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if TARGET_LABEL_COLUMN in dataset.columns:
            return dataset.copy()

        score_source = self._find_result_source(dataset)
        if score_source:
            dataset = self._map_target_from_result(dataset, score_source)
            return dataset

        if self._has_score_columns(dataset):
            dataset = self._map_target_from_scores(dataset)
            return dataset

        raise InvalidMatchError(
            "Could not determine target result from existing match information."
        )

    def _find_result_source(self, dataset: pd.DataFrame) -> Optional[str]:
        for column in RESULT_COLUMNS:
            if column in dataset.columns:
                return column
        return None

    def _has_score_columns(self, dataset: pd.DataFrame) -> bool:
        return all(column in dataset.columns for column in ["FTHG", "FTAG"])

    def _map_target_from_result(self, dataset: pd.DataFrame, result_column: str) -> pd.DataFrame:
        dataset = dataset.copy()
        values = dataset[result_column].astype(str).str.upper()
        mapped = values.map(TARGET_LABEL_MAP)
        if mapped.isna().any():
            invalid_rows = dataset.loc[mapped.isna(), ["Date", "HomeTeam", "AwayTeam", result_column]]
            raise InvalidMatchError(
                f"Unable to map result values to target label: {invalid_rows.to_dict(orient='records')}"
            )
        dataset[TARGET_LABEL_COLUMN] = mapped
        return dataset

    def _map_target_from_scores(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.copy()
        home_goals = pd.to_numeric(dataset["FTHG"], errors="coerce")
        away_goals = pd.to_numeric(dataset["FTAG"], errors="coerce")
        if home_goals.isna().any() or away_goals.isna().any():
            invalid_rows = dataset.loc[
                home_goals.isna() | away_goals.isna(),
                ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"],
            ]
            raise InvalidMatchError(
                f"Invalid score values for target label generation: {invalid_rows.to_dict(orient='records')}"
            )

        target = pd.Series(index=dataset.index, dtype="object")
        target.loc[home_goals > away_goals] = "HOME_WIN"
        target.loc[home_goals < away_goals] = "AWAY_WIN"
        target.loc[home_goals == away_goals] = "DRAW"
        dataset[TARGET_LABEL_COLUMN] = target
        return dataset

    def _discover_features(self, dataset: pd.DataFrame) -> List[str]:
        columns = list(dataset.columns)
        return columns
