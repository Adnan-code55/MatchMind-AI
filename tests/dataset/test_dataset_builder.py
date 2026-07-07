"""Unit tests for the MatchMind AI DatasetBuilder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.dataset.dataset_builder import DatasetBuilder
from backend.dataset.exceptions import (
    DuplicateMatchError,
    EmptyDatasetError,
    InvalidMatchError,
    MissingFeatureError,
)


def test_build_dataset_from_dataframe_creates_target_and_preserves_order() -> None:
    df = pd.DataFrame(
        [
            {"Date": "2023-08-02", "HomeTeam": "A", "AwayTeam": "B", "FTR": "D", "recent_form": 1},
            {"Date": "2023-08-01", "HomeTeam": "C", "AwayTeam": "D", "FTR": "H", "recent_form": 2},
            {"Date": "2023-08-03", "HomeTeam": "B", "AwayTeam": "C", "FTR": "A", "recent_form": 3},
        ]
    )
    original = df.copy(deep=True)

    builder = DatasetBuilder()
    dataset = builder.build_dataset(df)

    assert dataset.shape == (3, 6)
    assert dataset["target_label"].tolist() == ["HOME_WIN", "DRAW", "AWAY_WIN"]
    assert dataset["Date"].tolist() == sorted(dataset["Date"])
    assert original.equals(df)
    assert "recent_form" in dataset.columns


def test_build_dataset_from_records_with_score_columns_generates_target() -> None:
    matches = [
        {"Date": "2023-08-01", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 2, "FTAG": 1},
        {"Date": "2023-08-02", "HomeTeam": "C", "AwayTeam": "D", "FTHG": 0, "FTAG": 0},
        {"Date": "2023-08-03", "HomeTeam": "B", "AwayTeam": "C", "FTHG": 0, "FTAG": 3},
    ]

    builder = DatasetBuilder()
    dataset = builder.build_dataset(matches)

    assert dataset.shape == (3, 6)
    assert dataset["target_label"].tolist() == ["HOME_WIN", "DRAW", "AWAY_WIN"]
    assert pd.api.types.is_datetime64_any_dtype(dataset["Date"])


def test_build_dataset_empty_input_raises_empty_dataset_error() -> None:
    builder = DatasetBuilder()

    try:
        builder.build_dataset(pd.DataFrame())
        assert False, "Expected EmptyDatasetError"
    except EmptyDatasetError as exc:
        assert "Input dataset must contain at least one match" in str(exc)


def test_build_dataset_missing_required_fields_raises_missing_feature_error() -> None:
    df = pd.DataFrame(
        [{"Date": "2023-08-01", "HomeTeam": "A", "FTR": "H"}]
    )
    builder = DatasetBuilder()

    try:
        builder.build_dataset(df)
        assert False, "Expected MissingFeatureError"
    except MissingFeatureError as exc:
        assert "Required columns are missing" in str(exc)


def test_build_dataset_duplicate_matches_raises_duplicate_match_error() -> None:
    df = pd.DataFrame(
        [
            {"Date": "2023-08-01", "HomeTeam": "A", "AwayTeam": "B", "FTR": "H"},
            {"Date": "2023-08-01", "HomeTeam": "A", "AwayTeam": "B", "FTR": "D"},
        ]
    )
    builder = DatasetBuilder()

    try:
        builder.build_dataset(df)
        assert False, "Expected DuplicateMatchError"
    except DuplicateMatchError as exc:
        assert "Duplicate matches detected" in str(exc)


def test_build_dataset_throws_invalid_match_error_for_unknown_result() -> None:
    df = pd.DataFrame(
        [
            {"Date": "2023-08-01", "HomeTeam": "A", "AwayTeam": "B", "FTR": "X"},
        ]
    )
    builder = DatasetBuilder()

    try:
        builder.build_dataset(df)
        assert False, "Expected InvalidMatchError"
    except InvalidMatchError as exc:
        assert "Unable to map result values to target label" in str(exc)
