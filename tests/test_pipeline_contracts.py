"""
Tests for data contract validation.

This module verifies that business rule contracts detect invalid match records and
accept valid match data.
"""

import pandas as pd
import pytest

from backend.app.pipeline.contracts import ContractValidator
from backend.app.data.exceptions import (
    DuplicateRowError,
    InvalidDateError,
    InvalidScoreError,
    MissingColumnError,
)


@pytest.fixture
def valid_match_dataframe():
    return pd.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-02"],
            "HomeTeam": ["Arsenal", "Liverpool"],
            "AwayTeam": ["Chelsea", "Tottenham"],
            "FTHG": ["2", "1"],
            "FTAG": ["1", "1"],
            "FTR": ["H", "D"],
            "HS": ["10", "12"],
            "AS": ["7", "8"],
            "HST": ["5", "4"],
            "AST": ["3", "4"],
            "HC": ["8", "6"],
            "AC": ["3", "5"],
            "HY": ["1", "2"],
            "AY": ["2", "1"],
            "HR": ["0", "0"],
            "AR": ["0", "1"],
        }
    )


def test_contract_validator_accepts_valid_data(valid_match_dataframe):
    validator = ContractValidator()
    validator.validate(valid_match_dataframe)


def test_contract_validator_rejects_missing_columns(valid_match_dataframe):
    df = valid_match_dataframe.drop(columns=["FTR"])
    validator = ContractValidator()

    with pytest.raises(MissingColumnError):
        validator.validate(df)


def test_contract_validator_rejects_future_dates(valid_match_dataframe):
    df = valid_match_dataframe.copy()
    df.loc[1, "Date"] = "2099-01-01"
    validator = ContractValidator()

    with pytest.raises(InvalidDateError):
        validator.validate(df)


def test_contract_validator_rejects_negative_scores(valid_match_dataframe):
    df = valid_match_dataframe.copy()
    df.loc[0, "FTHG"] = "-1"
    validator = ContractValidator()

    with pytest.raises(InvalidScoreError):
        validator.validate(df)


def test_contract_validator_rejects_duplicate_matches(valid_match_dataframe):
    df = pd.concat([valid_match_dataframe, valid_match_dataframe.iloc[[0]]], ignore_index=True)
    validator = ContractValidator()

    with pytest.raises(DuplicateRowError):
        validator.validate(df)


def test_contract_validator_rejects_incorrect_result_label(valid_match_dataframe):
    df = valid_match_dataframe.copy()
    df.loc[0, "FTR"] = "A"
    validator = ContractValidator()

    with pytest.raises(InvalidScoreError):
        validator.validate(df)
