import pytest
import pandas as pd
from backend.ml.prediction.validators import validate_prediction_input
from backend.ml.prediction.exceptions import InvalidInputDataError

def test_validate_prediction_input_valid():
    df = pd.DataFrame({"feature1": [1.0], "feature2": [2.0]})
    # Should not raise
    validate_prediction_input(df)

def test_validate_prediction_input_not_dataframe():
    with pytest.raises(InvalidInputDataError, match="must be a pandas DataFrame"):
        validate_prediction_input([1.0, 2.0])

def test_validate_prediction_input_empty():
    df = pd.DataFrame()
    with pytest.raises(InvalidInputDataError, match="cannot be empty"):
        validate_prediction_input(df)
