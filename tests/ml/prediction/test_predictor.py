import pytest
import pandas as pd
from backend.ml.prediction.predictor import MatchPredictor

from unittest.mock import patch

def test_match_predictor_predict():
    with patch("backend.ml.prediction.predictor.PredictionPipeline") as mock_pipeline:
        mock_instance = mock_pipeline.return_value
        mock_instance.execute.return_value = ({"home_win": 0.8}, 0.8, "mock_metadata")

        predictor = MatchPredictor()
        df = pd.DataFrame({"feature": [1]})
        
        probs, conf, meta = predictor.predict(df)
        
        assert probs == {"home_win": 0.8}
        assert conf == 0.8
        assert meta == "mock_metadata"
        mock_instance.execute.assert_called_once_with(df)
