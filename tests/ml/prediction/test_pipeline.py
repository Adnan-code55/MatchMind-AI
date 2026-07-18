import pytest
import pandas as pd
import numpy as np
from backend.ml.prediction.pipeline import PredictionPipeline
from backend.ml.prediction.exceptions import PredictionError

class MockModel:
    def predict_proba(self, X):
        return np.array([[0.5, 0.3, 0.2]])

class MockModelNoProba:
    pass

from unittest.mock import patch

@pytest.fixture
def mock_loader():
    with patch("backend.ml.prediction.pipeline.ModelLoader") as mock:
        mock_instance = mock.return_value
        mock_instance.load_best_model.return_value = MockModel()
        yield mock_instance

def test_prediction_pipeline_execute(mock_loader):
    pipeline = PredictionPipeline()
    df = pd.DataFrame({"feature1": [1.0]})
    
    probs, confidence, metadata = pipeline.execute(df)
    
    assert probs["home_win"] == 0.5
    assert probs["draw"] == 0.3
    assert probs["away_win"] == 0.2
    assert confidence == 0.5
    assert metadata.model_name == "MockModel"
    assert metadata.confidence_score == 0.5
    assert metadata.inference_time_ms >= 0

def test_prediction_pipeline_no_proba():
    with patch("backend.ml.prediction.pipeline.ModelLoader") as mock:
        mock_instance = mock.return_value
        mock_instance.load_best_model.return_value = MockModelNoProba()
        
        pipeline = PredictionPipeline()
        df = pd.DataFrame({"feature1": [1.0]})
        
        with pytest.raises(PredictionError, match="does not support probability predictions"):
            pipeline.execute(df)
