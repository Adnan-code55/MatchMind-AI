import pytest
import numpy as np
from backend.ml.prediction.probability import ProbabilityGenerator

def test_generate_probabilities():
    generator = ProbabilityGenerator()
    # 0: home_win, 1: draw, 2: away_win
    predict_proba_output = np.array([[0.6, 0.3, 0.1]])
    
    probs, confidence = generator.generate_probabilities(predict_proba_output)
    
    assert probs["home_win"] == 0.6
    assert probs["draw"] == 0.3
    assert probs["away_win"] == 0.1
    assert confidence == 0.6

def test_generate_probabilities_wrong_shape():
    generator = ProbabilityGenerator()
    predict_proba_output = np.array([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2]])
    
    with pytest.raises(ValueError, match="Expected predict_proba_output for a single sample"):
        generator.generate_probabilities(predict_proba_output)

def test_generate_probabilities_missing_classes():
    # If the model only predicted two classes (e.g., 0 and 1)
    generator = ProbabilityGenerator()
    predict_proba_output = np.array([[0.7, 0.3]])
    
    probs, confidence = generator.generate_probabilities(predict_proba_output)
    
    assert probs["home_win"] == 0.7
    assert probs["draw"] == 0.3
    assert probs["away_win"] == 0.0
    assert confidence == 0.7
