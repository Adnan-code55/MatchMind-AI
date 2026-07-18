"""
Tests for ranking in the model selection module.
"""

from backend.ml.model_selection.ranking import ModelRanker

def test_model_ranker():
    ranker = ModelRanker(primary_metric="f1", higher_is_better=True)
    
    results = [
        {"model_name": "ModelA", "metrics": {"f1": 0.80}},
        {"model_name": "ModelB", "metrics": {"f1": 0.95}},
        {"model_name": "ModelC", "metrics": {"f1": 0.90}}
    ]
    
    ranked = ranker.rank(results)
    
    assert ranked[0]["model_name"] == "ModelB"
    assert ranked[1]["model_name"] == "ModelC"
    assert ranked[2]["model_name"] == "ModelA"
