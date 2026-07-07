"""
Tests for feature engineering architecture.

These tests verify that the feature engine abstractions can be imported,
registered, and executed without implementing concrete feature calculations.
"""

import pandas as pd

from backend.app.features import (
    FeatureCombiner,
    FeatureGenerator,
    FeaturePipeline,
    FeatureRegistry,
)


class DummyFeature(FeatureGenerator):
    name = "dummy_feature"
    required_columns = ["HomeTeam", "AwayTeam"]
    output_columns = ["dummy_score"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"dummy_score": [0] * len(df)})


def test_feature_registry_can_register_and_instantiate():
    FeatureRegistry.reset()
    FeatureRegistry.register(DummyFeature)

    generators = FeatureRegistry.instantiate_all(order=["dummy_feature"])

    assert len(generators) == 1
    assert generators[0].name == "dummy_feature"


def test_feature_pipeline_runs_and_combines_dummy_features():
    FeatureRegistry.reset()
    FeatureRegistry.register(DummyFeature)

    df = pd.DataFrame({"HomeTeam": ["Arsenal"], "AwayTeam": ["Chelsea"]})
    pipeline = FeaturePipeline(
        registry=FeatureRegistry,
        combiner=FeatureCombiner(strategy="concat"),
        feature_order=["dummy_feature"],
    )

    result = pipeline.run(df)

    assert "dummy_score" in result.columns
    assert result["dummy_score"].tolist() == [0]
