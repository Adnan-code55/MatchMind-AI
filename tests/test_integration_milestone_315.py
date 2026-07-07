"""Integration tests for Milestone 3.1.5."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.app.pipeline.integration import FeaturePipelineIntegration


def _build_sample_matches() -> pd.DataFrame:
    rows = [
        {"Date": "2023-08-01", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 0, "FTR": "H"},
        {"Date": "2023-08-02", "HomeTeam": "C", "AwayTeam": "D", "FTHG": 0, "FTAG": 0, "FTR": "D"},
        {"Date": "2023-08-03", "HomeTeam": "A", "AwayTeam": "C", "FTHG": 2, "FTAG": 1, "FTR": "H"},
        {"Date": "2023-08-04", "HomeTeam": "B", "AwayTeam": "D", "FTHG": 1, "FTAG": 2, "FTR": "A"},
        {"Date": "2023-08-05", "HomeTeam": "C", "AwayTeam": "A", "FTHG": 0, "FTAG": 1, "FTR": "A"},
        {"Date": "2023-08-06", "HomeTeam": "D", "AwayTeam": "B", "FTHG": 2, "FTAG": 2, "FTR": "D"},
        {"Date": "2023-08-07", "HomeTeam": "A", "AwayTeam": "D", "FTHG": 1, "FTAG": 1, "FTR": "D"},
        {"Date": "2023-08-08", "HomeTeam": "B", "AwayTeam": "C", "FTHG": 1, "FTAG": 0, "FTR": "H"},
        {"Date": "2023-08-09", "HomeTeam": "D", "AwayTeam": "A", "FTHG": 0, "FTAG": 2, "FTR": "A"},
        {"Date": "2023-08-10", "HomeTeam": "C", "AwayTeam": "B", "FTHG": 3, "FTAG": 1, "FTR": "H"},
        {"Date": "2023-08-11", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 0, "FTAG": 0, "FTR": "D"},
        {"Date": "2023-08-12", "HomeTeam": "B", "AwayTeam": "D", "FTHG": 2, "FTAG": 1, "FTR": "H"},
        {"Date": "2023-08-20", "HomeTeam": "C", "AwayTeam": "A", "FTHG": 1, "FTAG": 1, "FTR": "D"},
        {"Date": "2023-08-27", "HomeTeam": "B", "AwayTeam": "C", "FTHG": 2, "FTAG": 0, "FTR": "H"},
    ]
    return pd.DataFrame(rows)


def test_feature_pipeline_integration_end_to_end(tmp_path: Path) -> None:
    df = _build_sample_matches()
    runner = FeaturePipelineIntegration(
        feature_order=[
            "goal_difference",
            "home_advantage",
            "head_to_head",
            "recent_form",
            "rest_days",
            "team_performance",
        ]
    )

    result = runner.run(df, output_directory=tmp_path)

    assert len(result.manifest.generators) == 6
    assert all(entry.status == "passed" for entry in result.manifest.generators)
    assert result.dataset_quality_report.output_rows == len(df)
    assert result.dataset_quality_report.generated_feature_count > 0
    assert result.dataset_quality_report.fully_null_features == []
    assert result.dataset_quality_report.is_valid is True
    assert result.time_integrity_report.passed is True
    assert result.validation_report["is_valid"] is True
    assert result.benchmark_report.total_duration_seconds >= 0.0


def test_integration_reports_are_written_to_disk(tmp_path: Path) -> None:
    df = _build_sample_matches()
    runner = FeaturePipelineIntegration()
    result = runner.run(df, output_directory=tmp_path)

    paths = result.save_reports(tmp_path / "nested")
    assert paths["feature_manifest"].exists()
    assert paths["dataset_quality_report"].exists()
    assert paths["pipeline_benchmark"].exists()
    assert paths["time_integrity_report"].exists()
