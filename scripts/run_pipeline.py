"""
Entry point for MatchMind AI pipeline execution.

This script executes the data pipeline with runtime configuration overrides for
league, season, source directory, and output directory.
"""

import argparse
import sys
from pathlib import Path

from backend.app.pipeline.orchestrator import PipelineOrchestrator
from backend.app.config.settings import PipelineSettings


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for pipeline execution."""
    settings = PipelineSettings.load()

    parser = argparse.ArgumentParser(
        description="Run the MatchMind AI data pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--league",
        type=str,
        default=settings.default_league,
        help="League code to process (e.g. EPL, LA_LIGA).",
    )
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Season year to tag the dataset with.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(settings.data_raw_path),
        help="Path to the raw data source directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(settings.data_processed_path),
        help="Path to the processed data output directory.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the pipeline and report the execution status."""
    args = parse_arguments()
    orchestrator = PipelineOrchestrator(PipelineSettings.load())

    try:
        report = orchestrator.run(
            league=args.league,
            season=args.season,
            source=args.source,
            output=args.output,
        )
        print(f"Pipeline finished: {report.status}")
        print(f"Report file: {report.report_file}")
        return 0
    except Exception as error:
        print(f"Pipeline failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
