"""
Machine learning preprocessing pipeline.

This module defines the main PreprocessingPipeline that orchestrates
all preprocessing steps for preparing datasets for machine learning models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.app.data.logger import PipelineLogger
from .exceptions import (
    FeatureDetectionError,
    InvalidFeatureTypeError,
    MissingTargetError,
)
from .metadata import FeatureInfo, PreprocessingMetadata
from .transformers import (
    CategoricalEncoder,
    FeatureScaler,
    MissingValueHandler,
    TargetExtractor,
)


class FeatureDetector:
    """Detect and classify feature types in dataset."""

    NUMERICAL_DTYPES = {"int64", "int32", "float64", "float32", "int", "float"}
    CATEGORICAL_DTYPES = {"object", "category", "bool", "string"}
    DATE_INDICATORS = {"date", "time", "timestamp", "dt"}

    def __init__(self) -> None:
        """Initialize feature detector."""
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)

    def detect_features(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Detect and classify feature types.

        Args:
            df: Input DataFrame.
            target_column: Name of target column (will be excluded).

        Returns:
            Dictionary with keys: numerical, categorical, date, metadata.

        Raises:
            FeatureDetectionError: If detection fails.
        """
        try:
            numerical = []
            categorical = []
            date = []
            metadata = []

            for col in df.columns:
                if target_column and col == target_column:
                    continue

                dtype_str = str(df[col].dtype)
                col_lower = col.lower()

                # Check if date-like
                if any(indicator in col_lower for indicator in self.DATE_INDICATORS):
                    date.append(col)
                elif dtype_str in self.NUMERICAL_DTYPES:
                    numerical.append(col)
                elif dtype_str in self.CATEGORICAL_DTYPES or "string" in dtype_str.lower() or dtype_str.lower() == "str":
                    categorical.append(col)
                else:
                    metadata.append(col)

            self.logger.info(
                f"Detected features: {len(numerical)} numerical, "
                f"{len(categorical)} categorical, {len(date)} date"
            )

            return {
                "numerical": numerical,
                "categorical": categorical,
                "date": date,
                "metadata": metadata,
            }
        except Exception as exc:
            raise FeatureDetectionError(
                f"Failed to detect features: {exc}"
            ) from exc


class PreprocessingPipeline:
    """Main preprocessing pipeline for ML datasets.

    Orchestrates feature detection, missing value handling, encoding,
    scaling, and metadata generation.
    """

    def __init__(
        self,
        target_column: Optional[str] = None,
        numerical_missing_strategy: str = "mean",
        categorical_missing_strategy: str = "most_frequent",
        encoding_type: str = "auto",
        scaler_type: str = "standard",
        enable_scaling: bool = True,
        drop_date_columns: bool = True,
        drop_metadata_columns: bool = True,
    ) -> None:
        """Initialize preprocessing pipeline.

        Args:
            target_column: Name of target column.
            numerical_missing_strategy: Strategy for numerical missing values.
            categorical_missing_strategy: Strategy for categorical missing values.
            encoding_type: Type of categorical encoding (auto, label, onehot).
            scaler_type: Type of scaler (standard, minmax).
            enable_scaling: Whether to enable feature scaling.
            drop_date_columns: Whether to drop date columns.
            drop_metadata_columns: Whether to drop metadata columns.
        """
        self.target_column = target_column
        self.numerical_missing_strategy = numerical_missing_strategy
        self.categorical_missing_strategy = categorical_missing_strategy
        self.encoding_type = encoding_type
        self.scaler_type = scaler_type
        self.enable_scaling = enable_scaling
        self.drop_date_columns = drop_date_columns
        self.drop_metadata_columns = drop_metadata_columns

        self.logger = PipelineLogger.get_logger(self.__class__.__name__)

        # Initialize transformers
        self.detector = FeatureDetector()
        self.missing_value_handler: Optional[MissingValueHandler] = None
        self.categorical_encoder: Optional[CategoricalEncoder] = None
        self.feature_scaler: Optional[FeatureScaler] = None
        self.target_extractor: Optional[TargetExtractor] = None

        # State tracking
        self._fitted = False
        self.feature_types: Dict[str, List[str]] = {}
        self.metadata: Optional[PreprocessingMetadata] = None

    def fit(self, df: pd.DataFrame) -> PreprocessingPipeline:
        """Fit pipeline on training data.

        Args:
            df: Training DataFrame.

        Returns:
            Self for method chaining.

        Raises:
            MissingTargetError: If target column is missing.
        """
        self.logger.info("Starting preprocessing pipeline fit")

        # Validate input
        if df.empty:
            raise ValueError("Cannot fit pipeline on empty DataFrame")

        if self.target_column and self.target_column not in df.columns:
            raise MissingTargetError(
                f"Target column '{self.target_column}' not found in DataFrame"
            )

        # Detect features
        self.feature_types = self.detector.detect_features(df, self.target_column)

        # Initialize transformers
        self.missing_value_handler = MissingValueHandler(
            numerical_strategy=self.numerical_missing_strategy,
            categorical_strategy=self.categorical_missing_strategy,
        )

        self.categorical_encoder = CategoricalEncoder(
            encoding_type=self.encoding_type
        )

        self.feature_scaler = FeatureScaler(
            scaler_type=self.scaler_type,
            enabled=self.enable_scaling,
        )

        if self.target_column:
            self.target_extractor = TargetExtractor()
            self.target_extractor.fit(df, self.target_column)

        # Fit transformers
        working_df = df.copy()

        self.missing_value_handler.fit(
            working_df,
            numerical_cols=self.feature_types["numerical"],
            categorical_cols=self.feature_types["categorical"],
        )
        self.logger.info("Fitted missing value handler")

        working_df = self.missing_value_handler.transform(working_df)

        self.categorical_encoder.fit(
            working_df,
            categorical_cols=self.feature_types["categorical"],
        )
        self.logger.info("Fitted categorical encoder")

        working_df = self.categorical_encoder.transform(working_df)

        # Update feature types after encoding (one-hot creates new columns)
        encoded_cols = [
            col
            for col in working_df.columns
            if col not in df.columns
            and col != self.target_column
        ]

        self.feature_scaler.fit(
            working_df,
            numerical_cols=self.feature_types["numerical"] + encoded_cols,
        )
        self.logger.info("Fitted feature scaler")

        self._fitted = True
        self.logger.info("Pipeline fit completed successfully")

        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Transform dataset using fitted pipeline.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (processed_features, target) where target is None
            if not extracted.

        Raises:
            ValueError: If pipeline not fitted.
        """
        if not self._fitted:
            raise ValueError(
                "Pipeline must be fitted before transform. Call fit() first."
            )

        self.logger.info(
            f"Transforming dataset: {df.shape[0]} rows × {df.shape[1]} cols"
        )

        working_df = df.copy()
        original_shape = working_df.shape
        target = None

        # Handle missing values
        working_df = self.missing_value_handler.transform(working_df)

        # Encode categorical features
        working_df = self.categorical_encoder.transform(working_df)

        # Scale numerical features
        working_df = self.feature_scaler.transform(working_df)

        # Drop columns
        cols_to_drop = []
        if self.drop_date_columns:
            cols_to_drop.extend(
                [col for col in self.feature_types["date"] if col in working_df.columns]
            )
        if self.drop_metadata_columns:
            cols_to_drop.extend(
                [col for col in self.feature_types["metadata"] if col in working_df.columns]
            )

        working_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        # Extract target
        if self.target_extractor:
            working_df, target = self.target_extractor.transform(working_df)

        # Generate metadata
        self._generate_metadata(df, working_df, target, original_shape)

        self.logger.info(
            f"Transform completed: {working_df.shape[0]} rows × {working_df.shape[1]} cols"
        )

        return working_df, target

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Fit and transform in one call.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (processed_features, target).
        """
        return self.fit(df).transform(df)

    def _generate_metadata(
        self,
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        target: Optional[pd.Series],
        original_shape: Tuple[int, int],
    ) -> None:
        """Generate metadata about preprocessing.

        Args:
            original_df: Original DataFrame.
            processed_df: Processed DataFrame.
            target: Extracted target series.
            original_shape: Original DataFrame shape.
        """
        features = []

        # Get encoded columns
        encoded_cols = self.categorical_encoder.encoding_strategy.keys() if self.categorical_encoder else []
        scaled_cols = self.feature_scaler.scaled_columns if self.feature_scaler else []

        # Track dropped columns
        dropped_cols = []
        if self.drop_date_columns:
            dropped_cols.extend(self.feature_types["date"])
        if self.drop_metadata_columns:
            dropped_cols.extend(self.feature_types["metadata"])

        # Create feature info
        for col in original_df.columns:
            if col == self.target_column or col in dropped_cols:
                continue

            missing_count = original_df[col].isna().sum()
            feature_type = "unknown"

            if col in self.feature_types["numerical"]:
                feature_type = "numerical"
            elif col in self.feature_types["categorical"]:
                feature_type = "categorical"

            info = FeatureInfo(
                name=col,
                feature_type=feature_type,
                original_dtype=str(original_df[col].dtype),
                missing_count=int(missing_count),
                missing_strategy=(
                    self.numerical_missing_strategy
                    if feature_type == "numerical"
                    else self.categorical_missing_strategy
                ),
                encoded=col in encoded_cols,
                scaled=col in scaled_cols,
            )

            features.append(info)

        self.metadata = PreprocessingMetadata(
            original_shape=original_shape,
            processed_shape=processed_df.shape,
            target_column=self.target_column,
            target_dtype=str(original_df[self.target_column].dtype) if self.target_column else None,
            features=features,
            encoded_features=list(encoded_cols),
            scaled_features=scaled_cols,
            dropped_features=dropped_cols,
            pipeline_config={
                "numerical_missing_strategy": self.numerical_missing_strategy,
                "categorical_missing_strategy": self.categorical_missing_strategy,
                "encoding_type": self.encoding_type,
                "scaler_type": self.scaler_type,
                "enable_scaling": self.enable_scaling,
            },
        )

    def get_metadata(self) -> Optional[PreprocessingMetadata]:
        """Get preprocessing metadata.

        Returns:
            PreprocessingMetadata or None if not generated yet.
        """
        return self.metadata

    def describe(self) -> str:
        """Return description of pipeline configuration.

        Returns:
            Description string.
        """
        config = (
            f"PreprocessingPipeline("
            f"target={self.target_column}, "
            f"numerical_missing={self.numerical_missing_strategy}, "
            f"categorical_missing={self.categorical_missing_strategy}, "
            f"encoding={self.encoding_type}, "
            f"scaler={self.scaler_type}, "
            f"scaling_enabled={self.enable_scaling})"
        )
        return config
