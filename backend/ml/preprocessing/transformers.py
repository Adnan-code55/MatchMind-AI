"""
Reusable transformers for preprocessing pipeline.

This module provides individual transformer classes for handling
missing values, encoding, scaling, and target extraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
)

from backend.app.data.logger import PipelineLogger
from .exceptions import TransformationStateError, UnsupportedTransformationError


class Transformer(ABC):
    """Abstract base class for all transformers.

    Transformers follow the fit/transform pattern similar to scikit-learn.
    """

    def __init__(self) -> None:
        """Initialize transformer."""
        self._fitted = False
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)

    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs: Any) -> Transformer:
        """Fit transformer on data.

        Args:
            df: Input DataFrame.
            **kwargs: Additional arguments.

        Returns:
            Self for method chaining.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data.

        Args:
            df: Input DataFrame.

        Returns:
            Transformed DataFrame.

        Raises:
            TransformationStateError: If transformer not fitted.
        """
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Fit and transform in one call.

        Args:
            df: Input DataFrame.
            **kwargs: Additional arguments for fit.

        Returns:
            Transformed DataFrame.
        """
        return self.fit(df, **kwargs).transform(df)


class MissingValueHandler(Transformer):
    """Handle missing values in numerical and categorical columns.

    Supports configurable strategies for each type:
    - Numerical: mean, median, constant
    - Categorical: most_frequent, constant
    """

    def __init__(
        self,
        numerical_strategy: str = "mean",
        numerical_constant: float = 0.0,
        categorical_strategy: str = "most_frequent",
        categorical_constant: str = "MISSING",
    ) -> None:
        """Initialize missing value handler.

        Args:
            numerical_strategy: Strategy for numerical features.
            numerical_constant: Constant value for numerical replacement.
            categorical_strategy: Strategy for categorical features.
            categorical_constant: Constant value for categorical replacement.

        Raises:
            UnsupportedTransformationError: If strategy not supported.
        """
        super().__init__()
        self.numerical_strategy = numerical_strategy
        self.numerical_constant = numerical_constant
        self.categorical_strategy = categorical_strategy
        self.categorical_constant = categorical_constant
        self.fill_values: Dict[str, Any] = {}

        self._validate_strategies()

    def _validate_strategies(self) -> None:
        """Validate strategies are supported."""
        valid_numerical = {"mean", "median", "constant"}
        valid_categorical = {"most_frequent", "constant"}

        if self.numerical_strategy not in valid_numerical:
            raise UnsupportedTransformationError(
                f"Numerical strategy '{self.numerical_strategy}' not supported. "
                f"Valid: {valid_numerical}"
            )

        if self.categorical_strategy not in valid_categorical:
            raise UnsupportedTransformationError(
                f"Categorical strategy '{self.categorical_strategy}' not supported. "
                f"Valid: {valid_categorical}"
            )

    def fit(
        self,
        df: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> MissingValueHandler:
        """Learn missing value strategies from data.

        Args:
            df: Input DataFrame.
            numerical_cols: Numerical column names.
            categorical_cols: Categorical column names.

        Returns:
            Self for method chaining.
        """
        numerical_cols = numerical_cols or []
        categorical_cols = categorical_cols or []

        # Learn fill values for numerical columns
        for col in numerical_cols:
            if col in df.columns:
                if self.numerical_strategy == "mean":
                    self.fill_values[col] = df[col].mean()
                elif self.numerical_strategy == "median":
                    self.fill_values[col] = df[col].median()
                elif self.numerical_strategy == "constant":
                    self.fill_values[col] = self.numerical_constant

        # Learn fill values for categorical columns
        for col in categorical_cols:
            if col in df.columns:
                if self.categorical_strategy == "most_frequent":
                    self.fill_values[col] = df[col].mode().iloc[0] if not df[col].mode().empty else self.categorical_constant
                elif self.categorical_strategy == "constant":
                    self.fill_values[col] = self.categorical_constant

        self._fitted = True
        self.logger.info(
            f"Learned fill values for {len(self.fill_values)} columns"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using learned strategies.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with missing values filled.

        Raises:
            TransformationStateError: If not fitted.
        """
        if not self._fitted:
            raise TransformationStateError(
                "MissingValueHandler must be fitted before transform"
            )

        result = df.copy()
        for col, fill_value in self.fill_values.items():
            if col in result.columns:
                result[col] = result[col].fillna(fill_value)

        return result


class CategoricalEncoder(Transformer):
    """Encode categorical features using label encoding or one-hot encoding.

    Automatically chooses encoding type based on cardinality.
    """

    def __init__(
        self,
        encoding_type: str = "auto",
        max_categories_label: int = 10,
    ) -> None:
        """Initialize categorical encoder.

        Args:
            encoding_type: Type of encoding (auto, label, onehot).
            max_categories_label: Max categories for label encoding.

        Raises:
            UnsupportedTransformationError: If encoding type not supported.
        """
        super().__init__()
        if encoding_type not in {"auto", "label", "onehot"}:
            raise UnsupportedTransformationError(
                f"Encoding type '{encoding_type}' not supported"
            )

        self.encoding_type = encoding_type
        self.max_categories_label = max_categories_label
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.onehot_categories: Dict[str, List[str]] = {}
        self.encoding_strategy: Dict[str, str] = {}

    def fit(
        self,
        df: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None,
    ) -> CategoricalEncoder:
        """Learn encoding for categorical columns.

        Args:
            df: Input DataFrame.
            categorical_cols: Categorical column names.

        Returns:
            Self for method chaining.
        """
        categorical_cols = categorical_cols or []

        for col in categorical_cols:
            if col not in df.columns:
                continue

            cardinality = df[col].nunique()

            # Decide encoding type
            if self.encoding_type == "auto":
                chosen_type = (
                    "label"
                    if cardinality <= self.max_categories_label
                    else "onehot"
                )
            else:
                chosen_type = self.encoding_type

            self.encoding_strategy[col] = chosen_type

            # Fit encoder
            if chosen_type == "label":
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le
            elif chosen_type == "onehot":
                self.onehot_categories[col] = (
                    df[col].astype(str).unique().tolist()
                )

        self._fitted = True
        self.logger.info(
            f"Fitted encoding for {len(self.encoding_strategy)} categorical columns"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with encoded categorical features.

        Raises:
            TransformationStateError: If not fitted.
        """
        if not self._fitted:
            raise TransformationStateError(
                "CategoricalEncoder must be fitted before transform"
            )

        result = df.copy()

        for col, strategy in self.encoding_strategy.items():
            if col not in result.columns:
                continue

            if strategy == "label":
                result[col] = self.label_encoders[col].transform(
                    result[col].astype(str)
                )
            elif strategy == "onehot":
                onehot_df = pd.get_dummies(
                    result[col].astype(str),
                    prefix=col,
                    drop_first=False,
                )
                result = pd.concat([result, onehot_df], axis=1)
                result.drop(columns=[col], inplace=True)

        return result


class FeatureScaler(Transformer):
    """Scale numerical features using StandardScaler or MinMaxScaler."""

    def __init__(
        self,
        scaler_type: str = "standard",
        enabled: bool = True,
    ) -> None:
        """Initialize feature scaler.

        Args:
            scaler_type: Type of scaler (standard, minmax).
            enabled: Whether to enable scaling.

        Raises:
            UnsupportedTransformationError: If scaler type not supported.
        """
        super().__init__()
        if scaler_type not in {"standard", "minmax"}:
            raise UnsupportedTransformationError(
                f"Scaler type '{scaler_type}' not supported"
            )

        self.scaler_type = scaler_type
        self.enabled = enabled
        self.scaler: Optional[Any] = None
        self.scaled_columns: List[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None,
    ) -> FeatureScaler:
        """Fit scaler on numerical data.

        Args:
            df: Input DataFrame.
            numerical_cols: Numerical column names.

        Returns:
            Self for method chaining.
        """
        if not self.enabled:
            self._fitted = True
            return self

        numerical_cols = numerical_cols or []
        self.scaled_columns = [
            col for col in numerical_cols if col in df.columns
        ]

        if not self.scaled_columns:
            self._fitted = True
            return self

        # Create and fit scaler
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:  # minmax
            self.scaler = MinMaxScaler()

        self.scaler.fit(df[self.scaled_columns])
        self._fitted = True
        self.logger.info(
            f"Fitted {self.scaler_type} scaler for {len(self.scaled_columns)} columns"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with scaled features.

        Raises:
            TransformationStateError: If not fitted.
        """
        if not self._fitted:
            raise TransformationStateError(
                "FeatureScaler must be fitted before transform"
            )

        if not self.enabled or not self.scaled_columns or self.scaler is None:
            return df.copy()

        result = df.copy()
        scaled_values = self.scaler.transform(result[self.scaled_columns])
        result[self.scaled_columns] = scaled_values

        return result


class TargetExtractor(Transformer):
    """Extract and preserve target column separately."""

    def __init__(self) -> None:
        """Initialize target extractor."""
        super().__init__()
        self.target_column: Optional[str] = None

    def fit(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> TargetExtractor:
        """Learn target column.

        Args:
            df: Input DataFrame.
            target_column: Name of target column.

        Returns:
            Self for method chaining.
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not in DataFrame")

        self.target_column = target_column
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Extract target column.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (features_df, target_series).

        Raises:
            TransformationStateError: If not fitted.
        """
        if not self._fitted:
            raise TransformationStateError(
                "TargetExtractor must be fitted before transform"
            )

        if self.target_column is None or self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not in DataFrame"
            )

        target = df[self.target_column].copy()
        features = df.drop(columns=[self.target_column])

        return features, target
