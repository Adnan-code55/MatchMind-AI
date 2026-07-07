"""
Data preprocessor module for MatchMind AI pipeline.

Responsible for preparing cleaned data for machine learning. Encodes categorical
columns, normalizes data, prepares labels, splits train/test data, and saves
processed datasets.
"""

from typing import Dict, Optional, Tuple, List
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from .logger import PipelineLogger
from .exceptions import DataProcessingError
from .schema import FootballMatchSchema


MODULE_NAME = "preprocessor"


class DataPreprocessor:
    """
    Preprocessor for football match datasets.

    Performs feature engineering and preparation including categorical encoding,
    normalization, label preparation, and train/test splitting.
    """

    def __init__(self) -> None:
        """Initialize the data preprocessor."""
        self.schema = FootballMatchSchema
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        PipelineLogger.log_info(MODULE_NAME, "DataPreprocessor initialized")

    def preprocess(
        self,
        df: pd.DataFrame,
        train_size: float = 0.8,
        random_state: int = 42,
        normalize: bool = True,
        encode_categorical: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform all preprocessing operations on dataset.

        Args:
            df (pd.DataFrame): Cleaned dataset to preprocess.
            train_size (float): Proportion for training set. Defaults to 0.8.
            random_state (int): Random seed for reproducibility. Defaults to 42.
            normalize (bool): Whether to normalize numeric columns. Defaults to True.
            encode_categorical (bool): Whether to encode categorical columns. Defaults to True.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)

        Raises:
            DataProcessingError: If preprocessing operations fail.
        """
        try:
            df = df.copy()

            if encode_categorical:
                df = self.encode_categorical_columns(df)

            if normalize:
                df = self.normalize_numeric_columns(df)

            train_df, test_df = self.split_train_test(
                df,
                train_size=train_size,
                random_state=random_state,
            )

            PipelineLogger.log_info(
                MODULE_NAME,
                f"Preprocessing complete. Train: {len(train_df)}, Test: {len(test_df)}",
            )

            return train_df, test_df

        except Exception as e:
            message = f"Unexpected error during preprocessing: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e

    def encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical columns to numeric values.

        Converts categorical columns (HomeTeam, AwayTeam, FTR) to encoded
        numeric values using LabelEncoder.

        Args:
            df (pd.DataFrame): Dataset with categorical columns.

        Returns:
            pd.DataFrame: Dataset with encoded categorical columns.

        Raises:
            DataProcessingError: If encoding fails.
        """
        try:
            df = df.copy()
            categorical_cols = self.schema.get_categorical_columns()
            categorical_cols = categorical_cols & set(df.columns)

            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    encoder = self.label_encoders[col]
                    df[col] = df[col].map(
                        lambda x: encoder.transform([x])[0]
                        if x in encoder.classes_
                        else -1
                    )

            if categorical_cols:
                PipelineLogger.log_debug(
                    MODULE_NAME,
                    f"Encoded {len(categorical_cols)} categorical column(s)",
                )

            return df

        except Exception as e:
            message = f"Error encoding categorical columns: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e

    def normalize_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numeric columns using StandardScaler.

        Scales numeric columns to mean=0, std=1. Excludes date columns.

        Args:
            df (pd.DataFrame): Dataset with numeric columns.

        Returns:
            pd.DataFrame: Dataset with normalized numeric columns.

        Raises:
            DataProcessingError: If normalization fails.
        """
        try:
            df = df.copy()
            numeric_cols = self.schema.get_numeric_columns()
            numeric_cols = numeric_cols & set(df.columns)

            for col in numeric_cols:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df[[col]] = self.scalers[col].fit_transform(df[[col]])
                else:
                    scaler = self.scalers[col]
                    df[[col]] = scaler.transform(df[[col]])

            if numeric_cols:
                PipelineLogger.log_debug(
                    MODULE_NAME,
                    f"Normalized {len(numeric_cols)} numeric column(s)",
                )

            return df

        except Exception as e:
            message = f"Error normalizing numeric columns: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e

    def split_train_test(
        self,
        df: pd.DataFrame,
        train_size: float = 0.8,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train and test sets.

        Performs stratified split to maintain distribution of target variable.

        Args:
            df (pd.DataFrame): Dataset to split.
            train_size (float): Proportion for training set. Defaults to 0.8.
            random_state (int): Random seed for reproducibility. Defaults to 42.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)

        Raises:
            DataProcessingError: If split fails.
        """
        try:
            if not (0 < train_size < 1):
                raise ValueError("train_size must be between 0 and 1")

            train_df, test_df = train_test_split(
                df,
                train_size=train_size,
                random_state=random_state,
            )

            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)

            PipelineLogger.log_debug(
                MODULE_NAME,
                f"Dataset split: Train {len(train_df)} rows, Test {len(test_df)} rows",
            )

            return train_df, test_df

        except Exception as e:
            message = f"Error splitting train/test: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e

    def save_dataset(
        self,
        df: pd.DataFrame,
        output_path: str,
        name: str = "dataset",
    ) -> None:
        """
        Save processed dataset to CSV file.

        Args:
            df (pd.DataFrame): Dataset to save.
            output_path (str): Path to output directory.
            name (str): Name for the output file (without extension).

        Raises:
            DataProcessingError: If saving fails.
        """
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            file_path = output_dir / f"{name}.csv"
            df.to_csv(file_path, index=False)

            PipelineLogger.log_info(
                MODULE_NAME,
                f"Dataset saved to {file_path} ({len(df)} rows)",
            )

        except Exception as e:
            message = f"Error saving dataset: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e

    def save_encoders(self, output_path: str) -> None:
        """
        Save label encoders for future use.

        Saves encoder metadata to JSON file for production use.

        Args:
            output_path (str): Path to output directory.

        Raises:
            DataProcessingError: If saving fails.
        """
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            encoder_info: Dict[str, List[str]] = {}

            for col, encoder in self.label_encoders.items():
                encoder_info[col] = list(encoder.classes_)

            import json

            encoder_file = output_dir / "label_encoders.json"
            with open(encoder_file, "w") as f:
                json.dump(encoder_info, f, indent=2)

            PipelineLogger.log_info(
                MODULE_NAME,
                f"Encoders saved to {encoder_file}",
            )

        except Exception as e:
            message = f"Error saving encoders: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e

    def get_encoder(self, column_name: str) -> Optional[LabelEncoder]:
        """
        Get a specific label encoder by column name.

        Args:
            column_name (str): Name of the column.

        Returns:
            Optional[LabelEncoder]: Encoder for the column, or None if not found.
        """
        return self.label_encoders.get(column_name)

    def get_scaler(self, column_name: str) -> Optional[StandardScaler]:
        """
        Get a specific scaler by column name.

        Args:
            column_name (str): Name of the column.

        Returns:
            Optional[StandardScaler]: Scaler for the column, or None if not found.
        """
        return self.scalers.get(column_name)

    def prepare_features(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Prepare features for model training.

        Removes non-feature columns and ensures all features are numeric.

        Args:
            df (pd.DataFrame): Dataset to prepare.
            exclude_columns (Optional[List[str]]): Columns to exclude from features.

        Returns:
            pd.DataFrame: Features only dataset.

        Raises:
            DataProcessingError: If preparation fails.
        """
        try:
            df = df.copy()

            if exclude_columns is None:
                exclude_columns = []

            columns_to_keep = [
                col
                for col in df.columns
                if col not in exclude_columns
            ]

            features_df = df[columns_to_keep]

            PipelineLogger.log_debug(
                MODULE_NAME,
                f"Prepared {len(columns_to_keep)} feature column(s)",
            )

            return features_df

        except Exception as e:
            message = f"Error preparing features: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e

    def prepare_labels(
        self,
        df: pd.DataFrame,
        label_column: str,
    ) -> pd.Series:
        """
        Prepare labels for model training.

        Extracts label column and ensures it's numeric.

        Args:
            df (pd.DataFrame): Dataset containing labels.
            label_column (str): Name of the label column.

        Returns:
            pd.Series: Labels for training.

        Raises:
            DataProcessingError: If label preparation fails.
        """
        try:
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found")

            labels = df[label_column].copy()

            PipelineLogger.log_debug(
                MODULE_NAME,
                f"Labels prepared from column: {label_column}",
            )

            return labels

        except Exception as e:
            message = f"Error preparing labels: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e
