"""
Main trainer for the ML Training module.
"""
from typing import Any, Dict, Optional
import pandas as pd

from backend.app.data.logger import PipelineLogger
from .models import build_model, get_supported_models
from .validators import validate_model_name, validate_dataset
from .metadata import TrainingMetadata
from .persistence import save_model

class ModelTrainer:
    """
    Coordinates model initialization, training, evaluation, saving,
    and metadata generation.
    """

    def __init__(self, random_seed: int = 42) -> None:
        """
        Initialize the ModelTrainer.
        
        Args:
            random_seed: Random seed used for reproducibility.
        """
        self.random_seed = random_seed
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)
        self._model = None
        self._metadata: Optional[TrainingMetadata] = None

    def train(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
        **kwargs: Any
    ) -> "ModelTrainer":
        """
        Train a model, evaluate it, save it, and generate metadata.
        
        Args:
            model_name: The name of the model to train (e.g., 'logistic_regression').
            X_train: Training features.
            y_train: Training target.
            X_validation: Validation features.
            y_validation: Validation target.
            **kwargs: Additional hyperparameters for the model.
            
        Returns:
            self for method chaining.
        """
        self.logger.info(f"Starting training process for model: {model_name}")

        # Validation
        validate_model_name(model_name, get_supported_models())
        validate_dataset(X_train, y_train, "train")
        validate_dataset(X_validation, y_validation, "validation")

        # Build model
        self.logger.info("Initializing model...")
        model = build_model(model_name, self.random_seed, **kwargs)

        # Train model
        self.logger.info(f"Training on {len(X_train)} samples...")
        model.fit(X_train, y_train)
        
        # Evaluate
        self.logger.info("Evaluating model...")
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_validation, y_validation)

        # Save model
        self.logger.info("Saving model to disk...")
        saved_path = save_model(model, model_name)

        # Generate metadata
        self._metadata = TrainingMetadata(
            algorithm=model_name,
            dataset_size=len(X_train),
            feature_count=X_train.shape[1],
            training_score=train_score,
            validation_score=val_score,
            parameters=kwargs,
            random_seed=self.random_seed
        )
        
        self._model = model
        
        self.logger.info(
            f"Training complete. Train score: {train_score:.4f}, "
            f"Validation score: {val_score:.4f}. Saved to {saved_path}"
        )
        return self

    @property
    def model(self) -> Any:
        """The trained model instance."""
        return self._model

    def get_metadata(self) -> Optional[TrainingMetadata]:
        """Get the metadata generated from the last training run."""
        return self._metadata
