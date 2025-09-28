"""
Base model class for commodity price prediction.
Provides common interface for all model types.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Base class for all prediction models."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.feature_columns = []
        self.target_columns = []
        self.model_params = kwargs
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'BaseModel':
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions (if applicable)."""
        pass
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        return None
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'model_params': self.model_params,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'BaseModel':
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.target_columns = model_data['target_columns']
        self.model_params = model_data['model_params']
        self.is_fitted = model_data['is_fitted']
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare input data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Check if all required features are present
        missing_features = set(self.feature_columns) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select only the required features
        X_selected = X[self.feature_columns].copy()
        
        # Handle missing values
        X_selected = X_selected.fillna(method='ffill').fillna(method='bfill')
        
        return X_selected
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_columns),
            'target_count': len(self.target_columns),
            'model_params': self.model_params
        }
