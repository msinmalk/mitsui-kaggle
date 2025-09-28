"""
Ensemble model for commodity price prediction.
Combines multiple models for improved performance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models."""
    
    def __init__(self, base_models: List[BaseModel], ensemble_method: str = 'voting', **kwargs):
        super().__init__("EnsembleModel", **kwargs)
        self.base_models = base_models
        self.ensemble_method = ensemble_method
        self.ensemble_model = None
        self.model_weights = None
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'EnsembleModel':
        """Fit the ensemble model."""
        logger.info(f"Fitting ensemble model with {len(self.base_models)} base models")
        
        # Store feature and target columns
        self.feature_columns = X.columns.tolist()
        self.target_columns = y.columns.tolist()
        
        # Fit all base models
        fitted_models = []
        for i, model in enumerate(self.base_models):
            logger.info(f"Fitting base model {i+1}/{len(self.base_models)}: {model.model_name}")
            model.fit(X, y)
            fitted_models.append((f'model_{i}', model))
        
        # Create ensemble based on method
        if self.ensemble_method == 'voting':
            self.ensemble_model = VotingRegressor(fitted_models)
        elif self.ensemble_method == 'stacking':
            self.ensemble_model = StackingRegressor(
                estimators=fitted_models,
                final_estimator=LinearRegression(),
                cv=5
            )
        else:
            raise ValueError("ensemble_method must be 'voting' or 'stacking'")
        
        # Fit ensemble
        self.ensemble_model.fit(X, y)
        self.is_fitted = True
        
        logger.info("Ensemble model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the ensemble."""
        X_validated = self.validate_input(X)
        predictions = self.ensemble_model.predict(X_validated)
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Not applicable for regression ensemble."""
        raise NotImplementedError("Probability prediction not available for regression ensemble")
    
    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from individual base models."""
        X_validated = self.validate_input(X)
        individual_preds = {}
        
        for i, model in enumerate(self.base_models):
            pred = model.predict(X_validated)
            individual_preds[f'model_{i}_{model.model_name}'] = pred
        
        return individual_preds
    
    def get_model_weights(self) -> Optional[Dict[str, float]]:
        """Get model weights if available."""
        if hasattr(self.ensemble_model, 'estimators_'):
            weights = {}
            for i, (name, model) in enumerate(self.ensemble_model.estimators_):
                weights[name] = getattr(model, 'coef_', 1.0) if hasattr(model, 'coef_') else 1.0
            return weights
        return None

class WeightedEnsembleModel(BaseModel):
    """Weighted ensemble model with learned weights."""
    
    def __init__(self, base_models: List[BaseModel], weight_method: str = 'performance', **kwargs):
        super().__init__("WeightedEnsembleModel", **kwargs)
        self.base_models = base_models
        self.weight_method = weight_method
        self.model_weights = None
        self.validation_scores = None
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'WeightedEnsembleModel':
        """Fit the weighted ensemble model."""
        logger.info(f"Fitting weighted ensemble with {len(self.base_models)} base models")
        
        # Store feature and target columns
        self.feature_columns = X.columns.tolist()
        self.target_columns = y.columns.tolist()
        
        # Fit all base models
        for model in self.base_models:
            model.fit(X, y)
        
        # Calculate weights based on validation performance
        if self.weight_method == 'performance':
            self._calculate_performance_weights(X, y)
        elif self.weight_method == 'equal':
            self.model_weights = np.ones(len(self.base_models)) / len(self.base_models)
        else:
            raise ValueError("weight_method must be 'performance' or 'equal'")
        
        self.is_fitted = True
        logger.info(f"Model weights: {self.model_weights}")
        return self
    
    def _calculate_performance_weights(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Calculate weights based on cross-validation performance."""
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error
        
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for model in self.base_models:
            model_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Fit model on training fold
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                
                # Calculate MSE
                mse = mean_squared_error(y_val, pred)
                model_scores.append(mse)
            
            scores.append(np.mean(model_scores))
        
        # Convert MSE to weights (lower MSE = higher weight)
        # Use inverse MSE as weights
        inv_scores = 1.0 / (np.array(scores) + 1e-8)
        self.model_weights = inv_scores / np.sum(inv_scores)
        self.validation_scores = scores
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make weighted predictions."""
        X_validated = self.validate_input(X)
        
        # Get predictions from all models
        predictions = []
        for model in self.base_models:
            pred = model.predict(X_validated)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for i, (pred, weight) in enumerate(zip(predictions, self.model_weights)):
            weighted_pred += weight * pred
        
        return weighted_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Not applicable for regression ensemble."""
        raise NotImplementedError("Probability prediction not available for regression ensemble")
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get the learned model weights."""
        weights = {}
        for i, model in enumerate(self.base_models):
            weights[f'model_{i}_{model.model_name}'] = self.model_weights[i]
        return weights
