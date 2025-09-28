"""
Traditional machine learning models for commodity price prediction.
Includes XGBoost, LightGBM, Random Forest, and Linear models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class XGBoostModel(BaseModel):
    """XGBoost model for commodity price prediction."""
    
    def __init__(self, n_estimators: int = 1000, max_depth: int = 6, 
                 learning_rate: float = 0.1, subsample: float = 0.8, **kwargs):
        super().__init__("XGBoostModel", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'XGBoostModel':
        """Fit the XGBoost model."""
        logger.info("Fitting XGBoost model...")
        
        # Store feature and target columns
        self.feature_columns = X.columns.tolist()
        self.target_columns = y.columns.tolist()
        
        # For multi-output regression, we'll use separate models for each target
        self.models = {}
        
        for i, target_col in enumerate(self.target_columns):
            logger.info(f"Training model for target {i+1}/{len(self.target_columns)}: {target_col}")
            
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X, y[target_col])
            self.models[target_col] = model
        
        self.is_fitted = True
        logger.info("XGBoost model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using XGBoost."""
        X_validated = self.validate_input(X)
        
        predictions = []
        for target_col in self.target_columns:
            pred = self.models[target_col].predict(X_validated)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Not applicable for regression."""
        raise NotImplementedError("Probability prediction not available for regression")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance for all targets."""
        importance_data = []
        
        for target_col, model in self.models.items():
            importance = model.feature_importances_
            for i, feature in enumerate(self.feature_columns):
                importance_data.append({
                    'target': target_col,
                    'feature': feature,
                    'importance': importance[i]
                })
        
        return pd.DataFrame(importance_data)

class LightGBMModel(BaseModel):
    """LightGBM model for commodity price prediction."""
    
    def __init__(self, n_estimators: int = 1000, max_depth: int = 6, 
                 learning_rate: float = 0.1, subsample: float = 0.8, **kwargs):
        super().__init__("LightGBMModel", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'LightGBMModel':
        """Fit the LightGBM model."""
        logger.info("Fitting LightGBM model...")
        
        # Store feature and target columns
        self.feature_columns = X.columns.tolist()
        self.target_columns = y.columns.tolist()
        
        # For multi-output regression, we'll use separate models for each target
        self.models = {}
        
        for i, target_col in enumerate(self.target_columns):
            logger.info(f"Training model for target {i+1}/{len(self.target_columns)}: {target_col}")
            
            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            model.fit(X, y[target_col])
            self.models[target_col] = model
        
        self.is_fitted = True
        logger.info("LightGBM model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using LightGBM."""
        X_validated = self.validate_input(X)
        
        predictions = []
        for target_col in self.target_columns:
            pred = self.models[target_col].predict(X_validated)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Not applicable for regression."""
        raise NotImplementedError("Probability prediction not available for regression")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance for all targets."""
        importance_data = []
        
        for target_col, model in self.models.items():
            importance = model.feature_importances_
            for i, feature in enumerate(self.feature_columns):
                importance_data.append({
                    'target': target_col,
                    'feature': feature,
                    'importance': importance[i]
                })
        
        return pd.DataFrame(importance_data)

class RandomForestModel(BaseModel):
    """Random Forest model for commodity price prediction."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None, 
                 min_samples_split: int = 2, min_samples_leaf: int = 1, **kwargs):
        super().__init__("RandomForestModel", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'RandomForestModel':
        """Fit the Random Forest model."""
        logger.info("Fitting Random Forest model...")
        
        # Store feature and target columns
        self.feature_columns = X.columns.tolist()
        self.target_columns = y.columns.tolist()
        
        # For multi-output regression, we'll use separate models for each target
        self.models = {}
        
        for i, target_col in enumerate(self.target_columns):
            logger.info(f"Training model for target {i+1}/{len(self.target_columns)}: {target_col}")
            
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X, y[target_col])
            self.models[target_col] = model
        
        self.is_fitted = True
        logger.info("Random Forest model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using Random Forest."""
        X_validated = self.validate_input(X)
        
        predictions = []
        for target_col in self.target_columns:
            pred = self.models[target_col].predict(X_validated)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Not applicable for regression."""
        raise NotImplementedError("Probability prediction not available for regression")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance for all targets."""
        importance_data = []
        
        for target_col, model in self.models.items():
            importance = model.feature_importances_
            for i, feature in enumerate(self.feature_columns):
                importance_data.append({
                    'target': target_col,
                    'feature': feature,
                    'importance': importance[i]
                })
        
        return pd.DataFrame(importance_data)

class LinearModel(BaseModel):
    """Linear regression model for commodity price prediction."""
    
    def __init__(self, model_type: str = 'ridge', alpha: float = 1.0, **kwargs):
        super().__init__("LinearModel", **kwargs)
        self.model_type = model_type
        self.alpha = alpha
        self.scaler = StandardScaler()
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'LinearModel':
        """Fit the Linear model."""
        logger.info(f"Fitting {self.model_type} model...")
        
        # Store feature and target columns
        self.feature_columns = X.columns.tolist()
        self.target_columns = y.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Choose model type
        if self.model_type == 'ridge':
            base_model = Ridge(alpha=self.alpha)
        elif self.model_type == 'lasso':
            base_model = Lasso(alpha=self.alpha)
        elif self.model_type == 'elastic':
            base_model = ElasticNet(alpha=self.alpha)
        else:
            base_model = LinearRegression()
        
        # For multi-output regression, we'll use separate models for each target
        self.models = {}
        
        for i, target_col in enumerate(self.target_columns):
            logger.info(f"Training model for target {i+1}/{len(self.target_columns)}: {target_col}")
            
            model = base_model.__class__(**base_model.get_params())
            model.fit(X_scaled, y[target_col])
            self.models[target_col] = model
        
        self.is_fitted = True
        logger.info(f"{self.model_type} model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using Linear model."""
        X_validated = self.validate_input(X)
        X_scaled = self.scaler.transform(X_validated)
        
        predictions = []
        for target_col in self.target_columns:
            pred = self.models[target_col].predict(X_scaled)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Not applicable for regression."""
        raise NotImplementedError("Probability prediction not available for regression")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance for all targets."""
        importance_data = []
        
        for target_col, model in self.models.items():
            if hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
                for i, feature in enumerate(self.feature_columns):
                    importance_data.append({
                        'target': target_col,
                        'feature': feature,
                        'importance': importance[i]
                    })
        
        return pd.DataFrame(importance_data) if importance_data else None

class SVMModel(BaseModel):
    """Support Vector Machine model for commodity price prediction."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale', **kwargs):
        super().__init__("SVMModel", **kwargs)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.scaler = StandardScaler()
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'SVMModel':
        """Fit the SVM model."""
        logger.info("Fitting SVM model...")
        
        # Store feature and target columns
        self.feature_columns = X.columns.tolist()
        self.target_columns = y.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # For multi-output regression, we'll use separate models for each target
        self.models = {}
        
        for i, target_col in enumerate(self.target_columns):
            logger.info(f"Training model for target {i+1}/{len(self.target_columns)}: {target_col}")
            
            model = SVR(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma
            )
            
            model.fit(X_scaled, y[target_col])
            self.models[target_col] = model
        
        self.is_fitted = True
        logger.info("SVM model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using SVM."""
        X_validated = self.validate_input(X)
        X_scaled = self.scaler.transform(X_validated)
        
        predictions = []
        for target_col in self.target_columns:
            pred = self.models[target_col].predict(X_scaled)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Not applicable for regression."""
        raise NotImplementedError("Probability prediction not available for regression")
