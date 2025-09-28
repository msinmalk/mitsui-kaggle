#!/usr/bin/env python3
"""
Main training script for commodity price prediction challenge.
Trains multiple models and creates ensemble predictions.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any
import argparse
from datetime import datetime
import joblib

# Import our modules
from data.loader import CommodityDataLoader
from data.feature_engineering import FeatureEngineer
from models.traditional_ml import XGBoostModel, LightGBMModel, RandomForestModel, LinearModel
from models.neural_networks import LSTMModel, TransformerModel, CNNLSTMModel
from models.ensemble_model import EnsembleModel, WeightedEnsembleModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CommodityPredictionTrainer:
    """Main trainer class for commodity price prediction."""
    
    def __init__(self, data_dir: str = "data/raw", output_dir: str = "models"):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.loader = CommodityDataLoader(data_dir)
        self.feature_engineer = FeatureEngineer()
        
        # Data storage
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.target_pairs = None
        self.features = None
        
        # Models
        self.models = {}
        self.ensemble_model = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess all data."""
        logger.info("Loading and preprocessing data...")
        
        # Load raw data
        data = self.loader.load_all_data()
        self.train_data = data['train']
        self.test_data = data['test']
        self.train_labels = data['labels']
        self.target_pairs = data['target_pairs']
        
        # Preprocess data
        self.train_data = self.loader.preprocess_data(self.train_data, fill_method='forward')
        self.test_data = self.loader.preprocess_data(self.test_data, fill_method='forward')
        
        # Create features
        logger.info("Creating features...")
        self.features = self.feature_engineer.create_all_features(
            self.train_data, self.target_pairs
        )
        
        # Create features for test data
        test_features = self.feature_engineer.create_all_features(
            self.test_data, self.target_pairs
        )
        
        logger.info(f"Created {len(self.feature_engineer.feature_columns)} features")
        logger.info(f"Feature matrix shape: {self.features.shape}")
        
        return self.features, test_features
    
    def prepare_training_data(self, validation_split: float = 0.2):
        """Prepare training and validation data."""
        logger.info("Preparing training data...")
        
        # Get target columns
        target_cols = [col for col in self.train_labels.columns if col.startswith('target_')]
        
        # Align data by date_id
        train_with_labels = self.features.merge(
            self.train_labels[['date_id'] + target_cols], 
            on='date_id', 
            how='inner'
        )
        
        # Remove rows with missing targets
        train_with_labels = train_with_labels.dropna(subset=target_cols)
        
        # Split features and targets
        feature_cols = [col for col in train_with_labels.columns 
                       if col not in ['date_id'] + target_cols]
        
        X = train_with_labels[feature_cols]
        y = train_with_labels[target_cols]
        
        # Time series split (use last 20% for validation)
        split_idx = int(len(X) * (1 - validation_split))
        
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:]
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        
        return X_train, X_val, y_train, y_val, feature_cols
    
    def train_individual_models(self, X_train, y_train, X_val, y_val):
        """Train individual models."""
        logger.info("Training individual models...")
        
        # Define models to train
        models_config = {
            'xgboost': XGBoostModel(n_estimators=1000, max_depth=6, learning_rate=0.1),
            'lightgbm': LightGBMModel(n_estimators=1000, max_depth=6, learning_rate=0.1),
            'random_forest': RandomForestModel(n_estimators=100, max_depth=10),
            'ridge': LinearModel(model_type='ridge', alpha=1.0),
            'lasso': LinearModel(model_type='lasso', alpha=0.1),
            'lstm': LSTMModel(sequence_length=30, hidden_units=64),
            'transformer': TransformerModel(sequence_length=30, d_model=64, num_heads=8),
            'cnn_lstm': CNNLSTMModel(sequence_length=30, cnn_filters=64, lstm_units=64)
        }
        
        # Train each model
        for name, model in models_config.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                val_pred = model.predict(X_val)
                val_mse = np.mean((y_val.values - val_pred) ** 2)
                val_mae = np.mean(np.abs(y_val.values - val_pred))
                
                logger.info(f"{name} - Val MSE: {val_mse:.6f}, Val MAE: {val_mae:.6f}")
                
                # Save model
                model_path = self.output_dir / f"{name}_model.pkl"
                model.save_model(str(model_path))
                
                self.models[name] = model
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        logger.info(f"Successfully trained {len(self.models)} models")
    
    def train_ensemble_models(self, X_train, y_train, X_val, y_val):
        """Train ensemble models."""
        logger.info("Training ensemble models...")
        
        if len(self.models) < 2:
            logger.warning("Not enough individual models for ensemble")
            return
        
        # Create ensemble from best performing models
        best_models = list(self.models.values())[:5]  # Use top 5 models
        
        # Voting ensemble
        try:
            voting_ensemble = EnsembleModel(
                base_models=best_models,
                ensemble_method='voting'
            )
            voting_ensemble.fit(X_train, y_train)
            
            # Evaluate
            val_pred = voting_ensemble.predict(X_val)
            val_mse = np.mean((y_val.values - val_pred) ** 2)
            logger.info(f"Voting Ensemble - Val MSE: {val_mse:.6f}")
            
            # Save
            voting_ensemble.save_model(str(self.output_dir / "voting_ensemble.pkl"))
            self.models['voting_ensemble'] = voting_ensemble
            
        except Exception as e:
            logger.error(f"Error training voting ensemble: {e}")
        
        # Weighted ensemble
        try:
            weighted_ensemble = WeightedEnsembleModel(
                base_models=best_models,
                weight_method='performance'
            )
            weighted_ensemble.fit(X_train, y_train)
            
            # Evaluate
            val_pred = weighted_ensemble.predict(X_val)
            val_mse = np.mean((y_val.values - val_pred) ** 2)
            logger.info(f"Weighted Ensemble - Val MSE: {val_mse:.6f}")
            
            # Save
            weighted_ensemble.save_model(str(self.output_dir / "weighted_ensemble.pkl"))
            self.models['weighted_ensemble'] = weighted_ensemble
            self.ensemble_model = weighted_ensemble
            
        except Exception as e:
            logger.error(f"Error training weighted ensemble: {e}")
    
    def generate_predictions(self, test_features, feature_cols):
        """Generate predictions for test set."""
        logger.info("Generating test predictions...")
        
        # Prepare test data
        X_test = test_features[feature_cols]
        
        # Generate predictions for all models
        predictions = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_test)
                predictions[name] = pred
                logger.info(f"Generated predictions for {name}: {pred.shape}")
            except Exception as e:
                logger.error(f"Error generating predictions for {name}: {e}")
        
        # Save predictions
        for name, pred in predictions.items():
            pred_df = pd.DataFrame(pred, columns=[f'target_{i}' for i in range(pred.shape[1])])
            pred_df['date_id'] = test_features['date_id'].values
            pred_df.to_csv(self.output_dir / f"{name}_predictions.csv", index=False)
        
        return predictions
    
    def create_submission(self, test_features, feature_cols):
        """Create submission file for the competition."""
        logger.info("Creating submission file...")
        
        if self.ensemble_model is None:
            logger.warning("No ensemble model available for submission")
            return
        
        # Generate predictions
        X_test = test_features[feature_cols]
        predictions = self.ensemble_model.predict(X_test)
        
        # Create submission format
        submission = pd.DataFrame(predictions, columns=[f'target_{i}' for i in range(predictions.shape[1])])
        submission['date_id'] = test_features['date_id'].values
        
        # Save submission
        submission_path = self.output_dir / "submission.csv"
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")
        
        return submission
    
    def run_full_training(self):
        """Run the complete training pipeline."""
        logger.info("Starting full training pipeline...")
        
        try:
            # Load and preprocess data
            train_features, test_features = self.load_and_preprocess_data()
            
            # Prepare training data
            X_train, X_val, y_train, y_val, feature_cols = self.prepare_training_data()
            
            # Train individual models
            self.train_individual_models(X_train, y_train, X_val, y_val)
            
            # Train ensemble models
            self.train_ensemble_models(X_train, y_train, X_val, y_val)
            
            # Generate predictions
            predictions = self.generate_predictions(test_features, feature_cols)
            
            # Create submission
            submission = self.create_submission(test_features, feature_cols)
            
            logger.info("Training pipeline completed successfully!")
            
            return {
                'models': self.models,
                'predictions': predictions,
                'submission': submission
            }
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train commodity price prediction models')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick training with fewer models')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CommodityPredictionTrainer(args.data_dir, args.output_dir)
    
    # Run training
    results = trainer.run_full_training()
    
    print("Training completed! Check the models directory for saved models and predictions.")

if __name__ == "__main__":
    main()
