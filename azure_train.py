"""
Azure ML training script for Mitsui Commodity Prediction.
Optimized for cloud deployment with progress tracking.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

# Add src to path
sys.path.append('src')

from data.loader import CommodityDataLoader
from data.feature_engineering_simple import SimpleFeatureEngineer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

class AzureMLTrainer:
    """Azure ML optimized trainer for commodity prediction."""
    
    def __init__(self, data_dir: str = "data/raw", output_dir: str = "outputs"):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.loader = CommodityDataLoader(data_dir)
        self.feature_engineer = SimpleFeatureEngineer()
        
        # Results tracking
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for training."""
        print("🔄 Loading data...")
        
        # Load data
        data = self.loader.load_all_data()
        
        # Create features
        print("🔄 Creating features...")
        features = self.feature_engineer.create_all_features(
            data['train'], 
            data['target_pairs'], 
            verbose=True
        )
        
        # Prepare target variables
        target_cols = [col for col in data['labels'].columns if col.startswith('target_')]
        targets = data['labels'][target_cols]
        
        # Align features and targets
        common_idx = features.index.intersection(targets.index)
        features = features.loc[common_idx]
        targets = targets.loc[common_idx]
        
        # Handle NaN values
        print(f"   Handling NaN values...")
        print(f"   Features NaN: {features.isnull().sum().sum()}")
        print(f"   Targets NaN: {targets.isnull().sum().sum()}")
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        targets = targets.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"   After cleaning - Features NaN: {features.isnull().sum().sum()}")
        print(f"   After cleaning - Targets NaN: {targets.isnull().sum().sum()}")
        
        print(f"✅ Data prepared: {features.shape} features, {targets.shape} targets")
        
        return features, targets, target_cols
    
    def train_models(self, X, y, target_cols):
        """Train multiple models for comparison."""
        print("🔄 Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        print("   Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Train XGBoost
        print("   Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        
        # Train LightGBM
        print("   Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        
        # Evaluate models
        models = {
            'RandomForest': rf_model,
            'XGBoost': xgb_model,
            'LightGBM': lgb_model
        }
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            self.results[name] = {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
            
            print(f"   {name} - MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        return models, X_test, y_test
    
    def save_models_and_results(self, models, X_test, y_test):
        """Save models and results for Azure ML."""
        print("🔄 Saving models and results...")
        
        # Save models
        for name, model in models.items():
            model_path = self.output_dir / f"{name.lower()}_model.pkl"
            joblib.dump(model, model_path)
            print(f"   Saved {name} model to {model_path}")
        
        # Save feature engineer
        feature_engineer_path = self.output_dir / "feature_engineer.pkl"
        joblib.dump(self.feature_engineer, feature_engineer_path)
        print(f"   Saved feature engineer to {feature_engineer_path}")
        
        # Save results
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"   Saved results to {results_path}")
        
        # Save test data for evaluation
        test_data_path = self.output_dir / "test_data.csv"
        X_test.to_csv(test_data_path)
        print(f"   Saved test data to {test_data_path}")
        
        # Create Azure ML run info
        run_info = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': list(models.keys()),
            'results': self.results,
            'feature_count': X_test.shape[1],
            'test_samples': X_test.shape[0]
        }
        
        run_info_path = self.output_dir / "run_info.json"
        with open(run_info_path, 'w') as f:
            json.dump(run_info, f, indent=2)
        print(f"   Saved run info to {run_info_path}")
        
        return run_info
    
    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        print("🚀 Starting Azure ML training pipeline...")
        start_time = datetime.now()
        
        try:
            # Load and prepare data
            X, y, target_cols = self.load_and_prepare_data()
            
            # Train models
            models, X_test, y_test = self.train_models(X, y, target_cols)
            
            # Save everything
            run_info = self.save_models_and_results(models, X_test, y_test)
            
            # Print summary
            total_time = (datetime.now() - start_time).total_seconds()
            print(f"\n✅ Training pipeline completed in {total_time:.1f} seconds")
            print(f"   Models trained: {len(models)}")
            print(f"   Best model: {min(self.results.keys(), key=lambda k: self.results[k]['mse'])}")
            print(f"   Output directory: {self.output_dir}")
            
            return run_info
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise

def main():
    """Main training function."""
    # Get data directory from environment or use default
    data_dir = os.getenv('DATA_DIR', 'data/raw')
    output_dir = os.getenv('OUTPUT_DIR', 'outputs')
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create trainer and run pipeline
    trainer = AzureMLTrainer(data_dir, output_dir)
    results = trainer.run_training_pipeline()
    
    print("\n🎉 Training completed successfully!")
    print("Ready for Azure ML deployment!")

if __name__ == "__main__":
    main()
