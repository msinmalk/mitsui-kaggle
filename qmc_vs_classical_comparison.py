"""
Compare Quantum Monte Carlo vs Classical vs Simple Quantum for uncertainty calibration.
"""

import sys
sys.path.append('./src')

from data.loader import CommodityDataLoader
from data.feature_engineering_simple import SimpleFeatureEngineer
from quantum.quantum_monte_carlo import QuantumMonteCarlo
from quantum.working_quantum_uncertainty import WorkingQuantumUncertainty
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClassicalModel:
    """Pure classical model with classical uncertainty."""
    
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        self.xgb_model = xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        self.is_trained = False
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train classical models."""
        X_clean = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        X_values = X_clean.values
        y_values = y.fillna(method='ffill').fillna(method='bfill').fillna(0).values
        
        self.rf_model.fit(X_values, y_values)
        self.xgb_model.fit(X_values, y_values)
        self.is_trained = True
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with classical uncertainty."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_clean = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        X_values = X_clean.values
        
        # Get predictions
        rf_pred = self.rf_model.predict(X_values)
        xgb_pred = self.xgb_model.predict(X_values)
        
        # Ensemble prediction
        ensemble_pred = 0.5 * rf_pred + 0.5 * xgb_pred
        
        # Classical uncertainty (variance of predictions)
        classical_uncertainty = np.var([rf_pred, xgb_pred], axis=0)
        
        return ensemble_pred, classical_uncertainty

class SimpleQuantumModel:
    """Model with simple quantum uncertainty (entropy-based)."""
    
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        self.xgb_model = xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        self.quantum_model = WorkingQuantumUncertainty(n_qubits=4)
        self.is_trained = False
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train models."""
        X_clean = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        X_values = X_clean.values
        y_values = y.fillna(method='ffill').fillna(method='bfill').fillna(0).values
        
        self.rf_model.fit(X_values, y_values)
        self.xgb_model.fit(X_values, y_values)
        self.quantum_model.train(X_values, y_values)
        self.is_trained = True
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with simple quantum uncertainty."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_clean = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        X_values = X_clean.values
        
        # Get predictions
        rf_pred = self.rf_model.predict(X_values)
        xgb_pred = self.xgb_model.predict(X_values)
        ensemble_pred = 0.5 * rf_pred + 0.5 * xgb_pred
        
        # Get quantum uncertainty
        quantum_unc = self.quantum_model.batch_predict_uncertainty(X_values)
        quantum_uncertainty = np.array([q['total_uncertainty'] for q in quantum_unc])
        
        return ensemble_pred, quantum_uncertainty

class QMCModel:
    """Model with Quantum Monte Carlo uncertainty."""
    
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        self.xgb_model = xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        self.qmc_model = QuantumMonteCarlo(n_qubits=4, n_samples=200, n_circuits=5)
        self.is_trained = False
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train models."""
        X_clean = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        X_values = X_clean.values
        y_values = y.fillna(method='ffill').fillna(method='bfill').fillna(0).values
        
        self.rf_model.fit(X_values, y_values)
        self.xgb_model.fit(X_values, y_values)
        self.qmc_model.train(X_values, y_values)
        self.is_trained = True
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with QMC uncertainty."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_clean = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        X_values = X_clean.values
        
        # Get predictions
        rf_pred = self.rf_model.predict(X_values)
        xgb_pred = self.xgb_model.predict(X_values)
        ensemble_pred = 0.5 * rf_pred + 0.5 * xgb_pred
        
        # Get QMC uncertainty
        qmc_unc = self.qmc_model.batch_predict_uncertainty(X_values)
        qmc_uncertainty = np.array([q['qmc_uncertainty'] for q in qmc_unc])
        
        return ensemble_pred, qmc_uncertainty

def evaluate_uncertainty_calibration(y_true: pd.Series, y_pred: np.ndarray, uncertainty: np.ndarray, model_name: str) -> Dict[str, float]:
    """Evaluate uncertainty calibration."""
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Uncertainty metrics
    avg_uncertainty = np.mean(uncertainty)
    uncertainty_std = np.std(uncertainty)
    
    # Calibration metrics
    actual_errors = np.abs(y_true - y_pred)
    
    # Correlation between uncertainty and actual errors
    uncertainty_correlation = np.corrcoef(actual_errors, uncertainty)[0, 1] if len(uncertainty) > 1 else 0
    
    # Calibration score (how well uncertainty predicts actual errors)
    # Higher is better
    calibration_score = uncertainty_correlation
    
    # Reliability score (how consistent uncertainty is with error magnitude)
    # We want uncertainty to be proportional to actual errors
    if np.std(uncertainty) > 0 and np.std(actual_errors) > 0:
        reliability_score = np.corrcoef(uncertainty, actual_errors)[0, 1]
    else:
        reliability_score = 0
    
    # Sharpness (how tight the uncertainty bounds are)
    # Lower is better (more precise)
    sharpness = np.mean(uncertainty)
    
    return {
        'model': model_name,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Avg_Uncertainty': avg_uncertainty,
        'Uncertainty_Std': uncertainty_std,
        'Uncertainty_Correlation': uncertainty_correlation,
        'Calibration_Score': calibration_score,
        'Reliability_Score': reliability_score,
        'Sharpness': sharpness
    }

def main():
    print("🔬 Quantum Monte Carlo vs Classical vs Simple Quantum Comparison")
    print("=================================================================")
    
    # Load data
    print("📊 Loading data...")
    loader = CommodityDataLoader(data_dir='data/raw')
    raw_data = loader.load_all_data()
    train_df = raw_data['train']
    target_pairs = raw_data['target_pairs']
    train_labels = raw_data['labels']
    
    # Feature engineering
    print("🔧 Creating features...")
    feature_engineer = SimpleFeatureEngineer()
    features_df = feature_engineer.create_all_features(train_df, target_pairs, verbose=False)
    
    # Align data
    target_cols = [col for col in train_labels.columns if col.startswith('target_')]
    targets_df = train_labels[target_cols]
    
    common_idx = features_df.index.intersection(targets_df.index)
    features_df = features_df.loc[common_idx]
    targets_df = targets_df.loc[common_idx]
    
    # Clean data
    features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    targets_df = targets_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"✅ Data prepared: {features_df.shape} features, {targets_df.shape} targets")
    
    # Test on multiple targets
    num_targets = 3
    sample_targets = target_cols[:num_targets]
    
    classical_results = []
    simple_quantum_results = []
    qmc_results = []
    
    for i, target_col in enumerate(sample_targets):
        print(f"\n--- Target {i+1}/{num_targets}: {target_col} ---")
        
        X = features_df.copy()
        y = targets_df[target_col].copy()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Test Classical Model
        print("🌲 Testing Classical Model...")
        classical_model = ClassicalModel()
        classical_model.train(X_train, y_train)
        classical_pred, classical_unc = classical_model.predict_with_uncertainty(X_test)
        classical_metrics = evaluate_uncertainty_calibration(y_test, classical_pred, classical_unc, "Classical")
        classical_results.append(classical_metrics)
        
        # Test Simple Quantum Model
        print("🔮 Testing Simple Quantum Model...")
        simple_quantum_model = SimpleQuantumModel()
        simple_quantum_model.train(X_train, y_train)
        sq_pred, sq_unc = simple_quantum_model.predict_with_uncertainty(X_test)
        sq_metrics = evaluate_uncertainty_calibration(y_test, sq_pred, sq_unc, "Simple Quantum")
        simple_quantum_results.append(sq_metrics)
        
        # Test QMC Model
        print("🔬 Testing Quantum Monte Carlo Model...")
        qmc_model = QMCModel()
        qmc_model.train(X_train, y_train)
        qmc_pred, qmc_unc = qmc_model.predict_with_uncertainty(X_test)
        qmc_metrics = evaluate_uncertainty_calibration(y_test, qmc_pred, qmc_unc, "QMC")
        qmc_results.append(qmc_metrics)
        
        print(f"  Classical Calibration: {classical_metrics['Calibration_Score']:.3f}")
        print(f"  Simple Quantum Calibration: {sq_metrics['Calibration_Score']:.3f}")
        print(f"  QMC Calibration: {qmc_metrics['Calibration_Score']:.3f}")
    
    # Summary comparison
    print("\n📊 UNCERTAINTY CALIBRATION COMPARISON")
    print("=" * 50)
    
    # Average metrics
    classical_avg = {key: np.mean([r[key] for r in classical_results]) for key in classical_results[0].keys() if key != 'model'}
    sq_avg = {key: np.mean([r[key] for r in simple_quantum_results]) for key in simple_quantum_results[0].keys() if key != 'model'}
    qmc_avg = {key: np.mean([r[key] for r in qmc_results]) for key in qmc_results[0].keys() if key != 'model'}
    
    print("\n🌲 Classical Model (Average):")
    for key, value in classical_avg.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n🔮 Simple Quantum Model (Average):")
    for key, value in sq_avg.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n🔬 Quantum Monte Carlo Model (Average):")
    for key, value in qmc_avg.items():
        print(f"  {key}: {value:.6f}")
    
    # Calibration comparison
    print("\n📈 CALIBRATION IMPROVEMENT")
    print("=" * 30)
    
    classical_cal = classical_avg['Calibration_Score']
    sq_cal = sq_avg['Calibration_Score']
    qmc_cal = qmc_avg['Calibration_Score']
    
    print(f"Classical Calibration: {classical_cal:.3f}")
    print(f"Simple Quantum Calibration: {sq_cal:.3f}")
    print(f"QMC Calibration: {qmc_cal:.3f}")
    
    print(f"\nQMC vs Classical: {((qmc_cal - classical_cal) / abs(classical_cal) * 100):+.1f}%")
    print(f"QMC vs Simple Quantum: {((qmc_cal - sq_cal) / abs(sq_cal) * 100):+.1f}%")
    
    # Reliability comparison
    print("\n🔍 RELIABILITY COMPARISON")
    print("=" * 30)
    
    classical_rel = classical_avg['Reliability_Score']
    sq_rel = sq_avg['Reliability_Score']
    qmc_rel = qmc_avg['Reliability_Score']
    
    print(f"Classical Reliability: {classical_rel:.3f}")
    print(f"Simple Quantum Reliability: {sq_rel:.3f}")
    print(f"QMC Reliability: {qmc_rel:.3f}")
    
    # Sharpness comparison
    print("\n⚡ SHARPNESS COMPARISON")
    print("=" * 30)
    
    classical_sharp = classical_avg['Sharpness']
    sq_sharp = sq_avg['Sharpness']
    qmc_sharp = qmc_avg['Sharpness']
    
    print(f"Classical Sharpness: {classical_sharp:.6f}")
    print(f"Simple Quantum Sharpness: {sq_sharp:.6f}")
    print(f"QMC Sharpness: {qmc_sharp:.6f}")
    
    # Conclusion
    print("\n🎯 CONCLUSION")
    print("=" * 15)
    
    if qmc_cal > classical_cal and qmc_cal > sq_cal:
        print("✅ QMC provides the BEST uncertainty calibration!")
    elif classical_cal > qmc_cal and classical_cal > sq_cal:
        print("✅ Classical uncertainty is still the best")
    else:
        print("✅ Simple quantum uncertainty is the best")
    
    if qmc_rel > classical_rel and qmc_rel > sq_rel:
        print("✅ QMC provides the BEST reliability!")
    elif classical_rel > qmc_rel and classical_rel > sq_rel:
        print("✅ Classical uncertainty is most reliable")
    else:
        print("✅ Simple quantum uncertainty is most reliable")
    
    print("\n💡 QMC should provide better uncertainty calibration by sampling from quantum distributions!")

if __name__ == "__main__":
    main()


