"""
Compare quantum-enhanced models vs classical models to see if quantum uncertainty helps.
"""

import sys
sys.path.append('./src')

from data.loader import CommodityDataLoader
from data.feature_engineering_simple import SimpleFeatureEngineer
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
    """Pure classical model without quantum uncertainty."""
    
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        self.xgb_model = xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        self.is_trained = False
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train classical models."""
        print("🌲 Training Classical Models...")
        
        # Prepare data
        X_clean = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        X_values = X_clean.values
        y_values = y.fillna(method='ffill').fillna(method='bfill').fillna(0).values
        
        # Train models
        self.rf_model.fit(X_values, y_values)
        self.xgb_model.fit(X_values, y_values)
        self.is_trained = True
        print("✅ Classical models trained")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions and return classical uncertainty."""
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

class QuantumEnhancedModel:
    """Quantum-enhanced model with real quantum uncertainty."""
    
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        self.xgb_model = xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        self.quantum_uncertainty_model = WorkingQuantumUncertainty(n_qubits=n_qubits)
        self.is_trained = False
    
    def _add_quantum_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add quantum-inspired features."""
        df_quantum = X.copy()
        
        # Add limited quantum features for a few columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns[:5]
        
        for col in numeric_cols:
            df_quantum[f'{col}_quantum_entropy'] = X[col].rolling(5).apply(
                lambda x: -sum(p * np.log2(p) for p in x.value_counts(normalize=True) if p > 0) if len(x) > 0 else 0
            )
            df_quantum[f'{col}_quantum_variance'] = X[col].rolling(5).var()
        
        df_quantum = df_quantum.replace([np.inf, -np.inf], 0)
        return df_quantum
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train quantum-enhanced models."""
        print("🔮 Training Quantum-Enhanced Models...")
        
        # Add quantum features
        X_enhanced = self._add_quantum_features(X)
        
        # Prepare data
        X_clean = X_enhanced.fillna(method='ffill').fillna(method='bfill').fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        X_values = X_clean.values
        y_values = y.fillna(method='ffill').fillna(method='bfill').fillna(0).values
        
        # Train classical models
        self.rf_model.fit(X_values, y_values)
        self.xgb_model.fit(X_values, y_values)
        
        # Train quantum uncertainty model
        self.quantum_uncertainty_model.train(X_values, y_values)
        
        self.is_trained = True
        print("✅ Quantum-enhanced models trained")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Make predictions with quantum uncertainty."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Add quantum features
        X_enhanced = self._add_quantum_features(X)
        X_clean = X_enhanced.fillna(method='ffill').fillna(method='bfill').fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        X_values = X_clean.values
        
        # Get classical predictions
        rf_pred = self.rf_model.predict(X_values)
        xgb_pred = self.xgb_model.predict(X_values)
        ensemble_pred = 0.5 * rf_pred + 0.5 * xgb_pred
        
        # Get quantum uncertainty
        try:
            quantum_unc = self.quantum_uncertainty_model.batch_predict_uncertainty(X_values)
            quantum_uncertainty = np.array([q['total_uncertainty'] for q in quantum_unc])
            
            quantum_metrics = {
                'quantum_entropy': np.mean([q['quantum_entropy'] for q in quantum_unc]),
                'quantum_variance': np.mean([q['quantum_variance'] for q in quantum_unc]),
                'quantum_coherence': np.mean([q['quantum_coherence'] for q in quantum_unc]),
                'quantum_superposition': np.mean([q['quantum_superposition'] for q in quantum_unc])
            }
        except Exception as e:
            print(f"⚠️ Quantum uncertainty failed: {e}")
            quantum_uncertainty = np.ones_like(ensemble_pred) * 0.1
            quantum_metrics = {'error': str(e)}
        
        return ensemble_pred, quantum_uncertainty, quantum_metrics

def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, uncertainty: np.ndarray, model_name: str) -> Dict[str, float]:
    """Evaluate model performance."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Uncertainty metrics
    avg_uncertainty = np.mean(uncertainty)
    uncertainty_std = np.std(uncertainty)
    
    # Calibration (how well uncertainty correlates with actual error)
    actual_errors = np.abs(y_true - y_pred)
    uncertainty_correlation = np.corrcoef(actual_errors, uncertainty)[0, 1]
    
    return {
        'model': model_name,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Avg_Uncertainty': avg_uncertainty,
        'Uncertainty_Std': uncertainty_std,
        'Uncertainty_Correlation': uncertainty_correlation
    }

def main():
    print("🔬 Quantum vs Classical Model Comparison")
    print("========================================")
    
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
    num_targets = 5
    sample_targets = target_cols[:num_targets]
    
    classical_results = []
    quantum_results = []
    
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
        classical_pred, classical_unc = classical_model.predict(X_test)
        classical_metrics = evaluate_model(y_test, classical_pred, classical_unc, "Classical")
        classical_results.append(classical_metrics)
        
        # Test Quantum Model
        print("🔮 Testing Quantum-Enhanced Model...")
        quantum_model = QuantumEnhancedModel(n_qubits=4)
        quantum_model.train(X_train, y_train)
        quantum_pred, quantum_unc, quantum_metrics = quantum_model.predict(X_test)
        quantum_eval = evaluate_model(y_test, quantum_pred, quantum_unc, "Quantum")
        quantum_eval.update(quantum_metrics)
        quantum_results.append(quantum_eval)
        
        print(f"  Classical MSE: {classical_metrics['MSE']:.6f}")
        print(f"  Quantum MSE: {quantum_eval['MSE']:.6f}")
        print(f"  Classical Uncertainty: {classical_metrics['Avg_Uncertainty']:.6f}")
        print(f"  Quantum Uncertainty: {quantum_eval['Avg_Uncertainty']:.6f}")
    
    # Summary comparison
    print("\n📊 COMPARISON SUMMARY")
    print("=" * 50)
    
    # Average metrics
    classical_avg = {key: np.mean([r[key] for r in classical_results]) for key in classical_results[0].keys() if key != 'model'}
    quantum_avg = {key: np.mean([r[key] for r in quantum_results]) for key in quantum_results[0].keys() if key != 'model'}
    
    print("\n🌲 Classical Model (Average):")
    for key, value in classical_avg.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n🔮 Quantum-Enhanced Model (Average):")
    for key, value in quantum_avg.items():
        print(f"  {key}: {value:.6f}")
    
    # Performance comparison
    print("\n📈 PERFORMANCE COMPARISON")
    print("=" * 30)
    
    mse_improvement = (classical_avg['MSE'] - quantum_avg['MSE']) / classical_avg['MSE'] * 100
    rmse_improvement = (classical_avg['RMSE'] - quantum_avg['RMSE']) / classical_avg['RMSE'] * 100
    r2_improvement = (quantum_avg['R2'] - classical_avg['R2']) / abs(classical_avg['R2']) * 100
    
    print(f"MSE Improvement: {mse_improvement:+.2f}%")
    print(f"RMSE Improvement: {rmse_improvement:+.2f}%")
    print(f"R² Improvement: {r2_improvement:+.2f}%")
    
    # Uncertainty comparison
    print("\n⚛️ UNCERTAINTY COMPARISON")
    print("=" * 30)
    print(f"Classical Uncertainty: {classical_avg['Avg_Uncertainty']:.6f}")
    print(f"Quantum Uncertainty: {quantum_avg['Avg_Uncertainty']:.6f}")
    print(f"Classical Uncertainty Correlation: {classical_avg['Uncertainty_Correlation']:.6f}")
    print(f"Quantum Uncertainty Correlation: {quantum_avg['Uncertainty_Correlation']:.6f}")
    
    # Quantum metrics
    print("\n🔬 QUANTUM METRICS")
    print("=" * 20)
    print(f"Quantum Entropy: {quantum_avg.get('quantum_entropy', 0):.6f}")
    print(f"Quantum Variance: {quantum_avg.get('quantum_variance', 0):.6f}")
    print(f"Quantum Coherence: {quantum_avg.get('quantum_coherence', 0):.6f}")
    print(f"Quantum Superposition: {quantum_avg.get('quantum_superposition', 0):.6f}")
    
    # Conclusion
    print("\n🎯 CONCLUSION")
    print("=" * 15)
    if mse_improvement > 0:
        print(f"✅ Quantum-enhanced model performs BETTER by {mse_improvement:.2f}%")
    else:
        print(f"❌ Classical model performs better by {abs(mse_improvement):.2f}%")
    
    if quantum_avg['Uncertainty_Correlation'] > classical_avg['Uncertainty_Correlation']:
        print("✅ Quantum uncertainty is more calibrated with actual errors")
    else:
        print("❌ Classical uncertainty is more calibrated")
    
    print("\n💡 This comparison shows whether quantum uncertainty actually helps!")

if __name__ == "__main__":
    main()


