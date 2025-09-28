"""
Quantum-enhanced model with real Qiskit integration for uncertainty quantification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import sys
from pathlib import Path

# Add quantum module to path
sys.path.append(str(Path(__file__).parent.parent / 'quantum'))

try:
    from quantum.real_quantum_uncertainty import RealQuantumUncertaintyModel, QuantumVariationalUncertainty, create_quantum_uncertainty_features
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    print("⚠️ Quantum modules not available. Install Qiskit for quantum uncertainty.")

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

class QuantumEnhancedModel:
    """Quantum-enhanced model with real quantum uncertainty quantification."""
    
    def __init__(self, use_quantum_uncertainty: bool = True, n_qubits: int = 4):
        self.use_quantum_uncertainty = use_quantum_uncertainty and QUANTUM_AVAILABLE
        self.n_qubits = n_qubits
        
        # Classical models
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Quantum uncertainty models
        self.quantum_uncertainty_model = None
        self.variational_uncertainty_model = None
        
        # Training state
        self.is_trained = False
        self.feature_columns = None
        self.target_column = None
        
        # Initialize quantum models if available
        if self.use_quantum_uncertainty:
            self._initialize_quantum_models()
    
    def _initialize_quantum_models(self):
        """Initialize quantum uncertainty models."""
        if not QUANTUM_AVAILABLE:
            return
        
        try:
            self.quantum_uncertainty_model = RealQuantumUncertaintyModel(
                n_qubits=self.n_qubits, 
                n_layers=2
            )
            self.variational_uncertainty_model = QuantumVariationalUncertainty(
                n_qubits=self.n_qubits, 
                n_layers=2
            )
            print("✅ Quantum uncertainty models initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize quantum models: {e}")
            self.use_quantum_uncertainty = False
    
    def _add_quantum_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add quantum-inspired features to the dataset."""
        if not self.use_quantum_uncertainty:
            return X
        
        try:
            # Create quantum uncertainty features
            X_quantum = create_quantum_uncertainty_features(X)
            print(f"   Added {X_quantum.shape[1] - X.shape[1]} quantum features")
            return X_quantum
        except Exception as e:
            print(f"⚠️ Could not add quantum features: {e}")
            return X
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the quantum-enhanced model."""
        print("🚀 Training Quantum-Enhanced Model")
        print("=" * 40)
        
        # Store feature info
        self.feature_columns = X.columns.tolist()
        self.target_column = y.name
        
        # Add quantum features
        print("🔮 Adding quantum uncertainty features...")
        X_enhanced = self._add_quantum_features(X)
        
        # Prepare data
        X_values = X_enhanced.fillna(method='ffill').fillna(method='bfill').fillna(0).values
        y_values = y.fillna(method='ffill').fillna(method='bfill').fillna(0).values
        
        # Train classical models
        print("🌲 Training Random Forest...")
        self.rf_model.fit(X_values, y_values)
        
        print("🚀 Training XGBoost...")
        self.xgb_model.fit(X_values, y_values)
        
        # Train quantum uncertainty models
        if self.use_quantum_uncertainty:
            print("⚛️ Training quantum uncertainty models...")
            try:
                self.quantum_uncertainty_model.train(X_values, y_values)
                self.variational_uncertainty_model.train(X_values, y_values)
                print("✅ Quantum uncertainty models trained")
            except Exception as e:
                print(f"⚠️ Quantum training failed: {e}")
                self.use_quantum_uncertainty = False
        
        self.is_trained = True
        print("✅ Quantum-enhanced model training completed!")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using ensemble of classical models."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Add quantum features
        X_enhanced = self._add_quantum_features(X)
        X_values = X_enhanced.fillna(method='ffill').fillna(method='bfill').fillna(0).values
        
        # Get predictions from both models
        rf_pred = self.rf_model.predict(X_values)
        xgb_pred = self.xgb_model.predict(X_values)
        
        # Ensemble prediction (weighted average)
        ensemble_pred = 0.5 * rf_pred + 0.5 * xgb_pred
        
        return ensemble_pred
    
    def predict_with_quantum_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Make predictions with quantum uncertainty quantification."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get base predictions
        predictions = self.predict(X)
        
        # Add quantum features
        X_enhanced = self._add_quantum_features(X)
        X_values = X_enhanced.fillna(method='ffill').fillna(method='bfill').fillna(0).values
        
        # Get quantum uncertainty
        quantum_uncertainty = None
        quantum_metrics = {}
        
        if self.use_quantum_uncertainty:
            try:
                # Get quantum uncertainty from both models
                quantum_unc = self.quantum_uncertainty_model.batch_predict_quantum_uncertainty(X_values)
                variational_unc = self.variational_uncertainty_model.predict_uncertainty(X_values)
                
                # Combine quantum uncertainties
                quantum_uncertainty = np.array([q['total_uncertainty'] for q in quantum_unc])
                quantum_uncertainty = 0.5 * quantum_uncertainty + 0.5 * variational_unc
                
                # Store detailed quantum metrics
                quantum_metrics = {
                    'quantum_entropy': np.mean([q['quantum_entropy'] for q in quantum_unc]),
                    'quantum_variance': np.mean([q['quantum_variance'] for q in quantum_unc]),
                    'quantum_coherence': np.mean([q['quantum_coherence'] for q in quantum_unc]),
                    'quantum_superposition': np.mean([q['quantum_superposition'] for q in quantum_unc]),
                    'variational_uncertainty': np.mean(variational_unc)
                }
                
            except Exception as e:
                print(f"⚠️ Quantum uncertainty calculation failed: {e}")
                quantum_uncertainty = np.ones_like(predictions) * 0.1
                quantum_metrics = {'error': str(e)}
        else:
            # Fallback to classical uncertainty
            quantum_uncertainty = np.ones_like(predictions) * 0.1
            quantum_metrics = {'method': 'classical_fallback'}
        
        return predictions, quantum_uncertainty, quantum_metrics
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate the model with quantum uncertainty."""
        predictions, uncertainties, quantum_metrics = self.predict_with_quantum_uncertainty(X)
        
        # Standard metrics
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mse)
        
        # Uncertainty metrics
        avg_uncertainty = np.mean(uncertainties)
        uncertainty_std = np.std(uncertainties)
        
        # Combine all metrics
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Avg_Uncertainty': avg_uncertainty,
            'Uncertainty_Std': uncertainty_std,
            'Quantum_Available': self.use_quantum_uncertainty
        }
        
        # Add quantum metrics if available
        if quantum_metrics:
            metrics.update(quantum_metrics)
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from both models."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        return {
            'RandomForest': self.rf_model.feature_importances_,
            'XGBoost': self.xgb_model.feature_importances_
        }
    
    def get_quantum_status(self) -> Dict[str, bool]:
        """Get status of quantum components."""
        return {
            'quantum_available': QUANTUM_AVAILABLE,
            'quantum_enabled': self.use_quantum_uncertainty,
            'quantum_uncertainty_trained': self.quantum_uncertainty_model is not None and self.quantum_uncertainty_model.is_trained,
            'variational_uncertainty_trained': self.variational_uncertainty_model is not None and self.variational_uncertainty_model.is_trained
        }

def test_quantum_enhanced_model():
    """Test the quantum-enhanced model."""
    print("🔮 Testing Quantum-Enhanced Model")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                    columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.randn(n_samples), name='target')
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = QuantumEnhancedModel(use_quantum_uncertainty=True, n_qubits=4)
    model.train(X_train, y_train)
    
    # Test predictions
    predictions, uncertainties, quantum_metrics = model.predict_with_quantum_uncertainty(X_test)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    print("\n📊 Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n🔮 Quantum Status:")
    quantum_status = model.get_quantum_status()
    for key, value in quantum_status.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Quantum-enhanced model test completed!")

if __name__ == "__main__":
    test_quantum_enhanced_model()


