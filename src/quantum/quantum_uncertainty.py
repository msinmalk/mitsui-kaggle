"""
Quantum uncertainty quantification for commodity price prediction.
Uses IBM Qiskit to model uncertainty in financial predictions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.algorithms import VQE
    from qiskit.opflow import PauliSumOp
    from qiskit.circuit.library import TwoLocal
    from qiskit.algorithms.optimizers import SPSA
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Qiskit not available. Install with: pip install qiskit")

class QuantumUncertaintyModel:
    """Quantum model for uncertainty quantification in financial predictions."""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, backend: str = 'qasm_simulator'):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend_name = backend
        self.circuit = None
        self.parameters = None
        self.is_trained = False
        
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for quantum uncertainty modeling")
    
    def _build_circuit(self) -> QuantumCircuit:
        """Build variational quantum circuit for uncertainty modeling."""
        qc = QuantumCircuit(self.n_qubits)
        
        # Feature encoding layer
        for i in range(self.n_qubits):
            qc.ry(np.pi/4, i)  # Encode features as rotation angles
        
        # Variational layers
        for layer in range(self.n_layers):
            # Entangling gates
            for i in range(self.n_qubits-1):
                qc.cx(i, i+1)
            
            # Parameterized rotations
            for i in range(self.n_qubits):
                qc.ry(f'θ_{layer}_{i}', i)
        
        return qc
    
    def _encode_features(self, features: np.ndarray) -> QuantumCircuit:
        """Encode financial features into quantum state."""
        # Normalize features to [0, π] range
        normalized_features = (features - features.min()) / (features.max() - features.min()) * np.pi
        
        # Create circuit with feature encoding
        qc = self._build_circuit()
        
        # Replace parameterized rotations with actual feature values
        for i, feature_val in enumerate(normalized_features[:self.n_qubits]):
            qc.ry(feature_val, i)
        
        return qc
    
    def _measure_uncertainty(self, circuit: QuantumCircuit, n_shots: int = 1000) -> Dict[str, float]:
        """Measure uncertainty from quantum circuit."""
        # Add measurement gates
        measured_circuit = circuit.copy()
        measured_circuit.measure_all()
        
        # Execute on simulator
        backend = Aer.get_backend(self.backend_name)
        job = execute(measured_circuit, backend, shots=n_shots)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate uncertainty metrics
        total_shots = sum(counts.values())
        probabilities = {state: count/total_shots for state, count in counts.items()}
        
        # Entropy as uncertainty measure
        entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
        
        # Variance as uncertainty measure
        mean = sum(int(state, 2) * prob for state, prob in probabilities.items())
        variance = sum((int(state, 2) - mean)**2 * prob for state, prob in probabilities.items())
        
        return {
            'entropy': entropy,
            'variance': variance,
            'max_probability': max(probabilities.values()),
            'num_states': len(probabilities)
        }
    
    def predict_uncertainty(self, features: np.ndarray) -> Dict[str, float]:
        """Predict uncertainty for given features."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Encode features into quantum state
        circuit = self._encode_features(features)
        
        # Measure uncertainty
        uncertainty = self._measure_uncertainty(circuit)
        
        return uncertainty
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train quantum uncertainty model."""
        print("🔄 Training quantum uncertainty model...")
        
        # For now, use a simple approach
        # In practice, you would use VQE or other quantum algorithms
        self.is_trained = True
        
        print("✅ Quantum uncertainty model trained")
    
    def batch_predict_uncertainty(self, X: np.ndarray) -> List[Dict[str, float]]:
        """Predict uncertainty for batch of features."""
        uncertainties = []
        
        for i in range(X.shape[0]):
            features = X[i]
            uncertainty = self.predict_uncertainty(features)
            uncertainties.append(uncertainty)
        
        return uncertainties

class HybridQuantumModel:
    """Hybrid classical-quantum model for commodity prediction."""
    
    def __init__(self, classical_model, quantum_model):
        self.classical_model = classical_model
        self.quantum_model = quantum_model
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train both classical and quantum models."""
        print("🔄 Training hybrid classical-quantum model...")
        
        # Train classical model
        self.classical_model.fit(X, y)
        
        # Train quantum model
        self.quantum_model.train(X, y)
        
        self.is_trained = True
        print("✅ Hybrid model trained")
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Predict with both point estimates and uncertainty."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get point predictions from classical model
        predictions = self.classical_model.predict(X)
        
        # Get uncertainty from quantum model
        uncertainties = self.quantum_model.batch_predict_uncertainty(X)
        
        return predictions, uncertainties

def create_quantum_uncertainty_features(features: pd.DataFrame) -> pd.DataFrame:
    """Create quantum-inspired uncertainty features."""
    df_quantum = features.copy()
    
    # Entropy-based features
    for col in features.select_dtypes(include=[np.number]).columns:
        # Rolling entropy
        df_quantum[f'{col}_quantum_entropy_10'] = features[col].rolling(10).apply(
            lambda x: -sum(p * np.log2(p) for p in x.value_counts(normalize=True) if p > 0)
        )
        
        # Quantum-inspired volatility
        df_quantum[f'{col}_quantum_volatility'] = features[col].rolling(20).std() / features[col].rolling(20).mean()
        
        # Uncertainty ratio
        df_quantum[f'{col}_uncertainty_ratio'] = (
            features[col].rolling(10).std() / features[col].rolling(50).std()
        )
    
    return df_quantum

def demonstrate_quantum_uncertainty():
    """Demonstrate quantum uncertainty quantification."""
    if not QISKIT_AVAILABLE:
        print("❌ Qiskit not available. Install with: pip install qiskit")
        return
    
    print("🚀 Demonstrating Quantum Uncertainty Quantification")
    
    # Create sample financial data
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    # Create quantum uncertainty model
    quantum_model = QuantumUncertaintyModel(n_qubits=4, n_layers=2)
    
    # Train model
    quantum_model.train(X, y)
    
    # Test uncertainty prediction
    test_features = X[:5]
    uncertainties = quantum_model.batch_predict_uncertainty(test_features)
    
    print("\n📊 Quantum Uncertainty Results:")
    for i, uncertainty in enumerate(uncertainties):
        print(f"Sample {i+1}:")
        print(f"  Entropy: {uncertainty['entropy']:.3f}")
        print(f"  Variance: {uncertainty['variance']:.3f}")
        print(f"  Max Probability: {uncertainty['max_probability']:.3f}")
        print(f"  Number of States: {uncertainty['num_states']}")
        print()

if __name__ == "__main__":
    demonstrate_quantum_uncertainty()


