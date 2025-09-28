"""
ALF-Inspired Quantum Monte Carlo for Financial Markets
=====================================================

This module implements quantum Monte Carlo methods inspired by the ALF
(Algorithms for lattice fermions) project for condensed matter physics,
adapted for commodity price prediction and uncertainty quantification.

Reference: https://git.physik.uni-wuerzburg.de/ALF
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit_aer import Aer
    from qiskit.quantum_info import Statevector, Operator
    QISKIT_AVAILABLE = True
    print("✅ Qiskit available for ALF-inspired QMC")
except ImportError as e:
    QISKIT_AVAILABLE = False
    print(f"❌ Qiskit not available: {e}")

class MarketLattice:
    """Market lattice structure inspired by ALF's lattice fermions."""
    
    def __init__(self, n_time: int, n_exchange: int, n_commodity: int):
        self.n_time = n_time
        self.n_exchange = n_exchange
        self.n_commodity = n_commodity
        
        # Lattice dimensions
        self.dimensions = (n_time, n_exchange, n_commodity)
        
        # Exchange mapping
        self.exchange_map = {
            0: 'LME',    # London Metal Exchange
            1: 'JPX',    # Japan Exchange
            2: 'US',     # US Stock Markets
            3: 'FX'      # Foreign Exchange
        }
        
        # Build connectivity matrix (correlations between lattice sites)
        self.connectivity = self._build_connectivity_matrix()
        
        print(f"🏗️ Market Lattice initialized: {self.dimensions}")
    
    def _build_connectivity_matrix(self) -> np.ndarray:
        """Build connectivity matrix for lattice sites."""
        # 3D connectivity: (time, exchange, commodity)
        connectivity = np.zeros(self.dimensions + self.dimensions)
        
        # Temporal connectivity (time series)
        for t in range(self.n_time - 1):
            for e in range(self.n_exchange):
                for c in range(self.n_commodity):
                    connectivity[t, e, c, t+1, e, c] = 1.0  # Next time step
        
        # Exchange connectivity (cross-exchange correlations)
        for t in range(self.n_time):
            for e1 in range(self.n_exchange):
                for e2 in range(self.n_exchange):
                    if e1 != e2:
                        for c in range(self.n_commodity):
                            connectivity[t, e1, c, t, e2, c] = 0.5  # Cross-exchange
        
        # Commodity connectivity (cross-commodity correlations)
        for t in range(self.n_time):
            for e in range(self.n_exchange):
                for c1 in range(self.n_commodity):
                    for c2 in range(self.n_commodity):
                        if c1 != c2:
                            connectivity[t, e, c1, t, e, c2] = 0.3  # Cross-commodity
        
        return connectivity
    
    def get_neighbors(self, time_idx: int, exchange_idx: int, commodity_idx: int) -> List[Tuple[int, int, int]]:
        """Get neighboring lattice sites."""
        neighbors = []
        
        # Temporal neighbors
        if time_idx > 0:
            neighbors.append((time_idx - 1, exchange_idx, commodity_idx))
        if time_idx < self.n_time - 1:
            neighbors.append((time_idx + 1, exchange_idx, commodity_idx))
        
        # Exchange neighbors
        for e in range(self.n_exchange):
            if e != exchange_idx:
                neighbors.append((time_idx, e, commodity_idx))
        
        # Commodity neighbors (limited to avoid explosion)
        for c in range(min(10, self.n_commodity)):  # Limit to first 10 commodities
            if c != commodity_idx:
                neighbors.append((time_idx, exchange_idx, c))
        
        return neighbors

class FinancialFieldOperators:
    """Financial field operators with meaningful financial interpretations."""
    
    def __init__(self, lattice: MarketLattice):
        self.lattice = lattice
        self.n_qubits = self._calculate_required_qubits()
        
        print(f"⚛️ Financial Field Operators: {self.n_qubits} qubits")
    
    def _calculate_required_qubits(self) -> int:
        """Calculate required qubits for the lattice."""
        # Use log2 of total lattice sites
        total_sites = self.lattice.n_time * self.lattice.n_exchange * self.lattice.n_commodity
        n_qubits = int(np.ceil(np.log2(total_sites)))
        return min(n_qubits, 20)  # Limit to 20 qubits for practicality
    
    def create_financial_operators(self) -> Dict[str, np.ndarray]:
        """Create financially meaningful quantum operators."""
        n_qubits = self.n_qubits
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        identity = np.array([[1, 0], [0, 1]])
        
        operators = {}
        
        # 1. Price Movement Operators
        for i in range(n_qubits):
            # Price increase operator (|↑⟩)
            op_up = np.eye(1)
            for j in range(n_qubits):
                if j == i:
                    op_up = np.kron(op_up, (identity + sigma_z) / 2)  # |1⟩⟨1|
                else:
                    op_up = np.kron(op_up, identity)
            operators[f'price_increase_{i}'] = op_up
            
            # Price decrease operator (|↓⟩)
            op_down = np.eye(1)
            for j in range(n_qubits):
                if j == i:
                    op_down = np.kron(op_down, (identity - sigma_z) / 2)  # |0⟩⟨0|
                else:
                    op_down = np.kron(op_down, identity)
            operators[f'price_decrease_{i}'] = op_down
        
        # 2. Market Regime Operators
        for i in range(min(3, n_qubits)):  # Bull, Bear, Sideways
            # Bull market operator
            op_bull = np.eye(1)
            for j in range(n_qubits):
                if j == i:
                    op_bull = np.kron(op_bull, (identity + sigma_x) / 2)  # |+⟩⟨+|
                else:
                    op_bull = np.kron(op_bull, identity)
            operators[f'bull_market_{i}'] = op_bull
            
            # Bear market operator
            op_bear = np.eye(1)
            for j in range(n_qubits):
                if j == i:
                    op_bear = np.kron(op_bear, (identity - sigma_x) / 2)  # |-⟩⟨-|
                else:
                    op_bear = np.kron(op_bear, identity)
            operators[f'bear_market_{i}'] = op_bear
        
        # 3. Volatility Operators
        for i in range(min(2, n_qubits)):  # High, Low volatility
            # High volatility operator
            op_high_vol = np.eye(1)
            for j in range(n_qubits):
                if j == i:
                    op_high_vol = np.kron(op_high_vol, (identity + sigma_y) / 2)  # |i⟩⟨i|
                else:
                    op_high_vol = np.kron(op_high_vol, identity)
            operators[f'high_volatility_{i}'] = op_high_vol
            
            # Low volatility operator
            op_low_vol = np.eye(1)
            for j in range(n_qubits):
                if j == i:
                    op_low_vol = np.kron(op_low_vol, (identity - sigma_y) / 2)  # |-i⟩⟨-i|
                else:
                    op_low_vol = np.kron(op_low_vol, identity)
            operators[f'low_volatility_{i}'] = op_low_vol
        
        return operators
    
    def create_market_hamiltonian(self, price_data: np.ndarray) -> np.ndarray:
        """Create market Hamiltonian from price data."""
        n_qubits = self.n_qubits
        
        # Initialize Hamiltonian
        hamiltonian = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
        
        # Kinetic term (price momentum)
        for i in range(n_qubits):
            # Price momentum operator
            momentum_op = self._create_momentum_operator(i)
            hamiltonian += 0.5 * momentum_op
        
        # Potential term (price interactions)
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                # Interaction between qubits i and j
                interaction_op = self._create_interaction_operator(i, j, price_data)
                hamiltonian += interaction_op
        
        return hamiltonian
    
    def _create_momentum_operator(self, qubit_idx: int) -> np.ndarray:
        """Create momentum operator for a specific qubit."""
        n_qubits = self.n_qubits
        sigma_x = np.array([[0, 1], [1, 0]])
        identity = np.array([[1, 0], [0, 1]])
        
        op = np.eye(1)
        for j in range(n_qubits):
            if j == qubit_idx:
                op = np.kron(op, sigma_x)
            else:
                op = np.kron(op, identity)
        
        return op
    
    def _create_interaction_operator(self, i: int, j: int, price_data: np.ndarray) -> np.ndarray:
        """Create interaction operator between qubits i and j."""
        n_qubits = self.n_qubits
        sigma_z = np.array([[1, 0], [0, -1]])
        identity = np.array([[1, 0], [0, 1]])
        
        # Interaction strength based on price correlation
        if len(price_data) > max(i, j):
            interaction_strength = np.corrcoef(price_data[i], price_data[j])[0, 1]
        else:
            interaction_strength = 0.1
        
        op = np.eye(1)
        for k in range(n_qubits):
            if k == i or k == j:
                op = np.kron(op, sigma_z)
            else:
                op = np.kron(op, identity)
        
        return interaction_strength * op

class ALFInspiredQMC:
    """ALF-inspired Quantum Monte Carlo for financial markets."""
    
    def __init__(self, lattice: MarketLattice, n_samples: int = 1000, n_circuits: int = 10):
        self.lattice = lattice
        self.n_samples = n_samples
        self.n_circuits = n_circuits
        
        # Initialize components
        self.field_operators = FinancialFieldOperators(lattice)
        self.operators = self.field_operators.create_financial_operators()
        
        # Quantum backend
        if QISKIT_AVAILABLE:
            self.backend = Aer.get_backend('qasm_simulator')
        else:
            self.backend = None
        
        # Monte Carlo state
        self.is_trained = False
        self.hamiltonian = None
        
        print(f"🔬 ALF-Inspired QMC initialized")
        print(f"   Lattice: {lattice.dimensions}")
        print(f"   Samples: {n_samples}")
        print(f"   Circuits: {n_circuits}")
    
    def _create_quantum_circuit(self, price_data: np.ndarray, circuit_idx: int) -> QuantumCircuit:
        """Create quantum circuit inspired by ALF's lattice structure."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for quantum circuits")
        
        n_qubits = self.field_operators.n_qubits
        
        # Create quantum circuit
        qr = QuantumRegister(n_qubits, 'q')
        cr = ClassicalRegister(n_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # Encode price data into quantum state
        self._encode_price_data(qc, price_data, circuit_idx)
        
        # Apply lattice interactions
        self._apply_lattice_interactions(qc)
        
        # Apply variational layers
        self._apply_variational_layers(qc, circuit_idx)
        
        return qc
    
    def _encode_price_data(self, qc: QuantumCircuit, price_data: np.ndarray, circuit_idx: int):
        """Encode price data into quantum state."""
        n_qubits = self.field_operators.n_qubits
        
        # Normalize price data
        if len(price_data) > 0:
            normalized_data = (price_data - np.mean(price_data)) / (np.std(price_data) + 1e-8)
        else:
            normalized_data = np.random.randn(n_qubits)
        
        # Encode as rotation angles
        for i in range(min(n_qubits, len(normalized_data))):
            angle = normalized_data[i] * np.pi / 4  # Scale to [-π/4, π/4]
            qc.ry(angle, i)
    
    def _apply_lattice_interactions(self, qc: QuantumCircuit):
        """Apply lattice interactions inspired by ALF."""
        n_qubits = self.field_operators.n_qubits
        
        # Nearest neighbor interactions
        for i in range(n_qubits - 1):
            # Ising interaction (ZZ coupling)
            qc.cx(i, i + 1)
            qc.rz(np.pi / 4, i + 1)
            qc.cx(i, i + 1)
        
        # Long-range interactions (every other qubit)
        for i in range(0, n_qubits - 2, 2):
            qc.cx(i, i + 2)
            qc.rz(np.pi / 8, i + 2)
            qc.cx(i, i + 2)
    
    def _apply_variational_layers(self, qc: QuantumCircuit, circuit_idx: int):
        """Apply variational layers with random parameters."""
        n_qubits = self.field_operators.n_qubits
        
        # Random seed for reproducibility
        np.random.seed(42 + circuit_idx)
        
        # Variational layers
        for layer in range(3):
            # Single qubit rotations
            for i in range(n_qubits):
                qc.ry(np.random.uniform(0, 2*np.pi), i)
                qc.rz(np.random.uniform(0, 2*np.pi), i)
            
            # Entangling gates
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
    
    def _quantum_monte_carlo_step(self, price_data: np.ndarray) -> Dict[str, float]:
        """Single quantum Monte Carlo step."""
        if not QISKIT_AVAILABLE:
            # Fallback to classical Monte Carlo
            return self._classical_monte_carlo_step(price_data)
        
        # Create multiple quantum circuits
        circuits = []
        for i in range(self.n_circuits):
            circuit = self._create_quantum_circuit(price_data, i)
            circuit.measure_all()
            circuits.append(circuit)
        
        # Run quantum circuits
        job = self.backend.run(circuits, shots=self.n_samples)
        result = job.result()
        
        # Process results
        all_counts = []
        for i, circuit in enumerate(circuits):
            counts = result.get_counts(circuit)
            all_counts.append(counts)
        
        # Calculate quantum metrics
        return self._calculate_quantum_metrics(all_counts, price_data)
    
    def _classical_monte_carlo_step(self, price_data: np.ndarray) -> Dict[str, float]:
        """Classical Monte Carlo fallback."""
        # Sample from normal distribution
        samples = np.random.normal(np.mean(price_data), np.std(price_data), self.n_samples)
        
        return {
            'qmc_mean': np.mean(samples),
            'qmc_std': np.std(samples),
            'qmc_var': np.var(samples),
            'qmc_entropy': -np.sum(samples * np.log(np.abs(samples) + 1e-8)),
            'qmc_uncertainty': np.std(samples),
            'method': 'classical_fallback'
        }
    
    def _calculate_quantum_metrics(self, all_counts: List[Dict], price_data: np.ndarray) -> Dict[str, float]:
        """Calculate quantum metrics from measurement counts."""
        # Combine all measurement results
        combined_counts = {}
        total_shots = 0
        
        for counts in all_counts:
            for state, count in counts.items():
                if state in combined_counts:
                    combined_counts[state] += count
                else:
                    combined_counts[state] += count
                total_shots += count
        
        # Convert to probabilities
        probabilities = {state: count/total_shots for state, count in combined_counts.items()}
        
        # Calculate quantum metrics
        states = list(probabilities.keys())
        probs = list(probabilities.values())
        
        # Quantum entropy
        entropy = -np.sum(probs * np.log2(np.array(probs) + 1e-8))
        
        # Quantum variance
        state_values = [int(state, 2) for state in states]
        mean_value = np.sum(np.array(state_values) * np.array(probs))
        variance = np.sum(np.array(probs) * (np.array(state_values) - mean_value)**2)
        
        # Quantum uncertainty
        uncertainty = np.sqrt(variance)
        
        return {
            'qmc_mean': mean_value,
            'qmc_std': np.sqrt(variance),
            'qmc_var': variance,
            'qmc_entropy': entropy,
            'qmc_uncertainty': uncertainty,
            'qmc_coherence': self._calculate_coherence(probabilities),
            'qmc_superposition': self._calculate_superposition(probabilities),
            'method': 'quantum'
        }
    
    def _calculate_coherence(self, probabilities: Dict[str, float]) -> float:
        """Calculate quantum coherence measure."""
        # Coherence as off-diagonal elements
        n_qubits = self.field_operators.n_qubits
        coherence = 0.0
        
        for state, prob in probabilities.items():
            if prob > 0:
                # Count superposition states (non-classical states)
                if state.count('1') > 0 and state.count('0') > 0:
                    coherence += prob
        
        return coherence
    
    def _calculate_superposition(self, probabilities: Dict[str, float]) -> float:
        """Calculate quantum superposition measure."""
        # Superposition as entropy of the distribution
        probs = list(probabilities.values())
        probs = np.array(probs)
        probs = probs[probs > 0]
        
        if len(probs) > 1:
            superposition = -np.sum(probs * np.log2(probs))
        else:
            superposition = 0.0
        
        return superposition
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the ALF-inspired QMC model."""
        print("🔄 Training ALF-inspired QMC model...")
        
        # Create market Hamiltonian
        self.hamiltonian = self.field_operators.create_market_hamiltonian(X.flatten())
        
        self.is_trained = True
        print("✅ ALF-inspired QMC model trained")
    
    def predict_uncertainty(self, features: np.ndarray) -> Dict[str, float]:
        """Predict uncertainty using ALF-inspired QMC."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self._quantum_monte_carlo_step(features)
    
    def batch_predict_uncertainty(self, X: np.ndarray) -> List[Dict[str, float]]:
        """Predict uncertainty for batch of features."""
        uncertainties = []
        
        for i in range(X.shape[0]):
            features = X[i]
            uncertainty = self.predict_uncertainty(features)
            uncertainties.append(uncertainty)
        
        return uncertainties

def demonstrate_alf_inspired_qmc():
    """Demonstrate ALF-inspired QMC on sample data."""
    print("🔬 ALF-Inspired Quantum Monte Carlo Demonstration")
    print("=" * 60)
    
    # Create market lattice
    lattice = MarketLattice(n_time=100, n_exchange=4, n_commodity=10)
    
    # Initialize QMC
    qmc = ALFInspiredQMC(lattice, n_samples=500, n_circuits=5)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 50
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    # Train model
    qmc.train(X, y)
    
    # Test uncertainty prediction
    test_features = X[:5]
    uncertainties = qmc.batch_predict_uncertainty(test_features)
    
    print("\n📊 ALF-Inspired QMC Results:")
    for i, uncertainty in enumerate(uncertainties):
        print(f"\nSample {i+1}:")
        for key, value in uncertainty.items():
            print(f"  {key}: {value:.6f}")
    
    print("\n✅ ALF-inspired QMC demonstration completed!")
    return qmc, uncertainties

if __name__ == "__main__":
    qmc, uncertainties = demonstrate_alf_inspired_qmc()
