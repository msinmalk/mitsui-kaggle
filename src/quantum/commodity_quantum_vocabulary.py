"""
Commodity-Specific Quantum Vocabulary
Adapted from FOREX inspiration to our multi-dimensional commodity challenge.

Our challenge is fundamentally different from FOREX:
- FOREX: Single currency pair, time series
- COMMODITY: Multi-dimensional (time, exchange, commodity) with 424 targets

This adapts the FOREX quantum vocabulary to our specific needs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector, Operator
    QISKIT_AVAILABLE = True
    print("✅ Qiskit available for Commodity Quantum Vocabulary")
except ImportError as e:
    QISKIT_AVAILABLE = False
    print(f"❌ Qiskit not available: {e}")

class CommodityQuantumVocabulary:
    """
    Quantum vocabulary adapted for multi-dimensional commodity prediction.
    
    Key differences from FOREX:
    1. Multi-dimensional state space: (time, exchange, commodity)
    2. 424 target variables vs single currency pair
    3. Cross-commodity correlations vs single exchange rate
    4. Multi-exchange dynamics vs single market
    """
    
    def __init__(self, n_time: int = 1961, n_exchange: int = 4, n_commodity: int = 424):
        self.n_time = n_time
        self.n_exchange = n_exchange
        self.n_commodity = n_commodity
        
        # Exchange mapping
        self.exchange_map = {
            0: 'LME',    # London Metal Exchange (metals)
            1: 'JPX',    # Japan Exchange (precious metals, rubber)
            2: 'US',     # US Stock Markets (100+ stocks/ETFs)
            3: 'FX'      # Foreign Exchange (50+ currency pairs)
        }
        
        # Commodity categories
        self.commodity_categories = self._define_commodity_categories()
        
        # Multi-dimensional quantum states
        self.market_states = self._define_multi_dimensional_states()
        
        # Cross-dimensional operators
        self.cross_operators = self._create_cross_dimensional_operators()
        
        print(f"🌾 Commodity Quantum Vocabulary initialized")
        print(f"   Dimensions: Time({n_time}) × Exchange({n_exchange}) × Commodity({n_commodity})")
    
    def _define_commodity_categories(self) -> Dict[str, List[int]]:
        """Define commodity categories for quantum state organization."""
        categories = {
            'metals': list(range(0, 50)),      # LME metals
            'precious': list(range(50, 100)),  # Gold, Platinum
            'stocks': list(range(100, 200)),   # US stocks
            'etfs': list(range(200, 300)),     # US ETFs
            'currencies': list(range(300, 350)), # FX pairs
            'spreads': list(range(350, 424))   # Cross-asset spreads
        }
        return categories
    
    def _define_multi_dimensional_states(self) -> Dict[str, np.ndarray]:
        """
        Define multi-dimensional quantum states for commodity markets.
        
        Unlike FOREX single currency pair, we have:
        - Time dimension: Market evolution over time
        - Exchange dimension: Cross-market correlations
        - Commodity dimension: Cross-commodity relationships
        """
        states = {}
        
        # Ground state: Stable market across all dimensions
        ground_state = np.zeros((self.n_time, self.n_exchange, self.n_commodity))
        ground_state[:, :, :] = 0.0  # All prices stable
        states['ground_state'] = ground_state
        
        # Excited states: Volatile market conditions
        # Time-volatile state: High volatility over time
        time_volatile = np.zeros((self.n_time, self.n_exchange, self.n_commodity))
        time_volatile[:, :, :] = np.random.normal(0, 0.1, time_volatile.shape)
        states['time_volatile'] = time_volatile
        
        # Exchange-correlated state: High cross-exchange correlation
        exchange_correlated = np.zeros((self.n_time, self.n_exchange, self.n_commodity))
        for t in range(self.n_time):
            for e in range(self.n_exchange):
                exchange_correlated[t, e, :] = np.random.normal(0, 0.05) + 0.1 * e
        states['exchange_correlated'] = exchange_correlated
        
        # Commodity-correlated state: High cross-commodity correlation
        commodity_correlated = np.zeros((self.n_time, self.n_exchange, self.n_commodity))
        for t in range(self.n_time):
            for c in range(self.n_commodity):
                commodity_correlated[t, :, c] = np.random.normal(0, 0.03) + 0.05 * c
        states['commodity_correlated'] = commodity_correlated
        
        # Crisis state: High volatility across all dimensions
        crisis_state = np.zeros((self.n_time, self.n_exchange, self.n_commodity))
        crisis_state[:, :, :] = np.random.normal(0, 0.2, crisis_state.shape)
        states['crisis_state'] = crisis_state
        
        return states
    
    def _create_cross_dimensional_operators(self) -> Dict[str, np.ndarray]:
        """
        Create operators for cross-dimensional interactions.
        
        Unlike FOREX single transition, we need:
        - Time evolution operators
        - Exchange correlation operators  
        - Commodity correlation operators
        - Cross-dimensional coupling operators
        """
        operators = {}
        
        # Time evolution operator: H_time
        # Captures temporal dynamics of commodity prices
        time_operator = np.zeros((self.n_time, self.n_time))
        for t in range(self.n_time - 1):
            time_operator[t, t+1] = 1.0  # Forward time evolution
            time_operator[t+1, t] = 0.5  # Backward time evolution
        operators['time_evolution'] = time_operator
        
        # Exchange correlation operator: H_exchange
        # Captures cross-exchange correlations
        exchange_operator = np.zeros((self.n_exchange, self.n_exchange))
        for e1 in range(self.n_exchange):
            for e2 in range(self.n_exchange):
                if e1 != e2:
                    exchange_operator[e1, e2] = 0.3  # Cross-exchange correlation
                else:
                    exchange_operator[e1, e2] = 1.0  # Self-correlation
        operators['exchange_correlation'] = exchange_operator
        
        # Commodity correlation operator: H_commodity
        # Captures cross-commodity relationships
        commodity_operator = np.zeros((self.n_commodity, self.n_commodity))
        for c1 in range(self.n_commodity):
            for c2 in range(self.n_commodity):
                if c1 != c2:
                    # Higher correlation for same category
                    if self._same_category(c1, c2):
                        commodity_operator[c1, c2] = 0.5
                    else:
                        commodity_operator[c1, c2] = 0.1
                else:
                    commodity_operator[c1, c2] = 1.0
        operators['commodity_correlation'] = commodity_operator
        
        # Cross-dimensional coupling: H_coupling
        # Captures interactions between dimensions
        coupling_operator = np.zeros((self.n_time * self.n_exchange * self.n_commodity, 
                                    self.n_time * self.n_exchange * self.n_commodity))
        # This would be a very large matrix - simplified for demonstration
        operators['cross_dimensional_coupling'] = coupling_operator
        
        return operators
    
    def _same_category(self, c1: int, c2: int) -> bool:
        """Check if two commodities are in the same category."""
        for category, indices in self.commodity_categories.items():
            if c1 in indices and c2 in indices:
                return True
        return False
    
    def encode_commodity_data(self, price_tensor: np.ndarray) -> QuantumCircuit:
        """
        Encode multi-dimensional commodity data into quantum states.
        
        Args:
            price_tensor: 3D array (time, exchange, commodity)
        """
        # For simplicity, we'll encode a subset of the data
        # In practice, this would be more sophisticated
        
        # Determine number of qubits needed
        n_qubits = min(8, int(np.ceil(np.log2(self.n_time * self.n_exchange * 10))))
        
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Encode time dimension
        time_features = np.mean(price_tensor, axis=(1, 2))  # Average over exchange and commodity
        for i, feature in enumerate(time_features[:n_qubits//2]):
            if feature > 0:
                qc.x(i)
        
        # Encode exchange dimension
        exchange_features = np.mean(price_tensor, axis=(0, 2))  # Average over time and commodity
        for i, feature in enumerate(exchange_features[:n_qubits//2]):
            if feature > 0:
                qc.x(i + n_qubits//2)
        
        return qc
    
    def simulate_commodity_transition(self, initial_state: str, transition_type: str) -> Dict[str, float]:
        """
        Simulate commodity price transitions across multiple dimensions.
        
        Unlike FOREX single transition, we simulate:
        - Time evolution of commodity prices
        - Cross-exchange correlations
        - Cross-commodity relationships
        """
        if not QISKIT_AVAILABLE:
            return {'error': 'Qiskit not available'}
        
        # Get initial state
        if initial_state not in self.market_states:
            initial_state = 'ground_state'
        
        initial_tensor = self.market_states[initial_state]
        
        # Create quantum circuit
        qc = QuantumCircuit(4, 4)  # Simplified for demonstration
        
        # Encode initial state
        qc.h(0)  # Superposition for time dimension
        qc.h(1)  # Superposition for exchange dimension
        qc.h(2)  # Superposition for commodity dimension
        qc.h(3)  # Additional qubit for correlations
        
        # Apply transition operator
        if transition_type == 'time_evolution':
            qc.cx(0, 1)  # Time-exchange correlation
        elif transition_type == 'exchange_correlation':
            qc.cx(1, 2)  # Exchange-commodity correlation
        elif transition_type == 'commodity_correlation':
            qc.cx(2, 3)  # Commodity-commodity correlation
        
        # Measure
        qc.measure_all()
        
        # Run simulation
        simulator = AerSimulator()
        transpiled_circuit = transpile(qc, simulator)
        job = simulator.run(transpiled_circuit, shots=1000)
        result = job.result()
        counts = result.get_counts(transpiled_circuit)
        
        # Calculate transition metrics
        total_shots = sum(counts.values())
        probabilities = {state: count/total_shots for state, count in counts.items()}
        
        sample_values = np.array([int(state.replace(' ', ''), 2) for state in probabilities.keys()])
        sample_probs = np.array(list(probabilities.values()))
        
        if len(sample_values) == 0:
            return {'transition_probability': 0.0, 'entropy': 0.0}
        
        mean_value = np.sum(sample_values * sample_probs)
        entropy = -np.sum(sample_probs * np.log2(sample_probs + 1e-10))
        
        return {
            'transition_probability': mean_value / (2**4 - 1),
            'entropy': entropy,
            'mean_value': mean_value,
            'state_counts': counts,
            'transition_type': transition_type
        }
    
    def simulate_market_crisis(self, crisis_strength: float = 1.0) -> Dict[str, float]:
        """
        Simulate market crisis across all dimensions.
        
        Unlike FOREX single perturbation, we simulate:
        - Time dimension: Market crash over time
        - Exchange dimension: Cross-exchange contagion
        - Commodity dimension: Cross-commodity correlation breakdown
        """
        if not QISKIT_AVAILABLE:
            return {'error': 'Qiskit not available'}
        
        # Create quantum circuit for crisis simulation
        qc = QuantumCircuit(6, 6)  # More qubits for multi-dimensional crisis
        
        # Prepare crisis state
        qc.h(0)  # Time dimension in superposition
        qc.h(1)  # Exchange dimension in superposition
        qc.h(2)  # Commodity dimension in superposition
        qc.h(3)  # Crisis propagation
        qc.h(4)  # Cross-correlation breakdown
        qc.h(5)  # Recovery mechanism
        
        # Apply crisis perturbations
        for i in range(6):
            qc.rx(crisis_strength * np.pi/4, i)  # Crisis perturbation
            qc.rz(crisis_strength * np.pi/4, i)  # Volatility shock
        
        # Add cross-dimensional coupling during crisis
        qc.cx(0, 1)  # Time-Exchange coupling
        qc.cx(1, 2)  # Exchange-Commodity coupling
        qc.cx(2, 3)  # Commodity-Crisis coupling
        qc.cx(3, 4)  # Crisis-Correlation coupling
        qc.cx(4, 5)  # Correlation-Recovery coupling
        
        # Measure
        qc.measure_all()
        
        # Run simulation
        simulator = AerSimulator()
        transpiled_circuit = transpile(qc, simulator)
        job = simulator.run(transpiled_circuit, shots=1000)
        result = job.result()
        counts = result.get_counts(transpiled_circuit)
        
        # Calculate crisis metrics
        total_shots = sum(counts.values())
        probabilities = {state: count/total_shots for state, count in counts.items()}
        
        sample_values = np.array([int(state.replace(' ', ''), 2) for state in probabilities.keys()])
        sample_probs = np.array(list(probabilities.values()))
        
        if len(sample_values) == 0:
            return {'crisis_effect': 0.0, 'volatility': 0.0}
        
        mean_value = np.sum(sample_values * sample_probs)
        volatility = np.sqrt(np.sum((sample_values - mean_value)**2 * sample_probs))
        
        return {
            'crisis_effect': mean_value / (2**6 - 1),
            'volatility': volatility,
            'mean_value': mean_value,
            'crisis_strength': crisis_strength,
            'state_counts': counts,
            'cross_dimensional_coupling': True
        }

def demonstrate_commodity_quantum_vocabulary():
    """Demonstrate the commodity-specific quantum vocabulary."""
    if not QISKIT_AVAILABLE:
        print("❌ Qiskit not available. Install with: pip install qiskit qiskit-aer")
        return
    
    print("🌾 Commodity Quantum Vocabulary Demo")
    print("Adapted from FOREX inspiration for multi-dimensional commodity prediction")
    print("=" * 80)
    
    # Initialize commodity quantum vocabulary
    commodity_qv = CommodityQuantumVocabulary(n_time=100, n_exchange=4, n_commodity=50)
    
    # Demonstrate multi-dimensional states
    print("\n📊 Multi-Dimensional Market States:")
    for state_name, state_tensor in commodity_qv.market_states.items():
        print(f"  {state_name}: Shape {state_tensor.shape}")
        print(f"    Sample values: {state_tensor[0, 0, :5]}...")
    
    # Demonstrate cross-dimensional transitions
    print("\n⚛️ Cross-Dimensional Transitions:")
    transitions = ['time_evolution', 'exchange_correlation', 'commodity_correlation']
    for transition in transitions:
        result = commodity_qv.simulate_commodity_transition('ground_state', transition)
        print(f"  {transition}: Probability = {result['transition_probability']:.3f}")
        print(f"    Entropy: {result['entropy']:.3f}")
    
    # Demonstrate market crisis simulation
    print("\n💥 Multi-Dimensional Market Crisis:")
    crisis_result = commodity_qv.simulate_market_crisis(crisis_strength=0.8)
    print(f"  Crisis effect: {crisis_result['crisis_effect']:.3f}")
    print(f"  Volatility: {crisis_result['volatility']:.3f}")
    print(f"  Cross-dimensional coupling: {crisis_result['cross_dimensional_coupling']}")
    
    # Demonstrate data encoding
    print("\n🔢 Multi-Dimensional Data Encoding:")
    # Create sample 3D price tensor
    price_tensor = np.random.randn(100, 4, 50)  # (time, exchange, commodity)
    encoded_circuit = commodity_qv.encode_commodity_data(price_tensor)
    print(f"  Encoded circuit depth: {encoded_circuit.depth()}")
    print(f"  Encoded circuit gates: {len(encoded_circuit.data)}")
    
    print("\n✅ Key Differences from FOREX:")
    print("  ✓ Multi-dimensional state space (time × exchange × commodity)")
    print("  ✓ 424 target variables vs single currency pair")
    print("  ✓ Cross-commodity correlations vs single exchange rate")
    print("  ✓ Multi-exchange dynamics vs single market")
    print("  ✓ Cross-dimensional coupling operators")
    
    print("\n✅ Commodity Quantum Vocabulary Demo Completed!")
    print("This adapts FOREX inspiration to our specific multi-dimensional challenge.")

if __name__ == "__main__":
    demonstrate_commodity_quantum_vocabulary()
