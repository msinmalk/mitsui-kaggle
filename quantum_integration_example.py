"""
Example integration of quantum uncertainty quantification with commodity prediction.
Demonstrates how to use quantum computing for uncertainty quantification.
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import our modules
from data.loader import CommodityDataLoader
from data.feature_engineering_simple import SimpleFeatureEngineer
from quantum.quantum_uncertainty import QuantumUncertaintyModel, HybridQuantumModel, create_quantum_uncertainty_features

def demonstrate_quantum_integration():
    """Demonstrate quantum uncertainty integration with commodity prediction."""
    
    print("🚀 Quantum Uncertainty Integration Demo")
    print("=" * 50)
    
    # Load data
    print("📊 Loading commodity data...")
    loader = CommodityDataLoader('data/raw')
    data = loader.load_all_data()
    
    # Use small sample for demo
    sample_data = data['train'].head(200).copy()
    target_cols = [col for col in data['labels'].columns if col.startswith('target_')]
    targets = data['labels'][target_cols].head(200)
    
    # Create features
    print("🔧 Creating features...")
    feature_engineer = SimpleFeatureEngineer()
    features = feature_engineer.create_all_features(sample_data, data['target_pairs'], verbose=False)
    
    # Add quantum uncertainty features
    print("⚛️ Adding quantum uncertainty features...")
    features_with_quantum = create_quantum_uncertainty_features(features)
    
    # Clean data
    features_clean = features_with_quantum.fillna(method='ffill').fillna(method='bfill').fillna(0)
    targets_clean = targets.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"✅ Data prepared: {features_clean.shape} features, {targets_clean.shape} targets")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_clean, targets_clean, test_size=0.2, random_state=42
    )
    
    # Train classical model
    print("🤖 Training classical model...")
    classical_model = RandomForestRegressor(n_estimators=50, random_state=42)
    classical_model.fit(X_train, y_train)
    
    # Get classical predictions
    classical_predictions = classical_model.predict(X_test)
    classical_mse = mean_squared_error(y_test, classical_predictions)
    classical_mae = mean_absolute_error(y_test, classical_predictions)
    
    print(f"   Classical MSE: {classical_mse:.4f}")
    print(f"   Classical MAE: {classical_mae:.4f}")
    
    # Demonstrate quantum uncertainty (if Qiskit available)
    try:
        print("⚛️ Training quantum uncertainty model...")
        quantum_model = QuantumUncertaintyModel(n_qubits=4, n_layers=2)
        quantum_model.train(X_train.values, y_train.values)
        
        # Get quantum uncertainty predictions
        print("🔮 Predicting with quantum uncertainty...")
        test_features = X_test.values[:5]  # Test on first 5 samples
        uncertainties = quantum_model.batch_predict_uncertainty(test_features)
        
        print("\n📊 Quantum Uncertainty Analysis:")
        print("-" * 40)
        for i, uncertainty in enumerate(uncertainties):
            print(f"Sample {i+1}:")
            print(f"  Entropy: {uncertainty['entropy']:.3f}")
            print(f"  Variance: {uncertainty['variance']:.3f}")
            print(f"  Max Probability: {uncertainty['max_probability']:.3f}")
            print(f"  Number of States: {uncertainty['num_states']}")
            print()
        
        # Create hybrid model
        print("🔗 Creating hybrid classical-quantum model...")
        hybrid_model = HybridQuantumModel(classical_model, quantum_model)
        hybrid_model.train(X_train.values, y_train.values)
        
        # Get hybrid predictions
        predictions, uncertainties = hybrid_model.predict_with_uncertainty(X_test.values[:5])
        
        print("\n🎯 Hybrid Model Results:")
        print("-" * 30)
        for i in range(len(predictions)):
            print(f"Sample {i+1}:")
            print(f"  Prediction: {predictions[i]:.4f}")
            print(f"  Uncertainty (Entropy): {uncertainties[i]['entropy']:.3f}")
            print(f"  Actual: {y_test.iloc[i].values[0]:.4f}")
            print()
        
    except ImportError:
        print("❌ Qiskit not available. Install with: pip install qiskit")
        print("   Quantum uncertainty features still added to classical model")
    
    # Analyze quantum uncertainty features
    print("📈 Quantum Uncertainty Feature Analysis:")
    print("-" * 40)
    
    quantum_features = [col for col in features_with_quantum.columns if 'quantum' in col or 'uncertainty' in col]
    print(f"Number of quantum uncertainty features: {len(quantum_features)}")
    
    if quantum_features:
        # Show feature importance
        feature_importance = classical_model.feature_importances_
        feature_names = features_clean.columns
        
        # Get top quantum features
        quantum_feature_indices = [i for i, name in enumerate(feature_names) if name in quantum_features]
        quantum_importances = feature_importance[quantum_feature_indices]
        
        if len(quantum_importances) > 0:
            top_quantum_features = sorted(zip(quantum_features, quantum_importances), 
                                        key=lambda x: x[1], reverse=True)[:5]
            
            print("\nTop 5 Quantum Uncertainty Features:")
            for feature, importance in top_quantum_features:
                print(f"  {feature}: {importance:.4f}")
    
    print("\n✅ Quantum integration demo completed!")
    print("\n💡 Next Steps:")
    print("1. Install Qiskit: pip install qiskit")
    print("2. Access IBM Quantum: https://quantum-computing.ibm.com/")
    print("3. Experiment with real quantum hardware")
    print("4. Scale to full dataset with quantum uncertainty")

if __name__ == "__main__":
    demonstrate_quantum_integration()


