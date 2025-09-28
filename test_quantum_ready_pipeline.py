"""
Test the quantum-ready pipeline with the commodity prediction data.
Demonstrates how the base program is structured for quantum integration.
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

from data.loader import CommodityDataLoader
from data.feature_engineering_simple import SimpleFeatureEngineer
from models.quantum_ready_model import create_quantum_ready_pipeline

def test_quantum_ready_pipeline():
    """Test the complete quantum-ready pipeline."""
    
    print("🔮 Testing Quantum-Ready Pipeline")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    loader = CommodityDataLoader('data/raw')
    data = loader.load_all_data()
    
    # Create features
    print("🔧 Creating features...")
    feature_engineer = SimpleFeatureEngineer()
    features = feature_engineer.create_all_features(data['train'], data['target_pairs'], verbose=False)
    
    # Prepare targets
    target_cols = [col for col in data['labels'].columns if col.startswith('target_')]
    targets = data['labels'][target_cols]
    
    print(f"✅ Data prepared: {features.shape} features, {targets.shape} targets")
    
    # Create quantum-ready pipeline
    print("\\n🔮 Creating quantum-ready pipeline...")
    pipeline = create_quantum_ready_pipeline()
    
    # Test on first few targets
    test_targets = targets.iloc[:, :3]  # First 3 targets
    
    print(f"\\n🎯 Testing on {test_targets.shape[1]} targets...")
    
    results = {}
    
    for i, target_col in enumerate(test_targets.columns):
        print(f"\\n--- Target {i+1}: {target_col} ---")
        
        # Prepare data for this target
        y = test_targets[target_col]
        
        # Align data
        common_idx = features.index.intersection(y.index)
        X_aligned = features.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        # Clean data
        X_clean = X_aligned.fillna(method='ffill').fillna(method='bfill').fillna(0)
        y_clean = y_aligned.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        # Train pipeline
        start_time = time.time()
        pipeline.train(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions with uncertainty
        start_time = time.time()
        predictions = pipeline.predict_with_uncertainty(X_test)
        prediction_time = time.time() - start_time
        
        # Evaluate results
        target_results = {}
        
        for model_name, (pred, unc) in predictions.items():
            mse = mean_squared_error(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mse)
            
            # Uncertainty metrics
            avg_uncertainty = np.mean(unc)
            uncertainty_std = np.std(unc)
            
            target_results[model_name] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'avg_uncertainty': avg_uncertainty,
                'uncertainty_std': uncertainty_std
            }
            
            print(f"  {model_name}:")
            print(f"    MSE: {mse:.6f}")
            print(f"    MAE: {mae:.6f}")
            print(f"    RMSE: {rmse:.6f}")
            print(f"    Avg Uncertainty: {avg_uncertainty:.6f}")
            print(f"    Uncertainty Std: {uncertainty_std:.6f}")
        
        results[target_col] = {
            'target_results': target_results,
            'training_time': training_time,
            'prediction_time': prediction_time
        }
    
    # Summary
    print("\\n📊 Pipeline Summary")
    print("=" * 30)
    
    # Average performance across targets
    all_mse = []
    all_uncertainty = []
    
    for target, result in results.items():
        for model_name, metrics in result['target_results'].items():
            all_mse.append(metrics['mse'])
            all_uncertainty.append(metrics['avg_uncertainty'])
    
    print(f"Average MSE across all models and targets: {np.mean(all_mse):.6f}")
    print(f"Average Uncertainty across all models and targets: {np.mean(all_uncertainty):.6f}")
    print(f"Total training time: {sum(r['training_time'] for r in results.values()):.2f}s")
    print(f"Total prediction time: {sum(r['prediction_time'] for r in results.values()):.2f}s")
    
    # Quantum readiness assessment
    print("\\n🔮 Quantum Readiness Assessment")
    print("=" * 35)
    
    print("✅ Architecture supports:")
    print("  - Uncertainty quantification")
    print("  - Multiple model types")
    print("  - Ensemble methods")
    print("  - Easy quantum model integration")
    
    print("\\n💡 Ready for quantum enhancement:")
    print("  - Add quantum uncertainty models")
    print("  - Implement quantum feature encoding")
    print("  - Use quantum optimization")
    print("  - Deploy on quantum hardware")
    
    return results

def demonstrate_quantum_integration_points():
    """Demonstrate where quantum models can be integrated."""
    
    print("\\n🔗 Quantum Integration Points")
    print("=" * 30)
    
    print("1. Feature Engineering:")
    print("   - Quantum uncertainty features")
    print("   - Quantum correlation analysis")
    print("   - Quantum pattern recognition")
    
    print("\\n2. Model Training:")
    print("   - Quantum optimization algorithms")
    print("   - Quantum neural networks")
    print("   - Quantum ensemble methods")
    
    print("\\n3. Uncertainty Quantification:")
    print("   - Quantum Monte Carlo methods")
    print("   - Quantum state estimation")
    print("   - Quantum confidence intervals")
    
    print("\\n4. Trading Strategy:")
    print("   - Quantum risk assessment")
    print("   - Quantum portfolio optimization")
    print("   - Quantum scenario analysis")

if __name__ == "__main__":
    # Test the pipeline
    results = test_quantum_ready_pipeline()
    
    # Show integration points
    demonstrate_quantum_integration_points()
    
    print("\\n🎉 Quantum-ready pipeline test completed!")
    print("\\nNext steps:")
    print("1. Install Qiskit: pip install qiskit")
    print("2. Implement quantum uncertainty models")
    print("3. Test on real quantum hardware")
    print("4. Deploy to Azure with quantum capabilities")


