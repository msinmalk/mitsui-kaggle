"""
FOREX-Inspired Quantum Monte Carlo Comparison
Based on Alaminos et al. (2023) Nature article methodology.

Reference: https://www.nature.com/articles/s41599-023-01836-2
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.loader import CommodityDataLoader
from data.feature_engineering import FeatureEngineer
from quantum.auxiliary_field_qmc import FOREXInspiredCommodityPredictor, AuxiliaryFieldQMC
from quantum.quantum_monte_carlo import QuantumMonteCarlo
from models.traditional_ml import RandomForestModel

class FOREXMethodologyComparison:
    """
    Compare different Monte Carlo methods following the Nature article methodology.
    """
    
    def __init__(self, data_dir: str = 'data/raw'):
        self.data_dir = data_dir
        self.loader = CommodityDataLoader(data_dir)
        self.feature_engineer = FeatureEngineer()
        self.results = {}
        
    def load_and_prepare_data(self, sample_size: int = 1000):
        """
        Load and prepare data following the Nature article's sample size methodology.
        """
        print(f"📊 Loading data with sample size: {sample_size}")
        
        # Load raw data
        raw_data = self.loader.load_all_data()
        train_df = raw_data['train']
        target_pairs_df = raw_data['target_pairs']
        
        # Feature engineering
        features_df = self.feature_engineer.create_all_features(train_df, target_pairs_df)
        
        # Select target variables (following FOREX methodology)
        target_cols = [col for col in target_pairs_df.columns if col != 'date_id']
        
        # Sample data to match FOREX study sample sizes
        if len(features_df) > sample_size:
            features_df = features_df.sample(n=sample_size, random_state=42)
        
        # Prepare features and targets
        feature_cols = [col for col in features_df.columns if col not in ['date_id'] + target_cols]
        X = features_df[feature_cols].fillna(0).values
        
        # Use first target for simplicity (following FOREX single currency pair approach)
        y = features_df[target_cols[0]].fillna(0).values if target_cols else np.zeros(len(X))
        
        # Standardize features (as done in FOREX study)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"✅ Data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled, y, scaler
    
    def run_auxiliary_field_qmc(self, X: np.ndarray, y: np.ndarray, n_qubits: int = 4) -> Dict[str, float]:
        """
        Run Auxiliary-Field Quantum Monte Carlo (main method from Nature article).
        """
        print("🔬 Running Auxiliary-Field Quantum Monte Carlo...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize AFQMC
        afqmc = FOREXInspiredCommodityPredictor(n_qubits=n_qubits, n_samples=1000)
        
        # Train
        afqmc.train(X_train, y_train)
        
        # Predict
        predictions, uncertainties = afqmc.predict_with_uncertainty(X_test)
        
        # Evaluate
        performance = afqmc.evaluate_performance(X_test, y_test)
        
        # Calculate standard deviation (main metric from Nature article)
        std_deviation = performance['std_deviation']
        
        print(f"  AFQMC Standard Deviation: {std_deviation:.4f}")
        
        return {
            'method': 'Auxiliary-Field QMC',
            'std_deviation': std_deviation,
            'mse': performance['mse'],
            'mae': performance['mae'],
            'avg_uncertainty': performance['avg_uncertainty'],
            'predictions': predictions,
            'uncertainties': uncertainties
        }
    
    def run_standard_qmc(self, X: np.ndarray, y: np.ndarray, n_qubits: int = 4) -> Dict[str, float]:
        """
        Run standard Quantum Monte Carlo for comparison.
        """
        print("⚛️ Running Standard Quantum Monte Carlo...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize QMC
        qmc = QuantumMonteCarlo(n_qubits=n_qubits, n_samples=1000)
        
        # Train
        qmc.train(X_train, y_train)
        
        # Predict
        uncertainties = qmc.batch_predict_uncertainty(X_test)
        predictions = np.array([unc['qmc_mean'] for unc in uncertainties])
        
        # Calculate standard deviation
        mse = np.mean((predictions - y_test) ** 2)
        std_deviation = np.sqrt(mse)
        
        print(f"  Standard QMC Standard Deviation: {std_deviation:.4f}")
        
        return {
            'method': 'Standard QMC',
            'std_deviation': std_deviation,
            'mse': mse,
            'mae': np.mean(np.abs(predictions - y_test)),
            'avg_uncertainty': np.mean([unc['qmc_uncertainty'] for unc in uncertainties]),
            'predictions': predictions,
            'uncertainties': uncertainties
        }
    
    def run_classical_monte_carlo(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Run classical Monte Carlo for baseline comparison.
        """
        print("📊 Running Classical Monte Carlo...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize classical model
        classical_model = RandomForestModel()
        
        # Prepare data for traditional ML (expects DataFrames)
        X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
        X_test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
        y_train_df = pd.DataFrame(y_train, columns=['target'])
        y_test_df = pd.DataFrame(y_test, columns=['target'])
        
        # Train
        classical_model.fit(X_train_df, y_train_df)
        
        # Predict with Monte Carlo sampling
        predictions = classical_model.predict(X_test_df).flatten()
        
        # Add Monte Carlo uncertainty estimation
        n_samples = 1000
        mc_predictions = []
        
        for _ in range(n_samples):
            # Add noise to simulate Monte Carlo sampling
            noise = np.random.normal(0, 0.1, len(predictions))
            mc_pred = predictions + noise
            mc_predictions.append(mc_pred)
        
        mc_predictions = np.array(mc_predictions)
        mean_predictions = np.mean(mc_predictions, axis=0)
        uncertainty = np.std(mc_predictions, axis=0)
        
        # Calculate standard deviation
        mse = np.mean((mean_predictions - y_test) ** 2)
        std_deviation = np.sqrt(mse)
        
        print(f"  Classical MC Standard Deviation: {std_deviation:.4f}")
        
        return {
            'method': 'Classical Monte Carlo',
            'std_deviation': std_deviation,
            'mse': mse,
            'mae': np.mean(np.abs(mean_predictions - y_test)),
            'avg_uncertainty': np.mean(uncertainty),
            'predictions': mean_predictions,
            'uncertainties': [{'mc_uncertainty': u} for u in uncertainty]
        }
    
    def run_stress_testing(self, X: np.ndarray, y: np.ndarray, n_qubits: int = 4) -> Dict[str, Dict]:
        """
        Run stress testing as described in the Nature article.
        """
        print("🧪 Running Stress Testing...")
        
        # Initialize AFQMC
        afqmc = AuxiliaryFieldQMC(n_qubits=n_qubits, n_samples=1000)
        
        # Train
        afqmc.train(X, y)
        
        # Stress factors from Nature article
        stress_factors = [0.5, 1.0, 1.5, 2.0]
        
        # Run stress tests
        stress_results = afqmc.stress_test(X[:50], stress_factors)  # Use subset for stress testing
        
        # Calculate performance for each stress level
        stress_performance = {}
        for stress_level, results in stress_results.items():
            avg_uncertainty = np.mean([r['afqmc_uncertainty'] for r in results])
            stress_performance[stress_level] = {
                'avg_uncertainty': avg_uncertainty,
                'n_samples': len(results)
            }
        
        print("  Stress Testing Results:")
        for level, perf in stress_performance.items():
            print(f"    {level}: Avg Uncertainty = {perf['avg_uncertainty']:.4f}")
        
        return stress_performance
    
    def compare_methods(self, sample_sizes: List[int] = [100, 500, 1000]):
        """
        Compare methods across different sample sizes as in the Nature article.
        """
        print("🔍 FOREX-Inspired Method Comparison")
        print("=" * 60)
        
        all_results = []
        
        for sample_size in sample_sizes:
            print(f"\n📊 Testing with sample size: {sample_size}")
            
            # Load data
            X, y, scaler = self.load_and_prepare_data(sample_size)
            
            # Run methods
            methods = [
                self.run_auxiliary_field_qmc,
                self.run_standard_qmc,
                self.run_classical_monte_carlo
            ]
            
            sample_results = []
            for method in methods:
                try:
                    result = method(X, y)
                    result['sample_size'] = sample_size
                    sample_results.append(result)
                except Exception as e:
                    print(f"  ❌ Error in {method.__name__}: {e}")
                    continue
            
            all_results.extend(sample_results)
        
        # Analyze results
        self.analyze_results(all_results)
        
        return all_results
    
    def analyze_results(self, results: List[Dict]):
        """
        Analyze results following the Nature article methodology.
        """
        print("\n📈 Results Analysis (Following Nature Article Methodology)")
        print("=" * 70)
        
        # Group by method
        methods = {}
        for result in results:
            method = result['method']
            if method not in methods:
                methods[method] = []
            methods[method].append(result)
        
        # Calculate statistics for each method
        print("\nStandard Deviation Results (Main Metric from Nature Article):")
        print("-" * 60)
        
        for method, method_results in methods.items():
            std_deviations = [r['std_deviation'] for r in method_results]
            sample_sizes = [r['sample_size'] for r in method_results]
            
            print(f"\n{method}:")
            for i, (std_dev, sample_size) in enumerate(zip(std_deviations, sample_sizes)):
                print(f"  Sample Size {sample_size}: {std_dev:.4f}")
            
            avg_std = np.mean(std_deviations)
            min_std = np.min(std_deviations)
            max_std = np.max(std_deviations)
            
            print(f"  Average: {avg_std:.4f}")
            print(f"  Range: {min_std:.4f} - {max_std:.4f}")
        
        # Compare with Nature article results
        print("\n📚 Comparison with Nature Article Results:")
        print("-" * 50)
        print("Nature Article (FOREX):")
        print("  - AFQMC: Significantly enhanced accuracy")
        print("  - Minimal error and consistent estimations")
        print("  - Standard deviation range: 0.40-0.83 (large samples)")
        print("  - Standard deviation range: 0.75-1.24 (small samples)")
        
        print("\nOur Results (Commodity):")
        for method, method_results in methods.items():
            std_deviations = [r['std_deviation'] for r in method_results]
            avg_std = np.mean(std_deviations)
            print(f"  - {method}: {avg_std:.4f} average")
    
    def create_visualization(self, results: List[Dict]):
        """
        Create visualization of results.
        """
        print("\n📊 Creating visualization...")
        
        # Group by method
        methods = {}
        for result in results:
            method = result['method']
            if method not in methods:
                methods[method] = {'sample_sizes': [], 'std_deviations': []}
            methods[method]['sample_sizes'].append(result['sample_size'])
            methods[method]['std_deviations'].append(result['std_deviation'])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        for method, data in methods.items():
            plt.plot(data['sample_sizes'], data['std_deviations'], 
                    marker='o', label=method, linewidth=2, markersize=8)
        
        plt.xlabel('Sample Size')
        plt.ylabel('Standard Deviation')
        plt.title('FOREX-Inspired Quantum Monte Carlo Comparison\n(Following Nature Article Methodology)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Add reference lines from Nature article
        plt.axhline(y=0.83, color='red', linestyle='--', alpha=0.7, label='Nature Article: Large Samples (0.83)')
        plt.axhline(y=1.24, color='orange', linestyle='--', alpha=0.7, label='Nature Article: Small Samples (1.24)')
        
        plt.tight_layout()
        plt.savefig('forex_inspired_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved as 'forex_inspired_comparison.png'")

def main():
    """Main function to run the FOREX-inspired comparison."""
    print("🌍 FOREX-Inspired Quantum Monte Carlo for Commodity Prediction")
    print("Based on Alaminos et al. (2023) Nature Article")
    print("=" * 70)
    
    # Initialize comparison
    comparison = FOREXMethodologyComparison()
    
    # Run comparison
    results = comparison.compare_methods(sample_sizes=[100, 500, 1000])
    
    # Create visualization
    comparison.create_visualization(results)
    
    print("\n✅ FOREX-Inspired Comparison Completed!")
    print("Reference: https://www.nature.com/articles/s41599-023-01836-2")

if __name__ == "__main__":
    main()
