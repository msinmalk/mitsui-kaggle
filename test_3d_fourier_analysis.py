"""
Test 3D Fourier Analysis on Commodity Prediction Data
====================================================

This script demonstrates the 3D Fourier analysis framework on the actual
Mitsui commodity prediction challenge data.
"""

import sys
import os
sys.path.append('./src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from signal_processing.fourier_3d_analysis import Fourier3DAnalyzer
from data.loader import CommodityDataLoader

def test_3d_fourier_on_real_data():
    """Test 3D Fourier analysis on real commodity data."""
    print("🔬 Testing 3D Fourier Analysis on Real Commodity Data")
    print("=" * 60)
    
    # Load data
    print("📊 Loading commodity data...")
    loader = CommodityDataLoader(data_dir='data/raw')
    data = loader.load_all_data()
    train_data = data['train']
    
    print(f"   Data shape: {train_data.shape}")
    print(f"   Columns: {len(train_data.columns)}")
    
    # Initialize 3D Fourier analyzer
    print("\n🌊 Initializing 3D Fourier Analyzer...")
    analyzer = Fourier3DAnalyzer(
        n_time=len(train_data),
        n_exchange=4,
        n_commodity=424  # Number of target variables
    )
    
    # Perform 3D Fourier analysis
    print("\n🚀 Performing 3D Fourier Analysis...")
    results = analyzer.analyze(train_data)
    
    # Display results summary
    print("\n📈 Analysis Results Summary:")
    print("-" * 40)
    
    # Tensor information
    print(f"Price tensor shape: {results['tensor_shape']}")
    print(f"Fourier coefficients shape: {results['fourier_coeffs'].shape}")
    
    # Frequency analysis
    temporal_dominant = results['frequency_analysis']['temporal']['dominant_frequencies'][:5]
    print(f"\nTop 5 Temporal Frequencies:")
    for freq, power in temporal_dominant:
        print(f"   Frequency {freq}: Power = {power:.2e}")
    
    exchange_dominant = results['frequency_analysis']['exchange']['dominant_frequencies']
    print(f"\nExchange Frequencies:")
    for i, (freq, power) in enumerate(exchange_dominant):
        exchange_name = analyzer.exchange_map[i]
        print(f"   {exchange_name}: Power = {power:.2e}")
    
    # Cross-exchange correlations
    exchange_corr = results['frequency_analysis']['exchange']['cross_exchange_correlations']
    print(f"\nCross-Exchange Correlations:")
    for i in range(4):
        for j in range(4):
            if i != j:
                corr = exchange_corr[i, j]
                print(f"   {analyzer.exchange_map[i]} - {analyzer.exchange_map[j]}: {corr:.3f}")
    
    # Fourier features
    fourier_features = results['fourier_features']
    print(f"\nFourier Features Extracted:")
    for feature_name, feature_data in fourier_features.items():
        print(f"   {feature_name}: {feature_data.shape}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    try:
        fig = analyzer.visualize_frequency_analysis(results, save_path='fourier_3d_analysis.png')
        print("✅ Visualizations created successfully!")
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
    
    return results

def analyze_frequency_bands(results):
    """Analyze different frequency bands for trading insights."""
    print("\n🎯 Frequency Band Analysis for Trading")
    print("=" * 50)
    
    # Extract frequency bands
    temporal_bands = results['frequency_analysis']['temporal']['frequency_bands']
    
    # Calculate band statistics
    band_stats = {}
    for band_name, band_data in temporal_bands.items():
        if len(band_data) > 0:
            band_stats[band_name] = {
                'mean_power': np.mean(band_data),
                'std_power': np.std(band_data),
                'max_power': np.max(band_data),
                'shape': band_data.shape
            }
    
    print("Frequency Band Statistics:")
    for band_name, stats in band_stats.items():
        print(f"\n{band_name.upper()}:")
        print(f"   Mean Power: {stats['mean_power']:.2e}")
        print(f"   Std Power:  {stats['std_power']:.2e}")
        print(f"   Max Power:  {stats['max_power']:.2e}")
        print(f"   Shape:      {stats['shape']}")
    
    # Trading insights
    print("\n💡 Trading Insights:")
    
    # Low frequency (long-term trends)
    if 'low_frequency' in band_stats:
        low_power = band_stats['low_frequency']['mean_power']
        print(f"   Long-term trends (low freq): Power = {low_power:.2e}")
        if low_power > 1e6:
            print("   → Strong long-term trends detected")
        else:
            print("   → Weak long-term trends")
    
    # High frequency (short-term noise)
    if 'high_frequency' in band_stats:
        high_power = band_stats['high_frequency']['mean_power']
        print(f"   Short-term noise (high freq): Power = {high_power:.2e}")
        if high_power > 1e5:
            print("   → High short-term volatility")
        else:
            print("   → Low short-term volatility")
    
    # Mid frequency (medium-term cycles)
    if 'mid_frequency' in band_stats:
        mid_power = band_stats['mid_frequency']['mean_power']
        print(f"   Medium-term cycles (mid freq): Power = {mid_power:.2e}")
        if mid_power > 1e5:
            print("   → Strong medium-term cycles")
        else:
            print("   → Weak medium-term cycles")

def extract_ml_features(results):
    """Extract features suitable for machine learning models."""
    print("\n🤖 Extracting ML Features from Fourier Analysis")
    print("=" * 50)
    
    fourier_features = results['fourier_features']
    
    # Create feature matrix
    feature_matrix = []
    feature_names = []
    
    for feature_name, feature_data in fourier_features.items():
        if len(feature_data.shape) == 1:  # 1D features
            feature_matrix.append(feature_data)
            feature_names.append(feature_name)
        elif len(feature_data.shape) == 2:  # 2D features - flatten
            flattened = feature_data.flatten()
            feature_matrix.append(flattened)
            feature_names.append(feature_name)
    
    # Stack features
    X_fourier = np.column_stack(feature_matrix)
    
    print(f"Fourier Feature Matrix:")
    print(f"   Shape: {X_fourier.shape}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Samples: {X_fourier.shape[0]}")
    
    # Feature statistics
    print(f"\nFeature Statistics:")
    for i, name in enumerate(feature_names):
        feature_data = X_fourier[:, i]
        print(f"   {name}:")
        print(f"      Mean: {np.mean(feature_data):.2e}")
        print(f"      Std:  {np.std(feature_data):.2e}")
        print(f"      Range: [{np.min(feature_data):.2e}, {np.max(feature_data):.2e}]")
    
    return X_fourier, feature_names

def main():
    """Main test function."""
    print("🚀 3D Fourier Analysis Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: 3D Fourier analysis on real data
        results = test_3d_fourier_on_real_data()
        
        # Test 2: Frequency band analysis
        analyze_frequency_bands(results)
        
        # Test 3: Extract ML features
        X_fourier, feature_names = extract_ml_features(results)
        
        print("\n✅ All tests completed successfully!")
        print(f"   Generated {X_fourier.shape[1]} Fourier features")
        print(f"   Ready for integration with ML models")
        
        return results, X_fourier, feature_names
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    results, X_fourier, feature_names = main()
