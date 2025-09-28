"""
3D Fourier Analysis for Commodity Prediction Challenge
=====================================================

This module implements 3D Fourier analysis on the multi-dimensional signal space:
- Time dimension: 1,961 time steps
- Exchange dimension: 4 exchanges (LME, JPX, US, FX)  
- Commodity dimension: 424 target variables

The 3D Fourier transform captures:
- Temporal frequencies (daily, weekly, monthly cycles)
- Exchange frequencies (cross-exchange correlations)
- Commodity frequencies (cross-commodity relationships)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import fft
from scipy.signal import periodogram
import warnings
warnings.filterwarnings('ignore')

class Fourier3DAnalyzer:
    """3D Fourier Analysis for commodity price prediction."""
    
    def __init__(self, n_time: int = 1961, n_exchange: int = 4, n_commodity: int = 424):
        self.n_time = n_time
        self.n_exchange = n_exchange
        self.n_commodity = n_commodity
        
        # Exchange mapping
        self.exchange_map = {
            0: 'LME',    # London Metal Exchange
            1: 'JPX',    # Japan Exchange
            2: 'US',     # US Stock Markets
            3: 'FX'      # Foreign Exchange
        }
        
        # Initialize data structures
        self.price_tensor = None
        self.fourier_coeffs = None
        self.frequency_analysis = None
        
        print(f"🔬 Initialized 3D Fourier Analyzer")
        print(f"   Time dimension: {n_time} samples")
        print(f"   Exchange dimension: {n_exchange} exchanges")
        print(f"   Commodity dimension: {n_commodity} targets")
    
    def _organize_data_into_3d_tensor(self, train_data: pd.DataFrame) -> np.ndarray:
        """Organize raw data into 3D tensor: (time, exchange, commodity)."""
        print("📊 Organizing data into 3D tensor...")
        
        # Initialize 3D tensor
        tensor = np.full((self.n_time, self.n_exchange, self.n_commodity), np.nan)
        
        # Get all columns
        columns = train_data.columns.tolist()
        
        # Group columns by exchange
        exchange_columns = {
            'LME': [col for col in columns if col.startswith('LME_')],
            'JPX': [col for col in columns if col.startswith('JPX_')],
            'US': [col for col in columns if col.startswith('US_')],
            'FX': [col for col in columns if col.startswith('FX_')]
        }
        
        print(f"   LME columns: {len(exchange_columns['LME'])}")
        print(f"   JPX columns: {len(exchange_columns['JPX'])}")
        print(f"   US columns: {len(exchange_columns['US'])}")
        print(f"   FX columns: {len(exchange_columns['FX'])}")
        
        # Fill tensor for each exchange
        for exchange_idx, (exchange_name, cols) in enumerate(exchange_columns.items()):
            if cols:  # If exchange has columns
                # Take first few columns to fill commodity dimension
                n_cols_to_use = min(len(cols), self.n_commodity)
                selected_cols = cols[:n_cols_to_use]
                
                # Extract data for this exchange
                exchange_data = train_data[selected_cols].values
                
                # Fill tensor
                tensor[:, exchange_idx, :n_cols_to_use] = exchange_data
                
                print(f"   {exchange_name}: Using {n_cols_to_use} columns")
        
        # Handle remaining commodity slots with interpolated data
        for exchange_idx in range(self.n_exchange):
            exchange_data = tensor[:, exchange_idx, :]
            valid_data = exchange_data[~np.isnan(exchange_data)]
            
            if len(valid_data) > 0:
                # Fill NaN values with forward fill, then backward fill
                exchange_df = pd.DataFrame(exchange_data)
                exchange_df = exchange_df.fillna(method='ffill').fillna(method='bfill')
                tensor[:, exchange_idx, :] = exchange_df.values
        
        # Final cleanup - replace any remaining NaN with 0
        tensor = np.nan_to_num(tensor, nan=0.0)
        
        print(f"✅ 3D tensor created: {tensor.shape}")
        print(f"   Data range: [{tensor.min():.2f}, {tensor.max():.2f}]")
        
        return tensor
    
    def _compute_3d_fourier_transform(self, tensor: np.ndarray) -> np.ndarray:
        """Compute 3D Fourier transform of the price tensor."""
        print("🌊 Computing 3D Fourier Transform...")
        
        # 3D FFT
        fourier_coeffs = fft.fftn(tensor, axes=(0, 1, 2))
        
        # Compute magnitude spectrum
        magnitude_spectrum = np.abs(fourier_coeffs)
        
        print(f"✅ 3D FFT computed: {fourier_coeffs.shape}")
        print(f"   Frequency range: [0, {self.n_time-1}] x [0, {self.n_exchange-1}] x [0, {self.n_commodity-1}]")
        
        return fourier_coeffs, magnitude_spectrum
    
    def _analyze_frequency_components(self, fourier_coeffs: np.ndarray) -> Dict:
        """Analyze frequency components in each dimension."""
        print("📈 Analyzing frequency components...")
        
        analysis = {}
        
        # Temporal frequency analysis (axis 0)
        temporal_fft = fft.fft(fourier_coeffs, axis=0)
        temporal_power = np.abs(temporal_fft) ** 2
        
        # Exchange frequency analysis (axis 1) 
        exchange_fft = fft.fft(fourier_coeffs, axis=1)
        exchange_power = np.abs(exchange_fft) ** 2
        
        # Commodity frequency analysis (axis 2)
        commodity_fft = fft.fft(fourier_coeffs, axis=2)
        commodity_power = np.abs(commodity_fft) ** 2
        
        # Extract dominant frequencies
        analysis['temporal'] = {
            'power_spectrum': temporal_power,
            'dominant_frequencies': self._find_dominant_frequencies(temporal_power, axis=0),
            'frequency_bands': self._extract_frequency_bands(temporal_power, axis=0)
        }
        
        analysis['exchange'] = {
            'power_spectrum': exchange_power,
            'dominant_frequencies': self._find_dominant_frequencies(exchange_power, axis=1),
            'cross_exchange_correlations': self._compute_cross_exchange_correlations(exchange_power)
        }
        
        analysis['commodity'] = {
            'power_spectrum': commodity_power,
            'dominant_frequencies': self._find_dominant_frequencies(commodity_power, axis=2),
            'cross_commodity_correlations': self._compute_cross_commodity_correlations(commodity_power)
        }
        
        print("✅ Frequency analysis completed")
        return analysis
    
    def _find_dominant_frequencies(self, power_spectrum: np.ndarray, axis: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Find dominant frequencies in the power spectrum."""
        # Sum over other axes to get 1D power spectrum
        if axis == 0:
            power_1d = np.sum(power_spectrum, axis=(1, 2))
        elif axis == 1:
            power_1d = np.sum(power_spectrum, axis=(0, 2))
        else:  # axis == 2
            power_1d = np.sum(power_spectrum, axis=(0, 1))
        
        # Find top frequencies
        top_indices = np.argsort(power_1d)[-top_k:][::-1]
        top_powers = power_1d[top_indices]
        
        return list(zip(top_indices, top_powers))
    
    def _extract_frequency_bands(self, power_spectrum: np.ndarray, axis: int) -> Dict[str, np.ndarray]:
        """Extract different frequency bands."""
        if axis == 0:  # Temporal
            n_freqs = power_spectrum.shape[0]
            
            # Define frequency bands
            low_freq_end = n_freqs // 20      # Lowest 5%
            mid_freq_start = n_freqs // 20    # 5% to 50%
            mid_freq_end = n_freqs // 2
            high_freq_start = n_freqs // 2    # 50% to 100%
            
            bands = {
                'low_frequency': power_spectrum[:low_freq_end],
                'mid_frequency': power_spectrum[mid_freq_start:mid_freq_end],
                'high_frequency': power_spectrum[high_freq_start:]
            }
        else:
            # For exchange and commodity, use simpler bands
            n_freqs = power_spectrum.shape[axis]
            bands = {
                'low_frequency': power_spectrum[:n_freqs//3],
                'mid_frequency': power_spectrum[n_freqs//3:2*n_freqs//3],
                'high_frequency': power_spectrum[2*n_freqs//3:]
            }
        
        return bands
    
    def _compute_cross_exchange_correlations(self, exchange_power: np.ndarray) -> np.ndarray:
        """Compute correlations between different exchanges."""
        # Reshape to (time, commodity, exchange) for correlation analysis
        reshaped = np.transpose(exchange_power, (0, 2, 1))  # (time, commodity, exchange)
        
        # Compute correlation matrix between exchanges
        correlations = np.corrcoef(reshaped.reshape(-1, reshaped.shape[-1]).T)
        
        return correlations
    
    def _compute_cross_commodity_correlations(self, commodity_power: np.ndarray) -> np.ndarray:
        """Compute correlations between different commodities."""
        # Reshape to (time, exchange, commodity) for correlation analysis
        reshaped = np.transpose(commodity_power, (0, 1, 2))  # (time, exchange, commodity)
        
        # Compute correlation matrix between commodities
        correlations = np.corrcoef(reshaped.reshape(-1, reshaped.shape[-1]).T)
        
        return correlations
    
    def _extract_fourier_features(self, fourier_coeffs: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features from Fourier coefficients for ML models."""
        print("🔧 Extracting Fourier features...")
        
        features = {}
        
        # 1. Magnitude spectrum features
        magnitude = np.abs(fourier_coeffs)
        features['magnitude_mean'] = np.mean(magnitude, axis=(1, 2))  # Temporal
        features['magnitude_std'] = np.std(magnitude, axis=(1, 2))
        features['magnitude_max'] = np.max(magnitude, axis=(1, 2))
        
        # 2. Phase features
        phase = np.angle(fourier_coeffs)
        features['phase_mean'] = np.mean(phase, axis=(1, 2))
        features['phase_std'] = np.std(phase, axis=(1, 2))
        
        # 3. Cross-dimensional features
        # Temporal-Exchange interactions
        temporal_exchange = np.mean(magnitude, axis=2)  # (time, exchange)
        features['temporal_exchange_interaction'] = temporal_exchange
        
        # Temporal-Commodity interactions  
        temporal_commodity = np.mean(magnitude, axis=1)  # (time, commodity)
        features['temporal_commodity_interaction'] = temporal_commodity
        
        # Exchange-Commodity interactions
        exchange_commodity = np.mean(magnitude, axis=0)  # (exchange, commodity)
        features['exchange_commodity_interaction'] = exchange_commodity
        
        # 4. Frequency band features
        n_time = magnitude.shape[0]
        low_freq = magnitude[:n_time//4]      # Lowest 25%
        mid_freq = magnitude[n_time//4:3*n_time//4]  # Middle 50%
        high_freq = magnitude[3*n_time//4:]   # Highest 25%
        
        features['low_frequency_power'] = np.mean(low_freq, axis=(1, 2))
        features['mid_frequency_power'] = np.mean(mid_freq, axis=(1, 2))
        features['high_frequency_power'] = np.mean(high_freq, axis=(1, 2))
        
        print(f"✅ Extracted {len(features)} Fourier feature types")
        return features
    
    def analyze(self, train_data: pd.DataFrame) -> Dict:
        """Perform complete 3D Fourier analysis."""
        print("🚀 Starting 3D Fourier Analysis")
        print("=" * 50)
        
        # Step 1: Organize data into 3D tensor
        self.price_tensor = self._organize_data_into_3d_tensor(train_data)
        
        # Step 2: Compute 3D Fourier transform
        self.fourier_coeffs, magnitude_spectrum = self._compute_3d_fourier_transform(self.price_tensor)
        
        # Step 3: Analyze frequency components
        self.frequency_analysis = self._analyze_frequency_components(self.fourier_coeffs)
        
        # Step 4: Extract features for ML models
        fourier_features = self._extract_fourier_features(self.fourier_coeffs)
        
        # Compile results
        results = {
            'price_tensor': self.price_tensor,
            'fourier_coeffs': self.fourier_coeffs,
            'magnitude_spectrum': magnitude_spectrum,
            'frequency_analysis': self.frequency_analysis,
            'fourier_features': fourier_features,
            'tensor_shape': self.price_tensor.shape
        }
        
        print("\n✅ 3D Fourier Analysis Complete!")
        print(f"   Tensor shape: {self.price_tensor.shape}")
        print(f"   Fourier features: {len(fourier_features)} types")
        
        return results
    
    def visualize_frequency_analysis(self, results: Dict, save_path: Optional[str] = None):
        """Visualize the frequency analysis results."""
        print("📊 Creating frequency analysis visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('3D Fourier Analysis Results', fontsize=16)
        
        # 1. Temporal frequency spectrum
        temporal_power = results['frequency_analysis']['temporal']['power_spectrum']
        temporal_1d = np.sum(temporal_power, axis=(1, 2))
        
        axes[0, 0].plot(temporal_1d)
        axes[0, 0].set_title('Temporal Frequency Spectrum')
        axes[0, 0].set_xlabel('Frequency')
        axes[0, 0].set_ylabel('Power')
        axes[0, 0].grid(True)
        
        # 2. Exchange frequency spectrum
        exchange_power = results['frequency_analysis']['exchange']['power_spectrum']
        exchange_1d = np.sum(exchange_power, axis=(0, 2))
        
        axes[0, 1].bar(range(len(exchange_1d)), exchange_1d)
        axes[0, 1].set_title('Exchange Frequency Spectrum')
        axes[0, 1].set_xlabel('Exchange Index')
        axes[0, 1].set_ylabel('Power')
        axes[0, 1].set_xticks(range(4))
        axes[0, 1].set_xticklabels(['LME', 'JPX', 'US', 'FX'])
        
        # 3. Commodity frequency spectrum (first 50)
        commodity_power = results['frequency_analysis']['commodity']['power_spectrum']
        commodity_1d = np.sum(commodity_power, axis=(0, 1))
        
        axes[0, 2].plot(commodity_1d[:50])
        axes[0, 2].set_title('Commodity Frequency Spectrum (First 50)')
        axes[0, 2].set_xlabel('Commodity Index')
        axes[0, 2].set_ylabel('Power')
        axes[0, 2].grid(True)
        
        # 4. Cross-exchange correlations
        exchange_corr = results['frequency_analysis']['exchange']['cross_exchange_correlations']
        im1 = axes[1, 0].imshow(exchange_corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title('Cross-Exchange Correlations')
        axes[1, 0].set_xlabel('Exchange')
        axes[1, 0].set_ylabel('Exchange')
        axes[1, 0].set_xticks(range(4))
        axes[1, 0].set_yticks(range(4))
        axes[1, 0].set_xticklabels(['LME', 'JPX', 'US', 'FX'])
        axes[1, 0].set_yticklabels(['LME', 'JPX', 'US', 'FX'])
        plt.colorbar(im1, ax=axes[1, 0])
        
        # 5. Frequency band analysis
        temporal_bands = results['frequency_analysis']['temporal']['frequency_bands']
        band_names = ['Low', 'Mid', 'High']
        band_powers = [
            np.mean(temporal_bands['low_frequency']),
            np.mean(temporal_bands['mid_frequency']),
            np.mean(temporal_bands['high_frequency'])
        ]
        
        axes[1, 1].bar(band_names, band_powers)
        axes[1, 1].set_title('Temporal Frequency Band Power')
        axes[1, 1].set_ylabel('Average Power')
        
        # 6. 3D magnitude spectrum (2D slice)
        magnitude_2d = np.mean(results['magnitude_spectrum'], axis=2)  # Average over commodity
        im2 = axes[1, 2].imshow(magnitude_2d, aspect='auto', cmap='viridis')
        axes[1, 2].set_title('2D Magnitude Spectrum (Time vs Exchange)')
        axes[1, 2].set_xlabel('Exchange')
        axes[1, 2].set_ylabel('Time')
        axes[1, 2].set_xticks(range(4))
        axes[1, 2].set_xticklabels(['LME', 'JPX', 'US', 'FX'])
        plt.colorbar(im2, ax=axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
        
        plt.show()
        
        return fig

def demonstrate_3d_fourier_analysis():
    """Demonstrate 3D Fourier analysis on sample data."""
    print("🔬 3D Fourier Analysis Demonstration")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_time, n_exchange, n_commodity = 100, 4, 50
    
    # Create synthetic price data with different patterns
    sample_data = {}
    
    # LME data (metals) - trending
    sample_data['LME_AH_Close'] = 2000 + np.cumsum(np.random.randn(n_time) * 10)
    sample_data['LME_CA_Close'] = 7000 + np.cumsum(np.random.randn(n_time) * 20)
    
    # JPX data (futures) - cyclical
    t = np.linspace(0, 4*np.pi, n_time)
    sample_data['JPX_Gold_Close'] = 4000 + 100 * np.sin(t) + np.random.randn(n_time) * 50
    
    # US data (stocks) - random walk
    sample_data['US_Stock_Close'] = 100 + np.cumsum(np.random.randn(n_time) * 2)
    
    # FX data (currencies) - mean-reverting
    sample_data['FX_USDJPY'] = 110 + 10 * np.sin(t/2) + np.random.randn(n_time) * 1
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Initialize analyzer
    analyzer = Fourier3DAnalyzer(n_time=n_time, n_exchange=4, n_commodity=4)
    
    # Perform analysis
    results = analyzer.analyze(df)
    
    # Visualize results
    analyzer.visualize_frequency_analysis(results)
    
    print("\n✅ Demonstration completed!")
    return results

if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_3d_fourier_analysis()
