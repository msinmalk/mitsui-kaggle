"""
Simplified feature engineering for commodity price prediction.
Optimized for performance and progress tracking.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class SimpleFeatureEngineer:
    """Simplified and efficient feature engineering."""
    
    def __init__(self):
        self.feature_columns = []
        
    def create_basic_features(self, df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
        """Create basic price features efficiently."""
        print("   Creating basic price features...")
        
        df_features = df.copy()
        new_features = []
        
        for i, col in enumerate(price_cols):
            if col not in df.columns:
                continue
                
            if i % 10 == 0:  # Progress indicator
                print(f"      Processing column {i+1}/{len(price_cols)}: {col}")
            
            # Basic features only
            col_features = {
                f'{col}_log_price': np.log(df[col] + 1e-8),
                f'{col}_normalized': (df[col] - df[col].mean()) / (df[col].std() + 1e-8),
                f'{col}_ma_5': df[col].rolling(5).mean(),
                f'{col}_ma_20': df[col].rolling(20).mean(),
                f'{col}_return_1': df[col].pct_change(1, fill_method=None),
                f'{col}_return_5': df[col].pct_change(5, fill_method=None),
                f'{col}_volatility_10': df[col].pct_change(fill_method=None).rolling(10).std()
            }
            
            new_features.append(pd.DataFrame(col_features, index=df.index))
        
        # Concatenate all at once
        if new_features:
            df_features = pd.concat([df_features] + new_features, axis=1)
        
        print(f"      ✓ Created {len(new_features) * len(list(new_features[0].columns))} basic features")
        return df_features
    
    def create_technical_indicators_simple(self, df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
        """Create simplified technical indicators."""
        print("   Creating technical indicators...")
        
        df_tech = df.copy()
        new_features = []
        
        # Only process first 10 price columns to avoid hanging
        sample_cols = price_cols[:10]
        
        for i, col in enumerate(sample_cols):
            if col not in df.columns:
                continue
                
            print(f"      Processing technical indicators for {col} ({i+1}/{len(sample_cols)})")
            
            try:
                col_features = {}
                
                # Simple RSI calculation
                delta = df[col].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-8)
                col_features[f'{col}_rsi'] = 100 - (100 / (1 + rs))
                
                # Simple MACD
                ema_12 = df[col].ewm(span=12).mean()
                ema_26 = df[col].ewm(span=26).mean()
                col_features[f'{col}_macd'] = ema_12 - ema_26
                col_features[f'{col}_macd_signal'] = col_features[f'{col}_macd'].ewm(span=9).mean()
                
                # Bollinger Bands
                ma_20 = df[col].rolling(20).mean()
                std_20 = df[col].rolling(20).std()
                col_features[f'{col}_bb_upper'] = ma_20 + (std_20 * 2)
                col_features[f'{col}_bb_lower'] = ma_20 - (std_20 * 2)
                col_features[f'{col}_bb_width'] = (col_features[f'{col}_bb_upper'] - col_features[f'{col}_bb_lower']) / ma_20
                
                new_features.append(pd.DataFrame(col_features, index=df.index))
                
            except Exception as e:
                print(f"      Warning: Could not create technical indicators for {col}: {e}")
                continue
        
        if new_features:
            df_tech = pd.concat([df_tech] + new_features, axis=1)
        
        print(f"      ✓ Created technical indicators for {len(sample_cols)} columns")
        return df_tech
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        print("   Creating time features...")
        
        df_time = df.copy()
        
        if 'date_id' in df.columns:
            # Simple time features
            df_time['day_of_week'] = df['date_id'] % 7
            df_time['month'] = (df['date_id'] % 365) // 30
            df_time['quarter'] = (df['date_id'] % 365) // 90
            
            # Cyclical encoding
            df_time['day_sin'] = np.sin(2 * np.pi * df_time['day_of_week'] / 7)
            df_time['day_cos'] = np.cos(2 * np.pi * df_time['day_of_week'] / 7)
        
        print("      ✓ Created time features")
        return df_time
    
    def create_all_features(self, df: pd.DataFrame, target_pairs: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Create all features with progress tracking."""
        import time
        start_time = time.time()
        
        if verbose:
            print("🚀 Starting simplified feature engineering...")
            print(f"   Input shape: {df.shape}")
        
        # Get price columns
        price_cols = [col for col in df.columns if any(x in col for x in ['Close', 'adj_close'])]
        if verbose:
            print(f"   Found {len(price_cols)} price columns")
        
        # Create features step by step
        df_features = self.create_basic_features(df, price_cols)
        df_features = self.create_technical_indicators_simple(df_features, price_cols)
        df_features = self.create_time_features(df_features)
        
        # Store feature columns
        self.feature_columns = [col for col in df_features.columns if col not in df.columns]
        
        total_time = time.time() - start_time
        if verbose:
            print(f"✅ Feature engineering completed!")
            print(f"   Final shape: {df_features.shape}")
            print(f"   New features: {len(self.feature_columns)}")
            print(f"   Total time: {total_time:.1f}s")
        
        return df_features


