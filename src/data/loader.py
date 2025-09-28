"""
Data loading and preprocessing module for commodity price prediction challenge.
Handles loading of LME, JPX, US Stock, and Forex data with proper preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CommodityDataLoader:
    """Main data loader for commodity prediction challenge."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.target_pairs = None
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available datasets."""
        try:
            # Load main datasets
            self.train_data = pd.read_csv(self.data_dir / "train.csv")
            self.test_data = pd.read_csv(self.data_dir / "test.csv")
            self.train_labels = pd.read_csv(self.data_dir / "train_labels.csv")
            self.target_pairs = pd.read_csv(self.data_dir / "target_pairs.csv")
            
            logger.info(f"Loaded train data: {self.train_data.shape}")
            logger.info(f"Loaded test data: {self.test_data.shape}")
            logger.info(f"Loaded train labels: {self.train_labels.shape}")
            logger.info(f"Loaded target pairs: {self.target_pairs.shape}")
            
            return {
                'train': self.train_data,
                'test': self.test_data,
                'labels': self.train_labels,
                'target_pairs': self.target_pairs
            }
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_data_info(self) -> Dict:
        """Get comprehensive information about the loaded data."""
        if self.train_data is None:
            self.load_all_data()
            
        info = {
            'train_shape': self.train_data.shape,
            'test_shape': self.test_data.shape,
            'labels_shape': self.train_labels.shape,
            'target_pairs_shape': self.target_pairs.shape,
            'date_range': {
                'train_start': self.train_data['date_id'].min(),
                'train_end': self.train_data['date_id'].max(),
                'test_start': self.test_data['date_id'].min(),
                'test_end': self.test_data['date_id'].max()
            },
            'missing_values': self.train_data.isnull().sum().sum(),
            'data_types': self.train_data.dtypes.value_counts().to_dict()
        }
        
        return info
    
    def get_market_data(self, market: str) -> pd.DataFrame:
        """Extract data for specific market (LME, JPX, US_Stock, FX)."""
        if self.train_data is None:
            self.load_all_data()
            
        market_cols = [col for col in self.train_data.columns if col.startswith(market)]
        market_data = self.train_data[['date_id'] + market_cols].copy()
        
        logger.info(f"Extracted {len(market_cols)} columns for {market} market")
        return market_data
    
    def get_price_columns(self) -> List[str]:
        """Get all price-related columns (Close, Open, High, Low)."""
        if self.train_data is None:
            self.load_all_data()
            
        price_keywords = ['Close', 'Open', 'High', 'Low', 'adj_close', 'adj_open', 'adj_high', 'adj_low']
        price_cols = []
        
        for col in self.train_data.columns:
            if any(keyword in col for keyword in price_keywords):
                price_cols.append(col)
                
        return price_cols
    
    def get_volume_columns(self) -> List[str]:
        """Get all volume-related columns."""
        if self.train_data is None:
            self.load_all_data()
            
        volume_cols = [col for col in self.train_data.columns if 'volume' in col.lower()]
        return volume_cols
    
    def create_price_differences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price difference features based on target_pairs."""
        if self.target_pairs is None:
            self.load_all_data()
            
        df_diff = df.copy()
        
        for _, row in self.target_pairs.iterrows():
            target_name = row['target']
            pair = row['pair']
            
            if ' - ' in pair:
                # Handle price differences
                asset1, asset2 = pair.split(' - ')
                if asset1 in df.columns and asset2 in df.columns:
                    df_diff[f'{target_name}_diff'] = df[asset1] - df[asset2]
            else:
                # Handle single asset (log returns)
                if pair in df.columns:
                    df_diff[f'{target_name}_log_return'] = np.log(df[pair] / df[pair].shift(1))
        
        return df_diff
    
    def preprocess_data(self, df: pd.DataFrame, fill_method: str = 'forward') -> pd.DataFrame:
        """Preprocess data with missing value handling."""
        df_processed = df.copy()
        
        if fill_method == 'forward':
            df_processed = df_processed.fillna(method='ffill')
        elif fill_method == 'backward':
            df_processed = df_processed.fillna(method='bfill')
        elif fill_method == 'interpolate':
            df_processed = df_processed.interpolate()
        elif fill_method == 'zero':
            df_processed = df_processed.fillna(0)
        
        # Fill any remaining NaN values
        df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
        
        return df_processed
    
    def create_lagged_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Create lagged features for time series modeling."""
        df_lagged = df.copy()
        price_cols = self.get_price_columns()
        
        for col in price_cols:
            if col in df.columns:
                for lag in lags:
                    df_lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df_lagged
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators for price data."""
        df_tech = df.copy()
        price_cols = self.get_price_columns()
        
        for col in price_cols:
            if col in df.columns:
                # Moving averages
                df_tech[f'{col}_ma_5'] = df[col].rolling(window=5).mean()
                df_tech[f'{col}_ma_10'] = df[col].rolling(window=10).mean()
                df_tech[f'{col}_ma_20'] = df[col].rolling(window=20).mean()
                
                # Volatility
                df_tech[f'{col}_volatility_5'] = df[col].rolling(window=5).std()
                df_tech[f'{col}_volatility_10'] = df[col].rolling(window=10).std()
                
                # Price changes
                df_tech[f'{col}_pct_change'] = df[col].pct_change()
                df_tech[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
                
                # RSI-like indicator
                delta = df[col].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df_tech[f'{col}_rsi'] = 100 - (100 / (1 + rs))
        
        return df_tech
