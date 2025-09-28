"""
Feature engineering module for commodity price prediction.
Creates advanced features from raw price and volume data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import ta
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for commodity prediction."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = []
        
    def create_price_features(self, df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
        """Create comprehensive price-based features."""
        df_features = df.copy()
        new_features = []
        
        for col in price_cols:
            if col not in df.columns:
                continue
                
            # Collect all new features for this column
            col_features = {}
            
            # Basic price features
            col_features[f'{col}_log_price'] = np.log(df[col] + 1e-8)
            col_features[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()
            
            # Rolling statistics
            for window in [5, 10, 20, 50]:
                ma = df[col].rolling(window).mean()
                col_features[f'{col}_ma_{window}'] = ma
                col_features[f'{col}_std_{window}'] = df[col].rolling(window).std()
                col_features[f'{col}_min_{window}'] = df[col].rolling(window).min()
                col_features[f'{col}_max_{window}'] = df[col].rolling(window).max()
                
                # Price relative to moving average
                col_features[f'{col}_ma_ratio_{window}'] = df[col] / ma
                
            # Momentum features
            for period in [1, 3, 5, 10, 20]:
                col_features[f'{col}_momentum_{period}'] = df[col] / df[col].shift(period)
                col_features[f'{col}_return_{period}'] = df[col].pct_change(period, fill_method=None)
                col_features[f'{col}_log_return_{period}'] = np.log(df[col] / df[col].shift(period))
            
            # Volatility features
            for window in [5, 10, 20]:
                returns = df[col].pct_change(fill_method=None)
                col_features[f'{col}_volatility_{window}'] = returns.rolling(window).std()
                col_features[f'{col}_skewness_{window}'] = returns.rolling(window).skew()
                col_features[f'{col}_kurtosis_{window}'] = returns.rolling(window).kurt()
            
            # Add to new features list
            new_features.append(pd.DataFrame(col_features, index=df.index))
        
        # Concatenate all new features at once
        if new_features:
            df_features = pd.concat([df_features] + new_features, axis=1)
        
        return df_features
    
    def create_technical_indicators(self, df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
        """Create technical analysis indicators."""
        df_tech = df.copy()
        new_features = []
        
        for col in price_cols:
            if col not in df.columns:
                continue
                
            try:
                # Collect all new features for this column
                col_features = {}
                
                # RSI
                col_features[f'{col}_rsi_14'] = ta.momentum.RSIIndicator(df[col], window=14).rsi()
                col_features[f'{col}_rsi_7'] = ta.momentum.RSIIndicator(df[col], window=7).rsi()
                
                # MACD
                macd = ta.trend.MACD(df[col])
                col_features[f'{col}_macd'] = macd.macd()
                col_features[f'{col}_macd_signal'] = macd.macd_signal()
                col_features[f'{col}_macd_histogram'] = macd.macd_diff()
                
                # Bollinger Bands
                bb = ta.volatility.BollingerBands(df[col])
                col_features[f'{col}_bb_upper'] = bb.bollinger_hband()
                col_features[f'{col}_bb_lower'] = bb.bollinger_lband()
                col_features[f'{col}_bb_width'] = bb.bollinger_wband()
                col_features[f'{col}_bb_percent'] = bb.bollinger_pband()
                
                # Stochastic Oscillator
                stoch = ta.momentum.StochasticOscillator(df[col], df[col], df[col])
                col_features[f'{col}_stoch_k'] = stoch.stoch()
                col_features[f'{col}_stoch_d'] = stoch.stoch_signal()
                
                # Williams %R
                col_features[f'{col}_williams_r'] = ta.momentum.WilliamsRIndicator(df[col], df[col], df[col]).williams_r()
                
                # Commodity Channel Index
                col_features[f'{col}_cci'] = ta.trend.CCIIndicator(df[col], df[col], df[col]).cci()
                
                # Add to new features list
                new_features.append(pd.DataFrame(col_features, index=df.index))
                
            except Exception as e:
                logger.warning(f"Could not create technical indicators for {col}: {e}")
                continue
        
        # Concatenate all new features at once
        if new_features:
            df_tech = pd.concat([df_tech] + new_features, axis=1)
        
        return df_tech
    
    def create_cross_asset_features(self, df: pd.DataFrame, target_pairs: pd.DataFrame) -> pd.DataFrame:
        """Create cross-asset correlation and spread features."""
        df_cross = df.copy()
        
        # Get all price columns
        price_cols = [col for col in df.columns if any(x in col for x in ['Close', 'adj_close'])]
        
        # Create correlation features
        for window in [10, 20, 50]:
            corr_matrix = df[price_cols].rolling(window).corr()
            
            # Add pairwise correlations as features
            for i, col1 in enumerate(price_cols[:10]):  # Limit to avoid too many features
                for j, col2 in enumerate(price_cols[:10]):
                    if i < j:
                        try:
                            corr_series = corr_matrix.loc[(slice(None), col1), col2].droplevel(1)
                            df_cross[f'corr_{col1}_{col2}_{window}'] = corr_series
                        except:
                            continue
        
        # Create spread features based on target pairs
        for _, row in target_pairs.iterrows():
            pair = row['pair']
            if ' - ' in pair:
                asset1, asset2 = pair.split(' - ')
                if asset1 in df.columns and asset2 in df.columns:
                    # Spread
                    df_cross[f'spread_{asset1}_{asset2}'] = df[asset1] - df[asset2]
                    # Spread ratio
                    df_cross[f'spread_ratio_{asset1}_{asset2}'] = df[asset1] / (df[asset2] + 1e-8)
                    # Spread z-score
                    spread = df[asset1] - df[asset2]
                    df_cross[f'spread_zscore_{asset1}_{asset2}'] = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
        
        return df_cross
    
    def create_market_regime_features(self, df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
        """Create market regime and regime change features."""
        df_regime = df.copy()
        
        # Market volatility regime
        for col in price_cols[:5]:  # Use first 5 price columns as market proxies
            if col in df.columns:
                returns = df[col].pct_change()
                vol = returns.rolling(20).std()
                
                # Volatility regime (high/medium/low)
                vol_quantiles = vol.quantile([0.33, 0.67])
                df_regime[f'{col}_vol_regime'] = pd.cut(vol, 
                    bins=[-np.inf, vol_quantiles[0.33], vol_quantiles[0.67], np.inf],
                    labels=[0, 1, 2])
                
                # Trend regime
                ma_short = df[col].rolling(10).mean()
                ma_long = df[col].rolling(30).mean()
                df_regime[f'{col}_trend_regime'] = np.where(ma_short > ma_long, 1, 0)
        
        return df_regime
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        df_time = df.copy()
        
        if 'date_id' in df.columns:
            # Convert date_id to datetime if needed
            df_time['date'] = pd.to_datetime(df['date_id'], errors='coerce')
            
            # Time features
            df_time['day_of_week'] = df_time['date'].dt.dayofweek
            df_time['month'] = df_time['date'].dt.month
            df_time['quarter'] = df_time['date'].dt.quarter
            df_time['year'] = df_time['date'].dt.year
            
            # Cyclical encoding
            df_time['day_sin'] = np.sin(2 * np.pi * df_time['day_of_week'] / 7)
            df_time['day_cos'] = np.cos(2 * np.pi * df_time['day_of_week'] / 7)
            df_time['month_sin'] = np.sin(2 * np.pi * df_time['month'] / 12)
            df_time['month_cos'] = np.cos(2 * np.pi * df_time['month'] / 12)
        
        return df_time
    
    def create_lag_features(self, df: pd.DataFrame, target_cols: List[str], lags: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
        """Create lagged features for target variables."""
        df_lagged = df.copy()
        
        for col in target_cols:
            if col in df.columns:
                for lag in lags:
                    df_lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df_lagged
    
    def create_lead_features(self, df: pd.DataFrame, target_cols: List[str], leads: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """Create lead features for target variables (for validation)."""
        df_lead = df.copy()
        
        for col in target_cols:
            if col in df.columns:
                for lead in leads:
                    df_lead[f'{col}_lead_{lead}'] = df[col].shift(-lead)
        
        return df_lead
    
    def scale_features(self, df: pd.DataFrame, feature_cols: List[str], method: str = 'standard') -> pd.DataFrame:
        """Scale features using specified method."""
        df_scaled = df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'standard' or 'robust'")
        
        # Fit and transform features
        df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
        self.scalers[method] = scaler
        
        return df_scaled
    
    def create_all_features(self, df: pd.DataFrame, target_pairs: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Create all features in one pipeline."""
        import time
        start_time = time.time()
        
        if verbose:
            print("🚀 Starting feature engineering pipeline...")
            print(f"   Input shape: {df.shape}")
        
        # Get price columns
        price_cols = [col for col in df.columns if any(x in col for x in ['Close', 'adj_close'])]
        if verbose:
            print(f"   Found {len(price_cols)} price columns")
        
        # Create all feature types with progress tracking
        step_start = time.time()
        df_features = self.create_price_features(df, price_cols)
        if verbose:
            print(f"   ✓ Price features: {df_features.shape[1] - df.shape[1]} new features ({time.time() - step_start:.1f}s)")
        
        step_start = time.time()
        df_features = self.create_technical_indicators(df_features, price_cols)
        if verbose:
            print(f"   ✓ Technical indicators: {df_features.shape[1] - df.shape[1] - (df_features.shape[1] - df.shape[1])} new features ({time.time() - step_start:.1f}s)")
        
        step_start = time.time()
        df_features = self.create_cross_asset_features(df_features, target_pairs)
        if verbose:
            print(f"   ✓ Cross-asset features: {df_features.shape[1] - df.shape[1] - (df_features.shape[1] - df.shape[1])} new features ({time.time() - step_start:.1f}s)")
        
        step_start = time.time()
        df_features = self.create_market_regime_features(df_features, price_cols)
        if verbose:
            print(f"   ✓ Market regime features: {df_features.shape[1] - df.shape[1] - (df_features.shape[1] - df.shape[1])} new features ({time.time() - step_start:.1f}s)")
        
        step_start = time.time()
        df_features = self.create_time_features(df_features)
        if verbose:
            print(f"   ✓ Time features: {df_features.shape[1] - df.shape[1] - (df_features.shape[1] - df.shape[1])} new features ({time.time() - step_start:.1f}s)")
        
        # Store feature columns
        self.feature_columns = [col for col in df_features.columns if col not in df.columns]
        
        total_time = time.time() - start_time
        if verbose:
            print(f"✅ Feature engineering completed!")
            print(f"   Final shape: {df_features.shape}")
            print(f"   New features: {len(self.feature_columns)}")
            print(f"   Total time: {total_time:.1f}s")
            print(f"   Memory usage: ~{df_features.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df_features
