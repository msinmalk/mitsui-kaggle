"""
Trading strategy implementation for commodity price prediction.
Converts model predictions into actionable trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Trading signal types."""
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class TradingSignal:
    """Trading signal data structure."""
    asset: str
    signal: SignalType
    confidence: float
    price: float
    timestamp: int
    target_name: str

class BaseTradingStrategy:
    """Base class for trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.signals = []
        self.positions = {}
        self.performance_metrics = {}
    
    def generate_signals(self, predictions: pd.DataFrame, 
                        prices: pd.DataFrame, 
                        target_pairs: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals from predictions."""
        raise NotImplementedError
    
    def backtest(self, signals: List[TradingSignal], 
                 prices: pd.DataFrame, 
                 initial_capital: float = 100000) -> Dict[str, float]:
        """Backtest the trading strategy."""
        raise NotImplementedError

class ThresholdStrategy(BaseTradingStrategy):
    """Strategy based on prediction thresholds."""
    
    def __init__(self, buy_threshold: float = 0.01, sell_threshold: float = -0.01, 
                 confidence_threshold: float = 0.6):
        super().__init__("ThresholdStrategy")
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.confidence_threshold = confidence_threshold
    
    def generate_signals(self, predictions: pd.DataFrame, 
                        prices: pd.DataFrame, 
                        target_pairs: pd.DataFrame) -> List[TradingSignal]:
        """Generate signals based on prediction thresholds."""
        signals = []
        
        for _, row in target_pairs.iterrows():
            target_name = row['target']
            pair = row['pair']
            
            if target_name not in predictions.columns:
                continue
            
            # Get predictions for this target
            pred_values = predictions[target_name].values
            dates = predictions['date_id'].values if 'date_id' in predictions.columns else range(len(pred_values))
            
            for i, (pred, date) in enumerate(zip(pred_values, dates)):
                if np.isnan(pred):
                    continue
                
                # Determine signal based on threshold
                if pred > self.buy_threshold:
                    signal_type = SignalType.BUY
                    confidence = min(abs(pred) / self.buy_threshold, 1.0)
                elif pred < self.sell_threshold:
                    signal_type = SignalType.SELL
                    confidence = min(abs(pred) / abs(self.sell_threshold), 1.0)
                else:
                    signal_type = SignalType.HOLD
                    confidence = 0.0
                
                # Only generate signal if confidence is above threshold
                if confidence >= self.confidence_threshold:
                    # Get current price for the asset
                    if ' - ' in pair:
                        # Price difference - use first asset
                        asset_name = pair.split(' - ')[0]
                    else:
                        # Single asset
                        asset_name = pair
                    
                    # Find price for this asset and date
                    price = self._get_asset_price(asset_name, date, prices)
                    
                    if price is not None:
                        signal = TradingSignal(
                            asset=asset_name,
                            signal=signal_type,
                            confidence=confidence,
                            price=price,
                            timestamp=date,
                            target_name=target_name
                        )
                        signals.append(signal)
        
        self.signals = signals
        logger.info(f"Generated {len(signals)} trading signals")
        return signals
    
    def _get_asset_price(self, asset_name: str, date: int, prices: pd.DataFrame) -> Optional[float]:
        """Get price for a specific asset and date."""
        if asset_name not in prices.columns:
            return None
        
        # Find the closest date
        date_col = 'date_id' if 'date_id' in prices.columns else prices.index.name
        if date_col in prices.columns:
            price_row = prices[prices[date_col] == date]
            if not price_row.empty:
                return price_row[asset_name].iloc[0]
        
        return None

class MomentumStrategy(BaseTradingStrategy):
    """Strategy based on momentum and trend following."""
    
    def __init__(self, lookback_period: int = 5, momentum_threshold: float = 0.02):
        super().__init__("MomentumStrategy")
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
    
    def generate_signals(self, predictions: pd.DataFrame, 
                        prices: pd.DataFrame, 
                        target_pairs: pd.DataFrame) -> List[TradingSignal]:
        """Generate signals based on momentum."""
        signals = []
        
        for _, row in target_pairs.iterrows():
            target_name = row['target']
            pair = row['pair']
            
            if target_name not in predictions.columns:
                continue
            
            # Get predictions for this target
            pred_values = predictions[target_name].values
            dates = predictions['date_id'].values if 'date_id' in predictions.columns else range(len(pred_values))
            
            # Calculate momentum
            for i in range(self.lookback_period, len(pred_values)):
                if np.isnan(pred_values[i]):
                    continue
                
                # Calculate momentum as change over lookback period
                momentum = pred_values[i] - pred_values[i - self.lookback_period]
                
                # Determine signal based on momentum
                if momentum > self.momentum_threshold:
                    signal_type = SignalType.BUY
                    confidence = min(abs(momentum) / self.momentum_threshold, 1.0)
                elif momentum < -self.momentum_threshold:
                    signal_type = SignalType.SELL
                    confidence = min(abs(momentum) / self.momentum_threshold, 1.0)
                else:
                    signal_type = SignalType.HOLD
                    confidence = 0.0
                
                if confidence > 0:
                    # Get asset name and price
                    if ' - ' in pair:
                        asset_name = pair.split(' - ')[0]
                    else:
                        asset_name = pair
                    
                    price = self._get_asset_price(asset_name, dates[i], prices)
                    
                    if price is not None:
                        signal = TradingSignal(
                            asset=asset_name,
                            signal=signal_type,
                            confidence=confidence,
                            price=price,
                            timestamp=dates[i],
                            target_name=target_name
                        )
                        signals.append(signal)
        
        self.signals = signals
        logger.info(f"Generated {len(signals)} momentum signals")
        return signals
    
    def _get_asset_price(self, asset_name: str, date: int, prices: pd.DataFrame) -> Optional[float]:
        """Get price for a specific asset and date."""
        if asset_name not in prices.columns:
            return None
        
        date_col = 'date_id' if 'date_id' in prices.columns else prices.index.name
        if date_col in prices.columns:
            price_row = prices[prices[date_col] == date]
            if not price_row.empty:
                return price_row[asset_name].iloc[0]
        
        return None

class PortfolioManager:
    """Manages portfolio and position sizing."""
    
    def __init__(self, initial_capital: float = 100000, max_position_size: float = 0.1):
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.portfolio_value_history = []
    
    def calculate_position_size(self, signal: TradingSignal, 
                              current_prices: Dict[str, float]) -> float:
        """Calculate position size based on signal and risk management."""
        if signal.asset not in current_prices:
            return 0.0
        
        # Kelly criterion for position sizing
        win_rate = 0.6  # Assume 60% win rate
        avg_win = 0.02  # Assume 2% average win
        avg_loss = 0.01  # Assume 1% average loss
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
        
        # Adjust by confidence
        position_size = kelly_fraction * signal.confidence
        
        # Calculate dollar amount
        position_value = self.capital * position_size
        
        return position_value
    
    def execute_trade(self, signal: TradingSignal, current_prices: Dict[str, float]):
        """Execute a trade based on signal."""
        if signal.asset not in current_prices:
            return
        
        current_price = current_prices[signal.asset]
        position_value = self.calculate_position_size(signal, current_prices)
        
        if position_value <= 0:
            return
        
        # Calculate shares
        shares = position_value / current_price
        
        # Update positions
        if signal.asset not in self.positions:
            self.positions[signal.asset] = 0
        
        if signal.signal == SignalType.BUY:
            self.positions[signal.asset] += shares
            self.capital -= position_value
        elif signal.signal == SignalType.SELL:
            self.positions[signal.asset] -= shares
            self.capital += position_value
        
        # Record trade
        trade = {
            'timestamp': signal.timestamp,
            'asset': signal.asset,
            'signal': signal.signal.name,
            'shares': shares,
            'price': current_price,
            'value': position_value,
            'confidence': signal.confidence
        }
        self.trade_history.append(trade)
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value."""
        total_value = self.capital
        
        for asset, shares in self.positions.items():
            if asset in current_prices and shares > 0:
                total_value += shares * current_prices[asset]
        
        return total_value
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not self.portfolio_value_history:
            return {}
        
        returns = np.diff(self.portfolio_value_history) / self.portfolio_value_history[:-1]
        
        metrics = {
            'total_return': (self.portfolio_value_history[-1] - self.initial_capital) / self.initial_capital,
            'annualized_return': np.mean(returns) * 252,  # Assuming daily data
            'volatility': np.std(returns) * np.sqrt(252),
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'win_rate': self._calculate_win_rate()
        }
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        peak = self.portfolio_value_history[0]
        max_dd = 0
        
        for value in self.portfolio_value_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history."""
        if not self.trade_history:
            return 0.0
        
        # This is a simplified calculation
        # In practice, you'd track P&L for each trade
        return 0.6  # Placeholder

class Backtester:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.portfolio_manager = PortfolioManager(initial_capital)
    
    def run_backtest(self, strategy: BaseTradingStrategy, 
                    predictions: pd.DataFrame,
                    prices: pd.DataFrame,
                    target_pairs: pd.DataFrame) -> Dict[str, float]:
        """Run backtest for a trading strategy."""
        logger.info(f"Running backtest for {strategy.name}")
        
        # Generate signals
        signals = strategy.generate_signals(predictions, prices, target_pairs)
        
        if not signals:
            logger.warning("No signals generated")
            return {}
        
        # Group signals by timestamp
        signals_by_time = {}
        for signal in signals:
            if signal.timestamp not in signals_by_time:
                signals_by_time[signal.timestamp] = []
            signals_by_time[signal.timestamp].append(signal)
        
        # Execute trades chronologically
        for timestamp in sorted(signals_by_time.keys()):
            # Get current prices for this timestamp
            current_prices = self._get_prices_at_timestamp(timestamp, prices)
            
            # Execute all signals for this timestamp
            for signal in signals_by_time[timestamp]:
                self.portfolio_manager.execute_trade(signal, current_prices)
            
            # Record portfolio value
            portfolio_value = self.portfolio_manager.calculate_portfolio_value(current_prices)
            self.portfolio_manager.portfolio_value_history.append(portfolio_value)
        
        # Calculate performance metrics
        metrics = self.portfolio_manager.get_performance_metrics()
        
        logger.info(f"Backtest completed. Total return: {metrics.get('total_return', 0):.2%}")
        
        return metrics
    
    def _get_prices_at_timestamp(self, timestamp: int, prices: pd.DataFrame) -> Dict[str, float]:
        """Get prices for all assets at a specific timestamp."""
        current_prices = {}
        
        date_col = 'date_id' if 'date_id' in prices.columns else prices.index.name
        if date_col in prices.columns:
            price_row = prices[prices[date_col] == timestamp]
            if not price_row.empty:
                for col in prices.columns:
                    if col != date_col:
                        current_prices[col] = price_row[col].iloc[0]
        
        return current_prices
