#!/usr/bin/env python3
"""
Evaluation script for commodity price prediction models.
Provides comprehensive evaluation metrics and trading strategy backtesting.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Any
import argparse
import joblib

# Import our modules
from data.loader import CommodityDataLoader
from data.feature_engineering import FeatureEngineer
from models.base_model import BaseModel
from trading.strategy import ThresholdStrategy, MomentumStrategy, Backtester
from trading.portfolio_manager import PortfolioManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, data_dir: str = "data/raw", models_dir: str = "models"):
        self.data_dir = data_dir
        self.models_dir = Path(models_dir)
        self.loader = CommodityDataLoader(data_dir)
        self.feature_engineer = FeatureEngineer()
        
        # Load data
        self.data = self.loader.load_all_data()
        self.train_data = self.loader.preprocess_data(self.data['train'])
        self.test_data = self.loader.preprocess_data(self.data['test'])
        
        # Create features
        self.train_features = self.feature_engineer.create_all_features(
            self.train_data, self.data['target_pairs']
        )
        self.test_features = self.feature_engineer.create_all_features(
            self.test_data, self.data['target_pairs']
        )
        
        # Load models
        self.models = self._load_models()
    
    def _load_models(self) -> Dict[str, BaseModel]:
        """Load all trained models."""
        models = {}
        
        for model_file in self.models_dir.glob("*_model.pkl"):
            try:
                model_name = model_file.stem.replace("_model", "")
                model = joblib.load(model_file)
                models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading {model_file}: {e}")
        
        return models
    
    def evaluate_predictions(self, model_name: str, predictions: np.ndarray, 
                           actual: np.ndarray) -> Dict[str, float]:
        """Evaluate prediction quality."""
        # Calculate metrics
        mse = np.mean((actual - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predictions))
        mape = np.mean(np.abs((actual - predictions) / (actual + 1e-8))) * 100
        
        # Direction accuracy
        actual_direction = np.sign(actual)
        pred_direction = np.sign(predictions)
        direction_accuracy = np.mean(actual_direction == pred_direction)
        
        # Correlation
        correlation = np.corrcoef(actual.flatten(), predictions.flatten())[0, 1]
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'correlation': correlation
        }
        
        return metrics
    
    def evaluate_all_models(self) -> pd.DataFrame:
        """Evaluate all models on test set."""
        logger.info("Evaluating all models...")
        
        # Prepare test data
        target_cols = [col for col in self.data['train_labels'].columns if col.startswith('target_')]
        
        # Align test data with labels (using last part of train labels as test)
        test_with_labels = self.test_features.merge(
            self.data['train_labels'][['date_id'] + target_cols], 
            on='date_id', 
            how='inner'
        )
        
        if test_with_labels.empty:
            logger.warning("No test data with labels available")
            return pd.DataFrame()
        
        # Get feature columns
        feature_cols = [col for col in test_with_labels.columns 
                       if col not in ['date_id'] + target_cols]
        
        X_test = test_with_labels[feature_cols]
        y_test = test_with_labels[target_cols]
        
        # Evaluate each model
        results = []
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Evaluating {model_name}...")
                
                # Make predictions
                predictions = model.predict(X_test)
                
                # Evaluate
                metrics = self.evaluate_predictions(model_name, predictions, y_test.values)
                metrics['model'] = model_name
                
                results.append(metrics)
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        return pd.DataFrame(results)
    
    def plot_predictions(self, model_name: str, target_idx: int = 0, 
                        num_samples: int = 100):
        """Plot predictions vs actual values."""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return
        
        # Get test data
        target_cols = [col for col in self.data['train_labels'].columns if col.startswith('target_')]
        test_with_labels = self.test_features.merge(
            self.data['train_labels'][['date_id'] + target_cols], 
            on='date_id', 
            how='inner'
        )
        
        if test_with_labels.empty:
            logger.warning("No test data available")
            return
        
        feature_cols = [col for col in test_with_labels.columns 
                       if col not in ['date_id'] + target_cols]
        
        X_test = test_with_labels[feature_cols]
        y_test = test_with_labels[target_cols]
        
        # Make predictions
        model = self.models[model_name]
        predictions = model.predict(X_test)
        
        # Plot
        plt.figure(figsize=(15, 8))
        
        # Sample data for plotting
        sample_idx = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
        sample_idx = np.sort(sample_idx)
        
        actual_sample = y_test.iloc[sample_idx, target_idx].values
        pred_sample = predictions[sample_idx, target_idx]
        
        plt.subplot(2, 1, 1)
        plt.plot(sample_idx, actual_sample, label='Actual', alpha=0.7)
        plt.plot(sample_idx, pred_sample, label='Predicted', alpha=0.7)
        plt.title(f'{model_name} - Target {target_idx} Predictions')
        plt.legend()
        plt.ylabel('Value')
        
        plt.subplot(2, 1, 2)
        plt.scatter(actual_sample, pred_sample, alpha=0.6)
        plt.plot([actual_sample.min(), actual_sample.max()], 
                [actual_sample.min(), actual_sample.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Scatter Plot')
        
        plt.tight_layout()
        plt.show()
    
    def backtest_trading_strategies(self) -> Dict[str, Dict[str, float]]:
        """Backtest trading strategies."""
        logger.info("Backtesting trading strategies...")
        
        # Get test data
        target_cols = [col for col in self.data['train_labels'].columns if col.startswith('target_')]
        test_with_labels = self.test_features.merge(
            self.data['train_labels'][['date_id'] + target_cols], 
            on='date_id', 
            how='inner'
        )
        
        if test_with_labels.empty:
            logger.warning("No test data available for backtesting")
            return {}
        
        # Prepare data for backtesting
        feature_cols = [col for col in test_with_labels.columns 
                       if col not in ['date_id'] + target_cols]
        
        X_test = test_with_labels[feature_cols]
        y_test = test_with_labels[target_cols]
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame(y_test.values, columns=target_cols)
        predictions_df['date_id'] = test_with_labels['date_id'].values
        
        # Price data for backtesting
        price_cols = self.loader.get_price_columns()
        price_data = test_with_labels[['date_id'] + price_cols]
        
        # Initialize strategies
        threshold_strategy = ThresholdStrategy(buy_threshold=0.01, sell_threshold=-0.01)
        momentum_strategy = MomentumStrategy(lookback_period=5, momentum_threshold=0.02)
        
        # Initialize backtester
        backtester = Backtester(initial_capital=100000)
        
        # Backtest strategies
        results = {}
        
        try:
            threshold_metrics = backtester.run_backtest(
                threshold_strategy, predictions_df, price_data, self.data['target_pairs']
            )
            results['threshold_strategy'] = threshold_metrics
        except Exception as e:
            logger.error(f"Error backtesting threshold strategy: {e}")
        
        try:
            momentum_metrics = backtester.run_backtest(
                momentum_strategy, predictions_df, price_data, self.data['target_pairs']
            )
            results['momentum_strategy'] = momentum_metrics
        except Exception as e:
            logger.error(f"Error backtesting momentum strategy: {e}")
        
        return results
    
    def generate_report(self, output_file: str = "evaluation_report.html"):
        """Generate comprehensive evaluation report."""
        logger.info("Generating evaluation report...")
        
        # Evaluate all models
        model_results = self.evaluate_all_models()
        
        # Backtest strategies
        strategy_results = self.backtest_trading_strategies()
        
        # Create HTML report
        html_content = f"""
        <html>
        <head>
            <title>Commodity Prediction Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ margin: 10px 0; }}
                .good {{ color: green; }}
                .bad {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Commodity Prediction Model Evaluation Report</h1>
            
            <h2>Model Performance Summary</h2>
            {model_results.to_html(index=False) if not model_results.empty else '<p>No model results available</p>'}
            
            <h2>Trading Strategy Performance</h2>
            {self._format_strategy_results(strategy_results)}
            
            <h2>Recommendations</h2>
            {self._generate_recommendations(model_results, strategy_results)}
        </body>
        </html>
        """
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report saved to {output_file}")
    
    def _format_strategy_results(self, strategy_results: Dict[str, Dict[str, float]]) -> str:
        """Format strategy results for HTML."""
        if not strategy_results:
            return "<p>No strategy results available</p>"
        
        html = "<table><tr><th>Strategy</th><th>Metric</th><th>Value</th></tr>"
        
        for strategy_name, metrics in strategy_results.items():
            for metric_name, value in metrics.items():
                html += f"<tr><td>{strategy_name}</td><td>{metric_name}</td><td>{value:.4f}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_recommendations(self, model_results: pd.DataFrame, 
                                 strategy_results: Dict[str, Dict[str, float]]) -> str:
        """Generate recommendations based on results."""
        recommendations = []
        
        if not model_results.empty:
            # Find best model
            best_model = model_results.loc[model_results['rmse'].idxmin()]
            recommendations.append(f"Best performing model: {best_model['model']} (RMSE: {best_model['rmse']:.4f})")
            
            # Check for overfitting
            if model_results['correlation'].max() < 0.3:
                recommendations.append("Low correlation suggests potential overfitting - consider regularization")
        
        if strategy_results:
            # Check strategy performance
            for strategy_name, metrics in strategy_results.items():
                if 'sharpe_ratio' in metrics and metrics['sharpe_ratio'] > 1.0:
                    recommendations.append(f"{strategy_name} shows good risk-adjusted returns")
                elif 'total_return' in metrics and metrics['total_return'] > 0.1:
                    recommendations.append(f"{strategy_name} shows positive returns")
        
        if not recommendations:
            recommendations.append("No specific recommendations available")
        
        return "<ul>" + "".join([f"<li>{rec}</li>" for rec in recommendations]) + "</ul>"

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate commodity price prediction models')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Data directory')
    parser.add_argument('--models_dir', type=str, default='models', help='Models directory')
    parser.add_argument('--output_file', type=str, default='evaluation_report.html', help='Output report file')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.data_dir, args.models_dir)
    
    # Generate report
    evaluator.generate_report(args.output_file)
    
    print(f"Evaluation completed! Report saved to {args.output_file}")

if __name__ == "__main__":
    main()
