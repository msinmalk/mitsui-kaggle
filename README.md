# Commodity Price Prediction Challenge

## Overview
This project provides a comprehensive solution for the Mitsui Commodity Price Prediction Challenge, developing AI models capable of predicting future commodity returns using historical data from multiple financial markets.

### Data Sources
- **London Metal Exchange (LME)**: Metal commodity prices and trading data
- **Japan Exchange Group (JPX)**: Gold, Platinum, and Rubber futures
- **US Stock Markets**: 100+ US stocks and ETFs across various sectors
- **Forex Markets**: 50+ currency pairs and exchange rates

### Challenge Goals
- Predict 424 target variables representing log returns and price differences
- Extract robust price-movement signals as features
- Deploy AI-driven trading techniques that convert signals into sustainable trading profits
- Achieve stable, long-term forecasts for optimizing trading strategies and managing risk

## Project Structure
```
mitsui/
├── data/                   # Raw and processed datasets
│   ├── raw/               # Original competition data
│   ├── processed/         # Cleaned and feature-engineered data
│   └── external/          # Additional external data sources
├── notebooks/             # Jupyter notebooks for analysis
│   └── 01_data_exploration.ipynb
├── src/                   # Source code
│   ├── data/             # Data processing and feature engineering
│   │   ├── loader.py     # Data loading and preprocessing
│   │   └── feature_engineering.py  # Advanced feature creation
│   ├── models/           # ML models and training
│   │   ├── base_model.py # Base model interface
│   │   ├── traditional_ml.py  # XGBoost, LightGBM, etc.
│   │   ├── neural_networks.py # LSTM, Transformer, CNN-LSTM
│   │   └── ensemble_model.py  # Ensemble methods
│   ├── trading/          # Trading strategy implementation
│   │   └── strategy.py   # Trading strategies and backtesting
│   └── utils/            # Utility functions
├── models/               # Trained model artifacts
├── results/              # Results and outputs
├── train.py              # Main training script
├── evaluate.py           # Model evaluation script
└── requirements.txt      # Python dependencies
```

## Key Features

### 🚀 Advanced Machine Learning
- **Multiple Model Architectures**: LSTM, Transformer, CNN-LSTM, XGBoost, LightGBM, Random Forest
- **Ensemble Methods**: Voting and weighted ensemble models
- **Time Series Modeling**: Specialized for financial time series data
- **Multi-target Prediction**: Handles 424 target variables simultaneously

### 📊 Comprehensive Feature Engineering
- **Price-based Features**: Moving averages, volatility, momentum indicators
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Stochastic Oscillator
- **Cross-asset Features**: Correlations and spreads between different markets
- **Market Regime Features**: Volatility and trend regime detection
- **Time Features**: Cyclical encoding of temporal patterns

### 💼 Trading Strategy Framework
- **Multiple Strategies**: Threshold-based and momentum-based trading
- **Risk Management**: Position sizing using Kelly criterion
- **Portfolio Management**: Multi-asset portfolio optimization
- **Backtesting Engine**: Comprehensive strategy validation

### 📈 Evaluation & Analysis
- **Performance Metrics**: MSE, RMSE, MAE, Direction Accuracy, Sharpe Ratio
- **Trading Metrics**: Total Return, Max Drawdown, Win Rate
- **Visualization**: Prediction plots, correlation analysis, performance charts
- **HTML Reports**: Comprehensive evaluation reports

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd mitsui

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup
```bash
# Place competition data in data/raw/
# The following files should be present:
# - train.csv
# - test.csv
# - train_labels.csv
# - target_pairs.csv
```

### 3. Training Models
```bash
# Train all models
python train.py

# Quick training (fewer models)
python train.py --quick

# Custom data/output directories
python train.py --data_dir data/raw --output_dir models
```

### 4. Evaluation
```bash
# Generate comprehensive evaluation report
python evaluate.py

# Custom evaluation
python evaluate.py --models_dir models --output_file report.html
```

### 5. Data Exploration
```bash
# Launch Jupyter notebook for data exploration
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Model Architecture

### Traditional Machine Learning
- **XGBoost**: Gradient boosting with hyperparameter optimization
- **LightGBM**: Fast gradient boosting with categorical features
- **Random Forest**: Ensemble of decision trees
- **Linear Models**: Ridge, Lasso, ElasticNet regression
- **SVM**: Support Vector Regression

### Deep Learning
- **LSTM**: Long Short-Term Memory networks for sequence modeling
- **Transformer**: Attention-based architecture for time series
- **CNN-LSTM**: Hybrid convolutional and recurrent networks
- **Multi-head Attention**: Captures complex temporal dependencies

### Ensemble Methods
- **Voting Ensemble**: Combines predictions from multiple models
- **Weighted Ensemble**: Learns optimal weights based on performance
- **Stacking**: Meta-learning approach for model combination

## Trading Strategies

### Threshold Strategy
- Generates buy/sell signals based on prediction thresholds
- Configurable confidence levels and thresholds
- Risk-adjusted position sizing

### Momentum Strategy
- Follows trend and momentum patterns
- Lookback period and momentum threshold configuration
- Adaptive to market conditions

### Portfolio Management
- Kelly Criterion for optimal position sizing
- Multi-asset portfolio optimization
- Risk management and drawdown control

## Performance Metrics

### Prediction Quality
- **MSE/RMSE**: Mean squared error and root mean squared error
- **MAE**: Mean absolute error
- **MAPE**: Mean absolute percentage error
- **Direction Accuracy**: Percentage of correct directional predictions
- **Correlation**: Linear correlation between predictions and actuals

### Trading Performance
- **Total Return**: Overall portfolio return
- **Sharpe Ratio**: Risk-adjusted return metric
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Volatility**: Portfolio return volatility

## Configuration

### Model Parameters
Models can be configured through the training script or by modifying the model classes directly:

```python
# Example: XGBoost configuration
xgboost_model = XGBoostModel(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8
)
```

### Trading Strategy Parameters
```python
# Example: Threshold strategy configuration
threshold_strategy = ThresholdStrategy(
    buy_threshold=0.01,
    sell_threshold=-0.01,
    confidence_threshold=0.6
)
```

## Dependencies

### Core Libraries
- **Python 3.8+**
- **pandas, numpy**: Data manipulation and numerical computing
- **scikit-learn**: Traditional machine learning algorithms
- **tensorflow**: Deep learning framework
- **xgboost, lightgbm**: Gradient boosting libraries

### Visualization
- **matplotlib, seaborn**: Statistical plotting
- **plotly**: Interactive visualizations

### Financial Data
- **ta-lib**: Technical analysis indicators
- **yfinance**: Additional financial data sources

### Development
- **jupyter**: Interactive notebooks
- **pytest**: Unit testing framework

## Results

The framework generates several outputs:

1. **Trained Models**: Saved in `models/` directory
2. **Predictions**: CSV files with model predictions
3. **Evaluation Report**: HTML report with comprehensive analysis
4. **Trading Signals**: Generated trading signals for backtesting
5. **Performance Metrics**: Detailed performance analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Mitsui & Co., Ltd. for providing the competition data
- AlpacaTech Co., Ltd. for technical support
- Exchange Rates API from APILayer for Forex data

