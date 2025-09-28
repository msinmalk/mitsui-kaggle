"""
Neural network models for commodity price prediction.
Includes LSTM, GRU, and Transformer architectures.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Input, 
    MultiHeadAttention, LayerNormalization,
    TimeDistributed, BatchNormalization,
    Conv1D, MaxPooling1D, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class LSTMModel(BaseModel):
    """LSTM model for time series prediction."""
    
    def __init__(self, sequence_length: int = 30, hidden_units: int = 64, 
                 dropout_rate: float = 0.2, **kwargs):
        super().__init__("LSTMModel", **kwargs)
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.scaler = StandardScaler()
        
    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences for LSTM input."""
        X_seq = []
        y_seq = []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def _build_model(self, input_shape: Tuple, output_dim: int) -> Model:
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(self.hidden_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            LSTM(self.hidden_units // 2, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(64, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(output_dim)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'LSTMModel':
        """Fit the LSTM model."""
        logger.info("Fitting LSTM model...")
        
        # Store feature and target columns
        self.feature_columns = X.columns.tolist()
        self.target_columns = y.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        y_values = y.values
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_values)
        
        # Build model
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        output_dim = y.shape[1]
        self.model = self._build_model(input_shape, output_dim)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train model
        self.model.fit(
            X_seq, y_seq,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_fitted = True
        logger.info("LSTM model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using LSTM."""
        X_validated = self.validate_input(X)
        X_scaled = self.scaler.transform(X_validated)
        
        # Create sequences for prediction
        X_seq, _ = self._create_sequences(X_scaled)
        
        # Make predictions
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Pad predictions to match input length
        padded_predictions = np.zeros((len(X), predictions.shape[1]))
        padded_predictions[self.sequence_length:] = predictions
        
        return padded_predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Not applicable for regression."""
        raise NotImplementedError("Probability prediction not available for regression")

class TransformerModel(BaseModel):
    """Transformer model for time series prediction."""
    
    def __init__(self, sequence_length: int = 30, d_model: int = 64, 
                 num_heads: int = 8, num_layers: int = 2, **kwargs):
        super().__init__("TransformerModel", **kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.scaler = StandardScaler()
        
    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences for Transformer input."""
        X_seq = []
        y_seq = []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def _transformer_encoder(self, inputs, d_model, num_heads, num_layers):
        """Create transformer encoder layers."""
        x = inputs
        
        for _ in range(num_layers):
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model
            )(x, x)
            attention_output = Dropout(0.1)(attention_output)
            x = LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # Feed forward
            ffn = Sequential([
                Dense(d_model * 4, activation='relu'),
                Dense(d_model)
            ])
            ffn_output = ffn(x)
            ffn_output = Dropout(0.1)(ffn_output)
            x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        return x
    
    def _build_model(self, input_shape: Tuple, output_dim: int) -> Model:
        """Build Transformer model architecture."""
        inputs = Input(shape=input_shape)
        
        # Positional encoding
        x = Dense(self.d_model)(inputs)
        
        # Transformer encoder
        x = self._transformer_encoder(x, self.d_model, self.num_heads, self.num_layers)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Output layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(output_dim)(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'TransformerModel':
        """Fit the Transformer model."""
        logger.info("Fitting Transformer model...")
        
        # Store feature and target columns
        self.feature_columns = X.columns.tolist()
        self.target_columns = y.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        y_values = y.values
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_values)
        
        # Build model
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        output_dim = y.shape[1]
        self.model = self._build_model(input_shape, output_dim)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train model
        self.model.fit(
            X_seq, y_seq,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_fitted = True
        logger.info("Transformer model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using Transformer."""
        X_validated = self.validate_input(X)
        X_scaled = self.scaler.transform(X_validated)
        
        # Create sequences for prediction
        X_seq, _ = self._create_sequences(X_scaled)
        
        # Make predictions
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Pad predictions to match input length
        padded_predictions = np.zeros((len(X), predictions.shape[1]))
        padded_predictions[self.sequence_length:] = predictions
        
        return padded_predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Not applicable for regression."""
        raise NotImplementedError("Probability prediction not available for regression")

class CNNLSTMModel(BaseModel):
    """CNN-LSTM hybrid model for time series prediction."""
    
    def __init__(self, sequence_length: int = 30, cnn_filters: int = 64, 
                 lstm_units: int = 64, **kwargs):
        super().__init__("CNNLSTMModel", **kwargs)
        self.sequence_length = sequence_length
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.scaler = StandardScaler()
        
    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences for CNN-LSTM input."""
        X_seq = []
        y_seq = []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def _build_model(self, input_shape: Tuple, output_dim: int) -> Model:
        """Build CNN-LSTM model architecture."""
        model = Sequential([
            Conv1D(filters=self.cnn_filters, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=self.cnn_filters//2, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(output_dim)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'CNNLSTMModel':
        """Fit the CNN-LSTM model."""
        logger.info("Fitting CNN-LSTM model...")
        
        # Store feature and target columns
        self.feature_columns = X.columns.tolist()
        self.target_columns = y.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        y_values = y.values
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_values)
        
        # Build model
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        output_dim = y.shape[1]
        self.model = self._build_model(input_shape, output_dim)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train model
        self.model.fit(
            X_seq, y_seq,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_fitted = True
        logger.info("CNN-LSTM model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using CNN-LSTM."""
        X_validated = self.validate_input(X)
        X_scaled = self.scaler.transform(X_validated)
        
        # Create sequences for prediction
        X_seq, _ = self._create_sequences(X_scaled)
        
        # Make predictions
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Pad predictions to match input length
        padded_predictions = np.zeros((len(X), predictions.shape[1]))
        padded_predictions[self.sequence_length:] = predictions
        
        return padded_predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Not applicable for regression."""
        raise NotImplementedError("Probability prediction not available for regression")
