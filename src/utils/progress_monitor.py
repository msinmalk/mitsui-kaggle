"""
Progress monitoring utilities for the commodity prediction project.
Provides real-time progress tracking and performance monitoring.
"""

import time
import psutil
import pandas as pd
from typing import Dict, Any, Optional
from contextlib import contextmanager

class ProgressMonitor:
    """Monitor progress and performance of long-running operations."""
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.checkpoints = {}
        self.memory_usage = []
        
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.memory_usage = []
        print(f"🚀 Starting {self.operation_name}...")
        self._log_memory()
        
    def checkpoint(self, name: str, data: Optional[Any] = None):
        """Record a checkpoint with optional data info."""
        if self.start_time is None:
            self.start()
            
        elapsed = time.time() - self.start_time
        self.checkpoints[name] = {
            'time': elapsed,
            'memory_mb': self._get_memory_usage(),
            'data_shape': getattr(data, 'shape', None) if data is not None else None
        }
        
        print(f"   ✓ {name}: {elapsed:.1f}s, {self._get_memory_usage():.1f}MB")
        if data is not None and hasattr(data, 'shape'):
            print(f"      Data shape: {data.shape}")
            
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
        
    def _log_memory(self):
        """Log current memory usage."""
        memory = self._get_memory_usage()
        self.memory_usage.append(memory)
        
    def finish(self, final_data: Optional[Any] = None):
        """Finish monitoring and print summary."""
        if self.start_time is None:
            return
            
        total_time = time.time() - self.start_time
        final_memory = self._get_memory_usage()
        
        print(f"✅ {self.operation_name} completed!")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Final memory: {final_memory:.1f}MB")
        print(f"   Memory peak: {max(self.memory_usage):.1f}MB")
        
        if final_data is not None and hasattr(final_data, 'shape'):
            print(f"   Final data shape: {final_data.shape}")
            
        return {
            'total_time': total_time,
            'final_memory': final_memory,
            'peak_memory': max(self.memory_usage),
            'checkpoints': self.checkpoints
        }

@contextmanager
def monitor_progress(operation_name: str, data: Optional[Any] = None):
    """Context manager for monitoring progress."""
    monitor = ProgressMonitor(operation_name)
    monitor.start()
    if data is not None:
        monitor.checkpoint("Initial data", data)
    
    try:
        yield monitor
    finally:
        monitor.finish(data)

def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """Log detailed information about a DataFrame."""
    print(f"📊 {name} Information:")
    print(f"   Shape: {df.shape}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Data types: {df.dtypes.value_counts().to_dict()}")
    
def estimate_feature_engineering_time(n_samples: int, n_price_cols: int) -> Dict[str, float]:
    """Estimate time for feature engineering based on data size."""
    # Rough estimates based on typical performance
    base_time = 0.1  # seconds per sample
    price_time = 0.05  # seconds per price column per sample
    technical_time = 0.1  # seconds per price column per sample
    
    total_time = n_samples * (base_time + n_price_cols * (price_time + technical_time))
    
    return {
        'estimated_seconds': total_time,
        'estimated_minutes': total_time / 60,
        'complexity': 'Low' if total_time < 60 else 'Medium' if total_time < 300 else 'High'
    }


