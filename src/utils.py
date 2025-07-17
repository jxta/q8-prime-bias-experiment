#!/usr/bin/env python3
"""
Utility functions for Q8 Prime Bias Experiment

Common utility functions for mathematical computations, file I/O, 
performance monitoring, and data processing.
"""

import os
import time
import json
import yaml
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import sympy as sp
from sympy import isprime, sieve
import multiprocessing as mp
from functools import wraps

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting: {self.description}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"Completed: {self.description} in {duration:.2f}s")
    
    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

def timing_decorator(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(f"{func.__name__}"):
            return func(*args, **kwargs)
    return wrapper

class DirectoryManager:
    """Manage directory structure for the experiment"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all necessary directories exist"""
        directories = [
            'data/results',
            'graphs/output', 
            'logs',
            'config',
            'tmp'
        ]
        
        for dir_path in directories:
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {full_path}")
    
    def get_path(self, relative_path: str) -> Path:
        """Get full path for a relative path"""
        return self.base_dir / relative_path
    
    def get_data_path(self, filename: str) -> Path:
        """Get path in data/results directory"""
        return self.base_dir / 'data' / 'results' / filename
    
    def get_graph_path(self, filename: str) -> Path:
        """Get path in graphs/output directory"""
        return self.base_dir / 'graphs' / 'output' / filename

class ConfigManager:
    """Manage experiment configuration"""
    
    def __init__(self, config_path: str = "config/experiment_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return self.get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'computation': {
                'small_scale': 10000,
                'medium_scale': 100000,
                'large_scale': 1000000,
                'huge_scale': 1000000000,
                'parallel': {
                    'enabled': True,
                    'max_processes': None,
                    'batch_size': 50000
                }
            },
            'visualization': {
                'graphs': {
                    'figsize': [18, 12],
                    'dpi': 150
                },
                'sampling': {
                    'target_points': 500
                }
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key path"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Saved config to {self.config_path}")

def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    
    logger.info(f"Saved JSON data to {filepath}")

def load_json(filepath: Union[str, Path]) -> Any:
    """Load data from JSON file"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded JSON data from {filepath}")
    return data

def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """Save data to pickle file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"Saved pickle data to {filepath}")

def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load data from pickle file"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded pickle data from {filepath}")
    return data

def get_system_info() -> Dict[str, Any]:
    """Get system information for performance monitoring"""
    import platform
    import psutil
    
    try:
        return {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': mp.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3)
        }
    except ImportError:
        return {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'cpu_count': mp.cpu_count()
        }

def estimate_memory_usage(max_prime: int, num_cases: int = 1) -> Dict[str, float]:
    """Estimate memory usage for computation"""
    # Rough estimates based on data structures
    prime_count = max_prime // (np.log(max_prime) if max_prime > 1 else 1)
    
    # Memory per prime (dict entry + metadata)
    bytes_per_prime = 32  # Conservative estimate
    
    # Total memory for frobenius data
    frobenius_memory_mb = (prime_count * bytes_per_prime * num_cases) / (1024**2)
    
    # Additional overhead
    overhead_mb = frobenius_memory_mb * 0.5
    
    return {
        'estimated_primes': prime_count,
        'frobenius_data_mb': frobenius_memory_mb,
        'overhead_mb': overhead_mb,
        'total_mb': frobenius_memory_mb + overhead_mb,
        'total_gb': (frobenius_memory_mb + overhead_mb) / 1024
    }

def estimate_computation_time(
    max_prime: int, 
    num_cases: int = 1, 
    num_processes: int = 1,
    base_time_per_prime_us: float = 10.0
) -> Dict[str, float]:
    """Estimate computation time"""
    prime_count = max_prime // (np.log(max_prime) if max_prime > 1 else 1)
    
    # Base computation time
    total_operations = prime_count * num_cases
    base_time_seconds = (total_operations * base_time_per_prime_us) / 1e6
    
    # Parallel processing efficiency (assume 80% efficiency)
    parallel_efficiency = 0.8
    parallel_time_seconds = base_time_seconds / (num_processes * parallel_efficiency)
    
    return {
        'estimated_primes': prime_count,
        'total_operations': total_operations,
        'base_time_seconds': base_time_seconds,
        'base_time_minutes': base_time_seconds / 60,
        'base_time_hours': base_time_seconds / 3600,
        'parallel_time_seconds': parallel_time_seconds,
        'parallel_time_minutes': parallel_time_seconds / 60,
        'parallel_time_hours': parallel_time_seconds / 3600,
        'speedup_factor': base_time_seconds / parallel_time_seconds
    }

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def format_number(num: Union[int, float]) -> str:
    """Format large numbers with appropriate suffixes"""
    if isinstance(num, float):
        if num < 1000:
            return f"{num:.1f}"
        elif num < 1e6:
            return f"{num/1e3:.1f}K"
        elif num < 1e9:
            return f"{num/1e6:.1f}M"
        else:
            return f"{num/1e9:.1f}B"
    else:
        if num < 1000:
            return str(num)
        elif num < 1e6:
            return f"{num//1000}K"
        elif num < 1e9:
            return f"{num//1000000}M"
        else:
            return f"{num//1000000000}B"

def kronecker_symbol(a: int, n: int) -> int:
    """Compute Kronecker symbol (optimized version)"""
    if n == 0:
        return 1 if abs(a) == 1 else 0
    if n < 0:
        result = kronecker_symbol(a, -n)
        if a < 0:
            result *= -1
        return result
    
    # Use sympy for reliability
    return sp.jacobi_symbol(a, n)

def legendre_symbol(a: int, p: int) -> int:
    """Compute Legendre symbol"""
    if not isprime(p):
        raise ValueError(f"p={p} must be prime")
    
    if a % p == 0:
        return 0
    
    # Use Euler's criterion: a^((p-1)/2) mod p
    result = pow(a, (p - 1) // 2, p)
    return -1 if result == p - 1 else result

def generate_prime_batches(
    start_prime: int, 
    end_prime: int, 
    batch_size: int
) -> List[Tuple[int, int]]:
    """Generate prime range batches for parallel processing"""
    batches = []
    current_start = start_prime
    
    while current_start < end_prime:
        current_end = min(current_start + batch_size, end_prime)
        batches.append((current_start, current_end))
        current_start = current_end
    
    return batches

def get_optimal_process_count() -> int:
    """Get optimal number of processes for computation"""
    cpu_count = mp.cpu_count()
    # Use 75% of available cores, minimum 1
    return max(1, int(cpu_count * 0.75))

class ProgressTracker:
    """Track and display progress for long-running computations"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, increment: int = 1) -> None:
        """Update progress"""
        self.current += increment
        current_time = time.time()
        
        # Update display every second
        if current_time - self.last_update >= 1.0:
            self.display_progress()
            self.last_update = current_time
    
    def display_progress(self) -> None:
        """Display current progress"""
        if self.total == 0:
            return
        
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = format_duration(eta)
        else:
            eta_str = "Unknown"
        
        logger.info(
            f"{self.description}: {self.current}/{self.total} "
            f"({percentage:.1f}%) - ETA: {eta_str}"
        )
    
    def finish(self) -> None:
        """Mark as completed"""
        self.current = self.total
        elapsed = time.time() - self.start_time
        logger.info(
            f"{self.description}: Completed in {format_duration(elapsed)}"
        )

# Global instances
directory_manager = DirectoryManager()
config_manager = ConfigManager()

if __name__ == "__main__":
    # Demo usage
    print("Utility Functions Demo")
    print("=====================")
    
    # System info
    print("\nSystem Information:")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Memory estimation
    print("\nMemory Estimation for 10^6 primes:")
    memory_est = estimate_memory_usage(10**6, num_cases=3)
    for key, value in memory_est.items():
        print(f"  {key}: {value}")
    
    # Time estimation
    print("\nTime Estimation for 10^6 primes (8 processes):")
    time_est = estimate_computation_time(10**6, num_cases=3, num_processes=8)
    for key, value in time_est.items():
        if 'time' in key:
            print(f"  {key}: {format_duration(value) if 'seconds' in key else value}")
        else:
            print(f"  {key}: {format_number(value) if isinstance(value, (int, float)) else value}")
    
    print("\n✅ Demo complete!")