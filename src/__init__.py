"""
Q8 Galois Extension Prime Bias Experiment

Complete experimental environment for Quaternion Galois extension prime bias analysis
based on Aoki-Koyama arXiv:2203.12266 with all 13 Omar S. cases.

Modules:
- omar_polynomials: 13 case polynomial definitions
- fast_frobenius_calculator: Optimized Frobenius element computation
- bias_analyzer: Bias analysis and graph generation
- experiment_runner: Main experiment management
- utils: Utility functions
"""

__version__ = "1.0.0"
__author__ = "Q8 Prime Bias Research Team"
__email__ = "research@example.com"
__description__ = "Numerical experiments for Chebyshev bias in Q8 Galois extensions"

# Import main classes for convenience
try:
    from .omar_polynomials import (
        get_case,
        get_all_cases,
        print_all_cases_summary,
        OMAR_COLLECTION
    )
    from .experiment_runner import QuaternionBiasExperiment
    from .bias_analyzer import BiasAnalyzer
    
    __all__ = [
        'get_case',
        'get_all_cases', 
        'print_all_cases_summary',
        'OMAR_COLLECTION',
        'QuaternionBiasExperiment',
        'BiasAnalyzer'
    ]
    
except ImportError as e:
    # Handle import errors gracefully during development
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}")
    __all__ = []

# Package metadata
__package_info__ = {
    'name': 'q8-prime-bias-experiment',
    'version': __version__,
    'description': __description__,
    'author': __author__,
    'license': 'MIT',
    'url': 'https://github.com/jxta/q8-prime-bias-experiment',
    'keywords': ['galois theory', 'prime bias', 'quaternions', 'number theory'],
    'python_requires': '>=3.8'
}