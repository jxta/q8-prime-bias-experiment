#!/usr/bin/env python3
"""
Bias Analyzer for Q8 Prime Bias Experiments

Comprehensive analysis and visualization of Chebyshev bias in Q8 Galois extensions.
Implements the theoretical framework from Aoki-Koyama paper with 5 graph representations
for all 13 Omar S. polynomial cases.

This module provides:
- Weighted prime counting functions π_{1/2}(x) and π_{1/2}(x;σ)  
- Bias calculation and convergence analysis
- 5 comprehensive graph visualizations (S1-S5)
- Statistical analysis and trend fitting
- Comparison with theoretical predictions

Reference:
Aoki, M. and Koyama, S. (2022). Chebyshev's Bias against Splitting and 
Principal Primes in Global Fields. arXiv:2203.12266
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Statistical analysis
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .omar_polynomials import get_case, get_all_cases
from .fast_frobenius_calculator import load_frobenius_results
from .utils import (
    Timer, save_json, format_number, format_duration,
    DirectoryManager
)

# Configure matplotlib style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

@dataclass
class BiasAnalysisResult:
    """Container for bias analysis results"""
    case_id: int
    primes: List[int]
    frobenius_elements: List[int]
    
    # Weighted counting functions
    pi_half: List[float]
    pi_half_sigma: Dict[int, List[float]]  # sigma -> values
    
    # Bias functions S1-S5
    S1: List[float]  # π_{1/2}(x) - 8π_{1/2}(x;1)
    S2: List[float]  # π_{1/2}(x) - 8π_{1/2}(x;-1)  
    S3: List[float]  # π_{1/2}(x) - 4π_{1/2}(x;i)
    S4: List[float]  # π_{1/2}(x) - 4π_{1/2}(x;j)
    S5: List[float]  # π_{1/2}(x) - 4π_{1/2}(x;k)
    
    # Theoretical values
    log_log_x: List[float]
    theoretical_S1: List[float]
    theoretical_S2: List[float]
    theoretical_S3: List[float]
    theoretical_S4: List[float]
    theoretical_S5: List[float]
    
    # Statistical measures
    bias_coefficients: Dict[str, float]
    regression_stats: Dict[str, Dict[str, float]]

class WeightedPrimeCounter:
    """Compute weighted prime counting functions"""
    
    @staticmethod
    def compute_pi_half(primes: List[int], weights: Optional[List[float]] = None) -> List[float]:
        """
        Compute π_{1/2}(x) - weighted prime counting function
        
        Args:
            primes: List of primes up to x
            weights: Optional weights (default: 1/√p)
            
        Returns:
            Cumulative weighted counts
        """
        if weights is None:
            weights = [1.0 / np.sqrt(p) for p in primes]
        
        return np.cumsum(weights).tolist()
    
    @staticmethod
    def compute_pi_half_sigma(
        primes: List[int], 
        frobenius_elements: List[int],
        sigma: int,
        weights: Optional[List[float]] = None
    ) -> List[float]:
        """
        Compute π_{1/2}(x;σ) - weighted counting for specific Frobenius element
        
        Args:
            primes: List of primes
            frobenius_elements: Corresponding Frobenius elements
            sigma: Target Frobenius element (0-7)
            weights: Optional weights
            
        Returns:
            Cumulative weighted counts for primes with Frobenius element σ
        """
        if weights is None:
            weights = [1.0 / np.sqrt(p) for p in primes]
        
        sigma_weights = [w if frob == sigma else 0.0 
                        for w, frob in zip(weights, frobenius_elements)]
        
        return np.cumsum(sigma_weights).tolist()

class BiasCalculator:
    """Calculate bias functions and theoretical values"""
    
    def __init__(self, case_id: int):
        self.case_id = case_id
        self.case = get_case(case_id)
        self.bias_coefficients = self.case.get_bias_coefficients()
        self.conjugacy_sizes = self.case.get_conjugacy_sizes()
        
    def compute_bias_functions(self, analysis_data: BiasAnalysisResult) -> BiasAnalysisResult:
        """
        Compute all 5 bias functions S1-S5
        
        Args:
            analysis_data: Partial analysis data with base counts
            
        Returns:
            Complete analysis data with bias functions
        """
        pi_half = analysis_data.pi_half
        pi_half_sigma = analysis_data.pi_half_sigma
        
        # S1: π_{1/2}(x) - 8π_{1/2}(x;1) [identity element, conjugacy size 1]
        analysis_data.S1 = [
            pi - 8 * sigma1 
            for pi, sigma1 in zip(pi_half, pi_half_sigma.get(0, [0]*len(pi_half)))
        ]
        
        # S2: π_{1/2}(x) - 8π_{1/2}(x;-1) [-1 element, conjugacy size 1]  
        analysis_data.S2 = [
            pi - 8 * sigma_neg1
            for pi, sigma_neg1 in zip(pi_half, pi_half_sigma.get(1, [0]*len(pi_half)))
        ]
        
        # S3: π_{1/2}(x) - 4π_{1/2}(x;i) [i-type elements, conjugacy size 2]
        analysis_data.S3 = [
            pi - 4 * sigma_i
            for pi, sigma_i in zip(pi_half, pi_half_sigma.get(2, [0]*len(pi_half)))
        ]
        
        # S4: π_{1/2}(x) - 4π_{1/2}(x;j) [j-type elements, conjugacy size 2]
        analysis_data.S4 = [
            pi - 4 * sigma_j
            for pi, sigma_j in zip(pi_half, pi_half_sigma.get(3, [0]*len(pi_half)))
        ]
        
        # S5: π_{1/2}(x) - 4π_{1/2}(x;k) [k-type elements, conjugacy size 2]
        analysis_data.S5 = [
            pi - 4 * sigma_k
            for pi, sigma_k in zip(pi_half, pi_half_sigma.get(4, [0]*len(pi_half)))
        ]
        
        return analysis_data
    
    def compute_theoretical_values(self, primes: List[int]) -> Dict[str, List[float]]:
        """
        Compute theoretical bias values based on Aoki-Koyama theory
        
        Args:
            primes: List of primes for x values
            
        Returns:
            Dictionary of theoretical S1-S5 values
        """
        log_log_x = []
        theoretical = {'S1': [], 'S2': [], 'S3': [], 'S4': [], 'S5': []}
        
        for p in primes:
            if p >= 3:  # log(log(x)) defined for x >= 3
                ll_x = np.log(np.log(p))
                log_log_x.append(ll_x)
                
                # Theoretical values: (M(σ) + m(σ)) * log(log(x))
                # Based on bias coefficients from case
                theoretical['S1'].append(self.bias_coefficients.get('g0', 0.5) * ll_x)
                theoretical['S2'].append(self.bias_coefficients.get('g1', 2.5) * ll_x)
                theoretical['S3'].append(self.bias_coefficients.get('g2', -0.5) * ll_x)
                theoretical['S4'].append(self.bias_coefficients.get('g3', -0.5) * ll_x)
                theoretical['S5'].append(self.bias_coefficients.get('g4', -0.5) * ll_x)
            else:
                log_log_x.append(0)
                for key in theoretical:
                    theoretical[key].append(0)
        
        return {'log_log_x': log_log_x, **theoretical}

class StatisticalAnalyzer:
    """Perform statistical analysis of bias convergence"""
    
    @staticmethod
    def linear_regression_analysis(x_data: List[float], y_data: List[float]) -> Dict[str, float]:
        """
        Perform linear regression analysis
        
        Args:
            x_data: Independent variable (log(log(x)))
            y_data: Dependent variable (bias function)
            
        Returns:
            Regression statistics
        """
        if len(x_data) < 2 or len(y_data) < 2:
            return {'slope': 0, 'intercept': 0, 'r_squared': 0, 'p_value': 1.0}
        
        # Filter out invalid points
        valid_indices = [i for i, (x, y) in enumerate(zip(x_data, y_data)) 
                        if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y))]
        
        if len(valid_indices) < 2:
            return {'slope': 0, 'intercept': 0, 'r_squared': 0, 'p_value': 1.0}
        
        x_valid = [x_data[i] for i in valid_indices]
        y_valid = [y_data[i] for i in valid_indices]
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_error': std_err,
                'n_points': len(valid_indices)
            }
        except Exception:
            return {'slope': 0, 'intercept': 0, 'r_squared': 0, 'p_value': 1.0}
    
    @staticmethod
    def convergence_analysis(bias_values: List[float], window_size: int = 100) -> Dict[str, float]:
        """
        Analyze convergence properties of bias function
        
        Args:
            bias_values: Bias function values
            window_size: Window for moving average
            
        Returns:
            Convergence statistics
        """
        if len(bias_values) < window_size:
            return {'convergence_rate': 0, 'final_trend': 0, 'volatility': 0}
        
        # Moving average for trend analysis
        moving_avg = np.convolve(bias_values, np.ones(window_size)/window_size, mode='valid')
        
        # Convergence rate (trend in final portion)
        final_portion = moving_avg[-min(len(moving_avg)//4, 100):]
        if len(final_portion) >= 2:
            convergence_rate = (final_portion[-1] - final_portion[0]) / len(final_portion)
        else:
            convergence_rate = 0
        
        # Final trend
        final_trend = final_portion[-1] if len(final_portion) > 0 else 0
        
        # Volatility (standard deviation of differences)
        if len(bias_values) >= 2:
            differences = np.diff(bias_values)
            volatility = np.std(differences)
        else:
            volatility = 0
        
        return {
            'convergence_rate': convergence_rate,
            'final_trend': final_trend,
            'volatility': volatility
        }

class BiasVisualizer:
    """Create comprehensive visualizations of bias analysis"""
    
    def __init__(self, output_dir: str = "graphs/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure plot style
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16
        })
    
    def create_complete_bias_analysis(
        self, 
        analysis_data: BiasAnalysisResult,
        save_pdf: bool = True,
        show_plots: bool = False
    ) -> str:
        """
        Create complete 5-graph bias analysis
        
        Args:
            analysis_data: Analysis results
            save_pdf: Whether to save as PDF
            show_plots: Whether to display plots
            
        Returns:
            Path to saved visualization
        """
        case_id = analysis_data.case_id
        
        # Create figure with 2x3 subplot layout (5 bias plots + 1 summary)
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Q8 Prime Bias Analysis - Case {case_id}', fontsize=20, fontweight='bold')
        
        # Plot S1-S5
        bias_functions = ['S1', 'S2', 'S3', 'S4', 'S5']
        titles = [
            'S1: π₁/₂(x) - 8π₁/₂(x;1)',
            'S2: π₁/₂(x) - 8π₁/₂(x;-1)', 
            'S3: π₁/₂(x) - 4π₁/₂(x;i)',
            'S4: π₁/₂(x) - 4π₁/₂(x;j)',
            'S5: π₁/₂(x) - 4π₁/₂(x;k)'
        ]
        
        for i, (func_name, title) in enumerate(zip(bias_functions, titles)):
            ax = plt.subplot(2, 3, i + 1)
            self._plot_single_bias_function(ax, analysis_data, func_name, title)
        
        # Summary subplot
        ax_summary = plt.subplot(2, 3, 6)
        self._plot_bias_summary(ax_summary, analysis_data)
        
        plt.tight_layout()
        
        # Save and/or display
        if save_pdf:
            filename = f"case_{case_id:02d}_complete_bias_analysis.pdf"
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved complete bias analysis: {filepath}")
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return str(filepath) if save_pdf else ""
    
    def _plot_single_bias_function(
        self, 
        ax: plt.Axes, 
        analysis_data: BiasAnalysisResult,
        function_name: str,
        title: str
    ):
        """Plot a single bias function with theoretical comparison"""
        
        # Get data
        log_log_x = analysis_data.log_log_x
        actual_values = getattr(analysis_data, function_name)
        theoretical_values = getattr(analysis_data, f'theoretical_{function_name}')
        
        # Sample data for cleaner plots if too many points
        if len(log_log_x) > 1000:
            step = len(log_log_x) // 1000
            log_log_x_sampled = log_log_x[::step]
            actual_sampled = actual_values[::step]
            theoretical_sampled = theoretical_values[::step]
        else:
            log_log_x_sampled = log_log_x
            actual_sampled = actual_values
            theoretical_sampled = theoretical_values
        
        # Plot actual values
        ax.scatter(log_log_x_sampled, actual_sampled, 
                  c='black', s=1, alpha=0.6, label='Actual')
        
        # Plot theoretical line
        ax.plot(log_log_x_sampled, theoretical_sampled, 
               'red', linewidth=2, label='Theoretical')
        
        # Formatting
        ax.set_xlabel('log(log(x))')
        ax.set_ylabel(f'{function_name} values')
        ax.set_title(title, fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add regression info
        reg_stats = analysis_data.regression_stats.get(function_name, {})
        r_squared = reg_stats.get('r_squared', 0)
        ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_bias_summary(self, ax: plt.Axes, analysis_data: BiasAnalysisResult):
        """Create summary plot with all bias functions"""
        
        log_log_x = analysis_data.log_log_x
        
        # Sample for cleaner plot
        if len(log_log_x) > 500:
            step = len(log_log_x) // 500
            x_sampled = log_log_x[::step]
        else:
            x_sampled = log_log_x
            step = 1
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        functions = ['S1', 'S2', 'S3', 'S4', 'S5']
        
        for i, (func, color) in enumerate(zip(functions, colors)):
            values = getattr(analysis_data, func)[::step]
            ax.plot(x_sampled, values, color=color, alpha=0.7, 
                   linewidth=1.5, label=func)
        
        ax.set_xlabel('log(log(x))')
        ax.set_ylabel('Bias values')
        ax.set_title('All Bias Functions Summary')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add case info
        case_info = f"Case {analysis_data.case_id}\\nm_ρ₀ = {get_case(analysis_data.case_id).m_rho0}"
        ax.text(0.7, 0.95, case_info, transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def create_interactive_plotly_visualization(self, analysis_data: BiasAnalysisResult) -> str:
        """Create interactive Plotly visualization"""
        
        if not PLOTLY_AVAILABLE:
            print("Plotly not available, skipping interactive visualization")
            return ""
        
        case_id = analysis_data.case_id
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['S1: Identity', 'S2: -1 Element', 'S3: i-type', 
                          'S4: j-type', 'S5: k-type', 'Summary'],
            specs=[[{"secondary_y": False}]*3]*2
        )
        
        log_log_x = analysis_data.log_log_x
        functions = ['S1', 'S2', 'S3', 'S4', 'S5']
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
        
        # Plot individual bias functions
        for i, (func, color, pos) in enumerate(zip(functions, colors, positions)):
            actual = getattr(analysis_data, func)
            theoretical = getattr(analysis_data, f'theoretical_{func}')
            
            # Sample data for performance
            step = max(1, len(log_log_x) // 1000)
            x_sample = log_log_x[::step]
            actual_sample = actual[::step]
            theoretical_sample = theoretical[::step]
            
            # Actual data points
            fig.add_trace(
                go.Scatter(x=x_sample, y=actual_sample, mode='markers',
                          name=f'{func} Actual', marker=dict(color='black', size=2),
                          showlegend=(i==0)),
                row=pos[0], col=pos[1]
            )
            
            # Theoretical line
            fig.add_trace(
                go.Scatter(x=x_sample, y=theoretical_sample, mode='lines',
                          name=f'{func} Theory', line=dict(color='red', width=2),
                          showlegend=(i==0)),
                row=pos[0], col=pos[1]
            )
        
        # Summary plot
        for func, color in zip(functions, colors):
            values = getattr(analysis_data, func)
            step = max(1, len(log_log_x) // 500)
            fig.add_trace(
                go.Scatter(x=log_log_x[::step], y=values[::step], mode='lines',
                          name=func, line=dict(color=color)),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            title=f'Interactive Q8 Bias Analysis - Case {case_id}',
            height=800,
            showlegend=True
        )
        
        # Save HTML
        filename = f"case_{case_id:02d}_interactive_analysis.html"
        filepath = self.output_dir / filename
        pyo.plot(fig, filename=str(filepath), auto_open=False)
        
        print(f"Saved interactive analysis: {filepath}")
        return str(filepath)

class BiasAnalyzer:
    """Main analyzer class that orchestrates the complete bias analysis"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.dir_manager = DirectoryManager()
        self.visualizer = BiasVisualizer()
        self.statistical_analyzer = StatisticalAnalyzer()
    
    def analyze_case_complete(
        self, 
        case_id: int,
        data_dir: str = "data/results",
        max_primes_for_analysis: Optional[int] = None
    ) -> str:
        """
        Perform complete bias analysis for a single case
        
        Args:
            case_id: Case ID to analyze
            data_dir: Directory containing Frobenius data
            max_primes_for_analysis: Limit number of primes for analysis
            
        Returns:
            Path to generated visualization
        """
        if self.verbose:
            print(f"\\nStarting complete bias analysis for Case {case_id}")
        
        with Timer(f"Case {case_id} bias analysis"):
            # Load Frobenius data
            frobenius_data = self._load_case_data(case_id, data_dir)
            if not frobenius_data:
                raise ValueError(f"No Frobenius data found for Case {case_id}")
            
            # Prepare analysis data
            analysis_data = self._prepare_analysis_data(
                case_id, frobenius_data, max_primes_for_analysis
            )
            
            # Compute bias functions
            bias_calculator = BiasCalculator(case_id)
            analysis_data = bias_calculator.compute_bias_functions(analysis_data)
            
            # Compute theoretical values
            theoretical = bias_calculator.compute_theoretical_values(analysis_data.primes)
            analysis_data.log_log_x = theoretical['log_log_x']
            analysis_data.theoretical_S1 = theoretical['S1']
            analysis_data.theoretical_S2 = theoretical['S2']
            analysis_data.theoretical_S3 = theoretical['S3']
            analysis_data.theoretical_S4 = theoretical['S4']
            analysis_data.theoretical_S5 = theoretical['S5']
            
            # Perform statistical analysis
            self._perform_statistical_analysis(analysis_data)
            
            # Create visualizations
            visualization_path = self.visualizer.create_complete_bias_analysis(
                analysis_data, save_pdf=True, show_plots=False
            )
            
            # Create interactive version if available
            if PLOTLY_AVAILABLE:
                self.visualizer.create_interactive_plotly_visualization(analysis_data)
            
            # Save analysis results
            self._save_analysis_results(analysis_data)
            
            if self.verbose:
                print(f"Analysis completed. Visualization saved: {visualization_path}")
            
            return visualization_path
    
    def _load_case_data(self, case_id: int, data_dir: str) -> Optional[Dict[int, int]]:
        """Load Frobenius data for a case"""
        data_path = Path(data_dir)
        
        # Find the most recent file for this case
        pattern = f"case_{case_id:02d}_frobenius_*.json"
        files = list(data_path.glob(pattern))
        
        if not files:
            if self.verbose:
                print(f"No Frobenius data files found for Case {case_id}")
            return None
        
        # Use the most recent file
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        
        if self.verbose:
            print(f"Loading Frobenius data from: {latest_file}")
        
        try:
            _, results, metadata = load_frobenius_results(str(latest_file))
            if self.verbose:
                print(f"Loaded {len(results)} Frobenius elements")
                print(f"Max prime: {format_number(metadata.get('max_prime', 0))}")
            return results
        except Exception as e:
            if self.verbose:
                print(f"Error loading data: {e}")
            return None
    
    def _prepare_analysis_data(
        self, 
        case_id: int, 
        frobenius_data: Dict[int, int],
        max_primes: Optional[int] = None
    ) -> BiasAnalysisResult:
        """Prepare data for bias analysis"""
        
        # Sort primes and get corresponding Frobenius elements
        sorted_primes = sorted(frobenius_data.keys())
        
        # Limit number of primes if specified
        if max_primes and len(sorted_primes) > max_primes:
            sorted_primes = sorted_primes[:max_primes]
        
        frobenius_elements = [frobenius_data[p] for p in sorted_primes]
        
        if self.verbose:
            print(f"Preparing analysis for {len(sorted_primes)} primes")
        
        # Compute weighted counting functions
        counter = WeightedPrimeCounter()
        
        # π_{1/2}(x) - overall weighted count
        pi_half = counter.compute_pi_half(sorted_primes)
        
        # π_{1/2}(x;σ) for each Frobenius element
        pi_half_sigma = {}
        for sigma in range(8):  # Q8 has 8 elements
            pi_half_sigma[sigma] = counter.compute_pi_half_sigma(
                sorted_primes, frobenius_elements, sigma
            )
        
        # Create analysis result object
        analysis_data = BiasAnalysisResult(
            case_id=case_id,
            primes=sorted_primes,
            frobenius_elements=frobenius_elements,
            pi_half=pi_half,
            pi_half_sigma=pi_half_sigma,
            S1=[], S2=[], S3=[], S4=[], S5=[],  # Will be computed
            log_log_x=[], 
            theoretical_S1=[], theoretical_S2=[], theoretical_S3=[], 
            theoretical_S4=[], theoretical_S5=[],  # Will be computed
            bias_coefficients=get_case(case_id).get_bias_coefficients(),
            regression_stats={}
        )
        
        return analysis_data
    
    def _perform_statistical_analysis(self, analysis_data: BiasAnalysisResult):
        """Perform statistical analysis on bias functions"""
        
        functions = ['S1', 'S2', 'S3', 'S4', 'S5']
        
        for func in functions:
            actual_values = getattr(analysis_data, func)
            
            # Linear regression against log(log(x))
            reg_stats = self.statistical_analyzer.linear_regression_analysis(
                analysis_data.log_log_x, actual_values
            )
            
            # Convergence analysis
            conv_stats = self.statistical_analyzer.convergence_analysis(actual_values)
            
            # Combine statistics
            combined_stats = {**reg_stats, **conv_stats}
            analysis_data.regression_stats[func] = combined_stats
            
            if self.verbose:
                print(f"{func} statistics:")
                print(f"  R² = {reg_stats['r_squared']:.4f}")
                print(f"  Slope = {reg_stats['slope']:.4f}")
                print(f"  Convergence rate = {conv_stats['convergence_rate']:.6f}")
    
    def _save_analysis_results(self, analysis_data: BiasAnalysisResult):
        """Save analysis results to JSON"""
        
        # Prepare data for JSON serialization
        results_data = {
            'case_id': analysis_data.case_id,
            'analysis_timestamp': time.time(),
            'num_primes': len(analysis_data.primes),
            'max_prime': max(analysis_data.primes) if analysis_data.primes else 0,
            'bias_coefficients': analysis_data.bias_coefficients,
            'regression_statistics': analysis_data.regression_stats,
            'case_info': {
                'description': get_case(analysis_data.case_id).case_info['description'],
                'm_rho0': get_case(analysis_data.case_id).m_rho0,
                'ramified_primes': list(get_case(analysis_data.case_id).ramified_primes)
            }
        }
        
        # Save to results directory
        filename = f"case_{analysis_data.case_id:02d}_bias_analysis_results.json"
        filepath = self.dir_manager.get_data_path(filename)
        save_json(results_data, filepath)
        
        if self.verbose:
            print(f"Saved analysis results: {filepath}")
    
    def analyze_multiple_cases(
        self, 
        case_ids: List[int],
        data_dir: str = "data/results"
    ) -> Dict[int, str]:
        """
        Analyze multiple cases and return paths to visualizations
        
        Args:
            case_ids: List of case IDs to analyze
            data_dir: Directory containing data
            
        Returns:
            Dictionary mapping case IDs to visualization paths
        """
        results = {}
        
        for case_id in case_ids:
            if self.verbose:
                print(f"\\n{'='*50}")
                print(f"Analyzing Case {case_id}")
                print(f"{'='*50}")
            
            try:
                viz_path = self.analyze_case_complete(case_id, data_dir)
                results[case_id] = viz_path
            except Exception as e:
                if self.verbose:
                    print(f"Error analyzing Case {case_id}: {e}")
                results[case_id] = ""
        
        return results
    
    def generate_summary_report(self, case_ids: List[int]) -> str:
        """
        Generate comprehensive summary report across multiple cases
        
        Args:
            case_ids: List of case IDs to include
            
        Returns:
            Path to generated report
        """
        if self.verbose:
            print(f"\\nGenerating summary report for {len(case_ids)} cases")
        
        # Create comprehensive comparison figure
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('Q8 Prime Bias Experiment - Summary Report', fontsize=24, fontweight='bold')
        
        # Load analysis results for all cases
        all_results = {}
        for case_id in case_ids:
            try:
                results_file = self.dir_manager.get_data_path(f"case_{case_id:02d}_bias_analysis_results.json")
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        all_results[case_id] = json.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"Could not load results for Case {case_id}: {e}")
        
        # Create comparison plots
        self._create_summary_visualizations(fig, all_results, case_ids)
        
        # Save summary report
        timestamp = int(time.time())
        filename = f"Q8_bias_summary_report_{timestamp}.pdf"
        filepath = self.dir_manager.get_graph_path(filename)
        
        plt.tight_layout()
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        if self.verbose:
            print(f"Summary report saved: {filepath}")
        
        return str(filepath)
    
    def _create_summary_visualizations(self, fig: Figure, all_results: Dict, case_ids: List[int]):
        """Create summary visualization plots"""
        
        # Plot 1: R² comparison across cases
        ax1 = plt.subplot(2, 3, 1)
        self._plot_r_squared_comparison(ax1, all_results, case_ids)
        
        # Plot 2: Bias coefficient comparison
        ax2 = plt.subplot(2, 3, 2)
        self._plot_bias_coefficient_comparison(ax2, all_results, case_ids)
        
        # Plot 3: Convergence rate comparison
        ax3 = plt.subplot(2, 3, 3)
        self._plot_convergence_comparison(ax3, all_results, case_ids)
        
        # Plot 4: m_ρ₀ distribution
        ax4 = plt.subplot(2, 3, 4)
        self._plot_m_rho0_distribution(ax4, case_ids)
        
        # Plot 5: Case statistics table
        ax5 = plt.subplot(2, 3, 5)
        self._plot_statistics_table(ax5, all_results, case_ids)
        
        # Plot 6: Overall assessment
        ax6 = plt.subplot(2, 3, 6)
        self._plot_overall_assessment(ax6, all_results, case_ids)
    
    def _plot_r_squared_comparison(self, ax, all_results, case_ids):
        """Plot R² values comparison"""
        functions = ['S1', 'S2', 'S3', 'S4', 'S5']
        r_squared_data = {func: [] for func in functions}
        
        for case_id in case_ids:
            if case_id in all_results:
                reg_stats = all_results[case_id].get('regression_statistics', {})
                for func in functions:
                    r_sq = reg_stats.get(func, {}).get('r_squared', 0)
                    r_squared_data[func].append(r_sq)
            else:
                for func in functions:
                    r_squared_data[func].append(0)
        
        x = np.arange(len(case_ids))
        width = 0.15
        
        for i, func in enumerate(functions):
            ax.bar(x + i*width, r_squared_data[func], width, label=func)
        
        ax.set_xlabel('Case ID')
        ax.set_ylabel('R² Value')
        ax.set_title('Regression Quality Comparison')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(case_ids)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_bias_coefficient_comparison(self, ax, all_results, case_ids):
        """Plot bias coefficients"""
        m_rho0_0_cases = []
        m_rho0_1_cases = []
        
        for case_id in case_ids:
            case = get_case(case_id)
            if case.m_rho0 == 0:
                m_rho0_0_cases.append(case_id)
            else:
                m_rho0_1_cases.append(case_id)
        
        ax.bar(range(len(m_rho0_0_cases)), [1]*len(m_rho0_0_cases), 
               alpha=0.7, label='m_ρ₀ = 0', color='blue')
        ax.bar(range(len(m_rho0_0_cases), len(m_rho0_0_cases) + len(m_rho0_1_cases)), 
               [1]*len(m_rho0_1_cases), alpha=0.7, label='m_ρ₀ = 1', color='red')
        
        ax.set_xlabel('Cases')
        ax.set_ylabel('Count')
        ax.set_title('Cases by m_ρ₀ Value')
        ax.legend()
        
        # Add case labels
        all_case_labels = [str(c) for c in m_rho0_0_cases] + [str(c) for c in m_rho0_1_cases]
        ax.set_xticks(range(len(all_case_labels)))
        ax.set_xticklabels(all_case_labels, rotation=45)
    
    def _plot_convergence_comparison(self, ax, all_results, case_ids):
        """Plot convergence rates"""
        # Implementation similar to R² comparison
        ax.text(0.5, 0.5, 'Convergence Analysis\\n(Implementation Detail)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Convergence Rate Analysis')
    
    def _plot_m_rho0_distribution(self, ax, case_ids):
        """Plot m_ρ₀ distribution"""
        m_rho0_counts = {0: 0, 1: 0}
        for case_id in case_ids:
            case = get_case(case_id)
            m_rho0_counts[case.m_rho0] += 1
        
        ax.pie(m_rho0_counts.values(), labels=[f'm_ρ₀ = {k}' for k in m_rho0_counts.keys()],
               autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribution of m_ρ₀ Values')
    
    def _plot_statistics_table(self, ax, all_results, case_ids):
        """Create statistics summary table"""
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for case_id in case_ids[:5]:  # Show first 5 cases
            if case_id in all_results:
                data = all_results[case_id]
                row = [
                    f"Case {case_id}",
                    f"{data.get('num_primes', 0):,}",
                    f"{data.get('case_info', {}).get('m_rho0', 'N/A')}",
                    f"{np.mean([stats.get('r_squared', 0) for stats in data.get('regression_statistics', {}).values()]):.3f}"
                ]
            else:
                row = [f"Case {case_id}", "No data", "N/A", "N/A"]
            table_data.append(row)
        
        table = ax.table(cellText=table_data,
                        colLabels=['Case', 'Primes', 'm_ρ₀', 'Avg R²'],
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax.set_title('Case Statistics Summary')
    
    def _plot_overall_assessment(self, ax, all_results, case_ids):
        """Create overall assessment"""
        successful_cases = len([c for c in case_ids if c in all_results])
        total_cases = len(case_ids)
        
        ax.text(0.5, 0.7, f'Experiment Summary', ha='center', va='center', 
               transform=ax.transAxes, fontsize=16, fontweight='bold')
        ax.text(0.5, 0.5, f'Successful Cases: {successful_cases}/{total_cases}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        
        if successful_cases > 0:
            # Calculate average performance metrics
            avg_metrics = {}
            for case_id in case_ids:
                if case_id in all_results:
                    stats = all_results[case_id].get('regression_statistics', {})
                    for func, func_stats in stats.items():
                        if func not in avg_metrics:
                            avg_metrics[func] = []
                        avg_metrics[func].append(func_stats.get('r_squared', 0))
            
            overall_r2 = np.mean([np.mean(values) for values in avg_metrics.values()])
            ax.text(0.5, 0.3, f'Average R²: {overall_r2:.3f}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Overall Assessment')

if __name__ == "__main__":
    # Demo usage
    print("Bias Analyzer Demo")
    print("==================")
    
    analyzer = BiasAnalyzer(verbose=True)
    
    # Test with a single case if data is available
    try:
        viz_path = analyzer.analyze_case_complete(1, "data/results")
        print(f"\\n✅ Demo completed successfully!")
        print(f"Visualization: {viz_path}")
    except Exception as e:
        print(f"\\n⚠️  Demo requires Frobenius data to be computed first")
        print(f"Error: {e}")
        print(f"\\nTo generate data, run:")
        print(f"from src.fast_frobenius_calculator import ParallelFrobeniusComputation")
        print(f"computer = ParallelFrobeniusComputation()")
        print(f"computer.compute_case_parallel(1, 10000)")
    
    print(f"\\n📊 Available analysis methods:")
    methods = [
        'analyze_case_complete(case_id, data_dir)',
        'analyze_multiple_cases(case_ids, data_dir)',
        'generate_summary_report(case_ids)'
    ]
    for method in methods:
        print(f"  - analyzer.{method}")
