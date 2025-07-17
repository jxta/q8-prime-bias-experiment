#!/usr/bin/env python3
"""
Experiment Runner for Q8 Prime Bias Analysis

Main orchestrator for running complete experiments across all 13 cases
with support for parallel processing, progress tracking, and result management.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from .utils import (
    Timer, DirectoryManager, ConfigManager,
    format_number, format_duration,
    estimate_memory_usage, estimate_computation_time,
    get_optimal_process_count, save_json
)
from .omar_polynomials import get_case, get_all_cases, print_all_cases_summary
from .fast_frobenius_calculator import (
    FastFrobeniusCalculator, ParallelFrobeniusComputation,
    save_frobenius_results
)
from .bias_analyzer import BiasAnalyzer

class QuaternionBiasExperiment:
    """Main experiment runner for Q8 prime bias analysis"""
    
    def __init__(self, config_path: str = "config/experiment_config.yaml", verbose: bool = True):
        """
        Initialize experiment runner
        
        Args:
            config_path: Path to configuration file
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        
        # Initialize managers
        self.dir_manager = DirectoryManager()
        self.config_manager = ConfigManager(config_path)
        
        # Initialize components
        self.parallel_computer = ParallelFrobeniusComputation(verbose=verbose)
        self.bias_analyzer = BiasAnalyzer(verbose=verbose)
        
        # Experiment state
        self.results = {}
        self.computation_stats = {}
        
        if self.verbose:
            print("QuaternionBiasExperiment initialized")
            print(f"Available CPU cores: {mp.cpu_count()}")
            print(f"Configuration loaded from: {self.config_manager.config_path}")
    
    def print_experiment_info(self):
        """Print comprehensive experiment information"""
        print("\n" + "=" * 60)
        print("Q8 GALOIS EXTENSION PRIME BIAS EXPERIMENT")
        print("=" * 60)
        
        print("\nBased on:")
        print("Aoki, M. and Koyama, S. (2022). Chebyshev's Bias against")
        print("Splitting and Principal Primes in Global Fields. arXiv:2203.12266")
        
        print("\nAvailable Cases:")
        print_all_cases_summary()
        
        print(f"\nSystem Configuration:")
        print(f"  CPU cores: {mp.cpu_count()}")
        print(f"  Optimal processes: {get_optimal_process_count()}")
        
        scales = {
            'small': self.config_manager.get('computation.small_scale', 10000),
            'medium': self.config_manager.get('computation.medium_scale', 100000),
            'large': self.config_manager.get('computation.large_scale', 1000000),
            'huge': self.config_manager.get('computation.huge_scale', 1000000000)
        }
        
        print(f"\nPredefined Scales:")
        for scale_name, max_prime in scales.items():
            mem_est = estimate_memory_usage(max_prime, 13)
            time_est = estimate_computation_time(max_prime, 13, get_optimal_process_count())
            print(f"  {scale_name.capitalize()}: {format_number(max_prime)} primes")
            print(f"    Memory: ~{mem_est['total_gb']:.1f} GB")
            print(f"    Time (parallel): ~{format_duration(time_est['parallel_time_seconds'])}")
        
        print("\n" + "=" * 60)
    
    def validate_experiment_setup(
        self,
        case_ids: List[int],
        max_prime: int,
        num_processes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate experiment setup and provide estimates
        
        Args:
            case_ids: List of case IDs to validate
            max_prime: Maximum prime for computation
            num_processes: Number of processes
            
        Returns:
            Validation results and estimates
        """
        if num_processes is None:
            num_processes = get_optimal_process_count()
        
        # Validate case IDs
        available_cases = [case.case_id for case in get_all_cases()]
        invalid_cases = [cid for cid in case_ids if cid not in available_cases]
        
        if invalid_cases:
            raise ValueError(f"Invalid case IDs: {invalid_cases}. Available: {available_cases}")
        
        # Compute estimates
        memory_est = estimate_memory_usage(max_prime, len(case_ids))
        time_est = estimate_computation_time(max_prime, len(case_ids), num_processes)
        
        validation_result = {
            'valid': True,
            'case_ids': case_ids,
            'max_prime': max_prime,
            'num_processes': num_processes,
            'estimated_primes': time_est['estimated_primes'],
            'memory_estimate_gb': memory_est['total_gb'],
            'time_estimate_hours': time_est['parallel_time_hours'],
            'total_operations': time_est['total_operations']
        }
        
        if self.verbose:
            print(f"\nExperiment Validation:")
            print(f"  Cases: {case_ids}")
            print(f"  Max prime: {format_number(max_prime)}")
            print(f"  Estimated primes: {format_number(validation_result['estimated_primes'])}")
            print(f"  Processes: {num_processes}")
            print(f"  Memory estimate: {memory_est['total_gb']:.1f} GB")
            print(f"  Time estimate: {format_duration(time_est['parallel_time_seconds'])}")
        
        return validation_result
    
    def run_case(
        self,
        case_id: int,
        max_prime: int,
        num_processes: Optional[int] = None,
        save_results: bool = True
    ) -> Dict[int, int]:
        """
        Run computation for a single case
        
        Args:
            case_id: Case ID to compute
            max_prime: Maximum prime
            num_processes: Number of processes
            save_results: Whether to save results to disk
            
        Returns:
            Frobenius computation results
        """
        if self.verbose:
            print(f"\nRunning Case {case_id} computation...")
        
        # Validate setup
        validation = self.validate_experiment_setup([case_id], max_prime, num_processes)
        
        start_time = time.time()
        
        # Compute Frobenius elements
        with Timer(f"Case {case_id} computation"):
            results = self.parallel_computer.compute_case_parallel(
                case_id, max_prime, num_processes
            )
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        # Store results and statistics
        self.results[case_id] = results
        self.computation_stats[case_id] = {
            'max_prime': max_prime,
            'num_primes': len(results),
            'computation_time': computation_time,
            'num_processes': num_processes or get_optimal_process_count(),
            'timestamp': time.time()
        }
        
        if save_results:
            save_path = save_frobenius_results(case_id, results, max_prime)
            self.computation_stats[case_id]['save_path'] = save_path
        
        if self.verbose:
            print(f"Case {case_id} completed:")
            print(f"  Computed: {len(results)} Frobenius elements")
            print(f"  Time: {format_duration(computation_time)}")
            print(f"  Rate: {len(results)/computation_time:.1f} primes/second")
        
        return results
    
    def run_multiple_cases(
        self,
        case_ids: List[int],
        max_prime: int,
        num_processes: Optional[int] = None,
        save_results: bool = True
    ) -> Dict[int, Dict[int, int]]:
        """
        Run computation for multiple cases sequentially
        
        Args:
            case_ids: List of case IDs
            max_prime: Maximum prime
            num_processes: Number of processes per case
            save_results: Whether to save results
            
        Returns:
            Dictionary mapping case IDs to results
        """
        if self.verbose:
            print(f"\nRunning computation for {len(case_ids)} cases sequentially")
        
        # Validate all cases
        validation = self.validate_experiment_setup(case_ids, max_prime, num_processes)
        
        all_results = {}
        total_start_time = time.time()
        
        for i, case_id in enumerate(case_ids, 1):
            if self.verbose:
                print(f"\n[{i}/{len(case_ids)}] Starting Case {case_id}...")
            
            try:
                case_results = self.run_case(case_id, max_prime, num_processes, save_results)
                all_results[case_id] = case_results
            except Exception as e:
                if self.verbose:
                    print(f"Error in Case {case_id}: {e}")
                all_results[case_id] = {}
        
        total_time = time.time() - total_start_time
        
        if self.verbose:
            successful_cases = sum(1 for results in all_results.values() if results)
            total_primes = sum(len(results) for results in all_results.values())
            
            print(f"\nMultiple cases computation completed:")
            print(f"  Successful cases: {successful_cases}/{len(case_ids)}")
            print(f"  Total primes computed: {format_number(total_primes)}")
            print(f"  Total time: {format_duration(total_time)}")
            print(f"  Average rate: {total_primes/total_time:.1f} primes/second")
        
        return all_results
    
    def run_parallel_experiment(
        self,
        case_ids: List[int],
        max_prime: int,
        max_concurrent_cases: int = 3,
        num_processes_per_case: Optional[int] = None
    ) -> Dict[int, Dict[int, int]]:
        """
        Run multiple cases in parallel (experimental)
        
        Args:
            case_ids: List of case IDs
            max_prime: Maximum prime
            max_concurrent_cases: Maximum cases to run concurrently
            num_processes_per_case: Processes per case
            
        Returns:
            Dictionary mapping case IDs to results
        """
        if num_processes_per_case is None:
            # Distribute available cores among concurrent cases
            available_cores = get_optimal_process_count()
            num_processes_per_case = max(1, available_cores // max_concurrent_cases)
        
        if self.verbose:
            print(f"\nRunning {len(case_ids)} cases with parallel processing:")
            print(f"  Max concurrent cases: {max_concurrent_cases}")
            print(f"  Processes per case: {num_processes_per_case}")
        
        # Validate setup
        validation = self.validate_experiment_setup(case_ids, max_prime, num_processes_per_case)
        
        all_results = {}
        
        def compute_case_wrapper(case_id):
            """Wrapper for parallel case computation"""
            try:
                results = self.parallel_computer.compute_case_parallel(
                    case_id, max_prime, num_processes_per_case
                )
                # Save results
                save_frobenius_results(case_id, results, max_prime)
                return case_id, results, None
            except Exception as e:
                return case_id, {}, str(e)
        
        with Timer("Parallel experiment execution"):
            with ProcessPoolExecutor(max_workers=max_concurrent_cases) as executor:
                # Submit all cases
                future_to_case = {executor.submit(compute_case_wrapper, case_id): case_id 
                                 for case_id in case_ids}
                
                # Collect results as they complete
                for future in as_completed(future_to_case):
                    case_id, results, error = future.result()
                    
                    if error:
                        if self.verbose:
                            print(f"Case {case_id} failed: {error}")
                        all_results[case_id] = {}
                    else:
                        all_results[case_id] = results
                        if self.verbose:
                            print(f"Case {case_id} completed: {len(results)} primes")
        
        if self.verbose:
            successful_cases = sum(1 for results in all_results.values() if results)
            total_primes = sum(len(results) for results in all_results.values())
            print(f"\nParallel experiment completed:")
            print(f"  Successful cases: {successful_cases}/{len(case_ids)}")
            print(f"  Total primes computed: {format_number(total_primes)}")
        
        return all_results
    
    def analyze_case(
        self,
        case_id: int,
        data_dir: str = "data/results"
    ) -> str:
        """
        Run bias analysis for a case
        
        Args:
            case_id: Case ID to analyze
            data_dir: Directory containing Frobenius data
            
        Returns:
            Path to generated graph
        """
        if self.verbose:
            print(f"\nAnalyzing bias for Case {case_id}...")
        
        try:
            graph_path = self.bias_analyzer.analyze_case_complete(case_id, data_dir)
            if self.verbose:
                print(f"Generated bias graphs: {graph_path}")
            return graph_path
        except Exception as e:
            if self.verbose:
                print(f"Error analyzing Case {case_id}: {e}")
            raise
    
    def analyze_multiple_cases(
        self,
        case_ids: List[int],
        data_dir: str = "data/results"
    ) -> Dict[int, str]:
        """
        Run bias analysis for multiple cases
        
        Args:
            case_ids: List of case IDs
            data_dir: Directory containing data
            
        Returns:
            Dictionary mapping case IDs to graph paths
        """
        if self.verbose:
            print(f"\nAnalyzing bias for {len(case_ids)} cases...")
        
        return self.bias_analyzer.analyze_multiple_cases(case_ids, data_dir)
    
    def generate_summary_report(
        self,
        case_ids: Optional[List[int]] = None
    ) -> str:
        """
        Generate comprehensive summary report
        
        Args:
            case_ids: Case IDs to include (None for all available)
            
        Returns:
            Path to generated report
        """
        if case_ids is None:
            # Find all available cases from results directory
            results_dir = Path("data/results")
            case_ids = []
            for file in results_dir.glob("case_*_frobenius_*.json"):
                try:
                    case_id = int(file.name.split('_')[1])
                    if case_id not in case_ids:
                        case_ids.append(case_id)
                except:
                    continue
            case_ids.sort()
        
        if self.verbose:
            print(f"\nGenerating summary report for cases: {case_ids}")
        
        return self.bias_analyzer.generate_summary_report(case_ids)
    
    def run_complete_experiment(
        self,
        case_ids: List[int],
        max_prime: int,
        num_processes: Optional[int] = None,
        analyze_bias: bool = True,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete experiment: computation + analysis + reporting
        
        Args:
            case_ids: List of case IDs
            max_prime: Maximum prime
            num_processes: Number of processes
            analyze_bias: Whether to run bias analysis
            generate_report: Whether to generate summary report
            
        Returns:
            Complete experiment results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"RUNNING COMPLETE Q8 PRIME BIAS EXPERIMENT")
            print(f"{'='*60}")
            print(f"Cases: {case_ids}")
            print(f"Max prime: {format_number(max_prime)}")
            print(f"Processes: {num_processes or get_optimal_process_count()}")
        
        experiment_start = time.time()
        results = {}
        
        # Phase 1: Computation
        if self.verbose:
            print(f"\n🔄 Phase 1: Frobenius Element Computation")
        
        try:
            computation_results = self.run_multiple_cases(
                case_ids, max_prime, num_processes, save_results=True
            )
            results['computation'] = computation_results
            results['computation_stats'] = self.computation_stats
        except Exception as e:
            if self.verbose:
                print(f"❌ Computation phase failed: {e}")
            results['computation'] = {}
            results['computation_error'] = str(e)
        
        # Phase 2: Bias Analysis
        if analyze_bias:
            if self.verbose:
                print(f"\n📊 Phase 2: Bias Analysis")
            
            try:
                analysis_results = self.analyze_multiple_cases(case_ids)
                results['analysis'] = analysis_results
                
                successful_analyses = sum(1 for path in analysis_results.values() if path)
                if self.verbose:
                    print(f"Generated graphs for {successful_analyses}/{len(case_ids)} cases")
            except Exception as e:
                if self.verbose:
                    print(f"❌ Analysis phase failed: {e}")
                results['analysis'] = {}
                results['analysis_error'] = str(e)
        
        # Phase 3: Report Generation
        if generate_report:
            if self.verbose:
                print(f"\n📝 Phase 3: Report Generation")
            
            try:
                report_path = self.generate_summary_report(case_ids)
                results['report_path'] = report_path
                if self.verbose:
                    print(f"Generated summary report: {report_path}")
            except Exception as e:
                if self.verbose:
                    print(f"❌ Report generation failed: {e}")
                results['report_error'] = str(e)
        
        # Summary
        experiment_time = time.time() - experiment_start
        results['experiment_duration'] = experiment_time
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"EXPERIMENT COMPLETED")
            print(f"{'='*60}")
            print(f"Total time: {format_duration(experiment_time)}")
            
            if 'computation' in results:
                total_primes = sum(len(case_results) for case_results in results['computation'].values())
                print(f"Frobenius elements computed: {format_number(total_primes)}")
            
            if 'analysis' in results:
                successful_graphs = sum(1 for path in results['analysis'].values() if path)
                print(f"Bias graphs generated: {successful_graphs}")
            
            if 'report_path' in results:
                print(f"Summary report: {results['report_path']}")
        
        return results
    
    def quick_test(
        self,
        test_cases: List[int] = [1, 2, 3],
        max_prime: int = 10000
    ) -> Dict[str, Any]:
        """
        Run quick test with small parameters
        
        Args:
            test_cases: Cases to test
            max_prime: Small max prime for testing
            
        Returns:
            Test results
        """
        if self.verbose:
            print(f"\n🧪 Running quick test with {len(test_cases)} cases")
        
        return self.run_complete_experiment(
            test_cases, max_prime, 
            num_processes=4,
            analyze_bias=True,
            generate_report=True
        )

def run_experiment_from_config(config_file: str = "config/experiment_config.yaml") -> Dict[str, Any]:
    """
    Run experiment based on configuration file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Experiment results
    """
    experiment = QuaternionBiasExperiment(config_file)
    config = experiment.config_manager
    
    # Get parameters from config
    enabled_cases = config.get('cases.enabled', list(range(1, 14)))
    scale = config.get('computation.large_scale', 1000000)
    parallel_enabled = config.get('computation.parallel.enabled', True)
    max_processes = config.get('computation.parallel.max_processes', None)
    
    experiment.print_experiment_info()
    
    if parallel_enabled:
        return experiment.run_complete_experiment(
            enabled_cases, scale, max_processes
        )
    else:
        return experiment.run_complete_experiment(
            enabled_cases, scale, 1
        )

if __name__ == "__main__":
    # Demo run
    print("Q8 Prime Bias Experiment Runner Demo")
    print("====================================")
    
    # Initialize experiment
    experiment = QuaternionBiasExperiment()
    
    # Print info
    experiment.print_experiment_info()
    
    # Run quick test
    print("\n🧪 Running quick test...")
    test_results = experiment.quick_test([1, 2], max_prime=5000)
    
    print(f"\n✅ Quick test completed!")
    print(f"Results: {list(test_results.keys())}")
    
    # Show available methods
    print("\n📚 Available methods:")
    methods = [
        'run_case(case_id, max_prime)',
        'run_multiple_cases(case_ids, max_prime)', 
        'run_parallel_experiment(case_ids, max_prime)',
        'analyze_case(case_id)',
        'run_complete_experiment(case_ids, max_prime)',
        'quick_test(test_cases, max_prime)'
    ]
    
    for method in methods:
        print(f"  - experiment.{method}")
    
    print("\n🎯 Ready for large-scale experiments!")
