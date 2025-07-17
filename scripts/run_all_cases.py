#!/usr/bin/env python3
"""
Batch processing script for Q8 Prime Bias Experiments

This script runs complete experiments for all 13 cases with configurable
parameters. Suitable for overnight runs and large-scale computations.

Usage:
    python scripts/run_all_cases.py --scale medium --parallel
    python scripts/run_all_cases.py --max-prime 1000000 --cases 1,2,3
    python scripts/run_all_cases.py --config config/experiment_config.yaml
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'src'))

from experiment_runner import QuaternionBiasExperiment, run_experiment_from_config
from omar_polynomials import get_all_cases
from utils import format_number, format_duration, get_system_info

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run Q8 Prime Bias Experiments for all cases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with first 3 cases
  python %(prog)s --scale small --cases 1,2,3

  # Medium scale research run
  python %(prog)s --scale medium --parallel

  # Large scale with specific parameters
  python %(prog)s --max-prime 1000000 --processes 16

  # Production run (overnight)
  python %(prog)s --scale large --parallel --save-logs
        """
    )
    
    # Scale presets
    parser.add_argument(
        '--scale', 
        choices=['small', 'medium', 'large', 'huge'],
        help='Predefined scale: small(10K), medium(100K), large(1M), huge(1B)'
    )
    
    # Custom parameters
    parser.add_argument(
        '--max-prime', 
        type=int,
        help='Maximum prime to compute (overrides scale)'
    )
    
    parser.add_argument(
        '--cases',
        type=str,
        default='1,2,3,4,5,6,7,8,9,10,11,12,13',
        help='Comma-separated case IDs (default: all 13 cases)'
    )
    
    parser.add_argument(
        '--processes',
        type=int,
        help='Number of parallel processes (default: auto-detect)'
    )
    
    # Execution options
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing'
    )
    
    parser.add_argument(
        '--no-analysis',
        action='store_true',
        help='Skip bias analysis (computation only)'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip summary report generation'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='config/experiment_config.yaml',
        help='Configuration file path'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--save-logs',
        action='store_true',
        help='Save detailed computation logs'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing'
    )
    
    # Verbosity
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Detailed output'
    )

    return parser.parse_args()

def get_scale_parameters(scale):
    """Get max_prime for predefined scales"""
    scales = {
        'small': 10000,
        'medium': 100000,
        'large': 1000000,
        'huge': 1000000000
    }
    return scales.get(scale)

def parse_case_list(case_string):
    """Parse comma-separated case IDs"""
    try:
        case_ids = []
        for part in case_string.split(','):
            part = part.strip()
            if '-' in part:
                # Handle ranges like "1-5"
                start, end = map(int, part.split('-'))
                case_ids.extend(range(start, end + 1))
            else:
                case_ids.append(int(part))
        
        # Validate case IDs
        valid_cases = [case.case_id for case in get_all_cases()]
        invalid_cases = [cid for cid in case_ids if cid not in valid_cases]
        
        if invalid_cases:
            raise ValueError(f"Invalid case IDs: {invalid_cases}")
        
        return sorted(list(set(case_ids)))
    
    except Exception as e:
        raise ValueError(f"Error parsing case list '{case_string}': {e}")

def print_banner():
    """Print program banner"""
    print("=" * 70)
    print("Q8 PRIME BIAS EXPERIMENT - BATCH PROCESSOR")
    print("=" * 70)
    print("Implementation of Aoki-Koyama 2022 Example 2.1")
    print("Batch processing for all 13 Omar S. polynomial cases")
    print("=" * 70)

def print_system_info(verbose=False):
    """Print system information"""
    if verbose:
        print("\n🖥️  System Information:")
        print("-" * 30)
        sys_info = get_system_info()
        for key, value in sys_info.items():
            print(f"  {key}: {value}")
        print()

def print_experiment_plan(args, case_ids, max_prime):
    """Print what will be executed"""
    print(f"\n📋 Experiment Plan:")
    print(f"-" * 30)
    print(f"Cases to run: {case_ids} ({len(case_ids)} total)")
    print(f"Maximum prime: {format_number(max_prime)}")
    print(f"Parallel processing: {'Yes' if args.parallel else 'No'}")
    if args.processes:
        print(f"Process count: {args.processes}")
    print(f"Bias analysis: {'No' if args.no_analysis else 'Yes'}")
    print(f"Summary report: {'No' if args.no_report else 'Yes'}")
    print(f"Configuration: {args.config}")
    
    # Estimate resources
    from utils import estimate_memory_usage, estimate_computation_time, get_optimal_process_count
    
    processes = args.processes or get_optimal_process_count()
    memory_est = estimate_memory_usage(max_prime, len(case_ids))
    time_est = estimate_computation_time(max_prime, len(case_ids), processes)
    
    print(f"\n⚡ Resource Estimates:")
    print(f"  Memory usage: ~{memory_est['total_gb']:.1f} GB")
    if args.parallel:
        print(f"  Estimated time: {format_duration(time_est['parallel_time_seconds'])}")
        print(f"  Processes: {processes}")
    else:
        print(f"  Estimated time: {format_duration(time_est['base_time_seconds'])}")
    print()

def confirm_execution(args):
    """Ask for confirmation before running"""
    if args.dry_run:
        print("🏃 DRY RUN MODE - No computation will be performed")
        return True
    
    if args.quiet:
        return True
    
    try:
        response = input("Proceed with experiment? (y/n): ").lower().strip()
        return response in ['y', 'yes']
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
        return False

def setup_logging(args):
    """Setup logging configuration"""
    if args.save_logs:
        import logging
        from datetime import datetime
        
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"batch_experiment_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        print(f"📝 Logging to: {log_file}")
        return str(log_file)
    
    return None

def run_batch_experiment(args):
    """Main batch experiment execution"""
    
    # Parse arguments
    case_ids = parse_case_list(args.cases)
    
    # Determine max_prime
    if args.max_prime:
        max_prime = args.max_prime
    elif args.scale:
        max_prime = get_scale_parameters(args.scale)
        if max_prime is None:
            raise ValueError(f"Unknown scale: {args.scale}")
    else:
        raise ValueError("Must specify either --scale or --max-prime")
    
    # Setup experiment
    verbose = args.verbose and not args.quiet
    experiment = QuaternionBiasExperiment(
        config_path=args.config,
        verbose=verbose
    )
    
    if not args.dry_run:
        # Setup logging
        log_file = setup_logging(args)
        
        # Run the complete experiment
        print(f"🚀 Starting batch experiment...")
        start_time = time.time()
        
        try:
            results = experiment.run_complete_experiment(
                case_ids=case_ids,
                max_prime=max_prime,
                num_processes=args.processes,
                analyze_bias=not args.no_analysis,
                generate_report=not args.no_report
            )
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Print results summary
            print(f"\n🎉 Batch Experiment Completed!")
            print(f"=" * 50)
            print(f"Total duration: {format_duration(total_duration)}")
            
            if 'computation' in results:
                total_primes = sum(len(case_results) for case_results in results['computation'].values())
                print(f"Frobenius elements computed: {format_number(total_primes)}")
                print(f"Average rate: {total_primes / total_duration:.1f} primes/second")
            
            if 'analysis' in results:
                successful_analyses = sum(1 for path in results['analysis'].values() if path)
                print(f"Bias analyses generated: {successful_analyses}/{len(case_ids)}")
            
            if 'report_path' in results:
                print(f"Summary report: {results['report_path']}")
            
            if log_file:
                print(f"Detailed logs: {log_file}")
            
            print(f"\n✅ All results saved to data/ and graphs/ directories")
            
            return 0
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Experiment interrupted by user")
            print(f"Partial results may be available in data/results/")
            return 1
            
        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    else:
        print(f"✅ Dry run completed - no computation performed")
        return 0

def main():
    """Main entry point"""
    
    try:
        args = parse_arguments()
        
        # Print banner
        if not args.quiet:
            print_banner()
            print_system_info(args.verbose)
        
        # Parse and validate arguments
        case_ids = parse_case_list(args.cases)
        
        if args.max_prime:
            max_prime = args.max_prime
        elif args.scale:
            max_prime = get_scale_parameters(args.scale)
            if max_prime is None:
                print(f"❌ Unknown scale: {args.scale}")
                return 1
        else:
            print(f"❌ Must specify either --scale or --max-prime")
            return 1
        
        # Print experiment plan
        if not args.quiet:
            print_experiment_plan(args, case_ids, max_prime)
        
        # Confirm execution
        if not confirm_execution(args):
            print("❌ Experiment cancelled")
            return 1
        
        # Run the experiment
        return run_batch_experiment(args)
        
    except ValueError as e:
        print(f"❌ Invalid arguments: {e}")
        return 1
    except KeyboardInterrupt:
        print(f"\n❌ Cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if '--verbose' in sys.argv:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
