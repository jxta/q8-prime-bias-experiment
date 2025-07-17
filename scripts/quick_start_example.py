#!/usr/bin/env python3
"""
Quick Start Example for Q8 Prime Bias Experiment

This script demonstrates the basic workflow of the Q8 prime bias experiment system.
Perfect for getting started and understanding how the system works.

Run this after installing dependencies:
    pip install -r requirements.txt
    python scripts/quick_start_example.py
"""

import sys
import time
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'src'))

print("🌟 Q8 Prime Bias Experiment - Quick Start Example")
print("=" * 55)
print("Based on Aoki-Koyama 2022, arXiv:2203.12266")
print("Implementation of Example 2.1 with all 13 Omar S. cases")
print("=" * 55)

# Test imports
print("\n📦 Testing imports...")
try:
    from omar_polynomials import get_case, print_all_cases_summary
    from fast_frobenius_calculator import ParallelFrobeniusComputation
    from bias_analyzer import BiasAnalyzer
    from experiment_runner import QuaternionBiasExperiment
    from utils import format_number, Timer
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)

# Show available cases
print("\n📚 Available Q8 Galois Extension Cases:")
print("-" * 40)
print_all_cases_summary()

# Examine Case 1 (Aoki-Koyama Example 2.1)
print("\n🔍 Case 1 Details (Aoki-Koyama Example 2.1):")
print("-" * 45)
case1 = get_case(1)
print(f"Polynomial: {case1.polynomial}")
print(f"m_ρ₀ value: {case1.m_rho0}")
print(f"Ramified primes: {list(case1.ramified_primes)}")
print(f"Expected bias coefficients:")
bias_coeffs = case1.get_bias_coefficients()
for coeff, value in bias_coeffs.items():
    if coeff.startswith('g') and int(coeff[1]) <= 4:
        func_name = ['S1(identity)', 'S2(-1)', 'S3(i-type)', 'S4(j-type)', 'S5(k-type)'][int(coeff[1])]
        print(f"  {func_name}: {value:+.1f}")

# Initialize experiment system
print("\n⚙️  Initializing experiment system...")
experiment = QuaternionBiasExperiment(verbose=False)
print("✅ System initialized!")

# Demo 1: Small computation
print("\n🧪 Demo 1: Small Frobenius Computation")
print("-" * 40)
print("Computing Frobenius elements for primes up to 1000...")

with Timer("Small computation"):
    small_results = experiment.run_case(
        case_id=1,
        max_prime=1000,
        num_processes=2,
        save_results=True
    )

print(f"✅ Computed {len(small_results)} Frobenius elements")

# Show sample results
print("\n📋 Sample results (first 10 primes):")
sample_primes = sorted(small_results.keys())[:10]
for p in sample_primes:
    frobenius = small_results[p]
    print(f"  p = {p:3d} → Frobenius element = {frobenius}")

# Demo 2: Bias analysis
print("\n📊 Demo 2: Bias Analysis")
print("-" * 30)
print("Generating 5-graph bias analysis...")

analyzer = BiasAnalyzer(verbose=False)

with Timer("Bias analysis"):
    try:
        viz_path = analyzer.analyze_case_complete(
            case_id=1,
            data_dir="data/results"
        )
        print(f"✅ Bias analysis completed!")
        print(f"📁 Visualization saved to: {viz_path}")
        print(f"📊 Generated 5 bias function graphs:")
        print(f"   • S1: π₁/₂(x) - 8π₁/₂(x;1)   [Identity bias]")
        print(f"   • S2: π₁/₂(x) - 8π₁/₂(x;-1)  [-1 element bias]")
        print(f"   • S3: π₁/₂(x) - 4π₁/₂(x;i)   [i-type bias]")
        print(f"   • S4: π₁/₂(x) - 4π₁/₂(x;j)   [j-type bias]")
        print(f"   • S5: π₁/₂(x) - 4π₁/₂(x;k)   [k-type bias]")
    except Exception as e:
        print(f"⚠️  Bias analysis had an issue: {e}")
        print("This is normal for very small datasets. Try with larger prime ranges.")

# Demo 3: Complete workflow
print("\n🎯 Demo 3: Complete Workflow")
print("-" * 35)
print("Running complete experiment: computation + analysis + reporting")
print("Cases 1-2, up to 3000 primes...")

with Timer("Complete workflow"):
    complete_results = experiment.run_complete_experiment(
        case_ids=[1, 2],
        max_prime=3000,
        num_processes=4,
        analyze_bias=True,
        generate_report=True
    )

print(f"✅ Complete workflow finished!")

# Show results summary
if 'computation' in complete_results:
    total_primes = sum(len(case_data) for case_data in complete_results['computation'].values())
    print(f"📊 Computed {format_number(total_primes)} total Frobenius elements")

if 'analysis' in complete_results:
    successful_analyses = sum(1 for path in complete_results['analysis'].values() if path)
    print(f"📈 Generated {successful_analyses} bias visualizations")

if 'report_path' in complete_results:
    print(f"📁 Summary report: {complete_results['report_path']}")

duration = complete_results.get('experiment_duration', 0)
print(f"⏱️  Total time: {duration:.1f} seconds")

# Performance estimates for larger scales
print("\n⚡ Performance Estimates for Larger Scales:")
print("-" * 45)

from utils import estimate_memory_usage, estimate_computation_time, get_optimal_process_count

scales = [
    ("Medium (100K primes)", 100000),
    ("Large (1M primes)", 1000000),
    ("Huge (10M primes)", 10000000),
    ("Production (100M primes)", 100000000)
]

optimal_processes = get_optimal_process_count()

for scale_name, max_prime in scales:
    memory_est = estimate_memory_usage(max_prime, 1)
    time_est = estimate_computation_time(max_prime, 1, optimal_processes)
    
    print(f"\n{scale_name}:")
    print(f"  Memory: ~{memory_est['total_gb']:.1f} GB")
    print(f"  Time (sequential): {time_est['base_time_hours']:.1f} hours")
    print(f"  Time (parallel, {optimal_processes} cores): {time_est['parallel_time_hours']:.1f} hours")

# Next steps
print(f"\n🚀 Next Steps:")
print("-" * 15)
print("1. 📓 Try the interactive demo:")
print("   jupyter notebook notebooks/demo.ipynb")
print()
print("2. 🔬 Run larger experiments:")
print("   python scripts/run_all_cases.py --scale medium --cases 1,2,3")
print()
print("3. 🧪 Test the system:")
print("   python tests/test_system.py")
print()
print("4. 📖 Read the documentation:")
print("   Check README.md for detailed usage instructions")
print()
print("5. 🎯 For production scale (10^9 primes):")
print("   python scripts/run_all_cases.py --scale huge --parallel")
print("   (Warning: This can take 24+ hours and requires significant resources)")

print(f"\n🎉 Quick start complete! The Q8 prime bias experiment system is ready.")
print("=" * 55)
print("📚 Reference: Aoki, M. and Koyama, S. (2022)")
print("    Chebyshev's Bias against Splitting and Principal Primes")
print("    in Global Fields. arXiv:2203.12266")
print("=" * 55)
