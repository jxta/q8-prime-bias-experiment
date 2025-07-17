#!/usr/bin/env python3
"""
Test suite for Q8 Prime Bias Experiment

Comprehensive tests to verify all components work correctly.
Run this to validate your installation and setup.

Usage:
    python tests/test_system.py
    python tests/test_system.py --verbose
    python tests/test_system.py --quick
"""

import os
import sys
import time
import unittest
from pathlib import Path
import tempfile
import shutil

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Test imports - this will reveal any missing dependencies
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import sympy as sp
    from sympy import isprime
    
    # Our modules
    from omar_polynomials import get_case, get_all_cases, print_all_cases_summary
    from fast_frobenius_calculator import (
        FastFrobeniusCalculator, ParallelFrobeniusComputation,
        FrobeniusElementComputer, save_frobenius_results, load_frobenius_results
    )
    from bias_analyzer import BiasAnalyzer, WeightedPrimeCounter, BiasCalculator
    from experiment_runner import QuaternionBiasExperiment
    from utils import (
        Timer, DirectoryManager, ConfigManager,
        estimate_memory_usage, estimate_computation_time,
        format_number, format_duration
    )
    
    IMPORTS_OK = True
    IMPORT_ERROR = None
    
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

class TestImports(unittest.TestCase):
    """Test that all required modules can be imported"""
    
    def test_imports(self):
        """Test all imports work"""
        if not IMPORTS_OK:
            self.fail(f"Import failed: {IMPORT_ERROR}")
        
        # Test basic functionality of key libraries
        self.assertTrue(callable(np.array))
        self.assertTrue(callable(plt.plot))
        self.assertTrue(callable(sp.sympify))
        self.assertTrue(callable(isprime))

class TestOmarPolynomials(unittest.TestCase):
    """Test Omar polynomial definitions and case management"""
    
    def test_get_all_cases(self):
        """Test that all 13 cases are available"""
        cases = get_all_cases()
        self.assertEqual(len(cases), 13, "Should have exactly 13 cases")
        
        # Check case IDs are 1-13
        case_ids = [case.case_id for case in cases]
        self.assertEqual(sorted(case_ids), list(range(1, 14)))
    
    def test_case_1_aoki_koyama(self):
        """Test Case 1 (Aoki-Koyama Example 2.1)"""
        case1 = get_case(1)
        
        # Basic properties
        self.assertEqual(case1.case_id, 1)
        self.assertEqual(case1.m_rho0, 0)
        self.assertIn(3, case1.ramified_primes)
        self.assertIn(5, case1.ramified_primes)
        self.assertIn(7, case1.ramified_primes)
        
        # Polynomial should be defined
        self.assertIsNotNone(case1.polynomial)
        
        # Bias coefficients
        bias_coeffs = case1.get_bias_coefficients()
        self.assertIn('g0', bias_coeffs)
        self.assertIn('g1', bias_coeffs)
        
        # For m_ρ₀ = 0, expect specific values
        self.assertEqual(bias_coeffs['g0'], 0.5)
        self.assertEqual(bias_coeffs['g1'], 2.5)
    
    def test_m_rho0_distribution(self):
        """Test distribution of m_ρ₀ values"""
        cases = get_all_cases()
        
        m_rho0_0_count = sum(1 for case in cases if case.m_rho0 == 0)
        m_rho0_1_count = sum(1 for case in cases if case.m_rho0 == 1)
        
        # Should have both types
        self.assertGreater(m_rho0_0_count, 0, "Should have cases with m_ρ₀ = 0")
        self.assertGreater(m_rho0_1_count, 0, "Should have cases with m_ρ₀ = 1")
        self.assertEqual(m_rho0_0_count + m_rho0_1_count, 13)

class TestFrobeniusCalculation(unittest.TestCase):
    """Test Frobenius element computation"""
    
    def setUp(self):
        """Setup for Frobenius tests"""
        self.case1 = get_case(1)
        self.computer = FrobeniusElementComputer(self.case1, verbose=False)
        self.calculator = FastFrobeniusCalculator(verbose=False)
    
    def test_frobenius_computer_initialization(self):
        """Test FrobeniusElementComputer initialization"""
        self.assertEqual(self.computer.case.case_id, 1)
        self.assertIsNotNone(self.computer.polynomial)
        self.assertIsInstance(self.computer.ramified_primes, set)
    
    def test_ramified_prime_detection(self):
        """Test ramified prime detection"""
        # Case 1 ramified primes: 3, 5, 7
        self.assertTrue(self.computer.is_prime_ramified(3))
        self.assertTrue(self.computer.is_prime_ramified(5))
        self.assertTrue(self.computer.is_prime_ramified(7))
        
        # Non-ramified primes
        self.assertFalse(self.computer.is_prime_ramified(2))
        self.assertFalse(self.computer.is_prime_ramified(11))
        self.assertFalse(self.computer.is_prime_ramified(13))
    
    def test_single_frobenius_computation(self):
        """Test computing Frobenius element for single prime"""
        # Test with a few small primes
        test_primes = [2, 11, 13, 17, 19]
        
        for p in test_primes:
            self.assertTrue(isprime(p), f"{p} should be prime")
            
            frobenius = self.computer.compute_frobenius_element(p)
            
            # Should return integer in range 0-7 (Q8 has 8 elements)
            self.assertIsInstance(frobenius, int)
            self.assertGreaterEqual(frobenius, 0)
            self.assertLessEqual(frobenius, 7)
    
    def test_ramified_prime_frobenius(self):
        """Test that ramified primes give identity element"""
        ramified_primes = [3, 5, 7]
        
        for p in ramified_primes:
            frobenius = self.computer.compute_frobenius_element(p)
            # Ramified primes should give identity (0)
            self.assertEqual(frobenius, 0, f"Ramified prime {p} should give Frobenius element 0")
    
    def test_batch_computation(self):
        """Test batch Frobenius computation"""
        test_primes = [2, 11, 13, 17, 19, 23, 29, 31]
        
        results = self.computer.compute_batch(test_primes)
        
        # Check results
        self.assertEqual(len(results), len(test_primes))
        
        for p in test_primes:
            self.assertIn(p, results)
            frobenius = results[p]
            self.assertGreaterEqual(frobenius, 0)
            self.assertLessEqual(frobenius, 7)
    
    def test_sequential_computation(self):
        """Test sequential computation with small range"""
        # Small test to avoid taking too long
        max_prime = 100
        
        results = self.calculator.compute_case_sequential(1, max_prime)
        
        # Check results
        self.assertGreater(len(results), 0, "Should compute some Frobenius elements")
        
        # All keys should be primes
        for p in results.keys():
            self.assertTrue(isprime(p), f"{p} should be prime")
        
        # All values should be valid Frobenius elements
        for frobenius in results.values():
            self.assertGreaterEqual(frobenius, 0)
            self.assertLessEqual(frobenius, 7)

class TestBiasAnalysis(unittest.TestCase):
    """Test bias analysis components"""
    
    def setUp(self):
        """Setup for bias tests"""
        self.analyzer = BiasAnalyzer(verbose=False)
        self.counter = WeightedPrimeCounter()
        self.bias_calc = BiasCalculator(1)
    
    def test_weighted_prime_counting(self):
        """Test weighted prime counting functions"""
        test_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        # Test π₁/₂(x)
        pi_half = self.counter.compute_pi_half(test_primes)
        
        self.assertEqual(len(pi_half), len(test_primes))
        
        # Should be monotonically increasing
        for i in range(1, len(pi_half)):
            self.assertGreaterEqual(pi_half[i], pi_half[i-1])
        
        # First value should be 1/√2 ≈ 0.707
        expected_first = 1.0 / np.sqrt(2)
        self.assertAlmostEqual(pi_half[0], expected_first, places=5)
    
    def test_frobenius_specific_counting(self):
        """Test π₁/₂(x;σ) counting"""
        test_primes = [2, 3, 5, 7, 11, 13, 17, 19]
        test_frobenius = [1, 0, 0, 0, 2, 3, 1, 2]  # Example values
        
        # Count for Frobenius element 0 (identity)
        pi_half_0 = self.counter.compute_pi_half_sigma(test_primes, test_frobenius, 0)
        
        # Should only count primes with Frobenius element 0
        # In our example: primes 3, 5, 7 have Frobenius 0
        expected_indices = [1, 2, 3]  # indices for primes 3, 5, 7
        
        # Check that values increase only at correct positions
        self.assertEqual(pi_half_0[0], 0)  # p=2 has Frobenius=1, not 0
        self.assertGreater(pi_half_0[1], 0)  # p=3 has Frobenius=0
    
    def test_bias_coefficient_calculation(self):
        """Test bias coefficient calculation"""
        case1_coeffs = self.bias_calc.bias_coefficients
        
        # Check that all required coefficients are present
        required_coeffs = ['g0', 'g1', 'g2', 'g3', 'g4']
        for coeff in required_coeffs:
            self.assertIn(coeff, case1_coeffs)
        
        # For Case 1 (m_ρ₀ = 0), check expected values
        self.assertEqual(case1_coeffs['g0'], 0.5)
        self.assertEqual(case1_coeffs['g1'], 2.5)
        self.assertEqual(case1_coeffs['g2'], -0.5)

class TestExperimentRunner(unittest.TestCase):
    """Test the complete experiment runner"""
    
    def setUp(self):
        """Setup for experiment tests"""
        # Create temporary directory for test results
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Initialize experiment with minimal verbosity for testing
        self.experiment = QuaternionBiasExperiment(verbose=False)
    
    def tearDown(self):
        """Cleanup after tests"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_experiment_initialization(self):
        """Test experiment initialization"""
        self.assertIsNotNone(self.experiment)
        self.assertIsNotNone(self.experiment.parallel_computer)
        self.assertIsNotNone(self.experiment.bias_analyzer)
    
    def test_validation(self):
        """Test experiment validation"""
        validation = self.experiment.validate_experiment_setup(
            case_ids=[1, 2],
            max_prime=1000,
            num_processes=2
        )
        
        self.assertTrue(validation['valid'])
        self.assertEqual(validation['case_ids'], [1, 2])
        self.assertEqual(validation['max_prime'], 1000)
        self.assertGreater(validation['estimated_primes'], 0)
    
    def test_small_computation(self):
        """Test small computation to verify system works"""
        # Very small test to avoid timeout
        max_prime = 50  # Only primes up to 50
        
        start_time = time.time()
        results = self.experiment.run_case(
            case_id=1,
            max_prime=max_prime,
            num_processes=1,
            save_results=False  # Don't save in test
        )
        end_time = time.time()
        
        # Check results
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0, "Should compute some results")
        
        # Should complete quickly
        self.assertLess(end_time - start_time, 10, "Small computation should be fast")
        
        # All keys should be primes ≤ max_prime
        for p in results.keys():
            self.assertTrue(isprime(p))
            self.assertLessEqual(p, max_prime)

class TestUtilities(unittest.TestCase):
    """Test utility functions"""
    
    def test_format_functions(self):
        """Test number and duration formatting"""
        # Test format_number
        self.assertEqual(format_number(500), "500")
        self.assertEqual(format_number(1500), "1K")
        self.assertEqual(format_number(1500000), "1M")
        self.assertEqual(format_number(1500000000), "1B")
        
        # Test format_duration
        self.assertEqual(format_duration(30), "30.0s")
        self.assertEqual(format_duration(90), "1.5m")
        self.assertEqual(format_duration(7200), "2.0h")
    
    def test_estimates(self):
        """Test resource estimation functions"""
        # Test memory estimation
        memory_est = estimate_memory_usage(10000, 1)
        self.assertIn('total_gb', memory_est)
        self.assertGreater(memory_est['total_gb'], 0)
        
        # Test time estimation
        time_est = estimate_computation_time(10000, 1, 4)
        self.assertIn('parallel_time_seconds', time_est)
        self.assertGreater(time_est['parallel_time_seconds'], 0)
    
    def test_timer(self):
        """Test Timer context manager"""
        with Timer("Test operation") as timer:
            time.sleep(0.1)  # Sleep for 0.1 seconds
        
        self.assertGreaterEqual(timer.duration, 0.09)  # Should be at least 0.09 seconds
        self.assertLessEqual(timer.duration, 0.2)     # Should be less than 0.2 seconds

class TestFileOperations(unittest.TestCase):
    """Test file save/load operations"""
    
    def setUp(self):
        """Setup temporary directory"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_frobenius_save_load(self):
        """Test saving and loading Frobenius results"""
        # Create test data
        test_results = {2: 1, 3: 0, 5: 0, 7: 0, 11: 2, 13: 3}
        
        # Save results
        filepath = save_frobenius_results(
            case_id=1,
            results=test_results,
            max_prime=13,
            output_dir=self.test_dir
        )
        
        # Check file was created
        self.assertTrue(Path(filepath).exists())
        
        # Load results back
        case_id, loaded_results, metadata = load_frobenius_results(filepath)
        
        # Verify loaded data
        self.assertEqual(case_id, 1)
        self.assertEqual(loaded_results, test_results)
        self.assertEqual(metadata['max_prime'], 13)

def run_quick_tests():
    """Run only quick tests"""
    suite = unittest.TestSuite()
    
    # Add quick tests
    suite.addTest(TestImports('test_imports'))
    suite.addTest(TestOmarPolynomials('test_get_all_cases'))
    suite.addTest(TestOmarPolynomials('test_case_1_aoki_koyama'))
    suite.addTest(TestFrobeniusCalculation('test_frobenius_computer_initialization'))
    suite.addTest(TestFrobeniusCalculation('test_ramified_prime_detection'))
    suite.addTest(TestBiasAnalysis('test_weighted_prime_counting'))
    suite.addTest(TestUtilities('test_format_functions'))
    
    return suite

def run_all_tests():
    """Run all tests"""
    return unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Q8 Prime Bias Experiment System')
    parser.add_argument('--quick', action='store_true', help='Run only quick tests')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    # Configure test runner
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    print("=" * 60)
    print("Q8 PRIME BIAS EXPERIMENT - SYSTEM TESTS")
    print("=" * 60)
    
    if not IMPORTS_OK:
        print(f"❌ CRITICAL: Import failed: {IMPORT_ERROR}")
        print("Please install required dependencies:")
        print("  pip install -r requirements.txt")
        return 1
    
    print("✅ All imports successful")
    print()
    
    # Run tests
    if args.quick:
        print("Running quick tests...")
        suite = run_quick_tests()
    else:
        print("Running all tests...")
        suite = run_all_tests()
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
        print("✅ System is ready for Q8 prime bias experiments")
        print("\nNext steps:")
        print("  1. Try the demo notebook: jupyter notebook notebooks/demo.ipynb")
        print("  2. Run a quick experiment: python scripts/run_all_cases.py --scale small")
        print("  3. Check the README for detailed usage instructions")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print("\nPlease check the error messages above and fix any issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
