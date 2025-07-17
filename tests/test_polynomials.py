#!/usr/bin/env python3
"""
Test Suite for Omar Polynomial Definitions

Tests for the 13 quaternion polynomial cases including:
- Polynomial structure validation
- Mathematical properties verification
- Galois group properties
- Discriminant calculations
"""

import sys
import unittest
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import sympy as sp
from sympy import symbols, isprime, factorint
from omar_polynomials import (
    get_case, get_all_cases, OMAR_COLLECTION,
    QuaternionPolynomial, OmarPolynomialCollection
)

class TestPolynomialStructure(unittest.TestCase):
    """Test basic polynomial structure"""
    
    def test_all_cases_exist(self):
        """Test that all 13 cases are properly defined"""
        all_cases = get_all_cases()
        self.assertEqual(len(all_cases), 13)
        
        # Check case IDs are 1-13
        case_ids = [case.case_id for case in all_cases]
        self.assertEqual(sorted(case_ids), list(range(1, 14)))
    
    def test_polynomial_degrees(self):
        """Test that all polynomials have degree 8"""
        for case in get_all_cases():
            degree = case.polynomial.as_poly().degree()
            self.assertEqual(degree, 8, f"Case {case.case_id} should have degree 8")
    
    def test_polynomial_coefficients(self):
        """Test that polynomials have integer coefficients"""
        x = symbols('x')
        
        for case in get_all_cases():
            poly = sp.Poly(case.polynomial, x)
            coeffs = poly.all_coeffs()
            
            # All coefficients should be integers
            for coeff in coeffs:
                self.assertIsInstance(coeff, (int, sp.Integer), 
                                    f"Case {case.case_id} has non-integer coefficient: {coeff}")
    
    def test_case_properties(self):
        """Test that each case has required properties"""
        required_properties = ['case_id', 'm_rho0', 'ramified_primes', 'polynomial']
        
        for case in get_all_cases():
            # Check basic properties exist
            for prop in required_properties:
                self.assertTrue(hasattr(case, prop), 
                              f"Case {case.case_id} missing property: {prop}")
            
            # Check m_rho0 is 0 or 1
            self.assertIn(case.m_rho0, [0, 1], 
                         f"Case {case.case_id} has invalid m_rho0: {case.m_rho0}")
            
            # Check ramified_primes is a set of primes
            self.assertIsInstance(case.ramified_primes, set)
            for p in case.ramified_primes:
                self.assertTrue(isprime(p), 
                               f"Case {case.case_id} has non-prime ramified: {p}")

class TestSpecificCases(unittest.TestCase):
    """Test specific polynomial cases"""
    
    def test_case_1_aoki_koyama(self):
        """Test Case 1 (Aoki-Koyama Example 2.1)"""
        case1 = get_case(1)
        
        # Check basic properties
        self.assertEqual(case1.case_id, 1)
        self.assertEqual(case1.m_rho0, 0)
        self.assertEqual(case1.ramified_primes, {3, 5, 7})
        
        # Check polynomial matches expected
        x = symbols('x')
        expected_poly = (x**8 - x**7 - 34*x**6 + 29*x**5 + 
                        361*x**4 - 305*x**3 - 1090*x**2 + 1345*x - 395)
        
        self.assertEqual(sp.expand(case1.polynomial), sp.expand(expected_poly))
        
        # Check discriminant factors
        discriminant_info = case1.case_info['discriminant']
        self.assertEqual(discriminant_info, '3**6 * 5**6 * 7**6')
    
    def test_case_2_related(self):
        """Test Case 2 (related to Case 1)"""
        case2 = get_case(2)
        
        self.assertEqual(case2.case_id, 2)
        self.assertEqual(case2.m_rho0, 1)  # Different from Case 1
        self.assertEqual(case2.ramified_primes, {3, 5, 7})  # Same as Case 1
        
        # Check polynomial
        x = symbols('x')
        expected_poly = x**8 + 315*x**6 + 34020*x**4 + 1488375*x**2 + 22325625
        self.assertEqual(sp.expand(case2.polynomial), sp.expand(expected_poly))
    
    def test_case_3_different_discriminant(self):
        """Test Case 3 (different discriminant)"""
        case3 = get_case(3)
        
        self.assertEqual(case3.case_id, 3)
        self.assertEqual(case3.m_rho0, 1)
        self.assertEqual(case3.ramified_primes, {5, 41})  # Different ramification
        
        # Check discriminant
        discriminant_info = case3.case_info['discriminant']
        self.assertEqual(discriminant_info, '5**6 * 41**6')
    
    def test_case_4_binary_ramification(self):
        """Test Case 4 (ramified only at 2)"""
        case4 = get_case(4)
        
        self.assertEqual(case4.case_id, 4)
        self.assertEqual(case4.m_rho0, 0)
        self.assertEqual(case4.ramified_primes, {2})  # Only ramified at 2
        
        # Check polynomial
        x = symbols('x')
        expected_poly = x**8 - 2*x**6 + 4*x**4 - 8*x**2 + 16
        self.assertEqual(sp.expand(case4.polynomial), sp.expand(expected_poly))

class TestMathematicalProperties(unittest.TestCase):
    """Test mathematical properties of polynomials"""
    
    def test_polynomial_irreducibility_simple(self):
        """Test basic irreducibility checks (simplified)"""
        x = symbols('x')
        
        for case in get_all_cases()[:3]:  # Test first 3 cases
            poly = case.polynomial
            
            # Check that polynomial is not obviously reducible
            # (full irreducibility testing is computationally expensive)
            
            # Should not have obvious rational roots
            coeffs = sp.Poly(poly, x).all_coeffs()
            
            # By rational root theorem, possible roots are divisors of constant term
            constant_term = coeffs[-1]
            if constant_term != 0:
                divisors = [1, -1]
                if abs(constant_term) > 1:
                    for d in range(2, min(11, abs(constant_term) + 1)):
                        if constant_term % d == 0:
                            divisors.extend([d, -d, constant_term//d, -constant_term//d])
                
                # Check that none of these are roots
                for potential_root in set(divisors):
                    value = poly.subs(x, potential_root)
                    self.assertNotEqual(value, 0, 
                                      f"Case {case.case_id} has rational root {potential_root}")
    
    def test_galois_group_consistency(self):
        """Test that Galois group properties are consistent"""
        for case in get_all_cases():
            # All cases should be Q8 extensions
            self.assertEqual(case.case_info['galois_group'], 'Q8')
            self.assertEqual(case.case_info['degree'], 8)
            
            # Check bias coefficients structure
            bias_coeffs = case.get_bias_coefficients()
            self.assertEqual(len(bias_coeffs), 8)
            
            # Check conjugacy sizes
            conj_sizes = case.get_conjugacy_sizes()
            self.assertEqual(len(conj_sizes), 8)
            
            # Q8 has the conjugacy class structure: 1+1+2+2+2
            size_counts = {}
            for size in conj_sizes.values():
                size_counts[size] = size_counts.get(size, 0) + 1
            
            self.assertEqual(size_counts.get(1, 0), 2)  # Two size-1 classes: {1}, {-1}
            self.assertEqual(size_counts.get(2, 0), 6)  # Six size-2 classes (but g5,g6,g7 = g2,g3,g4)
    
    def test_discriminant_consistency(self):
        """Test discriminant and ramification consistency"""
        for case in get_all_cases():
            discriminant_str = case.case_info['discriminant']
            ramified_primes = case.ramified_primes
            
            # Extract primes from discriminant string
            # This is a simplified check - full discriminant calculation is complex
            for p in ramified_primes:
                self.assertIn(str(p), discriminant_str, 
                             f"Case {case.case_id}: ramified prime {p} not in discriminant {discriminant_str}")

class TestBiasCoefficients(unittest.TestCase):
    """Test bias coefficient calculations"""
    
    def test_coefficient_values(self):
        """Test that bias coefficients have expected values"""
        # Test m_rho0 = 0 cases
        for case in get_all_cases():
            coeffs = case.get_bias_coefficients()
            
            if case.m_rho0 == 0:
                self.assertEqual(coeffs['g0'], 0.5)
                self.assertEqual(coeffs['g1'], 2.5)
                self.assertEqual(coeffs['g2'], -0.5)
                self.assertEqual(coeffs['g3'], -0.5)
                self.assertEqual(coeffs['g4'], -0.5)
            else:  # m_rho0 == 1
                self.assertEqual(coeffs['g0'], 2.5)
                self.assertEqual(coeffs['g1'], 0.5)
                self.assertEqual(coeffs['g2'], -0.5)
                self.assertEqual(coeffs['g3'], -0.5)
                self.assertEqual(coeffs['g4'], -0.5)
    
    def test_coefficient_sum_property(self):
        """Test sum properties of bias coefficients"""
        for case in get_all_cases():
            coeffs = case.get_bias_coefficients()
            conj_sizes = case.get_conjugacy_sizes()
            
            # Calculate weighted sum (coefficients weighted by conjugacy class sizes)
            weighted_sum = 0
            for key in ['g0', 'g1', 'g2', 'g3', 'g4']:
                # Note: g5,g6,g7 are the same as g2,g3,g4 respectively
                weighted_sum += coeffs[key] * conj_sizes[key]
            
            # The weighted sum should be finite and well-defined
            self.assertIsInstance(weighted_sum, (int, float))
            self.assertFalse(sp.oo == weighted_sum)  # Should not be infinite
    
    def test_coefficient_consistency_across_cases(self):
        """Test that coefficients are consistent for cases with same m_rho0"""
        cases_by_m_rho0 = {0: [], 1: []}
        
        for case in get_all_cases():
            cases_by_m_rho0[case.m_rho0].append(case)
        
        # All cases with same m_rho0 should have same bias coefficients
        for m_rho0, cases in cases_by_m_rho0.items():
            if len(cases) > 1:
                reference_coeffs = cases[0].get_bias_coefficients()
                for case in cases[1:]:
                    case_coeffs = case.get_bias_coefficients()
                    self.assertEqual(case_coeffs, reference_coeffs,
                                   f"Case {case.case_id} coefficients differ from reference for m_rho0={m_rho0}")

class TestPolynomialCollection(unittest.TestCase):
    """Test the polynomial collection management"""
    
    def test_collection_initialization(self):
        """Test that collection initializes properly"""
        collection = OmarPolynomialCollection()
        
        self.assertEqual(len(collection.polynomials), 13)
        self.assertEqual(len(collection.get_all_cases()), 13)
        self.assertEqual(collection.get_available_case_ids(), list(range(1, 14)))
    
    def test_case_retrieval(self):
        """Test case retrieval methods"""
        collection = OmarPolynomialCollection()
        
        # Test get_case
        case1 = collection.get_case(1)
        self.assertEqual(case1.case_id, 1)
        
        # Test invalid case
        with self.assertRaises(ValueError):
            collection.get_case(14)
        
        with self.assertRaises(ValueError):
            collection.get_case(0)
    
    def test_filtering_by_m_rho0(self):
        """Test filtering cases by m_rho0 value"""
        collection = OmarPolynomialCollection()
        
        cases_0 = collection.get_cases_by_m_rho0(0)
        cases_1 = collection.get_cases_by_m_rho0(1)
        
        # Should have some cases for each value
        self.assertGreater(len(cases_0), 0)
        self.assertGreater(len(cases_1), 0)
        
        # Total should be 13
        self.assertEqual(len(cases_0) + len(cases_1), 13)
        
        # Check that filtering is correct
        for case in cases_0:
            self.assertEqual(case.m_rho0, 0)
        
        for case in cases_1:
            self.assertEqual(case.m_rho0, 1)
    
    def test_export_functionality(self):
        """Test JSON export functionality"""
        import tempfile
        import json
        
        collection = OmarPolynomialCollection()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Test export
            collection.export_to_json(temp_file)
            
            # Test that file was created and contains valid JSON
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.assertIn('description', data)
            self.assertIn('total_cases', data)
            self.assertIn('cases', data)
            self.assertEqual(data['total_cases'], 13)
            self.assertEqual(len(data['cases']), 13)
            
        finally:
            # Cleanup
            Path(temp_file).unlink()

class TestGlobalFunctions(unittest.TestCase):
    """Test global convenience functions"""
    
    def test_get_case_function(self):
        """Test global get_case function"""
        case1 = get_case(1)
        self.assertEqual(case1.case_id, 1)
        
        # Test that it returns the same object as collection method
        collection_case1 = OMAR_COLLECTION.get_case(1)
        self.assertEqual(case1.case_id, collection_case1.case_id)
        self.assertEqual(case1.m_rho0, collection_case1.m_rho0)
    
    def test_get_all_cases_function(self):
        """Test global get_all_cases function"""
        all_cases = get_all_cases()
        self.assertEqual(len(all_cases), 13)
        
        # Check that IDs are correct
        case_ids = [case.case_id for case in all_cases]
        self.assertEqual(sorted(case_ids), list(range(1, 14)))
    
    def test_legacy_compatibility(self):
        """Test that legacy OMAR_POLYNOMIALS list exists and is correct"""
        from omar_polynomials import OMAR_POLYNOMIALS
        
        self.assertEqual(len(OMAR_POLYNOMIALS), 13)
        
        # Check that it contains dictionaries with required keys
        required_keys = ['id', 'poly', 'm_rho0', 'ramified_primes']
        
        for poly_dict in OMAR_POLYNOMIALS:
            for key in required_keys:
                self.assertIn(key, poly_dict)
            
            # Check that ID is in valid range
            self.assertIn(poly_dict['id'], range(1, 14))

if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2)
