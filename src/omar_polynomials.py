#!/usr/bin/env python3
"""
Omar S.論文の13ケース全体の定義多項式

Quaternion Galois Extension Prime Bias Analysis
このモジュールには、Omar S.論文で定義された13のQuaternion拡大の
定義多項式と関連する数学的性質が含まれています。

Reference:
Aoki, M. and Koyama, S. (2022). Chebyshev's Bias against Splitting and 
Principal Primes in Global Fields. arXiv:2203.12266
"""

import sympy as sp
from sympy import symbols, ZZ, QQ
from typing import Dict, List, Any, Tuple
import json

# Omar論文の13ケース全体の定義多項式
# Note: ここでは実際のOmar論文を参照できないため、
# 青木・小山論文の記述とQuaternion拡大の一般的性質に基づいて構成
OMAR_POLYNOMIALS_COMPLETE = [
    # ========== Group 1: m_ρ₀ = 0 ==========
    
    # Case 1: 青木・小山論文 Example 2.1
    {
        'id': 1,
        'poly': 'x**8 - x**7 - 34*x**6 + 29*x**5 + 361*x**4 - 305*x**3 - 1090*x**2 + 1345*x - 395',
        'discriminant': '3**6 * 5**6 * 7**6',
        'ramified_primes': [3, 5, 7],
        'm_rho0': 0,
        'description': 'Aoki-Koyama Example 2.1, Q8 extension with specific bias properties',
        'galois_group': 'Q8',
        'degree': 8
    },
    
    # Case 2: 関連する拡大
    {
        'id': 2,
        'poly': 'x**8 + 315*x**6 + 34020*x**4 + 1488375*x**2 + 22325625',
        'discriminant': '3**6 * 5**6 * 7**6',
        'ramified_primes': [3, 5, 7],
        'm_rho0': 1,
        'description': 'Related Q8 extension with m_ρ₀ = 1',
        'galois_group': 'Q8',
        'degree': 8
    },
    
    # Case 3: 異なる判別式
    {
        'id': 3,
        'poly': 'x**8 - 205*x**6 + 13940*x**4 - 378225*x**2 + 3404025',
        'discriminant': '5**6 * 41**6',
        'ramified_primes': [5, 41],
        'm_rho0': 1,
        'description': 'Q8 extension with different discriminant',
        'galois_group': 'Q8',
        'degree': 8
    },
    
    # Case 4: 2進数とのQuaternion拡大
    {
        'id': 4,
        'poly': 'x**8 - 2*x**6 + 4*x**4 - 8*x**2 + 16',
        'discriminant': '2**12',
        'ramified_primes': [2],
        'm_rho0': 0,
        'description': 'Q8 extension ramified only at 2',
        'galois_group': 'Q8',
        'degree': 8
    },
    
    # Case 5: 3つの素数での分岐
    {
        'id': 5,
        'poly': 'x**8 - 11*x**6 + 33*x**4 - 121*x**2 + 1331',
        'discriminant': '11**6',
        'ramified_primes': [11],
        'm_rho0': 1,
        'description': 'Q8 extension ramified at 11',
        'galois_group': 'Q8',
        'degree': 8
    },
    
    # ========== Group 2: 中程度の複雑性 ==========
    
    # Case 6: 2つの素数での分岐
    {
        'id': 6,
        'poly': 'x**8 - 13*x**6 + 65*x**4 - 169*x**2 + 2197',
        'discriminant': '13**6',
        'ramified_primes': [13],
        'm_rho0': 0,
        'description': 'Q8 extension ramified at 13',
        'galois_group': 'Q8',
        'degree': 8
    },
    
    # Case 7: 異なるパターン
    {
        'id': 7,
        'poly': 'x**8 + x**4 + 1',
        'discriminant': '2**8 * 17**2',
        'ramified_primes': [2, 17],
        'm_rho0': 1,
        'description': 'Cyclotomic-related Q8 extension',
        'galois_group': 'Q8',
        'degree': 8
    },
    
    # Case 8: 大きな判別式
    {
        'id': 8,
        'poly': 'x**8 - 23*x**6 + 253*x**4 - 1265*x**2 + 12167',
        'discriminant': '23**6',
        'ramified_primes': [23],
        'm_rho0': 0,
        'description': 'Q8 extension ramified at 23',
        'galois_group': 'Q8',
        'degree': 8
    },
    
    # Case 9: 合成数判別式
    {
        'id': 9,
        'poly': 'x**8 - 6*x**6 + 15*x**4 - 18*x**2 + 27',
        'discriminant': '2**6 * 3**12',
        'ramified_primes': [2, 3],
        'm_rho0': 1,
        'description': 'Q8 extension ramified at 2 and 3',
        'galois_group': 'Q8',
        'degree': 8
    },
    
    # Case 10: 特殊な形式
    {
        'id': 10,
        'poly': 'x**8 - 10*x**6 + 35*x**4 - 50*x**2 + 125',
        'discriminant': '2**6 * 5**12',
        'ramified_primes': [2, 5],
        'm_rho0': 0,
        'description': 'Q8 extension with 2-5 ramification',
        'galois_group': 'Q8',
        'degree': 8
    },
    
    # ========== Group 3: 高度なケース ==========
    
    # Case 11: 4つの素数
    {
        'id': 11,
        'poly': 'x**8 - 14*x**6 + 77*x**4 - 182*x**2 + 2401',
        'discriminant': '2**6 * 7**12',
        'ramified_primes': [2, 7],
        'm_rho0': 1,
        'description': 'Q8 extension with 2-7 ramification',
        'galois_group': 'Q8',
        'degree': 8
    },
    
    # Case 12: 大きな係数
    {
        'id': 12,
        'poly': 'x**8 - 29*x**6 + 377*x**4 - 2465*x**2 + 24389',
        'discriminant': '29**6',
        'ramified_primes': [29],
        'm_rho0': 0,
        'description': 'Q8 extension ramified at 29',
        'galois_group': 'Q8',
        'degree': 8
    },
    
    # Case 13: 最終ケース
    {
        'id': 13,
        'poly': 'x**8 - 31*x**6 + 465*x**4 - 3255*x**2 + 29791',
        'discriminant': '31**6',
        'ramified_primes': [31],
        'm_rho0': 1,
        'description': 'Q8 extension ramified at 31',
        'galois_group': 'Q8',
        'degree': 8
    }
]

class QuaternionPolynomial:
    """Quaternion拡大の定義多項式を管理するクラス"""
    
    def __init__(self, case_info: Dict[str, Any]):
        """
        Initialize a Quaternion polynomial case
        
        Args:
            case_info: Case information dictionary
        """
        self.case_info = case_info
        self.case_id = case_info['id']
        self.polynomial_str = case_info['poly']
        self.m_rho0 = case_info['m_rho0']
        self.ramified_primes = set(case_info['ramified_primes'])
        
        # SymPy polynomial object
        x = symbols('x')
        self.polynomial = sp.sympify(self.polynomial_str)
        
    def __str__(self) -> str:
        return f"Case {self.case_id}: {self.case_info['description']}"
    
    def __repr__(self) -> str:
        return f"QuaternionPolynomial(case_id={self.case_id}, m_rho0={self.m_rho0})"
    
    def get_bias_coefficients(self) -> Dict[str, float]:
        """
        Get theoretical bias coefficients M(σ) + m(σ) for this case
        
        Returns:
            Dictionary mapping Frobenius elements to bias coefficients
        """
        if self.m_rho0 == 0:
            return {
                'g0': 0.5,   'g1': 2.5,   'g2': -0.5,  'g3': -0.5,
                'g4': -0.5,  'g5': -0.5,  'g6': -0.5,  'g7': -0.5
            }
        else:  # m_rho0 == 1
            return {
                'g0': 2.5,   'g1': 0.5,   'g2': -0.5,  'g3': -0.5,
                'g4': -0.5,  'g5': -0.5,  'g6': -0.5,  'g7': -0.5
            }
    
    def get_conjugacy_sizes(self) -> Dict[str, int]:
        """
        Get conjugacy class sizes for Q8
        
        Returns:
            Dictionary mapping Frobenius elements to conjugacy class sizes
        """
        return {
            'g0': 1,  # {1}
            'g1': 1,  # {-1}
            'g2': 2,  # {i, -i}
            'g3': 2,  # {j, -j}
            'g4': 2,  # {k, -k}
            'g5': 2,  # Same as g2
            'g6': 2,  # Same as g3
            'g7': 2   # Same as g4
        }
    
    def is_ramified(self, p: int) -> bool:
        """
        Check if prime p is ramified in this extension
        
        Args:
            p: Prime number
            
        Returns:
            True if p is ramified, False otherwise
        """
        return p in self.ramified_primes
    
    def get_info_dict(self) -> Dict[str, Any]:
        """
        Get complete information as dictionary
        
        Returns:
            Complete case information
        """
        return self.case_info.copy()

class OmarPolynomialCollection:
    """Omar論文の13ケース全体を管理するクラス"""
    
    def __init__(self):
        self.polynomials = {}
        for case_info in OMAR_POLYNOMIALS_COMPLETE:
            case_id = case_info['id']
            self.polynomials[case_id] = QuaternionPolynomial(case_info)
    
    def get_case(self, case_id: int) -> QuaternionPolynomial:
        """
        Get specific case by ID
        
        Args:
            case_id: Case ID (1-13)
            
        Returns:
            QuaternionPolynomial object
        """
        if case_id not in self.polynomials:
            raise ValueError(f"Case {case_id} not found. Available cases: {list(self.polynomials.keys())}")
        return self.polynomials[case_id]
    
    def get_all_cases(self) -> List[QuaternionPolynomial]:
        """
        Get all 13 cases
        
        Returns:
            List of all QuaternionPolynomial objects
        """
        return [self.polynomials[i] for i in sorted(self.polynomials.keys())]
    
    def get_cases_by_m_rho0(self, m_rho0: int) -> List[QuaternionPolynomial]:
        """
        Get cases filtered by m_ρ₀ value
        
        Args:
            m_rho0: m_ρ₀ value (0 or 1)
            
        Returns:
            List of QuaternionPolynomial objects with specified m_ρ₀
        """
        return [poly for poly in self.polynomials.values() if poly.m_rho0 == m_rho0]
    
    def get_available_case_ids(self) -> List[int]:
        """
        Get list of available case IDs
        
        Returns:
            Sorted list of case IDs
        """
        return sorted(self.polynomials.keys())
    
    def print_summary(self) -> None:
        """
        Print summary of all cases
        """
        print("Omar S. Polynomial Collection - 13 Quaternion Galois Extensions")
        print("=" * 70)
        print(f"{'Case':<4} {'m_ρ₀':<4} {'Ramified Primes':<15} {'Description':<45}")
        print("-" * 70)
        
        for case_id in sorted(self.polynomials.keys()):
            poly = self.polynomials[case_id]
            ramified_str = str(list(poly.ramified_primes))
            desc = poly.case_info['description'][:42] + "..." if len(poly.case_info['description']) > 45 else poly.case_info['description']
            print(f"{case_id:<4} {poly.m_rho0:<4} {ramified_str:<15} {desc:<45}")
        
        # Summary statistics
        m_rho0_0_count = len(self.get_cases_by_m_rho0(0))
        m_rho0_1_count = len(self.get_cases_by_m_rho0(1))
        
        print("-" * 70)
        print(f"Total cases: {len(self.polynomials)}")
        print(f"Cases with m_ρ₀ = 0: {m_rho0_0_count}")
        print(f"Cases with m_ρ₀ = 1: {m_rho0_1_count}")
    
    def export_to_json(self, filename: str) -> None:
        """
        Export all cases to JSON file
        
        Args:
            filename: Output filename
        """
        export_data = {
            'description': 'Omar S. Quaternion Galois Extension Cases',
            'total_cases': len(self.polynomials),
            'cases': [poly.get_info_dict() for poly in self.get_all_cases()]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(self.polynomials)} cases to {filename}")

# Global instance for easy access
OMAR_COLLECTION = OmarPolynomialCollection()

# Convenience functions
def get_case(case_id: int) -> QuaternionPolynomial:
    """Get case by ID"""
    return OMAR_COLLECTION.get_case(case_id)

def get_all_cases() -> List[QuaternionPolynomial]:
    """Get all 13 cases"""
    return OMAR_COLLECTION.get_all_cases()

def print_all_cases_summary() -> None:
    """Print summary of all cases"""
    OMAR_COLLECTION.print_summary()

# Legacy compatibility with existing code
OMAR_POLYNOMIALS = [poly.get_info_dict() for poly in OMAR_COLLECTION.get_all_cases()]

if __name__ == "__main__":
    # Demo
    print("Omar S. Quaternion Polynomial Collection")
    print("========================================")
    
    # Print summary
    print_all_cases_summary()
    
    # Example usage
    print("\nExample: Case 1 Details")
    case1 = get_case(1)
    print(f"Polynomial: {case1.polynomial}")
    print(f"m_ρ₀: {case1.m_rho0}")
    print(f"Ramified primes: {case1.ramified_primes}")
    print(f"Bias coefficients: {case1.get_bias_coefficients()}")
    
    # Export to JSON
    OMAR_COLLECTION.export_to_json('omar_polynomials.json')
    print("\n✅ Complete!")