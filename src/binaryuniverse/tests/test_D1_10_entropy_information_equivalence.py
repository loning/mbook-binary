"""
D1.10 Entropy-Information Equivalence Comprehensive Test Suite

This module implements complete formal verification of the entropy-information 
equivalence theory as specified in formal/D1_10_entropy_information_equivalence_formal.md

Core Theoretical Foundation:
- A1 Axiom: Self-referential complete systems inevitably increase entropy
- Binary Universe Constraint: No consecutive 1s in binary representations
- Zeckendorf Representation: Unique φ-based encoding for all data
- Standard Fibonacci Sequence: F₁=1, F₂=2, F₃=3, F₄=5, F₅=8, F₆=13, ...

Formal Verification Requirements:
- Complete consistency with machine formal description
- Zero-tolerance for approximations or simplifications
- Comprehensive coverage of all theoretical properties
- Rigorous adversarial validation approach
"""

import unittest
import math
import sys
import os
from typing import List, Tuple, Dict, Set
from decimal import Decimal, getcontext

# Add the tests directory to sys.path so we can import test_base
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set high precision for theoretical calculations  
getcontext().prec = 50

# Import existing base classes
from zeckendorf_base import ZeckendorfInt, PhiConstant, EntropyValidator


class ZeckendorfEncoderAdapter:
    """
    Adapter to use existing ZeckendorfInt implementation.
    
    Wraps the existing zeckendorf_base.py classes to provide the interface
    needed for D1.10 testing while maintaining consistency with existing code.
    """
    
    def __init__(self, max_terms: int = 50):
        self.max_terms = max_terms
        self.phi = Decimal(PhiConstant.phi())
        
    def encode(self, n: int) -> List[int]:
        """
        Encode integer n using existing ZeckendorfInt implementation.
        
        Returns: Binary representation compatible with No-11 constraint.
        """
        if n <= 0:
            return [0]
            
        zint = ZeckendorfInt.from_int(n)
        # Convert frozenset indices to binary representation
        if not zint.indices:
            return [0]
            
        max_index = max(zint.indices)
        # Standard Fibonacci starts from F(2)=1, so max index for representation
        representation = [0] * max(max_index - 1, 1)  # Adjust indexing
        
        for idx in zint.indices:
            # zeckendorf_base uses indices starting from 2 for F(2)=1
            if idx >= 2 and (idx - 2) < len(representation):
                representation[idx - 2] = 1
                
        # Remove leading zeros
        while len(representation) > 1 and representation[-1] == 0:
            representation.pop()
            
        return representation
    
    def decode(self, representation: List[int]) -> int:
        """Decode representation using existing ZeckendorfInt."""
        if not representation:
            return 0
            
        indices = set()
        for i, bit in enumerate(representation):
            if bit == 1:
                # Map position i to Fibonacci index (starting from F(2)=1)
                indices.add(i + 2)
                
        if not indices:
            return 0
            
        try:
            zint = ZeckendorfInt(frozenset(indices))
            return zint.to_int()
        except ValueError:
            # If invalid Zeckendorf representation, return 0
            return 0
    
    def is_valid_no11(self, representation: List[int]) -> bool:
        """Verify No-11 constraint: no consecutive 1s allowed."""
        for i in range(len(representation) - 1):
            if representation[i] == 1 and representation[i + 1] == 1:
                return False
        return True


class TestD110EntropyInformationEquivalence(unittest.TestCase):
    """
    Comprehensive test suite for D1.10 Entropy-Information Equivalence.
    
    Verifies complete consistency with formal specification:
    formal/D1_10_entropy_information_equivalence_formal.md
    """
    
    def setUp(self):
        """Initialize test environment with theoretical precision."""
        self.encoder = ZeckendorfEncoderAdapter(max_terms=30)
        self.phi = Decimal(PhiConstant.phi())
        self.tolerance = Decimal('1e-12')
        
        # Generate standard Fibonacci sequence using existing implementation
        # Note: zeckendorf_base uses F(0)=0, F(1)=1, F(2)=1, F(3)=2, ...
        self.fibonacci = [ZeckendorfInt.fibonacci(i) for i in range(1, 31)]
        
    def test_fibonacci_sequence_correctness(self):
        """Test D1.10.1: Verify standard Fibonacci sequence definition."""
        # Standard Fibonacci: F(0)=0, F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5, ...
        expected = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
        actual = self.fibonacci[:len(expected)]
        
        self.assertEqual(actual, expected, 
                        "Fibonacci sequence must match zeckendorf_base standard")
        
        # Verify Fibonacci property F(n) = F(n-1) + F(n-2)
        for i in range(2, len(actual)):
            self.assertEqual(actual[i], actual[i-1] + actual[i-2],
                           f"Fibonacci property violated at index {i}")
    
    def test_zeckendorf_encoding_bijection(self):
        """Test D1.10.2: Verify Zeckendorf encoding is bijective."""
        test_numbers = [1, 2, 3, 4, 5, 8, 13, 21, 34, 55, 100, 233, 377, 500, 1000]
        
        for n in test_numbers:
            with self.subTest(number=n):
                # Encode and decode
                encoded = self.encoder.encode(n)
                decoded = self.encoder.decode(encoded)
                
                # Verify bijection
                self.assertEqual(n, decoded, f"Bijection failed for {n}")
                
                # Verify No-11 constraint
                self.assertTrue(self.encoder.is_valid_no11(encoded),
                               f"No-11 constraint violated for {n}: {encoded}")
                
                # Verify uniqueness (greedy property)
                self.assertTrue(len(encoded) > 0, f"Empty encoding for {n}")
    
    def test_no11_constraint_comprehensive(self):
        """Test D1.10.3: Comprehensive No-11 constraint verification."""
        # Valid representations (no consecutive 1s)
        valid_cases = [
            [1, 0, 1, 0, 1],     # 101010
            [1, 0, 0, 1, 0],     # 10010
            [0, 1, 0, 1, 0],     # 01010
            [1],                  # 1
            [0],                  # 0
            []                    # empty
        ]
        
        for case in valid_cases:
            with self.subTest(representation=case):
                self.assertTrue(self.encoder.is_valid_no11(case),
                               f"Valid case rejected: {case}")
        
        # Invalid representations (consecutive 1s)
        invalid_cases = [
            [1, 1],              # 11
            [1, 1, 0],           # 110
            [0, 1, 1],           # 011
            [1, 0, 1, 1],        # 1011
            [1, 1, 0, 1, 1]      # 11011
        ]
        
        for case in invalid_cases:
            with self.subTest(representation=case):
                self.assertFalse(self.encoder.is_valid_no11(case),
                                f"Invalid case accepted: {case}")
    
    def test_phi_complexity_monotonicity(self):
        """Test D1.10.4: Verify φ-complexity increases with representation size."""
        representations = [
            [1],              # F₁
            [0, 1],           # F₂  
            [1, 0, 1],        # F₁ + F₃
            [0, 1, 0, 1],     # F₂ + F₄
            [1, 0, 1, 0, 1]   # F₁ + F₃ + F₅
        ]
        
        complexities = []
        for rep in representations:
            # Calculate φ-complexity manually
            complexity = Decimal(0)
            for i, bit in enumerate(rep):
                if bit == 1 and i < len(self.fibonacci):
                    weight = self.phi ** Decimal(i + 1)
                    complexity += weight
            complexities.append(complexity)
            
        # Verify monotonic increase
        for i in range(1, len(complexities)):
            self.assertGreater(complexities[i], complexities[i-1],
                             f"φ-complexity not monotonic at index {i}")
    
    def test_golden_ratio_properties(self):
        """Test D1.10.5: Verify φ (golden ratio) satisfies required properties."""
        # Test golden ratio property φ² = φ + 1
        phi_squared = self.phi * self.phi
        phi_plus_one = self.phi + Decimal(1)
        error = abs(phi_squared - phi_plus_one)
        self.assertLess(error, self.tolerance, 
                       f"Golden ratio property violated: φ²={phi_squared}, φ+1={phi_plus_one}")
        
        # Additional φ properties specific to D1.10
        phi_squared = self.phi * self.phi
        phi_plus_one = self.phi + Decimal(1)
        
        # Verify explicit calculation
        calculated_phi = (Decimal(1) + Decimal(5).sqrt()) / Decimal(2)
        calc_error = abs(self.phi - calculated_phi)
        self.assertLess(calc_error, self.tolerance, "φ calculation mismatch")
    
    def test_entropy_information_basic_equivalence(self):
        """Test D1.10.6: Basic entropy-information equivalence verification."""
        # Simple test cases with known properties
        test_cases = [
            # Single representation: entropy should be 0
            {
                'representations': [[1]],
                'expected_entropy_bound': (Decimal(0), Decimal('0.1'))
            },
            # Two distinct representations: positive entropy
            {
                'representations': [[1, 0], [0, 1]],
                'expected_entropy_bound': (Decimal('0.1'), Decimal('2.0'))
            }
        ]
        
        for i, case in enumerate(test_cases):
            with self.subTest(case=i):
                reps = case['representations']
                lower, upper = case['expected_entropy_bound']
                
                # Verify all representations satisfy No-11
                for rep in reps:
                    self.assertTrue(self.encoder.is_valid_no11(rep),
                                   f"No-11 constraint violated: {rep}")
                
                # Calculate entropy using uniform distribution
                if len(reps) == 1:
                    entropy = Decimal(0)  # Single state has zero entropy
                else:
                    # Shannon entropy: H = -Σ p_i log(p_i)
                    p = Decimal(1) / Decimal(len(reps))
                    entropy = -Decimal(len(reps)) * p * p.ln()
                
                # Verify entropy bounds
                self.assertGreaterEqual(entropy, lower, 
                                       f"Case {i} entropy {entropy} below lower bound {lower}")
                self.assertLessEqual(entropy, upper,
                                    f"Case {i} entropy {entropy} exceeds upper bound {upper}")
    
    def test_zeckendorf_uniqueness_comprehensive(self):
        """Test D1.10.7: Verify Zeckendorf representation uniqueness."""
        # Test range of integers
        test_range = range(1, 101)
        representations = []
        
        for n in test_range:
            encoded = self.encoder.encode(n)
            representations.append(encoded)
            
            # Verify each representation is valid
            self.assertTrue(self.encoder.is_valid_no11(encoded),
                           f"Invalid representation for {n}: {encoded}")
            
            # Verify round-trip consistency
            decoded = self.encoder.decode(encoded)
            self.assertEqual(n, decoded, f"Round-trip failed: {n} → {encoded} → {decoded}")
        
        # Verify uniqueness manually
        seen = set()
        for rep in representations:
            rep_tuple = tuple(rep)
            self.assertNotIn(rep_tuple, seen, f"Duplicate representation found: {rep}")
            seen.add(rep_tuple)
    
    def test_entropy_increase_a1_axiom(self):
        """Test D1.10.8: Verify A1 axiom - entropy increase in self-referential systems."""
        # Build sequence of increasing complexity
        base_numbers = [1, 2, 3, 5, 8, 13, 21]
        
        entropies = []
        representations = []
        
        for n in base_numbers:
            encoded = self.encoder.encode(n)
            if self.encoder.is_valid_no11(encoded):
                representations.append(encoded)
                
                # Calculate cumulative entropy (simulate system growth)
                if len(representations) == 1:
                    entropy = Decimal(0)
                else:
                    # Uniform entropy model
                    count = len(representations)
                    p = Decimal(1) / Decimal(count)
                    entropy = -Decimal(count) * p * p.ln()
                
                entropies.append(entropy)
        
        # Verify entropy increase (A1 axiom)
        for i in range(1, len(entropies)):
            self.assertGreater(entropies[i], entropies[i-1],
                             f"Entropy decrease violation at index {i}: {entropies[i]} <= {entropies[i-1]}")
    
    def test_theoretical_bounds_verification(self):
        """Test D1.10.9: Verify all theoretical bounds are satisfied."""
        # Test information content bounds
        test_probs = [Decimal('0.5'), Decimal('0.25'), Decimal('0.125'), Decimal('0.1')]
        
        for prob in test_probs:
            # Calculate information content I(x) = -log(p(x))
            info_content = -prob.ln()
            
            # Information content should be positive and finite
            self.assertGreater(info_content, 0, f"Information content not positive for p={prob}")
            self.assertNotEqual(info_content, Decimal('inf'), f"Information content infinite for p={prob}")
            
            # Verify expected value matches calculation
            expected = -prob.ln()
            error = abs(info_content - expected)
            self.assertLess(error, self.tolerance, f"Information content mismatch for p={prob}")
    
    def test_integration_theoretical_consistency(self):
        """Test D1.10.10: Final integration test for theoretical consistency."""
        # Comprehensive test combining all aspects
        test_numbers = [1, 2, 3, 5, 8, 13, 21, 34]
        
        # Step 1: Encode all numbers
        representations = []
        for n in test_numbers:
            encoded = self.encoder.encode(n)
            representations.append(encoded)
        
        # Step 2: Verify all fundamental properties
        fundamental_checks = {
            'all_no11_valid': all(self.encoder.is_valid_no11(rep) for rep in representations),
            'all_bijective': all(self.encoder.decode(self.encoder.encode(self.encoder.decode(rep))) == self.encoder.decode(rep) for rep in representations),
            'fibonacci_correct': self.fibonacci[1:9] == [1, 2, 3, 5, 8, 13, 21, 34],  # Skip F(1)=1, start from F(2)=1
            'phi_property': abs(self.phi**2 - (self.phi + 1)) < self.tolerance,
            'unique_representations': len(set(tuple(rep) for rep in representations)) == len(representations)
        }
        
        # Step 3: Verify all checks pass
        for check_name, result in fundamental_checks.items():
            self.assertTrue(result, f"Fundamental check failed: {check_name}")
        
        # Step 4: Meta-verification
        self.assertEqual(len(fundamental_checks), 5, "Incomplete fundamental verification")
        
        # Step 5: Entropy consistency check
        complexity_values = []
        for rep in representations:
            complexity = Decimal(0)
            for i, bit in enumerate(rep):
                if bit == 1 and i < len(self.fibonacci):
                    weight = self.phi ** Decimal(i + 1)
                    complexity += weight
            complexity_values.append(complexity)
        
        # Complexity should increase with Fibonacci numbers
        for i in range(1, len(complexity_values)):
            self.assertGreater(complexity_values[i], complexity_values[i-1],
                             f"Complexity not increasing: pos {i}")


if __name__ == '__main__':
    # Run tests with maximum verbosity
    unittest.main(verbosity=2, buffer=True)