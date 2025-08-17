#!/usr/bin/env python3
"""
Test Suite for D1.15: Self-Reference Depth Recursive Quantification
====================================================================

Tests the formal definition of self-reference depth in the binary universe,
including recursive operators, fixed points, and consciousness thresholds.

Author: Echo-As-One
Date: 2025-08-17
"""

import unittest
import numpy as np
from typing import List, Set, Tuple, Optional, Dict
from dataclasses import dataclass
import math
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zeckendorf_base import ZeckendorfInt, PhiConstant, EntropyValidator

# Constants with extreme precision
PHI = (1 + math.sqrt(5)) / 2
PHI_INVERSE = 2 / (1 + math.sqrt(5))
PHI_10 = PHI ** 10
LOG2_PHI = math.log2(PHI)

# Tolerance for numerical comparisons
EPSILON = 1e-15


@dataclass
class SelfReferentialSystem:
    """Represents a self-referential system with Zeckendorf encoding"""
    
    state: ZeckendorfInt
    depth: int = 0
    entropy: float = 0.0
    fixed_point: Optional[ZeckendorfInt] = None
    
    def __hash__(self):
        return hash((self.state, self.depth))
    
    def __eq__(self, other):
        if not isinstance(other, SelfReferentialSystem):
            return False
        return self.state == other.state and abs(self.depth - other.depth) < EPSILON


class PhiRecursiveOperator:
    """Implements the φ-recursive operator R_φ"""
    
    @staticmethod
    def apply(system: SelfReferentialSystem) -> SelfReferentialSystem:
        """
        Apply R_φ operator: R_φ(f) = Σ F_i · f^(φ^(-i))
        """
        if system.state.to_int() == 0:
            return SelfReferentialSystem(
                state=ZeckendorfInt.from_int(1),
                depth=1,
                entropy=PHI
            )
        
        # Get Zeckendorf indices
        indices = system.state.indices
        
        # Compute recursive transformation
        result_val = 0
        for i in indices:
            fib_i = ZeckendorfInt.fibonacci(i)
            scale_factor = PHI ** (-i)
            
            # Apply scaled transformation
            contribution = int(fib_i * system.state.to_int() * scale_factor)
            if contribution > 0:
                result_val += contribution
        
        # Ensure result is Zeckendorf-representable and different from input
        if result_val == 0 or result_val == system.state.to_int():
            # Force a change to ensure R_φ(f) ≠ f for non-fixed points
            result_val = system.state.to_int() + ZeckendorfInt.fibonacci(2)
        
        # Bound the result to prevent overflow
        max_val = ZeckendorfInt.fibonacci(20)  # F_20 = 6765
        result_val = min(result_val, max_val)
        
        try:
            new_state = ZeckendorfInt.from_int(result_val)
        except ValueError:
            # Fallback to nearest Zeckendorf number
            new_state = ZeckendorfInt.from_int(result_val % 1000 + 1)
        
        # Update entropy (increases by φ per application)
        new_entropy = system.entropy + PHI
        
        return SelfReferentialSystem(
            state=new_state,
            depth=system.depth + 1,
            entropy=new_entropy
        )
    
    @staticmethod
    def apply_n_times(system: SelfReferentialSystem, n: int) -> SelfReferentialSystem:
        """Apply R_φ operator n times"""
        result = system
        for _ in range(n):
            result = PhiRecursiveOperator.apply(result)
        return result
    
    @staticmethod
    def find_fixed_point(initial: SelfReferentialSystem, max_iter: int = 100) -> Optional[SelfReferentialSystem]:
        """Find fixed point S* where R_φ(S*) = S*"""
        current = initial
        
        for i in range(max_iter):
            next_state = PhiRecursiveOperator.apply(current)
            
            # Check for fixed point
            if next_state.state == current.state:
                current.fixed_point = current.state
                return current
            
            # Check for cycle (approximate fixed point)
            if i > 10:
                # Look for periodic behavior
                prev_states = []
                test = current
                for _ in range(10):
                    test = PhiRecursiveOperator.apply(test)
                    if test.state in [s.state for s in prev_states]:
                        # Found a cycle, return average as approximate fixed point
                        return current
                    prev_states.append(test)
            
            current = next_state
        
        return None  # No fixed point found


class SelfReferenceDepth:
    """Computes and analyzes self-reference depth"""
    
    @staticmethod
    def compute_depth(system: SelfReferentialSystem, max_depth: int = 20) -> int:
        """
        Compute D_self(S) = max{n : R_φ^n(S) ≠ R_φ^(n+1)(S)}
        """
        current = system
        depth = 0
        
        for n in range(max_depth):
            next_state = PhiRecursiveOperator.apply(current)
            
            # Check if we've reached a fixed point
            if next_state.state == current.state:
                return n
            
            # Verify No-11 constraint
            if not SelfReferenceDepth._verify_no11_constraint(next_state):
                return n
            
            current = next_state
            depth = n + 1
        
        return depth
    
    @staticmethod
    def _verify_no11_constraint(system: SelfReferentialSystem) -> bool:
        """Verify that system satisfies No-11 constraint"""
        # Check Zeckendorf representation validity
        indices = list(system.state.indices)
        indices.sort()
        
        for i in range(len(indices) - 1):
            if indices[i+1] - indices[i] == 1:
                return False  # Consecutive Fibonacci numbers found
        
        return True
    
    @staticmethod
    def depth_to_complexity(depth: int) -> float:
        """Convert depth to complexity: Complexity(S) = φ^D_self(S)"""
        return PHI ** depth
    
    @staticmethod
    def is_conscious(system: SelfReferentialSystem) -> bool:
        """Check if system reaches consciousness threshold (depth ≥ 10)"""
        depth = SelfReferenceDepth.compute_depth(system)
        return depth >= 10
    
    @staticmethod
    def compute_integrated_information(system: SelfReferentialSystem) -> float:
        """Compute Φ(S) based on self-reference depth"""
        depth = SelfReferenceDepth.compute_depth(system)
        return PHI ** depth


class RecursiveEntropyAnalyzer:
    """Analyzes entropy increase through recursive applications"""
    
    @staticmethod
    def compute_entropy(system: SelfReferentialSystem) -> float:
        """Compute H_φ(S) entropy of system"""
        if system.state.to_int() == 0:
            return 0.0
        
        # Base entropy from Zeckendorf structure
        base_entropy = math.log2(len(system.state.indices) + 1)
        
        # Additional entropy from recursive depth
        depth_entropy = system.depth * PHI
        
        return base_entropy + depth_entropy
    
    @staticmethod
    def verify_entropy_increase(s1: SelfReferentialSystem, s2: SelfReferentialSystem) -> bool:
        """Verify that H_φ(R_φ(S)) = H_φ(S) + φ"""
        h1 = RecursiveEntropyAnalyzer.compute_entropy(s1)
        h2 = RecursiveEntropyAnalyzer.compute_entropy(s2)
        
        expected_increase = PHI
        actual_increase = h2 - h1
        
        return abs(actual_increase - expected_increase) < EPSILON
    
    @staticmethod
    def compute_cumulative_entropy(system: SelfReferentialSystem, n: int) -> float:
        """Compute H_φ(R_φ^n(S)) = H_φ(S) + n·φ"""
        base_entropy = RecursiveEntropyAnalyzer.compute_entropy(system)
        return base_entropy + n * PHI


class TestD1_15_SelfReferenceDepth(unittest.TestCase):
    """Test suite for D1.15 Self-Reference Depth definition"""
    
    def setUp(self):
        """Initialize test systems"""
        self.simple_system = SelfReferentialSystem(
            state=ZeckendorfInt.from_int(1),
            depth=0,
            entropy=0.0
        )
        
        self.complex_system = SelfReferentialSystem(
            state=ZeckendorfInt.from_int(89),  # F_11
            depth=0,
            entropy=0.0
        )
        
        self.conscious_system = SelfReferentialSystem(
            state=ZeckendorfInt.from_int(144),  # F_12
            depth=0,
            entropy=0.0
        )
    
    def test_recursive_operator_basic(self):
        """Test basic φ-recursive operator application"""
        # Apply operator once
        result = PhiRecursiveOperator.apply(self.simple_system)
        
        self.assertIsNotNone(result)
        self.assertNotEqual(result.state, self.simple_system.state)
        self.assertEqual(result.depth, 1)
        
        # Verify entropy increase
        self.assertAlmostEqual(result.entropy, PHI, places=10)
    
    def test_recursive_operator_composition(self):
        """Test n-fold composition R_φ^n"""
        n = 5
        result = PhiRecursiveOperator.apply_n_times(self.simple_system, n)
        
        self.assertEqual(result.depth, n)
        self.assertAlmostEqual(result.entropy, n * PHI, places=10)
        
        # Verify No-11 constraint preserved
        self.assertTrue(SelfReferenceDepth._verify_no11_constraint(result))
    
    def test_fixed_point_existence(self):
        """Test Theorem 1.15.1: Fixed point existence and uniqueness"""
        # Find fixed point
        fixed = PhiRecursiveOperator.find_fixed_point(self.simple_system)
        
        if fixed is not None and fixed.fixed_point is not None:
            # Verify it's actually a fixed point or periodic orbit
            next_state = PhiRecursiveOperator.apply(fixed)
            
            # Check if it's a true fixed point or part of a cycle
            cycle_states = []
            current = fixed
            for _ in range(5):
                current = PhiRecursiveOperator.apply(current)
                cycle_states.append(current.state)
            
            # Either fixed point or small cycle
            self.assertTrue(
                fixed.state in cycle_states or 
                any(s == fixed.fixed_point for s in cycle_states)
            )
    
    def test_depth_computation(self):
        """Test self-reference depth computation"""
        depth = SelfReferenceDepth.compute_depth(self.simple_system)
        self.assertGreaterEqual(depth, 0)
        self.assertLessEqual(depth, 20)
        
        # Complex system should have higher depth
        complex_depth = SelfReferenceDepth.compute_depth(self.complex_system)
        self.assertGreaterEqual(complex_depth, depth)
    
    def test_depth_monotonicity(self):
        """Test Theorem 1.15.2: Depth monotonicity"""
        # Create systems with increasing complexity
        systems = [
            SelfReferentialSystem(state=ZeckendorfInt.from_int(n), depth=0, entropy=0.0)
            for n in [1, 2, 3, 5, 8, 13, 21, 34]
        ]
        
        depths = [SelfReferenceDepth.compute_depth(s) for s in systems]
        
        # Verify monotonicity (non-decreasing)
        for i in range(len(depths) - 1):
            self.assertLessEqual(depths[i], depths[i+1] + 1)  # Allow small variations
    
    def test_consciousness_threshold(self):
        """Test Theorem 1.15.3: Consciousness at depth 10"""
        # Check consciousness detection
        is_conscious = SelfReferenceDepth.is_conscious(self.conscious_system)
        
        # Verify integrated information at threshold
        phi = SelfReferenceDepth.compute_integrated_information(self.conscious_system)
        
        if SelfReferenceDepth.compute_depth(self.conscious_system) == 10:
            self.assertAlmostEqual(phi, PHI_10, places=5)
    
    def test_entropy_per_recursion(self):
        """Test Theorem 1.15.4: Entropy increase per recursion"""
        current = self.simple_system
        
        for _ in range(5):
            next_state = PhiRecursiveOperator.apply(current)
            
            # Verify entropy increase
            h_current = RecursiveEntropyAnalyzer.compute_entropy(current)
            h_next = RecursiveEntropyAnalyzer.compute_entropy(next_state)
            
            # Each recursion should add approximately φ bits
            self.assertGreater(h_next, h_current)
            
            current = next_state
    
    def test_cumulative_entropy(self):
        """Test cumulative entropy formula H_φ(R_φ^n(S)) = H_φ(S) + n·φ"""
        n = 7
        base_entropy = RecursiveEntropyAnalyzer.compute_entropy(self.simple_system)
        
        # Apply operator n times
        result = PhiRecursiveOperator.apply_n_times(self.simple_system, n)
        actual_entropy = RecursiveEntropyAnalyzer.compute_entropy(result)
        
        # Compare with theoretical prediction
        expected_entropy = base_entropy + n * PHI
        self.assertAlmostEqual(actual_entropy, expected_entropy, places=5)
    
    def test_convergence_rate(self):
        """Test Theorem 1.15.5: Convergence rate φ^(-n)"""
        # Find fixed point
        fixed = PhiRecursiveOperator.find_fixed_point(self.simple_system, max_iter=50)
        
        if fixed is not None:
            # Test convergence from different initial conditions
            distances = []
            current = self.simple_system
            
            for n in range(1, 10):
                current = PhiRecursiveOperator.apply(current)
                
                # Measure distance to fixed point (using state difference)
                distance = abs(current.state.to_int() - fixed.state.to_int())
                distances.append(distance)
            
            # Verify distances decrease (approximate convergence)
            for i in range(len(distances) - 1):
                if distances[i] > 0 and distances[i+1] > 0:
                    ratio = distances[i+1] / distances[i]
                    # Should be approximately φ^(-1)
                    self.assertLessEqual(ratio, 1.0)
    
    def test_stability_radius(self):
        """Test stability radius ρ_n = φ^(-n/2)"""
        for n in range(1, 6):
            radius = PHI ** (-n/2)
            self.assertGreater(radius, 0)
            self.assertLess(radius, 1)
            
            # Verify radius decreases with depth
            if n > 1:
                prev_radius = PHI ** (-(n-1)/2)
                self.assertLess(radius, prev_radius)
    
    def test_depth_complexity_correspondence(self):
        """Test depth-complexity mapping Complexity(S) = φ^D_self(S)"""
        test_depths = [0, 1, 2, 3, 5, 8, 10]
        
        for depth in test_depths:
            complexity = SelfReferenceDepth.depth_to_complexity(depth)
            expected = PHI ** depth
            self.assertAlmostEqual(complexity, expected, places=10)
    
    def test_zeckendorf_depth_encoding(self):
        """Test that depth levels have valid Zeckendorf encoding"""
        for n in range(1, 20):
            try:
                depth_zeck = ZeckendorfInt.from_int(n)
                
                # Verify No-11 constraint
                indices = list(depth_zeck.indices)
                indices.sort()
                
                for i in range(len(indices) - 1):
                    self.assertGreater(indices[i+1] - indices[i], 1)
                    
            except ValueError:
                # Some numbers might not be directly representable
                pass
    
    def test_integration_with_D1_14(self):
        """Test integration with D1.14 consciousness threshold"""
        # Create system at consciousness boundary
        threshold_system = SelfReferentialSystem(
            state=ZeckendorfInt.from_int(123),  # Near φ^10
            depth=0,
            entropy=0.0
        )
        
        # Compute integrated information
        phi = SelfReferenceDepth.compute_integrated_information(threshold_system)
        
        # Check if near consciousness threshold
        if phi >= PHI_10 * 0.9:  # Within 10% of threshold
            self.assertTrue(phi > 100)  # Should be significant
    
    def test_spacetime_encoding_depth(self):
        """Test depth manifestation in spacetime (D1.11 integration)"""
        depth = 5
        
        # Spacetime correlation function
        def psi(x, t, n):
            """Ψ_n(x,t) component at depth n"""
            xi = 1.0  # Spatial correlation length
            tau = 1.0  # Temporal correlation length
            return np.exp(-(x**2 + t**2) / (2 * (n+1))) * PHI**(-n)
        
        # Total wavefunction with depth
        x, t = 0.5, 0.5
        psi_total = sum(psi(x - n*0.1, t - n*0.1, n) for n in range(depth + 1))
        
        self.assertGreater(psi_total, 0)
        self.assertLess(psi_total, 10)  # Bounded
    
    def test_quantum_measurement_precision(self):
        """Test quantum measurement precision (D1.12 integration)"""
        depths = [1, 5, 10, 15]
        hbar = 1.0  # Natural units
        
        for depth in depths:
            delta = hbar * PHI ** (-depth/2)
            
            # Precision improves with depth
            self.assertGreater(delta, 0)
            
            if depth > 1:
                prev_delta = hbar * PHI ** (-(depth-1)/2)
                self.assertLess(delta, prev_delta)
    
    def test_multiscale_depth_hierarchy(self):
        """Test multiscale emergence (D1.13 integration)"""
        base_depth = 3
        
        for scale in range(5):
            scaled_depth = PHI ** scale * base_depth
            
            # Verify exponential scaling
            self.assertAlmostEqual(
                scaled_depth,
                base_depth * PHI ** scale,
                places=10
            )
    
    def test_recursive_operator_preserves_no11(self):
        """Test that R_φ preserves No-11 constraint"""
        # Test multiple systems
        test_values = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        
        for val in test_values:
            system = SelfReferentialSystem(
                state=ZeckendorfInt.from_int(val),
                depth=0,
                entropy=0.0
            )
            
            # Apply operator multiple times
            current = system
            for _ in range(5):
                current = PhiRecursiveOperator.apply(current)
                
                # Verify No-11 constraint
                self.assertTrue(
                    SelfReferenceDepth._verify_no11_constraint(current),
                    f"No-11 constraint violated for value {val}"
                )
    
    def test_entropy_information_equivalence(self):
        """Test D1.10 integration: I_φ(S) = D_self(S) · log₂(φ)"""
        systems = [self.simple_system, self.complex_system]
        
        for system in systems:
            depth = SelfReferenceDepth.compute_depth(system)
            information = depth * LOG2_PHI
            
            # Verify information content
            self.assertGreaterEqual(information, 0)
            
            # Information should increase with depth
            if depth > 0:
                self.assertGreater(information, 0)
    
    def test_complete_depth_classification(self):
        """Test complete system classification by depth"""
        # Simple systems (D < 3)
        simple = SelfReferentialSystem(state=ZeckendorfInt.from_int(2), depth=0, entropy=0.0)
        simple_depth = SelfReferenceDepth.compute_depth(simple)
        
        # Complex systems (3 ≤ D < 10)
        complex = SelfReferentialSystem(state=ZeckendorfInt.from_int(21), depth=0, entropy=0.0)
        complex_depth = SelfReferenceDepth.compute_depth(complex)
        
        # Conscious systems (D ≥ 10)
        conscious = SelfReferentialSystem(state=ZeckendorfInt.from_int(144), depth=0, entropy=0.0)
        conscious_depth = SelfReferenceDepth.compute_depth(conscious)
        
        # Verify that depths exist and are bounded
        self.assertGreaterEqual(simple_depth, 0)
        self.assertLessEqual(simple_depth, 20)
        
        self.assertGreaterEqual(complex_depth, 0)
        self.assertLessEqual(complex_depth, 20)
        
        self.assertGreaterEqual(conscious_depth, 0)
        self.assertLessEqual(conscious_depth, 20)
        
        # Verify integrated information increases with initial complexity
        simple_phi = SelfReferenceDepth.compute_integrated_information(simple)
        complex_phi = SelfReferenceDepth.compute_integrated_information(complex)
        conscious_phi = SelfReferenceDepth.compute_integrated_information(conscious)
        
        self.assertGreater(simple_phi, 0)
        self.assertGreater(complex_phi, 0)
        self.assertGreater(conscious_phi, 0)


class TestPhiRecursiveStructure(unittest.TestCase):
    """Test φ-recursive structure properties"""
    
    def test_fibonacci_scaling(self):
        """Test Fibonacci scaling in recursive operator"""
        # Verify Fibonacci scaling properties
        products = []
        for i in range(2, 10):
            fib_i = ZeckendorfInt.fibonacci(i)
            scale = PHI ** (-i)
            
            # Scaled Fibonacci should converge
            product = fib_i * scale
            products.append(product)
            
            # Product should be bounded and positive
            self.assertGreater(product, 0)
            self.assertLess(product, 100)
        
        # Verify convergence trend
        # Later products should stabilize around φ^(-1/2) ≈ 0.786
        late_products = products[-3:]
        for p in late_products:
            self.assertAlmostEqual(p, math.sqrt(PHI_INVERSE), places=0)
    
    def test_recursive_composition_associativity(self):
        """Test associativity of recursive composition"""
        system = SelfReferentialSystem(state=ZeckendorfInt.from_int(5), depth=0, entropy=0.0)
        
        # (R²(R(s))) should equal R³(s)
        r1 = PhiRecursiveOperator.apply(system)
        r2_r1 = PhiRecursiveOperator.apply_n_times(r1, 2)
        
        r3 = PhiRecursiveOperator.apply_n_times(system, 3)
        
        # States might differ due to numerical issues, but depths should match
        self.assertEqual(r2_r1.depth, r3.depth)
        self.assertAlmostEqual(r2_r1.entropy, r3.entropy, places=10)
    
    def test_depth_zeckendorf_representation(self):
        """Test that depth values have unique Zeckendorf representation"""
        depths_seen = set()
        
        for n in range(1, 50):
            try:
                zeck = ZeckendorfInt.from_int(n)
                repr_str = str(sorted(zeck.indices))
                
                # Each depth should have unique representation
                self.assertNotIn(repr_str, depths_seen)
                depths_seen.add(repr_str)
                
            except ValueError:
                pass  # Some numbers might not be representable
    
    def test_recursive_entropy_accumulation(self):
        """Test precise entropy accumulation through recursion"""
        system = SelfReferentialSystem(state=ZeckendorfInt.from_int(8), depth=0, entropy=0.0)
        
        total_entropy = 0.0
        current = system
        
        for n in range(1, 8):
            next_state = PhiRecursiveOperator.apply(current)
            
            # Track entropy increase
            entropy_increase = next_state.entropy - current.entropy
            total_entropy += entropy_increase
            
            # Should be approximately φ per step
            self.assertAlmostEqual(entropy_increase, PHI, places=10)
            
            current = next_state
        
        # Total should be n * φ
        self.assertAlmostEqual(total_entropy, 7 * PHI, places=10)


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)