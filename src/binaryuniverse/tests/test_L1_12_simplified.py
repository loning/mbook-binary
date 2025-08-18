#!/usr/bin/env python3
"""
L1.12 Information Integration Complexity Threshold Lemma - Simplified Test Suite
================================================================================

Tests the essential functionality of information integration complexity thresholds
in the Binary Universe framework.

Author: Echo-As-One
Date: 2025-08-17
"""

import unittest
import math
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zeckendorf_base import ZeckendorfInt, PhiConstant

# Constants
PHI = PhiConstant.phi()
PHI_5 = PHI ** 5    # ≈ 11.09 - Partial integration threshold
PHI_10 = PHI ** 10  # ≈ 122.97 - Full integration threshold
EPSILON = 1e-10


class IntegrationPhase:
    """Integration phase enumeration"""
    SEGREGATED = "Segregated"    # I_φ < φ^5
    PARTIAL = "Partial"          # φ^5 ≤ I_φ < φ^10
    INTEGRATED = "Integrated"    # I_φ ≥ φ^10


class IntegrationComplexityCalculator:
    """Calculate information integration complexity"""
    
    @staticmethod
    def compute_integration_complexity(system_size: int, coupling: float = 0.5) -> float:
        """
        Compute I_φ(S) for a system
        
        Args:
            system_size: Number of components
            coupling: Coupling strength (0-1)
            
        Returns:
            Integration complexity in φ-bits
        """
        # Base complexity from system size (Fibonacci scaling)
        base_complexity = 0
        for i in range(2, min(system_size + 2, 15)):
            base_complexity += ZeckendorfInt.fibonacci(i) * coupling
        
        # Scale by golden ratio structure
        phi_complexity = base_complexity * (PHI ** math.log2(system_size + 1))
        
        return phi_complexity
    
    @staticmethod
    def determine_phase(integration_complexity: float) -> str:
        """Determine integration phase based on complexity"""
        if integration_complexity < PHI_5:
            return IntegrationPhase.SEGREGATED
        elif integration_complexity < PHI_10:
            return IntegrationPhase.PARTIAL
        else:
            return IntegrationPhase.INTEGRATED
    
    @staticmethod
    def compute_entropy_rate(phase: str) -> float:
        """Compute entropy rate for each phase"""
        rates = {
            IntegrationPhase.SEGREGATED: 1.0 / PHI,  # φ^(-1) ≈ 0.618
            IntegrationPhase.PARTIAL: 1.0,           # Unity
            IntegrationPhase.INTEGRATED: PHI         # φ ≈ 1.618
        }
        return rates.get(phase, 1.0)


class TestL1_12_Simplified(unittest.TestCase):
    """Simplified test suite for L1.12"""
    
    def setUp(self):
        """Initialize test systems"""
        self.calc = IntegrationComplexityCalculator()
        
        # Test systems
        self.small_system = {"size": 2, "coupling": 0.1}
        self.medium_system = {"size": 5, "coupling": 0.5}
        self.large_system = {"size": 10, "coupling": 0.9}
    
    def test_integration_complexity_calculation(self):
        """Test basic integration complexity calculation"""
        # Small system should have low complexity
        i_small = self.calc.compute_integration_complexity(
            self.small_system["size"], 
            self.small_system["coupling"]
        )
        self.assertGreater(i_small, 0, "Integration complexity should be positive")
        self.assertLess(i_small, PHI_5, "Small system should be below partial threshold")
        
        # Large system should have high complexity
        i_large = self.calc.compute_integration_complexity(
            self.large_system["size"], 
            self.large_system["coupling"]
        )
        self.assertGreater(i_large, i_small, "Large system should have higher complexity")
    
    def test_phase_determination(self):
        """Test phase determination from integration complexity"""
        # Test phase boundaries
        self.assertEqual(
            self.calc.determine_phase(PHI_5 - 0.1), 
            IntegrationPhase.SEGREGATED
        )
        
        self.assertEqual(
            self.calc.determine_phase(PHI_5 + 0.1), 
            IntegrationPhase.PARTIAL
        )
        
        self.assertEqual(
            self.calc.determine_phase(PHI_10 + 0.1), 
            IntegrationPhase.INTEGRATED
        )
    
    def test_entropy_rates(self):
        """Test entropy rates for different phases"""
        # Segregated phase
        rate_seg = self.calc.compute_entropy_rate(IntegrationPhase.SEGREGATED)
        self.assertAlmostEqual(rate_seg, 1.0/PHI, places=5, 
                             msg="Segregated entropy rate should be φ^(-1)")
        
        # Partial phase
        rate_partial = self.calc.compute_entropy_rate(IntegrationPhase.PARTIAL)
        self.assertAlmostEqual(rate_partial, 1.0, places=5,
                             msg="Partial entropy rate should be 1.0")
        
        # Integrated phase
        rate_int = self.calc.compute_entropy_rate(IntegrationPhase.INTEGRATED)
        self.assertAlmostEqual(rate_int, PHI, places=5,
                             msg="Integrated entropy rate should be φ")
    
    def test_threshold_values(self):
        """Test threshold values match theory"""
        self.assertAlmostEqual(PHI_5, 11.090169943749474, places=10,
                             msg="φ^5 threshold value incorrect")
        self.assertAlmostEqual(PHI_10, 122.99186938124426, places=10,
                             msg="φ^10 threshold value incorrect")
    
    def test_consciousness_threshold_integration(self):
        """Test integration with D1.14 consciousness threshold"""
        # System at consciousness threshold should be in integrated phase
        consciousness_complexity = PHI_10
        phase = self.calc.determine_phase(consciousness_complexity)
        self.assertEqual(phase, IntegrationPhase.INTEGRATED,
                        "Consciousness threshold should trigger integrated phase")
    
    def test_fibonacci_scaling(self):
        """Test that complexity scales with Fibonacci structure"""
        # Test systems with Fibonacci-based sizes
        fib_sizes = [1, 2, 3, 5, 8]
        complexities = []
        
        for size in fib_sizes:
            complexity = self.calc.compute_integration_complexity(size, 0.5)
            complexities.append(complexity)
        
        # Should show increasing complexity
        for i in range(1, len(complexities)):
            self.assertGreater(complexities[i], complexities[i-1],
                             f"Complexity should increase with Fibonacci size")
    
    def test_coupling_effect(self):
        """Test effect of coupling strength on integration complexity"""
        system_size = 5
        
        # Low coupling
        i_low = self.calc.compute_integration_complexity(system_size, 0.1)
        
        # High coupling
        i_high = self.calc.compute_integration_complexity(system_size, 0.9)
        
        self.assertGreater(i_high, i_low, 
                          "Higher coupling should increase integration complexity")
    
    def test_phase_transition_discreteness(self):
        """Test that phase transitions are discrete"""
        # Test points around thresholds
        test_points = [
            PHI_5 - 0.001,  # Just below partial threshold
            PHI_5 + 0.001,  # Just above partial threshold
            PHI_10 - 0.001, # Just below integrated threshold
            PHI_10 + 0.001  # Just above integrated threshold
        ]
        
        expected_phases = [
            IntegrationPhase.SEGREGATED,
            IntegrationPhase.PARTIAL,
            IntegrationPhase.PARTIAL,
            IntegrationPhase.INTEGRATED
        ]
        
        for point, expected in zip(test_points, expected_phases):
            phase = self.calc.determine_phase(point)
            self.assertEqual(phase, expected,
                           f"Phase transition should be discrete at {point}")
    
    def test_zeckendorf_structure_preservation(self):
        """Test that calculations preserve Zeckendorf structure"""
        # Test with Fibonacci numbers
        for fib_index in range(2, 10):
            fib_num = ZeckendorfInt.fibonacci(fib_index)
            
            # Should be able to compute complexity without errors
            try:
                complexity = self.calc.compute_integration_complexity(fib_num, 0.5)
                self.assertGreater(complexity, 0, 
                                 f"Should compute valid complexity for F_{fib_index}")
            except Exception as e:
                self.fail(f"Fibonacci number F_{fib_index} caused error: {e}")
    
    def test_integration_with_multiscale_cascade(self):
        """Test integration with L1.10 multiscale cascade"""
        # Different scale levels should have different integration complexities
        scale_levels = [0, 1, 2, 3, 4]
        
        for level in scale_levels:
            # Simulate cascade effect on integration
            base_complexity = 5.0  # Base complexity
            cascaded_complexity = base_complexity * (PHI ** level)
            
            phase = self.calc.determine_phase(cascaded_complexity)
            
            # Higher levels should tend toward higher phases
            if level == 0:
                self.assertEqual(phase, IntegrationPhase.SEGREGATED)
            elif level >= 3:
                # Should be at least partial integration
                self.assertIn(phase, [IntegrationPhase.PARTIAL, IntegrationPhase.INTEGRATED])
    
    def test_observer_hierarchy_connection(self):
        """Test connection with L1.11 observer hierarchy"""
        # Systems above consciousness threshold should support observer hierarchy
        observer_capable_complexity = PHI_10 * 1.5
        phase = self.calc.determine_phase(observer_capable_complexity)
        
        self.assertEqual(phase, IntegrationPhase.INTEGRATED,
                        "Observer-capable systems should be fully integrated")
        
        # Calculate implied observer depth (from L1.11: D_observer = D_self - 10)
        implied_depth = math.log(observer_capable_complexity / PHI_10) / math.log(PHI)
        self.assertGreater(implied_depth, 0,
                          "Should have positive observer hierarchy depth")


if __name__ == "__main__":
    unittest.main(verbosity=2)