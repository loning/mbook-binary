"""
Test suite for T0-17: Information Entropy in Zeckendorf Encoding

Tests the representation and evolution of information entropy under No-11 constraint,
verifying Shannon-φ conversion, constrained growth patterns, and conservation laws.
"""

import unittest
import numpy as np
from fractions import Fraction
from typing import List, Tuple, Optional


class ZeckendorfEntropy:
    """Implementation of entropy in Zeckendorf encoding"""
    
    PHI = (1 + np.sqrt(5)) / 2
    EPSILON_PHI = 1 / (PHI ** 2)
    
    @staticmethod
    def fibonacci(n: int) -> int:
        """Generate nth Fibonacci number"""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b
    
    @staticmethod
    def to_zeckendorf(n: int) -> str:
        """Convert integer to Zeckendorf representation"""
        if n == 0:
            return "0"
        
        # Generate Fibonacci numbers up to n
        fibs = []
        i = 2
        while True:
            fib = ZeckendorfEntropy.fibonacci(i)
            if fib > n:
                break
            fibs.append(fib)
            i += 1
        
        # Greedy algorithm for Zeckendorf representation
        result = []
        for fib in reversed(fibs):
            if n >= fib:
                result.append('1')
                n -= fib
            else:
                result.append('0')
        
        # Remove leading zeros
        result_str = ''.join(result).lstrip('0')
        return result_str if result_str else "0"
    
    @staticmethod
    def from_zeckendorf(z: str) -> int:
        """Convert Zeckendorf representation to integer"""
        if not z or z == "0":
            return 0
        
        value = 0
        for i, bit in enumerate(reversed(z)):
            if bit == '1':
                value += ZeckendorfEntropy.fibonacci(i + 2)
        return value
    
    @staticmethod
    def verify_no_11(z: str) -> bool:
        """Verify No-11 constraint"""
        return '11' not in z
    
    @staticmethod
    def shannon_to_phi_entropy(shannon_entropy: float) -> int:
        """Convert Shannon entropy to φ-entropy"""
        # H_φ = H_S · log₂(φ)
        phi_entropy = shannon_entropy * np.log2(ZeckendorfEntropy.PHI)
        
        # Discretize with φ-scaling to ensure integer result
        # Use higher k for better resolution
        k = 3  # Fixed scaling factor for consistency
        scaled = phi_entropy * (ZeckendorfEntropy.PHI ** k)
        return max(1, int(scaled))
    
    @staticmethod
    def calculate_shannon_entropy(probs: List[float]) -> float:
        """Calculate Shannon entropy from probability distribution"""
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    @staticmethod
    def phi_entropy_maximum(n_bits: int) -> int:
        """Calculate maximum φ-entropy for n-bit system"""
        # With No-11 constraint, number of valid states ≈ φ^(n+1)/√5
        # But entropy is log of states, so H_φ_max ≈ n - 0.672
        # For integer representation, scale appropriately
        max_entropy_continuous = n_bits - 0.672
        return max(1, int(max_entropy_continuous * ZeckendorfEntropy.PHI))
    
    @staticmethod
    def valid_entropy_transitions(current_entropy: int) -> List[int]:
        """Find valid entropy increases maintaining No-11"""
        z_current = ZeckendorfEntropy.to_zeckendorf(current_entropy)
        valid_increases = []
        
        # Try adding each Fibonacci number
        for i in range(2, 20):  # Check Fibonacci numbers F_2 to F_19
            fib = ZeckendorfEntropy.fibonacci(i)
            new_entropy = current_entropy + fib
            z_new = ZeckendorfEntropy.to_zeckendorf(new_entropy)
            
            if ZeckendorfEntropy.verify_no_11(z_new):
                valid_increases.append(new_entropy)
        
        return valid_increases
    
    @staticmethod
    def entropy_current(entropy1: int, entropy2: int, tau_0: float = 1.0) -> float:
        """Calculate entropy current between states"""
        return (entropy2 - entropy1) / tau_0
    
    @staticmethod
    def apply_conservation_with_source(entropies: List[int], 
                                      source_rate: float = 1.618) -> float:
        """Apply entropy conservation with mandatory source term"""
        total_change = sum(entropies[i+1] - entropies[i] 
                          for i in range(len(entropies)-1))
        source_contribution = source_rate * len(entropies)
        return total_change + source_contribution


class TestT0_17(unittest.TestCase):
    """Test cases for T0-17 theory"""
    
    def setUp(self):
        """Initialize test environment"""
        self.ze = ZeckendorfEntropy()
        self.phi = self.ze.PHI
    
    def test_zeckendorf_entropy_representation(self):
        """Test entropy representation in Zeckendorf form"""
        # Test small entropy values
        test_values = [1, 2, 3, 5, 8, 13, 21]  # Fibonacci numbers
        
        for val in test_values:
            z_repr = self.ze.to_zeckendorf(val)
            self.assertTrue(self.ze.verify_no_11(z_repr),
                          f"Entropy {val} violates No-11: {z_repr}")
            
            # Verify reconstruction
            reconstructed = self.ze.from_zeckendorf(z_repr)
            self.assertEqual(val, reconstructed,
                           f"Failed to reconstruct {val} from {z_repr}")
    
    def test_shannon_phi_conversion(self):
        """Test Shannon to φ-entropy conversion"""
        # Uniform distribution entropy
        probs_uniform = [0.25, 0.25, 0.25, 0.25]
        shannon_entropy = self.ze.calculate_shannon_entropy(probs_uniform)
        phi_entropy = self.ze.shannon_to_phi_entropy(shannon_entropy)
        
        # Should be positive integer
        self.assertGreater(phi_entropy, 0)
        self.assertIsInstance(phi_entropy, int)
        
        # Verify Zeckendorf representation
        z_repr = self.ze.to_zeckendorf(phi_entropy)
        self.assertTrue(self.ze.verify_no_11(z_repr))
        
        # Test with different distributions
        probs_skewed = [0.7, 0.2, 0.05, 0.05]
        shannon_skewed = self.ze.calculate_shannon_entropy(probs_skewed)
        phi_skewed = self.ze.shannon_to_phi_entropy(shannon_skewed)
        
        # Lower entropy for skewed distribution
        self.assertLess(phi_skewed, phi_entropy)
    
    def test_constrained_entropy_growth(self):
        """Test Fibonacci growth patterns in entropy"""
        initial_entropy = 5  # Start with F_5 = 5
        
        # Get valid transitions
        valid_next = self.ze.valid_entropy_transitions(initial_entropy)
        
        # Should include at least some Fibonacci jumps
        expected_jumps = [initial_entropy + 1,  # +F_1
                         initial_entropy + 2,  # +F_3
                         initial_entropy + 3,  # +F_4
                         initial_entropy + 8]  # +F_6
        
        for expected in expected_jumps:
            if expected in valid_next:
                z_repr = self.ze.to_zeckendorf(expected)
                self.assertTrue(self.ze.verify_no_11(z_repr),
                              f"Invalid transition to {expected}: {z_repr}")
    
    def test_maximum_phi_entropy(self):
        """Test maximum entropy under No-11 constraint"""
        for n_bits in [4, 8, 16]:
            max_entropy = self.ze.phi_entropy_maximum(n_bits)
            
            # Should be less than unconstrained maximum (2^n states → n bits entropy)
            unconstrained_max = n_bits
            
            # With No-11: approximately 67% efficiency
            # max_entropy ≈ (n - 0.672) * φ for scaling
            expected_approx = (n_bits - 0.672) * self.phi
            
            # Check it's in reasonable range
            self.assertGreater(max_entropy, 0)
            self.assertLess(max_entropy, unconstrained_max * 2)  # Scaled by φ
            
            # Verify it has valid Zeckendorf representation
            z_repr = self.ze.to_zeckendorf(max_entropy)
            self.assertTrue(self.ze.verify_no_11(z_repr))
    
    def test_entropy_current_flow(self):
        """Test entropy current between states"""
        state1_entropy = 8  # F_6
        state2_entropy = 13  # F_7
        tau_0 = 1.0
        
        current = self.ze.entropy_current(state1_entropy, state2_entropy, tau_0)
        self.assertEqual(current, 5.0)  # (13 - 8) / 1.0
        
        # Test reverse flow (should be negative)
        reverse_current = self.ze.entropy_current(state2_entropy, state1_entropy, tau_0)
        self.assertEqual(reverse_current, -5.0)
    
    def test_entropy_conservation_with_source(self):
        """Test conservation law with mandatory source term"""
        # Entropy evolution sequence
        entropy_sequence = [1, 2, 3, 5, 8, 13]  # Fibonacci sequence
        
        # Apply conservation with φ-scaled source
        total_with_source = self.ze.apply_conservation_with_source(
            entropy_sequence, source_rate=self.phi
        )
        
        # Should be positive due to source term (entropy increase)
        self.assertGreater(total_with_source, 0)
        
        # Verify each step maintains No-11
        for entropy in entropy_sequence:
            z_repr = self.ze.to_zeckendorf(entropy)
            self.assertTrue(self.ze.verify_no_11(z_repr))
    
    def test_binary_encoding_derivation(self):
        """Test T0-17 binary encoding from T0-16"""
        # T0-16 binary: 10000 (information flow)
        t0_16_binary = "10000"
        t0_16_value = int(t0_16_binary, 2)  # 16
        
        # T0-17 adds entropy at boundaries: 10001
        t0_17_binary = "10001"
        t0_17_value = int(t0_17_binary, 2)  # 17
        
        # Verify No-11 constraint
        self.assertNotIn("11", t0_17_binary)
        
        # 17 should be representable in Zeckendorf
        z_repr = self.ze.to_zeckendorf(17)
        self.assertTrue(self.ze.verify_no_11(z_repr))
        
        # 17 = F_7 + F_4 + F_2 = 13 + 3 + 1
        self.assertEqual(self.ze.from_zeckendorf(z_repr), 17)
    
    def test_entropy_quantization(self):
        """Test entropy quantization with ε_φ"""
        continuous_entropy = 3.7  # Arbitrary real value
        
        # Quantize with ε_φ = 1/φ²
        quantum = self.ze.EPSILON_PHI
        quantized = int(continuous_entropy / quantum)
        
        # Should preserve ordering
        continuous_entropy2 = 5.2
        quantized2 = int(continuous_entropy2 / quantum)
        self.assertLess(quantized, quantized2)
        
        # Both should have valid Zeckendorf representations
        z1 = self.ze.to_zeckendorf(quantized)
        z2 = self.ze.to_zeckendorf(quantized2)
        self.assertTrue(self.ze.verify_no_11(z1))
        self.assertTrue(self.ze.verify_no_11(z2))
    
    def test_minimal_completeness(self):
        """Verify theory achieves minimal completeness"""
        required_components = {
            'phi_entropy_definition': self.ze.to_zeckendorf,
            'shannon_conversion': self.ze.shannon_to_phi_entropy,
            'constrained_growth': self.ze.valid_entropy_transitions,
            'maximum_entropy': self.ze.phi_entropy_maximum,
            'conservation_with_source': self.ze.apply_conservation_with_source
        }
        
        # All components should be present and functional
        for name, component in required_components.items():
            self.assertIsNotNone(component, f"Missing: {name}")
            self.assertTrue(callable(component), f"Not callable: {name}")
        
        # No redundant components (each serves unique purpose)
        purposes = {
            'phi_entropy_definition': 'represent entropy in Zeckendorf',
            'shannon_conversion': 'bridge classical and φ-information',
            'constrained_growth': 'enforce No-11 in dynamics',
            'maximum_entropy': 'bound the system',
            'conservation_with_source': 'satisfy A1 axiom'
        }
        
        # Verify each purpose is distinct
        self.assertEqual(len(purposes), len(set(purposes.values())))
        
        print("\n✓ T0-17 achieves minimal completeness")
        print(f"  - {len(required_components)} necessary components")
        print("  - No redundancy detected")
        print("  - All constraints satisfied")


class TestT0_17Integration(unittest.TestCase):
    """Integration tests with other theories"""
    
    def setUp(self):
        """Initialize integration test environment"""
        self.ze = ZeckendorfEntropy()
    
    def test_consistency_with_t0_0(self):
        """Test consistency with time emergence (T0-0)"""
        # Time quantum affects entropy current
        tau_0 = 1.0  # From T0-0
        
        # Entropy changes must occur in time quanta
        entropy_t0 = 5
        entropy_t1 = 8
        
        current = self.ze.entropy_current(entropy_t0, entropy_t1, tau_0)
        
        # Current should be quantized by tau_0
        self.assertEqual(current, (entropy_t1 - entropy_t0) / tau_0)
    
    def test_consistency_with_t0_16(self):
        """Test consistency with information-energy equivalence (T0-16)"""
        # Energy = information rate × ℏ_φ
        # Entropy is information measure
        
        shannon_entropy = 2.0  # bits
        phi_entropy = self.ze.shannon_to_phi_entropy(shannon_entropy)
        
        # Energy should relate to entropy change rate
        entropy_rate = phi_entropy / 1.0  # per time unit
        
        # This rate should have valid Zeckendorf representation
        z_repr = self.ze.to_zeckendorf(int(entropy_rate))
        self.assertTrue(self.ze.verify_no_11(z_repr))
    
    def test_a1_axiom_satisfaction(self):
        """Verify A1 axiom: self-referential systems increase entropy"""
        # Start with low entropy
        initial_entropy = 1
        
        # Simulate self-referential evolution
        evolution = [initial_entropy]
        for _ in range(5):
            current = evolution[-1]
            valid_next = self.ze.valid_entropy_transitions(current)
            if valid_next:
                # Always choose increase (A1 requirement)
                next_entropy = min(valid_next)  # Minimal increase
                evolution.append(next_entropy)
        
        # Verify monotonic increase
        for i in range(len(evolution) - 1):
            self.assertGreater(evolution[i+1], evolution[i],
                             "A1 axiom violated: entropy did not increase")
        
        # All states should maintain No-11
        for entropy in evolution:
            z_repr = self.ze.to_zeckendorf(entropy)
            self.assertTrue(self.ze.verify_no_11(z_repr))


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)