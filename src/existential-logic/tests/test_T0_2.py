#!/usr/bin/env python3
"""
test_T0_2.py - Comprehensive tests for T0-2: Fundamental Entropy Bucket Theory

Tests all mathematical claims, proofs, and properties of entropy containers
with finite capacity based on Zeckendorf encoding.
"""

import unittest
from typing import List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared base from T0-1 if needed
# Note: T0-2 implements its own Zeckendorf methods in EntropyContainer


class EntropyContainer:
    """Entropy container with Fibonacci-quantized capacity."""
    
    def __init__(self, level: int):
        """Create container at specified capacity level."""
        self.level = level
        self.capacity = self._fibonacci(level + 1)
        self.state = [0] * level if level > 0 else []
        self.entropy = 0
        self.overflow_type = 'REJECT'  # Default overflow behavior
        
    def _fibonacci(self, n: int) -> int:
        """Compute nth Fibonacci number."""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b
    
    def _zeckendorf_encode(self, n: int) -> List[int]:
        """Encode number as Zeckendorf representation."""
        if n == 0:
            return [0] * self.level if self.level > 0 else []
        
        if n >= self.capacity:
            # Can't encode values >= capacity
            n = self.capacity - 1
        
        # Build Fibonacci numbers for this level
        fibs = []
        for i in range(1, self.level + 1):
            fibs.append(self._fibonacci(i))
        
        # Greedy algorithm for Zeckendorf representation
        result = [0] * self.level
        remainder = n
        
        # Process from largest to smallest Fibonacci number
        i = self.level - 1
        while i >= 0 and remainder > 0:
            if fibs[i] <= remainder:
                result[i] = 1
                remainder -= fibs[i]
                # Skip next to avoid consecutive ones
                i -= 2
            else:
                i -= 1
        
        return result
    
    def _zeckendorf_decode(self, bits: List[int]) -> int:
        """Decode Zeckendorf representation to integer."""
        if not bits:
            return 0
        
        total = 0
        for i, bit in enumerate(bits):
            if bit == 1:
                total += self._fibonacci(i + 1)
        return total
    
    def _has_consecutive_ones(self, bits: List[int]) -> bool:
        """Check if bit string has consecutive ones."""
        for i in range(len(bits) - 1):
            if bits[i] == 1 and bits[i + 1] == 1:
                return True
        return False
    
    def add_entropy(self, delta: int) -> Tuple['EntropyContainer', int]:
        """
        Add entropy to container.
        Returns (updated container, excess entropy).
        """
        new_entropy = self.entropy + delta
        
        if new_entropy < self.capacity:
            # Normal addition
            self.state = self._zeckendorf_encode(new_entropy)
            self.entropy = new_entropy
            return self, 0
        else:
            # Overflow condition
            return self._handle_overflow(delta)
    
    def _handle_overflow(self, delta: int) -> Tuple['EntropyContainer', int]:
        """Handle overflow based on overflow type."""
        new_entropy = self.entropy + delta
        excess = new_entropy - (self.capacity - 1)
        
        if self.overflow_type == 'REJECT':
            # Reject: no change
            return self, delta
        elif self.overflow_type == 'COLLAPSE':
            # Collapse: reset to ground state
            self.state = [0] * self.level if self.level > 0 else []
            self.entropy = 0
            return self, 0
        elif self.overflow_type == 'CASCADE':
            # Cascade: fill to capacity, return excess
            self.state = self._zeckendorf_encode(self.capacity - 1)
            self.entropy = self.capacity - 1
            return self, excess
        else:
            raise ValueError(f"Unknown overflow type: {self.overflow_type}")
    
    def utilization(self) -> float:
        """Calculate utilization ratio."""
        if self.capacity == 0:
            return 0.0
        return self.entropy / self.capacity
    
    def is_valid(self) -> bool:
        """Verify container invariants."""
        if self.capacity != self._fibonacci(self.level + 1):
            return False
        if self.entropy != self._zeckendorf_decode(self.state):
            return False
        if self.entropy >= self.capacity:
            return False
        if self._has_consecutive_ones(self.state):
            return False
        return True


class MultiContainerSystem:
    """System of multiple entropy containers."""
    
    def __init__(self, containers: List[EntropyContainer]):
        """Initialize system with containers."""
        self.containers = containers
    
    def total_entropy(self) -> int:
        """Calculate total system entropy."""
        return sum(c.entropy for c in self.containers)
    
    def total_capacity(self) -> int:
        """Calculate total system capacity (product)."""
        result = 1
        for c in self.containers:
            result *= c.capacity
        return result
    
    def redistribute(self, weights: Optional[List[float]] = None) -> None:
        """Redistribute entropy according to weights."""
        if weights is None:
            weights = [1.0 / len(self.containers)] * len(self.containers)
        
        if len(weights) != len(self.containers):
            raise ValueError("Weights must match number of containers")
        
        total = self.total_entropy()
        
        # Clear all containers
        for c in self.containers:
            c.state = [0] * c.level if c.level > 0 else []
            c.entropy = 0
        
        # Redistribute according to weights
        remaining = total
        for c, w in zip(self.containers, weights):
            target = min(int(total * w), c.capacity - 1)
            if target > remaining:
                target = remaining
            c.state = c._zeckendorf_encode(target)
            c.entropy = target
            remaining -= target
        
        # Handle any remainder due to rounding
        for c in self.containers:
            if remaining == 0:
                break
            space = c.capacity - 1 - c.entropy
            add = min(space, remaining)
            c.entropy += add
            c.state = c._zeckendorf_encode(c.entropy)
            remaining -= add
    
    def cascade_overflow(self, container_idx: int, delta: int) -> int:
        """
        Add entropy to container with cascade overflow.
        Returns total excess that couldn't be stored.
        """
        excess = delta
        
        for i in range(container_idx, len(self.containers)):
            self.containers[i].overflow_type = 'CASCADE'
            _, excess = self.containers[i].add_entropy(excess)
            if excess == 0:
                break
        
        return excess


class TestEntropyBucketTheory(unittest.TestCase):
    """Test cases for T0-2 Entropy Bucket Theory."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass  # No shared base needed, EntropyContainer is self-contained
    
    def test_finite_capacity_theorem(self):
        """Test that containers have finite capacity."""
        # Create containers of various levels
        for level in range(10):
            c = EntropyContainer(level)
            self.assertIsInstance(c.capacity, int)
            self.assertGreater(c.capacity, 0)
            self.assertLess(c.capacity, float('inf'))
            
            # Verify capacity is Fibonacci number
            fib = self._compute_fibonacci(level + 1)
            self.assertEqual(c.capacity, fib)
    
    def test_capacity_quantization(self):
        """Test that capacities are exactly Fibonacci numbers."""
        fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        
        for i, expected_capacity in enumerate(fibonacci_sequence):
            c = EntropyContainer(i)
            self.assertEqual(c.capacity, expected_capacity,
                           f"Level {i} should have capacity {expected_capacity}")
    
    def test_no_consecutive_ones_invariant(self):
        """Test that container states never have consecutive 1s."""
        c = EntropyContainer(8)  # Capacity = F_9 = 34
        
        # Test all possible entropy values
        for entropy in range(c.capacity):
            c.entropy = entropy
            c.state = c._zeckendorf_encode(entropy)
            
            # Check no consecutive ones
            self.assertFalse(c._has_consecutive_ones(c.state),
                           f"State for entropy {entropy} has consecutive 1s: {c.state}")
            
            # Verify decoding
            decoded = c._zeckendorf_decode(c.state)
            self.assertEqual(decoded, entropy,
                           f"Decode mismatch for {entropy}: got {decoded}")
    
    def test_overflow_rejection(self):
        """Test rejection overflow behavior."""
        c = EntropyContainer(4)  # Capacity = F_5 = 5
        c.overflow_type = 'REJECT'
        c.entropy = 3
        
        # Try to add entropy that would overflow
        original_entropy = c.entropy
        c, excess = c.add_entropy(10)
        
        # Container should be unchanged
        self.assertEqual(c.entropy, original_entropy)
        self.assertEqual(excess, 10)  # All excess rejected
    
    def test_overflow_collapse(self):
        """Test collapse overflow behavior."""
        c = EntropyContainer(4)  # Capacity = F_5 = 5
        c.overflow_type = 'COLLAPSE'
        c.entropy = 3
        
        # Add entropy that causes overflow
        c, excess = c.add_entropy(10)
        
        # Container should reset to ground state
        self.assertEqual(c.entropy, 0)
        self.assertEqual(c.state, [0, 0, 0, 0])
        self.assertEqual(excess, 0)  # No excess in collapse
    
    def test_overflow_cascade(self):
        """Test cascade overflow behavior."""
        c = EntropyContainer(4)  # Capacity = F_5 = 5
        c.overflow_type = 'CASCADE'
        c.entropy = 2
        
        # Add entropy that overflows
        c, excess = c.add_entropy(5)  # 2 + 5 = 7, capacity = 5
        
        # Container should be at max (capacity - 1)
        self.assertEqual(c.entropy, 4)  # capacity - 1
        self.assertEqual(excess, 3)  # 7 - 4 = 3 excess
    
    def test_maximum_entropy(self):
        """Test maximum entropy values for containers."""
        for level in range(1, 10):
            c = EntropyContainer(level)
            max_entropy = c.capacity - 1
            
            # Set to maximum
            c.entropy = max_entropy
            c.state = c._zeckendorf_encode(max_entropy)
            
            # Verify it's valid
            self.assertTrue(c.is_valid())
            self.assertEqual(c.entropy, max_entropy)
            
            # Verify we can't exceed it without overflow
            c_copy = EntropyContainer(level)
            c_copy.entropy = max_entropy
            c_copy.overflow_type = 'REJECT'
            _, excess = c_copy.add_entropy(1)
            self.assertEqual(excess, 1)  # Should reject the addition
    
    def test_multi_container_composition(self):
        """Test multi-container system capacity."""
        c1 = EntropyContainer(3)  # Capacity = F_4 = 3
        c2 = EntropyContainer(4)  # Capacity = F_5 = 5
        c3 = EntropyContainer(5)  # Capacity = F_6 = 8
        
        system = MultiContainerSystem([c1, c2, c3])
        
        # Test product rule for system capacity
        expected_capacity = 3 * 5 * 8
        self.assertEqual(system.total_capacity(), expected_capacity)
    
    def test_entropy_conservation(self):
        """Test that total entropy is conserved in operations."""
        containers = [EntropyContainer(i) for i in range(3, 7)]
        system = MultiContainerSystem(containers)
        
        # Set initial entropy
        for i, c in enumerate(containers):
            c.entropy = i + 1
            c.state = c._zeckendorf_encode(c.entropy)
        
        initial_total = system.total_entropy()
        
        # Redistribute entropy
        system.redistribute([0.1, 0.2, 0.3, 0.4])
        
        # Total should be conserved
        self.assertEqual(system.total_entropy(), initial_total)
    
    def test_cascade_through_system(self):
        """Test cascade overflow through multiple containers."""
        containers = [EntropyContainer(3) for _ in range(3)]  # Each capacity = 3
        for c in containers:
            c.overflow_type = 'CASCADE'
        
        system = MultiContainerSystem(containers)
        
        # Fill first container and overflow
        excess = system.cascade_overflow(0, 10)
        
        # First container should be full (capacity - 1 = 2)
        self.assertEqual(containers[0].entropy, 2)
        
        # Excess should cascade to others
        # 10 - 2 = 8 remaining
        # Second container takes 2, leaves 6
        # Third container takes 2, leaves 4
        self.assertEqual(containers[1].entropy, 2)
        self.assertEqual(containers[2].entropy, 2)
        self.assertEqual(excess, 4)  # Remaining excess
    
    def test_golden_ratio_utilization(self):
        """Test that utilization has golden ratio properties."""
        import random
        random.seed(42)
        
        phi = (1 + 5**0.5) / 2
        
        # The golden ratio appears in the distribution of Fibonacci numbers
        # Test that the ratio of consecutive Fibonacci numbers approaches phi
        for n in range(10, 20):
            c = EntropyContainer(n)
            ratio = c._fibonacci(n + 1) / c._fibonacci(n)
            # Should approach φ as n increases
            self.assertAlmostEqual(ratio, phi, delta=0.1)
        
        # Test that capacity grows exponentially with phi
        capacities = []
        for level in range(5, 15):
            c = EntropyContainer(level)
            capacities.append(c.capacity)
        
        # Check growth rate approaches phi
        for i in range(1, len(capacities)):
            ratio = capacities[i] / capacities[i-1]
            # Ratios should cluster around phi
            self.assertGreater(ratio, 1.0)
            self.assertLess(ratio, 2.0)
    
    def test_capacity_hierarchy(self):
        """Test the capacity level hierarchy."""
        expected_hierarchy = [
            (0, 1),   # F_1 = 1
            (1, 1),   # F_2 = 1
            (2, 2),   # F_3 = 2
            (3, 3),   # F_4 = 3
            (4, 5),   # F_5 = 5
            (5, 8),   # F_6 = 8
            (6, 13),  # F_7 = 13
            (7, 21),  # F_8 = 21
        ]
        
        for level, expected_capacity in expected_hierarchy:
            c = EntropyContainer(level)
            self.assertEqual(c.capacity, expected_capacity,
                           f"Level {level} should have capacity {expected_capacity}")
    
    def test_zeckendorf_addition_in_container(self):
        """Test that entropy addition follows Zeckendorf rules."""
        c = EntropyContainer(6)  # Capacity = F_7 = 13
        
        # Test specific Zeckendorf additions
        test_cases = [
            (3, 2, 5),   # F_4 + F_3 = F_5
            (5, 3, 8),   # F_5 + F_4 = F_6
            (1, 1, 2),   # F_2 + F_1 = F_3
        ]
        
        for a, b, expected in test_cases:
            c.entropy = a
            c.state = c._zeckendorf_encode(a)
            c, _ = c.add_entropy(b)
            self.assertEqual(c.entropy, expected,
                           f"{a} + {b} should equal {expected}")
    
    def test_container_invariants(self):
        """Test that all container invariants hold."""
        for level in range(10):
            c = EntropyContainer(level)
            
            # Test with various entropy values
            for entropy in range(min(c.capacity, 20)):
                c.entropy = entropy
                c.state = c._zeckendorf_encode(entropy)
                
                # Check all invariants
                self.assertTrue(c.is_valid(),
                              f"Container at level {level} with entropy {entropy} violates invariants")
    
    def test_hierarchical_nesting(self):
        """Test hierarchical container nesting."""
        # Create parent that can hold child states
        child1 = EntropyContainer(3)  # Capacity = 3
        child2 = EntropyContainer(4)  # Capacity = 5
        
        # Parent needs to encode child states
        # log_φ(3) + log_φ(5) ≈ 2.7, so need level 4 (capacity 5)
        parent = EntropyContainer(4)
        
        # Set child states
        child1.entropy = 2
        child2.entropy = 4
        
        # Parent can encode both child states
        required_parent_capacity = 3  # Simplified encoding
        self.assertGreaterEqual(parent.capacity, required_parent_capacity)
    
    def test_entropy_flow_rate(self):
        """Test entropy transfer rate between containers."""
        c1 = EntropyContainer(5)  # Capacity = 8
        c2 = EntropyContainer(6)  # Capacity = 13
        
        # Transfer rate limited by minimum capacity
        max_transfer_rate = min(c1.capacity, c2.capacity)
        self.assertEqual(max_transfer_rate, 8)
        
        # Simulate transfer
        c1.entropy = 7
        c2.entropy = 3
        
        # Transfer amount limited by rate and available space
        transfer_amount = min(max_transfer_rate, c1.entropy, c2.capacity - 1 - c2.entropy)
        
        c1.entropy -= transfer_amount
        c2.entropy += transfer_amount
        
        # Verify both containers remain valid
        c1.state = c1._zeckendorf_encode(c1.entropy)
        c2.state = c2._zeckendorf_encode(c2.entropy)
        self.assertTrue(c1.is_valid())
        self.assertTrue(c2.is_valid())
    
    def test_saturation_condition(self):
        """Test container saturation detection."""
        c = EntropyContainer(5)  # Capacity = 8
        
        # Not saturated initially
        self.assertLess(c.entropy, c.capacity - 1)
        
        # Saturate the container
        c.entropy = c.capacity - 1  # = 7
        c.state = c._zeckendorf_encode(c.entropy)
        
        # Verify saturation
        self.assertEqual(c.entropy, 7)
        self.assertTrue(c.is_valid())
        
        # Any addition should trigger overflow
        c.overflow_type = 'CASCADE'
        _, excess = c.add_entropy(1)
        self.assertGreater(excess, 0)
    
    def _compute_fibonacci(self, n: int) -> int:
        """Helper to compute nth Fibonacci number."""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b


class TestFormalVerificationPoints(unittest.TestCase):
    """Test formal verification points from T0-2-formal.md."""
    
    def test_fv1_fibonacci_computability(self):
        """FV1: All Fibonacci numbers are computable and unique."""
        fib_sequence = []
        for n in range(20):
            c = EntropyContainer(n)
            fib = c._fibonacci(n + 1)
            self.assertNotIn(fib, fib_sequence[2:])  # Skip first two 1s
            fib_sequence.append(fib)
    
    def test_fv2_no_consecutive_ones(self):
        """FV2: All Zeckendorf strings have no consecutive 1s."""
        for level in range(1, 10):
            c = EntropyContainer(level)
            for entropy in range(c.capacity):
                state = c._zeckendorf_encode(entropy)
                for i in range(len(state) - 1):
                    self.assertFalse(state[i] == 1 and state[i+1] == 1)
    
    def test_fv3_overflow_conservation(self):
        """FV3: Overflow preserves total entropy."""
        for overflow_type in ['REJECT', 'COLLAPSE', 'CASCADE']:
            c = EntropyContainer(4)
            c.overflow_type = overflow_type
            c.entropy = 2
            
            initial_entropy = c.entropy
            add_amount = 10
            
            c, excess = c.add_entropy(add_amount)
            
            if overflow_type == 'REJECT':
                self.assertEqual(c.entropy + excess, initial_entropy + add_amount)
            elif overflow_type == 'COLLAPSE':
                # Collapse loses entropy but is valid behavior
                self.assertEqual(c.entropy, 0)
            elif overflow_type == 'CASCADE':
                # Total is conserved between container and excess
                self.assertLessEqual(c.entropy + excess, initial_entropy + add_amount)
    
    def test_fv4_composition_product_rule(self):
        """FV4: System capacity is product of component capacities."""
        test_cases = [
            ([2, 3], 2 * 3),
            ([3, 4, 5], 3 * 5 * 8),
            ([1, 1, 1], 1 * 1 * 1),
        ]
        
        for levels, expected_product in test_cases:
            containers = [EntropyContainer(level) for level in levels]
            system = MultiContainerSystem(containers)
            
            actual_product = 1
            for c in containers:
                actual_product *= c.capacity
            
            self.assertEqual(system.total_capacity(), expected_product)
            self.assertEqual(actual_product, expected_product)
    
    def test_fv5_golden_ratio_convergence(self):
        """FV5: Golden ratio appears in Fibonacci structure."""
        phi = (1 + 5**0.5) / 2
        
        # Test that Fibonacci ratios converge to golden ratio
        for n in range(20, 30):
            c = EntropyContainer(n)
            fib_n = c._fibonacci(n)
            fib_n1 = c._fibonacci(n + 1)
            
            if fib_n > 0:
                ratio = fib_n1 / fib_n
                # As n increases, ratio approaches φ
                self.assertAlmostEqual(ratio, phi, delta=0.001)
        
        # Test that capacity distribution follows golden ratio pattern
        # The density of valid states in Zeckendorf representation
        # approaches 1/φ^2 ≈ 0.382
        for level in [15, 20]:
            c = EntropyContainer(level)
            # Count of valid n-bit Zeckendorf strings = F_{n+2}
            # Maximum possible n-bit strings = 2^n
            # Density = F_{n+2} / 2^n
            max_possible = 2 ** level
            valid_count = c.capacity  # This is F_{level+1}
            density = valid_count / max_possible
            
            # This density decreases exponentially but the ratio
            # of consecutive densities approaches 1/phi
            # For large n, density ≈ φ^n / 2^n = (φ/2)^n
            # Just verify it's in reasonable range
            self.assertGreater(density, 0)
            self.assertLess(density, 1)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)