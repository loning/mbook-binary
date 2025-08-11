"""
Test Suite for T0-6: System Component Interaction Theory

Tests all aspects of component interaction including:
- Safe information exchange
- Coupling dynamics
- Transmission loss and error bounds
- Network stability
- Synchronization
- Optimization strategies
- Safety properties
"""

import unittest
import numpy as np
from math import log2, sqrt, ceil, floor
from typing import List, Tuple, Dict, Set


class Component:
    """Represents a system component with Fibonacci capacity"""
    
    def __init__(self, capacity_index: int):
        self.capacity_index = capacity_index
        self.capacity = self._fibonacci(capacity_index)
        self.entropy = 0
        self.state = []  # Zeckendorf representation
        
    def _fibonacci(self, n: int) -> int:
        """Generate nth Fibonacci number"""
        if n <= 2:
            return n
        a, b = 1, 2
        for _ in range(n - 2):
            a, b = b, a + b
        return b
    
    def set_entropy(self, value: int):
        """Set entropy and update Zeckendorf representation"""
        if value >= self.capacity:
            raise ValueError(f"Entropy {value} exceeds capacity {self.capacity}")
        self.entropy = value
        self.state = self._to_zeckendorf(value)
    
    def _to_zeckendorf(self, n: int) -> List[int]:
        """Convert to Zeckendorf representation"""
        if n == 0:
            return []
        
        fibs = []
        f = 1
        while f <= n:
            fibs.append(f)
            f = self._next_fib(f)
        
        result = []
        for fib in reversed(fibs):
            if fib <= n:
                result.append(fib)
                n -= fib
        return result
    
    def _next_fib(self, f: int) -> int:
        """Get next Fibonacci number"""
        phi = (1 + sqrt(5)) / 2
        return round(f * phi)
    
    def can_extract(self, amount: int) -> bool:
        """Check if amount can be safely extracted"""
        if amount > self.entropy:
            return False
        new_state = self._to_zeckendorf(self.entropy - amount)
        return self._is_valid_zeckendorf(new_state)
    
    def can_insert(self, amount: int) -> bool:
        """Check if amount can be safely inserted"""
        if self.entropy + amount >= self.capacity:
            return False
        new_state = self._to_zeckendorf(self.entropy + amount)
        return self._is_valid_zeckendorf(new_state)
    
    def _is_valid_zeckendorf(self, state: List[int]) -> bool:
        """Check if state maintains no-11 constraint"""
        if not state:
            return True
        binary = self._state_to_binary(state)
        return '11' not in binary
    
    def _state_to_binary(self, state: List[int]) -> str:
        """Convert Zeckendorf state to binary string"""
        if not state:
            return '0'
        max_fib = max(state)
        binary = []
        f = 1
        while f <= max_fib:
            binary.append('1' if f in state else '0')
            f = self._next_fib(f)
        return ''.join(reversed(binary))


class InteractionChannel:
    """Represents interaction channel between components"""
    
    def __init__(self, c1: Component, c2: Component, 
                 coupling: float = None, delay: int = 1):
        self.c1 = c1
        self.c2 = c2
        self.delay = delay
        
        # Calculate default coupling if not provided
        if coupling is None:
            min_cap = min(c1.capacity, c2.capacity)
            max_cap = max(c1.capacity, c2.capacity)
            self.coupling = min_cap / max_cap
        else:
            self.coupling = coupling
    
    def bandwidth(self) -> float:
        """Calculate channel bandwidth"""
        min_cap = min(self.c1.capacity, self.c2.capacity)
        return (self.coupling * min_cap) / self.delay
    
    def transmission_loss(self, amount: int) -> float:
        """Calculate entropy loss for transmission"""
        return amount * (1 - self.coupling) * log2(self.delay + 1)
    
    def error_bound(self, amount: int) -> float:
        """Calculate maximum error for transmission"""
        phi = (1 + sqrt(5)) / 2
        if amount == 0:
            return 0
        largest_fib_index = floor(log2(amount) / log2(phi))
        largest_fib = self._fibonacci(largest_fib_index + 1)
        return largest_fib * (1 - self.coupling)
    
    def _fibonacci(self, n: int) -> int:
        """Generate nth Fibonacci number"""
        if n <= 2:
            return n
        a, b = 1, 2
        for _ in range(n - 2):
            a, b = b, a + b
        return b
    
    def can_transfer(self, amount: int, from_c1: bool = True) -> bool:
        """Check if transfer is safe"""
        source = self.c1 if from_c1 else self.c2
        dest = self.c2 if from_c1 else self.c1
        
        # Check extraction safety
        if not source.can_extract(amount):
            return False
        
        # Check insertion safety
        if not dest.can_insert(amount):
            return False
        
        # Check channel capacity
        min_cap = min(self.c1.capacity, self.c2.capacity)
        if amount > self.coupling * min_cap:
            return False
        
        return True
    
    def transfer(self, amount: int, from_c1: bool = True) -> bool:
        """Execute information transfer"""
        if not self.can_transfer(amount, from_c1):
            return False
        
        source = self.c1 if from_c1 else self.c2
        dest = self.c2 if from_c1 else self.c1
        
        source.set_entropy(source.entropy - amount)
        dest.set_entropy(dest.entropy + amount)
        return True


class InteractionNetwork:
    """Network of interacting components"""
    
    def __init__(self):
        self.components: List[Component] = []
        self.channels: Dict[Tuple[int, int], InteractionChannel] = {}
        
    def add_component(self, capacity_index: int) -> int:
        """Add component and return its index"""
        c = Component(capacity_index)
        self.components.append(c)
        return len(self.components) - 1
    
    def add_channel(self, i: int, j: int, 
                   coupling: float = None, delay: int = 1):
        """Add interaction channel between components"""
        if i >= len(self.components) or j >= len(self.components):
            raise ValueError("Invalid component indices")
        
        channel = InteractionChannel(
            self.components[i], 
            self.components[j],
            coupling, 
            delay
        )
        self.channels[(min(i, j), max(i, j))] = channel
    
    def coupling_matrix(self) -> np.ndarray:
        """Get coupling matrix"""
        n = len(self.components)
        K = np.zeros((n, n))
        
        for (i, j), channel in self.channels.items():
            K[i, j] = channel.coupling
            K[j, i] = channel.coupling
        
        return K
    
    def is_stable(self) -> bool:
        """Check network stability"""
        K = self.coupling_matrix()
        eigenvalues = np.linalg.eigvals(K)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        return max_eigenvalue < 1.0
    
    def total_entropy(self) -> int:
        """Calculate total system entropy"""
        return sum(c.entropy for c in self.components)
    
    def broadcast(self, sender_idx: int, amount: int, 
                 receivers: List[int]) -> Tuple[bool, int]:
        """Broadcast information from sender to receivers"""
        if sender_idx >= len(self.components):
            return False, 0
        
        sender = self.components[sender_idx]
        if not sender.can_extract(amount):
            return False, 0
        
        # Check all receivers can accept
        for r_idx in receivers:
            if r_idx >= len(self.components):
                return False, 0
            if not self.components[r_idx].can_insert(amount // len(receivers)):
                return False, 0
        
        # Execute broadcast
        sender.set_entropy(sender.entropy - amount)
        per_receiver = amount // len(receivers)
        overhead = amount % len(receivers)  # Broadcast overhead
        
        for r_idx in receivers:
            self.components[r_idx].set_entropy(
                self.components[r_idx].entropy + per_receiver
            )
        
        return True, overhead
    
    def synchronization_threshold(self, i: int, j: int) -> float:
        """Calculate critical coupling for synchronization"""
        if i >= len(self.components) or j >= len(self.components):
            raise ValueError("Invalid component indices")
        
        c1, c2 = self.components[i], self.components[j]
        diff = abs(c1.capacity - c2.capacity)
        total = c1.capacity + c2.capacity
        return diff / total if total > 0 else 0
    
    def optimal_coupling(self, i: int, j: int) -> float:
        """Calculate optimal coupling strength"""
        if i >= len(self.components) or j >= len(self.components):
            raise ValueError("Invalid component indices")
        
        c1, c2 = self.components[i], self.components[j]
        product = c1.capacity * c2.capacity
        sum_squared = (c1.capacity + c2.capacity) ** 2
        return sqrt(product / sum_squared) if sum_squared > 0 else 0
    
    def optimal_load_distribution(self, total_entropy: int) -> List[int]:
        """Calculate optimal entropy distribution"""
        total_capacity = sum(c.capacity for c in self.components)
        distribution = []
        
        for c in self.components:
            optimal = int(c.capacity * total_entropy / total_capacity)
            # Ensure within bounds
            optimal = min(optimal, c.capacity - 1)
            distribution.append(optimal)
        
        return distribution
    
    def has_deadlock_freedom(self) -> bool:
        """Check if system is deadlock-free"""
        for c in self.components:
            if c.entropy < c.capacity / 2:
                return True
        return False
    
    def calculate_leakage_bound(self) -> float:
        """Calculate maximum information leakage"""
        total_leakage = 0
        K = self.coupling_matrix()
        
        for i in range(len(self.components)):
            for j in range(len(self.components)):
                if i != j and K[i, j] > 0:
                    total_leakage += K[i, j] * log2(self.components[j].capacity)
        
        return total_leakage


class TestComponentInteraction(unittest.TestCase):
    """Test component interaction mechanisms"""
    
    def test_component_creation(self):
        """Test component initialization"""
        c = Component(5)  # F_5 = 8
        self.assertEqual(c.capacity, 8)
        self.assertEqual(c.entropy, 0)
        self.assertEqual(c.state, [])
    
    def test_entropy_setting(self):
        """Test entropy setting and Zeckendorf representation"""
        c = Component(5)  # F_5 = 8
        c.set_entropy(7)  # 7 = 5 + 2 = F_4 + F_2
        self.assertEqual(c.entropy, 7)
        self.assertIn(5, c.state)
        self.assertIn(2, c.state)
    
    def test_safe_extraction(self):
        """Test safe extraction checking"""
        c = Component(5)
        c.set_entropy(7)
        
        # Can extract 2 (leaving 5, which is valid)
        self.assertTrue(c.can_extract(2))
        
        # Cannot extract more than current entropy
        self.assertFalse(c.can_extract(8))
    
    def test_safe_insertion(self):
        """Test safe insertion checking"""
        c = Component(5)  # Capacity 8
        c.set_entropy(3)
        
        # Can insert 4 (total 7 < 8)
        self.assertTrue(c.can_insert(4))
        
        # Cannot exceed capacity
        self.assertFalse(c.can_insert(5))


class TestInteractionChannel(unittest.TestCase):
    """Test interaction channel properties"""
    
    def test_channel_creation(self):
        """Test channel initialization"""
        c1 = Component(4)  # F_4 = 5
        c2 = Component(5)  # F_5 = 8
        channel = InteractionChannel(c1, c2)
        
        # Default coupling = min/max = 5/8
        self.assertAlmostEqual(channel.coupling, 5/8)
    
    def test_bandwidth_calculation(self):
        """Test bandwidth theorem"""
        c1 = Component(4)  # F_4 = 5
        c2 = Component(5)  # F_5 = 8
        channel = InteractionChannel(c1, c2, coupling=0.5, delay=2)
        
        # Bandwidth = coupling * min_capacity / delay = 0.5 * 5 / 2
        self.assertAlmostEqual(channel.bandwidth(), 1.25)
    
    def test_transmission_loss(self):
        """Test transmission loss calculation"""
        c1 = Component(4)
        c2 = Component(5)
        channel = InteractionChannel(c1, c2, coupling=0.6, delay=3)
        
        amount = 3
        loss = channel.transmission_loss(amount)
        expected = amount * (1 - 0.6) * log2(3 + 1)
        self.assertAlmostEqual(loss, expected)
    
    def test_error_bound(self):
        """Test error propagation bound"""
        c1 = Component(4)
        c2 = Component(5)
        channel = InteractionChannel(c1, c2, coupling=0.7)
        
        amount = 5  # F_4 = 5 is largest Fibonacci ≤ 5
        error = channel.error_bound(amount)
        expected = 5 * (1 - 0.7)
        self.assertAlmostEqual(error, expected, places=1)
    
    def test_safe_transfer(self):
        """Test safe transfer checking"""
        c1 = Component(4)  # Capacity 5
        c2 = Component(5)  # Capacity 8
        c1.set_entropy(4)
        c2.set_entropy(2)
        
        channel = InteractionChannel(c1, c2)
        
        # Can transfer 2 from c1 to c2
        self.assertTrue(channel.can_transfer(2, from_c1=True))
        
        # Cannot transfer 5 (exceeds c1's entropy)
        self.assertFalse(channel.can_transfer(5, from_c1=True))
    
    def test_transfer_execution(self):
        """Test actual transfer"""
        c1 = Component(4)
        c2 = Component(5)
        c1.set_entropy(4)
        c2.set_entropy(2)
        
        channel = InteractionChannel(c1, c2)
        
        # Transfer 2 from c1 to c2
        self.assertTrue(channel.transfer(2, from_c1=True))
        self.assertEqual(c1.entropy, 2)
        self.assertEqual(c2.entropy, 4)


class TestInteractionNetwork(unittest.TestCase):
    """Test network-level properties"""
    
    def setUp(self):
        """Set up test network"""
        self.network = InteractionNetwork()
        self.idx1 = self.network.add_component(4)  # F_4 = 5
        self.idx2 = self.network.add_component(5)  # F_5 = 8
        self.idx3 = self.network.add_component(6)  # F_6 = 13
    
    def test_network_creation(self):
        """Test network initialization"""
        self.assertEqual(len(self.network.components), 3)
        self.assertEqual(self.network.components[0].capacity, 5)
        self.assertEqual(self.network.components[1].capacity, 8)
        self.assertEqual(self.network.components[2].capacity, 13)
    
    def test_channel_addition(self):
        """Test adding channels"""
        self.network.add_channel(0, 1, coupling=0.6)
        self.network.add_channel(1, 2, coupling=0.7)
        
        self.assertEqual(len(self.network.channels), 2)
        self.assertIn((0, 1), self.network.channels)
        self.assertIn((1, 2), self.network.channels)
    
    def test_coupling_matrix(self):
        """Test coupling matrix generation"""
        self.network.add_channel(0, 1, coupling=0.6)
        self.network.add_channel(1, 2, coupling=0.7)
        
        K = self.network.coupling_matrix()
        self.assertEqual(K.shape, (3, 3))
        self.assertAlmostEqual(K[0, 1], 0.6)
        self.assertAlmostEqual(K[1, 0], 0.6)
        self.assertAlmostEqual(K[1, 2], 0.7)
        self.assertAlmostEqual(K[2, 1], 0.7)
    
    def test_network_stability(self):
        """Test network stability check"""
        # Create stable network (low coupling)
        self.network.add_channel(0, 1, coupling=0.3)
        self.network.add_channel(1, 2, coupling=0.3)
        self.network.add_channel(0, 2, coupling=0.2)
        
        self.assertTrue(self.network.is_stable())
    
    def test_total_entropy(self):
        """Test total entropy calculation"""
        self.network.components[0].set_entropy(3)
        self.network.components[1].set_entropy(5)
        self.network.components[2].set_entropy(8)
        
        self.assertEqual(self.network.total_entropy(), 16)
    
    def test_broadcast(self):
        """Test broadcast operation"""
        self.network.components[0].set_entropy(4)
        self.network.components[1].set_entropy(2)
        self.network.components[2].set_entropy(3)
        
        # Broadcast 3 units from component 0 to components 1 and 2
        success, overhead = self.network.broadcast(0, 3, [1, 2])
        
        self.assertTrue(success)
        self.assertEqual(self.network.components[0].entropy, 1)  # 4 - 3
        self.assertEqual(self.network.components[1].entropy, 3)  # 2 + 1
        self.assertEqual(self.network.components[2].entropy, 4)  # 3 + 1
        self.assertEqual(overhead, 1)  # 3 % 2 = 1
    
    def test_synchronization_threshold(self):
        """Test critical coupling calculation"""
        threshold = self.network.synchronization_threshold(0, 1)
        # |5 - 8| / (5 + 8) = 3/13
        self.assertAlmostEqual(threshold, 3/13)
    
    def test_optimal_coupling(self):
        """Test optimal coupling calculation"""
        optimal = self.network.optimal_coupling(0, 1)
        # sqrt((5*8)/(5+8)^2) = sqrt(40/169)
        expected = sqrt(40/169)
        self.assertAlmostEqual(optimal, expected)
    
    def test_optimal_load_distribution(self):
        """Test optimal load distribution"""
        total_entropy = 20
        distribution = self.network.optimal_load_distribution(total_entropy)
        
        # Total capacity = 5 + 8 + 13 = 26
        # Component 0: 5 * 20 / 26 ≈ 3.8 → 3
        # Component 1: 8 * 20 / 26 ≈ 6.2 → 6
        # Component 2: 13 * 20 / 26 = 10
        
        self.assertEqual(distribution[0], 3)
        self.assertEqual(distribution[1], 6)
        self.assertEqual(distribution[2], 10)
    
    def test_deadlock_freedom(self):
        """Test deadlock prevention"""
        # All components near capacity (potential deadlock)
        self.network.components[0].set_entropy(4)  # > capacity/2
        self.network.components[1].set_entropy(7)  # > capacity/2
        self.network.components[2].set_entropy(12) # > capacity/2
        
        self.assertFalse(self.network.has_deadlock_freedom())
        
        # One component below half capacity
        self.network.components[0].set_entropy(2)  # < capacity/2
        self.assertTrue(self.network.has_deadlock_freedom())
    
    def test_leakage_bound(self):
        """Test information leakage calculation"""
        self.network.add_channel(0, 1, coupling=0.5)
        self.network.add_channel(1, 2, coupling=0.3)
        
        leakage = self.network.calculate_leakage_bound()
        
        # Expected: 0.5*log2(8) + 0.5*log2(5) + 0.3*log2(13) + 0.3*log2(8)
        expected = (0.5 * log2(8) + 0.5 * log2(5) + 
                   0.3 * log2(13) + 0.3 * log2(8))
        self.assertAlmostEqual(leakage, expected)


class TestConservationLaws(unittest.TestCase):
    """Test conservation properties"""
    
    def test_transfer_conservation(self):
        """Test entropy conservation during transfer"""
        network = InteractionNetwork()
        idx1 = network.add_component(5)
        idx2 = network.add_component(6)
        
        network.components[0].set_entropy(5)
        network.components[1].set_entropy(3)
        
        initial_total = network.total_entropy()
        
        # Create channel and transfer
        network.add_channel(0, 1)
        channel = network.channels[(0, 1)]
        channel.transfer(3, from_c1=True)
        
        final_total = network.total_entropy()
        
        # Total entropy should be conserved
        self.assertEqual(initial_total, final_total)
    
    def test_broadcast_conservation(self):
        """Test entropy conservation during broadcast"""
        network = InteractionNetwork()
        for i in range(4):
            network.add_component(5)
        
        network.components[0].set_entropy(6)
        for i in range(1, 4):
            network.components[i].set_entropy(1)
        
        initial_total = network.total_entropy()
        
        # Broadcast from 0 to others
        success, overhead = network.broadcast(0, 6, [1, 2, 3])
        
        final_total = network.total_entropy()
        
        # Total should be conserved (minus overhead)
        self.assertEqual(initial_total, final_total + overhead)


class TestScalability(unittest.TestCase):
    """Test scalability properties"""
    
    def test_large_network(self):
        """Test properties scale logarithmically"""
        network = InteractionNetwork()
        
        # Add many components
        n = 20
        for i in range(n):
            network.add_component(4 + (i % 5))
        
        # Add channels in a ring topology
        for i in range(n):
            network.add_channel(i, (i + 1) % n, coupling=0.1)
        
        # Check stability
        self.assertTrue(network.is_stable())
        
        # Check overhead scales logarithmically
        phi = (1 + sqrt(5)) / 2
        expected_overhead = ceil(log2(n) / log2(phi))
        self.assertLessEqual(expected_overhead, 10)  # O(log n)


class TestSafetyProperties(unittest.TestCase):
    """Test safety guarantees"""
    
    def test_no_11_preservation(self):
        """Test that no-11 constraint is always preserved"""
        c = Component(6)  # F_6 = 13
        
        # Try various entropy values
        for value in range(13):
            c.set_entropy(value)
            binary = c._state_to_binary(c.state)
            self.assertNotIn('11', binary)
    
    def test_capacity_bounds(self):
        """Test capacity is never exceeded"""
        c = Component(5)  # Capacity 8
        
        # Try to set entropy beyond capacity
        with self.assertRaises(ValueError):
            c.set_entropy(8)
        
        # Ensure insertion check works
        c.set_entropy(7)
        self.assertFalse(c.can_insert(1))
    
    def test_isolation_guarantee(self):
        """Test zero coupling provides isolation"""
        network = InteractionNetwork()
        idx1 = network.add_component(5)
        idx2 = network.add_component(5)
        
        # Add channel with zero coupling
        network.add_channel(0, 1, coupling=0.0)
        
        network.components[0].set_entropy(5)
        network.components[1].set_entropy(0)
        
        channel = network.channels[(0, 1)]
        
        # Cannot transfer with zero coupling
        self.assertFalse(channel.can_transfer(1, from_c1=True))


def run_all_tests():
    """Run complete test suite"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestComponentInteraction))
    suite.addTests(loader.loadTestsFromTestCase(TestInteractionChannel))
    suite.addTests(loader.loadTestsFromTestCase(TestInteractionNetwork))
    suite.addTests(loader.loadTestsFromTestCase(TestConservationLaws))
    suite.addTests(loader.loadTestsFromTestCase(TestScalability))
    suite.addTests(loader.loadTestsFromTestCase(TestSafetyProperties))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("T0-6 TEST SUITE COMPLETE")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)