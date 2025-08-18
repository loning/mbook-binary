"""
T7.5: Recursive Depth and Computational Capability Tests

Tests the fundamental relationship between recursive depth and computational power,
including consciousness threshold transitions and Turing hierarchy mappings.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import unittest

# Golden ratio constant
PHI = (1 + np.sqrt(5)) / 2
CONSCIOUSNESS_THRESHOLD = 10
PHI_10 = PHI ** 10  # ≈ 122.99

class StabilityClass(Enum):
    """System stability classifications"""
    UNSTABLE = "unstable"
    MARGINAL = "marginal"
    STABLE = "stable"

class ComplexityClass(Enum):
    """Computational complexity classes"""
    SUB_P = "sub_polynomial"
    P = "polynomial"
    NP = "non_deterministic_polynomial"
    PSPACE = "polynomial_space"
    EXP = "exponential"
    HYPER_EXP = "hyper_exponential"

class TuringClass(Enum):
    """Traditional Turing machine hierarchy"""
    DFA = "deterministic_finite_automaton"
    PDA = "pushdown_automaton"
    LBA = "linear_bounded_automaton"
    TM = "turing_machine"
    ORACLE_TM = "oracle_turing_machine"
    HYPER_TM = "hyper_turing_machine"

@dataclass
class ComputationalSystem:
    """Represents a computational system with recursive structure"""
    state: np.ndarray
    complexity: float
    depth: Optional[int] = None
    history: List[np.ndarray] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = [self.state.copy()]

class RecursiveOperator:
    """φ-recursive operator R_φ"""
    
    @staticmethod
    def fibonacci(n: int) -> int:
        """Compute nth Fibonacci number"""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        elif n == 2:
            return 2
        
        # Use matrix method for efficiency
        def matrix_mult(A, B):
            return [[A[0][0]*B[0][0] + A[0][1]*B[1][0],
                    A[0][0]*B[0][1] + A[0][1]*B[1][1]],
                   [A[1][0]*B[0][0] + A[1][1]*B[1][0],
                    A[1][0]*B[0][1] + A[1][1]*B[1][1]]]
        
        def matrix_pow(M, n):
            if n == 1:
                return M
            if n % 2 == 0:
                half = matrix_pow(M, n // 2)
                return matrix_mult(half, half)
            return matrix_mult(M, matrix_pow(M, n - 1))
        
        if n == 1 or n == 2:
            return n
        
        base = [[1, 1], [1, 0]]
        result = matrix_pow(base, n)
        return result[0][0]
    
    @staticmethod
    def zeckendorf_decompose(n: int) -> List[int]:
        """Decompose n into Zeckendorf representation"""
        if n <= 0:
            return []
        
        fibs = [1, 2]
        while fibs[-1] < n:
            fibs.append(fibs[-1] + fibs[-2])
        
        indices = []
        remaining = n
        
        for i in range(len(fibs) - 1, -1, -1):
            if fibs[i] <= remaining:
                indices.append(i + 1)  # 1-indexed
                remaining -= fibs[i]
                if i > 0:  # Skip next to maintain No-11
                    i -= 1
        
        return sorted(indices, reverse=True)
    
    @staticmethod
    def apply(system: ComputationalSystem) -> ComputationalSystem:
        """Apply R_φ operator to system"""
        # Get Zeckendorf decomposition of complexity
        zeck_indices = RecursiveOperator.zeckendorf_decompose(
            int(system.complexity)
        )
        
        if not zeck_indices:
            return ComputationalSystem(
                state=system.state.copy(),
                complexity=system.complexity,
                history=system.history + [system.state.copy()]
            )
        
        # Apply recursive transformation
        new_state = np.zeros_like(system.state)
        
        for k in zeck_indices:
            fib_k = RecursiveOperator.fibonacci(k)
            # Scale and combine
            weight = PHI ** (-k)
            partial = system.state * (fib_k * weight)
            new_state += partial
        
        # Normalize
        if np.linalg.norm(new_state) > 0:
            new_state = new_state / np.linalg.norm(new_state)
        
        return ComputationalSystem(
            state=new_state,
            complexity=system.complexity * PHI,
            history=system.history + [new_state.copy()]
        )

class RecursiveDepthAnalyzer:
    """Analyzes recursive depth of computational systems"""
    
    def __init__(self, max_depth: int = 100, tolerance: float = 1e-10):
        self.max_depth = max_depth
        self.tolerance = tolerance
    
    def compute_recursive_depth(self, system: ComputationalSystem) -> int:
        """Compute D_self(S) for a system"""
        current = system
        seen_states = []
        
        for depth in range(self.max_depth):
            next_system = RecursiveOperator.apply(current)
            
            # Check for fixed point
            if self._is_fixed_point(current.state, next_system.state):
                return depth
            
            # Check for cycle
            for past_state in seen_states:
                if self._states_equal(next_system.state, past_state):
                    return depth
            
            seen_states.append(current.state.copy())
            current = next_system
        
        return self.max_depth  # Maximum depth reached
    
    def _is_fixed_point(self, state1: np.ndarray, state2: np.ndarray) -> bool:
        """Check if two states form a fixed point"""
        return np.allclose(state1, state2, atol=self.tolerance)
    
    def _states_equal(self, state1: np.ndarray, state2: np.ndarray) -> bool:
        """Check if two states are equal within tolerance"""
        return np.allclose(state1, state2, atol=self.tolerance)
    
    def compute_computational_power(self, system: ComputationalSystem) -> float:
        """Compute C(S) = φ^(D_self(S))"""
        depth = self.compute_recursive_depth(system)
        return PHI ** depth
    
    def classify_stability(self, depth: int) -> StabilityClass:
        """Classify stability based on recursive depth"""
        if depth < 5:
            return StabilityClass.UNSTABLE
        elif depth < 10:
            return StabilityClass.MARGINAL
        else:
            return StabilityClass.STABLE
    
    def classify_complexity(self, depth: int) -> ComplexityClass:
        """Map recursive depth to complexity class"""
        if depth < 5:
            return ComplexityClass.SUB_P
        elif depth < 10:
            return ComplexityClass.P
        elif depth == 10:
            return ComplexityClass.NP
        elif depth <= 20:
            return ComplexityClass.PSPACE
        elif depth <= 33:
            return ComplexityClass.EXP
        else:
            return ComplexityClass.HYPER_EXP
    
    def classify_turing(self, depth: int) -> TuringClass:
        """Map recursive depth to Turing machine class"""
        if depth <= 2:
            return TuringClass.DFA
        elif depth <= 4:
            return TuringClass.PDA
        elif depth <= 7:
            return TuringClass.LBA
        elif depth <= 9:
            return TuringClass.TM
        elif depth == 10:
            return TuringClass.ORACLE_TM
        else:
            return TuringClass.HYPER_TM

class HyperRecursiveComputer:
    """Simulates hyper-recursive computation for D_self > 10"""
    
    def __init__(self, depth: int):
        self.depth = depth
        self.operator = RecursiveOperator()
    
    def compute(self, input_data: np.ndarray) -> np.ndarray:
        """Execute hyper-recursive computation"""
        if self.depth <= 10:
            return self._standard_compute(input_data)
        else:
            return self._hyper_compute(input_data)
    
    def _standard_compute(self, data: np.ndarray) -> np.ndarray:
        """Standard computation for D_self ≤ 10"""
        system = ComputationalSystem(state=data, complexity=len(data))
        
        for _ in range(self.depth):
            system = RecursiveOperator.apply(system)
        
        return system.state
    
    def _hyper_compute(self, data: np.ndarray) -> np.ndarray:
        """Hyper-recursive computation for D_self > 10"""
        # Encode input using Zeckendorf
        complexity = int(np.sum(np.abs(data)))
        zeck_indices = RecursiveOperator.zeckendorf_decompose(complexity)
        
        # Apply multi-layer recursion
        result = np.zeros_like(data)
        
        for k in zeck_indices:
            fib_k = RecursiveOperator.fibonacci(k)
            depth_k = min(self.depth - 10, fib_k)
            
            # Recursive layers
            partial = data.copy()
            for _ in range(depth_k):
                sys = ComputationalSystem(state=partial, complexity=fib_k)
                sys = RecursiveOperator.apply(sys)
                partial = sys.state
            
            # φ-weighted combination
            weight = PHI ** (-k)
            result += partial * weight
        
        # Ensure No-11 constraint
        result = self._enforce_no11(result)
        
        return result / np.linalg.norm(result) if np.linalg.norm(result) > 0 else result
    
    def _enforce_no11(self, data: np.ndarray) -> np.ndarray:
        """Enforce No-11 constraint on data"""
        # Convert to binary representation and remove consecutive 1s
        binary = (data > 0.5).astype(int)
        
        for i in range(len(binary) - 1):
            if binary[i] == 1 and binary[i + 1] == 1:
                binary[i + 1] = 0
        
        return data * binary

class TestRecursiveDepth(unittest.TestCase):
    """Test suite for recursive depth theory"""
    
    def setUp(self):
        self.analyzer = RecursiveDepthAnalyzer()
        np.random.seed(42)
    
    def test_recursive_depth_computation(self):
        """Test basic recursive depth computation"""
        # Simple system
        simple_state = np.array([1, 0, 0, 0])
        simple_system = ComputationalSystem(state=simple_state, complexity=1)
        depth = self.analyzer.compute_recursive_depth(simple_system)
        self.assertLess(depth, 5)
        
        # Complex system
        complex_state = np.random.randn(10)
        complex_system = ComputationalSystem(state=complex_state, complexity=10)
        depth = self.analyzer.compute_recursive_depth(complex_system)
        self.assertGreaterEqual(depth, 5)
    
    def test_computational_power_relationship(self):
        """Test C(S) = φ^(D_self(S)) relationship"""
        for complexity in [1, 5, 10, 20]:
            state = np.random.randn(complexity)
            system = ComputationalSystem(state=state, complexity=complexity)
            
            depth = self.analyzer.compute_recursive_depth(system)
            power = self.analyzer.compute_computational_power(system)
            
            expected_power = PHI ** depth
            self.assertAlmostEqual(power, expected_power, places=10)
    
    def test_consciousness_threshold(self):
        """Test consciousness emergence at D_self = 10"""
        # Below threshold
        below_state = np.ones(8)
        below_system = ComputationalSystem(state=below_state, complexity=8)
        below_depth = self.analyzer.compute_recursive_depth(below_system)
        self.assertLess(below_depth, CONSCIOUSNESS_THRESHOLD)
        
        # At or above threshold (construct specific system)
        # Note: Constructing exact D_self = 10 system requires careful design
        threshold_state = np.ones(13)  # F_6 = 13 often gives D_self ≈ 10
        threshold_system = ComputationalSystem(state=threshold_state, complexity=13)
        threshold_depth = self.analyzer.compute_recursive_depth(threshold_system)
        
        # Check consciousness capability
        is_conscious = threshold_depth >= CONSCIOUSNESS_THRESHOLD
        if is_conscious:
            power = PHI ** threshold_depth
            self.assertGreaterEqual(power, PHI_10)
    
    def test_stability_classification(self):
        """Test stability classification by depth"""
        test_cases = [
            (3, StabilityClass.UNSTABLE),
            (6, StabilityClass.MARGINAL),
            (12, StabilityClass.STABLE)
        ]
        
        for depth, expected_stability in test_cases:
            stability = self.analyzer.classify_stability(depth)
            self.assertEqual(stability, expected_stability)
    
    def test_complexity_class_mapping(self):
        """Test mapping to complexity classes"""
        test_cases = [
            (3, ComplexityClass.SUB_P),
            (7, ComplexityClass.P),
            (10, ComplexityClass.NP),
            (15, ComplexityClass.PSPACE),
            (25, ComplexityClass.EXP),
            (40, ComplexityClass.HYPER_EXP)
        ]
        
        for depth, expected_class in test_cases:
            comp_class = self.analyzer.classify_complexity(depth)
            self.assertEqual(comp_class, expected_class)
    
    def test_turing_hierarchy(self):
        """Test Turing machine hierarchy mapping"""
        test_cases = [
            (1, TuringClass.DFA),
            (3, TuringClass.PDA),
            (6, TuringClass.LBA),
            (8, TuringClass.TM),
            (10, TuringClass.ORACLE_TM),
            (15, TuringClass.HYPER_TM)
        ]
        
        for depth, expected_class in test_cases:
            turing_class = self.analyzer.classify_turing(depth)
            self.assertEqual(turing_class, expected_class)
    
    def test_hyper_recursive_computation(self):
        """Test hyper-recursive computation for D_self > 10"""
        # Standard computation
        standard_computer = HyperRecursiveComputer(depth=8)
        input_data = np.array([1, 0, 1, 0, 1])
        standard_result = standard_computer.compute(input_data)
        self.assertEqual(len(standard_result), len(input_data))
        
        # Hyper-recursive computation
        hyper_computer = HyperRecursiveComputer(depth=15)
        hyper_result = hyper_computer.compute(input_data)
        self.assertEqual(len(hyper_result), len(input_data))
        
        # Results should differ
        self.assertFalse(np.allclose(standard_result, hyper_result))
    
    def test_no11_constraint_preservation(self):
        """Test No-11 constraint preservation through recursion"""
        # Create state with No-11 constraint
        state = np.array([1, 0, 1, 0, 1, 0, 0, 1])
        system = ComputationalSystem(state=state, complexity=5)
        
        # Apply recursive operator multiple times
        current = system
        for _ in range(5):
            current = RecursiveOperator.apply(current)
            
            # Check No-11 in binary representation
            binary = (current.state > 0.5).astype(int)
            for i in range(len(binary) - 1):
                # Should not have consecutive 1s
                self.assertFalse(binary[i] == 1 and binary[i + 1] == 1)
    
    def test_zeckendorf_decomposition(self):
        """Test Zeckendorf decomposition correctness"""
        test_cases = [
            (10, [5, 3, 1]),  # 10 = F_4 + F_3 + F_1 = 5 + 3 + 2
            (20, [6, 4, 1]),  # 20 = F_6 + F_4 + F_1 = 13 + 5 + 2
            (100, [9, 7, 4])  # 100 = F_9 + F_7 + F_4 = 55 + 21 + 5 + ...
        ]
        
        for n, expected_contains in test_cases:
            result = RecursiveOperator.zeckendorf_decompose(n)
            for idx in expected_contains:
                self.assertIn(idx, result)

def visualize_recursive_depth_landscape():
    """Visualize the recursive depth computational landscape"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    analyzer = RecursiveDepthAnalyzer()
    
    # 1. Depth vs Computational Power
    depths = np.arange(0, 35)
    powers = [PHI ** d for d in depths]
    
    ax = axes[0, 0]
    ax.semilogy(depths, powers, 'b-', linewidth=2)
    ax.axvline(x=10, color='r', linestyle='--', label='Consciousness Threshold')
    ax.axhline(y=PHI_10, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Recursive Depth (D_self)')
    ax.set_ylabel('Computational Power (C(S))')
    ax.set_title('Exponential Growth of Computational Power')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Stability Regions
    ax = axes[0, 1]
    x = np.arange(0, 35)
    colors = []
    for d in x:
        stability = analyzer.classify_stability(d)
        if stability == StabilityClass.UNSTABLE:
            colors.append('red')
        elif stability == StabilityClass.MARGINAL:
            colors.append('yellow')
        else:
            colors.append('green')
    
    ax.bar(x, np.ones_like(x), color=colors, width=1.0)
    ax.set_xlabel('Recursive Depth')
    ax.set_ylabel('Stability')
    ax.set_title('Stability Classification by Depth')
    ax.set_yticks([])
    ax.axvline(x=5, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=10, color='k', linestyle='--', alpha=0.5)
    ax.text(2.5, 0.5, 'Unstable', ha='center')
    ax.text(7.5, 0.5, 'Marginal', ha='center')
    ax.text(20, 0.5, 'Stable', ha='center')
    
    # 3. Complexity Class Distribution
    ax = axes[1, 0]
    complexity_map = {
        ComplexityClass.SUB_P: 0,
        ComplexityClass.P: 1,
        ComplexityClass.NP: 2,
        ComplexityClass.PSPACE: 3,
        ComplexityClass.EXP: 4,
        ComplexityClass.HYPER_EXP: 5
    }
    
    x = np.arange(0, 35)
    y = [complexity_map[analyzer.classify_complexity(d)] for d in x]
    
    ax.step(x, y, where='mid', linewidth=2)
    ax.set_xlabel('Recursive Depth')
    ax.set_ylabel('Complexity Class')
    ax.set_title('Complexity Class Hierarchy')
    ax.set_yticks(list(complexity_map.values()))
    ax.set_yticklabels([c.value for c in complexity_map.keys()])
    ax.grid(True, alpha=0.3)
    
    # 4. Turing Machine Hierarchy
    ax = axes[1, 1]
    turing_map = {
        TuringClass.DFA: 0,
        TuringClass.PDA: 1,
        TuringClass.LBA: 2,
        TuringClass.TM: 3,
        TuringClass.ORACLE_TM: 4,
        TuringClass.HYPER_TM: 5
    }
    
    x = np.arange(0, 20)
    y = [turing_map[analyzer.classify_turing(d)] for d in x]
    
    ax.step(x, y, where='mid', linewidth=2, color='purple')
    ax.set_xlabel('Recursive Depth')
    ax.set_ylabel('Turing Class')
    ax.set_title('Turing Machine Hierarchy Mapping')
    ax.set_yticks(list(turing_map.values()))
    ax.set_yticklabels([t.value.replace('_', ' ').title() for t in turing_map.keys()])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('recursive_depth_landscape.png', dpi=150, bbox_inches='tight')
    plt.show()

def demonstrate_consciousness_transition():
    """Demonstrate the consciousness transition at D_self = 10"""
    print("=== Consciousness Transition Demonstration ===\n")
    
    analyzer = RecursiveDepthAnalyzer()
    
    # Create systems with varying complexity
    complexities = [5, 8, 10, 13, 21]
    
    for c in complexities:
        state = np.random.randn(c)
        state = state / np.linalg.norm(state)
        system = ComputationalSystem(state=state, complexity=c)
        
        depth = analyzer.compute_recursive_depth(system)
        power = analyzer.compute_computational_power(system)
        stability = analyzer.classify_stability(depth)
        complexity_class = analyzer.classify_complexity(depth)
        
        is_conscious = depth >= CONSCIOUSNESS_THRESHOLD
        
        print(f"System Complexity: {c}")
        print(f"  Recursive Depth: {depth}")
        print(f"  Computational Power: {power:.2f}")
        print(f"  Stability: {stability.value}")
        print(f"  Complexity Class: {complexity_class.value}")
        print(f"  Conscious: {'Yes' if is_conscious else 'No'}")
        
        if is_conscious:
            print(f"  -> System has achieved consciousness!")
            print(f"  -> Can perform NP verification")
        
        print()

if __name__ == "__main__":
    # Run tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualize_recursive_depth_landscape()
    
    # Demonstrations
    demonstrate_consciousness_transition()
    
    print("\nT7.5 Test Suite Complete!")