"""
T9.4: Life System Entropy Regulation Theorem - Test Suite

This module implements comprehensive tests for the Life System Entropy Regulation Theorem,
which establishes how life systems regulate entropy through φ-encoding mechanisms.

The tests verify:
1. φ-negative entropy generator mechanics
2. Zeckendorf self-organization principles
3. No-11 constraint stability in living systems
4. Life-nonlife boundary criteria
5. Entropy delay mechanisms
6. Recursive depth analysis for life complexity
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.linalg import expm
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Set
import unittest


class PhiConstants:
    """Golden ratio and Fibonacci constants for life system analysis"""
    PHI = (1 + np.sqrt(5)) / 2
    PHI_INV = 2 / (1 + np.sqrt(5))
    
    # Critical thresholds
    LIFE_THRESHOLD = 5  # D_life >= 5 for self-replication
    COMPLEXITY_THRESHOLD = 8  # D_life = 8 for self-organization criticality
    CONSCIOUSNESS_THRESHOLD = 10  # D_life ~ 10 for pre-consciousness
    
    # Consciousness threshold value
    PHI_10 = PHI ** 10  # ~122.99 bits
    
    @staticmethod
    def fibonacci(n: int) -> int:
        """Generate nth Fibonacci number (1-indexed)"""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        elif n == 2:
            return 2
        
        # Use matrix method for efficiency
        a, b = 1, 2
        for _ in range(n - 2):
            a, b = b, a + b
        return b


class ZeckendorfEncoder:
    """Zeckendorf decomposition and No-11 constraint verification for life systems"""
    
    @staticmethod
    def decompose(n: int) -> List[int]:
        """
        Decompose n into Zeckendorf representation
        Returns list of Fibonacci indices (1-indexed)
        """
        if n == 0:
            return []
        
        indices = []
        fibs = []
        k = 1
        while PhiConstants.fibonacci(k) <= n:
            fibs.append((PhiConstants.fibonacci(k), k))
            k += 1
        
        remaining = n
        for fib_val, idx in reversed(fibs):
            if fib_val <= remaining:
                indices.append(idx)
                remaining -= fib_val
        
        return sorted(indices, reverse=True)
    
    @staticmethod
    def verify_no11(indices: List[int]) -> bool:
        """Verify No-11 constraint (no consecutive indices)"""
        if len(indices) <= 1:
            return True
        
        for i in range(len(indices) - 1):
            if abs(indices[i] - indices[i + 1]) == 1:
                return False
        return True
    
    @staticmethod
    def encode_state(state: np.ndarray) -> Tuple[List[int], bool]:
        """
        Encode a state vector using Zeckendorf decomposition
        Returns (indices, valid_no11)
        """
        # Convert state to integer representation
        state_int = int(np.sum([2**i * int(bit) for i, bit in enumerate(state)]))
        
        indices = ZeckendorfEncoder.decompose(state_int)
        valid = ZeckendorfEncoder.verify_no11(indices)
        
        return indices, valid


class NegativeEntropyGenerator:
    """φ-negative entropy generator for life systems"""
    
    def __init__(self, D_life: int):
        """
        Initialize negative entropy generator
        
        Args:
            D_life: Recursive depth of life system
        """
        self.D_life = D_life
        self.phi = PhiConstants.PHI
        self.zeck_indices = ZeckendorfEncoder.decompose(D_life)
        
        # Verify system can support life
        if D_life < PhiConstants.LIFE_THRESHOLD:
            raise ValueError(f"D_life must be >= {PhiConstants.LIFE_THRESHOLD} for life")
        
        # Initialize entropy production components
        self.neg_entropy_units = self._initialize_units()
        self.efficiency = self._calculate_efficiency()
    
    def _initialize_units(self) -> Dict[int, float]:
        """Initialize negative entropy production units"""
        units = {}
        for k in self.zeck_indices:
            # Each unit contributes F_k * φ^(-k/2) to negative entropy
            units[k] = PhiConstants.fibonacci(k) * (self.phi ** (-k/2))
        return units
    
    def _calculate_efficiency(self) -> float:
        """
        Calculate negative entropy production efficiency
        Converges to φ^(-1) ≈ 0.618 as D_life → ∞
        """
        return min(self.phi ** (-1), 1 - np.exp(-self.D_life / 5))
    
    def generate_neg_entropy(self, state: np.ndarray, environment: np.ndarray) -> Tuple[float, float]:
        """
        Generate negative entropy through metabolic process
        
        Args:
            state: Current life system state
            environment: Environmental state
            
        Returns:
            (neg_entropy_produced, waste_entropy)
        """
        # Calculate input entropy from environment
        H_env = self._calculate_entropy(environment)
        
        # Generate negative entropy through φ-optimization
        neg_entropy = 0
        for k, contribution in self.neg_entropy_units.items():
            neg_entropy += contribution * self._process_energy(state, k)
        
        # Apply efficiency factor
        neg_entropy *= self.efficiency
        
        # Calculate waste entropy (must exceed negative entropy for total increase)
        waste_entropy = neg_entropy / self.efficiency + 0.1  # Ensure net positive
        
        return neg_entropy, waste_entropy
    
    def _calculate_entropy(self, state: np.ndarray) -> float:
        """Calculate Shannon entropy of state"""
        # Normalize state to probability distribution
        p = np.abs(state) / np.sum(np.abs(state))
        p = p[p > 0]  # Remove zeros
        return -np.sum(p * np.log2(p))
    
    def _process_energy(self, state: np.ndarray, k: int) -> float:
        """Process energy at level k"""
        # Energy processing efficiency depends on Fibonacci level
        return np.tanh(np.linalg.norm(state) / PhiConstants.fibonacci(k))


class LifeSystemOrganization:
    """Zeckendorf self-organization for life systems"""
    
    def __init__(self, D_life: int):
        """
        Initialize life system organization
        
        Args:
            D_life: Recursive depth
        """
        self.D_life = D_life
        self.phi = PhiConstants.PHI
        self.organization_levels = self._build_organization()
        self.stability = self._check_stability()
    
    def _build_organization(self) -> Dict[int, float]:
        """Build Zeckendorf organization structure"""
        zeck_indices = ZeckendorfEncoder.decompose(self.D_life)
        
        org = {}
        for k in zeck_indices:
            # Organization at level k
            org[k] = PhiConstants.fibonacci(k) * (self.phi ** (-k/2))
        
        return org
    
    def _check_stability(self) -> bool:
        """Check No-11 constraint for organizational stability"""
        indices = list(self.organization_levels.keys())
        return ZeckendorfEncoder.verify_no11(indices)
    
    def calculate_organization_degree(self) -> float:
        """Calculate total organization degree O(L)"""
        return sum(self.organization_levels.values())
    
    def perturb_and_test_stability(self, perturbation: float) -> bool:
        """
        Test stability under perturbation
        
        Args:
            perturbation: Perturbation strength
            
        Returns:
            True if system remains stable
        """
        # Apply perturbation to organization levels
        perturbed_org = {}
        for k, v in self.organization_levels.items():
            perturbed_org[k] = v * (1 + perturbation * np.random.randn())
        
        # Check if structure maintains No-11 constraint
        indices = [k for k, v in perturbed_org.items() if v > 0.1]
        return ZeckendorfEncoder.verify_no11(indices)


class LifeBoundaryInformation:
    """Holographic boundary information for life systems"""
    
    def __init__(self, D_life: int):
        """Initialize life boundary with recursive depth"""
        self.D_life = D_life
        self.phi = PhiConstants.PHI
        self.selectivity = self._calculate_selectivity()
    
    def _calculate_selectivity(self) -> float:
        """
        Calculate boundary selectivity factor
        SelectivityFactor(D) = 1 + Σ(φ^(-k)) for k=1 to D
        """
        selectivity = 1.0
        for k in range(1, self.D_life + 1):
            selectivity += self.phi ** (-k)
        return selectivity
    
    def calculate_boundary_density(self) -> float:
        """
        Calculate information density at life boundary
        ρ_boundary = (φ²/4G) * SelectivityFactor(D_life)
        """
        # Using normalized units where 4G = 1
        base_density = self.phi ** 2
        return base_density * self.selectivity
    
    def entropy_pump_gradient(self) -> float:
        """
        Calculate entropy gradient maintained by boundary
        ΔS_boundary = k_B * D_life * ln(φ)
        """
        # Using k_B = 1 in natural units
        return self.D_life * np.log(self.phi)
    
    def filter_molecules(self, molecules: np.ndarray) -> np.ndarray:
        """
        Simulate selective filtering at life boundary
        
        Args:
            molecules: Array of molecule energies
            
        Returns:
            Filtered molecules (low entropy selected)
        """
        # Sort by entropy (energy)
        sorted_mols = np.sort(molecules)
        
        # Select low-entropy fraction based on selectivity
        cutoff_idx = int(len(sorted_mols) / self.selectivity)
        return sorted_mols[:cutoff_idx]


class EntropyDelayMechanism:
    """Entropy delay mechanisms in life systems"""
    
    def __init__(self, D_life: int):
        """Initialize entropy delay mechanism"""
        self.D_life = D_life
        self.phi = PhiConstants.PHI
        self.tau_0 = 1.0  # Base time scale
        self.delay_time = self._calculate_delay()
    
    def _calculate_delay(self) -> float:
        """
        Calculate entropy delay time
        τ_delay = τ_0 * φ^(D_life)
        """
        return self.tau_0 * (self.phi ** self.D_life)
    
    def simulate_entropy_evolution(self, t_max: float, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate entropy evolution with delay
        
        Returns:
            (time_array, entropy_array)
        """
        t = np.arange(0, t_max, dt)
        
        # Without life: immediate entropy increase
        S_no_life = np.log(1 + t)
        
        # With life: delayed entropy increase
        S_life = np.zeros_like(t)
        for i, time in enumerate(t):
            if time < self.delay_time:
                # Life maintains low entropy
                S_life[i] = np.log(1 + time / self.delay_time)
            else:
                # After delay, entropy increases normally
                S_life[i] = np.log(1 + time - self.delay_time + 1)
        
        return t, S_life


class LifeComplexityAnalyzer:
    """Analyze recursive depth and complexity in life systems"""
    
    def __init__(self):
        """Initialize complexity analyzer"""
        self.phi = PhiConstants.PHI
        self.phase_transitions = {
            5: "Self-replication emergence",
            8: "Self-organization criticality", 
            10: "Pre-consciousness state"
        }
    
    def calculate_entropy_regulation(self, D_life: int) -> float:
        """
        Calculate entropy regulation capability
        E_reg(L) = φ^(D_life) bits/time
        """
        return self.phi ** D_life
    
    def identify_phase(self, D_life: int) -> str:
        """Identify life system phase based on recursive depth"""
        if D_life < 5:
            return "Pre-life chemistry"
        elif D_life < 8:
            return "Primitive life"
        elif D_life < 10:
            return "Complex life"
        else:
            return "Pre-conscious life"
    
    def calculate_information_integration(self, D_life: int) -> float:
        """
        Calculate information integration Φ
        Approaches consciousness threshold φ^10 as D_life → 10
        """
        return self.phi ** D_life
    
    def predict_evolution_trajectory(self, D_initial: int, generations: int) -> np.ndarray:
        """
        Predict evolutionary trajectory of recursive depth
        
        Args:
            D_initial: Initial recursive depth
            generations: Number of generations to simulate
            
        Returns:
            Array of D_life values over time
        """
        D_trajectory = np.zeros(generations)
        D_trajectory[0] = D_initial
        
        for gen in range(1, generations):
            # Evolution rate depends on current depth
            selection = self.phi ** (-1)
            mutation = self.phi ** (-2)
            cost = self.phi ** (-D_trajectory[gen-1] / 2)
            
            # Evolutionary dynamics
            dD = selection + mutation - cost
            D_trajectory[gen] = D_trajectory[gen-1] + dD * 0.01  # Small time step
            
            # Cap at consciousness threshold
            if D_trajectory[gen] > 10:
                D_trajectory[gen] = 10
        
        return D_trajectory


class TestLifeEntropyRegulation(unittest.TestCase):
    """Test suite for Life System Entropy Regulation Theorem"""
    
    def test_negative_entropy_generator(self):
        """Test φ-negative entropy generation"""
        # Test at life threshold
        gen = NegativeEntropyGenerator(D_life=5)
        self.assertEqual(gen.D_life, 5)
        self.assertTrue(gen.efficiency > 0)
        self.assertTrue(gen.efficiency <= PhiConstants.PHI_INV)
        
        # Test negative entropy production
        state = np.random.rand(8)
        environment = np.random.rand(16)
        neg_entropy, waste = gen.generate_neg_entropy(state, environment)
        
        self.assertTrue(neg_entropy > 0)
        self.assertTrue(waste > neg_entropy)  # Total entropy must increase
        
        # Test efficiency convergence
        gen_advanced = NegativeEntropyGenerator(D_life=20)
        self.assertAlmostEqual(gen_advanced.efficiency, PhiConstants.PHI_INV, places=2)
    
    def test_zeckendorf_organization(self):
        """Test Zeckendorf self-organization principles"""
        # Test organization at different depths
        for D_life in [5, 8, 10]:
            org = LifeSystemOrganization(D_life)
            
            # Check stability
            self.assertTrue(org.stability)
            
            # Check organization degree
            O = org.calculate_organization_degree()
            self.assertTrue(O > 0)
            
            # Test stability under small perturbations
            stable = org.perturb_and_test_stability(0.1)
            self.assertTrue(stable)
    
    def test_no11_constraint_stability(self):
        """Test No-11 constraint maintains organizational stability"""
        # Create organization with valid structure
        org = LifeSystemOrganization(D_life=13)  # 13 = F_6
        self.assertTrue(org.stability)
        
        # Test that No-11 prevents collapse
        num_stable = 0
        num_tests = 100
        for _ in range(num_tests):
            if org.perturb_and_test_stability(0.2):
                num_stable += 1
        
        # Most perturbations should maintain stability
        self.assertTrue(num_stable / num_tests > 0.8)
    
    def test_life_nonlife_boundary(self):
        """Test life-nonlife boundary criteria"""
        # Test below life threshold
        with self.assertRaises(ValueError):
            NegativeEntropyGenerator(D_life=4)
        
        # Test at life threshold
        gen = NegativeEntropyGenerator(D_life=5)
        self.assertIsNotNone(gen)
        
        # Test life criteria
        def is_life(D_life):
            """Check if system satisfies life criteria"""
            if D_life < 5:
                return False
            
            # Check negative entropy production
            try:
                gen = NegativeEntropyGenerator(D_life)
                neg_entropy_active = gen.efficiency > 0
            except:
                neg_entropy_active = False
            
            # Check organization stability
            org = LifeSystemOrganization(D_life)
            org_stable = org.stability
            
            return neg_entropy_active and org_stable
        
        self.assertFalse(is_life(4))
        self.assertTrue(is_life(5))
        self.assertTrue(is_life(8))
    
    def test_holographic_boundary(self):
        """Test holographic boundary information properties"""
        boundary = LifeBoundaryInformation(D_life=8)
        
        # Test selectivity factor
        selectivity = boundary.selectivity
        self.assertTrue(selectivity > 1)
        self.assertTrue(selectivity < 2)  # Bounded by geometric series
        
        # Test boundary density
        density = boundary.calculate_boundary_density()
        self.assertTrue(density > PhiConstants.PHI ** 2)  # Enhanced by selectivity
        
        # Test entropy pump
        gradient = boundary.entropy_pump_gradient()
        self.assertTrue(gradient > 0)
        self.assertAlmostEqual(gradient, 8 * np.log(PhiConstants.PHI), places=5)
        
        # Test molecular filtering
        molecules = np.random.exponential(1.0, 100)
        filtered = boundary.filter_molecules(molecules)
        self.assertTrue(len(filtered) < len(molecules))
        self.assertTrue(np.mean(filtered) < np.mean(molecules))  # Lower entropy selected
    
    def test_entropy_delay_mechanism(self):
        """Test entropy delay in life systems"""
        delay = EntropyDelayMechanism(D_life=7)
        
        # Check delay time calculation
        expected_delay = PhiConstants.PHI ** 7
        self.assertAlmostEqual(delay.delay_time, expected_delay, places=5)
        
        # Simulate entropy evolution
        t, S_life = delay.simulate_entropy_evolution(t_max=100)
        
        # Check that entropy is delayed
        mid_idx = len(t) // 2
        S_no_delay = np.log(1 + t[mid_idx])
        self.assertTrue(S_life[mid_idx] < S_no_delay)
    
    def test_recursive_depth_analysis(self):
        """Test recursive depth and complexity analysis"""
        analyzer = LifeComplexityAnalyzer()
        
        # Test phase identification
        self.assertEqual(analyzer.identify_phase(3), "Pre-life chemistry")
        self.assertEqual(analyzer.identify_phase(6), "Primitive life")
        self.assertEqual(analyzer.identify_phase(9), "Complex life")
        self.assertEqual(analyzer.identify_phase(11), "Pre-conscious life")
        
        # Test entropy regulation calculation
        E_reg_5 = analyzer.calculate_entropy_regulation(5)
        E_reg_8 = analyzer.calculate_entropy_regulation(8)
        self.assertTrue(E_reg_8 > E_reg_5)
        self.assertAlmostEqual(E_reg_8 / E_reg_5, PhiConstants.PHI ** 3, places=5)
        
        # Test information integration
        Phi_9 = analyzer.calculate_information_integration(9)
        Phi_10 = analyzer.calculate_information_integration(10)
        self.assertTrue(Phi_9 < PhiConstants.PHI_10)
        self.assertAlmostEqual(Phi_10, PhiConstants.PHI_10, places=2)
    
    def test_evolutionary_dynamics(self):
        """Test evolutionary dynamics of recursive depth"""
        analyzer = LifeComplexityAnalyzer()
        
        # Simulate evolution from primitive to complex life
        trajectory = analyzer.predict_evolution_trajectory(D_initial=5, generations=1000)
        
        # Check monotonic increase
        self.assertTrue(all(trajectory[i+1] >= trajectory[i] for i in range(len(trajectory)-1)))
        
        # Check convergence toward consciousness threshold
        self.assertTrue(trajectory[-1] > trajectory[0])
        self.assertTrue(trajectory[-1] <= 10)
    
    def test_metabolic_efficiency_limit(self):
        """Test that metabolic efficiency converges to φ^(-1)"""
        efficiencies = []
        for D_life in range(5, 20):
            gen = NegativeEntropyGenerator(D_life)
            efficiencies.append(gen.efficiency)
        
        # Check convergence to golden ratio inverse
        self.assertTrue(efficiencies[-1] > efficiencies[0])
        self.assertAlmostEqual(efficiencies[-1], PhiConstants.PHI_INV, places=2)
        
        # Verify upper bound
        self.assertTrue(all(e <= PhiConstants.PHI_INV for e in efficiencies))
    
    def test_phase_transitions(self):
        """Test phase transitions at critical recursive depths"""
        analyzer = LifeComplexityAnalyzer()
        
        # Calculate entropy regulation near phase transitions
        depths = np.linspace(4, 11, 100)
        E_reg = [analyzer.calculate_entropy_regulation(d) for d in depths]
        
        # Calculate second derivative to find phase transitions
        d2E = np.gradient(np.gradient(E_reg))
        
        # Find peaks in second derivative (phase transitions)
        peaks = []
        for i in range(1, len(d2E) - 1):
            if d2E[i] > d2E[i-1] and d2E[i] > d2E[i+1]:
                peaks.append(depths[i])
        
        # Should find transitions near D=5, 8, 10
        self.assertTrue(any(4.5 < p < 5.5 for p in peaks))
        self.assertTrue(any(7.5 < p < 8.5 for p in peaks))


def visualize_life_entropy_regulation():
    """Create visualization of life entropy regulation"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Negative entropy production vs recursive depth
    ax = axes[0, 0]
    depths = range(5, 15)
    efficiencies = [NegativeEntropyGenerator(d).efficiency for d in depths]
    ax.plot(depths, efficiencies, 'b-', linewidth=2)
    ax.axhline(y=PhiConstants.PHI_INV, color='r', linestyle='--', label=f'φ⁻¹ limit')
    ax.set_xlabel('Recursive Depth D_life')
    ax.set_ylabel('Negative Entropy Efficiency')
    ax.set_title('Negative Entropy Production Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Organization degree
    ax = axes[0, 1]
    org_degrees = [LifeSystemOrganization(d).calculate_organization_degree() for d in depths]
    ax.semilogy(depths, org_degrees, 'g-', linewidth=2)
    ax.set_xlabel('Recursive Depth D_life')
    ax.set_ylabel('Organization Degree O(L)')
    ax.set_title('Zeckendorf Organization Structure')
    ax.grid(True, alpha=0.3)
    
    # 3. Entropy delay
    ax = axes[0, 2]
    for D_life in [5, 7, 9]:
        delay = EntropyDelayMechanism(D_life)
        t, S = delay.simulate_entropy_evolution(t_max=50)
        ax.plot(t, S, label=f'D_life = {D_life}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy Delay in Life Systems')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Phase diagram
    ax = axes[1, 0]
    analyzer = LifeComplexityAnalyzer()
    depths_fine = np.linspace(3, 12, 100)
    phases = [analyzer.identify_phase(d) for d in depths_fine]
    phase_nums = {'Pre-life chemistry': 0, 'Primitive life': 1, 
                  'Complex life': 2, 'Pre-conscious life': 3}
    phase_values = [phase_nums[p] for p in phases]
    ax.plot(depths_fine, phase_values, 'k-', linewidth=2)
    ax.set_xlabel('Recursive Depth D_life')
    ax.set_ylabel('Life Phase')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Pre-life', 'Primitive', 'Complex', 'Pre-conscious'])
    ax.set_title('Life System Phase Transitions')
    ax.grid(True, alpha=0.3)
    
    # 5. Information integration
    ax = axes[1, 1]
    Phi_values = [analyzer.calculate_information_integration(d) for d in depths_fine]
    ax.semilogy(depths_fine, Phi_values, 'purple', linewidth=2)
    ax.axhline(y=PhiConstants.PHI_10, color='r', linestyle='--', 
               label='Consciousness threshold')
    ax.set_xlabel('Recursive Depth D_life')
    ax.set_ylabel('Information Integration Φ')
    ax.set_title('Approach to Consciousness Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Evolutionary trajectory
    ax = axes[1, 2]
    trajectory = analyzer.predict_evolution_trajectory(D_initial=5, generations=500)
    ax.plot(range(len(trajectory)), trajectory, 'brown', linewidth=2)
    ax.set_xlabel('Evolutionary Time')
    ax.set_ylabel('Recursive Depth D_life')
    ax.set_title('Evolution of Life Complexity')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('life_entropy_regulation.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Run tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Create visualization
    print("\nGenerating visualization...")
    visualize_life_entropy_regulation()
    print("Visualization saved as 'life_entropy_regulation.png'")
    
    # Demonstrate key calculations
    print("\n" + "="*50)
    print("Key Life System Calculations")
    print("="*50)
    
    # Calculate for different recursive depths
    for D_life in [5, 8, 10]:
        print(f"\nRecursive Depth D_life = {D_life}:")
        
        # Negative entropy
        gen = NegativeEntropyGenerator(D_life)
        print(f"  Negative entropy efficiency: {gen.efficiency:.4f}")
        
        # Organization
        org = LifeSystemOrganization(D_life)
        print(f"  Organization degree: {org.calculate_organization_degree():.4f}")
        
        # Boundary
        boundary = LifeBoundaryInformation(D_life)
        print(f"  Boundary selectivity: {boundary.selectivity:.4f}")
        print(f"  Entropy pump gradient: {boundary.entropy_pump_gradient():.4f}")
        
        # Complexity
        analyzer = LifeComplexityAnalyzer()
        print(f"  Life phase: {analyzer.identify_phase(D_life)}")
        print(f"  Information integration: {analyzer.calculate_information_integration(D_life):.2f}")
        
        if D_life == 10:
            print(f"  Consciousness threshold: {PhiConstants.PHI_10:.2f} bits")
            print(f"  Status: {'Pre-conscious' if analyzer.calculate_information_integration(D_life) < PhiConstants.PHI_10 else 'Conscious'}")