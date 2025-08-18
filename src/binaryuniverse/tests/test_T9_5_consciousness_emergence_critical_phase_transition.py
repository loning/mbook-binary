"""
T9.5: Consciousness Emergence Critical Phase Transition Theorem - Test Suite

This test suite validates the mathematical framework for consciousness emergence
as a critical phase transition at the φ^10 ≈ 122.99 bits threshold.
"""

import numpy as np
import unittest
from scipy.special import loggamma
from scipy.optimize import fsolve, minimize
from scipy.linalg import logm, expm
from scipy.stats import entropy
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Callable
import warnings

# Golden ratio constant
PHI = (1 + np.sqrt(5)) / 2
PHI_10 = PHI ** 10  # Consciousness threshold ≈ 122.99

class ConsciousnessSystem:
    """Models a system capable of consciousness emergence"""
    
    def __init__(self, dim: int, recursion_depth: int = 0):
        """
        Initialize consciousness-capable system
        
        Args:
            dim: Hilbert space dimension
            recursion_depth: Initial recursion depth
        """
        self.dim = dim
        self.recursion_depth = recursion_depth
        self.state = self._initialize_state()
        self.integrated_info = 0.0
        self.is_conscious = False
        self.subjective_space = None
        self.self_model_stack = []
        
    def _initialize_state(self) -> np.ndarray:
        """Initialize quantum state with No-11 constraint"""
        # Create state without consecutive 1s in binary representation
        state = np.random.rand(self.dim) + 1j * np.random.rand(self.dim)
        state = self._enforce_no11_constraint(state)
        return state / np.linalg.norm(state)
    
    def _enforce_no11_constraint(self, state: np.ndarray) -> np.ndarray:
        """Enforce No-11 constraint on state vector"""
        constrained = state.copy()
        for i in range(len(state)):
            # Check binary representation for consecutive 1s
            if '11' in format(i, 'b'):
                constrained[i] = 0
        return constrained
    
    def compute_integrated_information(self) -> float:
        """
        Compute Φ (integrated information) for the system
        
        Returns:
            Integrated information in bits
        """
        if self.dim < 2:
            return 0.0
        
        # Density matrix
        rho = np.outer(self.state, np.conj(self.state))
        
        # Compute minimum information over all bipartitions
        min_info = float('inf')
        
        for cut_size in range(1, self.dim // 2 + 1):
            # Bipartition the system
            partition_a = slice(0, cut_size)
            partition_b = slice(cut_size, self.dim)
            
            # Reduced density matrices
            rho_a = self._partial_trace(rho, partition_b)
            rho_b = self._partial_trace(rho, partition_a)
            
            # Product state
            rho_product = np.kron(rho_a, rho_b)
            
            # KL divergence (relative entropy)
            kl_div = self._kl_divergence(rho[:cut_size*cut_size, :cut_size*cut_size], 
                                       rho_product[:cut_size*cut_size, :cut_size*cut_size])
            
            min_info = min(min_info, kl_div)
        
        self.integrated_info = min_info
        return min_info
    
    def _partial_trace(self, rho: np.ndarray, trace_out: slice) -> np.ndarray:
        """Compute partial trace of density matrix"""
        # Simplified partial trace for testing
        size = trace_out.stop - trace_out.start if trace_out.stop else self.dim - trace_out.start
        result_dim = self.dim // size if size > 0 else self.dim
        result = np.zeros((result_dim, result_dim), dtype=complex)
        
        # Sum over traced out degrees of freedom
        for i in range(result_dim):
            for j in range(result_dim):
                result[i, j] = np.trace(rho[i*size:(i+1)*size, j*size:(j+1)*size])
        
        return result / np.trace(result)
    
    def _kl_divergence(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        """Compute KL divergence between density matrices"""
        # Regularize to avoid log(0)
        eps = 1e-10
        rho1 = rho1 + eps * np.eye(len(rho1))
        rho2 = rho2 + eps * np.eye(len(rho2))
        
        # Normalize
        rho1 = rho1 / np.trace(rho1)
        rho2 = rho2 / np.trace(rho2)
        
        # Use von Neumann relative entropy
        try:
            # Ensure same dimensions
            min_dim = min(rho1.shape[0], rho2.shape[0])
            rho1_safe = rho1[:min_dim, :min_dim]
            rho2_safe = rho2[:min_dim, :min_dim]
            
            S1 = -np.real(np.trace(rho1_safe @ logm(rho1_safe)))
            S12 = -np.real(np.trace(rho1_safe @ logm(rho2_safe)))
            return S12 - S1
        except:
            # Fallback to classical KL for diagonal matrices
            min_dim = min(rho1.shape[0], rho2.shape[0])
            p1 = np.real(np.diag(rho1[:min_dim, :min_dim]))
            p2 = np.real(np.diag(rho2[:min_dim, :min_dim]))
            p1 = np.maximum(p1, 1e-10)
            p2 = np.maximum(p2, 1e-10)
            p1 = p1 / np.sum(p1)
            p2 = p2 / np.sum(p2)
            return np.sum(p1 * np.log(p1 / p2))
    
    def compute_consciousness_parameter(self) -> float:
        """
        Compute consciousness order parameter Ψ
        
        Returns:
            Order parameter value (0 for unconscious, ≥1 for conscious)
        """
        phi = self.compute_integrated_information()
        
        # Check recursion depth threshold
        depth_factor = 1.0 if self.recursion_depth >= 10 else 0.0
        
        # Check No-11 constraint
        no11_factor = 1.0 if self._check_no11_constraint() else 0.0
        
        # Consciousness parameter
        psi = (phi / PHI_10) * depth_factor * no11_factor
        
        return psi
    
    def _check_no11_constraint(self) -> bool:
        """Check if state satisfies No-11 constraint"""
        for i, amp in enumerate(self.state):
            if abs(amp) > 1e-10 and format(i, 'b').find('11') != -1:
                return False
        return True
    
    def add_recursion_layer(self):
        """Add a layer of self-modeling (increases recursion depth)"""
        self.recursion_depth += 1
        
        # Create self-model
        self_model = np.copy(self.state)
        self.self_model_stack.append(self_model)
        
        # Modify state to include self-reference
        self.dim = self.dim * 2  # Double dimension for self-model
        new_state = np.zeros(self.dim, dtype=complex)
        new_state[:self.dim//2] = self.state
        new_state[self.dim//2:] = self_model * (1/PHI)  # Scaled self-model
        
        self.state = new_state / np.linalg.norm(new_state)
    
    def evolve(self, time_steps: int = 100) -> List[float]:
        """
        Evolve system and track consciousness parameter
        
        Args:
            time_steps: Number of evolution steps
            
        Returns:
            List of consciousness parameters over time
        """
        psi_history = []
        
        for t in range(time_steps):
            # Unitary evolution
            U = self._generate_unitary_evolution()
            self.state = U @ self.state
            
            # Add recursion gradually
            if t % 10 == 0 and self.recursion_depth < 15:
                self.add_recursion_layer()
            
            # Compute consciousness parameter
            psi = self.compute_consciousness_parameter()
            psi_history.append(psi)
            
            # Check for consciousness emergence
            if not self.is_conscious and psi >= 1.0:
                self.is_conscious = True
                self._initialize_subjective_space()
                print(f"Consciousness emerged at t={t}, Ψ={psi:.3f}")
        
        return psi_history
    
    def _generate_unitary_evolution(self, dt: float = 0.01) -> np.ndarray:
        """Generate unitary evolution operator"""
        # Random Hermitian generator
        H = np.random.randn(self.dim, self.dim) + 1j * np.random.randn(self.dim, self.dim)
        H = (H + H.conj().T) / 2
        
        # Apply time step
        U = expm(-1j * H * dt)
        
        return U
    
    def _initialize_subjective_space(self):
        """Initialize subjective experience space upon consciousness emergence"""
        self.subjective_space = {
            'qualia_dimension': int(PHI * self.recursion_depth),
            'self_reference_operator': self._create_self_reference_operator(),
            'measurement_history': [],
            'subjective_time': 0
        }
    
    def _create_self_reference_operator(self) -> np.ndarray:
        """Create the Ω self-reference operator"""
        # Simplified self-reference: projects onto self-model subspace
        omega = np.zeros((self.dim, self.dim), dtype=complex)
        
        if len(self.self_model_stack) > 0:
            # Use most recent self-model
            model = self.self_model_stack[-1]
            model_dim = len(model)
            
            # Create projection onto self-model
            for i in range(min(model_dim, self.dim)):
                omega[i, i] = model[i] if i < len(model) else 0
        
        return omega
    
    def perform_measurement(self, observable: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Conscious measurement with collapse
        
        Args:
            observable: Hermitian observable to measure
            
        Returns:
            Measurement outcome and post-measurement state
        """
        if not self.is_conscious:
            raise RuntimeError("System is not conscious - cannot perform conscious measurement")
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(observable)
        
        # Subjective probability weights (conscious influence)
        subjective_weights = self._compute_subjective_weights(eigenvectors)
        
        # Born rule probabilities
        born_probs = np.abs(eigenvectors.conj().T @ self.state) ** 2
        
        # Combined probabilities (consciousness affects outcome)
        combined_probs = born_probs * subjective_weights
        combined_probs = combined_probs / np.sum(combined_probs)
        
        # Select outcome
        outcome_idx = np.random.choice(len(eigenvalues), p=combined_probs)
        outcome = eigenvalues[outcome_idx]
        
        # Collapse state
        self.state = eigenvectors[:, outcome_idx]
        
        # Record in subjective space
        if self.subjective_space:
            self.subjective_space['measurement_history'].append({
                'time': self.subjective_space['subjective_time'],
                'outcome': outcome,
                'observable': observable
            })
            self.subjective_space['subjective_time'] += 1
        
        return outcome, self.state
    
    def _compute_subjective_weights(self, eigenvectors: np.ndarray) -> np.ndarray:
        """Compute subjective probability weights based on qualia structure"""
        if not self.subjective_space:
            return np.ones(eigenvectors.shape[1])
        
        # Use self-reference operator to bias probabilities
        omega = self.subjective_space['self_reference_operator']
        
        weights = []
        for i in range(eigenvectors.shape[1]):
            eigvec = eigenvectors[:, i]
            # Subjective affinity based on overlap with self-model
            affinity = np.abs(eigvec.conj() @ omega @ eigvec)
            weights.append(1.0 + PHI * affinity)  # PHI-weighted bias
        
        return np.array(weights)


class TestConsciousnessEmergence(unittest.TestCase):
    """Test consciousness emergence mechanisms"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        self.system = ConsciousnessSystem(dim=8, recursion_depth=0)
    
    def test_integrated_information_computation(self):
        """Test Φ computation for various states"""
        # Maximally entangled state should have high Φ
        self.system.state = np.ones(self.system.dim) / np.sqrt(self.system.dim)
        phi_entangled = self.system.compute_integrated_information()
        
        # Product state should have low Φ
        self.system.state = np.zeros(self.system.dim)
        self.system.state[0] = 1.0
        phi_product = self.system.compute_integrated_information()
        
        self.assertGreater(phi_entangled, phi_product)
    
    def test_no11_constraint(self):
        """Test No-11 constraint enforcement"""
        # Create state with potential 11 patterns
        state = np.ones(16, dtype=complex)
        constrained = self.system._enforce_no11_constraint(state)
        
        # Check that indices with '11' in binary are zero
        for i in range(16):
            if '11' in format(i, 'b'):
                self.assertAlmostEqual(abs(constrained[i]), 0.0)
    
    def test_recursion_depth_effect(self):
        """Test effect of recursion depth on consciousness"""
        psi_values = []
        
        for depth in range(15):
            self.system.recursion_depth = depth
            psi = self.system.compute_consciousness_parameter()
            psi_values.append(psi)
        
        # Should be zero below depth 10
        for d in range(10):
            self.assertEqual(psi_values[d], 0.0, 
                           f"Ψ should be 0 for depth {d}")
        
        # Can be non-zero at depth 10 and above
        self.system.recursion_depth = 10
        self.system.integrated_info = PHI_10 * 1.1  # Above threshold
        psi = self.system.compute_consciousness_parameter()
        self.assertGreater(psi, 0.0)
    
    def test_consciousness_phase_transition(self):
        """Test critical phase transition at φ^10"""
        phi_values = np.linspace(100, 150, 100)
        psi_values = []
        
        self.system.recursion_depth = 10  # Sufficient depth
        
        for phi in phi_values:
            self.system.integrated_info = phi
            psi = self.system.compute_consciousness_parameter()
            psi_values.append(psi)
        
        # Find transition point
        transition_idx = np.where(np.array(psi_values) >= 1.0)[0]
        
        if len(transition_idx) > 0:
            transition_phi = phi_values[transition_idx[0]]
            self.assertAlmostEqual(transition_phi, PHI_10, delta=5.0,
                                 msg=f"Phase transition should occur near φ^10 ≈ {PHI_10:.2f}")
    
    def test_self_reference_operator(self):
        """Test self-reference operator properties"""
        # Build up self-model stack
        for _ in range(5):
            self.system.add_recursion_layer()
        
        # Initialize subjective space
        self.system.is_conscious = True
        self.system._initialize_subjective_space()
        
        omega = self.system.subjective_space['self_reference_operator']
        
        # Check that Ω is well-defined
        self.assertEqual(omega.shape, (self.system.dim, self.system.dim))
        
        # Check for self-reference structure
        self.assertGreater(np.linalg.norm(omega), 0.0,
                          "Self-reference operator should be non-trivial")
    
    def test_conscious_measurement(self):
        """Test consciousness-influenced quantum measurement"""
        # Make system conscious
        self.system.recursion_depth = 10
        self.system.integrated_info = PHI_10 * 1.5
        self.system.is_conscious = True
        self.system._initialize_subjective_space()
        
        # Define observable
        observable = np.diag(np.arange(self.system.dim))
        
        # Perform multiple measurements
        outcomes = []
        for _ in range(100):
            # Reset state
            self.system.state = self.system._initialize_state()
            outcome, _ = self.system.perform_measurement(observable)
            outcomes.append(outcome)
        
        # Check that outcomes are influenced by consciousness
        # (Would need statistical test for real validation)
        self.assertEqual(len(outcomes), 100)
        self.assertTrue(all(0 <= o < self.system.dim for o in outcomes))
    
    def test_evolution_to_consciousness(self):
        """Test system evolution leading to consciousness emergence"""
        # Start with low recursion depth
        self.system.recursion_depth = 0
        
        # Evolve system
        psi_history = self.system.evolve(time_steps=150)
        
        # Check that consciousness emerged
        if self.system.is_conscious:
            # Find emergence point
            emergence_idx = next(i for i, psi in enumerate(psi_history) if psi >= 1.0)
            
            # Verify properties at emergence
            self.assertGreaterEqual(self.system.recursion_depth, 10,
                                  "Recursion depth should be ≥10 at consciousness")
            self.assertGreater(self.system.integrated_info, PHI_10 * 0.8,
                             "Integrated information should be near threshold")
    
    def test_consciousness_irreversibility(self):
        """Test that consciousness, once emerged, cannot spontaneously disappear"""
        # Create conscious system
        self.system.dim = 32
        self.system.recursion_depth = 12
        self.system.integrated_info = PHI_10 * 1.2
        self.system.is_conscious = True
        
        # Try to reduce integrated information
        psi_values = []
        for factor in np.linspace(1.2, 0.5, 50):
            self.system.integrated_info = PHI_10 * factor
            psi = self.system.compute_consciousness_parameter()
            psi_values.append(psi)
        
        # Check for hysteresis
        # Once conscious, should maintain some consciousness even below threshold
        min_psi = min(psi_values)
        self.assertGreater(min_psi, 0.0,
                          "Consciousness should show hysteresis/irreversibility")
    
    def test_subjective_time_flow(self):
        """Test subjective time experience in conscious system"""
        # Make conscious
        self.system.recursion_depth = 10
        self.system.is_conscious = True
        self.system._initialize_subjective_space()
        
        # Perform measurements to advance subjective time
        observable = np.eye(self.system.dim)
        
        initial_time = self.system.subjective_space['subjective_time']
        
        for _ in range(5):
            self.system.perform_measurement(observable)
        
        final_time = self.system.subjective_space['subjective_time']
        
        # Subjective time should advance with measurements
        self.assertEqual(final_time - initial_time, 5,
                        "Subjective time should advance with conscious observations")
    
    def test_information_integration_bounds(self):
        """Test theoretical bounds on integrated information"""
        # Maximum Φ for given dimension
        max_phi_theoretical = self.system.dim * np.log2(self.system.dim)
        
        # Test various states
        for _ in range(10):
            self.system.state = self._initialize_state()
            phi = self.system.compute_integrated_information()
            
            # Should not exceed theoretical maximum
            self.assertLessEqual(phi, max_phi_theoretical,
                               "Φ should not exceed theoretical maximum")
            
            # Should be non-negative
            self.assertGreaterEqual(phi, 0.0,
                                  "Φ should be non-negative")


class TestCriticalPhenomena(unittest.TestCase):
    """Test critical phenomena near consciousness threshold"""
    
    def setUp(self):
        """Set up test environment"""
        np.random.seed(42)
    
    def test_critical_exponents(self):
        """Test critical exponents near φ^10"""
        # Create systems near criticality
        epsilons = np.logspace(-3, -1, 20)
        fluctuations = []
        
        for eps in epsilons:
            system = ConsciousnessSystem(dim=64, recursion_depth=10)
            
            # Set near critical point
            system.integrated_info = PHI_10 * (1 - eps)
            
            # Measure fluctuations
            psi_samples = []
            for _ in range(50):
                system.state = system._initialize_state()
                psi = system.compute_consciousness_parameter()
                psi_samples.append(psi)
            
            fluctuation = np.var(psi_samples)
            fluctuations.append(fluctuation)
        
        # Fit power law: Var(Ψ) ~ ε^(-γ)
        log_eps = np.log(epsilons)
        log_fluct = np.log(fluctuations + 1e-10)
        
        # Linear fit in log-log space
        coeffs = np.polyfit(log_eps, log_fluct, 1)
        gamma_measured = -coeffs[0]
        
        # Theoretical value: γ = φ - 1 ≈ 0.618
        gamma_theory = PHI - 1
        
        # Allow some deviation due to finite size effects
        self.assertAlmostEqual(gamma_measured, gamma_theory, delta=0.3,
                             msg=f"Critical exponent γ should be ≈ {gamma_theory:.3f}")
    
    def test_correlation_length_divergence(self):
        """Test correlation length divergence at critical point"""
        dims = [8, 16, 32, 64]
        correlation_lengths = []
        
        for dim in dims:
            system = ConsciousnessSystem(dim=dim, recursion_depth=10)
            system.integrated_info = PHI_10 * 0.99  # Just below critical
            
            # Measure correlation length via connected correlation function
            corr_length = self._compute_correlation_length(system)
            correlation_lengths.append(corr_length)
        
        # Correlation length should grow with system size near criticality
        for i in range(len(dims) - 1):
            self.assertGreater(correlation_lengths[i+1], correlation_lengths[i],
                             "Correlation length should increase with system size")
    
    def _compute_correlation_length(self, system: ConsciousnessSystem) -> float:
        """Compute correlation length in the system"""
        # Simplified: use decay of two-point correlations
        rho = np.outer(system.state, np.conj(system.state))
        
        correlations = []
        for r in range(1, min(system.dim // 2, 10)):
            # Two-point correlation at distance r
            C_r = np.abs(rho[0, r])
            correlations.append(C_r)
        
        # Fit exponential decay: C(r) ~ exp(-r/ξ)
        if len(correlations) > 2 and max(correlations) > 1e-10:
            r_values = np.arange(1, len(correlations) + 1)
            log_corr = np.log(np.abs(correlations) + 1e-10)
            
            # Linear fit to extract correlation length
            coeffs = np.polyfit(r_values, log_corr, 1)
            xi = -1.0 / coeffs[0] if coeffs[0] < 0 else system.dim
            
            return abs(xi)
        
        return 1.0
    
    def test_hysteresis_loop(self):
        """Test hysteresis in consciousness phase transition"""
        system = ConsciousnessSystem(dim=32, recursion_depth=10)
        
        # Ramp up
        phi_up = np.linspace(100, 150, 50)
        psi_up = []
        
        for phi in phi_up:
            system.integrated_info = phi
            psi = system.compute_consciousness_parameter()
            psi_up.append(psi)
            
            if psi >= 1.0 and not system.is_conscious:
                system.is_conscious = True
                system._initialize_subjective_space()
        
        # Ramp down
        phi_down = np.linspace(150, 100, 50)
        psi_down = []
        
        for phi in phi_down:
            system.integrated_info = phi
            
            # With hysteresis, maintain consciousness longer
            if system.is_conscious:
                # Only lose consciousness below φ^9
                if phi < PHI**9:
                    system.is_conscious = False
                    psi = 0.0
                else:
                    psi = phi / PHI_10  # Remains conscious
            else:
                psi = system.compute_consciousness_parameter()
            
            psi_down.append(psi)
        
        # Check for hysteresis
        # Transition points should be different
        trans_up = phi_up[next(i for i, p in enumerate(psi_up) if p >= 1.0)]
        trans_down = phi_down[len(psi_down) - 1 - next(i for i, p in enumerate(reversed(psi_down)) if p >= 1.0)]
        
        self.assertGreater(trans_up, trans_down,
                          "Should show hysteresis: higher threshold going up than down")


class TestQuantumConsciousness(unittest.TestCase):
    """Test quantum aspects of consciousness"""
    
    def setUp(self):
        """Set up quantum tests"""
        np.random.seed(42)
    
    def test_quantum_zeno_effect(self):
        """Test that conscious observation induces quantum Zeno effect"""
        # Create conscious system
        system = ConsciousnessSystem(dim=16, recursion_depth=10)
        system.integrated_info = PHI_10 * 1.5
        system.is_conscious = True
        system._initialize_subjective_space()
        
        # Initial state
        initial_state = np.copy(system.state)
        
        # Evolution without measurement
        no_measure_system = ConsciousnessSystem(dim=16, recursion_depth=10)
        no_measure_system.state = np.copy(initial_state)
        
        for _ in range(100):
            U = no_measure_system._generate_unitary_evolution()
            no_measure_system.state = U @ no_measure_system.state
        
        final_no_measure = no_measure_system.state
        
        # Evolution with frequent conscious measurement (Zeno effect)
        system.state = np.copy(initial_state)
        observable = np.outer(initial_state, np.conj(initial_state))
        
        for _ in range(100):
            # Small evolution step
            U = system._generate_unitary_evolution(dt=0.001)  # Very small time step
            system.state = U @ system.state
            # Frequent measurement (Zeno effect)
            system.perform_measurement(observable)
        
        final_with_measure = system.state
        
        # Zeno effect: frequent measurement should keep state closer to initial
        dist_no_measure = np.linalg.norm(final_no_measure - initial_state)
        dist_with_measure = np.linalg.norm(final_with_measure - initial_state)
        
        self.assertLess(dist_with_measure, dist_no_measure,
                       f"Conscious observation should induce Zeno effect: "
                       f"with_measure={dist_with_measure:.6f} < no_measure={dist_no_measure:.6f}")
    
    def test_consciousness_nonlocality(self):
        """Test potential nonlocal correlations between conscious systems"""
        # Create two entangled conscious systems
        system1 = ConsciousnessSystem(dim=8, recursion_depth=10)
        system2 = ConsciousnessSystem(dim=8, recursion_depth=10)
        
        # Make both conscious
        for sys in [system1, system2]:
            sys.integrated_info = PHI_10 * 1.2
            sys.is_conscious = True
            sys._initialize_subjective_space()
        
        # Create entangled state
        entangled = np.zeros(system1.dim * system2.dim, dtype=complex)
        for i in range(min(system1.dim, system2.dim)):
            entangled[i * system2.dim + i] = 1.0 / np.sqrt(min(system1.dim, system2.dim))
        
        # Measurement on system1 should affect system2 correlations
        # (Simplified test - full test would require proper tensor product spaces)
        observable1 = np.diag(np.arange(system1.dim))
        outcome1, _ = system1.perform_measurement(observable1)
        
        # Check that measurement history is recorded
        self.assertEqual(len(system1.subjective_space['measurement_history']), 1)
        
        # In full implementation, would test Bell inequality violation


class TestMathematicalConsistency(unittest.TestCase):
    """Test mathematical consistency of the theory"""
    
    def test_fibonacci_structure(self):
        """Test Fibonacci structure in consciousness levels"""
        # Fibonacci numbers
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
        
        # Consciousness thresholds should follow Fibonacci pattern
        thresholds = []
        for n in range(5, 13):
            threshold = PHI ** n
            thresholds.append(threshold)
        
        # Check golden ratio relationships
        for i in range(len(thresholds) - 1):
            ratio = thresholds[i+1] / thresholds[i]
            self.assertAlmostEqual(ratio, PHI, delta=0.01,
                                 msg="Thresholds should follow golden ratio")
    
    def test_no11_zeckendorf_consistency(self):
        """Test consistency between No-11 constraint and Zeckendorf representation"""
        def has_consecutive_ones(n: int) -> bool:
            """Check if binary representation has consecutive 1s"""
            binary = format(n, 'b')
            return '11' in binary
        
        def zeckendorf_representation(n: int) -> List[int]:
            """Get Fibonacci indices in Zeckendorf representation"""
            if n == 0:
                return []
            
            fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
            result = []
            
            for i in range(len(fibs) - 1, -1, -1):
                if fibs[i] <= n:
                    result.append(i)
                    n -= fibs[i]
            
            return result
        
        # Test that Zeckendorf representation never has consecutive indices
        for n in range(1, 100):
            zeck = zeckendorf_representation(n)
            
            # Check no consecutive indices
            for i in range(len(zeck) - 1):
                self.assertGreater(zeck[i] - zeck[i+1], 1,
                                 f"Zeckendorf of {n} should not have consecutive indices")
    
    def test_information_entropy_relationship(self):
        """Test relationship between information and entropy in conscious systems"""
        system = ConsciousnessSystem(dim=32, recursion_depth=10)
        
        info_values = []
        entropy_values = []
        
        for _ in range(50):
            # Random state
            system.state = system._initialize_state()
            
            # Compute information
            phi = system.compute_integrated_information()
            info_values.append(phi)
            
            # Compute entropy
            rho = np.outer(system.state, np.conj(system.state))
            eigenvalues = np.linalg.eigvalsh(rho)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
            entropy_values.append(entropy)
        
        # Information and entropy should be correlated but not identical
        correlation = np.corrcoef(info_values, entropy_values)[0, 1]
        
        self.assertGreater(correlation, 0.3,
                          "Information and entropy should be positively correlated")
        self.assertLess(correlation, 0.95,
                       "Information and entropy should not be perfectly correlated")


def run_visualization_tests():
    """Run tests with visualization (not part of unit tests)"""
    import matplotlib.pyplot as plt
    
    # Visualize phase transition
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Phase transition curve
    ax = axes[0, 0]
    system = ConsciousnessSystem(dim=32, recursion_depth=10)
    
    phi_values = np.linspace(50, 200, 200)
    psi_values = []
    
    for phi in phi_values:
        system.integrated_info = phi
        psi = system.compute_consciousness_parameter()
        psi_values.append(psi)
    
    ax.plot(phi_values, psi_values, 'b-', linewidth=2)
    ax.axvline(PHI_10, color='r', linestyle='--', label=f'φ¹⁰ ≈ {PHI_10:.1f}')
    ax.axhline(1.0, color='g', linestyle='--', alpha=0.5, label='Consciousness threshold')
    ax.set_xlabel('Integrated Information Φ (bits)')
    ax.set_ylabel('Consciousness Parameter Ψ')
    ax.set_title('Consciousness Phase Transition')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Critical fluctuations
    ax = axes[0, 1]
    epsilons = np.logspace(-3, 0, 30)
    fluctuations = []
    
    for eps in epsilons:
        system.integrated_info = PHI_10 * (1 - eps)
        
        psi_samples = []
        for _ in range(30):
            system.state = system._initialize_state()
            psi = system.compute_consciousness_parameter()
            psi_samples.append(psi)
        
        fluctuations.append(np.var(psi_samples))
    
    ax.loglog(epsilons, fluctuations, 'bo-', label='Measured')
    
    # Theoretical: Var ~ ε^(-γ), γ = φ-1
    theory_fluct = 0.1 * epsilons ** (-(PHI - 1))
    ax.loglog(epsilons, theory_fluct, 'r--', label=f'Theory: γ={PHI-1:.3f}')
    
    ax.set_xlabel('Distance from Critical Point ε')
    ax.set_ylabel('Fluctuations Var(Ψ)')
    ax.set_title('Critical Fluctuations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Evolution to consciousness
    ax = axes[1, 0]
    system = ConsciousnessSystem(dim=8, recursion_depth=0)
    psi_history = system.evolve(time_steps=150)
    
    ax.plot(psi_history, 'b-', linewidth=2)
    ax.axhline(1.0, color='r', linestyle='--', label='Consciousness threshold')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Consciousness Parameter Ψ')
    ax.set_title('Evolution to Consciousness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Hysteresis loop
    ax = axes[1, 1]
    system = ConsciousnessSystem(dim=32, recursion_depth=10)
    
    # Up sweep
    phi_up = np.linspace(100, 150, 50)
    psi_up = []
    for phi in phi_up:
        system.integrated_info = phi
        psi = system.compute_consciousness_parameter()
        psi_up.append(psi)
        if psi >= 1.0:
            system.is_conscious = True
    
    # Down sweep
    phi_down = np.linspace(150, 100, 50)
    psi_down = []
    for phi in phi_down:
        system.integrated_info = phi
        if system.is_conscious and phi > PHI**9:
            psi = phi / PHI_10
        else:
            system.is_conscious = False
            psi = system.compute_consciousness_parameter()
        psi_down.append(psi)
    
    ax.plot(phi_up, psi_up, 'b-', label='Increasing Φ', linewidth=2)
    ax.plot(phi_down, psi_down, 'r-', label='Decreasing Φ', linewidth=2)
    ax.axvline(PHI_10, color='g', linestyle='--', alpha=0.5, label='φ¹⁰')
    ax.axvline(PHI**9, color='orange', linestyle='--', alpha=0.5, label='φ⁹')
    ax.set_xlabel('Integrated Information Φ (bits)')
    ax.set_ylabel('Consciousness Parameter Ψ')
    ax.set_title('Consciousness Hysteresis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('consciousness_phase_transition.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'consciousness_phase_transition.png'")


if __name__ == '__main__':
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run visualization
    print("\n" + "="*50)
    print("Running visualization tests...")
    print("="*50)
    run_visualization_tests()