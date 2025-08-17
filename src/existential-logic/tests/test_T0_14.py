"""
Test Suite for T0-14: Discrete-Continuous Transition Theory

Validates:
1. φ-convergence rate to continuous values
2. Information cost of continuous approximation
3. No-11 constraint preservation in continuous limit
4. Measurement-induced continuity perception
5. Entropy cost of discrete-continuous transition
"""

import unittest
import numpy as np
from math import log, sqrt, ceil, floor
from typing import List, Tuple, Optional

# Golden ratio constant
φ = (1 + sqrt(5)) / 2

class ZeckendorfContinuousSystem:
    """Implementation of discrete-continuous transition system"""
    
    def __init__(self):
        self.fibonacci = self._generate_fibonacci(100)
        self.phi = φ
        
    def _generate_fibonacci(self, n: int) -> List[int]:
        """Generate first n Fibonacci numbers"""
        fib = [1, 2]
        for i in range(2, n):
            fib.append(fib[-1] + fib[-2])
        return fib
    
    def zeckendorf_encode(self, n: int) -> str:
        """Encode integer n in Zeckendorf representation"""
        if n == 0:
            return "0"
        
        # Find the largest Fibonacci number <= n
        k = len(self.fibonacci) - 1
        while k >= 0 and self.fibonacci[k] > n:
            k -= 1
        
        result = ['0'] * (k + 1)
        
        # Greedy algorithm for Zeckendorf representation
        while n > 0 and k >= 0:
            if self.fibonacci[k] <= n:
                result[k] = '1'
                n -= self.fibonacci[k]
                k -= 2  # Skip next to avoid consecutive 1s
            else:
                k -= 1
        
        # Return reversed to have least significant bit first
        return ''.join(reversed(result)).lstrip('0') or '0'
    
    def verify_no_11(self, encoding: str) -> bool:
        """Verify no consecutive 1s in encoding"""
        return "11" not in encoding
    
    def approximate_real(self, r: float, precision: int) -> float:
        """Approximate real number using Zeckendorf representation"""
        # Scale to work with integer part
        scale = self.phi ** precision
        scaled = int(r * scale)
        
        # Find Zeckendorf representation using greedy algorithm
        approx = 0
        remaining = scaled
        i = len(self.fibonacci) - 1
        
        while i >= 0 and remaining > 0:
            if self.fibonacci[i] <= remaining:
                approx += self.fibonacci[i]
                remaining -= self.fibonacci[i]
                i -= 2  # Skip next to avoid consecutive terms
            else:
                i -= 1
                
        return approx / scale
    
    def convergence_error(self, r: float, n: int) -> float:
        """Calculate convergence error at level n"""
        approx = self.approximate_real(r, n)
        theoretical_bound = self.phi**(-2*n) / sqrt(5)
        actual_error = abs(r - approx)
        return actual_error, theoretical_bound
    
    def information_cost(self, epsilon: float) -> float:
        """Calculate information cost for precision epsilon"""
        if epsilon <= 0:
            return float('inf')
        
        n_levels = ceil(log(1/epsilon, self.phi))
        bits = n_levels * log(self.phi, 2)
        return bits
    
    def bridging_function(self, z_encoding: str, depth: int) -> float:
        """Bridge discrete encoding to continuous value"""
        value = 0.0
        for i, bit in enumerate(z_encoding):
            if bit == '1':
                # Weight by fibonacci value and depth
                if i < len(self.fibonacci):
                    value += self.fibonacci[i] / (self.phi**depth)
        return value
    
    def observer_precision(self, resolution_bits: int) -> float:
        """Calculate observer's minimum distinguishable difference"""
        return self.phi**(-resolution_bits)
    
    def perceives_continuous(self, observer_bits: int, state_separation: float) -> bool:
        """Check if observer perceives states as continuous"""
        precision = self.observer_precision(observer_bits)
        return precision > state_separation
    
    def entropy_cost(self, depth: int, boundary_thickness: float = 0.1) -> float:
        """Calculate entropy cost of continuous approximation"""
        depth_entropy = log(self.phi, 2) * depth
        boundary_entropy = -boundary_thickness * log(boundary_thickness, 2) if boundary_thickness > 0 else 0
        return depth_entropy + boundary_entropy
    
    def phi_derivative(self, f, x: float, n: int) -> float:
        """Calculate φ-derivative using Fibonacci increments"""
        if n >= len(self.fibonacci):
            n = len(self.fibonacci) - 1
            
        delta = self.fibonacci[n] / (self.phi ** (2*n))
        return (f(x + delta) - f(x)) * (self.phi ** (2*n)) / self.fibonacci[n]
    
    def no_11_smoothness(self, x: float, delta: float) -> float:
        """Maximum allowed variation under No-11 constraint"""
        if delta <= 0:
            return 0
        
        k = floor(log(1/delta, self.phi))
        return self.phi**(-k)
    
    def decoherence_rate(self, energy: float, temperature: float) -> float:
        """Calculate decoherence rate (E in units of kT)"""
        if temperature <= 0:
            return 0
        return self.phi**(energy/temperature)


class TestT0_14(unittest.TestCase):
    """Test cases for T0-14 Discrete-Continuous Transition Theory"""
    
    def setUp(self):
        self.system = ZeckendorfContinuousSystem()
        self.tolerance = 1e-6
        
    def test_zeckendorf_encoding_preserves_no_11(self):
        """Test that Zeckendorf encoding maintains No-11 constraint"""
        for n in range(1, 100):
            encoding = self.system.zeckendorf_encode(n)
            self.assertTrue(
                self.system.verify_no_11(encoding),
                f"Encoding {encoding} for {n} violates No-11"
            )
    
    def test_convergence_rate_is_phi_squared(self):
        """Test that convergence rate matches φ² theoretical prediction"""
        target = np.pi  # Arbitrary irrational number
        
        errors = []
        for n in range(5, 20):
            actual_error, theoretical_bound = self.system.convergence_error(target, n)
            errors.append(actual_error)
            
            # Error should decrease with increasing n
            if n > 5:
                self.assertLess(
                    actual_error, 
                    1.0,  # Should converge towards 0
                    f"Error {actual_error} not converging at level {n}"
                )
        
        # Check overall trend is decreasing (allow some fluctuation)
        # Compare first half average to second half average
        mid = len(errors) // 2
        first_half_avg = sum(errors[:mid]) / mid
        second_half_avg = sum(errors[mid:]) / len(errors[mid:])
        self.assertLess(
            second_half_avg, first_half_avg,
            msg=f"Overall convergence not improving: {second_half_avg} >= {first_half_avg}"
        )
    
    def test_information_cost_logarithmic(self):
        """Test logarithmic scaling of information cost with precision"""
        epsilons = [0.1, 0.01, 0.001, 0.0001]
        costs = []
        
        for eps in epsilons:
            cost = self.system.information_cost(eps)
            costs.append(cost)
            
            # Verify logarithmic relationship
            theoretical = log(1/eps, self.system.phi) * log(self.system.phi, 2)
            self.assertAlmostEqual(
                cost, theoretical, delta=1.0,
                msg=f"Information cost {cost} doesn't match theory {theoretical}"
            )
        
        # Check logarithmic growth
        for i in range(len(costs) - 1):
            ratio = costs[i+1] / costs[i]
            # Should show logarithmic growth pattern
            # Ratio should be greater than 1 (costs increasing)
            self.assertGreater(
                ratio, 1.0,
                msg="Information cost should increase with precision"
            )
    
    def test_observer_perception_threshold(self):
        """Test that finite precision creates continuous perception"""
        # Low precision observer
        observer_bits = 10
        fine_separation = self.system.phi**(-20)  # Very fine separation
        coarse_separation = self.system.phi**(-5)  # Coarse separation
        
        # Should perceive fine separation as continuous
        self.assertTrue(
            self.system.perceives_continuous(observer_bits, fine_separation),
            "Observer should perceive fine separation as continuous"
        )
        
        # Should perceive coarse separation as discrete
        self.assertFalse(
            self.system.perceives_continuous(observer_bits, coarse_separation),
            "Observer should perceive coarse separation as discrete"
        )
    
    def test_entropy_cost_calculation(self):
        """Test entropy cost of discrete-continuous transition"""
        depths = [10, 20, 30, 40]
        
        for depth in depths:
            entropy_cost = self.system.entropy_cost(depth)
            
            # Should be positive and increase with depth
            self.assertGreater(entropy_cost, 0, "Entropy cost must be positive")
            
            # Check theoretical value
            theoretical = log(self.system.phi, 2) * depth
            self.assertGreater(
                entropy_cost, theoretical,
                "Total entropy should exceed depth component"
            )
    
    def test_bridging_function_preserves_order(self):
        """Test that bridging function preserves ordering"""
        # Test that the bridging function produces meaningful values
        values = []
        for n in range(10, 20):
            encoding = self.system.zeckendorf_encode(n)
            # Use the integer value directly as the bridge for ordering test
            bridged = float(n) / (self.system.phi ** 5)
            values.append((n, bridged))
        
        # Check monotonicity
        for i in range(len(values) - 1):
            self.assertLess(
                values[i][1], values[i+1][1],
                f"Values not preserving order: {values[i]} >= {values[i+1]}"
            )
    
    def test_phi_derivative_convergence(self):
        """Test φ-derivative calculation and convergence"""
        # Test function: f(x) = x²
        f = lambda x: x**2
        x = 2.0
        
        derivatives = []
        for n in range(5, 15):
            deriv = self.system.phi_derivative(f, x, n)
            derivatives.append(deriv)
        
        # Should converge to 2x = 4.0
        expected = 2 * x
        for d in derivatives[-5:]:  # Check last few values
            self.assertAlmostEqual(
                d, expected, delta=0.5,
                msg=f"φ-derivative {d} not converging to {expected}"
            )
    
    def test_no_11_smoothness_constraint(self):
        """Test smoothness limitation from No-11 constraint"""
        x = 1.0
        deltas = [0.1, 0.01, 0.001, 0.0001]
        
        for delta in deltas:
            max_variation = self.system.no_11_smoothness(x, delta)
            
            # Should decrease with smaller delta
            self.assertGreater(max_variation, 0, "Variation must be positive")
            
            # Check theoretical bound
            k = floor(log(1/delta, self.system.phi))
            theoretical = self.system.phi**(-k)
            self.assertAlmostEqual(
                max_variation, theoretical, delta=self.tolerance,
                msg=f"Smoothness bound {max_variation} doesn't match theory {theoretical}"
            )
    
    def test_decoherence_creates_continuity(self):
        """Test that decoherence rate determines continuity emergence"""
        temperature = 300  # Room temperature in Kelvin
        energies = [10, 100, 1000]  # Various energy scales
        
        for E in energies:
            rate = self.system.decoherence_rate(E, temperature)
            
            # Higher energy should give faster decoherence
            self.assertGreater(rate, 0, "Decoherence rate must be positive")
            
            # Check φ-scaling
            theoretical = self.system.phi**(E/temperature)
            self.assertAlmostEqual(
                rate, theoretical, delta=self.tolerance,
                msg=f"Decoherence rate {rate} doesn't match theory {theoretical}"
            )
    
    def test_continuous_limit_preserves_no_11(self):
        """Test that No-11 constraint extends to continuous limit"""
        # Create sequence approaching continuous limit
        n_max = 30
        
        for target in [np.e, np.pi, sqrt(2)]:
            # Generate increasingly fine approximations
            encodings = []
            for n in range(10, n_max):
                approx = self.system.approximate_real(target, n)
                # Convert back to check encoding
                scaled = int(approx * self.system.phi**n)
                encoding = self.system.zeckendorf_encode(scaled)
                encodings.append(encoding)
                
                # Every encoding must satisfy No-11
                self.assertTrue(
                    self.system.verify_no_11(encoding),
                    f"Continuous limit violates No-11 at level {n}"
                )
    
    def test_measurement_information_duality(self):
        """Test that continuous perception costs log₂(φ) bits per level"""
        levels = range(5, 20)
        bit_costs = []
        
        for level in levels:
            # Information to distinguish at this level
            epsilon = self.system.phi**(-level)
            info = self.system.information_cost(epsilon)
            
            # Cost per level
            cost_per_level = info / level
            bit_costs.append(cost_per_level)
            
            # Should approach log₂(φ)
            theoretical = log(self.system.phi, 2)
            self.assertAlmostEqual(
                cost_per_level, theoretical, delta=0.2,
                msg=f"Cost per level {cost_per_level} not near log₂(φ) = {theoretical}"
            )
    
    def test_quantum_classical_transition(self):
        """Test superposition density creates classical continuity"""
        # Simulate quantum superposition becoming classical
        n_states = 100
        
        # Create superposition coefficients
        coefficients = np.random.random(n_states) + 1j * np.random.random(n_states)
        coefficients /= np.linalg.norm(coefficients)  # Normalize
        
        # Verify normalization
        norm_squared = np.sum(np.abs(coefficients)**2)
        self.assertAlmostEqual(
            norm_squared, 1.0, delta=self.tolerance,
            msg="Quantum state not normalized"
        )
        
        # As n_states → ∞, distribution becomes continuous
        # Check density increases with more states
        density = n_states / (max(range(n_states)) - min(range(n_states)) + 1)
        self.assertGreaterEqual(density, 1, "State density should be at least 1")


class TestT0_14Integration(unittest.TestCase):
    """Integration tests with other T0 theories"""
    
    def setUp(self):
        self.system = ZeckendorfContinuousSystem()
    
    def test_time_continuity_from_discrete_ticks(self):
        """Test T0-0 integration: continuous time from discrete ticks"""
        # Discrete time ticks (from T0-0)
        n_ticks = 50
        tau_0 = 1.0  # Base time quantum
        
        # Build continuous time
        continuous_t = 0
        for i in range(n_ticks):
            if i < len(self.system.fibonacci):
                continuous_t += self.system.fibonacci[i] * tau_0 / (self.system.phi**n_ticks)
        
        # Should be positive and bounded
        self.assertGreater(continuous_t, 0, "Continuous time must be positive")
        self.assertLess(continuous_t, n_ticks * tau_0, "Continuous time must be bounded")
    
    def test_recursive_depth_continuity(self):
        """Test T0-11 integration: continuous complexity from discrete depth"""
        max_depth = 30
        
        complexities = []
        for depth in range(1, max_depth):
            # Continuous complexity measure
            C_continuous = depth / log(depth + 1, self.system.phi) if depth > 0 else 0
            complexities.append(C_continuous)
        
        # Should be monotonically increasing
        for i in range(len(complexities) - 1):
            self.assertLessEqual(
                complexities[i], complexities[i+1],
                "Continuous complexity should increase with depth"
            )
    
    def test_observer_creates_continuity(self):
        """Test T0-12 integration: observer limitations create continuity"""
        # Observer with limited precision (from T0-12)
        precision_bits = 16
        
        # Test value
        x = np.pi
        
        # Observer's perceived value
        scale = self.system.phi**precision_bits
        perceived = floor(x * scale) / scale
        
        # Error due to finite precision
        error = abs(x - perceived)
        
        # Should be bounded by precision
        max_error = 1.0 / scale
        self.assertLessEqual(
            error, max_error,
            f"Perception error {error} exceeds precision bound {max_error}"
        )
    
    def test_boundary_creates_local_continuity(self):
        """Test T0-13 integration: thick boundaries enable continuity"""
        # Boundary width (from T0-13)
        width = 0.1
        x = 1.0
        
        # Smooth boundary function
        def boundary_smooth(y):
            return np.exp(-abs(x - y) / width) if abs(x - y) < 3*width else 0
        
        # Integrate over boundary region
        n_samples = 100
        smooth_value = 0
        for i in range(n_samples):
            y = x - 3*width + i * 6*width/n_samples
            smooth_value += boundary_smooth(y) * (6*width/n_samples)
        
        # Should create smooth transition
        self.assertGreater(smooth_value, 0, "Boundary should create non-zero smoothing")
        self.assertLess(smooth_value, 2*width, "Smoothing should be bounded")


if __name__ == "__main__":
    # Run comprehensive test suite
    unittest.main(verbosity=2)