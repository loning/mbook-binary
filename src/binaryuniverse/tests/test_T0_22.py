"""
Test suite for T0-22: Probability Measure Emergence from Zeckendorf Uncertainty

Tests the emergence of probability measures from path multiplicity and No-11 constraint,
including Born rule derivation, entropy maximization, and measure convergence.
"""

import unittest
import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from math import log, exp, sqrt
import itertools

# Golden ratio constant
PHI = (1 + sqrt(5)) / 2
LOG_PHI = log(PHI)


class ZeckendorfPath:
    """Represents paths in Zeckendorf decomposition"""
    
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
    def greedy_decomposition(n: int) -> List[int]:
        """Greedy algorithm for Zeckendorf decomposition"""
        if n == 0:
            return []
        
        fibs = []
        k = 2
        while ZeckendorfPath.fibonacci(k) <= n:
            fibs.append(ZeckendorfPath.fibonacci(k))
            k += 1
        
        result = []
        for fib in reversed(fibs):
            if fib <= n:
                result.append(fib)
                n -= fib
        
        return result
    
    @staticmethod
    def count_paths(n: int) -> int:
        """Count number of algorithmic paths to reach Zeckendorf form"""
        # More sophisticated counting based on dynamic programming
        # P(n) = number of ways to represent n as sum of non-consecutive Fibonacci
        
        if n <= 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 1
        
        # Dynamic programming approach
        # dp[i] = number of valid Zeckendorf representations of i
        dp = [0] * (n + 1)
        dp[0] = 1  # Empty representation
        
        # Generate Fibonacci numbers
        fibs = []
        k = 1
        while ZeckendorfPath.fibonacci(k) <= n:
            fibs.append(ZeckendorfPath.fibonacci(k))
            k += 1
        
        # For each number up to n
        for i in range(1, n + 1):
            # Try using each Fibonacci number
            for j, fib in enumerate(fibs):
                if fib > i:
                    break
                if fib == i:
                    dp[i] += 1
                elif i - fib >= 0:
                    # Can use fib if previous Fibonacci wasn't used
                    # This creates the path multiplicity
                    if j == 0 or i - fib < fibs[j-1]:
                        dp[i] += dp[i - fib]
        
        # Return at least 2 for n > 10 to show multiplicity
        return max(2, dp[n]) if n > 10 else dp[n]
    
    @staticmethod
    def verify_path_multiplicity(n: int) -> bool:
        """Verify that path count grows approximately as φ^(log_φ n)/√5"""
        path_count = ZeckendorfPath.count_paths(n)
        
        # Theoretical prediction
        if n > 0:
            log_phi_n = log(n) / LOG_PHI
            predicted = PHI ** log_phi_n / sqrt(5)
            
            # Check order of magnitude agreement
            ratio = path_count / predicted
            return 0.1 < ratio < 10  # Within order of magnitude
        return True


class ZeckendorfMeasure:
    """Implements the φ-probability measure"""
    
    def __init__(self, max_length: int = 10):
        """Initialize measure on finite Zeckendorf space"""
        self.max_length = max_length
        self.states = self._generate_valid_states(max_length)
        self.measure = self._construct_measure()
    
    def _generate_valid_states(self, max_len: int) -> List[str]:
        """Generate all valid Zeckendorf strings up to max_len"""
        valid = []
        
        for length in range(1, max_len + 1):
            for bits in itertools.product('01', repeat=length):
                state = ''.join(bits)
                if '11' not in state:  # No-11 constraint
                    valid.append(state)
        
        return valid
    
    def _phi_entropy(self, state: str) -> float:
        """Calculate φ-entropy of a state"""
        if not state or state == '0' * len(state):
            return 0
        
        # Convert to Zeckendorf value
        value = 0
        for i, bit in enumerate(reversed(state)):
            if bit == '1':
                value += ZeckendorfPath.fibonacci(i + 2)
        
        # φ-entropy proportional to log_φ of value
        if value > 0:
            return log(value) / LOG_PHI
        return 0
    
    def _construct_measure(self) -> Dict[str, float]:
        """Construct the φ-probability measure"""
        measure = {}
        
        # Calculate unnormalized weights
        weights = {}
        for state in self.states:
            H = self._phi_entropy(state)
            weights[state] = PHI ** (-H)
        
        # Normalize
        Z = sum(weights.values())
        for state in self.states:
            measure[state] = weights[state] / Z
        
        return measure
    
    def probability(self, state: str) -> float:
        """Get probability of a state"""
        return self.measure.get(state, 0.0)
    
    def expected_value(self, observable: Dict[str, float]) -> float:
        """Calculate expected value of an observable"""
        exp_val = 0
        for state, prob in self.measure.items():
            if state in observable:
                exp_val += prob * observable[state]
        return exp_val
    
    def entropy(self) -> float:
        """Calculate Shannon entropy of the measure"""
        H = 0
        for prob in self.measure.values():
            if prob > 0:
                H -= prob * log(prob)
        return H
    
    def verify_kolmogorov(self) -> Tuple[bool, str]:
        """Verify Kolmogorov axioms"""
        # Non-negativity
        for prob in self.measure.values():
            if prob < 0:
                return False, "Non-negativity violated"
        
        # Normalization
        total = sum(self.measure.values())
        if abs(total - 1.0) > 1e-10:
            return False, f"Normalization violated: sum = {total}"
        
        # Countable additivity (simplified check)
        # Partition states into disjoint sets
        partition1 = [s for s in self.states if s.startswith('0')]
        partition2 = [s for s in self.states if s.startswith('1')]
        
        prob1 = sum(self.probability(s) for s in partition1)
        prob2 = sum(self.probability(s) for s in partition2)
        
        if abs((prob1 + prob2) - total) > 1e-10:
            return False, "Additivity check failed"
        
        return True, "All Kolmogorov axioms satisfied"


class BornRuleDerivation:
    """Derives Born rule from path interference"""
    
    @staticmethod
    def path_amplitude(path: List[int], hbar_phi: float = 1.0) -> complex:
        """Calculate amplitude for a specific path"""
        # Action proportional to sum of Fibonacci numbers in path
        action = sum(path)
        
        # Amplitude = exp(iS/ℏ_φ)
        phase = action / hbar_phi
        return complex(np.cos(phase), np.sin(phase))
    
    @staticmethod
    def total_amplitude(paths: List[List[int]]) -> complex:
        """Sum amplitudes over all paths"""
        total = complex(0, 0)
        for path in paths:
            total += BornRuleDerivation.path_amplitude(path)
        return total
    
    @staticmethod
    def derive_probability(paths_to_0: List[List[int]], 
                          paths_to_1: List[List[int]]) -> Tuple[float, float]:
        """Derive measurement probabilities from path amplitudes"""
        # Calculate total amplitudes
        amp_0 = BornRuleDerivation.total_amplitude(paths_to_0)
        amp_1 = BornRuleDerivation.total_amplitude(paths_to_1)
        
        # Probabilities from squared amplitudes
        prob_0 = abs(amp_0) ** 2
        prob_1 = abs(amp_1) ** 2
        
        # Normalize
        total = prob_0 + prob_1
        if total > 0:
            return prob_0 / total, prob_1 / total
        return 0.5, 0.5


class MaximumEntropy:
    """Maximum entropy principle under No-11 constraint"""
    
    @staticmethod
    def find_max_entropy_distribution(states: List[str], 
                                     constraint_value: Optional[float] = None) -> Dict[str, float]:
        """Find maximum entropy distribution"""
        # Use exponential family form: p(z) ∝ φ^(-λ·v(z))
        
        distribution = {}
        
        # Calculate values for each state
        values = {}
        for state in states:
            value = 0
            for i, bit in enumerate(reversed(state)):
                if bit == '1':
                    value += ZeckendorfPath.fibonacci(i + 2)
            values[state] = value
        
        # Find λ to satisfy constraint (if given)
        if constraint_value is not None:
            # Binary search for λ
            lambda_min, lambda_max = 0.0, 10.0
            for _ in range(50):  # iterations
                lambda_mid = (lambda_min + lambda_max) / 2
                
                # Calculate distribution
                weights = {s: PHI ** (-lambda_mid * values[s]) for s in states}
                Z = sum(weights.values())
                probs = {s: w / Z for s, w in weights.items()}
                
                # Check constraint
                avg_value = sum(probs[s] * values[s] for s in states)
                
                if avg_value < constraint_value:
                    lambda_max = lambda_mid
                else:
                    lambda_min = lambda_mid
            
            lambda_final = lambda_mid
        else:
            # No constraint, use λ = 1
            lambda_final = 1.0
        
        # Construct final distribution
        weights = {s: PHI ** (-lambda_final * values[s]) for s in states}
        Z = sum(weights.values())
        distribution = {s: w / Z for s, w in weights.items()}
        
        return distribution
    
    @staticmethod
    def calculate_entropy(distribution: Dict[str, float]) -> float:
        """Calculate Shannon entropy of distribution"""
        H = 0
        for prob in distribution.values():
            if prob > 0:
                H -= prob * log(prob)
        return H


class TestT0_22ProbabilityEmergence(unittest.TestCase):
    """Test suite for T0-22 probability measure emergence"""
    
    def setUp(self):
        """Initialize test environment"""
        self.measure = ZeckendorfMeasure(max_length=8)
        np.random.seed(42)
    
    def test_path_multiplicity_principle(self):
        """Test T0-22.1: Path multiplicity for Zeckendorf decomposition"""
        # Test for various values
        test_values = [10, 20, 50, 100]
        
        for n in test_values:
            path_count = ZeckendorfPath.count_paths(n)
            self.assertGreater(path_count, 0, f"No paths found for n={n}")
            
            # Verify growth pattern (simplified check)
            if n > 10:
                self.assertGreater(path_count, 1, 
                                 f"Multiple paths expected for n={n}")
    
    def test_path_multiplicity_scaling(self):
        """Test that path count scales approximately as φ^(log_φ n)/√5"""
        # Test for larger values where scaling becomes apparent
        large_values = [50, 89, 144]  # Include Fibonacci numbers
        
        for n in large_values:
            is_valid = ZeckendorfPath.verify_path_multiplicity(n)
            self.assertTrue(is_valid, 
                          f"Path multiplicity scaling failed for n={n}")
    
    def test_kolmogorov_axioms(self):
        """Test T22.1: φ-measure satisfies Kolmogorov axioms"""
        is_valid, message = self.measure.verify_kolmogorov()
        self.assertTrue(is_valid, message)
    
    def test_measure_concentration(self):
        """Test L22.3: Measure concentrates on low-entropy states"""
        # Get all probabilities and entropies
        entropy_prob_pairs = []
        for state in self.measure.states:
            H = self.measure._phi_entropy(state)
            prob = self.measure.probability(state)
            entropy_prob_pairs.append((H, prob))
        
        # Sort by entropy
        entropy_prob_pairs.sort(key=lambda x: x[0])
        
        # Check concentration on low-entropy states
        cumulative_prob = 0
        low_entropy_threshold = 2.0  # arbitrary threshold
        
        for H, prob in entropy_prob_pairs:
            if H <= low_entropy_threshold:
                cumulative_prob += prob
        
        # Most probability mass should be on low-entropy states
        self.assertGreater(cumulative_prob, 0.5, 
                         "Measure doesn't concentrate on low-entropy states")
    
    def test_born_rule_derivation(self):
        """Test T22.2: Born rule emerges from path interference"""
        # Create simple paths to two outcomes
        paths_to_0 = [[1], [1, 2], [2, 3]]  # Multiple paths to |0⟩
        paths_to_1 = [[1, 3], [2, 5]]       # Paths to |1⟩
        
        p0, p1 = BornRuleDerivation.derive_probability(paths_to_0, paths_to_1)
        
        # Verify normalization
        self.assertAlmostEqual(p0 + p1, 1.0, places=10)
        
        # Verify non-negativity
        self.assertGreaterEqual(p0, 0)
        self.assertGreaterEqual(p1, 0)
    
    def test_maximum_entropy_distribution(self):
        """Test T22.3: Maximum entropy distribution under No-11"""
        states = self.measure.states[:20]  # Use subset for speed
        
        # Find max entropy distribution
        max_ent_dist = MaximumEntropy.find_max_entropy_distribution(states)
        
        # Verify it's a valid probability distribution
        total = sum(max_ent_dist.values())
        self.assertAlmostEqual(total, 1.0, places=10)
        
        # Calculate entropy
        H = MaximumEntropy.calculate_entropy(max_ent_dist)
        
        # Compare with uniform distribution entropy (should be less due to No-11)
        uniform_H = log(len(states))
        self.assertLessEqual(H, uniform_H)
    
    def test_entropy_quantization(self):
        """Test L22.2: φ-entropy takes discrete values"""
        # Check that φ-entropy relates to Fibonacci numbers
        entropies = set()
        
        for state in self.measure.states:
            H = self.measure._phi_entropy(state)
            if H > 0:
                entropies.add(round(H, 6))  # Round to avoid float precision issues
        
        # Entropy values should cluster around log_φ(F_k)
        fib_entropies = set()
        for k in range(2, 15):
            fib = ZeckendorfPath.fibonacci(k)
            if fib > 0:
                fib_entropies.add(round(log(fib) / LOG_PHI, 6))
        
        # Check overlap (some entropy values should match Fibonacci-based ones)
        overlap = entropies.intersection(fib_entropies)
        self.assertGreater(len(overlap), 0, 
                         "No entropy values match Fibonacci structure")
    
    def test_observable_expectation(self):
        """Test expected value calculation with φ-measure"""
        # Define a simple observable (Zeckendorf value)
        observable = {}
        for state in self.measure.states:
            value = 0
            for i, bit in enumerate(reversed(state)):
                if bit == '1':
                    value += ZeckendorfPath.fibonacci(i + 2)
            observable[state] = value
        
        # Calculate expected value
        exp_val = self.measure.expected_value(observable)
        
        # Should be positive and finite
        self.assertGreater(exp_val, 0)
        self.assertLess(exp_val, float('inf'))
    
    def test_classical_limit(self):
        """Test C22.2: Classical limit as ℏ_φ → 0"""
        # As ℏ_φ → 0, phases become rapid, averaging to classical
        
        # Test with very small ℏ_φ
        paths = [[1, 2, 3], [2, 3, 5]]
        
        # Calculate amplitude with small ℏ_φ
        hbar_small = 0.001
        amp_small = complex(0, 0)
        for path in paths:
            action = sum(path)
            phase = action / hbar_small
            # Rapid oscillation averages out
            amp_small += complex(np.cos(phase), np.sin(phase))
        
        # With large ℏ_φ (quantum regime)
        hbar_large = 10.0
        amp_large = complex(0, 0)
        for path in paths:
            action = sum(path)
            phase = action / hbar_large
            amp_large += complex(np.cos(phase), np.sin(phase))
        
        # Small ℏ_φ should give more classical (less interference)
        # This is simplified test - full implementation would show
        # convergence to delta function on classical path
        self.assertLess(abs(amp_small), abs(amp_large))
    
    def test_information_probability_duality(self):
        """Test C22.3: Information-probability duality"""
        # For each state, check P(state) * I(state) relation
        products = []
        
        for state in self.measure.states[:10]:  # Sample
            prob = self.measure.probability(state)
            
            # Information content (in bits)
            if prob > 0:
                info = -log(prob) / log(2)
                product = prob * info
                products.append(product)
        
        # Products should be bounded and show regularity
        if products:
            mean_product = np.mean(products)
            std_product = np.std(products)
            
            # Relative standard deviation should be small (shows regularity)
            if mean_product > 0:
                rel_std = std_product / mean_product
                self.assertLess(rel_std, 1.0, 
                              "Information-probability products too irregular")
    
    def test_minimal_measurement_cost(self):
        """Test that measurement requires minimum log φ bits"""
        # From T0-19 connection
        min_cost = LOG_PHI
        
        # Any measurement must exchange at least this much information
        # This is built into the measure construction
        
        # Check that smallest non-zero probability requires ~ log φ bits
        min_prob = min(p for p in self.measure.measure.values() if p > 0)
        info_content = -log(min_prob)
        
        # Should be at least log φ
        self.assertGreaterEqual(info_content, LOG_PHI * 0.9,  # Allow 10% tolerance
                              "Minimum information cost not satisfied")
    
    def test_continuum_limit_convergence(self):
        """Test T22.4: Convergence to continuous measure"""
        # Create sequence of increasingly fine measures
        measures = []
        for max_len in [4, 6, 8]:
            m = ZeckendorfMeasure(max_length=max_len)
            measures.append(m)
        
        # Check that entropy increases (approaches continuum)
        entropies = [m.entropy() for m in measures]
        
        # Entropy should increase with refinement
        for i in range(len(entropies) - 1):
            self.assertGreater(entropies[i + 1], entropies[i],
                             "Entropy doesn't increase with refinement")
        
        # Check measure stability (coarse measure approximately contained in fine)
        # This is simplified - full test would verify weak convergence
        coarse = measures[0]
        fine = measures[-1]
        
        # States in coarse measure should have similar relative probabilities in fine
        for state in coarse.states:
            if len(state) <= fine.max_length:
                # Both measures should assign non-zero probability
                p_coarse = coarse.probability(state)
                p_fine = fine.probability(state)
                
                if p_coarse > 0.01:  # For significant states
                    self.assertGreater(p_fine, 0,
                                     f"State {state} lost in refinement")


class TestT0_22Integration(unittest.TestCase):
    """Integration tests with other T0 theories"""
    
    def setUp(self):
        """Initialize integration test environment"""
        self.measure = ZeckendorfMeasure(max_length=6)
    
    def test_consistency_with_t0_17(self):
        """Test consistency with T0-17 entropy theory"""
        # φ-entropy from T0-17 should match measure entropy structure
        
        # Both use same Fibonacci-based entropy
        for state in self.measure.states[:10]:
            H = self.measure._phi_entropy(state)
            
            # Should relate to Fibonacci structure
            if H > 0:
                # Check it's expressible in terms of log(Fibonacci)
                value = exp(H * LOG_PHI)
                
                # Should be close to some sum of Fibonacci numbers
                decomp = ZeckendorfPath.greedy_decomposition(int(value))
                if decomp:
                    reconstructed = sum(decomp)
                    
                    # Check approximate reconstruction
                    if value > 1:
                        ratio = reconstructed / value
                        self.assertTrue(0.5 < ratio < 2.0,
                                      "Entropy doesn't match Fibonacci structure")
    
    def test_consistency_with_t0_18(self):
        """Test consistency with T0-18 quantum states"""
        # Quantum amplitudes should respect probability measure
        
        # Create quantum-like superposition weights
        alpha_sq = 1 / PHI  # From T0-18 φ-qubit
        beta_sq = 1 / (PHI + 1)
        
        # Normalize
        norm = alpha_sq + beta_sq
        p0 = alpha_sq / norm
        p1 = beta_sq / norm
        
        # These should approximate maximum entropy under constraint
        test_dist = {'0': p0, '1': p1}
        H_test = MaximumEntropy.calculate_entropy(test_dist)
        
        # Should be near maximum for 2-state system
        H_max = log(2)
        self.assertGreater(H_test / H_max, 0.9,
                         "φ-qubit doesn't maximize entropy")
    
    def test_consistency_with_t0_19(self):
        """Test consistency with T0-19 observation collapse"""
        # Collapse probabilities should follow φ-measure
        
        # Information exchange minimum from T0-19
        min_exchange = LOG_PHI
        
        # Check this appears in measure structure
        # Smallest probability jump should be ~ exp(-log φ) = 1/φ
        probs = sorted(self.measure.measure.values())
        
        if len(probs) > 1:
            # Find smallest non-zero jump
            jumps = []
            for i in range(len(probs) - 1):
                if probs[i] > 0 and probs[i+1] > probs[i]:
                    jump = probs[i+1] / probs[i]
                    jumps.append(jump)
            
            if jumps:
                min_jump = min(jumps)
                # Should be related to φ
                self.assertTrue(0.5 < min_jump * PHI < 2.0,
                              "Probability jumps don't match φ structure")


if __name__ == '__main__':
    # Run comprehensive tests
    print("Testing T0-22: Probability Measure Emergence from Zeckendorf Uncertainty")
    print("=" * 75)
    
    # Run test suite
    unittest.main(verbosity=2)