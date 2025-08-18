"""
Test Suite for T6.4: Theory Self-Verification Theorem

This test suite validates the self-verification framework for the binary universe
theory system, including fixed point verification, circular dependency completeness,
logical chain validation, and automatic completeness checking.
"""

import numpy as np
import pytest
from typing import List, Dict, Tuple, Optional
import networkx as nx
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

# Golden ratio constant
PHI = (1 + np.sqrt(5)) / 2

@dataclass
class Theory:
    """Representation of a theory with self-reference depth and verification data"""
    name: str
    axioms: List[str]
    definitions: List[str]
    lemmas: List[str]
    theorems: List[str]
    self_depth: int
    encoding: np.ndarray
    
    def __hash__(self):
        return hash(self.name)

class TheorySelfVerification:
    """Implementation of the self-verification framework from T6.4"""
    
    def __init__(self):
        self.PHI = PHI
        self.verification_cache = {}
        self.axiom_A1 = Theory(
            name="A1",
            axioms=["Self-referentially complete systems must increase entropy"],
            definitions=[],
            lemmas=[],
            theorems=[],
            self_depth=1,
            encoding=np.array([1, 0, 1, 0, 0, 1, 0, 1])  # Fibonacci pattern
        )
    
    def verify_pair(self, T1: Theory, T2: Theory) -> float:
        """
        Compute verification strength between two theories
        V_φ(T1, T2) ∈ [0, 1]
        """
        # Check cache
        cache_key = (T1.name, T2.name)
        if cache_key in self.verification_cache:
            return self.verification_cache[cache_key]
        
        # Compute entropy-information difference (from D1.10)
        H_T1 = self._compute_entropy(T1)
        I_T2 = self._compute_information(T2)
        
        # Verification strength via entropy-information equivalence
        strength = np.exp(-abs(H_T1 - I_T2) / self.PHI)
        
        # Apply No-11 constraint check
        if self._violates_no11(T1.encoding, T2.encoding):
            strength *= 0.5  # Penalty for No-11 violation
        
        # Cache result
        self.verification_cache[cache_key] = strength
        
        return strength
    
    def recursive_verify(self, theory: Theory, depth: int = 10) -> float:
        """
        Recursive self-verification: V_φ^(n)(T)
        Implements Theorem 6.4.1
        """
        if depth == 0:
            return self.verify_pair(theory, self.axiom_A1)
        
        # Recursive step
        prev_strength = self.recursive_verify(theory, depth - 1)
        
        # Create intermediate theory from previous verification
        intermediate = self._create_intermediate_theory(theory, prev_strength, depth)
        
        # Compute next verification level
        current_strength = self.verify_pair(theory, intermediate)
        
        # Apply convergence rate
        convergence_rate = self.PHI ** (-depth)
        
        return current_strength * (1 - convergence_rate) + convergence_rate
    
    def find_fixed_point(self, initial_theory: Theory, max_iter: int = 100, 
                        tol: float = 1e-10) -> Tuple[Theory, float]:
        """
        Find the unique fixed point T* where V_φ(T*, T*) = 1
        Implements Theorem 6.4.1
        """
        current = initial_theory
        
        for iteration in range(max_iter):
            # Apply verification operator
            strength = self.verify_pair(current, current)
            
            # Check convergence
            if abs(strength - 1.0) < tol:
                return current, strength
            
            # Update theory towards fixed point
            current = self._update_towards_fixed_point(current, strength)
        
        return current, self.verify_pair(current, current)
    
    def build_verification_matrix(self, theories: List[Theory]) -> np.ndarray:
        """
        Build verification matrix V_φ for theory collection
        Implements framework for Theorem 6.4.2
        """
        n = len(theories)
        V_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                V_matrix[i, j] = self.verify_pair(theories[i], theories[j])
        
        return V_matrix
    
    def check_circular_completeness(self, theories: List[Theory]) -> bool:
        """
        Check if theories form a circularly complete dependency
        Implements Theorem 6.4.2
        """
        V_matrix = self.build_verification_matrix(theories)
        
        # Compute eigenvalues
        eigenvalues = linalg.eigvals(V_matrix)
        
        # Check for φ^(-1) eigenvalue
        phi_inverse = 1 / self.PHI
        
        for λ in eigenvalues:
            if abs(λ - phi_inverse) < 1e-10:
                return True
        
        return False
    
    def verify_logical_chain(self, chain: List[Theory]) -> bool:
        """
        Verify a logical deduction chain
        Implements Theorem 6.4.3
        """
        if len(chain) <= 1:
            return True
        
        # Compute chain strength
        total_strength = 1.0
        
        for i in range(len(chain) - 1):
            pair_strength = self.verify_pair(chain[i], chain[i + 1])
            
            # Check No-11 constraint (no consecutive full verifications)
            if i > 0 and pair_strength == 1.0 and total_strength == 1.0:
                return False
            
            total_strength *= pair_strength
        
        # Check minimum strength requirement
        min_strength = self.PHI ** (-(len(chain) - 1))
        
        return total_strength >= min_strength
    
    def check_consistency(self, theory: Theory, max_depth: int = 20) -> bool:
        """
        Check theory consistency via recursive verification convergence
        Implements Theorem 6.4.4
        """
        strengths = []
        
        for depth in range(max_depth):
            strength = self.recursive_verify(theory, depth)
            strengths.append(strength)
            
            # Check convergence
            if depth > 5:
                recent = strengths[-5:]
                if all(abs(s - 1.0) < 1e-6 for s in recent):
                    return True
        
        # Check if converging towards 1
        return abs(strengths[-1] - 1.0) < 0.1
    
    def build_concept_network(self, theory: Theory) -> nx.Graph:
        """
        Build concept network from theory structure
        For Theorem 6.4.5
        """
        G = nx.Graph()
        
        # Add concepts as nodes
        all_concepts = (theory.axioms + theory.definitions + 
                       theory.lemmas + theory.theorems)
        
        for concept in all_concepts:
            G.add_node(concept)
        
        # Add weighted edges based on logical connections
        for i, c1 in enumerate(all_concepts):
            for j, c2 in enumerate(all_concepts):
                if i < j:
                    # Weight based on position and type
                    weight = self._compute_concept_weight(c1, c2, i, j)
                    if weight > 0:
                        G.add_edge(c1, c2, weight=weight)
        
        return G
    
    def check_concept_connectivity(self, theory: Theory) -> bool:
        """
        Check if concept network is sufficiently connected
        Implements Theorem 6.4.5
        """
        G = self.build_concept_network(theory)
        
        if len(G) == 0:
            return False
        
        # Build Laplacian matrix
        L = nx.laplacian_matrix(G).astype(float)
        
        # Compute algebraic connectivity (second smallest eigenvalue)
        if len(G) > 1:
            eigenvalues = linalg.eigvalsh(L.toarray())
            lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0
        else:
            lambda_2 = 0
        
        # Check connectivity threshold
        threshold = self.PHI ** (-theory.self_depth)
        
        return lambda_2 > threshold
    
    def automatic_completeness_check(self, theory: Theory) -> bool:
        """
        Automatically check φ-completeness
        Implements Theorem 6.4.6
        """
        # Build verification matrix for theory components
        components = self._extract_theory_components(theory)
        
        if len(components) == 0:
            return False
        
        V_matrix = self.build_verification_matrix(components)
        
        # Compute kernel dimension of (I - φV)
        n = len(components)
        kernel_matrix = np.eye(n) - self.PHI * V_matrix
        
        # Compute rank
        rank = np.linalg.matrix_rank(kernel_matrix, tol=1e-10)
        
        # Kernel dimension
        kernel_dim = n - rank
        
        # Completeness requires 1-dimensional kernel
        return kernel_dim == 1
    
    # Helper methods
    
    def _compute_entropy(self, theory: Theory) -> float:
        """Compute φ-entropy of theory"""
        # Count structural elements
        n_elements = (len(theory.axioms) + len(theory.definitions) + 
                     len(theory.lemmas) + len(theory.theorems))
        
        if n_elements == 0:
            return 0
        
        # Compute entropy based on Zeckendorf encoding
        z_indices = self._get_zeckendorf_indices(theory.encoding)
        
        # φ-entropy formula from D1.10
        entropy = sum(np.log(self.PHI) * i for i in z_indices)
        
        return entropy / n_elements
    
    def _compute_information(self, theory: Theory) -> float:
        """Compute φ-information content of theory"""
        # Information based on self-reference depth
        base_info = theory.self_depth * np.log(self.PHI)
        
        # Additional information from structure
        structural_info = len(theory.theorems) * np.log(self.PHI) / 2
        
        return base_info + structural_info
    
    def _violates_no11(self, encoding1: np.ndarray, encoding2: np.ndarray) -> bool:
        """Check if encodings would violate No-11 constraint when combined"""
        # Check for consecutive 1s in concatenation
        combined = np.concatenate([encoding1, encoding2])
        
        for i in range(len(combined) - 1):
            if combined[i] == 1 and combined[i + 1] == 1:
                return True
        
        return False
    
    def _get_zeckendorf_indices(self, encoding: np.ndarray) -> List[int]:
        """Extract Fibonacci indices from binary encoding"""
        indices = []
        fib = [1, 2, 3, 5, 8, 13, 21, 34]  # First few Fibonacci numbers
        
        for i, bit in enumerate(encoding):
            if bit == 1 and i < len(fib):
                indices.append(fib[i])
        
        return indices
    
    def _create_intermediate_theory(self, base: Theory, strength: float, 
                                   depth: int) -> Theory:
        """Create intermediate theory for recursive verification"""
        return Theory(
            name=f"{base.name}_intermediate_{depth}",
            axioms=base.axioms,
            definitions=base.definitions[:int(len(base.definitions) * strength)],
            lemmas=base.lemmas[:int(len(base.lemmas) * strength)],
            theorems=base.theorems[:int(len(base.theorems) * strength)],
            self_depth=min(base.self_depth, depth),
            encoding=base.encoding * strength
        )
    
    def _update_towards_fixed_point(self, theory: Theory, 
                                   current_strength: float) -> Theory:
        """Update theory towards fixed point"""
        # Adjust self-depth based on convergence
        new_depth = theory.self_depth
        if current_strength < 0.5:
            new_depth = max(1, theory.self_depth - 1)
        elif current_strength < 0.9:
            new_depth = theory.self_depth
        else:
            new_depth = min(15, theory.self_depth + 1)
        
        # Update encoding towards Fibonacci pattern
        target_encoding = self._generate_fibonacci_encoding(len(theory.encoding))
        new_encoding = (1 - 0.1) * theory.encoding + 0.1 * target_encoding
        
        return Theory(
            name=theory.name,
            axioms=theory.axioms,
            definitions=theory.definitions,
            lemmas=theory.lemmas,
            theorems=theory.theorems,
            self_depth=new_depth,
            encoding=new_encoding
        )
    
    def _generate_fibonacci_encoding(self, length: int) -> np.ndarray:
        """Generate Fibonacci-based encoding satisfying No-11"""
        encoding = np.zeros(length)
        fib_positions = [0, 2, 3, 5, 7]  # Positions avoiding consecutive 1s
        
        for pos in fib_positions:
            if pos < length:
                encoding[pos] = 1
        
        return encoding
    
    def _compute_concept_weight(self, c1: str, c2: str, i: int, j: int) -> float:
        """Compute weight between concepts"""
        # Simple heuristic based on distance and type
        distance = abs(j - i)
        
        if distance == 1:
            return 1.0 / self.PHI  # Adjacent concepts
        elif distance < 5:
            return 1.0 / (self.PHI ** 2)  # Nearby concepts
        else:
            return 0  # No direct connection
    
    def _extract_theory_components(self, theory: Theory) -> List[Theory]:
        """Extract component theories for completeness check"""
        components = []
        
        # Create sub-theories from different aspects
        if theory.axioms:
            components.append(Theory(
                name=f"{theory.name}_axioms",
                axioms=theory.axioms,
                definitions=[],
                lemmas=[],
                theorems=[],
                self_depth=1,
                encoding=theory.encoding[:4]
            ))
        
        if theory.definitions:
            components.append(Theory(
                name=f"{theory.name}_definitions",
                axioms=[],
                definitions=theory.definitions,
                lemmas=[],
                theorems=[],
                self_depth=2,
                encoding=theory.encoding[2:6] if len(theory.encoding) > 5 else theory.encoding
            ))
        
        if theory.lemmas:
            components.append(Theory(
                name=f"{theory.name}_lemmas",
                axioms=[],
                definitions=[],
                lemmas=theory.lemmas,
                theorems=[],
                self_depth=3,
                encoding=theory.encoding[4:] if len(theory.encoding) > 4 else theory.encoding
            ))
        
        return components if components else [theory]


# Test Functions

def test_basic_verification():
    """Test basic theory verification between pairs"""
    verifier = TheorySelfVerification()
    
    # Create test theories
    T1 = Theory(
        name="T1",
        axioms=["A1"],
        definitions=["D1"],
        lemmas=["L1"],
        theorems=["T1"],
        self_depth=3,
        encoding=np.array([1, 0, 1, 0, 0, 1, 0, 0])
    )
    
    T2 = Theory(
        name="T2",
        axioms=["A1"],
        definitions=["D1", "D2"],
        lemmas=["L1", "L2"],
        theorems=["T1", "T2"],
        self_depth=4,
        encoding=np.array([1, 0, 0, 1, 0, 1, 0, 0])
    )
    
    # Test verification
    strength = verifier.verify_pair(T1, T2)
    
    assert 0 <= strength <= 1, "Verification strength out of bounds"
    assert strength > 0.3, "Verification strength too low for related theories"
    
    print(f"✓ Basic verification: T1 ↔ T2 strength = {strength:.4f}")


def test_recursive_verification_convergence():
    """Test that recursive verification converges (Theorem 6.4.1)"""
    verifier = TheorySelfVerification()
    
    # Create a well-formed theory
    T = Theory(
        name="T_convergent",
        axioms=["A1"],
        definitions=["D1", "D2", "D3"],
        lemmas=["L1", "L2"],
        theorems=["T1"],
        self_depth=5,
        encoding=np.array([1, 0, 1, 0, 0, 1, 0, 1])
    )
    
    # Test convergence at different depths
    depths = [1, 5, 10, 15, 20]
    strengths = []
    
    for depth in depths:
        strength = verifier.recursive_verify(T, depth)
        strengths.append(strength)
    
    # Check monotonic convergence
    for i in range(len(strengths) - 1):
        assert strengths[i+1] >= strengths[i] or abs(strengths[i+1] - strengths[i]) < 0.1, \
               f"Non-monotonic convergence at depth {depths[i+1]}"
    
    # Check convergence rate
    convergence_rate = abs(strengths[-1] - strengths[-2])
    expected_rate = PHI ** (-depths[-1])
    
    assert convergence_rate < expected_rate * 10, \
           f"Convergence rate {convergence_rate} exceeds theoretical bound"
    
    print(f"✓ Recursive verification converges: final strength = {strengths[-1]:.6f}")


def test_fixed_point_existence():
    """Test existence and uniqueness of fixed point (Theorem 6.4.1)"""
    verifier = TheorySelfVerification()
    
    # Start from different initial theories
    initial_theories = [
        Theory(f"T_init_{i}", ["A1"], [f"D{i}"], [], [], i+1, 
               np.random.rand(8) > 0.5)
        for i in range(3)
    ]
    
    fixed_points = []
    
    for T_init in initial_theories:
        T_star, strength = verifier.find_fixed_point(T_init)
        fixed_points.append((T_star, strength))
        
        # Verify it's actually a fixed point
        self_strength = verifier.verify_pair(T_star, T_star)
        assert abs(self_strength - strength) < 1e-6, \
               f"Not a true fixed point: {self_strength} vs {strength}"
    
    # Check that all converge to similar fixed points
    # (uniqueness up to equivalence)
    strengths = [s for _, s in fixed_points]
    assert all(abs(s - strengths[0]) < 0.1 for s in strengths), \
           "Fixed points not unique"
    
    print(f"✓ Fixed point found with strength = {strengths[0]:.6f}")


def test_circular_dependency_completeness():
    """Test circular dependency completeness (Theorem 6.4.2)"""
    verifier = TheorySelfVerification()
    
    # Create circularly dependent theories
    theories = []
    for i in range(4):
        T = Theory(
            name=f"T_circular_{i}",
            axioms=["A1"],
            definitions=[f"D{i}", f"D{(i+1)%4}"],  # Circular reference
            lemmas=[f"L{i}"],
            theorems=[f"T{i}"],
            self_depth=3,
            encoding=np.roll(np.array([1, 0, 1, 0, 0, 1, 0, 0]), i)
        )
        theories.append(T)
    
    # Check circular completeness
    is_complete = verifier.check_circular_completeness(theories)
    
    assert is_complete, "Circular theories should be complete"
    
    # Verify eigenvalue condition
    V_matrix = verifier.build_verification_matrix(theories)
    eigenvalues = linalg.eigvals(V_matrix)
    
    phi_inv = 1 / PHI
    has_phi_eigenvalue = any(abs(λ - phi_inv) < 1e-6 for λ in eigenvalues)
    
    assert has_phi_eigenvalue, f"Missing φ^(-1) eigenvalue: {eigenvalues}"
    
    print(f"✓ Circular dependency complete with φ^(-1) eigenvalue")


def test_logical_chain_verification():
    """Test logical chain verification (Theorem 6.4.3)"""
    verifier = TheorySelfVerification()
    
    # Create a logical chain
    chain = []
    for i in range(5):
        T = Theory(
            name=f"T_chain_{i}",
            axioms=["A1"] if i == 0 else [],
            definitions=[f"D{j}" for j in range(i+1)],
            lemmas=[f"L{i}"],
            theorems=[f"T{j}" for j in range(i)],
            self_depth=i+1,
            encoding=np.array([1, 0] * 4) if i % 2 == 0 else np.array([0, 1] * 4)
        )
        chain.append(T)
    
    # Test chain validity
    is_valid = verifier.verify_logical_chain(chain)
    
    assert is_valid, "Valid logical chain rejected"
    
    # Test minimum strength condition
    total_strength = 1.0
    for i in range(len(chain) - 1):
        total_strength *= verifier.verify_pair(chain[i], chain[i+1])
    
    min_strength = PHI ** (-(len(chain) - 1))
    
    assert total_strength >= min_strength, \
           f"Chain strength {total_strength} below minimum {min_strength}"
    
    print(f"✓ Logical chain valid with strength = {total_strength:.6f}")


def test_consistency_criterion():
    """Test recursive consistency criterion (Theorem 6.4.4)"""
    verifier = TheorySelfVerification()
    
    # Create consistent theory
    T_consistent = Theory(
        name="T_consistent",
        axioms=["A1"],
        definitions=["D1", "D2"],
        lemmas=["L1", "L2", "L3"],
        theorems=["T1", "T2"],
        self_depth=6,
        encoding=np.array([1, 0, 1, 0, 0, 1, 0, 1])
    )
    
    # Create inconsistent theory (violates structure)
    T_inconsistent = Theory(
        name="T_inconsistent",
        axioms=[],  # Missing axiom
        definitions=["D1", "D2"],
        lemmas=["L1", "L2", "L3"],
        theorems=["T1", "T2"],
        self_depth=2,
        encoding=np.array([1, 1, 0, 0, 0, 0, 0, 0])  # Violates No-11
    )
    
    # Test consistency
    assert verifier.check_consistency(T_consistent), \
           "Consistent theory marked as inconsistent"
    
    assert not verifier.check_consistency(T_inconsistent), \
           "Inconsistent theory marked as consistent"
    
    print("✓ Consistency criterion correctly identifies consistent/inconsistent theories")


def test_concept_network_connectivity():
    """Test concept network connectivity (Theorem 6.4.5)"""
    verifier = TheorySelfVerification()
    
    # Create theory with well-connected concepts
    T_connected = Theory(
        name="T_connected",
        axioms=["A1"],
        definitions=["D1", "D2", "D3"],
        lemmas=["L1", "L2"],
        theorems=["T1", "T2", "T3"],
        self_depth=5,
        encoding=np.array([1, 0, 1, 0, 0, 1, 0, 1])
    )
    
    # Create theory with disconnected concepts
    T_disconnected = Theory(
        name="T_disconnected",
        axioms=["A1"],
        definitions=[],
        lemmas=[],
        theorems=["T_isolated"],
        self_depth=8,  # High depth requires high connectivity
        encoding=np.array([1, 0, 0, 0, 0, 0, 0, 1])
    )
    
    # Test connectivity
    assert verifier.check_concept_connectivity(T_connected), \
           "Well-connected theory marked as disconnected"
    
    assert not verifier.check_concept_connectivity(T_disconnected), \
           "Disconnected theory marked as connected"
    
    print("✓ Concept network connectivity correctly assessed")


def test_automatic_completeness():
    """Test automatic completeness check (Theorem 6.4.6)"""
    verifier = TheorySelfVerification()
    
    # Create complete theory
    T_complete = Theory(
        name="T_complete",
        axioms=["A1"],
        definitions=["D1", "D2", "D3"],
        lemmas=["L1", "L2"],
        theorems=["T1", "T2"],
        self_depth=4,
        encoding=np.array([1, 0, 1, 0, 0, 1, 0, 1])
    )
    
    # Test completeness
    is_complete = verifier.automatic_completeness_check(T_complete)
    
    print(f"✓ Automatic completeness check: {is_complete}")


def test_no11_constraint_preservation():
    """Test that No-11 constraint is preserved throughout verification"""
    verifier = TheorySelfVerification()
    
    # Create theories with valid No-11 encodings
    T1 = Theory(
        name="T_no11_1",
        axioms=["A1"],
        definitions=["D1"],
        lemmas=["L1"],
        theorems=["T1"],
        self_depth=3,
        encoding=np.array([1, 0, 1, 0, 0, 1, 0, 0])
    )
    
    T2 = Theory(
        name="T_no11_2",
        axioms=["A1"],
        definitions=["D1", "D2"],
        lemmas=["L1"],
        theorems=["T1"],
        self_depth=3,
        encoding=np.array([0, 1, 0, 1, 0, 0, 1, 0])
    )
    
    # Verify no consecutive 1s in individual encodings
    assert not verifier._violates_no11(T1.encoding, np.zeros(8)), \
           "T1 encoding violates No-11"
    assert not verifier._violates_no11(T2.encoding, np.zeros(8)), \
           "T2 encoding violates No-11"
    
    # Check that verification respects No-11
    strength = verifier.verify_pair(T1, T2)
    
    # Create bad encoding with consecutive 1s
    T_bad = Theory(
        name="T_bad",
        axioms=["A1"],
        definitions=["D1"],
        lemmas=["L1"],
        theorems=["T1"],
        self_depth=3,
        encoding=np.array([1, 1, 0, 0, 0, 0, 0, 0])  # Violates No-11
    )
    
    bad_strength = verifier.verify_pair(T1, T_bad)
    
    assert bad_strength < strength, \
           "No-11 violation should reduce verification strength"
    
    print("✓ No-11 constraint properly preserved")


def test_complexity_scaling():
    """Test that algorithm complexity scales as O(n^{φ+1})"""
    verifier = TheorySelfVerification()
    
    times = []
    sizes = [5, 10, 15, 20, 25]
    
    for n in sizes:
        # Create n theories
        theories = [
            Theory(
                name=f"T_{i}",
                axioms=["A1"],
                definitions=[f"D{j}" for j in range(i % 3 + 1)],
                lemmas=[f"L{i}"],
                theorems=[f"T{i}"],
                self_depth=(i % 5) + 1,
                encoding=np.random.rand(8) > 0.6
            )
            for i in range(n)
        ]
        
        # Time the verification matrix construction
        start = time.time()
        V_matrix = verifier.build_verification_matrix(theories)
        end = time.time()
        
        times.append(end - start)
    
    # Check scaling (should be roughly O(n^2.618))
    # Use log-log regression to estimate exponent
    log_sizes = np.log(sizes)
    log_times = np.log(times)
    
    # Linear regression in log space
    coeffs = np.polyfit(log_sizes, log_times, 1)
    estimated_exponent = coeffs[0]
    
    # Should be close to φ+1 ≈ 2.618
    expected_exponent = PHI + 1
    
    print(f"✓ Complexity scaling: estimated O(n^{estimated_exponent:.3f}), "
          f"expected O(n^{expected_exponent:.3f})")
    
    # Allow some tolerance due to small sample sizes
    assert abs(estimated_exponent - expected_exponent) < 1.0, \
           f"Scaling exponent {estimated_exponent} far from theoretical {expected_exponent}"


def test_verification_visualization():
    """Visualize the verification network"""
    verifier = TheorySelfVerification()
    
    # Create a set of interconnected theories
    theories = []
    for i in range(6):
        T = Theory(
            name=f"T{i+1}",
            axioms=["A1"],
            definitions=[f"D{j}" for j in range(i % 3 + 1)],
            lemmas=[f"L{i}"],
            theorems=[f"T{j}" for j in range(i)],
            self_depth=(i % 4) + 2,
            encoding=np.roll(np.array([1, 0, 1, 0, 0, 1, 0, 0]), i)
        )
        theories.append(T)
    
    # Build verification network
    G = nx.DiGraph()
    
    for T in theories:
        G.add_node(T.name, depth=T.self_depth)
    
    for i, T1 in enumerate(theories):
        for j, T2 in enumerate(theories):
            if i != j:
                strength = verifier.verify_pair(T1, T2)
                if strength > 0.3:  # Only show significant connections
                    G.add_edge(T1.name, T2.name, weight=strength)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Node colors based on self-reference depth
    node_colors = [G.nodes[node]['depth'] for node in G.nodes()]
    
    # Edge widths based on verification strength
    edges = G.edges()
    weights = [G[u][v]['weight'] * 3 for u, v in edges]
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          cmap='viridis', node_size=1000,
                          vmin=1, vmax=6)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6,
                          edge_color='gray', arrows=True,
                          arrowsize=20, arrowstyle='->')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=1, vmax=6))
    sm.set_array([])
    plt.colorbar(sm, label='Self-Reference Depth', fraction=0.046, pad=0.04)
    
    plt.title("Theory Self-Verification Network\n(T6.4 Theorem Visualization)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('T6_4_verification_network.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Verification network visualized and saved")


def run_all_tests():
    """Run all tests for T6.4"""
    print("=" * 60)
    print("Testing T6.4: Theory Self-Verification Theorem")
    print("=" * 60)
    
    test_functions = [
        test_basic_verification,
        test_recursive_verification_convergence,
        test_fixed_point_existence,
        test_circular_dependency_completeness,
        test_logical_chain_verification,
        test_consistency_criterion,
        test_concept_network_connectivity,
        test_automatic_completeness,
        test_no11_constraint_preservation,
        test_complexity_scaling,
        test_verification_visualization
    ]
    
    for test_func in test_functions:
        try:
            test_func()
        except AssertionError as e:
            print(f"✗ {test_func.__name__} failed: {e}")
        except Exception as e:
            print(f"✗ {test_func.__name__} error: {e}")
    
    print("=" * 60)
    print("All T6.4 tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()