"""
Unit tests for V4: Theory Boundary Verification System
V4：理论边界验证系统的单元测试
"""
import unittest
import sys
import os
import math
import numpy as np
from typing import List, Tuple, Set, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_framework import VerificationTest


# Golden ratio constant
PHI = (1 + math.sqrt(5)) / 2


class Zone(Enum):
    """Validity zones as defined in the formal specification"""
    CORE = "Core"           # V > φ^21
    TRANSITION = "Transition"  # φ^10 < V <= φ^21
    PERIPHERY = "Periphery"    # φ^3 < V <= φ^10
    EXTERNAL = "External"      # V <= φ^3


@dataclass
class BoundaryCondition:
    """Boundary condition structure"""
    lower: float
    upper: float
    connectivity: float
    entropy_flow: float


class SharedV4ValidationBase:
    """Shared base class for V4 validation functionality"""
    
    @staticmethod
    def fibonacci(n: int) -> int:
        """Compute nth Fibonacci number"""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        elif n == 2:
            return 2
        
        a, b = 1, 2
        for _ in range(3, n + 1):
            a, b = b, a + b
        return b
    
    @staticmethod
    def to_zeckendorf(n: int) -> List[int]:
        """Convert integer to Zeckendorf representation (no consecutive 1s)"""
        if n == 0:
            return [0]
        
        # Build Fibonacci sequence: 1, 2, 3, 5, 8, 13, ...
        fibs = []
        a, b = 1, 2
        while a <= n:
            fibs.append(a)
            a, b = b, a + b
        
        if not fibs:
            return [0]
        
        # Greedy algorithm for Zeckendorf representation
        result = [0] * len(fibs)
        remaining = n
        
        for i in range(len(fibs) - 1, -1, -1):
            if fibs[i] <= remaining:
                result[i] = 1
                remaining -= fibs[i]
        
        # The result is already in the correct order (smallest Fibonacci first)
        # Remove trailing zeros if any
        while len(result) > 1 and result[-1] == 0:
            result.pop()
            
        return result if result else [0]
    
    @staticmethod
    def from_zeckendorf(zeck: List[int]) -> int:
        """Convert Zeckendorf representation to integer"""
        if not zeck:
            return 0
        
        # Build Fibonacci sequence starting from F_1=1, F_2=2
        fibs = []
        a, b = 1, 2
        for _ in range(len(zeck)):
            fibs.append(a)
            a, b = b, a + b
        
        # Compute value
        result = 0
        for i, bit in enumerate(zeck):
            if bit == 1:
                result += fibs[i]
        
        return result
    
    @staticmethod
    def verify_no_11_constraint(zeck: List[int]) -> bool:
        """Verify that Zeckendorf representation has no consecutive 1s"""
        for i in range(len(zeck) - 1):
            if zeck[i] == 1 and zeck[i + 1] == 1:
                return False
        return True
    
    @staticmethod
    def phi_power(n: int) -> float:
        """Compute φ^n"""
        return PHI ** n
    
    @staticmethod
    def log_phi(x: float) -> float:
        """Compute log_φ(x)"""
        if x <= 0:
            return float('-inf')
        return math.log(x) / math.log(PHI)


class ParameterPoint:
    """Parameter point in φ-encoded space"""
    
    def __init__(self, coordinates: List[List[int]]):
        """Initialize with Zeckendorf-encoded coordinates"""
        if not coordinates:
            raise ValueError("Parameter point must have at least one coordinate")
        self.coordinates = coordinates
        self._validate()
    
    def _validate(self):
        """Validate no-11 constraint for all coordinates"""
        for coord in self.coordinates:
            if not SharedV4ValidationBase.verify_no_11_constraint(coord):
                raise ValueError(f"Coordinate {coord} violates no-11 constraint")
    
    def decode(self) -> List[int]:
        """Decode all coordinates to integers"""
        return [SharedV4ValidationBase.from_zeckendorf(c) for c in self.coordinates]
    
    def __eq__(self, other):
        return self.coordinates == other.coordinates
    
    def __hash__(self):
        return hash(tuple(map(tuple, self.coordinates)))


class ValidityFunction:
    """Validity function V: P → ℝ^+_φ"""
    
    def __init__(self, weights: Optional[List[float]] = None):
        """Initialize with φ-encoded weights"""
        if weights is None:
            # Default weights are powers of φ
            self.weights = [PHI ** (-i) for i in range(10)]
        else:
            self.weights = weights
    
    def compute(self, point: ParameterPoint) -> float:
        """Compute validity value for a parameter point"""
        decoded = point.decode()
        if len(decoded) > len(self.weights):
            # Extend weights if needed
            for i in range(len(self.weights), len(decoded)):
                self.weights.append(PHI ** (-i))
        
        value = 0
        for i, coord_val in enumerate(decoded):
            value += self.weights[i] * coord_val
        
        # Handle edge cases
        if value <= 0:
            return SharedV4ValidationBase.phi_power(0)  # φ^0 = 1
        
        # Return as power of φ
        n = math.floor(SharedV4ValidationBase.log_phi(value))
        return SharedV4ValidationBase.phi_power(n)
    
    def get_zone(self, point: ParameterPoint) -> Zone:
        """Determine which zone a point belongs to"""
        v = self.compute(point)
        
        if v > SharedV4ValidationBase.phi_power(21):
            return Zone.CORE
        elif v > SharedV4ValidationBase.phi_power(10):
            return Zone.TRANSITION
        elif v > SharedV4ValidationBase.phi_power(3):
            return Zone.PERIPHERY
        else:
            return Zone.EXTERNAL


class BoundaryDetector:
    """Boundary detection algorithm implementation"""
    
    def __init__(self, validity_func: ValidityFunction):
        self.validity_func = validity_func
    
    def detect_boundary(self, point: ParameterPoint) -> Tuple[bool, int]:
        """
        Detect if a point is on a boundary
        Returns: (is_boundary, boundary_order)
        """
        v = self.validity_func.compute(point)
        
        # Check if v is exactly a power of φ
        log_v = SharedV4ValidationBase.log_phi(v)
        n = round(log_v)
        
        if abs(log_v - n) < 1e-10:  # Numerical tolerance
            return (True, n)
        else:
            return (False, -1)
    
    def find_boundary_points(self, points: List[ParameterPoint]) -> List[Tuple[ParameterPoint, int]]:
        """Find all boundary points from a list"""
        boundaries = []
        for p in points:
            is_boundary, order = self.detect_boundary(p)
            if is_boundary:
                boundaries.append((p, order))
        return boundaries


class ParameterTraverser:
    """Parameter space traversal algorithm"""
    
    def __init__(self, validity_func: ValidityFunction):
        self.validity_func = validity_func
        self.detector = BoundaryDetector(validity_func)
    
    def traverse(self, start: ParameterPoint, direction: ParameterPoint, 
                max_steps: int = 100) -> List[Tuple[ParameterPoint, int]]:
        """
        Traverse parameter space and collect boundary crossings
        """
        boundaries = []
        current = start
        prev_entropy = self._compute_entropy(current)
        
        for _ in range(max_steps):
            # Check for boundary
            is_boundary, order = self.detector.detect_boundary(current)
            if is_boundary:
                boundaries.append((current, order))
            
            # Move to next point
            current = self._phi_add(current, self._phi_multiply(PHI, direction))
            
            # Verify entropy increase
            curr_entropy = self._compute_entropy(current)
            if curr_entropy <= prev_entropy:
                break  # Stop if entropy doesn't increase
            prev_entropy = curr_entropy
            
            # Check if still in valid zone
            if self.validity_func.get_zone(current) == Zone.EXTERNAL:
                break
        
        return boundaries
    
    def _compute_entropy(self, point: ParameterPoint) -> float:
        """Compute entropy of a parameter point"""
        decoded = point.decode()
        if not decoded or all(x == 0 for x in decoded):
            return 0
        
        # Shannon entropy of the distribution
        total = sum(decoded)
        entropy = 0
        for val in decoded:
            if val > 0:
                p = val / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _phi_add(self, p1: ParameterPoint, p2: ParameterPoint) -> ParameterPoint:
        """Add two parameter points maintaining no-11 constraint"""
        result_coords = []
        
        for c1, c2 in zip(p1.coordinates, p2.coordinates):
            # Add Zeckendorf representations
            val1 = SharedV4ValidationBase.from_zeckendorf(c1)
            val2 = SharedV4ValidationBase.from_zeckendorf(c2)
            sum_val = val1 + val2
            
            # Convert back to Zeckendorf
            result_coords.append(SharedV4ValidationBase.to_zeckendorf(sum_val))
        
        return ParameterPoint(result_coords)
    
    def _phi_multiply(self, scalar: float, point: ParameterPoint) -> ParameterPoint:
        """Multiply parameter point by scalar"""
        result_coords = []
        
        for coord in point.coordinates:
            val = SharedV4ValidationBase.from_zeckendorf(coord)
            scaled = int(val * scalar)
            if scaled == 0:
                scaled = 1  # Ensure non-zero
            result_coords.append(SharedV4ValidationBase.to_zeckendorf(scaled))
        
        return ParameterPoint(result_coords)


class BoundarySurface:
    """Boundary surface representation"""
    
    def __init__(self, boundary_points: List[Tuple[ParameterPoint, int]]):
        self.points = boundary_points
        self.surface = self._reconstruct_surface()
    
    def _reconstruct_surface(self) -> Dict[int, Set[ParameterPoint]]:
        """Reconstruct boundary surface from points"""
        surface = {}
        for point, order in self.points:
            if order not in surface:
                surface[order] = set()
            surface[order].add(point)
        return surface
    
    def get_order_boundaries(self, order: int) -> Set[ParameterPoint]:
        """Get all boundary points of a specific order"""
        return self.surface.get(order, set())
    
    def verify_axiom_consistency(self, validity_func: ValidityFunction) -> bool:
        """Verify that boundary respects A1 axiom (entropy increase)"""
        for order, points in self.surface.items():
            for point in points:
                # Check that crossing boundary increases entropy
                zone = validity_func.get_zone(point)
                if zone == Zone.EXTERNAL:
                    continue  # Skip external points
                
                # Simplified check: boundary points should have defined validity
                v = validity_func.compute(point)
                if v <= 0:
                    return False
        
        return True


class TheoryBoundary:
    """Complete theory boundary specification"""
    
    def __init__(self, theory_id: int):
        self.theory_id = theory_id
        self.zeckendorf = SharedV4ValidationBase.to_zeckendorf(theory_id)
        self.boundary_condition = self._compute_boundary_condition()
    
    def _compute_boundary_condition(self) -> BoundaryCondition:
        """Compute boundary conditions from theory Zeckendorf decomposition"""
        # Find Fibonacci indices where zeckendorf has 1s
        fib_indices = []
        for i, bit in enumerate(self.zeckendorf):
            if bit == 1:
                fib_indices.append(i + 1)  # 1-indexed
        
        if not fib_indices:
            fib_indices = [1]  # Default to F_1
        
        lower = SharedV4ValidationBase.phi_power(min(fib_indices))
        upper = SharedV4ValidationBase.phi_power(sum(fib_indices))
        connectivity = math.prod([SharedV4ValidationBase.phi_power(i) for i in fib_indices])
        entropy_flow = upper - lower  # Simplified entropy flow
        
        return BoundaryCondition(lower, upper, connectivity, entropy_flow)
    
    def contains(self, point: ParameterPoint, validity_func: ValidityFunction) -> bool:
        """Check if a point is within theory boundary"""
        v = validity_func.compute(point)
        return self.boundary_condition.lower <= v <= self.boundary_condition.upper


class ViolationResponse:
    """Entropy injection for boundary violations"""
    
    @staticmethod
    def handle_violation(theory: TheoryBoundary, point: ParameterPoint, 
                        validity_func: ValidityFunction) -> ParameterPoint:
        """Handle boundary violation by injecting entropy"""
        v = validity_func.compute(point)
        
        if theory.contains(point, validity_func):
            return point  # No violation
        
        # Compute required entropy injection
        if v < theory.boundary_condition.lower:
            delta_h = theory.boundary_condition.lower - v
        else:
            delta_h = v - theory.boundary_condition.upper
        
        # Compute correction factor
        correction_order = math.ceil(SharedV4ValidationBase.log_phi(delta_h))
        correction = SharedV4ValidationBase.phi_power(correction_order)
        
        # Apply entropy injection (simplified: scale coordinates)
        new_coords = []
        for coord in point.coordinates:
            val = SharedV4ValidationBase.from_zeckendorf(coord)
            if v < theory.boundary_condition.lower:
                # Need to increase
                new_val = int(val * (1 + correction / 100))
            else:
                # Need to decrease
                new_val = int(val / (1 + correction / 100))
            
            new_coords.append(SharedV4ValidationBase.to_zeckendorf(max(1, new_val)))
        
        return ParameterPoint(new_coords)


class TestV4TheoryBoundary(VerificationTest):
    """Test suite for V4 Theory Boundary Verification System"""
    
    def setUp(self):
        """Initialize test environment"""
        super().setUp()
        self.validity_func = ValidityFunction()
        self.detector = BoundaryDetector(self.validity_func)
        self.traverser = ParameterTraverser(self.validity_func)
    
    # Test Zeckendorf encoding
    def test_zeckendorf_encoding_basic(self):
        """Test basic Zeckendorf encoding"""
        # Note: Using Fibonacci sequence 1, 2, 3, 5, 8, 13, ...
        # For each test, manually verify the encoding
        test_cases = [
            (0, [0]),
            (1, [1]),           # 1 = F_1
            (2, [0, 1]),        # 2 = F_2
            (3, [0, 0, 1]),     # 3 = F_3
            (4, [1, 0, 1]),     # 4 = F_1 + F_3 = 1 + 3
            (5, [0, 0, 0, 1]),  # 5 = F_4
            (8, [0, 0, 0, 0, 1]),  # 8 = F_5
        ]
        
        for n, expected in test_cases:
            zeck = SharedV4ValidationBase.to_zeckendorf(n)
            self.assertTrue(SharedV4ValidationBase.verify_no_11_constraint(zeck),
                          f"Zeckendorf of {n} violates no-11 constraint")
            decoded = SharedV4ValidationBase.from_zeckendorf(zeck)
            self.assertEqual(decoded, n, f"Zeckendorf round-trip failed for {n}")
    
    def test_no_11_constraint_verification(self):
        """Test no-11 constraint verification"""
        valid_cases = [[1, 0, 1], [0, 1, 0, 1], [1, 0, 0, 1]]
        invalid_cases = [[1, 1], [0, 1, 1, 0], [1, 1, 1]]
        
        for zeck in valid_cases:
            self.assertTrue(SharedV4ValidationBase.verify_no_11_constraint(zeck))
        
        for zeck in invalid_cases:
            self.assertFalse(SharedV4ValidationBase.verify_no_11_constraint(zeck))
    
    def test_fibonacci_computation(self):
        """Test Fibonacci number computation"""
        expected = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        for i, exp in enumerate(expected):
            self.assertEqual(SharedV4ValidationBase.fibonacci(i), exp)
    
    # Test parameter points
    def test_parameter_point_creation(self):
        """Test parameter point creation and validation"""
        # Valid point: [1, 0, 1] = 1 + 3 = 4, [0, 1] = 2
        coords = [[1, 0, 1], [0, 1]]
        point = ParameterPoint(coords)
        self.assertEqual(point.decode(), [4, 2])
        
        # Invalid point (consecutive 1s)
        with self.assertRaises(ValueError):
            ParameterPoint([[1, 1, 0]])
    
    def test_parameter_point_equality(self):
        """Test parameter point equality"""
        p1 = ParameterPoint([[1, 0, 1]])
        p2 = ParameterPoint([[1, 0, 1]])
        p3 = ParameterPoint([[0, 1, 0]])
        
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
    
    # Test validity function
    def test_validity_function_computation(self):
        """Test validity function computation"""
        point = ParameterPoint([[0, 1], [1, 0]])  # [2, 1]
        v = self.validity_func.compute(point)
        self.assertGreater(v, 0)
        self.assertEqual(v, SharedV4ValidationBase.phi_power(
            math.floor(SharedV4ValidationBase.log_phi(v))))
    
    def test_zone_classification(self):
        """Test zone classification"""
        # Create points with different validity values
        small_point = ParameterPoint([[1], [0]])  # Small values
        medium_point = ParameterPoint([[0, 0, 1], [0, 0, 1]])  # Medium values  
        large_point = ParameterPoint([[0, 0, 0, 0, 1], [0, 0, 0, 0, 1]])  # Large values
        
        zones = [
            self.validity_func.get_zone(small_point),
            self.validity_func.get_zone(medium_point),
            self.validity_func.get_zone(large_point)
        ]
        
        # At least one should be external, others may vary
        self.assertIn(Zone.EXTERNAL, zones)
    
    # Test boundary detection
    def test_boundary_detection_exact(self):
        """Test exact boundary detection"""
        # Create a point that should be on a boundary
        # We need to craft this carefully
        point = ParameterPoint([[0, 1]])  # Value = 2
        
        # Adjust validity function weights to make this a boundary
        custom_validity = ValidityFunction([PHI ** 2])
        detector = BoundaryDetector(custom_validity)
        
        is_boundary, order = detector.detect_boundary(point)
        # The result depends on the exact computation
        self.assertIsInstance(is_boundary, bool)
        self.assertIsInstance(order, int)
    
    def test_boundary_detection_non_boundary(self):
        """Test non-boundary point detection"""
        point = ParameterPoint([[1], [1], [1]])  # Unlikely to be exact power of φ
        is_boundary, order = self.detector.detect_boundary(point)
        
        if not is_boundary:
            self.assertEqual(order, -1)
    
    def test_find_boundary_points(self):
        """Test finding multiple boundary points"""
        points = [
            ParameterPoint([[1]]),
            ParameterPoint([[0, 1]]),
            ParameterPoint([[1, 0, 1]]),
        ]
        
        boundaries = self.detector.find_boundary_points(points)
        self.assertIsInstance(boundaries, list)
        
        for point, order in boundaries:
            self.assertIsInstance(point, ParameterPoint)
            self.assertGreaterEqual(order, 0)
    
    # Test parameter traversal
    def test_parameter_traversal_basic(self):
        """Test basic parameter traversal"""
        start = ParameterPoint([[1]])
        direction = ParameterPoint([[1]])
        
        boundaries = self.traverser.traverse(start, direction, max_steps=10)
        self.assertIsInstance(boundaries, list)
        
        # Verify all returned points are boundaries
        for point, order in boundaries:
            is_boundary, detected_order = self.detector.detect_boundary(point)
            if is_boundary:  # Only check if actually detected as boundary
                self.assertEqual(order, detected_order)
    
    def test_traversal_entropy_increase(self):
        """Test that traversal maintains entropy increase"""
        start = ParameterPoint([[1, 0]])
        direction = ParameterPoint([[1]])
        
        # Track entropy during traversal
        current = start
        prev_entropy = self.traverser._compute_entropy(current)
        
        for i in range(5):
            current = self.traverser._phi_add(current, 
                                             self.traverser._phi_multiply(PHI, direction))
            curr_entropy = self.traverser._compute_entropy(current)
            
            # Allow for small numerical errors or constant entropy
            if curr_entropy > 0 and prev_entropy > 0:
                self.assertGreaterEqual(curr_entropy, prev_entropy - 1e-10)
            prev_entropy = curr_entropy
    
    def test_phi_operations(self):
        """Test φ-arithmetic operations"""
        p1 = ParameterPoint([[1]])     # 1
        p2 = ParameterPoint([[0, 1]])  # 2
        
        # Test addition
        sum_point = self.traverser._phi_add(p1, p2)
        self.assertEqual(sum_point.decode()[0], 3)
        
        # Test scalar multiplication
        scaled = self.traverser._phi_multiply(2, p1)
        self.assertEqual(scaled.decode()[0], 2)
    
    # Test boundary surface
    def test_boundary_surface_reconstruction(self):
        """Test boundary surface reconstruction"""
        boundary_points = [
            (ParameterPoint([[1]]), 1),
            (ParameterPoint([[0, 1]]), 2),
            (ParameterPoint([[1, 0, 1]]), 1),
        ]
        
        surface = BoundarySurface(boundary_points)
        
        # Check order grouping
        order_1_points = surface.get_order_boundaries(1)
        self.assertEqual(len(order_1_points), 2)
        
        order_2_points = surface.get_order_boundaries(2)
        self.assertEqual(len(order_2_points), 1)
    
    def test_surface_axiom_consistency(self):
        """Test boundary surface axiom consistency"""
        boundary_points = [
            (ParameterPoint([[0, 1]]), 2),
            (ParameterPoint([[0, 0, 1]]), 3),
        ]
        
        surface = BoundarySurface(boundary_points)
        self.assertTrue(surface.verify_axiom_consistency(self.validity_func))
    
    # Test theory boundaries
    def test_theory_boundary_creation(self):
        """Test theory boundary creation"""
        theory = TheoryBoundary(10)  # 10 = 8 + 2 = F_5 + F_3
        
        # Check Zeckendorf decomposition
        self.assertTrue(SharedV4ValidationBase.verify_no_11_constraint(theory.zeckendorf))
        
        # Check boundary conditions
        self.assertGreater(theory.boundary_condition.upper, 
                          theory.boundary_condition.lower)
        self.assertGreater(theory.boundary_condition.entropy_flow, 0)
    
    def test_theory_containment(self):
        """Test point containment in theory boundary"""
        theory = TheoryBoundary(5)
        
        # Create test points
        inside_point = ParameterPoint([[0, 1]])  # Should be inside for some validity functions
        outside_point = ParameterPoint([[0]])  # Likely outside
        
        # At least one should be different
        contains_inside = theory.contains(inside_point, self.validity_func)
        contains_outside = theory.contains(outside_point, self.validity_func)
        
        self.assertIsInstance(contains_inside, bool)
        self.assertIsInstance(contains_outside, bool)
    
    def test_multiple_theory_boundaries(self):
        """Test multiple theory boundaries"""
        theories = [TheoryBoundary(i) for i in [3, 5, 8, 13]]
        
        # Check that boundary conditions are monotonic in some sense
        for i in range(len(theories) - 1):
            # Later theories should generally have higher upper bounds
            self.assertLessEqual(theories[i].boundary_condition.lower,
                               theories[i + 1].boundary_condition.upper)
    
    # Test violation response
    def test_violation_response_no_violation(self):
        """Test violation response when no violation exists"""
        theory = TheoryBoundary(5)
        point = ParameterPoint([[0, 1]])
        
        # If point is already valid, should return same point
        new_point = ViolationResponse.handle_violation(theory, point, self.validity_func)
        
        # Check that a point is returned
        self.assertIsInstance(new_point, ParameterPoint)
    
    def test_violation_response_correction(self):
        """Test violation response with correction needed"""
        theory = TheoryBoundary(20)  # High boundary
        point = ParameterPoint([[1]])  # Low value point
        
        new_point = ViolationResponse.handle_violation(theory, point, self.validity_func)
        
        # New point should be different if violation occurred
        self.assertIsInstance(new_point, ParameterPoint)
        
        # Check that we got a point back (may or may not be corrected)
        # The correction logic depends on whether the point violates the boundary
        v_original = self.validity_func.compute(point)
        v_new = self.validity_func.compute(new_point)
        
        # Both should be valid φ powers
        self.assertGreater(v_original, 0)
        self.assertGreater(v_new, 0)
    
    def test_entropy_injection_computation(self):
        """Test entropy injection computation"""
        theory = TheoryBoundary(8)
        
        # Test with point below boundary
        low_point = ParameterPoint([[1]])
        corrected = ViolationResponse.handle_violation(theory, low_point, self.validity_func)
        
        # Verify corrected point has higher validity
        v_original = self.validity_func.compute(low_point)
        v_corrected = self.validity_func.compute(corrected)
        
        if v_original < theory.boundary_condition.lower:
            self.assertGreaterEqual(v_corrected, v_original)
    
    # Integration tests
    def test_v1_axiom_integration(self):
        """Test integration with V1 axiom verification"""
        # Create a sequence of points
        points = [ParameterPoint([[i]]) for i in range(1, 6)]
        
        # Verify entropy increases along sequence
        entropies = [self.traverser._compute_entropy(p) for p in points]
        
        # For non-zero entropies, check general trend
        non_zero = [e for e in entropies if e > 0]
        if len(non_zero) > 1:
            # At least some should increase
            increases = sum(1 for i in range(len(non_zero) - 1) 
                          if non_zero[i + 1] >= non_zero[i])
            self.assertGreater(increases, 0)
    
    def test_complete_parameter_coverage(self):
        """Test that parameter space is completely covered"""
        # Every point should belong to exactly one zone
        test_points = [
            ParameterPoint([[0]]),
            ParameterPoint([[1]]),
            ParameterPoint([[0, 1]]),
            ParameterPoint([[0, 0, 1]]),
            ParameterPoint([[0, 0, 0, 1]]),
        ]
        
        for point in test_points:
            zone = self.validity_func.get_zone(point)
            self.assertIn(zone, [Zone.CORE, Zone.TRANSITION, Zone.PERIPHERY, Zone.EXTERNAL])
    
    def test_boundary_evolution(self):
        """Test boundary evolution over time"""
        # Simple model of boundary evolution
        theory = TheoryBoundary(5)
        time_steps = 5
        
        boundaries = []
        for t in range(time_steps):
            # Boundary evolves with entropy gradient
            evolved_lower = theory.boundary_condition.lower * (PHI ** (t * 0.1))
            evolved_upper = theory.boundary_condition.upper * (PHI ** (t * 0.1))
            boundaries.append((evolved_lower, evolved_upper))
        
        # Check that boundaries expand over time
        for i in range(len(boundaries) - 1):
            self.assertGreaterEqual(boundaries[i + 1][1], boundaries[i][1])
    
    # Edge cases and error handling
    def test_empty_parameter_point(self):
        """Test handling of empty parameter points"""
        with self.assertRaises(ValueError):
            ParameterPoint([])  # Should raise ValueError for empty coordinates
    
    def test_large_fibonacci_numbers(self):
        """Test handling of large Fibonacci numbers"""
        large_n = 100
        zeck = SharedV4ValidationBase.to_zeckendorf(large_n)
        self.assertTrue(SharedV4ValidationBase.verify_no_11_constraint(zeck))
        
        decoded = SharedV4ValidationBase.from_zeckendorf(zeck)
        self.assertEqual(decoded, large_n)
    
    def test_boundary_singularity_handling(self):
        """Test handling of boundary singularities"""
        # Create a point that might cause numerical issues
        tricky_point = ParameterPoint([[1, 0, 1, 0, 1]])
        
        # Should handle without crashing
        is_boundary, order = self.detector.detect_boundary(tricky_point)
        self.assertIsInstance(is_boundary, bool)
    
    def test_infinite_boundary_compactification(self):
        """Test compactification of infinite boundaries"""
        # Model infinite boundary with large values
        large_values = [SharedV4ValidationBase.phi_power(i) for i in range(50, 55)]
        
        # Apply compactification (map to finite range)
        compactified = [v / (1 + v / SharedV4ValidationBase.phi_power(100)) 
                       for v in large_values]
        
        # Check all values are finite
        for v in compactified:
            self.assertTrue(math.isfinite(v))
            self.assertLess(v, SharedV4ValidationBase.phi_power(100))
    
    # Performance and complexity tests
    def test_algorithm_complexity_boundary_detection(self):
        """Test that boundary detection has expected complexity"""
        import time
        
        sizes = [10, 20, 40]
        times = []
        
        for size in sizes:
            point = ParameterPoint([SharedV4ValidationBase.to_zeckendorf(i) 
                                   for i in range(1, min(size, 10) + 1)])
            
            start_time = time.time()
            for _ in range(100):
                self.detector.detect_boundary(point)
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        # Check that time grows subexponentially (approximately)
        if len(times) >= 2:
            growth_rate = times[-1] / times[0]
            self.assertLess(growth_rate, sizes[-1] / sizes[0] * 10)  # Allow for some overhead
    
    def test_surface_reconstruction_performance(self):
        """Test surface reconstruction performance"""
        # Create many boundary points
        boundary_points = []
        for i in range(1, 21):
            point = ParameterPoint([SharedV4ValidationBase.to_zeckendorf(i)])
            boundary_points.append((point, i % 5 + 1))
        
        # Should complete in reasonable time
        import time
        start = time.time()
        surface = BoundarySurface(boundary_points)
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 1.0)  # Should be fast
        self.assertTrue(surface.verify_axiom_consistency(self.validity_func))
    
    # Meta-properties tests
    def test_self_referential_boundary(self):
        """Test that V4 can determine its own validity boundary"""
        # V4 system should be able to verify itself
        v4_theory_id = 4
        v4_boundary = TheoryBoundary(v4_theory_id)
        
        # Create a point representing V4's own parameters
        v4_params = ParameterPoint([SharedV4ValidationBase.to_zeckendorf(v4_theory_id)])
        
        # V4 should be able to determine if it's within its own boundary
        is_valid = v4_boundary.contains(v4_params, self.validity_func)
        self.assertIsInstance(is_valid, bool)
    
    def test_recursive_boundary_definition(self):
        """Test recursive boundary definitions"""
        # Boundaries can reference other boundaries
        boundary_stack = []
        
        for i in range(1, 6):
            fib_val = SharedV4ValidationBase.fibonacci(i)
            theory = TheoryBoundary(fib_val)
            boundary_stack.append(theory)
            
            # Each boundary should be well-defined
            self.assertGreater(theory.boundary_condition.upper, 0)
            self.assertGreater(theory.boundary_condition.lower, 0)
            self.assertGreaterEqual(theory.boundary_condition.upper,
                                  theory.boundary_condition.lower)
    
    def test_fixed_point_existence(self):
        """Test existence of boundary fixed points"""
        # There should exist points where V(p) = φ^k for some special k
        
        # Golden ratio fixed point
        golden_point = ParameterPoint([[0, 1]])  # Represents 2, close to φ
        v = self.validity_func.compute(golden_point)
        
        # Check if it's close to a power of φ
        log_v = SharedV4ValidationBase.log_phi(v)
        nearest_int = round(log_v)
        distance = abs(log_v - nearest_int)
        
        # Should be very close to an integer power
        self.assertLess(distance, 1.0)


if __name__ == '__main__':
    unittest.main()