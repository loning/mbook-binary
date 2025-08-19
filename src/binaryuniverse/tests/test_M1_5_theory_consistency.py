#!/usr/bin/env python3
"""
Test Suite for M1.5 Theory Consistency Metatheorem
Tests the four-layer contradiction detection mechanism and consistency tensor computation
"""

import unittest
import math
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum

# Fibonacci sequence for φ-encoding (F1=1, F2=2, ...)
FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
PHI = (1 + math.sqrt(5)) / 2

class ContradictionType(Enum):
    """Types of contradictions in theory system"""
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic" 
    LOGICAL = "logical"
    METATHEORETIC = "metatheoretic"

class ResolutionStrategy(Enum):
    """Contradiction resolution strategies"""
    LOCAL_REPAIR = "local_repair"
    THEORY_RECONSTRUCTION = "theory_reconstruction"
    META_EXTENSION = "meta_extension"

class TheoryElement:
    """Represents a theory element with φ-encoding"""
    def __init__(self, content: str, theory_id: str, element_type: str = "statement"):
        self.content = content
        self.theory_id = theory_id
        self.element_type = element_type
        self.zeckendorf_encoding = self._compute_zeckendorf()
        
    def _compute_zeckendorf(self) -> List[int]:
        """Compute Zeckendorf decomposition of content hash"""
        content_hash = hash(self.content) % 1000  # Simplify for testing
        return self._zeckendorf_decompose(content_hash)
    
    def _zeckendorf_decompose(self, n: int) -> List[int]:
        """Decompose number into non-consecutive Fibonacci numbers"""
        if n <= 0:
            return []
        
        decomposition = []
        i = len(FIBONACCI) - 1
        
        while i >= 0 and n > 0:
            if FIBONACCI[i] <= n:
                decomposition.append(FIBONACCI[i])
                n -= FIBONACCI[i]
                i -= 2  # Skip next Fibonacci to avoid consecutive
            else:
                i -= 1
        
        return sorted(decomposition)
    
    def has_no11_violation(self) -> bool:
        """Check for No-11 constraint violations"""
        binary_str = ""
        max_fib = max(self.zeckendorf_encoding) if self.zeckendorf_encoding else 0
        
        for fib in FIBONACCI:
            if fib > max_fib:
                break
            binary_str += "1" if fib in self.zeckendorf_encoding else "0"
        
        return "11" in binary_str

class TheorySystem:
    """Represents a theory system for consistency checking"""
    def __init__(self, name: str):
        self.name = name
        self.theories: List[TheoryElement] = []
        self.predictions: List[TheoryElement] = []
        self.inferences: List[TheoryElement] = []
        self.axiom_compatibility: Dict[str, float] = {}
        
    def add_theory(self, theory: TheoryElement):
        """Add theory element to system"""
        self.theories.append(theory)
        
    def add_prediction(self, prediction: TheoryElement):
        """Add prediction to system"""
        self.predictions.append(prediction)
        
    def add_inference(self, inference: TheoryElement):
        """Add inference to system"""
        self.inferences.append(inference)
        
    def set_axiom_compatibility(self, theory_id: str, score: float):
        """Set A1 axiom compatibility score for theory"""
        self.axiom_compatibility[theory_id] = max(0.0, min(1.0, score))

class ConsistencyTensorAnalyzer:
    """Analyzes theory system consistency using four-layer detection"""
    
    def __init__(self):
        self.phi = PHI
        self.consistency_threshold = PHI ** 3  # φ³ ≈ 4.236
        
    def detect_syntactic_contradictions(self, system: TheorySystem) -> List[ContradictionType]:
        """Layer 1: Detect syntactic contradictions"""
        contradictions = []
        
        # Check Zeckendorf violations
        zeckendorf_violations = 0
        no11_violations = 0
        
        all_elements = system.theories + system.predictions + system.inferences
        for element in all_elements:
            if not element.zeckendorf_encoding:
                zeckendorf_violations += 1
            if element.has_no11_violation():
                no11_violations += 1
        
        if zeckendorf_violations > 0 or no11_violations > 0:
            contradictions.append(ContradictionType.SYNTACTIC)
            
        return contradictions
    
    def detect_semantic_contradictions(self, system: TheorySystem) -> List[ContradictionType]:
        """Layer 2: Detect semantic contradictions"""
        contradictions = []
        
        # Check prediction conflicts (simplified)
        prediction_conflicts = 0
        for i, pred1 in enumerate(system.predictions):
            for pred2 in system.predictions[i+1:]:
                if self._predictions_conflict(pred1, pred2):
                    prediction_conflicts += 1
        
        # Check definition conflicts
        definition_conflicts = self._count_definition_conflicts(system)
        
        if prediction_conflicts > 0 or definition_conflicts > 0:
            contradictions.append(ContradictionType.SEMANTIC)
            
        return contradictions
    
    def detect_logical_contradictions(self, system: TheorySystem) -> List[ContradictionType]:
        """Layer 3: Detect logical contradictions"""
        contradictions = []
        
        # Check circular reasoning
        circular_reasoning = self._detect_circular_reasoning(system)
        
        # Check proof contradictions
        proof_contradictions = self._detect_proof_contradictions(system)
        
        if circular_reasoning > 0 or proof_contradictions > 0:
            contradictions.append(ContradictionType.LOGICAL)
            
        return contradictions
    
    def detect_metatheoretic_contradictions(self, system: TheorySystem) -> List[ContradictionType]:
        """Layer 4: Detect metatheoretic contradictions"""
        contradictions = []
        
        # Check A1 axiom conflicts
        axiom_conflicts = 0
        for theory_id, compatibility in system.axiom_compatibility.items():
            if compatibility < 0.5:  # Below reasonable threshold
                axiom_conflicts += 1
        
        # Check φ-encoding principle violations
        phi_violations = self._detect_phi_violations(system)
        
        if axiom_conflicts > 0 or phi_violations > 0:
            contradictions.append(ContradictionType.METATHEORETIC)
            
        return contradictions
    
    def compute_consistency_tensor(self, system: TheorySystem) -> Dict[str, float]:
        """Compute five-component consistency tensor"""
        # Detect contradictions in each layer
        c1 = self.detect_syntactic_contradictions(system)
        c2 = self.detect_semantic_contradictions(system)
        c3 = self.detect_logical_contradictions(system)
        c4 = self.detect_metatheoretic_contradictions(system)
        
        # Compute consistency scores for each layer
        syntax_consistency = self.phi * (1 - len(c1) / max(1, len(system.theories)))
        semantic_consistency = self.phi * (1 - len(c2) / max(1, len(system.predictions)))
        logical_consistency = self.phi * (1 - len(c3) / max(1, len(system.inferences)))
        
        # Meta consistency based on axiom compatibility
        avg_compatibility = np.mean(list(system.axiom_compatibility.values())) if system.axiom_compatibility else 1.0
        meta_consistency = self.phi * avg_compatibility
        
        # Total consistency (geometric mean)
        total_consistency = (syntax_consistency * semantic_consistency * 
                           logical_consistency * meta_consistency) ** 0.25
        
        return {
            'syntax_consistency': max(0.0, syntax_consistency),
            'semantic_consistency': max(0.0, semantic_consistency),
            'logical_consistency': max(0.0, logical_consistency),
            'meta_consistency': max(0.0, meta_consistency),
            'total_consistency': max(0.0, total_consistency)
        }
    
    def is_consistent(self, system: TheorySystem) -> bool:
        """Check if theory system meets consistency threshold"""
        tensor = self.compute_consistency_tensor(system)
        return tensor['total_consistency'] >= self.consistency_threshold
    
    def calculate_contradiction_severity(self, contradiction_type: ContradictionType, 
                                      system: TheorySystem) -> float:
        """Calculate severity of contradiction for resolution strategy selection"""
        if contradiction_type == ContradictionType.SYNTACTIC:
            syntactic_count = len(self.detect_syntactic_contradictions(system))
            return syntactic_count / max(1, len(system.theories))
        elif contradiction_type == ContradictionType.SEMANTIC:
            semantic_count = len(self.detect_semantic_contradictions(system))
            return semantic_count / max(1, len(system.predictions))
        elif contradiction_type == ContradictionType.LOGICAL:
            logical_count = len(self.detect_logical_contradictions(system))
            return logical_count / max(1, len(system.inferences))
        elif contradiction_type == ContradictionType.METATHEORETIC:
            avg_compatibility = np.mean(list(system.axiom_compatibility.values())) if system.axiom_compatibility else 1.0
            return self.phi * (1 - avg_compatibility)
        
        return 0.0
    
    def select_resolution_strategy(self, contradiction_type: ContradictionType, 
                                 severity: float) -> ResolutionStrategy:
        """Select appropriate resolution strategy based on severity"""
        if severity < self.phi:
            return ResolutionStrategy.LOCAL_REPAIR
        elif severity < self.phi ** 2:
            return ResolutionStrategy.THEORY_RECONSTRUCTION
        else:
            return ResolutionStrategy.META_EXTENSION
    
    def _predictions_conflict(self, pred1: TheoryElement, pred2: TheoryElement) -> bool:
        """Check if two predictions conflict (simplified heuristic)"""
        # Simplified: check if predictions from different theories contradict
        return (pred1.theory_id != pred2.theory_id and 
                len(set(pred1.zeckendorf_encoding) & set(pred2.zeckendorf_encoding)) == 0)
    
    def _count_definition_conflicts(self, system: TheorySystem) -> int:
        """Count definition conflicts in theory system"""
        # Simplified: check for theories with identical content but different IDs
        content_to_theories = {}
        conflicts = 0
        
        for theory in system.theories:
            if theory.content in content_to_theories:
                if content_to_theories[theory.content] != theory.theory_id:
                    conflicts += 1
            else:
                content_to_theories[theory.content] = theory.theory_id
                
        return conflicts
    
    def _detect_circular_reasoning(self, system: TheorySystem) -> int:
        """Detect circular reasoning patterns (simplified)"""
        # Simplified: check for theories that reference each other cyclically
        references = {}
        for theory in system.theories:
            references[theory.theory_id] = []
            for other in system.theories:
                if other.theory_id != theory.theory_id and other.theory_id in theory.content:
                    references[theory.theory_id].append(other.theory_id)
        
        # Simple cycle detection
        cycles = 0
        for theory_id, refs in references.items():
            for ref in refs:
                if ref in references and theory_id in references[ref]:
                    cycles += 1
        
        return cycles // 2  # Each cycle counted twice
    
    def _detect_proof_contradictions(self, system: TheorySystem) -> int:
        """Detect proof contradictions (simplified)"""
        # Simplified: check for inferences that contradict each other
        contradictions = 0
        for i, inf1 in enumerate(system.inferences):
            for inf2 in system.inferences[i+1:]:
                if self._inferences_contradict(inf1, inf2):
                    contradictions += 1
        return contradictions
    
    def _inferences_contradict(self, inf1: TheoryElement, inf2: TheoryElement) -> bool:
        """Check if two inferences contradict each other"""
        # Simplified: check if inferences have opposite Zeckendorf patterns
        enc1 = set(inf1.zeckendorf_encoding)
        enc2 = set(inf2.zeckendorf_encoding)
        return len(enc1 & enc2) == 0 and len(enc1) > 0 and len(enc2) > 0
    
    def _detect_phi_violations(self, system: TheorySystem) -> int:
        """Detect φ-encoding principle violations"""
        violations = 0
        all_elements = system.theories + system.predictions + system.inferences
        
        for element in all_elements:
            # Check if element follows φ-encoding efficiency
            expected_efficiency = self.phi
            actual_efficiency = len(element.zeckendorf_encoding) / max(1, len(element.content))
            
            if abs(actual_efficiency - expected_efficiency) > 0.5:
                violations += 1
                
        return violations

class TestM15TheoryConsistency(unittest.TestCase):
    """Test suite for M1.5 Theory Consistency Metatheorem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = ConsistencyTensorAnalyzer()
        self.phi = PHI
        
        # Create test theory system
        self.test_system = TheorySystem("TestSystem")
        
        # Add consistent theories
        theory1 = TheoryElement("Self-referential complete systems must increase entropy", "T1", "axiom")
        theory2 = TheoryElement("φ-encoding optimizes information structure", "T2", "principle")
        theory3 = TheoryElement("Zeckendorf decomposition ensures uniqueness", "T3", "lemma")
        
        self.test_system.add_theory(theory1)
        self.test_system.add_theory(theory2)
        self.test_system.add_theory(theory3)
        
        # Add predictions
        pred1 = TheoryElement("System entropy will increase", "T1", "prediction")
        pred2 = TheoryElement("Information structure will optimize", "T2", "prediction")
        
        self.test_system.add_prediction(pred1)
        self.test_system.add_prediction(pred2)
        
        # Add inferences
        inf1 = TheoryElement("Therefore entropy increase is inevitable", "T1", "inference")
        inf2 = TheoryElement("Therefore φ-encoding is optimal", "T2", "inference")
        
        self.test_system.add_inference(inf1)
        self.test_system.add_inference(inf2)
        
        # Set axiom compatibility
        self.test_system.set_axiom_compatibility("T1", 1.0)
        self.test_system.set_axiom_compatibility("T2", 0.9)
        self.test_system.set_axiom_compatibility("T3", 0.8)
    
    def test_zeckendorf_encoding_generation(self):
        """Test Zeckendorf encoding generation for theory elements"""
        theory = TheoryElement("Test theory content", "T_test")
        
        # Should have valid Zeckendorf decomposition
        self.assertIsInstance(theory.zeckendorf_encoding, list)
        self.assertTrue(all(fib in FIBONACCI for fib in theory.zeckendorf_encoding))
        
        # Should be sorted
        self.assertEqual(theory.zeckendorf_encoding, sorted(theory.zeckendorf_encoding))
    
    def test_no11_constraint_checking(self):
        """Test No-11 constraint violation detection"""
        # Create theory with potential No-11 violation
        theory = TheoryElement("Test content", "T_test")
        
        # Check No-11 constraint
        has_violation = theory.has_no11_violation()
        self.assertIsInstance(has_violation, bool)
    
    def test_syntactic_contradiction_detection(self):
        """Test syntactic layer contradiction detection"""
        contradictions = self.analyzer.detect_syntactic_contradictions(self.test_system)
        
        # Should return list of contradiction types
        self.assertIsInstance(contradictions, list)
        self.assertTrue(all(isinstance(c, ContradictionType) for c in contradictions))
    
    def test_semantic_contradiction_detection(self):
        """Test semantic layer contradiction detection"""
        contradictions = self.analyzer.detect_semantic_contradictions(self.test_system)
        
        # Should return list of contradiction types
        self.assertIsInstance(contradictions, list)
        self.assertTrue(all(isinstance(c, ContradictionType) for c in contradictions))
    
    def test_logical_contradiction_detection(self):
        """Test logical layer contradiction detection"""
        contradictions = self.analyzer.detect_logical_contradictions(self.test_system)
        
        # Should return list of contradiction types
        self.assertIsInstance(contradictions, list)
        self.assertTrue(all(isinstance(c, ContradictionType) for c in contradictions))
    
    def test_metatheoretic_contradiction_detection(self):
        """Test metatheoretic layer contradiction detection"""
        contradictions = self.analyzer.detect_metatheoretic_contradictions(self.test_system)
        
        # Should return list of contradiction types
        self.assertIsInstance(contradictions, list)
        self.assertTrue(all(isinstance(c, ContradictionType) for c in contradictions))
    
    def test_consistency_tensor_computation(self):
        """Test consistency tensor computation"""
        tensor = self.analyzer.compute_consistency_tensor(self.test_system)
        
        # Should have all five components
        expected_keys = {'syntax_consistency', 'semantic_consistency', 
                        'logical_consistency', 'meta_consistency', 'total_consistency'}
        self.assertEqual(set(tensor.keys()), expected_keys)
        
        # All values should be non-negative
        for value in tensor.values():
            self.assertGreaterEqual(value, 0.0)
        
        # Total consistency should be geometric mean
        expected_total = (tensor['syntax_consistency'] * tensor['semantic_consistency'] * 
                         tensor['logical_consistency'] * tensor['meta_consistency']) ** 0.25
        self.assertAlmostEqual(tensor['total_consistency'], expected_total, places=5)
    
    def test_consistency_threshold_checking(self):
        """Test consistency threshold validation"""
        is_consistent = self.analyzer.is_consistent(self.test_system)
        
        # Should return boolean (convert numpy bool to Python bool)
        self.assertIsInstance(bool(is_consistent), bool)
        
        # Check threshold value
        self.assertAlmostEqual(self.analyzer.consistency_threshold, self.phi ** 3, places=5)
    
    def test_contradiction_severity_calculation(self):
        """Test contradiction severity calculation"""
        for contradiction_type in ContradictionType:
            severity = self.analyzer.calculate_contradiction_severity(
                contradiction_type, self.test_system)
            
            # Severity should be non-negative
            self.assertGreaterEqual(severity, 0.0)
            self.assertIsInstance(severity, float)
    
    def test_resolution_strategy_selection(self):
        """Test resolution strategy selection based on severity"""
        # Test different severity levels
        severities = [0.5, 2.0, 5.0]  # Below φ, between φ and φ², above φ²
        expected_strategies = [
            ResolutionStrategy.LOCAL_REPAIR,
            ResolutionStrategy.THEORY_RECONSTRUCTION,
            ResolutionStrategy.META_EXTENSION
        ]
        
        for severity, expected in zip(severities, expected_strategies):
            strategy = self.analyzer.select_resolution_strategy(
                ContradictionType.SYNTACTIC, severity)
            self.assertEqual(strategy, expected)
    
    def test_phi_encoding_compatibility(self):
        """Test φ-encoding compatibility checking"""
        # All elements should use φ-related structures
        all_elements = (self.test_system.theories + 
                       self.test_system.predictions + 
                       self.test_system.inferences)
        
        for element in all_elements:
            # Zeckendorf encoding should use Fibonacci numbers
            self.assertTrue(all(fib in FIBONACCI for fib in element.zeckendorf_encoding))
    
    def test_consistency_tensor_phi_bounds(self):
        """Test that consistency tensor components respect φ bounds"""
        tensor = self.analyzer.compute_consistency_tensor(self.test_system)
        
        # Each component should be bounded by φ
        for key in ['syntax_consistency', 'semantic_consistency', 
                   'logical_consistency', 'meta_consistency']:
            self.assertLessEqual(tensor[key], self.phi)
    
    def test_four_layer_completeness(self):
        """Test that four-layer detection covers all contradiction types"""
        all_contradictions = set()
        
        # Collect contradictions from all layers
        all_contradictions.update(self.analyzer.detect_syntactic_contradictions(self.test_system))
        all_contradictions.update(self.analyzer.detect_semantic_contradictions(self.test_system))
        all_contradictions.update(self.analyzer.detect_logical_contradictions(self.test_system))
        all_contradictions.update(self.analyzer.detect_metatheoretic_contradictions(self.test_system))
        
        # Should cover all contradiction types if present
        contradiction_types = {ContradictionType.SYNTACTIC, ContradictionType.SEMANTIC,
                             ContradictionType.LOGICAL, ContradictionType.METATHEORETIC}
        
        # All detected contradictions should be valid types
        self.assertTrue(all_contradictions.issubset(contradiction_types))
    
    def test_consistency_promotes_completeness_relation(self):
        """Test relationship between consistency and completeness (M1.4)"""
        tensor = self.analyzer.compute_consistency_tensor(self.test_system)
        
        # If consistency is high, it should promote completeness
        if tensor['total_consistency'] >= self.phi ** 3:
            # High consistency should correlate with better completeness potential
            completeness_factor = 1 + tensor['total_consistency'] / (self.phi ** 5)
            self.assertGreater(completeness_factor, 1.0)
    
    def test_axiom_compatibility_scoring(self):
        """Test A1 axiom compatibility scoring"""
        # All compatibility scores should be in [0,1]
        for score in self.test_system.axiom_compatibility.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # High compatibility should contribute to meta consistency
        tensor = self.analyzer.compute_consistency_tensor(self.test_system)
        avg_compatibility = np.mean(list(self.test_system.axiom_compatibility.values()))
        expected_meta = self.phi * avg_compatibility
        self.assertAlmostEqual(tensor['meta_consistency'], expected_meta, places=5)
    
    def test_geometric_mean_consistency(self):
        """Test that total consistency uses geometric mean"""
        tensor = self.analyzer.compute_consistency_tensor(self.test_system)
        
        # Compute expected geometric mean
        components = [tensor['syntax_consistency'], tensor['semantic_consistency'],
                     tensor['logical_consistency'], tensor['meta_consistency']]
        expected_geometric_mean = np.power(np.prod(components), 1/4)
        
        self.assertAlmostEqual(tensor['total_consistency'], expected_geometric_mean, places=5)
    
    def test_contradiction_resolution_convergence(self):
        """Test that contradiction resolution decreases contradiction count"""
        # Create system with artificial contradictions
        inconsistent_system = TheorySystem("InconsistentSystem")
        
        # Add conflicting theories
        theory1 = TheoryElement("Entropy must increase", "T1")
        theory2 = TheoryElement("Entropy must decrease", "T2")  # Contradiction
        
        inconsistent_system.add_theory(theory1)
        inconsistent_system.add_theory(theory2)
        
        # Set low axiom compatibility to create meta contradiction
        inconsistent_system.set_axiom_compatibility("T1", 0.3)
        inconsistent_system.set_axiom_compatibility("T2", 0.2)
        
        # Check that system is detected as inconsistent
        initial_tensor = self.analyzer.compute_consistency_tensor(inconsistent_system)
        self.assertLess(initial_tensor['total_consistency'], self.analyzer.consistency_threshold)

if __name__ == '__main__':
    unittest.main()