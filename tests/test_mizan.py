"""
tests/test_mizan.py - Unit Tests for Al-Mizan Protocol
============================================================================
Ensures mathematical justice and constitutional integrity of:
- MizanValue (engine.py)
- ZakatOrchestrator (zakat_manager.py)
- Adaptive Zakat Rate (adaptive_zakat_rate)

Author: Amine Rekab (@rekabamine91-lgtm)
Version: 1.0.0 (June 2026)
============================================================================
"""

import unittest
import math
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from almizan.engine import MizanValue, adaptive_zakat_rate
from almizan.zakat_manager import ZakatOrchestrator


class TestMizanProtocol(unittest.TestCase):
    """
    Test suite for Al-Mizan Protocol constitutional guarantees.
    """
    
    def setUp(self):
        """Setup test environment before each test."""
        self.tau_tyr = 10.0
        self.v = MizanValue(5.0, label="TestNeuron", tau_tyr=self.tau_tyr)
        self.orchestrator = ZakatOrchestrator(poverty_threshold=0.5, base_rate=0.025)
    
    # ========================================================================
    # Tests for Al-Qist (Anti-Tyranny)
    # ========================================================================
    
    def test_al_qist_dampening(self):
        """Test that Al-Qist dampens tyrannical gradients above threshold."""
        # Set tyrannical gradient (20 > threshold 10)
        self.v.grad = 20.0
        self.v.apply_al_qist()
        
        # Gradient should be reduced
        self.assertLess(self.v.grad, 20.0, "Al-Qist failed to dampen tyrannical gradient")
        self.assertGreater(self.v.tyranny_count, 0, "Tyranny count did not increment")
    
    def test_al_qist_no_dampening_for_normal_gradients(self):
        """Test that Al-Qist does NOT dampen normal gradients below threshold."""
        # Set normal gradient (5 < threshold 10)
        self.v.grad = 5.0
        original_grad = self.v.grad
        self.v.apply_al_qist()
        
        # Gradient should remain unchanged
        self.assertEqual(self.v.grad, original_grad, "Normal gradient was incorrectly dampened")
        self.assertEqual(self.v.tyranny_count, 0, "Tyranny count incremented for normal gradient")
    
    def test_tyranny_count_accumulation(self):
        """Test that tyranny count accumulates over multiple violations."""
        # Simulate multiple tyranny events
        for _ in range(3):
            self.v.grad = 50.0
            self.v.apply_al_qist()
        
        self.assertEqual(self.v.tyranny_count, 3, "Tyranny count should be 3 after 3 violations")
    
    # ========================================================================
    # Tests for Integrity Score
    # ========================================================================
    
    def test_integrity_decay(self):
        """Test that integrity score decays after repeated tyranny."""
        initial_integrity = self.v.integrity_score
        
        # Simulate two tyranny events
        for _ in range(2):
            self.v.grad = 50.0
            self.v.apply_al_qist()
        
        self.assertLess(self.v.integrity_score, initial_integrity, 
                        "Integrity score should decay after repeated tyranny")
    
    def test_integrity_bounds(self):
        """Test that integrity score stays within (0, 1] bounds."""
        # Test with extreme tyranny
        for _ in range(100):
            self.v.grad = 1000.0
            self.v.apply_al_qist()
        
        self.assertGreater(self.v.integrity_score, 0.0, "Integrity score should not go negative")
        self.assertLessEqual(self.v.integrity_score, 1.0, "Integrity score should not exceed 1.0")
    
    def test_integrity_formula_parameters(self):
        """Test that alpha and beta parameters affect integrity score correctly."""
        # Create two neurons with different parameters
        v1 = MizanValue(5.0, label="v1", tau_tyr=10.0, alpha=0.1, beta=0.05)
        v2 = MizanValue(5.0, label="v2", tau_tyr=10.0, alpha=0.2, beta=0.10)
        
        # Apply same tyranny to both
        for _ in range(3):
            v1.grad = 50.0
            v1.apply_al_qist()
            v2.grad = 50.0
            v2.apply_al_qist()
        
        # Higher alpha/beta should lead to lower integrity
        self.assertLess(v2.integrity_score, v1.integrity_score,
                        "Higher alpha/beta should cause faster integrity decay")
    
    # ========================================================================
    # Tests for Information Entropy (Poverty Detection)
    # ========================================================================
    
    def test_entropy_high_for_variable_gradients(self):
        """Test that neurons with variable gradients have high entropy (rich)."""
        rich = MizanValue(1.0, label="rich")
        # Simulate rich neuron: varied gradient history
        rich.grad_history = [10.0, 12.0, 8.0, 11.0, 9.0, 13.0, 10.0, 12.0, 11.0, 9.0]
        
        entropy = rich.compute_entropy()
        self.assertGreater(entropy, 0.5, "Rich neuron should have high entropy > 0.5")
        self.assertFalse(rich.is_poor(poverty_threshold=0.5), 
                         "Rich neuron should NOT be classified as poor")
    
    def test_entropy_low_for_constant_gradients(self):
        """Test that neurons with constant gradients have low entropy (poor)."""
        poor = MizanValue(1.0, label="poor")
        # Simulate poor neuron: constant gradient history
        poor.grad_history = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        
        entropy = poor.compute_entropy()
        self.assertLess(entropy, 0.5, "Poor neuron should have low entropy < 0.5")
        self.assertTrue(poor.is_poor(poverty_threshold=0.5), 
                        "Poor neuron should be classified as poor")
    
    # ========================================================================
    # Tests for Zakat Orchestrator
    # ========================================================================
    
    def test_zakat_collection_from_rich_neurons(self):
        """Test that rich neurons pay zakat to the pool."""
        orchestrator = ZakatOrchestrator(poverty_threshold=0.5, base_rate=0.1)
        
        # Rich neuron (non-minority) with high gradient
        rich = MizanValue(1.0, label="Rich")
        rich.grad = 15.0
        
        # Apply zakat collection
        orchestrator.collect_and_distribute([rich], [False], rate=0.1)
        
        self.assertGreater(rich.zakat_given, 0, "Rich neuron should pay zakat")
        self.assertEqual(orchestrator.zakat_pool, rich.zakat_given, 
                         "Zakat pool should equal amount paid")
    
    def test_zakat_distribution_to_poor_neurons(self):
        """Test that poor neurons receive zakat from the pool."""
        orchestrator = ZakatOrchestrator(poverty_threshold=0.5, base_rate=0.1)
        
        # Rich neuron (non-minority)
        rich = MizanValue(1.0, label="Rich")
        rich.grad = 15.0
        
        # Poor neuron (minority)
        poor = MizanValue(0.1, label="Poor")
        poor.grad = 0.0
        
        # Collect and distribute
        orchestrator.collect_and_distribute([rich, poor], [False, True], rate=0.1)
        
        self.assertGreater(poor.zakat_received, 0, "Poor neuron should receive zakat")
        self.assertAlmostEqual(poor.zakat_received, orchestrator.zakat_pool,
                               delta=0.001, msg="Poor neuron should receive entire pool")
    
    def test_zakat_multiple_poor_neurons_equal_distribution(self):
        """Test that zakat is distributed equally among all poor neurons."""
        orchestrator = ZakatOrchestrator(poverty_threshold=0.5, base_rate=0.1)
        
        # One rich
        rich = MizanValue(1.0, label="Rich")
        rich.grad = 30.0
        
        # Two poor
        poor1 = MizanValue(0.1, label="Poor1")
        poor2 = MizanValue(0.2, label="Poor2")
        poor1.grad = 0.0
        poor2.grad = 0.0
        
        orchestrator.collect_and_distribute([rich, poor1, poor2], [False, True, True], rate=0.1)
        
        self.assertAlmostEqual(poor1.zakat_received, poor2.zakat_received, delta=0.001,
                               msg="Poor neurons should receive equal zakat shares")
    
    # ========================================================================
    # Tests for Adaptive Zakat Rate
    # ========================================================================
    
    def test_adaptive_zakat_initial_boost(self):
        """Test that adaptive zakat rate starts higher for small minorities."""
        minority_ratio = 0.01  # 1% minority
        rate = adaptive_zakat_rate(minority_ratio, iteration=0, tau=1000, base_rate=0.025)
        
        # Initial rate should be boosted
        self.assertGreater(rate, 0.025, "Adaptive zakat should boost initial rate for minorities")
        self.assertLessEqual(rate, 0.10, "Adaptive zakat rate should not exceed cap (10%)")
    
    def test_adaptive_zakat_decay_over_time(self):
        """Test that adaptive zakat rate decays to baseline over time."""
        minority_ratio = 0.01  # 1% minority
        
        initial_rate = adaptive_zakat_rate(minority_ratio, iteration=0, tau=1000, base_rate=0.025)
        final_rate = adaptive_zakat_rate(minority_ratio, iteration=2000, tau=1000, base_rate=0.025)
        
        self.assertGreater(initial_rate, final_rate, 
                           "Adaptive zakat rate should decay over time")
        self.assertAlmostEqual(final_rate, 0.025, delta=0.005,
                               msg="Rate should return to baseline after long training")
    
    def test_adaptive_zakat_larger_minority_smaller_boost(self):
        """Test that larger minority groups receive smaller initial boost."""
        small_minority = adaptive_zakat_rate(0.01, iteration=0, tau=1000, base_rate=0.025)
        large_minority = adaptive_zakat_rate(0.10, iteration=0, tau=1000, base_rate=0.025)
        
        self.assertGreater(small_minority, large_minority,
                           "Smaller minority groups should receive larger initial boost")
    
    def test_adaptive_zakat_no_boost_for_majority(self):
        """Test that majority groups (no protection needed) get baseline rate."""
        majority_rate = adaptive_zakat_rate(0.50, iteration=0, tau=1000, base_rate=0.025)
        
        self.assertAlmostEqual(majority_rate, 0.025, delta=0.001,
                               msg="Majority groups should receive baseline zakat rate")
    
    # ========================================================================
    # Tests for Arithmetic Operations
    # ========================================================================
    
    def test_addition_with_al_qist(self):
        """Test that addition operation triggers Al-Qist in backward pass."""
        a = MizanValue(2.0, label="a")
        b = MizanValue(3.0, label="b")
        c = a + b
        c.backward()
        
        # Both gradients should be 1.0 (chain rule for addition)
        self.assertAlmostEqual(a.grad, 1.0, delta=0.001)
        self.assertAlmostEqual(b.grad, 1.0, delta=0.001)
    
    def test_multiplication_backward(self):
        """Test multiplication backward pass correctness."""
        a = MizanValue(2.0, label="a")
        b = MizanValue(3.0, label="b")
        c = a * b
        c.backward()
        
        # ∂c/∂a = b = 3, ∂c/∂b = a = 2
        self.assertAlmostEqual(a.grad, 3.0, delta=0.001)
        self.assertAlmostEqual(b.grad, 2.0, delta=0.001)
    
    def test_complex_expression_with_tyranny(self):
        """Test complex expression with tyranny detection (c = a*b + a*a)."""
        a = MizanValue(2.0, label="a")
        b = MizanValue(10.0, label="b")  # Large value may cause gradient spikes
        c = a * b + a * a
        c.backward()
        
        # Expected: ∂c/∂a = b + 2a = 10 + 4 = 14
        self.assertAlmostEqual(a.grad, 14.0, delta=0.001)
        self.assertAlmostEqual(b.grad, 2.0, delta=0.001)
    
    # ========================================================================
    # Tests for Audit Trail
    # ========================================================================
    
    def test_audit_trail_contains_all_attributes(self):
        """Test that audit trail includes all constitutional attributes."""
        v = MizanValue(10.0, label="AuditTest")
        v.grad = 5.0
        v.apply_al_qist()
        
        audit = v.get_audit_trail()
        
        required_keys = ['label', 'data', 'grad', 'tyranny_count', 'integrity_score', 
                         'zakat_given', 'zakat_received', 'grad_history', 'entropy']
        
        for key in required_keys:
            self.assertIn(key, audit, f"Audit trail missing key: {key}")
    
    def test_audit_trail_serializable(self):
        """Test that audit trail is JSON-serializable."""
        import json
        v = MizanValue(10.0, label="SerializableTest")
        v.grad = 3.0
        v.apply_al_qist()
        
        audit = v.get_audit_trail()
        
        # Should not raise exception
        try:
            json_str = json.dumps(audit)
            self.assertIsInstance(json_str, str)
        except TypeError as e:
            self.fail(f"Audit trail not JSON serializable: {e}")
    
    # ========================================================================
    # Edge Cases and Stress Tests
    # ========================================================================
    
    def test_edge_case_zero_gradient(self):
        """Test behavior with zero gradient (not tyrannical)."""
        self.v.grad = 0.0
        self.v.apply_al_qist()
        
        self.assertEqual(self.v.grad, 0.0, "Zero gradient should remain zero")
        self.assertEqual(self.v.tyranny_count, 0, "Zero gradient should not trigger tyranny")
    
    def test_edge_case_negative_tyrannical_gradient(self):
        """Test that Al-Qist works with negative tyrannical gradients."""
        self.v.grad = -20.0  # Negative tyrannical gradient
        original_grad = self.v.grad
        self.v.apply_al_qist()
        
        self.assertNotEqual(self.v.grad, original_grad, 
                            "Negative tyrannical gradient should be dampened")
        self.assertGreater(self.v.tyranny_count, 0, "Negative gradient should increment tyranny count")
    
    def test_extreme_tyranny(self):
        """Test handling of extremely large gradients."""
        self.v.grad = 1e6  # Extremely large gradient
        
        # Should not crash
        self.v.apply_al_qist()
        
        self.assertGreater(self.v.tyranny_count, 0, "Tyranny count should increment")
        self.assertLess(self.v.grad, 1e6, "Gradient should be dampened")
    
    def test_gradient_history_management(self):
        """Test that gradient history doesn't grow indefinitely."""
        # Add many gradients
        for i in range(100):
            self.v.grad = float(i)
            self.v.apply_al_qist()
        
        # History should be bounded (max 20 entries)
        self.assertLessEqual(len(self.v.grad_history), 20, 
                             f"Gradient history should be bounded, but has {len(self.v.grad_history)} entries")
    
    # ========================================================================
    # Integration Tests
    # ========================================================================
    
    def test_end_to_end_constitutional_learning_cycle(self):
        """Test complete constitutional learning cycle with multiple neurons."""
        # Create neurons
        neurons = [MizanValue(np.random.uniform(-1, 1), label=f"N{i}") for i in range(5)]
        
        # Simulate training steps
        for step in range(3):
            # Generate raw gradients
            for n in neurons:
                n.grad = np.random.uniform(-20, 20)
                n.apply_al_qist()
            
            # Redistribute zakat
            entropies = [n.compute_entropy() for n in neurons]
            minority_mask = [e < 0.5 for e in entropies]
            self.orchestrator.collect_and_distribute(neurons, minority_mask)
        
        # Verify that all neurons have valid state
        for n in neurons:
            self.assertGreaterEqual(n.integrity_score, 0.0, "Integrity score out of bounds")
            self.assertLessEqual(n.integrity_score, 1.0, "Integrity score out of bounds")
            self.assertGreaterEqual(n.tyranny_count, 0, "Tyranny count cannot be negative")
            self.assertGreaterEqual(n.zakat_given, 0, "Zakat given cannot be negative")
            self.assertGreaterEqual(n.zakat_received, 0, "Zakat received cannot be negative")


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("🧪 Al-Mizan Protocol - Constitutional Justice Test Suite")
    print("=" * 70)
    
    # Run tests with verbosity
    unittest.main(verbosity=2)
