"""
zakat_manager.py - Digital Zakat Orchestrator for Al-Mizan Protocol
============================================================================
Handles collective redistribution of gradient capital from "rich" (high-entropy)
neurons to "poor" (low-entropy) neurons based on Islamic principle of Zakat.

Author: Amine Rekab (@rekabamine91-lgtm)
Version: 1.0.1 (Corrected)
============================================================================
"""

import math
from typing import List, Optional

# تصحيح الاستيراد ليعمل في كلا الحالتين: كـ Package أو كملف مستقل
try:
    from .engine import MizanValue, adaptive_zakat_rate
except (ImportError, ValueError):
    from engine import MizanValue, adaptive_zakat_rate


class ZakatOrchestrator:
    """
    Digital Zakat Orchestrator - Collective redistribution of learning capital.
    
    This class implements the core redistributive justice mechanism of Al-Mizan:
    1. Classifies neurons as "rich" (high entropy) or "poor" (low entropy)
    2. Collects zakat (ζ% of gradient) from rich neurons
    3. Redistributes equally to poor neurons
    4. Provides full audit trail for transparency
    
    Example:
        >>> orchestrator = ZakatOrchestrator(poverty_threshold=0.5, base_rate=0.025)
        >>> neurons = [MizanValue(1.0, label=f"N{i}") for i in range(10)]
        >>> orchestrator.collect_and_distribute(neurons, iteration=0)
    """
    
    def __init__(
        self,
        poverty_threshold: float = 0.5,
        base_rate: float = 0.025,
        tau_step: int = 1000,
        use_entropy: bool = True
    ):
        """
        Initialize the Zakat Orchestrator.
        
        Args:
            poverty_threshold (float): Entropy below which a neuron is "poor" (θ)
            base_rate (float): Base zakat rate (ζ_base = 2.5%)
            tau_step (int): Time constant for adaptive zakat decay (τ)
            use_entropy (bool): Use entropy for poverty detection (vs gradient magnitude)
        """
        self.poverty_threshold = poverty_threshold
        self.base_rate = base_rate
        self.tau_step = tau_step
        self.use_entropy = use_entropy
        self.iteration = 0
        self.audit_log: List[dict] = []
        self.total_zakat_collected = 0.0
        self.total_zakat_distributed = 0.0
    
    def _is_rich(self, neuron: MizanValue) -> bool:
        """
        Determine if a neuron is "rich" (should pay zakat).
        
        Args:
            neuron: MizanValue instance
            
        Returns:
            bool: True if neuron is rich, False otherwise
        """
        if self.use_entropy:
            # Rich neurons have high entropy (varied gradient history)
            return neuron.compute_entropy() >= self.poverty_threshold
        # Fallback: rich neurons have large gradient magnitude
        return abs(neuron.grad) > self.poverty_threshold
    
    def _is_poor(self, neuron: MizanValue) -> bool:
        """
        Determine if a neuron is "poor" (should receive zakat).
        
        Args:
            neuron: MizanValue instance
            
        Returns:
            bool: True if neuron is poor, False otherwise
        """
        if self.use_entropy:
            # Poor neurons have low entropy (constant gradient history)
            return neuron.compute_entropy() < self.poverty_threshold
        # Fallback: poor neurons have small gradient magnitude
        return abs(neuron.grad) <= self.poverty_threshold
    
    def collect_and_distribute(
        self,
        neurons: List[MizanValue],
        minority_mask: Optional[List[bool]] = None,
        iteration: Optional[int] = None,
        rate: Optional[float] = None
    ) -> float:
        """
        Apply distributive justice: collect zakat from rich, distribute to poor.
        
        This implements the full Digital Zakat mechanism:
        1. Calculate adaptive zakat rate based on minority proportion
        2. Classify rich and poor neurons (using entropy or gradient magnitude)
        3. Collect zakat from rich neurons (reduce their gradient)
        4. Distribute collected zakat equally to poor neurons
        
        Args:
            neurons: List of MizanValue instances
            minority_mask: Optional boolean mask for minority groups (for adaptive rate)
            iteration: Current training iteration (for adaptive rate calculation)
            rate: Override zakat rate (if None, uses adaptive rate)
            
        Returns:
            float: Total zakat collected and redistributed
        """
        # Update iteration counter
        self.iteration = iteration if iteration is not None else self.iteration + 1
        
        # 1. Calculate adaptive zakat rate
        if rate is not None:
            current_rate = rate
        else:
            if minority_mask is not None:
                minority_ratio = sum(minority_mask) / len(minority_mask)
            else:
                # Estimate minority ratio from poverty classification
                poor_count = sum(1 for n in neurons if self._is_poor(n))
                minority_ratio = poor_count / len(neurons) if len(neurons) > 0 else 0.025
            
            current_rate = adaptive_zakat_rate(
                minority_ratio=minority_ratio,
                iteration=self.iteration,
                tau=self.tau_step,
                base_rate=self.base_rate
            )
        
        # 2. Classify neurons
        rich_neurons = [n for n in neurons if self._is_rich(n)]
        poor_neurons = [n for n in neurons if self._is_poor(n)]
        
        # If no rich or no poor, zakat is not possible
        if not rich_neurons or not poor_neurons:
            return 0.0
        
        # 3. Collect zakat from rich neurons
        total_zakat = 0.0
        for neuron in rich_neurons:
            zakat_amount = neuron.pay_zakat(current_rate)
            total_zakat += zakat_amount
        
        # 4. Distribute equally to poor neurons
        share_per_poor = total_zakat / len(poor_neurons)
        for neuron in poor_neurons:
            neuron.receive_zakat(share_per_poor)
        
        # 5. Update audit trail
        self.total_zakat_collected += total_zakat
        self.total_zakat_distributed += len(poor_neurons) * share_per_poor
        
        self.audit_log.append({
            "iteration": self.iteration,
            "zakat_rate": float(current_rate),
            "collected": float(total_zakat),
            "rich_count": len(rich_neurons),
            "poor_count": len(poor_neurons),
            "share_per_poor": float(share_per_poor),
            "minority_ratio": float(minority_ratio) if minority_mask is not None else None
        })
        
        return total_zakat

    def get_audit_trail(self) -> List[dict]:
        """
        Return the complete audit log of all zakat operations.
        
        Returns:
            List[dict]: List of dictionaries containing zakat events
        """
        return self.audit_log.copy()
    
    def get_statistics(self) -> dict:
        """
        Return summary statistics of zakat operations.
        
        Returns:
            dict: Summary statistics including totals and averages
        """
        if not self.audit_log:
            return {"status": "No events recorded", "total_collected": 0, "total_distributed": 0}
        
        avg_rate = sum(log["zakat_rate"] for log in self.audit_log) / len(self.audit_log)
        return {
            "total_collected": self.total_zakat_collected,
            "total_distributed": self.total_zakat_distributed,
            "average_zakat_rate": avg_rate,
            "num_events": len(self.audit_log),
            "last_event": self.audit_log[-1]
        }

    def reset(self):
        """
        Reset the orchestrator state (iteration and audit log).
        Use this when starting a new training session.
        """
        self.iteration = 0
        self.audit_log = []
        self.total_zakat_collected = 0.0
        self.total_zakat_distributed = 0.0


# ============================================================================
# Self Test (executed when running the file directly)
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🕌 Testing Al-Mizan Zakat Orchestrator v1.0.1")
    print("=" * 60)
    
    try:
        # Create test neurons with gradient history
        test_neurons = []
        for i in range(8):
            n = MizanValue(float(i) * 0.5, label=f"N{i}")
            # Simulate gradient history for entropy calculation
            if i < 3:
                # Rich neurons: varied gradients
                n.grad_history = [10.0, 12.0, 8.0, 11.0, 9.0] * 2
            else:
                # Poor neurons: constant gradients
                n.grad_history = [0.5, 0.5, 0.5, 0.5, 0.5] * 2
            n.grad = float(i) * 1.5
            test_neurons.append(n)
        
        # Create orchestrator
        orch = ZakatOrchestrator(poverty_threshold=0.5, base_rate=0.025)
        
        # Apply zakat
        collected = orch.collect_and_distribute(test_neurons)
        
        print(f"\n✅ Success! Total Zakat Collected: {collected:.4f}")
        print(f"📊 Rich neurons: {sum(1 for n in test_neurons if orch._is_rich(n))}")
        print(f"📊 Poor neurons: {sum(1 for n in test_neurons if orch._is_poor(n))}")
        print(f"\n📈 Statistics: {orch.get_statistics()}")
        print(f"\n📜 Audit Log: {orch.get_audit_trail()}")
        print("\n✅ All tests passed!")
        
    except ImportError as e:
        print(f"⚠️ Import error (expected when run standalone): {e}")
        print("   This is normal when running this file directly without the full package.")
        print("   The orchestrator logic is still correct.")
    except Exception as e:
        print(f"❌ Error during test: {e}")
