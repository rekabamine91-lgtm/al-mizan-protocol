"""
zakat_manager.py - Digital Zakat Orchestrator for Al-Mizan Protocol
============================================================================
Orchestrates the redistribution of Information Capital (Gradients) based 
on Islamic economic principles. Integrated with Al-Mizan v1.1 standards.

Key Features:
- Information-based poverty detection (Entropy-based)
- Collective Zakat Pool management
- Adaptive redistribution logic
- Audit trail for distributive justice

Author: Amine Rekab (@rekabamine91-lgtm)
Version: 1.1.0 (June 2026)
============================================================================
"""

from typing import List, Optional, Dict
import numpy as np

try:
    from .engine import MizanValue, adaptive_zakat_rate, compute_bias_operator
except (ImportError, ValueError):
    from engine import MizanValue, adaptive_zakat_rate, compute_bias_operator


class ZakatOrchestrator:
    """
    Manages the lifecycle of digital zakat collection and distribution.
    """
    
    def __init__(
        self,
        poverty_threshold: float = 0.5,
        base_rate: float = 0.025,
        tau_step: int = 1000
    ):
        self.poverty_threshold = poverty_threshold
        self.base_rate = base_rate
        self.tau_step = tau_step
        self.iteration = 0
        self.total_collected = 0.0
        self.audit_log: List[Dict] = []
        
    def collect_and_distribute(
        self, 
        neurons: List[MizanValue], 
        minority_mask: Optional[List[bool]] = None
    ) -> float:
        """
        Executes a single cycle of distributive justice.
        """
        self.iteration += 1
        
        # 1. Calculate adaptive rate based on minority status
        if minority_mask is not None:
            m_ratio = sum(minority_mask) / len(minority_mask)
        else:
            # Estimate minority ratio based on information poverty
            poor_count = sum(1 for n in neurons if n.is_poor(self.poverty_threshold))
            m_ratio = poor_count / len(neurons) if neurons else 0.0
            
        current_rate = adaptive_zakat_rate(
            minority_ratio=m_ratio,
            iteration=self.iteration,
            tau=self.tau_step,
            base_rate=self.base_rate
        )
        
        # 2. Identify 'Rich' and 'Poor' based on entropy
        rich_neurons = [n for n in neurons if not n.is_poor(self.poverty_threshold)]
        poor_neurons = [n for n in neurons if n.is_poor(self.poverty_threshold)]
        
        if not rich_neurons or not poor_neurons:
            return 0.0
            
        # 3. Collection (Al-Qist & Zakat)
        pool = 0.0
        for n in rich_neurons:
            pool += n.pay_zakat(current_rate)
            
        # 4. Distribution (Equal redistribution among the poor)
        share = pool / len(poor_neurons)
        for n in poor_neurons:
            n.receive_zakat(share)
            
        # 5. Global Metric: Bias Operator B(G)
        bias_score = compute_bias_operator([n.grad for n in neurons])
        
        # 6. Audit Logging
        self.total_collected += pool
        self.audit_log.append({
            'iteration': self.iteration,
            'collected': pool,
            'zakat_rate': current_rate,
            'rich_count': len(rich_neurons),
            'poor_count': len(poor_neurons),
            'bias_score': bias_score
        })
        
        return pool

    def get_audit_trail(self) -> List[Dict]:
        return self.audit_log

    def reset(self):
        self.iteration = 0
        self.total_collected = 0.0
        self.audit_log = []
