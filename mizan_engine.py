"""
engine.py - Constitutional Autograd Engine for Al-Mizan Protocol
============================================================================
Core implementation of MizanValue with:
- Historical integrity (TyrannyCount)
- Anti-tyranny constraint (Al-Qist)
- Information entropy for poverty detection
- Digital Zakat (pay/receive)
- Bias operator B(G) for group fairness

Author: Amine Rekab (@rekabamine91-lgtm)
Version: 1.1.0 (June 2026)
============================================================================
"""

import numpy as np
import math
from typing import List, Set, Callable, Dict, Optional, Union


class MizanValue:
    """
    Constitutional variable with historical integrity and self-regulation.
    Integrated with Al-Mizan Protocol v1.1 standards.
    """
    
    def __init__(
        self, 
        data: float, 
        label: str = "",
        tau_tyr: float = 10.0,
        alpha: float = 0.1,
        beta: float = 0.05,
        zakat_rate_base: float = 0.025,
        _children: tuple = (),
        _op: str = ''
    ):
        self.data = data
        self.grad = 0.0
        self.label = label
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable = lambda: None
        
        self.tau_tyr = tau_tyr
        self.alpha = alpha
        self.beta = beta
        self.zakat_rate_base = zakat_rate_base
        
        self.tyranny_count = 0
        self.integrity_score = 1.0
        self.grad_history: List[float] = []
        self.zakat_given = 0.0
        self.zakat_received = 0.0
        
        self._al_qist_applied = False
    
    def __repr__(self) -> str:
        status = "⚔️" if self.tyranny_count > 0 else "✅"
        integrity_icon = "🌙" if self.integrity_score > 0.7 else "⚠️" if self.integrity_score > 0.3 else "⛔"
        return (f"MizanValue({self.label}: {self.data:.4f}, "
                f"grad={self.grad:.4f}, i={self.integrity_score:.2f}{integrity_icon}, "
                f"tyr={self.tyranny_count}{status})")
    
    def compute_entropy(self, n_bins: int = 5) -> float:
        """
        Calculates information entropy H(i) = -∑ p_k(a_i) log p_k(a_i)
        Used for poverty detection in learning dynamics.
        """
        if len(self.grad_history) < 2:
            return 1.0
        
        grads = np.array(self.grad_history[-20:])
        if np.all(grads == grads[0]):
            return 0.0
        
        bins = np.linspace(np.min(grads), np.max(grads) + 1e-8, n_bins + 1)
        counts, _ = np.histogram(grads, bins=bins)
        probs = counts / (np.sum(counts) + 1e-8)
        
        entropy = -np.sum([p * np.log(p + 1e-8) for p in probs if p > 0])
        
        max_entropy = np.log(n_bins)
        if max_entropy > 0:
            entropy = entropy / max_entropy
        
        return entropy
    
    def update_integrity(self) -> None:
        """
        Updates integrity score I(v) = 1 / (1 + α·TC + β·Var(G))
        """
        if len(self.grad_history) >= 2:
            variance = np.var(self.grad_history[-10:])
        else:
            variance = 0.0
        
        self.integrity_score = 1.0 / (1.0 + self.alpha * self.tyranny_count + self.beta * variance)
        self.integrity_score = min(1.0, max(0.01, self.integrity_score))
    
    def apply_al_qist(self) -> float:
        """
        Anti-tyranny constraint to dampen extreme gradients.
        """
        self.grad_history.append(abs(self.grad))
        if len(self.grad_history) > 30:
            self.grad_history.pop(0)
        
        if abs(self.grad) > self.tau_tyr:
            self.tyranny_count += 1
            self.update_integrity()
            self.grad *= self.integrity_score
            self._al_qist_applied = True
        else:
            self._al_qist_applied = False
        
        return self.grad
    
    def pay_zakat(self, rate: Optional[float] = None) -> float:
        """
        Deducts zakat from the gradient (Information Capital).
        """
        z_rate = rate if rate is not None else self.zakat_rate_base
        z_amount = self.grad * z_rate
        self.grad -= z_amount
        self.zakat_given += abs(z_amount)
        return abs(z_amount)
    
    def receive_zakat(self, amount: float) -> None:
        """
        Receives zakat support for weak gradients.
        """
        self.grad += amount
        self.zakat_received += abs(amount)
    
    def is_poor(self, poverty_threshold: float = 0.5) -> bool:
        """
        Detects if the neuron is information-poor (θ).
        """
        return self.compute_entropy() < poverty_threshold
    
    def get_audit_trail(self) -> Dict:
        """
        Returns full serializable audit log for the variable.
        """
        return {
            'label': self.label,
            'data': float(self.data),
            'grad': float(self.grad),
            'tyranny_count': self.tyranny_count,
            'integrity_score': float(self.integrity_score),
            'zakat_given': float(self.zakat_given),
            'zakat_received': float(self.zakat_received),
            'entropy': float(self.compute_entropy()),
            'al_qist_applied': self._al_qist_applied
        }
    
    def __add__(self, other: Union['MizanValue', float]) -> 'MizanValue':
        other = other if isinstance(other, MizanValue) else MizanValue(float(other))
        out = MizanValue(
            self.data + other.data,
            label=f"({self.label}+{other.label})",
            _children=(self, other),
            _op='+'
        )
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other: Union['MizanValue', float]) -> 'MizanValue':
        other = other if isinstance(other, MizanValue) else MizanValue(float(other))
        out = MizanValue(
            self.data * other.data,
            label=f"({self.label}*{other.label})",
            _children=(self, other),
            _op='*'
        )
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other: Union['MizanValue', float]) -> 'MizanValue':
        other = other if isinstance(other, MizanValue) else MizanValue(float(other))
        out = MizanValue(
            self.data ** other.data,
            label=f"({self.label}^{other.label})",
            _children=(self, other),
            _op='^'
        )
        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            if self.data > 0:
                other.grad += out.data * math.log(self.data) * out.grad
        out._backward = _backward
        return out
    
    def relu(self) -> 'MizanValue':
        out = MizanValue(
            max(0, self.data),
            label=f"ReLU({self.label})",
            _children=(self,),
            _op='ReLU'
        )
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def __neg__(self) -> 'MizanValue': return self * -1
    def __sub__(self, other) -> 'MizanValue': return self + (-other)
    def __truediv__(self, other) -> 'MizanValue': return self * (other**-1)
    
    def backward(self) -> None:
        """
        Constitutional backward pass with Al-Qist enforcement.
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
            node.apply_al_qist()
    
    def zero_grad(self) -> None:
        self.grad = 0.0


def compute_bias_operator(group_gradients: List[float]) -> float:
    """
    Implements B(G) formula from Technical Paper v1.1
    B(G) = mean((G_group - G_total)^2)
    """
    if not group_gradients:
        return 0.0
    total_mean = np.mean(group_gradients)
    bias_score = np.mean([(g - total_mean) ** 2 for g in group_gradients])
    return bias_score


def adaptive_zakat_rate(
    minority_ratio: float, 
    iteration: int, 
    tau: int = 1000, 
    base_rate: float = 0.025
) -> float:
    """
    Calculates adaptive zakat rate based on minority status.
    Formula: ζ_adaptive(t) = ζ_base * (1 + exp(-r_min * t/τ))
    """
    exponent = -minority_ratio * (iteration / tau)
    rate = base_rate * (1 + math.exp(exponent))
    return min(rate, 0.10)
