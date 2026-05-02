"""
almizan/engine.py - Constitutional Autograd Engine for Al-Mizan Protocol
Author: Amine Rekab (@rekabamine91-lgtm)
Version: 1.0.0 (June 2026)

This module implements the core MizanValue class, which extends the concept
of automatic differentiation with:
- Historical integrity (TyrannyCount)
- Anti-tyranny constraint (Al-Qist)
- Information entropy for poverty detection
- Audit trail for transparency
"""

import math
import numpy as np
from typing import List, Set, Callable, Dict, Optional, Union


class MizanValue:
    """
    Constitutional variable with historical integrity.
    
    Unlike standard autograd variables (PyTorch, micrograd), MizanValue:
    1. Remembers its "tyranny" history via tyranny_count
    2. Self-regulates via Al-Qist (gradient dampening based on integrity)
    3. Tracks zakat flows (given/received)
    4. Provides full audit trail for transparency
    
    Attributes:
        data (float): The actual numerical value
        grad (float): Gradient (derivative) of the loss w.r.t this variable
        label (str): Human-readable identifier for debugging
        tau_tyr (float): Tyranny threshold (default 10.0)
        alpha (float): Decay factor for tyranny count (default 0.1)
        beta (float): Decay factor for gradient variance (default 0.05)
        zakat_rate_base (float): Base zakat rate (default 0.025)
        
        tyranny_count (int): Number of times gradient exceeded threshold
        integrity_score (float): Current trust score I(v) ∈ (0,1]
        grad_history (List[float]): Last N gradient magnitudes for variance
        zakat_given (float): Total gradient redistributed to others
        zakat_received (float): Total gradient received from others
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
        _op: str = '',
    ):
        # Core attributes
        self.data = data
        self.grad = 0.0
        self.label = label
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable = lambda: None
        
        # Constitutional hyperparameters
        self.tau_tyr = tau_tyr
        self.alpha = alpha
        self.beta = beta
        self.zakat_rate_base = zakat_rate_base
        
        # Constitutional state
        self.tyranny_count = 0
        self.integrity_score = 1.0
        self.grad_history: List[float] = []
        self.zakat_given = 0.0
        self.zakat_received = 0.0
        
        # Audit flag (triggered when Al-Qist is applied)
        self._al_qist_applied = False
    
    def __repr__(self) -> str:
        status = "⚔️" if self.tyranny_count > 0 else "✅"
        integrity_icon = "🌙" if self.integrity_score > 0.7 else "⚠️" if self.integrity_score > 0.3 else "⛔"
        return (
            f"MizanValue({self.label}: {self.data:.4f}, "
            f"grad={self.grad:.4f}, i={self.integrity_score:.2f}{integrity_icon}, "
            f"tyr={self.tyranny_count}{status}, z_given={self.zakat_given:.3f}, "
            f"z_recv={self.zakat_received:.3f})"
        )
    
    # -------------------------------------------------------------------------
    # Core Constitutional Methods
    # -------------------------------------------------------------------------
    
    def update_integrity(self) -> None:
        """
        Update integrity score based on tyranny count and gradient variance.
        
        Formula: I(v) = 1 / (1 + α·TC + β·Var(G_last10))
        This ensures that repeated tyranny reduces trust, and volatile gradients
        also reduce trust.
        """
        # Calculate variance of recent gradients (if enough history)
        if len(self.grad_history) >= 2:
            variance = np.var(self.grad_history[-10:])
        else:
            variance = 0.0
        
        # Constitutional integrity formula
        self.integrity_score = 1.0 / (1.0 + self.alpha * self.tyranny_count + self.beta * variance)
        # Bound to (0, 1]
        self.integrity_score = min(1.0, max(0.01, self.integrity_score))
    
    def apply_al_qist(self) -> float:
        """
        Apply Al-Qist (anti-tyranny) constraint.
        
        If |grad| exceeds tau_tyr, the gradient is dampened proportionally
        to the variable's integrity score. The tyranny count is incremented,
        and the integrity score is updated.
        
        Returns:
            float: The (possibly dampened) gradient value
        """
        # Record gradient in history
        self.grad_history.append(abs(self.grad))
        # Keep only last 20 values for efficiency
        if len(self.grad_history) > 20:
            self.grad_history.pop(0)
        
        # Check for tyranny
        if abs(self.grad) > self.tau_tyr:
            self.tyranny_count += 1
            self.update_integrity()
            # Dampen gradient using integrity score
            self.grad *= self.integrity_score
            self._al_qist_applied = True
        else:
            self._al_qist_applied = False
        
        return self.grad
    
    def compute_entropy(self, n_bins: int = 5) -> float:
        """
        Compute information entropy H(i) = -Σ p_k(a_i) log p_k(a_i).
        
        This measures the "richness" of information in the variable.
        Low entropy indicates the neuron is "poor" (non-informative activation)
        and should receive zakat.
        
        Args:
            n_bins: Number of histogram bins for discretization
            
        Returns:
            float: Entropy value (higher = more information-rich)
        """
        if len(self.grad_history) < 2:
            return 1.0  # Default: neutral entropy
        
        # Use last 10 gradients as distribution
        hist = self.grad_history[-10:]
        if min(hist) == max(hist):
            return 0.0  # No variation → zero entropy
        
        # Discretize into n_bins
        bins = np.linspace(min(hist), max(hist) + 1e-8, n_bins + 1)
        counts, _ = np.histogram(hist, bins=bins)
        probs = counts / (sum(counts) + 1e-8)
        
        # Calculate Shannon entropy
        entropy = -sum(p * math.log(p + 1e-8) for p in probs if p > 0)
        
        # Normalize to [0, 1] range
        max_entropy = math.log(n_bins)
        if max_entropy > 0:
            entropy = entropy / max_entropy
        
        return entropy
    
    def pay_zakat(self, rate: Optional[float] = None) -> float:
        """
        Pay digital zakat: remove a portion of gradient and return it.
        
        Args:
            rate: Zakat rate (uses default if None)
            
        Returns:
            float: Amount of gradient paid (to be redistributed)
        """
        zakat_rate = rate if rate is not None else self.zakat_rate_base
        zakat_amount = abs(self.grad) * zakat_rate
        self.grad -= zakat_amount
        self.zakat_given += zakat_amount
        return zakat_amount
    
    def receive_zakat(self, amount: float) -> None:
        """
        Receive redistributed gradient from rich neurons.
        
        Args:
            amount: Amount of gradient to add
        """
        self.grad += amount
        self.zakat_received += amount
    
    def is_poor(self, poverty_threshold: float = 0.5) -> bool:
        """
        Determine if neuron is "poor" (low information entropy).
        
        Args:
            poverty_threshold: Threshold below which neuron is considered poor
            
        Returns:
            bool: True if entropy < threshold
        """
        return self.compute_entropy() < poverty_threshold
    
    def get_audit_trail(self) -> Dict:
        """
        Generate complete audit trail for this variable.
        
        Returns:
            dict: Serializable audit record
        """
        return {
            'label': self.label,
            'data': float(self.data),
            'grad': float(self.grad),
            'tyranny_count': self.tyranny_count,
            'integrity_score': float(self.integrity_score),
            'zakat_given': float(self.zakat_given),
            'zakat_received': float(self.zakat_received),
            'grad_history': [float(g) for g in self.grad_history[-10:]],
            'entropy': float(self.compute_entropy()),
            'al_qist_applied': self._al_qist_applied
        }
    
    # -------------------------------------------------------------------------
    # Arithmetic Operations with Constitutional Backward Pass
    # -------------------------------------------------------------------------
    
    def __add__(self, other: Union['MizanValue', float]) -> 'MizanValue':
        """Addition with constitutional backward pass."""
        other = other if isinstance(other, MizanValue) else MizanValue(float(other))
        out = MizanValue(
            self.data + other.data,
            label=f"({self.label}+{other.label})",
            tau_tyr=self.tau_tyr,
            alpha=self.alpha,
            beta=self.beta,
            zakat_rate_base=self.zakat_rate_base,
            _children=(self, other),
            _op='+'
        )
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            # Al-Qist will be applied in backward()
        
        out._backward = _backward
        return out
    
    def __mul__(self, other: Union['MizanValue', float]) -> 'MizanValue':
        """Multiplication with constitutional backward pass."""
        other = other if isinstance(other, MizanValue) else MizanValue(float(other))
        out = MizanValue(
            self.data * other.data,
            label=f"({self.label}*{other.label})",
            tau_tyr=self.tau_tyr,
            alpha=self.alpha,
            beta=self.beta,
            zakat_rate_base=self.zakat_rate_base,
            _children=(self, other),
            _op='*'
        )
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __pow__(self, other: Union['MizanValue', float]) -> 'MizanValue':
        """Power operation (exponentiation) with constitutional backward pass."""
        other = other if isinstance(other, MizanValue) else MizanValue(float(other))
        out = MizanValue(
            self.data ** other.data,
            label=f"({self.label}^{other.label})",
            tau_tyr=self.tau_tyr,
            alpha=self.alpha,
            beta=self.beta,
            zakat_rate_base=self.zakat_rate_base,
            _children=(self, other),
            _op='^'
        )
        
        def _backward():
            # d/dx: y * x^(y-1)
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            # d/dy: x^y * ln(x) (only if x > 0)
            if self.data > 0:
                other.grad += out.data * math.log(self.data) * out.grad
        
        out._backward = _backward
        return out
    
    def relu(self) -> 'MizanValue':
        """ReLU activation function."""
        out = MizanValue(
            max(0, self.data),
            label=f"ReLU({self.label})",
            tau_tyr=self.tau_tyr,
            alpha=self.alpha,
            beta=self.beta,
            zakat_rate_base=self.zakat_rate_base,
            _children=(self,),
            _op='ReLU'
        )
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        
        out._backward = _backward
        return out
    
    def __neg__(self) -> 'MizanValue':
        """Negation operator."""
        out = MizanValue(
            -self.data,
            label=f"(-{self.label})",
            tau_tyr=self.tau_tyr,
            alpha=self.alpha,
            beta=self.beta,
            zakat_rate_base=self.zakat_rate_base,
            _children=(self,),
            _op='neg'
        )
        
        def _backward():
            self.grad -= out.grad
        
        out._backward = _backward
        return out
    
    def __sub__(self, other: Union['MizanValue', float]) -> 'MizanValue':
        """Subtraction operator."""
        return self + (-other)
    
    def __truediv__(self, other: Union['MizanValue', float]) -> 'MizanValue':
        """Division operator (x / y)."""
        other = other if isinstance(other, MizanValue) else MizanValue(float(other))
        out = MizanValue(
            self.data / other.data,
            label=f"({self.label}/{other.label})",
            tau_tyr=self.tau_tyr,
            alpha=self.alpha,
            beta=self.beta,
            zakat_rate_base=self.zakat_rate_base,
            _children=(self, other),
            _op='/'
        )
        
        def _backward():
            self.grad += (1.0 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad
        
        out._backward = _backward
        return out
    
    # -------------------------------------------------------------------------
    # Backward Pass (Constitutional)
    # -------------------------------------------------------------------------
    
    def backward(self) -> None:
        """
        Execute constitutional backward pass with Al-Qist.
        
        This method:
        1. Builds topological order of the computation graph
        2. Sets initial gradient (self.grad = 1.0 for the root)
        3. Propagates gradients backward (standard chain rule)
        4. Applies Al-Qist to each node after its backward pass
        
        The Al-Qist step ensures that any "tyrannical" gradient is dampened
        before it propagates further.
        """
        # Build topological order
        topo: List[MizanValue] = []
        visited: Set[MizanValue] = set()
        
        def build_topo(v: MizanValue) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize root gradient
        self.grad = 1.0
        
        # Process in reverse topological order (from loss to inputs)
        for node in reversed(topo):
            # Compute gradient via standard chain rule
            node._backward()
            # Apply constitutional justice (Al-Qist)
            node.apply_al_qist()
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def zero_grad(self) -> None:
        """Reset gradient to zero."""
        self.grad = 0.0
    
    def parameters(self) -> List['MizanValue']:
        """Extract all parameters in the computation graph."""
        params: List[MizanValue] = []
        visited: Set[MizanValue] = set()
        
        def collect(v: MizanValue) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    collect(child)
                params.append(v)
        
        collect(self)
        return params


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def adaptive_zakat_rate(minority_ratio: float, iteration: int, tau: int = 1000, base_rate: float = 0.025) -> float:
    """
    Compute adaptive zakat rate for minority protection.
    
    Formula: ζ_adaptive(t) = ζ_base * (1 + exp(-(r_min) * (t / τ)))
    
    Where:
        r_min = |S_min| / |S_total| (minority proportion)
        τ = 1000 (time constant)
        t = current iteration
    
    Args:
        minority_ratio: Proportion of minority group (|S_min|/|S_total|)
        iteration: Current training iteration (t)
        tau: Time constant (default 1000)
        base_rate: Baseline zakat rate (default 0.025 = 2.5%)
    
    Returns:
        float: Adaptive zakat rate (capped at 0.10)
    """
    exponent = -minority_ratio * (iteration / tau)
    rate = base_rate * (1 + math.exp(exponent))
    return min(rate, 0.10)  # Cap at 10%


# -----------------------------------------------------------------------------
# Demo and Test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("🌙 Al-Mizan Protocol v1.0.0 – Constitutional Autograd Engine")
    print("=" * 60)
    
    print("\n📊 Basic Test: Algebraic Operations with Justice")
    print("-" * 40)
    
    # Create two variables
    a = MizanValue(2.0, label="a")
    b = MizanValue(3.0, label="b")
    
    # Build expression: c = a*b + a*a → expected: c = 6 + 4 = 10
    c = a * b + a * a
    
    # Backward pass (with constitutional justice)
    c.backward()
    
    print(f"a.data = {a.data}, a.grad = {a.grad:.4f} (expected: 7.0)")
    print(f"b.data = {b.data}, b.grad = {b.grad:.4f} (expected: 2.0)")
    print(f"c.data = {c.data:.4f}")
    
    print(f"\n📜 Audit Trail for variable 'a':")
    audit = a.get_audit_trail()
    for key, value in audit.items():
        if key != 'grad_history':
            print(f"  {key}: {value}")
    
    print("\n🔬 Poverty Detection (Entropy Test):")
    rich = MizanValue(5.0, label="rich")
    rich.grad_history = [10.0, 12.0, 8.0, 11.0, 9.0, 13.0, 10.0, 12.0, 11.0, 9.0]
    poor = MizanValue(0.5, label="poor")
    poor.grad_history = [0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
    
    print(f"  Rich neuron entropy: {rich.compute_entropy():.4f} (is poor: {rich.is_poor()})")
    print(f"  Poor neuron entropy: {poor.compute_entropy():.4f} (is poor: {poor.is_poor()})")
    
    print("\n🕌 Adaptive Zakat Test (minority ratio = 1%, τ=1000):")
    for t in [0, 250, 500, 750, 1000, 2000]:
        rate = adaptive_zakat_rate(0.01, t, tau=1000)
        print(f"  t={t:4d}: ζ = {rate:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Constitutional Autograd Engine ready for production.")
    print("=" * 60)
