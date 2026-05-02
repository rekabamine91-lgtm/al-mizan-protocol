"""
mizan_engine.py - Al-Mizan Protocol Core Engine
Official implementation of Constitutional Autograd with Digital Justice

Key innovations over micrograd (Karpathy):
- TyrannyCount: Historical memory of gradient dominance
- IntegrityScore: Self-regulation prevents algorithmic tyranny
- Digital Zakat: Automatic redistribution to info-poor neurons
- Adaptive Zakat: τ=1000 protection window for minorities
"""

import numpy as np
from typing import List, Set, Callable, Optional, Tuple, Dict
from dataclasses import dataclass, field
import math


@dataclass
class MizanValue:
    """
    The Constitutional Variable - a responsible entity with historical integrity.
    Unlike Karpathy's passive Scalar, MizanValue remembers and self-regulates.
    """
    
    data: float
    label: str = ""
    _prev: Set['MizanValue'] = field(default_factory=set)
    _op: str = ''
    grad: float = 0.0
    _backward: Callable = lambda: None
    
    # 🏛️ Constitutional attributes (The Al-Mizan Innovations)
    tyranny_count: int = field(default=0, init=False)
    integrity_score: float = field(default=1.0, init=False)
    gradient_history: List[float] = field(default_factory=list, init=False)
    zakat_received: float = field(default=0.0, init=False)
    zakat_given: float = field(default=0.0, init=False)
    
    # Hyperparameters (can be set globally)
    tyranny_threshold: float = 10.0
    zakat_rate: float = 0.025  # 2.5% Digital Zakat
    integrity_decay_alpha: float = 0.1
    integrity_decay_beta: float = 0.05
    
    def __post_init__(self):
        """Initialize tracking structures"""
        if not isinstance(self.data, (int, float)):
            raise ValueError(f"Data must be numeric. Got: {type(self.data)}")
        self.gradient_history = []
    
    def __add__(self, other):
        """Addition with justice awareness"""
        other = other if isinstance(other, MizanValue) else MizanValue(float(other))
        out = MizanValue(
            self.data + other.data,
            label=f"({self.label}+{other.label})",
            _prev={self, other},
            _op='+'
        )
        
        def _backward():
            # Standard chain rule
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            
            # 🕌 Digital Zakat: Info-rich help info-poor
            if self.integrity_score > 0.7:
                self._pay_zakat(out.grad)
            if other.integrity_score > 0.7:
                other._pay_zakat(out.grad)
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        """Multiplication with redistributive justice"""
        other = other if isinstance(other, MizanValue) else MizanValue(float(other))
        out = MizanValue(
            self.data * other.data,
            label=f"({self.label}*{other.label})",
            _prev={self, other},
            _op='*'
        )
        
        def _backward():
            # Standard chain rule for multiplication
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
            # 🕌 Zakat on multiplication (compound growth pays more)
            if self.data > 0 and other.data > 0:
                zakat_amount = (self.data + other.data) * self.zakat_rate
                self.data -= zakat_amount / 2
                other.data -= zakat_amount / 2
                self.zakat_given += zakat_amount / 2
                other.zakat_given += zakat_amount / 2
        
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        """Power operation with integrity monitoring"""
        other = other if isinstance(other, MizanValue) else MizanValue(float(other))
        out = MizanValue(
            self.data ** other.data,
            label=f"({self.label}^{other.label})",
            _prev={self, other},
            _op='^'
        )
        
        def _backward():
            # Chain rule for power: d/dx (x^y) = y * x^(y-1)
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            # d/dy (x^y) = x^y * log(x)
            other.grad += out.data * math.log(abs(self.data) + 1e-8) * out.grad
            
            # Monitor for tyranny in exponentiation
            if abs(self.grad) > self.tyranny_threshold:
                self.tyranny_count += 1
                self.integrity_score *= 0.95
        
        out._backward = _backward
        return out
    
    def relu(self):
        """ReLU activation with justice preservation"""
        out = MizanValue(
            max(0, self.data),
            label=f"ReLU({self.label})",
            _prev={self},
            _op='ReLU'
        )
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        
        out._backward = _backward
        return out
    
    def _pay_zakat(self, gradient_amount: float):
        """
        🕌 Digital Zakat: Redistribute learning capital
        When a variable is "rich" (high integrity), it gives away
        a portion of its gradient to poorer variables.
        """
        if abs(gradient_amount) > 0:
            zakat = gradient_amount * self.zakat_rate
            self.grad -= zakat
            self.zakat_given += abs(zakat)
            self.gradient_history.append(('zakat_given', zakat))
    
    def apply_al_qist(self) -> float:
        """
        ⚖️ Al-Qist: The anti-tyranny constraint
        If a gradient exceeds threshold, it is dampened proportionally
        to how many times it has been tyrannical before.
        """
        is_tyrannical = abs(self.grad) > self.tyranny_threshold
        
        if is_tyrannical:
            self.tyranny_count += 1
            # Update integrity score based on history
            self.integrity_score = 1.0 / (
                1.0 + self.integrity_decay_alpha * self.tyranny_count 
                + self.integrity_decay_beta * np.var(self.gradient_history[-10:]) 
                if len(self.gradient_history) >= 10 else 0
            )
            # Dampen the tyrannical gradient
            self.grad *= self.integrity_score
            self.gradient_history.append(('qist_applied', self.grad))
        
        return self.grad
    
    def receive_zakat(self, amount: float):
        """
        Receive redistributed learning capital from richer neurons
        """
        self.grad += amount
        self.zakat_received += abs(amount)
        self.gradient_history.append(('zakat_received', amount))
    
    def backward(self):
        """
        🏛️ Constitutional Backward Pass
        The crown jewel of Al-Mizan: applies Al-Qist before propagation
        """
        # Build topological order
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Set initial gradient
        self.grad = 1.0
        
        # 🏛️ Constitutional backward pass with Al-Qist
        for node in reversed(topo):
            # Apply justice before backward
            node.apply_al_qist()
            node._backward()
    
    def get_audit_trail(self) -> Dict:
        """
        📜 Transparency Log - The proof of algorithmic justice
        Returns complete history of variable's behavior
        """
        return {
            'label': self.label,
            'data': self.data,
            'grad': self.grad,
            'tyranny_count': self.tyranny_count,
            'integrity_score': self.integrity_score,
            'zakat_given': self.zakat_given,
            'zakat_received': self.zakat_received,
            'history': self.gradient_history[-20:],  # Last 20 events
            'is_tyrannical': abs(self.grad) > self.tyranny_threshold
        }
    
    def parameters(self) -> List['MizanValue']:
        """Extract all parameters in the computational graph"""
        params = [self]
        for child in self._prev:
            params.extend(child.parameters())
        return list(set(params))
    
    def zero_grad(self):
        """Reset gradients before next iteration"""
        self.grad = 0.0
        for param in self.parameters():
            param.grad = 0.0
    
    def __repr__(self):
        tyranny_symbol = "⚔️" if self.tyranny_count > 0 else "✅"
        integrity_symbol = "🌙" if self.integrity_score > 0.7 else "⚠️" if self.integrity_score > 0.3 else "⛔"
        return f"MizanValue({self.label}: {self.data:.4f}, grad: {self.grad:.4f}, i:{self.integrity_score:.2f}{integrity_symbol}, t:{self.tyranny_count}{tyranny_symbol})"


class JusticeLayer:
    """
    🏛️ Constitutional Layer - Multiple neurons with collective justice
    Maintains equilibrium across the entire layer, not just individual neurons.
    """
    
    def __init__(self, n_in: int, n_out: int, layer_name: str = "", zakat_rate: float = 0.025):
        self.neurons: List[MizanValue] = []
        self.layer_name = layer_name
        self.zakat_rate = zakat_rate
        
        for i in range(n_out):
            neuron = MizanValue(
                data=np.random.randn() * 0.1,
                label=f"{layer_name}_n{i}",
                zakat_rate=zakat_rate
            )
            self.neurons.append(neuron)
        
        self.n_in = n_in
        self.n_out = n_out
    
    def forward(self, x: List[MizanValue]) -> List[MizanValue]:
        """
        Forward pass with fair distribution
        """
        outputs = []
        for i, neuron in enumerate(self.neurons):
            # Weighted sum of inputs (simplified for demo)
            weighted_sum = sum(x) * neuron
            outputs.append(weighted_sum.relu())
        return outputs
    
    def redistribute_wealth(self):
        """
        🕌 Collective Zakat: Redistribute from rich to poor neurons in the layer
        This prevents "knowledge monopolies" within the same layer
        """
        # Calculate average integrity
        avg_integrity = sum(n.integrity_score for n in self.neurons) / len(self.neurons)
        
        # Identify poor neurons (below poverty line)
        poor_neurons = [n for n in self.neurons if n.integrity_score < avg_integrity * 0.5]
        rich_neurons = [n for n in self.neurons if n.integrity_score > avg_integrity * 1.5]
        
        if rich_neurons and poor_neurons:
            # Redistribute from rich to poor
            total_zakat = sum(n.grad * self.zakat_rate for n in rich_neurons)
            per_poor_share = total_zakat / len(poor_neurons)
            
            for rich in rich_neurons:
                rich.grad -= rich.grad * self.zakat_rate
                rich.zakat_given += rich.grad * self.zakat_rate
            
            for poor in poor_neurons:
                poor.receive_zakat(per_poor_share)
    
    def parameters(self) -> List[MizanValue]:
        return self.neurons
    
    def __repr__(self):
        return f"JusticeLayer({self.layer_name}: {self.n_in}→{self.n_out}, neurons={len(self.neurons)})"


class AlMizanMLP:
    """
    🏛️ The Complete Constitutional Neural Network
    Multi-layer perceptron with justice at every level
    """
    
    def __init__(self, n_in: int, n_out: List[int], zakat_rate: float = 0.025):
        self.layers = []
        prev_size = n_in
        for i, size in enumerate(n_out):
            layer = JusticeLayer(prev_size, size, f"layer_{i}", zakat_rate)
            self.layers.append(layer)
            prev_size = size
        
        self.zakat_rate = zakat_rate
    
    def forward(self, x: List[MizanValue]) -> MizanValue:
        """
        Forward pass through all constitutional layers
        """
        current = x
        for layer in self.layers:
            current = layer.forward(current)
        # Return single value (assuming last layer has 1 neuron)
        return current[0] if isinstance(current, list) else current
    
    def backward(self, loss: MizanValue):
        """
        Constitutional backward pass with justice at every layer
        """
        loss.backward()
        
        # Apply collective redistribution after backward
        for layer in self.layers:
            layer.redistribute_wealth()
    
    def parameters(self) -> List[MizanValue]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
    
    def get_full_audit(self) -> Dict:
        """
        📜 Complete transparency audit for the entire network
        """
        audit = {
            'layers': [],
            'total_tyranny_events': 0,
            'total_zakat_given': 0.0,
            'total_zakat_received': 0.0
        }
        
        for layer in self.layers:
            layer_audit = {
                'name': layer.layer_name,
                'neurons': [n.get_audit_trail() for n in layer.neurons]
            }
            audit['layers'].append(layer_audit)
            
            for n in layer.neurons:
                audit['total_tyranny_events'] += n.tyranny_count
                audit['total_zakat_given'] += n.zakat_given
                audit['total_zakat_received'] += n.zakat_received
        
        return audit


# ============================================================================
# 🎯 Adaptive Zakat for Minority Protection
# ============================================================================

def adaptive_zakat_rate(minority_ratio: float, iteration: int, tau: int = 1000) -> float:
    """
    🕌 Adaptive Zakat Formula (from Whitepaper Section 5)
    
    ζ_adaptive = ζ_base * (1 + exp(-|S_min|/|S_total| * t / τ))
    
    Where:
    - ζ_base = 0.025 (2.5%)
    - τ = 1000 (Sovereign Decay Constant)
    - t = current iteration
    
    This ensures minority groups receive enhanced protection that
    naturally decays as the model converges.
    """
    zakat_base = 0.025
    exponent = -minority_ratio * (iteration / tau)
    adaptive_rate = zakat_base * (1 + math.exp(exponent))
    return min(adaptive_rate, 0.10)  # Cap at 10%


# ============================================================================
# 📜 Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🌙 Al-Mizan Protocol v1.2.2 - Constitutional Autograd")
    print("=" * 60)
    
    # Demonstrate basic functionality
    print("\n📊 Basic Operations Test:")
    a = MizanValue(2.0, label="a")
    b = MizanValue(3.0, label="b")
    c = a * b + a * a  # c = a*b + a² = 6 + 4 = 10
    c.backward()
    
    print(f"c = {c.data}")
    print(f"∂c/∂a = {a.grad} (expected: b + 2a = 3 + 4 = 7)")
    print(f"∂c/∂b = {b.grad} (expected: a = 2)")
    
    # Show audit trail
    print("\n📜 Audit Trail for variable 'a':")
    audit = a.get_audit_trail()
    for key, value in audit.items():
        if key != 'history':
            print(f"  {key}: {value}")
    
    # Test adaptive zakat
    print("\n🕌 Adaptive Zakat Test (minority ratio = 0.01, 1%):")
    for t in [0, 250, 500, 750, 1000, 2000]:
        rate = adaptive_zakat_rate(0.01, t)
        print(f"  t={t:4d}: ζ = {rate:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Al-Mizan Engine ready. Digital Sovereignty engaged.")
    print("=" * 60)
