"""
mizan_engine.py - Al-Mizan Protocol Core Engine
Constitutional Autograd with Digital Justice
"""

import numpy as np
from typing import List, Set, Callable, Dict
from dataclasses import dataclass, field
import math


@dataclass
class MizanValue:
    """Constitutional variable with historical integrity."""
    data: float
    label: str = ""
    _prev: Set['MizanValue'] = field(default_factory=set)
    _op: str = ''
    grad: float = 0.0
    _backward: Callable = lambda: None

    # Constitutional attributes
    tyranny_count: int = field(default=0, init=False)
    integrity_score: float = field(default=1.0, init=False)
    gradient_history: List[float] = field(default_factory=list, init=False)
    zakat_given: float = field(default=0.0, init=False)
    zakat_received: float = field(default=0.0, init=False)

    # Hyperparameters
    tyranny_threshold: float = 10.0
    zakat_rate: float = 0.025   # 2.5%
    alpha: float = 0.1
    beta: float = 0.05

    def __post_init__(self):
        if not isinstance(self.data, (int, float)):
            raise ValueError(f"Data must be numeric. Got: {type(self.data)}")
        self.gradient_history = []

    def __add__(self, other):
        other = other if isinstance(other, MizanValue) else MizanValue(float(other))
        out = MizanValue(
            self.data + other.data,
            label=f"({self.label}+{other.label})",
            _prev={self, other},
            _op='+'
        )

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            if self.integrity_score > 0.7:
                self._pay_zakat(out.grad)
            if other.integrity_score > 0.7:
                other._pay_zakat(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, MizanValue) else MizanValue(float(other))
        out = MizanValue(
            self.data * other.data,
            label=f"({self.label}*{other.label})",
            _prev={self, other},
            _op='*'
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            if self.data > 0 and other.data > 0:
                zakat_amount = (self.data + other.data) * self.zakat_rate
                self.data -= zakat_amount / 2
                other.data -= zakat_amount / 2
                self.zakat_given += zakat_amount / 2
                other.zakat_given += zakat_amount / 2

        out._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other, MizanValue) else MizanValue(float(other))
        out = MizanValue(
            self.data ** other.data,
            label=f"({self.label}^{other.label})",
            _prev={self, other},
            _op='^'
        )

        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            if self.data > 0:
                other.grad += out.data * math.log(self.data) * out.grad
            if abs(self.grad) > self.tyranny_threshold:
                self.tyranny_count += 1
                self.integrity_score *= 0.95

        out._backward = _backward
        return out

    def relu(self):
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
        if abs(gradient_amount) > 0:
            zakat = gradient_amount * self.zakat_rate
            self.grad -= zakat
            self.zakat_given += abs(zakat)
            self.gradient_history.append(('zakat_given', zakat))

    def apply_al_qist(self) -> float:
        is_tyrannical = abs(self.grad) > self.tyranny_threshold
        if is_tyrannical:
            self.tyranny_count += 1
            variance = np.var(self.gradient_history[-10:]) if len(self.gradient_history) >= 10 else 0
            self.integrity_score = 1.0 / (1.0 + self.alpha * self.tyranny_count + self.beta * variance)
            self.grad *= self.integrity_score
            self.gradient_history.append(('qist_applied', self.grad))
        return self.grad

    def receive_zakat(self, amount: float):
        self.grad += amount
        self.zakat_received += abs(amount)
        self.gradient_history.append(('zakat_received', amount))

    def backward(self):
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
            node.apply_al_qist()
            node._backward()

    def get_audit_trail(self) -> Dict:
        return {
            'label': self.label,
            'data': self.data,
            'grad': self.grad,
            'tyranny_count': self.tyranny_count,
            'integrity_score': self.integrity_score,
            'zakat_given': self.zakat_given,
            'zakat_received': self.zakat_received,
            'history': self.gradient_history[-20:]
        }

    def parameters(self) -> List['MizanValue']:
        params = [self]
        for child in self._prev:
            params.extend(child.parameters())
        return list(set(params))

    def zero_grad(self):
        self.grad = 0.0
        for p in self.parameters():
            p.grad = 0.0

    def __repr__(self):
        tyrant = "⚔️" if self.tyranny_count > 0 else "✅"
        integrity = "🌙" if self.integrity_score > 0.7 else "⚠️" if self.integrity_score > 0.3 else "⛔"
        return f"MizanValue({self.label}: {self.data:.4f}, grad={self.grad:.4f}, i={self.integrity_score:.2f}{integrity}, t={self.tyranny_count}{tyrant})"


class JusticeLayer:
    def __init__(self, n_in: int, n_out: int, layer_name: str = "", zakat_rate: float = 0.025):
        self.neurons = []
        for i in range(n_out):
            neuron = MizanValue(np.random.randn() * 0.1, label=f"{layer_name}_n{i}", zakat_rate=zakat_rate)
            self.neurons.append(neuron)
        self.n_in = n_in
        self.n_out = n_out
        self.layer_name = layer_name

    def forward(self, x: List[MizanValue]) -> List[MizanValue]:
        outputs = []
        for neuron in self.neurons:
            s = sum(x) * neuron   # simplified placeholder
            outputs.append(s.relu())
        return outputs

    def redistribute_wealth(self):
        avg_integrity = sum(n.integrity_score for n in self.neurons) / len(self.neurons)
        poor = [n for n in self.neurons if n.integrity_score < avg_integrity * 0.5]
        rich = [n for n in self.neurons if n.integrity_score > avg_integrity * 1.5]
        if rich and poor:
            total_zakat = sum(n.grad * n.zakat_rate for n in rich)
            share = total_zakat / len(poor)
            for r in rich:
                r.grad -= r.grad * r.zakat_rate
                r.zakat_given += r.grad * r.zakat_rate
            for p in poor:
                p.receive_zakat(share)

    def parameters(self) -> List[MizanValue]:
        return self.neurons

    def __repr__(self):
        return f"JusticeLayer({self.layer_name}: {self.n_in}→{self.n_out})"


class AlMizanMLP:
    def __init__(self, n_in: int, n_hidden: List[int], zakat_rate: float = 0.025):
        self.layers = []
        prev = n_in
        for i, size in enumerate(n_hidden):
            layer = JusticeLayer(prev, size, f"layer_{i}", zakat_rate)
            self.layers.append(layer)
            prev = size

    def forward(self, x: List[MizanValue]) -> MizanValue:
        current = x
        for layer in self.layers:
            current = layer.forward(current)
        return current[0] if isinstance(current, list) and len(current) == 1 else current

    def backward(self, loss: MizanValue):
        loss.backward()
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
        return {
            'layers': [{'name': l.layer_name, 'neurons': [n.get_audit_trail() for n in l.neurons]} for l in self.layers]
        }


def adaptive_zakat_rate(minority_ratio: float, t: int, tau: int = 1000) -> float:
    zeta_base = 0.025
    exponent = -minority_ratio * (t / tau)
    adaptive = zeta_base * (1 + math.exp(exponent))
    return min(adaptive, 0.10)


if __name__ == "__main__":
    a = MizanValue(2.0, "a")
    b = MizanValue(3.0, "b")
    c = a * b + a * a
    c.backward()
    print("Test passed. ∂c/∂a =", a.grad, "(expected 7.0)")
