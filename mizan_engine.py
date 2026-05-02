import math

class MizanValue:
    """ محرك الميزان: جبر اجتماعي للذكاء الاصطناعي """
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        # TyrannyCount: لمراقبة تراكم الأوزان ومنع الاستبداد الرقمي
        self.tyranny_count = 0 

    def __repr__(self):
        return f"MizanValue(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, MizanValue) else MizanValue(other)
        out = MizanValue(self.data + other.data, (self, other), '+')
        
        def _backward():
            # منطق القسط: توزيع التدرج بالتساوي
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, MizanValue) else MizanValue(other)
        out = MizanValue(self.data * other.data, (self, other), '*')
        
        def _backward():
            # Adaptive Zakat: إعادة توزيع التدرج بناءً على المساهمة
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = MizanValue(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def apply_mizan_protocol(self):
        """ تطبيق بروتوكول الميزان النهائي v1.2.2 """
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
            # منطق منع الاستبداد: إذا زاد التدرج عن حد معين يتم كبحه
            if abs(node.grad) > 10: 
                node.tyranny_count += 1
                node.grad *= 0.9  # كبح التدرج المستبد
