from math import exp, log, tanh as math_tanh

class Var:
    """
    A variable which holds a float and enables gradient computations.
    """

    def __init__(self, val: float, grad_fn=lambda: []):
        assert isinstance(val, float), "val must be a float"
        self.v = val
        self.grad_fn = grad_fn
        self.grad = 0.0

    def backprop(self, bp):
        self.grad += bp
        for input_var, local_grad in self.grad_fn():
            input_var.backprop(local_grad * bp)

    def backward(self):
        self.backprop(1.0)

    # Arithmetic operations (unchanged)
    def __add__(self, other):
        return Var(self.v + other.v, lambda: [(self, 1.0), (other, 1.0)])

    def __mul__(self, other):
        return Var(self.v * other.v, lambda: [(self, other.v), (other, self.v)])

    def __pow__(self, power):
        assert isinstance(power, (float, int)), "power must be float or int"
        return Var(self.v ** power, lambda: [(self, power * self.v ** (power - 1))])

    def __neg__(self):
        return Var(-1.0) * self

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __repr__(self):
        return f"Var(v={self.v:.4f}, grad={self.grad:.4f})"

    # Activation functions (including new ones)
    def relu(self):
        return Var(self.v if self.v > 0.0 else 0.0,
                   lambda: [(self, 1.0 if self.v > 0.0 else 0.0)])

    def identity(self):
        """
        Derivative: d/dx(x) = 1
        """
        # The value is just self.v, and the local gradient is 1.0
        return Var(self.v, lambda: [(self, 1.0)])

    def tanh(self):
        """
        Derivative: d/dx(tanh(x)) = 1 - tanh(x)^2
        Uses math.tanh for stability.
        """
        t = math_tanh(self.v)
        # The local gradient is 1.0 - t**2
        return Var(t, lambda: [(self, 1.0 - t**2)])

    def sigmoid(self):
        """
        Derivative: d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
        """
        # Calculate sigmoid(x)
        s = 1.0 / (1.0 + exp(-self.v))
        # The local gradient is s * (1.0 - s)
        return Var(s, lambda: [(self, s * (1.0 - s))])