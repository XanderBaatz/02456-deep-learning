import numpy as np

class Activation:
    """Base class for activation functions."""
    def forward(self, Z):
        raise NotImplementedError

    def backward(self, dA):
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0.0, Z)

    def backward(self, dA):
        return dA * (self.Z > 0).astype(float)


class Sigmoid(Activation):
    def forward(self, Z):
        self.Z = Z
        self.A = 1 / (1 + np.exp(-Z))
        return self.A
    
    def backward(self, dA):
        return dA * self.A * (1 - self.A)


class Tanh(Activation):
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA):
        return dA * (1.0 - self.A**2)


class Identity(Activation):
    def forward(self, Z):
        self.Z = Z
        return Z

    def backward(self, dA):
        return dA


# Note: softmax is technically not an activation function, but we disguise it as one for easier implementation
class Softmax(Activation):
    def forward(self, Z):
        # Z: (n_out, m)
        Z_shift = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_shift)
        self.A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        return self.A

    def backward(self, dA):
        # handled with cross-entropy; raise if used alone
        raise NotImplementedError("Softmax derivative should be combined with cross-entropy.")