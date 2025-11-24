import numpy as np

class Initializer:
    """Base class for all weight initializers."""
    def init_weights(self, n_in, n_out): # W^[l]
        raise NotImplementedError

    def init_bias(self, n_out): # b^[l]
        raise NotImplementedError

class ConstantInitializer(Initializer):
    def __init__(self, weight=1.0, bias=0.0):
        self.weight = weight
        self.bias = bias

    def init_weights(self, n_in, n_out):
        return np.full((n_in, n_out), self.weight)

    def init_bias(self, n_out):
        return np.full((n_out,), self.bias)

class NormalInitializer(Initializer):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
    
    def init_weights(self, n_in, n_out):
        return np.random.normal(self.mean, self.std, (n_out, n_in))
    
    def init_bias(self, n_out):
        return np.zeros((n_out, 1))

# Specifically for ReLU activation function
class HeInitializer(Initializer):
    def init_weights(self, n_in, n_out):
        std = np.sqrt(2 / n_in)
        return np.random.normal(0, std, (n_out, n_in))
    
    def init_bias(self, n_out):
        return np.zeros((n_out, 1))