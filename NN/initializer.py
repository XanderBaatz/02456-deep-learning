import numpy as np

class Initializer:
    def init_weights(self, n_in, n_out):
        raise NotImplementedError

    def init_bias(self, n_out):
        raise NotImplementedError


class ConstantInitializer(Initializer):
    def __init__(self, weight=0.01, bias=0.0):
        self.weight = float(weight)
        self.bias = float(bias)

    def init_weights(self, n_in, n_out):
        return np.full((n_out, n_in), self.weight, dtype=float)

    def init_bias(self, n_out):
        return np.full((n_out, 1), self.bias, dtype=float)


class NormalInitializer(Initializer):
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def init_weights(self, n_in, n_out):
        return np.random.normal(self.mean, self.std, (n_out, n_in)).astype(float)

    def init_bias(self, n_out):
        return np.zeros((n_out, 1), dtype=float)


# Specifically for Tanh and Sigmoid
class XavierInitializer(Initializer):
    def init_weights(self, n_in, n_out):
        std = np.sqrt(2.0 / (n_in + n_out))
        return np.random.normal(0, std, (n_out, n_in))
    
    def init_bias(self, n_out):
        return np.zeros((n_out, 1))


# Specifically for ReLU activation function
class HeInitializer(Initializer):
    def init_weights(self, n_in, n_out):
        std = np.sqrt(2.0 / n_in)
        return np.random.normal(0, std, (n_out, n_in))
    
    def init_bias(self, n_out):
        return np.zeros((n_out, 1))