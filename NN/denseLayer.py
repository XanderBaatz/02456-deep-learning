import numpy as np
from .activations import Softmax
from .initializer import NormalInitializer, XavierInitializer, HeInitializer

class Layer:
    def forward(self, A_prev):
        raise NotImplementedError

    def backward(self, dA):
        raise NotImplementedError


class DenseLayer(Layer):
    """
    Dense layer using weight shape (n_out, n_in) and bias (n_out, 1).
    activation: an Activation instance
    """
    def __init__(
        self, 
        n_in:int,
        n_out:int, activation,
        initializer=None, 
        l2_coeff:float=0.0
        ):
        
        if initializer is None:
            if activation.__class__.__name__.lower().startswith("relu"):
                initializer = HeInitializer()
            if activation.__class__.__name__.lower().startswith("tanh") or activation.__class__.__name__.lower().startswith("sigmoid"):
                initializer = XavierInitializer()
            else:
                initializer = NormalInitializer(mean=0.0, std=0.01)

        self.initializer = initializer
        self.W = self.initializer.init_weights(n_in, n_out)  
        self.b = self.initializer.init_bias(n_out)           
        self.activation = activation
        self.l2_coeff = float(l2_coeff) # lambda

        # Caching during forward/backward
        self.Z = None
        self.A_prev = None

    def __repr__(self):
        act_name = self.activation.__class__.__name__
        return (
                      #We gave added more interesting metrics to the repr

            f"DenseLayer({self.W.shape[1]} -> {self.W.shape[0]}, Activation={act_name})\n"
            f"Weights shape: {self.W.shape}, min: {self.W.min():.6f}, max: {self.W.max():.6f}, mean: {self.W.mean():.6f}\n"
            f"Biases shape: {self.b.shape}, min: {self.b.min():.6f}, max: {self.b.max():.6f}, mean: {self.b.mean():.6f}"
        )

    def forward(self, A_prev):
        # Save for backward
        self.A_prev = A_prev                     # (n_in, m)
        self.Z = np.dot(self.W, A_prev) + self.b # (n_out, m)

        # Return activation
        return self.activation.forward(self.Z)

    def backward(self, dA):
        m = self.A_prev.shape[1] # batch size

        if isinstance(self.activation, Softmax):
            dZ = dA # G, see explainer notebook
        else:
            dZ = self.activation.backward(dA)

        # Gradients
        dW = (1.0 / m) * np.dot(dZ, self.A_prev.T) + (self.l2_coeff / m) * self.W # regularization applied only to weights (UDL book)
        db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Gradient for previous layer
        dA_prev = np.dot(self.W.T, dZ)

        return dA_prev, dW, db
