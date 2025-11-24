from NN.activations import *
from NN.initializer import *


class Layer:
    def forward(self, A_prev): # A^[l-1]
        raise NotImplementedError

    def backward(self, dA, learning_rate): # dA^[l]
        raise NotImplementedError

class DenseLayer(Layer):
    def __init__(
        self,
        n_in:int,
        n_out:int,
        activation:Activation,
        initializer:Initializer=None,
        l2_coeff:float=0.0
    ):
        # Initialize weights and bias
        self.initializer = initializer
        self.W = self.initializer.init_weights(n_in, n_out) # W^[l]
        self.b = self.initializer.init_bias(n_out)          # b^[l]
        self.activation = activation
        self.l2_coeff = l2_coeff # lambda

        # Caching during forward/backward
        self.Z = None
        self.A_prev = None
    
    def __repr__(self):
        act_name = self.activation.__class__.__name__
        return (
            f"DenseLayer({self.W.shape[1]} -> {self.W.shape[0]}, Activation={act_name})\n"
            f"Weights shape: {self.W.shape}, min: {self.W.min():.4f}, max: {self.W.max():.4f}, mean: {self.W.mean():.4f}\n"
            f"Biases shape: {self.b.shape}, min: {self.b.min():.4f}, max: {self.b.max():.4f}, mean: {self.b.mean():.4f}"
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
            dZ = dA
        else:
            dZ = self.activation.backward(dA)

        # Gradients
        dW = (1/m) * np.dot(dZ, self.A_prev.T) + (self.l2_coeff/m) * self.W # regularization applied only to weights
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Gradient for previous layer
        dA_prev = np.dot(self.W.T, dZ)
        
        return dA_prev, dW, db