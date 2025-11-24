from NN.denseLayer import *

class Optimizer:
    def update(self, layer, dW, db):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def update(self, layer, dW, db):
        layer.W -= self.lr * dW
        layer.b -= self.lr * db

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}  # momentum
        self.v = {}  # RMSprop
    
    def update(self, layer, dW, db):
        if layer not in self.m:
            # Initialize moments
            self.m[layer] = {'dW': np.zeros_like(dW), 'db': np.zeros_like(db)}
            self.v[layer] = {'dW': np.zeros_like(dW), 'db': np.zeros_like(db)}
        
        self.t += 1
        # Momentum
        self.m[layer]['dW'] = self.beta1 * self.m[layer]['dW'] + (1 - self.beta1) * dW
        self.m[layer]['db'] = self.beta1 * self.m[layer]['db'] + (1 - self.beta1) * db
        # RMS
        self.v[layer]['dW'] = self.beta2 * self.v[layer]['dW'] + (1 - self.beta2) * (dW**2)
        self.v[layer]['db'] = self.beta2 * self.v[layer]['db'] + (1 - self.beta2) * (db**2)
        # Bias-corrected
        m_hat_W = self.m[layer]['dW'] / (1 - self.beta1**self.t)
        m_hat_b = self.m[layer]['db'] / (1 - self.beta1**self.t)
        v_hat_W = self.v[layer]['dW'] / (1 - self.beta2**self.t)
        v_hat_b = self.v[layer]['db'] / (1 - self.beta2**self.t)
        # Update
        layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.eps)
        layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)