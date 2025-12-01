import numpy as np

class Optimizer:
    def update(self, layer, dW, db):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.lr = float(learning_rate)

    def update(self, layer, dW, db):
        layer.W -= self.lr * dW
        layer.b -= self.lr * db


#The adan optimizer is highly inspired by https://medium.com/the-ml-practitioner/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t = 0
        self.m = {}
        self.v = {}

    def update(self, layer, dW, db):
        # initialize per-layer moments
        if layer not in self.m:
            self.m[layer] = {'dW': np.zeros_like(dW), 'db': np.zeros_like(db)}
            self.v[layer] = {'dW': np.zeros_like(dW), 'db': np.zeros_like(db)}

        self.t += 1
        self.m[layer]['dW'] = self.beta1 * self.m[layer]['dW'] + (1.0 - self.beta1) * dW
        self.m[layer]['db'] = self.beta1 * self.m[layer]['db'] + (1.0 - self.beta1) * db

        self.v[layer]['dW'] = self.beta2 * self.v[layer]['dW'] + (1.0 - self.beta2) * (dW**2)
        self.v[layer]['db'] = self.beta2 * self.v[layer]['db'] + (1.0 - self.beta2) * (db**2)

        m_hat_W = self.m[layer]['dW'] / (1.0 - self.beta1**self.t)
        m_hat_b = self.m[layer]['db'] / (1.0 - self.beta1**self.t)
        v_hat_W = self.v[layer]['dW'] / (1.0 - self.beta2**self.t)
        v_hat_b = self.v[layer]['db'] / (1.0 - self.beta2**self.t)

        layer.W -= self.lr * (m_hat_W / (np.sqrt(v_hat_W) + self.eps))
        layer.b -= self.lr * (m_hat_b / (np.sqrt(v_hat_b) + self.eps))
