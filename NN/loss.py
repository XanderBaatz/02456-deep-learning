import numpy as np

class Loss:
    def forward(self, A, Y):
        raise NotImplementedError
    
    def backward(self, A, Y):
        raise NotImplementedError

class CrossEntropyLoss(Loss):
    def forward(self, A, Y):
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(A + 1e-8)) / m
        return loss
    
    def backward(self, A, Y):
        return A - Y  # derivative when combined with softmax

class MSELoss(Loss):
    def forward(self, A, Y):
        m = Y.shape[1]
        return np.sum((A - Y)**2) / (2 * m)
    
    def backward(self, A, Y):
        m = Y.shape[1]
        return (A - Y) / m