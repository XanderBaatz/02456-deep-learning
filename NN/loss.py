import numpy as np

class Loss:
    def forward(self, A, Y):
        raise NotImplementedError

    def backward(self, A, Y):
        raise NotImplementedError


class CrossEntropyLossOld(Loss):
    def forward(self, A, Y, eps=1e-15):

        # Clip only for numerical stability in forward loss
        A_clipped = np.clip(A, eps, 1.0 - eps)

        self.A = A                  
        self.Y = Y                  
        self.m = Y.shape[1]

        loss = -np.sum(Y * np.log(A_clipped)) / self.m
        return loss

    def backward(self, A, Y):
        # For softmax + CE combination, gradient = A - Y
        # remeber that we divid with the batchsize. 
        return (self.A - self.Y) / self.m


class CrossEntropyLoss:
    def forward(self, A, Y, eps=1e-15):
        A_clipped = np.clip(A, eps, 1 - eps)
        m = Y.shape[1]
        return -np.sum(Y * np.log(A_clipped)) / m 

    def backward(self, A, Y):
        m = Y.shape[1]
        return (A - Y) / m # G, derivative when combined with softmax