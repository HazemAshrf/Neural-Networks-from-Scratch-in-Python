import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        """Calculate the gradient of the L2 regularization."""
        return self.alpha * weights

    def norm(self, weights):
        """Calculate the L2 norm regularization loss."""
        return self.alpha * np.sum(weights ** 2)


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        """Calculate the gradient of the L1 regularization."""
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        """Calculate the L1 norm regularization loss."""
        return self.alpha * np.sum(np.abs(weights))