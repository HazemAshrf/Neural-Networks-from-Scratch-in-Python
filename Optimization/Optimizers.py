import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)

        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum = 0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None  # To store the running velocity

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)  # Initialize velocity to zero

        # Update velocity and weights
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient_tensor

        if self.regularizer:
            regularization_term = self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        else:
            regularization_term = 0

        updated_weights = weight_tensor + self.velocity - regularization_term
        return updated_weights


class Adam(Optimizer):
    def __init__(self, learning_rate, mu = 0.9, rho = 0.999):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu  # Decay rate for the first moment estimate (v)
        self.rho = rho  # Decay rate for the second moment estimate (r)
        self.epsilon = 1e-8  # To prevent division by zero
        self.v = None  # First moment vector
        self.r = None  # Second moment vector
        self.t = 0  # Time step (iteration count)

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)  # Initialize first moment vector to zero
        if self.r is None:
            self.r = np.zeros_like(weight_tensor)  # Initialize second moment vector to zero

        self.t += 1  # Increment time step

        # Update biased first and second moment estimates
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * (gradient_tensor ** 2)

        # Bias correction
        v_hat = self.v / (1 - self.mu ** self.t)
        r_hat = self.r / (1 - self.rho ** self.t)

        if self.regularizer:
            regularization_term = self.regularizer.calculate_gradient(weight_tensor)
        else:
            regularization_term = 0

        # Compute updated weights
        updated_weights = weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.epsilon) - self.learning_rate * regularization_term
        return updated_weights