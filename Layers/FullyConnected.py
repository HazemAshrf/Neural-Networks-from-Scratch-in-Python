import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_dim = input_size
        self.output_dim = output_size
        self.weights = np.random.rand(self.input_dim + 1, self.output_dim) # Shape: (input_size + 1, output_size)
        self._optimizer = None
        self.grad_weights = None  # Gradient with respect to weights (input_size + 1, output_size)
        self.input_tensor = None  # Store input for use in backward pass

    def initialize(self, weights_initializer, bias_initializer):
        # Initialize weights (all rows except the last)
        self.weights[:-1, :] = weights_initializer.initialize(
            (self.input_dim, self.output_dim), self.input_dim, self.output_dim
        )
        
        # Initialize bias (last row of weights)
        self.weights[-1, :] = bias_initializer.initialize(
            (1, self.output_dim), self.input_dim, self.output_dim
        ).flatten()

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def forward(self, input_tensor):
        """
        Forward pass:
        - input_tensor shape: (batch_size, input_size)
        - After adding bias term, shape becomes: (batch_size, input_size + 1)
        - Output shape (batch_size, output_size)
        """
        batch_size = input_tensor.shape[0]
        # Add a bias term (column of ones) to input_tensor
        self.input_tensor = np.hstack([input_tensor, np.ones((batch_size, 1))])  # Shape: (batch_size, input_size + 1)
        
        # (batch_size, input_size + 1) @ (input_size + 1, output_size) -> (batch_size, output_size)
        output = self.input_tensor @ self.weights
        return output

    def backward(self, error_tensor):
        """
        Backward pass:
        - error_tensor shape: (batch_size, output_size)
        - grad_weights shape: (input_size + 1, output_size)
        - grad_input shape: (batch_size, input_size)
        """
        # Calculate gradients for weights (including bias term)
        # (input_size + 1, batch_size) @ (batch_size, output_size) -> (input_size + 1, output_size)
        self.grad_weights = self.input_tensor.T @ error_tensor
        
        # Update weights using optimizer if set
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.grad_weights)
        
        # Calculate gradient with respect to input (excluding the bias term)
        # error_tensor @ self.weights[:-1].T gives shape (batch_size, input_size)
        grad_input = error_tensor @ self.weights[:-1].T
        return grad_input

    @property
    def gradient_weights(self):
        # Shape: (input_size + 1, output_size)
        return self.grad_weights