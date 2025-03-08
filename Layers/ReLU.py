import numpy as np
from .Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_tensor = None

    def forward(self, input_tensor):

        self.input_tensor = input_tensor  # Store input for use in backward pass
        return np.maximum(0, input_tensor) # Shape (batch_size, input_size)

    def backward(self, error_tensor):
        relu_gradient = (self.input_tensor > 0).astype(float)
        return error_tensor * relu_gradient # Shape (batch_size, input_size)