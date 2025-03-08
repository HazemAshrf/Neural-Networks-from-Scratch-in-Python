import numpy as np
from .Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.activations = None

    def forward(self, input_tensor):
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        return error_tensor * (1 - self.activations ** 2)
