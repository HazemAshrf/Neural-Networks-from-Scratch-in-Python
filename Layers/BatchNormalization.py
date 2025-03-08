import numpy as np
from .Helpers import compute_bn_gradients
from .Base import BaseLayer

class BatchNormalization(BaseLayer):
    def __init__(self, channels, epsilon=1e-10, alpha=0.8):
        super().__init__()
        self.channels = channels
        self.epsilon = epsilon
        self.alpha = alpha
        
        self.trainable = True
        self.testing_phase = False
        self.initialize()
        
        # Initialize running statistics
        self.running_mean = None
        self.running_var = None
        
        # For optimizer
        self.optimizer = None  # Single optimizer property
        self._gradient_weights = None
        self._gradient_bias = None
        
        # Storage for backward pass
        self.input_tensor = None
        self.input_shape = None
        self.batch_mean = None
        self.batch_var = None
        self.normalized = None

    def initialize(self, weights_initializer=None, bias_initializer=None):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def reformat(self, tensor):
        if tensor.ndim == 4:
            self.input_shape = tensor.shape
            B, H, M, N = tensor.shape
            return tensor.transpose(0, 2, 3, 1).reshape(-1, H)
        else:
            B, H, M, N = self.input_shape
            return tensor.reshape(B, M, N, H).transpose(0, 3, 1, 2)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        is_conv = (input_tensor.ndim == 4)
        
        if is_conv:
            input_tensor = self.reformat(input_tensor)
        
        if self.testing_phase:
            if self.running_mean is None:
                self.running_mean = np.mean(input_tensor, axis=0)
                self.running_var = np.var(input_tensor, axis=0)
            normalized = (input_tensor - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        else:
            self.batch_mean = np.mean(input_tensor, axis=0)
            self.batch_var = np.var(input_tensor, axis=0)
            
            if self.running_mean is None:
                self.running_mean = self.batch_mean
                self.running_var = self.batch_var
            else:
                self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.batch_mean
                self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.batch_var
            
            normalized = (input_tensor - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
        
        self.normalized = normalized
        output = self.weights * normalized + self.bias
        
        if is_conv:
            output = self.reformat(output)
        
        return output

    def backward(self, error_tensor):
        is_conv = (error_tensor.ndim == 4)
        if is_conv:
            error_tensor = self.reformat(error_tensor)
            input_tensor = self.reformat(self.input_tensor)
        else:
            input_tensor = self.input_tensor

        # Calculate gradients
        self._gradient_weights = np.sum(error_tensor * self.normalized, axis=0)
        self._gradient_bias = np.sum(error_tensor, axis=0)

        # Store old weights for gradient computation
        old_weights = self.weights.copy()

        # Update parameters if optimizer is set
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self._gradient_bias)

        # Calculate gradient with respect to input using old weights
        gradient_input = compute_bn_gradients(error_tensor, input_tensor, old_weights, 
                                           self.batch_mean, self.batch_var)

        if is_conv:
            gradient_input = self.reformat(gradient_input)

        return gradient_input

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
        self.weights_optimizer = value
        self.bias_optimizer = value