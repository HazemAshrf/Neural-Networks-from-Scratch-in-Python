import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
    
    def forward(self, input_tensor):
        """
        Computes the SoftMax probabilities for the input tensor.
        
        Parameters:
        - input_tensor: (batch_size, num_classes) array of logits.

        Returns:
        - softmax_output: (batch_size, num_classes) array of SoftMax probabilities.
        """
        # Shift inputs for numerical stability
        input_tensor = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        
        exp_values = np.exp(input_tensor)
        softmax_output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = softmax_output  # Save output for use in backward pass
        return softmax_output
    
    def backward(self, error_tensor):
        """
        Computes the gradient of the loss with respect to the input using the backward pass.

        Parameters:
        - error_tensor: (batch_size, num_classes) array of gradient from the next layer.
        
        Returns:
        - grad_input: (batch_size, num_classes) array, gradient with respect to the input.
        """
        # Calculate sum across classes for each batch, result is (batch_size, 1)
        weighted_error_sum = np.sum(error_tensor * self.output, axis=1, keepdims=True)
        
        # Compute gradient by the element-wise equation
        grad_input = self.output * (error_tensor - weighted_error_sum)
        
        return grad_input
