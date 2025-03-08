import numpy as np
from .Base import BaseLayer
from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self._optimizer = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Fully connected layers for input-hidden and hidden-output
        self.fc_xh = FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_hy = FullyConnected(hidden_size, output_size)

        # Activation functions
        self.tanh = TanH()
        self.sigmoid = Sigmoid()

        # Hidden state initialization
        self.h = None
        self.memorize = False

        # Storage for intermediate values during forward pass
        self.inputs = []
        self.h_states = []
        self.outputs = []

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_xh.initialize(weights_initializer, bias_initializer)
        self.fc_hy.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        batch_size, _ = input_tensor.shape

        if not self.memorize or self.h is None:
            self.h = np.zeros((batch_size, self.hidden_size))

        self.inputs.clear()
        self.h_states.clear()
        self.outputs.clear()

        for t in range(batch_size):
            # Combine input with previous hidden state
            combined_input = np.concatenate((input_tensor[t].reshape(1, -1), self.h[t-1].reshape(1, -1)), axis=1)
            self.inputs.append(combined_input[0])

            # Forward through input-hidden layer
            u = self.fc_xh.forward(combined_input)
            new_h = self.tanh.forward(u)
            self.h[t] = new_h[0]  # Store the hidden state
            self.h_states.append(new_h[0])

            # Forward through hidden-output layer
            o = self.fc_hy.forward(new_h)
            y = self.sigmoid.forward(o)
            self.outputs.append(y[0])

        return np.vstack(self.outputs)

    def backward(self, error_tensor):
        batch_size, _ = error_tensor.shape
        grad_h_next = np.zeros((1, self.hidden_size))
        grad_input = np.zeros((batch_size, self.input_size))

        self.grad_w_xh = np.zeros_like(self.fc_xh.weights)  # Initialize gradients for fc_xh weights
        self.grad_w_hy = np.zeros_like(self.fc_hy.weights)  # Initialize gradients for fc_hy weights

        for t in reversed(range(batch_size)):
            # Backward through sigmoid
            self.sigmoid.activations = self.outputs[t]
            grad_o = self.sigmoid.backward(error_tensor[t].reshape(1, -1))

            # Backward through hidden-output layer
            self.fc_hy.input_tensor = np.hstack([self.h_states[t].reshape(1, -1), np.ones((1, 1))])
            grad_h = self.fc_hy.backward(grad_o) + grad_h_next

            # Backward through tanh
            self.tanh.activations = self.h_states[t]
            grad_u = self.tanh.backward(grad_h)

            # Backward through input-hidden layer
            combined_input = np.hstack([self.inputs[t].reshape(1, -1), np.ones((1, 1))])
            self.fc_xh.input_tensor = combined_input
            grad_combined = self.fc_xh.backward(grad_u)

            # Split gradients
            grad_input[t] = grad_combined[0, :self.input_size]
            grad_h_next = grad_combined[0, self.input_size:].reshape(1, -1)

            self.grad_w_xh += self.fc_xh.gradient_weights
            self.grad_w_hy += self.fc_hy.gradient_weights

        # Apply optimizer updates if available
        if self._optimizer:
            self.fc_xh.weights = self._optimizer.calculate_update(self.fc_xh.weights, self.grad_w_xh)
            self.fc_hy.weights = self._optimizer.calculate_update(self.fc_hy.weights, self.grad_w_hy)

        return grad_input

    @property
    def gradient_weights(self):
        return self.grad_w_xh

    @property
    def weights(self):
        return self.fc_xh.weights

    @weights.setter
    def weights(self, new_weights):
        self.fc_xh.weights = new_weights

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer