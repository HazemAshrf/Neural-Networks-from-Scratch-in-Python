import numpy as np
from .Base import BaseLayer
from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid

class LSTM(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self._optimizer = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize states
        self.h = None  # hidden state
        self.c = None  # cell state
        self.memorize = False
        
        # Layers
        self.fc_gates = FullyConnected(input_size + hidden_size, 4 * hidden_size)
        self.fc_hy = FullyConnected(hidden_size, output_size)
        
        # Activation functions
        self.tanh = TanH()
        self.sigmoid = Sigmoid()
        
        # Storage for backward pass
        self.inputs = []
        self.h_states = []
        self.c_states = []
        self.gates = []
        self.outputs = []

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_gates.initialize(weights_initializer, bias_initializer)
        self.fc_hy.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        batch_size, _ = input_tensor.shape
        
        if not self.memorize or self.h is None:
            self.h = np.zeros((batch_size, self.hidden_size))
            self.c = np.zeros((batch_size, self.hidden_size))
        
        self.inputs.clear()
        self.h_states.clear()
        self.c_states.clear()
        self.gates.clear()
        self.outputs.clear()
        
        for t in range(batch_size):
            # Combine input with previous hidden state
            combined_input = np.concatenate((input_tensor[t].reshape(1, -1), self.h[t-1].reshape(1, -1)), axis=1)
            self.inputs.append(combined_input[0])
            
            # Forward through gates layer
            gates = self.fc_gates.forward(combined_input)
            f, i, g, o = np.split(gates, 4, axis=1)
            
            # Apply activations
            f = self.sigmoid.forward(f)
            i = self.sigmoid.forward(i)
            g = self.tanh.forward(g)
            o = self.sigmoid.forward(o)
            
            # Update cell state
            c_next = f * self.c[t-1] + i * g
            self.c[t] = c_next
            
            # Update hidden state
            h_next = o * self.tanh.forward(c_next)
            self.h[t] = h_next
            
            # Store states
            self.h_states.append(h_next[0])
            self.c_states.append(c_next[0])
            self.gates.append((f[0], i[0], g[0], o[0]))
            
            # Forward through output layer
            y = self.fc_hy.forward(h_next)
            self.outputs.append(y[0])
        
        return np.vstack(self.outputs)

    def backward(self, error_tensor):
        batch_size, _ = error_tensor.shape
        grad_h_next = np.zeros((1, self.hidden_size))
        grad_c_next = np.zeros((1, self.hidden_size))
        grad_input = np.zeros((batch_size, self.input_size))
        
        self.grad_w_gates = np.zeros_like(self.fc_gates.weights)
        self.grad_w_hy = np.zeros_like(self.fc_hy.weights)
        
        for t in reversed(range(batch_size)):
            # Get stored values
            f, i, g, o = self.gates[t]
            h_prev = self.h_states[t-1] if t > 0 else np.zeros_like(self.h_states[0])
            c_prev = self.c_states[t-1] if t > 0 else np.zeros_like(self.c_states[0])
            
            # Backward through output layer
            self.fc_hy.input_tensor = np.hstack([self.h_states[t].reshape(1, -1), np.ones((1, 1))])
            grad_h = self.fc_hy.backward(error_tensor[t].reshape(1, -1)) + grad_h_next
            
            # Accumulate gradients for output layer
            self.grad_w_hy += self.fc_hy.gradient_weights
            
            # Backward through hidden state
            c_tanh = np.tanh(self.c_states[t])
            grad_o = grad_h * c_tanh
            grad_c = grad_h * o * (1 - c_tanh**2) + grad_c_next
            
            # Backward through cell state
            grad_f = grad_c * c_prev
            grad_i = grad_c * g
            grad_g = grad_c * i
            grad_c_prev = grad_c * f
            
            # Combine gate gradients
            grad_gates = np.concatenate([
                grad_f * f * (1 - f),
                grad_i * i * (1 - i),
                grad_g * (1 - g**2),
                grad_o * o * (1 - o)
            ], axis=1)
            
            # Backward through gates layer
            combined_input = np.hstack([self.inputs[t].reshape(1, -1), np.ones((1, 1))])
            self.fc_gates.input_tensor = combined_input
            grad_combined = self.fc_gates.backward(grad_gates)
            
            # Accumulate gradients for gates layer
            self.grad_w_gates += self.fc_gates.gradient_weights
            
            # Split gradients
            grad_input[t] = grad_combined[0, :self.input_size]
            grad_h_next = grad_combined[0, self.input_size:].reshape(1, -1)
            grad_c_next = grad_c_prev
        
        # Apply optimizer updates if available
        if self._optimizer:
            self.fc_gates.weights = self._optimizer.calculate_update(self.fc_gates.weights, self.grad_w_gates)
            self.fc_hy.weights = self._optimizer.calculate_update(self.fc_hy.weights, self.grad_w_hy)
        
        return grad_input

    @property
    def gradient_weights(self):
        return self.grad_w_gates

    @property
    def weights(self):
        return self.fc_gates.weights

    @weights.setter
    def weights(self, new_weights):
        self.fc_gates.weights = new_weights

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        
    def calculate_regularization_loss(self):
        reg_loss = 0
        if self._optimizer and hasattr(self._optimizer, 'regularizer'):
            reg_loss += self._optimizer.regularizer.norm(self.fc_gates.weights)
            reg_loss += self._optimizer.regularizer.norm(self.fc_hy.weights)
        return reg_loss