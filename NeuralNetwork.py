import copy
import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []              # List to store loss values after each training iteration
        self.layers = []
        self.data_layer = None      # Layer providing input data and labels
        self.loss_layer = None

    @property
    def phase(self):
        return all(layer.testing_phase for layer in self.layers)

    @phase.setter
    def phase(self, phase):
        for layer in self.layers:
            layer.testing_phase = phase
    
    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.current_label_tensor = label_tensor  # Store label for use in backward pass

        reg_loss = 0
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            if layer.trainable and layer.optimizer.regularizer:
                reg_loss += layer.optimizer.regularizer.norm(layer.weights)

        data_loss = self.loss_layer.forward(input_tensor, self.current_label_tensor)
        total_loss = data_loss + reg_loss

        return total_loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.current_label_tensor)
        
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
    
    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = False
        for _ in range(iterations):
            prediction = self.forward()
            
            self.loss.append(prediction)
            
            self.backward()

    def test(self, input_tensor):
        self.phase = True
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        return input_tensor

    @staticmethod
    def save(filename, net):
        """Save the neural network to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(net, f)

    @staticmethod
    def load(filename, data_layer):
        """Load the neural network from a file."""
        with open(filename, 'rb') as f:
            net = pickle.load(f)
        net.data_layer = data_layer  # Reassign the data layer after loading
        return net

    def __getstate__(self):
        state = self.__dict__.copy()
        state['data_layer'] = None  # Exclude the data layer from being pickled
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.data_layer = None  # Reinitialize the data layer to None