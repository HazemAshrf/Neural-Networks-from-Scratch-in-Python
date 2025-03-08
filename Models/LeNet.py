import numpy as np
from Layers.Conv import Conv
from Layers.FullyConnected import FullyConnected
from Layers.SoftMax import SoftMax
from Layers.ReLU import ReLU
from Layers.Pooling import Pooling
from Layers.Flatten import Flatten
from Layers.Initializers import He
from NeuralNetwork import NeuralNetwork
from Optimization.Optimizers import Adam
from Optimization.Loss import CrossEntropyLoss
from Optimization.Constraints import L2_Regularizer

def build():
    # Create optimizer and regularizer
    optimizer = Adam(learning_rate=5e-4)
    regularizer = L2_Regularizer(alpha=4e-4)
    optimizer.add_regularizer(regularizer)
    Initializer = He()

    # Initialize the network
    net = NeuralNetwork(optimizer, weights_initializer=Initializer, bias_initializer=Initializer)
    # Add layers
    net.append_layer(Conv(stride_shape=(1, 1), convolution_shape=[1, 5, 5], num_kernels=6))  # Conv1
    net.append_layer(ReLU())
    net.append_layer(Pooling(pooling_shape=(2, 2), stride_shape=(2, 2)))  # Pool1

    net.append_layer(Conv(stride_shape=(1, 1), convolution_shape=[6, 5, 5], num_kernels=16))  # Conv2
    net.append_layer(ReLU())
    net.append_layer(Pooling(pooling_shape=(2, 2), stride_shape=(2, 2)))  # Pool2

    net.append_layer(Flatten())  # Flatten
    net.append_layer(FullyConnected(16 * 7 * 7, 120))  # FC1
    net.append_layer(ReLU())

    net.append_layer(FullyConnected(120, 84))  # FC2
    net.append_layer(ReLU())

    net.append_layer(FullyConnected(84, 10))  # FC3
    net.append_layer(SoftMax())
    net.loss_layer = CrossEntropyLoss()

    return net
