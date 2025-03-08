# Neural Networks from Scratch in Python

## Overview
This project implements neural networks from scratch using Python, without relying on deep learning frameworks like TensorFlow or PyTorch. It includes fundamental components such as fully connected layers, convolutional layers, LSTMs, RNNs, optimizers, loss functions, and batch normalization. The implementation is structured in a modular way to facilitate easy experimentation and learning.

## Features
- Custom-built neural network framework
- Implementations of various layers: Fully Connected, Convolutional, Pooling, LSTM, RNN, Dropout, Batch Normalization, etc.
- Training script for the LeNet architecture on the MNIST dataset
- Custom initializers, activation functions, and optimization methods
- No external deep learning libraries required

## File Structure
```
NeuralNetworksFromScratch/
├── NeuralNetwork.py              # Core implementation of neural networks
├── NeuralNetworkTests.py         # Unit tests for the implemented network
├── TrainLeNet.py                 # Training script for LeNet on MNIST
├── Data/                         # MNIST dataset (compressed binary format)
├── Layers/                       # Layer implementations
│   ├── Base.py                   # Base class for all layers
│   ├── BatchNormalization.py      # Batch normalization layer
│   ├── Conv.py                    # Convolutional layer
│   ├── Dropout.py                 # Dropout regularization layer
│   ├── Flatten.py                 # Flatten layer
│   ├── FullyConnected.py          # Dense layer
│   ├── LSTM.py                    # Long Short-Term Memory layer
│   ├── Pooling.py                 # Pooling layer (Max & Average)
│   ├── RNN.py                     # Recurrent Neural Network layer
│   ├── ReLU.py                    # ReLU activation function
│   ├── Sigmoid.py                 # Sigmoid activation function
│   ├── SoftMax.py                 # Softmax activation function
│   ├── TanH.py                    # TanH activation function
│   ├── Initializers.py            # Weight initialization strategies
│   ├── Helpers.py                 # Utility functions
├── Models/                        # Predefined neural network architectures
│   ├── LeNet.py                   # LeNet model implementation
├── Optimization/                   # Optimization-related components
│   ├── Constraints.py              # Regularization constraints
│   ├── Loss.py                     # Loss functions (Cross-Entropy, MSE, etc.)
│   ├── Optimizers.py               # Optimization algorithms (SGD, Adam, etc.)
├── trained/                        # Stored trained models or checkpoints
│   ├── LeNet                       # Pre-trained LeNet model
```

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/NeuralNetworksFromScratch.git
   cd NeuralNetworksFromScratch
   ```
2. Install dependencies (only NumPy and Matplotlib are required):
   ```bash
   pip install numpy matplotlib
   ```

## Usage
To train the LeNet model on the MNIST dataset, run:
```bash
python TrainLeNet.py
```

## License
This project is licensed under the MIT License.

