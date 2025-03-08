from .Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_shape = None  # To store the original shape of the input tensor for backward pass

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape  # Shape: (batch_size, channels, height, width)
        return input_tensor.reshape(input_tensor.shape[0], -1)  # shape: (batch_size, channels * height * width)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)  # Shape: (batch_size, channels, height, width)
