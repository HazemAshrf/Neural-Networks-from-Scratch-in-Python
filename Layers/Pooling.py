import numpy as np
from .Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.trainable = False
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.output_shape = None
        self.max_indices = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channels, height, width = input_tensor.shape
        pool_height, pool_width = self.pooling_shape
        stride_y, stride_x = self.stride_shape

        # Calculate output dimensions
        output_height = (height - pool_height) // stride_y + 1
        output_width = (width - pool_width) // stride_x + 1
        self.output_shape = (batch_size, channels, output_height, output_width)

        # Extract pooling regions
        sliding_windows = np.lib.stride_tricks.sliding_window_view(input_tensor, (pool_height, pool_width), axis=(2, 3))  # shape: (batch_size, channels, h, w, pool_height, pool_width)
        strided_windows = sliding_windows[:, :, ::stride_y, ::stride_x, :, :]  # shape: (batch_size, channels, output_height, output_width, pool_height, pool_width)
        pooled_regions = strided_windows.reshape(batch_size, channels, output_height, output_width, pool_height * pool_width)

        # Compute max values and their indices
        self.max_indices = np.argmax(pooled_regions, axis=-1)  # shape: (batch_size, channels, output_height, output_width)
        return np.max(pooled_regions, axis=-1)  # shape: (batch_size, channels, output_height, output_width)

    def backward(self, error_tensor):
        batch_size, channels, height, width = self.input_tensor.shape
        pool_height, pool_width = self.pooling_shape
        stride_y, stride_x = self.stride_shape
        output_height, output_width = self.output_shape[2:]

        # Initialize gradient tensor
        grad_input = np.zeros_like(self.input_tensor, dtype=np.float32)

        # Converts flat indices to (row, col) indices within the pooling window.
        row_indices, col_indices = np.unravel_index(self.max_indices, self.pooling_shape)

        # Adds offsets to the indices to map them from the pooling window coordinates to the input tensor coordinates.
        row_indices = row_indices + np.arange(output_height)[:, None] * stride_y
        col_indices = col_indices + np.arange(output_width)[None, :] * stride_x

        # Broadcast offsets for batch and channel dimensions
        batch_indices = np.arange(batch_size)[:, None, None, None]
        channel_indices = np.arange(channels)[None, :, None, None]

        # Use advanced indexing to distribute error values
        np.add.at(
            grad_input,
            (batch_indices, channel_indices, row_indices, col_indices),
            error_tensor,
        )

        return grad_input


'''
import numpy as np
from .Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        """
        Initialize the pooling layer.
        
        Parameters:
        - stride_shape: Tuple indicating the stride for pooling (stride_y, stride_x).
        - pooling_shape: Tuple indicating the pooling region size (pool_y, pool_x).
        """
        self.trainable = False
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.max_indices = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channels, input_height, input_width = input_tensor.shape
        pool_height, pool_width = self.pooling_shape
        stride_y, stride_x = self.stride_shape

        # Calculate the output dimensions
        output_height = (input_height - pool_height) // stride_y + 1
        output_width = (input_width - pool_width) // stride_x + 1

        # Initialize the output tensor
        output_tensor = np.zeros((batch_size, channels, output_height, output_width))
        self.max_indices = np.zeros_like(input_tensor, dtype=bool)

        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # Define the pooling region
                        start_y = i * stride_y
                        start_x = j * stride_x
                        end_y = start_y + pool_height
                        end_x = start_x + pool_width

                        pooling_region = input_tensor[b, c, start_y:end_y, start_x:end_x]
                        max_value = np.max(pooling_region)
                        output_tensor[b, c, i, j] = max_value

                        # Store the indices of the max value
                        max_idx = np.unravel_index(np.argmax(pooling_region, axis=None), pooling_region.shape)
                        self.max_indices[b, c, start_y + max_idx[0], start_x + max_idx[1]] = True

        return output_tensor

    def backward(self, error_tensor):
        """
        Perform the backward pass for the pooling layer.
    
        Parameters:
        - error_tensor: Gradient of the loss with respect to the output tensor.
    
        Returns:
        - grad_input_tensor: Gradient of the loss with respect to the input tensor.
        """
        batch_size, channels, output_height, output_width = error_tensor.shape
        stride_y, stride_x = self.stride_shape
        pool_height, pool_width = self.pooling_shape
    
        # Initialize the gradient tensor for the input
        grad_input_tensor = np.zeros_like(self.input_tensor)
    
        # Propagate the error tensor
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # Define the pooling region
                        start_y = i * stride_y
                        start_x = j * stride_x
                        end_y = start_y + pool_height
                        end_x = start_x + pool_width
    
                        # Avoid out-of-bounds indexing by ensuring the region fits
                        pooling_region = self.input_tensor[b, c, start_y:end_y, start_x:end_x]
                        max_value = np.max(pooling_region)
                        max_index = np.unravel_index(np.argmax(pooling_region, axis=None), pooling_region.shape)
    
                        # Map the max index back to the original input
                        grad_input_tensor[b, c, start_y + max_index[0], start_x + max_index[1]] += error_tensor[b, c, i, j]
    
        return grad_input_tensor
'''