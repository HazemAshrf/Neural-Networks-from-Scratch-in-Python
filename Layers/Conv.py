import numpy as np
from scipy.signal import correlate , convolve
from .Base import BaseLayer
import copy

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        """        
        Args:
            stride_shape (int or tuple): The stride in each spatial dimension. Can be a single value (e.g., 1) 
                                         or a tuple (e.g., (1, 1) or (2, 2)).
            convolution_shape (list): Shape of the filter, [c, m] for 1D or [c, m, n] for 2D convolution.
                                      `c` is the number of input channels, `m` and `n` are spatial dimensions.
            num_kernels (int): The number of kernels (filters) used in this convolution layer.
        """
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
                
        # Initialize weights and biases randomly
        self.weights = np.random.rand(num_kernels, *convolution_shape)  # Shape: [num_kernels, c, m, n] or [num_kernels, c, m]
        self.bias = np.random.rand(num_kernels)  # Shape: [num_kernels]

        self._optimizer = None
        self._bias_optimizer = None
        
        self._gradient_weights = None
        self._gradient_bias = None

        self.input_tensor = None  # Store input for use in backward pass
        
        self.stride_y, self.stride_x = stride_shape if isinstance(stride_shape, tuple) else (stride_shape, stride_shape)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        batch_size, channels, *input_dims = input_tensor.shape

        # Calculate output shape based on input size, kernel size, and stride
        if len(input_dims) == 1:  # 1D Convolution
            pad = (self.convolution_shape[1] - 1) // 2
            #output_length = (input_dims[0] + 2*pad - self.convolution_shape[1]) // self.stride_y + 1
            #output_tensor = np.zeros((batch_size, self.num_kernels, output_length))

            input_padded = np.pad(input_tensor, ((0, 0), (0, 0), (pad, pad)), mode='constant')
            input_patches = np.lib.stride_tricks.sliding_window_view(input_padded, self.convolution_shape[1], axis=2)  # Shape: (batch_size, channels, output_length, kernel_size)
            input_patches = np.transpose(input_patches, (0, 1, 3, 2))  # Shape: (batch_size, channels, kernel_size, output_length)

            # Rearrange the weights
            rearranged_weights = np.transpose(self.weights, (1, 0, 2))  # Shape: (channels, num_kernels, kernel_size)

            # Correlation via matrix multiplication
            output_tensor = np.matmul(rearranged_weights, input_patches)  # Shape: (batch_size, channels, num_kernels, output_length)
            output_tensor = np.transpose(output_tensor, (0, 2, 1, 3))        # Shape: (batch_size, num_kernels, channels, output_length)
            output_tensor = np.sum(output_tensor, axis=2)                    # Shape: (batch_size, num_kernels, output_length)

            output_tensor = output_tensor[:, :, ::self.stride_y] + self.bias.reshape(1, -1, 1)  # make the bias have a shape of (1, num_kernels, 1) so it broadcasts correctly across the batch and length dimensions

        elif len(input_dims) == 2:  # 2D Convolution
            pad_y = (self.convolution_shape[1] - 1) // 2
            pad_x = (self.convolution_shape[2] - 1) // 2
            #output_height = int((input_dims[0] + 2*pad_y - self.convolution_shape[1]) // self.stride_y + 1)
            #output_width = int((input_dims[1] + 2*pad_x - self.convolution_shape[2]) // self.stride_x + 1)
            #output_tensor = np.zeros((batch_size, self.num_kernels, output_height, output_width))

            # Padding dimensions
            pad_width = ((0, 0),  # No padding for the batch dimension
                        (0, 0),  # No padding for the channel dimension
                        (pad_y, pad_y if self.convolution_shape[1] % 2 != 0 else pad_y+1),  # Pad height with half the kernel height row on top and bottom
                        (pad_x, pad_x if self.convolution_shape[2] % 2 != 0 else pad_x+1))  # Pad width with half the kernel width column on left and right


            input_padded = np.pad(input_tensor, pad_width, mode='constant')
            input_patches = np.lib.stride_tricks.sliding_window_view(input_padded, (self.convolution_shape[1], self.convolution_shape[2]), axis=(2, 3))  # Shape: (batch_size, channels, output_height, output_width, kernel_size_y, kernel_size_x)
            input_patches = input_patches.reshape(batch_size, channels, input_dims[0]*input_dims[1], self.convolution_shape[1]*self.convolution_shape[2])  # Shape: (batch_size, channels, output_height*output_width, kernel_size_y*kernel_size_x) 
            input_patches = np.transpose(input_patches, (0, 1, 3, 2))  # Shape: (batch_size, channels, kernel_size_y*kernel_size_x, output_height*output_width)

            # Rearrange the weights
            rearranged_weights = np.transpose(self.weights, (1, 0, 2, 3))  # Shape: (channels, num_kernels, kernel_size_y, kernel_size_x)
            weights_reshaped = rearranged_weights.reshape(channels, self.num_kernels, self.convolution_shape[1]*self.convolution_shape[2])    # Shape: (channels, num_kernels, kernel_size_y*kernel_size_x)

            # Correlation via matrix multiplication
            output_tensor = np.matmul(weights_reshaped, input_patches)  # Shape: (batch_size, channels, num_kernels, output_height*output_width)
            output_tensor = np.transpose(output_tensor, (0, 2, 1, 3))        # Shape: (batch_size, num_kernels, channels, output_height*output_width)
            output_tensor = np.sum(output_tensor, axis=2)                    # Shape: (batch_size, num_kernels, output_height*output_width)
            output_tensor = output_tensor.reshape(batch_size, self.num_kernels, input_dims[0], input_dims[1])  # Shape: (batch_size, num_kernels, output_height*output_width)

            output_tensor = output_tensor[:, :, ::self.stride_y, ::self.stride_x] + self.bias.reshape(1, -1, 1, 1)  # make the bias have a shape of (1, num_kernels, 1, 1) so it broadcasts correctly across the batch and length dimensions

        return output_tensor
    
    def backward(self, error_tensor):
        batch_size, channels, *input_dims = self.input_tensor.shape

        if len(input_dims) == 1:  # 1D Convolution
            pad_l = self.convolution_shape[1] // 2

            padded_input_tensor = np.pad(self.input_tensor, pad_width=((0, 0), (0, 0), (pad_l, pad_l)), mode='constant', constant_values=0)

            # Upsampling the error tensor to match the input spatial dimensions
            upsample_length = (error_tensor.shape[2] - 1) * self.stride_y + 1
            upsampled_error_tensor = np.zeros((batch_size, self.num_kernels, upsample_length))
            upsampled_error_tensor[:, :, ::self.stride_y] = error_tensor  # Shape: (batch_size, num_kernels, input_length)

            input_patches = np.lib.stride_tricks.sliding_window_view(padded_input_tensor, upsample_length, axis=2)  # Shape: (batch_size, channels, kernel_size, input_length)
            input_patches = input_patches.reshape(batch_size, channels*self.convolution_shape[1], input_dims[0])    # Shape: (batch_size, channels*kernel_size, input_length)
            input_patches = input_patches.transpose(0, 2, 1)  # Shape: (batch_size, input_length, channels*kernel_size)

            gradient_weights = np.matmul(upsampled_error_tensor, input_patches)  # Shape: (batch_size, num_kernels, channels*kernel_size)
            # Sum gradients over the batch dimension
            gradient_weights = gradient_weights.sum(axis=0)  # Shape: (num_kernels, channels*kernel_size)
            # Reshape the gradients to match the weights shape
            gradient_weights = gradient_weights.reshape(self.num_kernels, channels, self.convolution_shape[1])  # Shape: (num_kernels, channels, kernel_size)

            gradient_bias = np.sum(error_tensor, axis=(0, 2))  # Shape: (num_kernels,). Sum across the batch dimension (axis 0) and the spatial dimension (axis 2).

            padded_error_tensor = np.pad(upsampled_error_tensor, pad_width=((0, 0), (0, 0), (pad_l, pad_l)), mode='constant', constant_values=0)
            error_patches = np.lib.stride_tricks.sliding_window_view(padded_error_tensor, self.convolution_shape[1], axis=2)  # Shape: (batch_size, num_kernels, input_length, kernel_size)
            error_patches = np.transpose(error_patches, (0, 1, 3, 2))  # Shape: (batch_size, num_kernels, kernel_size, input_length)

            # Flip the weights 180 degrees.
            flipped_weights = np.flip(self.weights, axis=2)  # Shape: (num_kernels, channels, kernel_size)

            gradient_input = np.matmul(flipped_weights, error_patches)  # Shape: (batch_size, num_kernels, channels, input_length)
            gradient_input = np.transpose(gradient_input, (0, 2, 1, 3))  # Shape: (batch_size, channels, num_kernels, input_length)
            gradient_input = np.sum(gradient_input, axis=2)  # Shape: (batch_size, channels, input_length)

        elif len(input_dims) == 2:  # 2D Convolution
            pad_y = self.convolution_shape[1] // 2 if self.convolution_shape[1] % 2 != 0 else (self.convolution_shape[1] // 2) - 1
            pad_x = self.convolution_shape[2] // 2 if self.convolution_shape[2] % 2 != 0 else (self.convolution_shape[2] // 2) - 1
            # Padding dimensions
            pad_width = ((0, 0),  # No padding for the batch dimension
                        (0, 0),  # No padding for the channel dimension
                        (pad_y, pad_y if self.convolution_shape[1] % 2 != 0 else pad_y+1),  # Pad height with half the kernel height row on top and bottom
                        (pad_x, pad_x if self.convolution_shape[2] % 2 != 0 else pad_x+1))  # Pad width with half the kernel width column on left and right

            # Apply padding
            padded_input_tensor = np.pad(self.input_tensor, pad_width=pad_width, mode='constant', constant_values=0)

            # Upsampling the error tensor to match the input spatial dimensions
            upsample_height = self.input_tensor.shape[2]
            upsample_width = self.input_tensor.shape[3]
            upsampled_error_tensor = np.zeros((batch_size, self.num_kernels, upsample_height, upsample_width))
            upsampled_error_tensor[:, :, ::self.stride_y, ::self.stride_x] = error_tensor  # Shape: (batch_size, num_kernels, input_height, input_width)

            input_patches = np.lib.stride_tricks.sliding_window_view(padded_input_tensor, (upsample_height, upsample_width), axis=(2, 3))  # Shape: (batch_size, channels, kernel_size_y, kernel_size_x, input_height, input_width)
            input_patches = input_patches.reshape(batch_size, channels, self.convolution_shape[1]*self.convolution_shape[2], upsample_height*upsample_width)  # Shape: (batch_size, channels, kernel_size_y*kernel_size_x, input_height*input_width)
            input_patches = input_patches.reshape(batch_size, channels*self.convolution_shape[1]*self.convolution_shape[2], upsample_height*upsample_width)    # Shape: (batch_size, channels*kernel_size_y*kernel_size_x, input_height*input_width)
            input_patches = input_patches.transpose(0, 2, 1)  # Shape: (batch_size, input_height*input_width, channels*kernel_size_y*kernel_size_x)

            upsampled_error_tensor_flattened = upsampled_error_tensor.reshape(batch_size, self.num_kernels, upsample_height*upsample_width)  # Shape: (batch_size, num_kernels, input_height*input_width)

            gradient_weights = np.matmul(upsampled_error_tensor_flattened, input_patches)  # Shape: (batch_size, num_kernels, channels*kernel_size_y*kernel_size_x)
            # Sum gradients over the batch dimension
            gradient_weights = gradient_weights.sum(axis=0)  # Shape: (num_kernels, channels*kernel_size_y*kernel_size_x)
            # Reshape the gradients to match the weights shape
            gradient_weights = gradient_weights.reshape(self.num_kernels, channels, self.convolution_shape[1], self.convolution_shape[2])  # Shape: (num_kernels, channels, kernel_size_y, kernel_size_x)
            
            gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))  # Shape: (num_kernels,). Sum across the batch dimension (axis 0) and the spatial dimensions (axis 2, axis 3).

            padded_error_tensor = np.pad(upsampled_error_tensor, pad_width=pad_width, mode='constant', constant_values=0)
            error_patches = np.lib.stride_tricks.sliding_window_view(padded_error_tensor, (self.convolution_shape[1], self.convolution_shape[2]), axis=(2, 3))  # Shape: (batch_size, num_kernels, input_height, input_width, kernel_size_y, kernel_size_x)
            error_patches = error_patches.reshape(batch_size, self.num_kernels, self.input_tensor.shape[2]*self.input_tensor.shape[3], self.convolution_shape[1]* self.convolution_shape[2])  # Shape: (batch_size, num_kernels, input_height*input_width, kernel_size_y*kernel_size_x)
            error_patches = np.transpose(error_patches, (0, 1, 3, 2))  # Shape: (batch_size, num_kernels, kernel_size_y*kernel_size_x, input_height*input_width)

            # Flip the weights 180 degrees.
            flipped_weights = np.flip(self.weights, axis=(2,3))  # Shape: (num_kernels, channels, kernel_size_y, kernel_size_x)
            flipped_weights = flipped_weights.reshape(self.num_kernels, channels, self.convolution_shape[1]*self.convolution_shape[2])  # Shape: (num_kernels, channels, kernel_size_y*kernel_size_x)

            gradient_input = np.matmul(flipped_weights, error_patches)  # Shape: (batch_size, num_kernels, channels, input_height*input_width)
            gradient_input = np.transpose(gradient_input, (0, 2, 1, 3))  # Shape: (batch_size, channels, num_kernels, input_height*input_width)
            gradient_input = np.sum(gradient_input, axis=2)  # Shape: (batch_size, channels, input_height*input_width)
            gradient_input = gradient_input.reshape(batch_size, channels, self.input_tensor.shape[2], self.input_tensor.shape[3])  # Shape: (batch_size, channels, input_height, input_width)

        # Update gradients
        self._gradient_weights = gradient_weights
        self._gradient_bias = gradient_bias

        # Update weights using optimizer if set
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer is not None:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return gradient_input


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._bias_optimizer = copy.deepcopy(optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.convolution_shape[0] * np.prod(self.convolution_shape[1:])
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)





'''
import numpy as np
from scipy.signal import correlate , convolve
from .Base import BaseLayer
import copy

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        """        
        Args:
            stride_shape (int or tuple): The stride in each spatial dimension. Can be a single value (e.g., 1) 
                                         or a tuple (e.g., (1, 1) or (2, 2)).
            convolution_shape (list): Shape of the filter, [c, m] for 1D or [c, m, n] for 2D convolution.
                                      `c` is the number of input channels, `m` and `n` are spatial dimensions.
            num_kernels (int): The number of kernels (filters) used in this convolution layer.
        """
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
                
        # Initialize weights and biases randomly
        self.weights = np.random.rand(num_kernels, *convolution_shape)  # Shape: [num_kernels, c, m, n] or [num_kernels, c, m]
        self.bias = np.random.rand(num_kernels)  # Shape: [num_kernels]

        self._optimizer = None
        self._bias_optimizer = None
        
        self._gradient_weights = None
        self._gradient_bias = None

        self.input_tensor = None  # Store input for use in backward pass
        
        self.stride_y, self.stride_x = stride_shape if isinstance(stride_shape, tuple) else (stride_shape[0], stride_shape[0])

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        batch_size, channels, *input_dims = input_tensor.shape

        # Calculate output shape based on input size, kernel size, and stride
        if len(input_dims) == 1:  # 1D Convolution
            pad = (self.convolution_shape[1] - 1) // 2
            output_length = (input_dims[0] + 2*pad - self.convolution_shape[1]) // self.stride_y + 1
            output_tensor = np.zeros((batch_size, self.num_kernels, output_length))
            
            for b in range(batch_size):
                for k in range(self.num_kernels):
                    conv_result = 0
                    for c in range(channels):
                        # Perform 1D convolution (correlation)
                        conv_result += correlate(input_tensor[b, c, :], self.weights[k, c, :], mode='same')
                    # Apply stride (subsampling)
                    output_tensor[b, k, :] = conv_result[::self.stride_y] + + self.bias[k]

        elif len(input_dims) == 2:  # 2D Convolution
            pad_y = (self.convolution_shape[1] - 1) / 2
            pad_x = (self.convolution_shape[2] - 1) / 2
            output_height = int((input_dims[0] + 2*pad_y - self.convolution_shape[1]) // self.stride_y + 1)
            output_width = int((input_dims[1] + 2*pad_x - self.convolution_shape[2]) // self.stride_x + 1)
            output_tensor = np.zeros((batch_size, self.num_kernels, output_height, output_width))

            for b in range(batch_size):
                for k in range(self.num_kernels):
                    conv_result = 0
                    for c in range(channels):
                        # Perform 2D convolution (correlation)
                        conv_result += correlate(input_tensor[b, c, :, :], self.weights[k, c, :, :], mode='same')
                    # Apply stride (subsampling)
                    output_tensor[b, k, :, :] += conv_result[::self.stride_y, ::self.stride_x] + + self.bias[k]
        return output_tensor
    
    def backward(self, error_tensor):
        gradient_input = np.zeros_like(self.input_tensor)
        gradient_weights = np.zeros_like(self.weights)
        gradient_bias = np.zeros_like(self.bias)

        batch_size, channels, *input_dims = self.input_tensor.shape

        if len(input_dims) == 1:  # 1D Convolution
            pad_l = self.convolution_shape[1] // 2
            # Padding dimensions
            pad_width = ((0, 0),  # No padding for the batch dimension
                        (0, 0),  # No padding for the channel dimension
                        (pad_l, pad_l))  # Pad length with half the kernel length

            # Apply padding
            padded_input_tensor = np.pad(self.input_tensor, pad_width=pad_width, mode='constant', constant_values=0)

            # Upsampling the error tensor to match the input spatial dimensions
            upsample_length = (error_tensor.shape[2] - 1) * self.stride_y + 1
            upsampled_error_tensor = np.zeros((batch_size, self.num_kernels, upsample_length))
            upsampled_error_tensor[:, :, ::self.stride_y] = error_tensor

            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for c in range(channels):
                        gradient_weights[k, c, :] += correlate(padded_input_tensor[b, c, :], upsampled_error_tensor[b, k, :], mode='valid')
                        gradient_input[b, c, :] += convolve(upsampled_error_tensor[b, k, :], self.weights[k, c, :], "same")
                    gradient_bias[k] += np.sum(error_tensor[b, k, :])

        elif len(input_dims) == 2:  # 2D Convolution
            pad_y = self.convolution_shape[1] // 2 if self.convolution_shape[1] % 2 != 0 else (self.convolution_shape[1] // 2) - 1
            pad_x = self.convolution_shape[2] // 2 if self.convolution_shape[2] % 2 != 0 else (self.convolution_shape[2] // 2) - 1
            # Padding dimensions
            pad_width = ((0, 0),  # No padding for the batch dimension
                        (0, 0),  # No padding for the channel dimension
                        (pad_y, pad_y if self.convolution_shape[1] % 2 != 0 else pad_y+1),  # Pad height with half the kernel height row on top and bottom
                        (pad_x, pad_x if self.convolution_shape[2] % 2 != 0 else pad_x+1))  # Pad width with half the kernel width column on left and right

            # Apply padding
            padded_input_tensor = np.pad(self.input_tensor, pad_width=pad_width, mode='constant', constant_values=0)

            # Upsampling the error tensor to match the input spatial dimensions
            upsample_height = self.input_tensor.shape[2]
            upsample_width = self.input_tensor.shape[3]
            upsampled_error_tensor = np.zeros((batch_size, self.num_kernels, upsample_height, upsample_width))
            upsampled_error_tensor[:, :, ::self.stride_y, ::self.stride_x] = error_tensor

            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for c in range(channels):
                        gradient_weights[k, c, :, :] += correlate(padded_input_tensor[b, c, :, :], upsampled_error_tensor[b, k, :, :], mode='valid')
                        gradient_input[b, c, :, :] += convolve(upsampled_error_tensor[b, k, :, :], self.weights[k, c, :, :], "same")
                    gradient_bias[k] += np.sum(error_tensor[b, k, :, :])

        # Update gradients
        self._gradient_weights = gradient_weights
        self._gradient_bias = gradient_bias

        # Update weights using optimizer if set
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer is not None:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return gradient_input


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._bias_optimizer = copy.deepcopy(optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.convolution_shape[0] * np.prod(self.convolution_shape[1:])
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)
'''

