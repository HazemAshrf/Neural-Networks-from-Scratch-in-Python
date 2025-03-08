import numpy as np

class Constant:
    def __init__(self, value=0.1):
        self.value = value

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        return np.full(weights_shape, self.value)

class UniformRandom:
    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        return np.random.uniform(0, 1, size=weights_shape).astype(np.float32)

class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        stddev = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, stddev, size=weights_shape).astype(np.float32)

class He:
    def initialize(self, weights_shape, fan_in, fan_out=None):
        stddev = np.sqrt(2 / fan_in)
        return np.random.normal(0, stddev, size=weights_shape).astype(np.float32)
