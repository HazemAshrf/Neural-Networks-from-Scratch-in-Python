import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.label_tensor = None
        self.epsilon = np.finfo(float).eps

    def forward(self, prediction_tensor, label_tensor):
        # Store predictions and labels for backward pass
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor
        
        loss = -np.sum(label_tensor * np.log(prediction_tensor + self.epsilon)) # a float value
        return loss

    def backward(self, label_tensor):
        error_tensor = -label_tensor / (self.prediction_tensor + self.epsilon)
        return error_tensor # shape (batch_size, num_classes)
