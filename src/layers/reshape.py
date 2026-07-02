from layers.layer import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (np.prod(input_shape), 1)

    def forward(self, input):
        return input.reshape(self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)

class Reshape(Layer):
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)