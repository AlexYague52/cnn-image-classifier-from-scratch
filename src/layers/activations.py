import numpy as np
from layers.layer import Layer


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_prime(self.input)


# -----------------------
# SIGMOID (whit clipping to avoid divergence)
# -----------------------
def sigmoid(x):
    x = np.clip(x, -50, 50)
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)


# -----------------------
# TANH
# -----------------------
def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


# -----------------------
# RELU
# -----------------------
def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return (x > 0).astype(float)


class ReLU(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)


# -----------------------
# LEAKY RELU
# -----------------------
class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        super().__init__(
            lambda x: np.where(x > 0, x, self.alpha * x),
            lambda x: np.where(x > 0, 1, self.alpha)
        )