import numpy as np
from layers.layer import Layer

class MaxPooling(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        C, H, W = input.shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1
        self.output_shape = (C, H_out, W_out)
        output = np.zeros(self.output_shape)

        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * self.stride #pasos regulados por stride
                    w_start = j * self.stride
                    patch = input[c,
                                  h_start:h_start + self.pool_size,
                                  w_start:w_start + self.pool_size] #esto te selecciona el recuadro del canal que estas considerando
                    output[c, i, j] = np.max(patch) #guardas el maximo del recuadro para el canal

        return output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros(self.input.shape)
        C, H_out, W_out = self.output_shape

        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * self.stride
                    w_start = j * self.stride
                    patch = self.input[c,
                                       h_start:h_start + self.pool_size,
                                       w_start:w_start + self.pool_size]
                    # solo el primer máximo recibe el gradiente (fix bug ties)
                    mask = np.zeros_like(patch)
                    idx = np.unravel_index(np.argmax(patch), patch.shape)
                    mask[idx] = 1
                    input_gradient[c,
                                   h_start:h_start + self.pool_size,
                                   w_start:w_start + self.pool_size] += mask * output_gradient[c, i, j]

        return input_gradient
    