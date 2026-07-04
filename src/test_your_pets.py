
from pathlib import Path
from utils.preproces_datav1 import dataset
import numpy as np
import os

from layers.dense import Dense
from layers.convolutional import Convolutional
from layers.reshape import Flatten
from layers.activations import Sigmoid, Tanh, ReLU, LeakyReLU
from layers.maxpool import MaxPooling
from utils.error import binary_cross_entropy, binary_cross_entropy_prime
from utils.training import train, predict, pool_output_shape, load_model
from utils.testing import Testing, visualize_test, show_failures



FOLDER = Path(r'C:\Users\Usuario\Documents\AI\neural_network\cnn\src\your_pets')

# ARCHITECTURE

input_shape = (3,32,32)
conv1  = Convolutional(input_shape, 3, 8)
shape1 = pool_output_shape(conv1.output_shape)
conv2  = Convolutional(shape1, 3, 16)
shape2 = pool_output_shape(conv2.output_shape)

encoder = [
    conv1, LeakyReLU(), MaxPooling(2, 2),
    conv2, LeakyReLU(), MaxPooling(2, 2),
    Flatten(shape2),
    Dense(int(np.prod(shape2)), 64), LeakyReLU(),
    Dense(64, 1), Sigmoid()
]

history = load_model(encoder, path=r"C:\Users\Usuario\Documents\AI\model45.npz")

visualize_test(encoder, FOLDER,32)


