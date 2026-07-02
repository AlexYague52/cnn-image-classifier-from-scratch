import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from utils.preproces_datav1 import dataset

from layers.dense import Dense
from layers.convolutional import Convolutional
from layers.reshape import Flatten
from layers.activations import Sigmoid, Tanh, ReLU, LeakyReLU
from layers.maxpool import MaxPooling
from utils.error import binary_cross_entropy, binary_cross_entropy_prime
from utils.training import train, predict, pool_output_shape, load_model
from utils.testing import Testing


CAT_FOLDER = Path(r'C:\Users\Usuario\Documents\AI\neural_network\cnn\src\catdog_dataset\cat')
DOG_FOLDER = Path(r'C:\Users\Usuario\Documents\AI\neural_network\cnn\src\catdog_dataset\dog')

SIZE = 32
TRAIN_RATIO = 0.8

x_train, y_train, x_test, y_test = dataset(
    CAT_FOLDER, DOG_FOLDER, SIZE, TRAIN_RATIO,
    split_path_x1=r"C:\Users\Usuario\Documents\AI\neural_network\cnn\src\training_data\cats_split.npy",
    split_path_x2=r"C:\Users\Usuario\Documents\AI\neural_network\cnn\src\training_data\dogs_split.npy",
    undersample_path=r"C:\Users\Usuario\Documents\AI\neural_network\cnn\src\training_data\undersample.npy",
    load_split=True,
    flags=True
)


# ARCHITECTURE

input_shape = x_train[0].shape  
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

history = load_model(encoder, path=r"C:\Users\Usuario\Documents\AI\model.npz")

loss = history['loss']
acc = history['acc']
epoches = [i+1 for i in range(len(acc))]

plt.plot(epoches,loss,'--', label = 'error')
plt.plot(epoches, acc, '-.', label = 'accuracy')
plt.xlabel('Epoches')
plt.legend()
plt.show()

Testing(encoder, x_test,y_test)





