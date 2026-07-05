import numpy as np
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

# SET-UP DATASET

PKMN_FOLDER = Path(r'...\src\dataset\pokemon')
DGMN_FOLDER = Path(r'...\src\dataset\digimon')

SIZE = 32
TRAIN_RATIO = 0.8

x_train, y_train, x_test, y_test = dataset(
    PKMN_FOLDER, DGMN_FOLDER, SIZE, TRAIN_RATIO,
    split_path_x1=r"...\src\training_data\PKMN_split.npy",
    split_path_x2=r"...\src\training_data\GMN_split.npy",
    undersample_path=r"...\src\training_data\undersample_PD.npy",
    flags=True,
    Y_flip=True,
    bright=True,
    jitter=True,
    noise=True,
)

print(type(y_test[0])) #returns array!!! (1,)
      
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
# TRANING AND SAVING

print('Training!...')

result = train(
    encoder,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=50,
    learning_rate=0.003,
    verbose=True, save_params=True, save_learning=True)

# TESTING

Testing(encoder, x_test, y_test)
