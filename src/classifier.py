import numpy as np
from pathlib import Path
from utils.preproces_data import augmented_dataset, dataset

from layers.dense import Dense
from layers.convolutional import Convolutional
from layers.reshape import Flatten
from layers.activations import Sigmoid, Tanh, ReLU, LeakyReLU
from layers.maxpool import MaxPooling
from utils.error import binary_cross_entropy, binary_cross_entropy_prime
from utils.training import train, predict, pool_output_shape
from utils.testing import Testing

# SET-UP DATASET

#fanarts
POKEMON_FOLDER = Path(r'C:\Users\Usuario\Documents\AI\neural_network\cnn\src\dataset\pokemon')
DIGIMON_FOLDER = Path(r'C:\Users\Usuario\Documents\AI\neural_network\cnn\src\dataset\digimon')

#sprites
#POKEMON_FOLDER = Path(r'C:\Users\Usuario\Documents\AI\neural_network\cnn\dataset\pokemon')
#DIGIMON_FOLDER = Path(r'C:\Users\Usuario\Documents\AI\neural_network\cnn\dataset\digimon')

SIZE = 8
TRAIN_RATIO = 0.8

x_train, y_train, x_test, y_test = dataset(
    POKEMON_FOLDER, DIGIMON_FOLDER, SIZE, TRAIN_RATIO,
    flags=True,
    angle_rot=True,
    Y_flip=True,
    bright=False,
    saturate=False,
    noise=True,
    jitter=True,
    erase=True
)

print(type(y_test[0])) #returns array!!! (1,)
      
# ARCHITECTURE

input_shape = x_train[0].shape
conv1 = Convolutional(input_shape, 3, 16)
shape1 = pool_output_shape(conv1.output_shape)

if SIZE == 8:
    # (3,8,8) → conv → (16,6,6) → pool → (16,3,3) → flatten(144) → 32 → 1
    encoder = [
        conv1, LeakyReLU(), MaxPooling(2, 2),
        Flatten(shape1),
        Dense(int(np.prod(shape1)), 32), LeakyReLU(),
        Dense(32, 1), Sigmoid()
    ]

elif SIZE == 16:
    # (3,16,16) → conv → (16,14,14) → pool → (16,7,7) → flatten(784) → 32 → 1
    encoder = [
        conv1, LeakyReLU(), MaxPooling(2, 2),
        Flatten(shape1),
        Dense(int(np.prod(shape1)), 32), LeakyReLU(),
        Dense(32, 1), Sigmoid()
    ]

elif SIZE == 32:
    # (3,32,32) → conv → (16,30,30) → pool → (16,15,15)
    #           → conv → (32,13,13) → pool → (32,6,6) → flatten(1152) → 64 → 1
    conv2  = Convolutional(shape1, 3, 32)
    shape2 = pool_output_shape(conv2.output_shape)
    encoder = [
        conv1, LeakyReLU(), MaxPooling(2, 2),
        conv2, LeakyReLU(), MaxPooling(2, 2),
        Flatten(shape2),
        Dense(int(np.prod(shape2)), 64), LeakyReLU(),
        Dense(64, 1), Sigmoid()
    ]

elif SIZE == 64:
    # (3,64,64) → conv → (16,62,62) → pool → (16,31,31)
    #           → conv → (32,29,29) → pool → (32,14,14)
    #           → conv → (64,12,12) → pool → (64,6,6) → flatten(2304) → 128 → 1
    conv2  = Convolutional(shape1, 3, 32)
    shape2 = pool_output_shape(conv2.output_shape)
    conv3  = Convolutional(shape2, 3, 64)
    shape3 = pool_output_shape(conv3.output_shape)
    encoder = [
        conv1, LeakyReLU(), MaxPooling(2, 2),
        conv2, LeakyReLU(), MaxPooling(2, 2),
        conv3, LeakyReLU(), MaxPooling(2, 2),
        Flatten(shape3),
        Dense(int(np.prod(shape3)), 128), LeakyReLU(),
        Dense(128, 1), Sigmoid()
    ]

else:
    raise ValueError(f"SIZE {SIZE} no soportado. Usa 8, 16, 32 o 64.")

# TRAINING AND SAVING

print('Training!...')

result = train(
    encoder,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.001,
    verbose=True)


# TESTING

Testing(encoder, x_test, y_test)

