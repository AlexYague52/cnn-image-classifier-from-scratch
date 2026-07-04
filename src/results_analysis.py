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
from utils.testing import Testing, show_failures


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

history = load_model(encoder, path=r"C:\Users\Usuario\Documents\AI\model45.npz")

loss = history['loss']
acc = history['acc']
epoches = [i+1 for i in range(len(acc))]

fig, ax1 = plt.subplots()

"""
ax1.plot(epoches, loss, '-', color='tab:blue', label='error')
ax1.set_xlabel('Epoches')
ax1.set_ylabel('Error', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.plot(epoches, acc, '-', color='tab:orange', label='accuracy')
ax2.set_ylabel('Accuracy', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.axvline(x=23, color='black', linestyle = '--')
ax2.axhline(y=0.691, color = 'black', linestyle = '--')
ax1.axvspan(0, 50, color='green', alpha=0.08)
ax1.axvspan(50, max(epoches), color='red', alpha=0.08)
ax1.text(25, ax1.get_ylim()[1] * 0.97, 'Learning',ha='center', color='green', fontweight='bold')
ax1.text((50 + max(epoches)) / 2, ax1.get_ylim()[1] * 0.97,'Memorizing (overfitting)', ha='center', color='red', fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
fig.tight_layout()
plt.show()
"""

plt.plot(epoches,loss,'--', label = 'error')
plt.plot(epoches, acc, '-', label = 'accuracy')
plt.xlabel('Epoches')
plt.legend()
plt.show()


Testing(encoder, x_test,y_test)

show_failures(
    encoder, x_test, y_test,
    CAT_FOLDER, DOG_FOLDER, SIZE, TRAIN_RATIO,
    split_path1=r"C:\Users\Usuario\Documents\AI\neural_network\cnn\src\training_data\cats_split.npy",
    split_path2=r"C:\Users\Usuario\Documents\AI\neural_network\cnn\src\training_data\dogs_split.npy",
    undersample_path=r"C:\Users\Usuario\Documents\AI\neural_network\cnn\src\training_data\undersample.npy",
    n=8, per_row=4
)







