import numpy as np
from pathlib import Path
from preproces_data import augmented_dataset

from dense import Dense
from convolutional import Convolutional
from reshape import Flatten
from activations import Sigmoid, Tanh, ReLU, LeakyReLU
from maxpool import MaxPooling
from error import binary_cross_entropy, binary_cross_entropy_prime
from training import train, predict


# ── rutas ──────────────────────────────────────────────────────────────
POKEMON_FOLDER = Path(r'C:\Users\Usuario\Documents\AI\neural_network\neural_network\cnn\dataset\pokemon')
DIGIMON_FOLDER = Path(r'C:\Users\Usuario\Documents\AI\neural_network\neural_network\cnn\dataset\digimon')

SIZE        = 16
TRAIN_RATIO = 0.8

# ── carga y split correcto ───────────────────────────────────────────────
print("Cargando pokemon...")
pkmn_train, pkmn_test = augmented_dataset(POKEMON_FOLDER, SIZE, train_ratio=TRAIN_RATIO)
print(f"  train: {len(pkmn_train)}  test: {len(pkmn_test)}")

print("Cargando digimon...")
dgmn_train, dgmn_test = augmented_dataset(DIGIMON_FOLDER, SIZE, train_ratio=TRAIN_RATIO)
print(f"  train: {len(dgmn_train)}  test: {len(dgmn_test)}")

# ── undersampling: recortar digimon para igualar pokemon ─────────────────
n_train = len(pkmn_train)
n_test  = len(pkmn_test)

idx_train = np.random.choice(len(dgmn_train), size=n_train, replace=False)
idx_test  = np.random.choice(len(dgmn_test),  size=n_test,  replace=False)
dgmn_train = dgmn_train[idx_train]
dgmn_test  = dgmn_test[idx_test]

print(f"\nTras undersampling:")
print(f"  pkmn_train: {len(pkmn_train)}  dgmn_train: {len(dgmn_train)}")
print(f"  pkmn_test:  {len(pkmn_test)}   dgmn_test:  {len(dgmn_test)}")


# ── concatenar y etiquetar ──────────────────────────────────────────────
def make_xy(data_0, data_1):
    x = np.concatenate([data_0, data_1], axis=0)
    y = np.concatenate([
        np.zeros((len(data_0), 1), dtype=np.float32),
        np.ones ((len(data_1), 1), dtype=np.float32),
    ], axis=0)
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]

x_train, y_train = make_xy(pkmn_train, dgmn_train)
x_test,  y_test  = make_xy(pkmn_test,  dgmn_test)


# ── resumen ─────────────────────────────────────────────────────────────
print(f"\nx_train: {x_train.shape}  y_train: {y_train.shape}")
print(f"x_test:  {x_test.shape}  y_test:  {y_test.shape}")
print(f"Balance train — pokemon: {int((y_train==0).sum())}  digimon: {int((y_train==1).sum())}")
print(f"Ratio pokemon/total: {(y_train==0).sum() / len(y_train):.3f}")
print("\nDatasets ready!")


# ── arquitectura ─────────────────────────────────────────────────────────
def pool_output_shape(input_shape, pool_size=2, stride=2):
    C, H, W = input_shape
    return (C,
            (H - pool_size) // stride + 1,
            (W - pool_size) // stride + 1)

input_shape = x_train[0].shape  # (3, 8, 8)

conv1  = Convolutional(input_shape, 3, 16)
shape1 = pool_output_shape(conv1.output_shape)

flat_size = int(np.prod(shape1))

encoder = [
    conv1, LeakyReLU(), MaxPooling(2, 2),
    Flatten(shape1),
    Dense(flat_size, 32), LeakyReLU(),
    Dense(32, 1), Sigmoid()
]

print(f"Arquitectura: input {input_shape} → conv(16,6,6) → pool(16,3,3) → flatten {flat_size} → 32 → 1")


# ── entrenamiento ────────────────────────────────────────────────────────
print('Training!...')
train(
    encoder,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.001,
    verbose=True
)


# ── test ─────────────────────────────────────────────────────────────────
n_test = min(100, len(x_test))
correct = 0

for i in range(n_test):
    pred  = predict(encoder, x_test[i]).item()
    label = y_test[i].item()
    if (pred > 0.5) == bool(label):
        correct += 1

print(f"\nTest accuracy: {correct}/{n_test} = {correct / n_test:.3f}")

print("\nPrimeros 5 pokemon del test:")
pkmn_idx = [i for i in range(len(y_test)) if y_test[i].item() == 0][:5]
for i in pkmn_idx:
    pred = predict(encoder, x_test[i]).item()
    print(f"  pred={pred:.4f}  label=pokemon  {'✓' if pred < 0.5 else '✗'}")

print("\nPrimeros 5 digimon del test:")
dgmn_idx = [i for i in range(len(y_test)) if y_test[i].item() == 1][:5]
for i in dgmn_idx:
    pred = predict(encoder, x_test[i]).item()
    print(f"  pred={pred:.4f}  label=digimon  {'✓' if pred > 0.5 else '✗'}")