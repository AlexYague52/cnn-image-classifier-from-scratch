import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import rotate as scipy_rotate


def regularize_data(folder, size):
    dataset = []
    extensions = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    for ext in extensions:
        files.extend(sorted(folder.rglob(ext)))
    for img_file in sorted(files):
        img = Image.open(img_file).convert("RGB")
        img = img.resize((size, size))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        dataset.append(img)
    return dataset


# ── augmentaciones ───────────────────────────────────────────────────────

def add_noise(image, std=0.05):
    noise = np.random.normal(0, std, image.shape)
    return np.clip(image + noise, 0.0, 1.0)

def color_jitter(image, strength=0.3):
    factors = np.random.uniform(1 - strength, 1 + strength, size=(3, 1, 1))
    return np.clip(image * factors, 0.0, 1.0)

def random_erasing(image, prob=0.5, max_area=0.2):
    if np.random.rand() > prob:
        return image.copy()
    img = image.copy()
    C, H, W = image.shape
    h = int(np.sqrt(max_area) * H)
    w = int(np.sqrt(max_area) * W)
    top  = np.random.randint(0, max(1, H - h))
    left = np.random.randint(0, max(1, W - w))
    img[:, top:top+h, left:left+w] = np.random.uniform(0, 1)
    return img

def grayscale(image):
    gray = np.mean(image, axis=0, keepdims=True)
    return np.repeat(gray, 3, axis=0)


def data_augmentation(image, angle_rot=True, Y_flip=True, bright=True,
                      saturate=True, noise=True, jitter=True, erase=True):
    aug = []

    if angle_rot:
        angle = np.random.uniform(-10, 10)
        img_hwc = np.transpose(image, (1, 2, 0))
        rotated_hwc = scipy_rotate(img_hwc, angle=angle, axes=(0, 1), reshape=False)
        rotated = np.transpose(rotated_hwc, (2, 0, 1))
        aug.append(np.clip(rotated, 0.0, 1.0))

    if Y_flip:
        aug.append(np.flip(image, axis=2).copy())

    if bright:
        aug.append(np.clip(image * 1.2, 0.0, 1.0))

    if saturate:
        gray = np.mean(image, axis=0, keepdims=True)
        saturated = np.clip(gray + 1.5 * (image - gray), 0.0, 1.0)
        aug.append(saturated)

    if noise:
        aug.append(add_noise(image))

    if jitter:
        aug.append(color_jitter(image))

    if erase:
        aug.append(random_erasing(image))

    return aug


def make_xy(data_0, data_1):
    x = np.concatenate([data_0, data_1], axis=0)
    y = np.concatenate([
        np.zeros((len(data_0), 1), dtype=np.float32),
        np.ones ((len(data_1), 1), dtype=np.float32),
    ], axis=0)
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


def augmented_dataset(folder, size, train_ratio=0.8,
                      angle_rot=True, Y_flip=True, bright=True, saturate=True,
                      noise=True, jitter=True, erase=True):

    base = regularize_data(folder, size)

    idx = np.random.permutation(len(base))
    base = [base[i] for i in idx]

    n_train = int(len(base) * train_ratio)
    base_train = base[:n_train]
    base_test  = base[n_train:]

    train_data = []
    for img in base_train:
        train_data.append(img)
        train_data.extend(data_augmentation(img, angle_rot, Y_flip, bright,
                                            saturate, noise, jitter, erase))

    test_data = list(base_test)

    return (np.array(train_data, dtype=np.float32),
            np.array(test_data,  dtype=np.float32))


def dataset(path1, path2, size, train_ratio, flags=True,
            angle_rot=True, Y_flip=True, bright=True, saturate=True,
            noise=True, jitter=True, erase=True):

    pokemon_folder = Path(path1)
    digimon_folder = Path(path2)

    pkmn_train, pkmn_test = augmented_dataset(pokemon_folder, size, train_ratio,
                                               angle_rot, Y_flip, bright, saturate,
                                               noise, jitter, erase)
    if flags:
        print("Loading pokemon dataset...")
        print(f"  train_pkmn: {len(pkmn_train)}  test_pkmn: {len(pkmn_test)}")

    dgmn_train, dgmn_test = augmented_dataset(digimon_folder, size, train_ratio,
                                               angle_rot, Y_flip, bright, saturate,
                                               noise, jitter, erase)
    if flags:
        print("Loading digimon dataset...")
        print(f"  train_dgmn: {len(dgmn_train)}  test_dgmn: {len(dgmn_test)}")

    if len(pkmn_test) != len(dgmn_test) or len(pkmn_train) != len(dgmn_train):
        if flags:
            print('Performing undersampling...')

        n_train = min(len(pkmn_train), len(dgmn_train))
        n_test  = min(len(pkmn_test),  len(dgmn_test))

        if len(dgmn_train) > len(pkmn_train):
            idx_train = np.random.choice(len(dgmn_train), size=n_train, replace=False)
            idx_test  = np.random.choice(len(dgmn_test),  size=n_test,  replace=False)
            dgmn_train = dgmn_train[idx_train]
            dgmn_test  = dgmn_test[idx_test]
        else:
            idx_train = np.random.choice(len(pkmn_train), size=n_train, replace=False)
            idx_test  = np.random.choice(len(pkmn_test),  size=n_test,  replace=False)
            pkmn_train = pkmn_train[idx_train]
            pkmn_test  = pkmn_test[idx_test]

        if flags:
            print(f"\nAfter undersampling:")
            print(f"  pkmn_train: {len(pkmn_train)}  dgmn_train: {len(dgmn_train)}")
            print(f"  pkmn_test:  {len(pkmn_test)}   dgmn_test:  {len(dgmn_test)}")

    x_train, y_train = make_xy(pkmn_train, dgmn_train)
    x_test,  y_test  = make_xy(pkmn_test,  dgmn_test)

    if flags:
        print("Datasets ready!")

    print(f"\nRatio balance — pokemon: {(y_train==0).sum()/len(y_train):.3f}")
    print(f"Loss inicial esperado: {-np.log(0.5):.3f}")
    return x_train, y_train, x_test, y_test