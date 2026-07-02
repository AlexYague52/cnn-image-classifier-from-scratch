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


def shuffle_and_split(folder, size, train_ratio,
                      split_path=None, load_split=False,
                      angle_rot=False, Y_flip=False, bright=False,
                      saturate=False, noise=False, jitter=False, erase=False):
    base = regularize_data(folder, size)

    if load_split and split_path and Path(split_path).exists():
        idx = np.load(split_path)
        print(f"  Split loaded from {split_path}")
    else:
        idx = np.random.permutation(len(base))
        if split_path:
            np.save(split_path, idx)
            print(f"  Split saved in {split_path}")

    base = [base[i] for i in idx]

    n_train    = int(len(base) * train_ratio)
    base_train = base[:n_train]
    base_test  = base[n_train:]

    # augmentacion solo en train
    any_aug = any([angle_rot, Y_flip, bright, saturate, noise, jitter, erase])
    if any_aug:
        train_data = []
        for img in base_train:
            train_data.append(img)
            train_data.extend(data_augmentation(img, angle_rot, Y_flip, bright,
                                                saturate, noise, jitter, erase))
        train = np.array(train_data, dtype=np.float32)
    else:
        train = np.array(base_train, dtype=np.float32)

    test = np.array(base_test, dtype=np.float32)

    return train, test


def dataset(path1, path2, size, train_ratio,
            split_path_x1=None, split_path_x2=None,
            undersample_path=None, load_split=False, flags=True,
            angle_rot=False, Y_flip=False, bright=False, saturate=False,
            noise=False, jitter=False, erase=False):

    folder1 = Path(path1)
    folder2 = Path(path2)

    x1_train, x1_test = shuffle_and_split(
        folder1, size, train_ratio,
        split_path=split_path_x1, load_split=load_split,
        angle_rot=angle_rot, Y_flip=Y_flip, bright=bright, saturate=saturate,
        noise=noise, jitter=jitter, erase=erase)
    if flags:
        print(f"Loaded {folder1.name}: train={len(x1_train)}  test={len(x1_test)}")

    x2_train, x2_test = shuffle_and_split(
        folder2, size, train_ratio,
        split_path=split_path_x2, load_split=load_split,
        angle_rot=angle_rot, Y_flip=Y_flip, bright=bright, saturate=saturate,
        noise=noise, jitter=jitter, erase=erase)
    if flags:
        print(f"Loaded {folder2.name}: train={len(x2_train)}  test={len(x2_test)}")

    # undersampling
    if len(x1_train) != len(x2_train) or len(x1_test) != len(x2_test):
        if flags:
            print("Performing undersampling...")

        n_train = min(len(x1_train), len(x2_train))
        n_test  = min(len(x1_test),  len(x2_test))

        if load_split and undersample_path and Path(undersample_path).exists():
            us = np.load(undersample_path, allow_pickle=True).item()
            idx_x1_train = us['idx_x1_train']
            idx_x2_train = us['idx_x2_train']
            idx_x1_test  = us['idx_x1_test']
            idx_x2_test  = us['idx_x2_test']
            print(f"  Undersample loaded from {undersample_path}")
        else:
            idx_x1_train = np.random.choice(len(x1_train), n_train, replace=False)
            idx_x2_train = np.random.choice(len(x2_train), n_train, replace=False)
            idx_x1_test  = np.random.choice(len(x1_test),  n_test,  replace=False)
            idx_x2_test  = np.random.choice(len(x2_test),  n_test,  replace=False)
            if undersample_path:
                np.save(undersample_path, {
                    'idx_x1_train': idx_x1_train,
                    'idx_x2_train': idx_x2_train,
                    'idx_x1_test':  idx_x1_test,
                    'idx_x2_test':  idx_x2_test,
                })
                print(f"  Undersample saved in {undersample_path}")

        x1_train = x1_train[idx_x1_train]
        x2_train = x2_train[idx_x2_train]
        x1_test  = x1_test [idx_x1_test]
        x2_test  = x2_test [idx_x2_test]

        if flags:
            print(f"  After undersampling:")
            print(f"  x1_train: {len(x1_train)}  x2_train: {len(x2_train)}")
            print(f"  x1_test:  {len(x1_test)}   x2_test:  {len(x2_test)}")

    x_train, y_train = make_xy(x1_train, x2_train)
    x_test,  y_test  = make_xy(x1_test,  x2_test)

    if flags:
        print(f"Balance ratio: {(y_train==0).sum()/len(y_train):.3f}")
        print(f"Initial predicted loss: {-np.log(0.5):.3f}")
        print("Datasets ready!")

    return x_train, y_train, x_test, y_test