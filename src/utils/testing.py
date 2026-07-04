from utils.training import predict
from utils.preproces_datav1 import regularize_data
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from pathlib import Path


def Testing(network, x_test, y_test, verbose = True):

    n_test = len(x_test)
    correct = 0

    for i in range(n_test):
        pred  = predict(network, x_test[i]).item()
        label = y_test[i].item()
        if (pred > 0.5) == bool(label):
            correct += 1

    print(f"\nTest accuracy: {correct}/{n_test} = {correct / n_test:.3f}")

    if verbose:

        print("\n First 5 data from dataset1:")
        data1_idx = [i for i in range(len(y_test)) if y_test[i].item() == 0][:5]
        for i in data1_idx:
            pred = predict(network, x_test[i]).item()
            print(f"  pred={pred:.4f}  label1  {'Good' if pred < 0.5 else 'Bad'}")

        print("\nFirst 5 data from dataset2:")
        data2_idx = [i for i in range(len(y_test)) if y_test[i].item() == 1][:5]
        for i in data2_idx:
            pred = predict(network, x_test[i]).item()
            print(f"  pred={pred:.4f}  label2  {'Good' if pred > 0.5 else 'Bad'}")


def visualize_test(network, folder, size, per_row=4):

    images = regularize_data(folder, size)

    files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        files.extend(sorted(folder.rglob(ext)))
    files = sorted(files)

    n = len(images)
    n_rows = (n + per_row - 1) // per_row

    fig = plt.figure(figsize=(2.6 * per_row, 5.2 * n_rows))
    subfigs = fig.subfigures(n_rows, per_row, wspace=0.05, hspace=0.1)
    subfigs = np.atleast_2d(subfigs)

    for i, (img, f) in enumerate(zip(images, files)):
        pred = predict(network, img).item()
        label = "Dog" if pred > 0.5 else "Cat"
        score = pred if pred > 0.5 else 1 - pred
        color = "#2a7a2a" if label == "Dog" else "#d97706"

        r, c = i // per_row, i % per_row
        sf = subfigs[r, c]

        # cada par: original arriba, resize abajo
        ax_top, ax_bot = sf.subplots(2, 1)

        original = Image.open(f).convert("RGB")
        ax_top.imshow(original)
        ax_top.axis("off")

        ax_bot.imshow(np.transpose(img, (1, 2, 0)))
        ax_bot.axis("off")
        ax_bot.set_title(f"{label} ({score:.2f})",
                         fontsize=11, color=color, fontweight="bold", pad=4)

    # apagar subfiguras sobrantes si n no llena la última fila
    for j in range(n, n_rows * per_row):
        r, c = j // per_row, j % per_row
        for ax in subfigs[r, c].subplots(2, 1):
            ax.axis("off")

    plt.savefig("pets_predictions.png", dpi=200, bbox_inches="tight")
    plt.show()

def show_failures(network, x_test, y_test,
                  folder1, folder2, size, train_ratio,
                  split_path1, split_path2, undersample_path,
                  n=8, per_row=4):

    # 1. Reconstruir las rutas de los archivos de test
    def get_test_files(folder, split_path, us_key):
        extensions = ["*.png", "*.jpg", "*.jpeg"]
        files = []
        for ext in extensions:
            files.extend(sorted(Path(folder).rglob(ext)))
        files = sorted(files)

        idx = np.load(split_path)
        files = [files[i] for i in idx]

        n_train = int(len(files) * train_ratio)
        test_files = files[n_train:]

        us = np.load(undersample_path, allow_pickle=True).item()
        test_files = [test_files[i] for i in us[us_key]]
        return test_files

    files_0 = get_test_files(folder1, split_path1, 'idx_x1_test')
    files_1 = get_test_files(folder2, split_path2, 'idx_x2_test')

    # 2. Lookup: hash de pixels → ruta original
    lookup = {}
    for f in files_0 + files_1:
        img = Image.open(f).convert("RGB").resize((size, size))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        lookup[img.tobytes()] = f

    # 3. Encontrar fallos
    failures = []
    for i in range(len(x_test)):
        pred = predict(network, x_test[i]).item()
        label = y_test[i].item()
        if (pred > 0.5) != bool(label):
            original_path = lookup.get(x_test[i].tobytes())
            failures.append((i, pred, label, original_path))

    print(f"Total fallos: {len(failures)} / {len(x_test)}")

    if len(failures) == 0:
        print("No hay fallos!")
        return

    # 4. Elegir n aleatorios y mostrar
    chosen = np.random.choice(len(failures), min(n, len(failures)), replace=False)
    n_show = len(chosen)
    n_rows = (n_show + per_row - 1) // per_row

    fig, axes = plt.subplots(n_rows, per_row, figsize=(5 * per_row, 5.5 * n_rows))
    axes = np.atleast_2d(axes)

    for ax in axes.flat:
        ax.axis("off")

    for j, idx in enumerate(chosen):
        i, pred, label, original_path = failures[idx]
        row = j // per_row
        col = j % per_row

        # cargar original a resolución completa
        if original_path:
            original = Image.open(original_path).convert("RGB")
            axes[row, col].imshow(original)
        else:
            axes[row, col].imshow(np.transpose(x_test[i], (1, 2, 0)))

        true_label = "Dog" if label == 1 else "Cat"
        pred_label = "Dog" if pred > 0.5 else "Cat"
        confidence = pred if pred > 0.5 else 1 - pred

        axes[row, col].set_title(
            f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})",
            fontsize=12, color="red", fontweight="bold"
        )

    plt.suptitle(f"Misclassified samples ({len(failures)}/{len(x_test)} total failures)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("failures.png", dpi=150, bbox_inches="tight")
    plt.show()

