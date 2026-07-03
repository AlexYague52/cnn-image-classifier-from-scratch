# CNN Image Classifier from Scratch

A convolutional neural network built entirely from scratch in NumPy, with no deep learning frameworks (no PyTorch, no TensorFlow). Every layer (convolution, max-pooling, dense, activations) implements its own forward and backward pass by hand, making this a transparent, educational project about how a CNN really works under the hood.


## What it does

Binary image classification on two datasets:

- **Pokémon vs. Digimon** (stylized sprites) → ~99% test accuracy
- **Cats vs. Dogs** (real photographs from Kaggle) → ~0.70 test accuracy, a harder generalization problem


## Highlights

- **Fully hand-written backpropagation** for every layer, including the non-trivial cases: gradient routing through max-pool ties, and conv backward via `correlate2d` / `convolve2d`.
- **Modular layer API** with `forward` / `backward` / `get_params` / `set_params` that lets you compose architectures as a Python list, the same pattern the major frameworks expose.
- **Training utilities**: mini-epoch subsampling, `.npz` checkpointing every N epochs, and save/load of full model state.
- **Reproducible data pipeline**: deterministic train/test splits and undersampling indices saved to disk, class balancing, and optional augmentation (rotation, flip, brightness, saturation, jitter, noise, random erasing).
- **Numerically stable losses**: binary cross-entropy and MSE with clipped sigmoid and log-clipping to avoid NaN / overflow.


## Project structure

\`\`\`
src/
├── layers/
│   ├── layer.py              # Base class (forward / backward interface)
│   ├── dense.py              # Fully connected layer with He initialization
│   ├── convolutional.py      # 2D convolution (scipy.signal)
│   ├── maxpool.py            # Max-pooling with correct gradient routing
│   ├── reshape.py            # Flatten and Reshape utilities
│   ├── activation.py         # Generic activation wrapper
│   └── activations.py        # Sigmoid, Tanh, ReLU, LeakyReLU
│
├── utils/
│   ├── training.py           # Training loop, save/load, predict
│   ├── testing.py            # Test evaluation with per-class examples
│   ├── preproces_datav1.py   # Image loading, augmentation, dataset builder
│   └── error.py              # Loss functions (BCE, MSE)
│
└── classifier_catdog.py      # Main script: data → architecture → train → test
\`\`\`


## Architecture (Cats vs. Dogs)

\`\`\`
Input (3, 32, 32)
  → Conv2D (8 filters, 3×3)   → LeakyReLU  → MaxPool (2×2)
  → Conv2D (16 filters, 3×3)  → LeakyReLU  → MaxPool (2×2)
  → Flatten (576)
  → Dense (576 → 64)          → LeakyReLU
  → Dense (64 → 1)            → Sigmoid
  → Binary Cross-Entropy Loss
\`\`\`


## Getting started

### Requirements

Python 3.8+ and three packages:

\`\`\`bash
pip install numpy scipy pillow matplotlib
\`\`\`

### Dataset setup

Download your images and place them with this structure:

\`\`\`
datasets/
├── cat/       # cat photos (.jpg / .png)
├── dog/       # dog photos
├── pokemon/   # pokémon sprites
└── digimon/   # digimon sprites
\`\`\`

The Cats vs. Dogs images come from the classic [Kaggle Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats). Update the folder paths at the top of \`classifier_catdog.py\` to point to your dataset directories.

### Run

\`\`\`bash
python src/classifier_catdog.py
\`\`\`

The script loads and preprocesses images, trains the network, saves checkpoints every 5 epochs, and prints test accuracy at the end.


## Notes on overfitting

The Cats vs. Dogs task is a good illustration of overfitting on a limited dataset. Training the network for many epochs drives training accuracy to 100% while test accuracy stalls and then degrades: the model memorizes the training images instead of learning generalizable patterns. In practice the best test accuracy is reached fairly early (around epoch 45–50 in this setup), after which further training hurts generalization. Techniques to address this include early stopping, data augmentation (already implemented in the preprocessing pipeline), and regularization.


## Tech

| Component | Library |
|-----------|---------|
| Linear algebra and array ops | NumPy |
| 2D convolution / correlation | SciPy (\`scipy.signal\`) |
| Image loading and resizing | Pillow |
| Plotting training curves | Matplotlib |
| Everything else | Written from scratch |


## License

MIT