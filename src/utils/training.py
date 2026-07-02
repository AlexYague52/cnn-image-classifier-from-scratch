import numpy as np
import time

def pool_output_shape(input_shape, pool_size=2, stride=2):
    C, H, W = input_shape
    return (C,
            (H - pool_size) // stride + 1,
            (W - pool_size) // stride + 1)

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True, save_params = False, save_learning = False):
    error_list = []
    acc_list = []
    result = {}

    for e in range(epochs):
        t_start = time.time()
        error = 0
        correct = 0

        idx = np.random.choice(len(x_train),1200,replace=False)
        x_epoch = x_train[idx]
        y_epoch = y_train[idx]

        for x, y in zip(x_epoch, y_epoch):
            output = predict(network, x)
            error += loss(y, output)
            # reutiliza output del forward, sin llamar predict de nuevo
            correct += int((output.item() > 0.5) == bool(y.item()))

            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        n = len(x_epoch)
        error /= n
        acc = correct / n

        t_epoch = time.time() - t_start

        error_list.append(error)
        acc_list.append(acc)

        if verbose:
            print(f"epoch {e+1}/{epochs}  loss={error:.4f}  acc={acc:.3f} time={t_epoch:.1f}s")
        
        if save_learning:
            result['loss'] = error_list
            result['acc'] = acc_list 

        if (e + 1) % 5 == 0:
            if save_params:
                result['params'] = [layer.get_params() for layer in network]
            save_model(network,result, path="model.npz", verbose = True)
           
    
    
    return result

def save_model(network, history=None, path="model.npz", verbose = True):
    data = {}

    # parámetros de cada capa
    for i, layer in enumerate(network):
        for key, val in layer.get_params().items():
            data[f"layer{i}_{key}"] = val

    # historial de entrenamiento
    if history:
        if 'loss' in history:
            data['loss'] = np.array(history['loss'])
        if 'acc' in history:
            data['acc'] = np.array(history['acc'])

    np.savez(path, **data)

    if verbose:
        print(f"Data saved in {path}")


def load_model(network, path="model.npz"):
    data = np.load(path)
    history = {}

    for i, layer in enumerate(network):
        params = {k.replace(f"layer{i}_", ""): data[k]
                  for k in data if k.startswith(f"layer{i}_")}
        if params:
            layer.set_params(params)

    if 'loss' in data:
        history['loss'] = data['loss'].tolist()
    if 'acc' in data:
        history['acc'] = data['acc'].tolist()

    print(f"Loaded from {path}")
    return history
