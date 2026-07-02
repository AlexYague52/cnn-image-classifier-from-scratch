from utils.training import predict

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
