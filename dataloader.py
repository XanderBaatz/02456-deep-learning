import numpy as np
import tarfile
import pickle
import os


def load_cifar10(cifar_path='data/cifar-10-python.tar.gz', val_fraction=0.1, random_seed=111):
    """
    Load CIFAR-10 dataset and return train, validation, and test sets.
    Automatically combines all training batches. No manual batch handling needed.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
        Shapes: (N, 3, 32, 32), (N,)
    """
    # Extract archive
    with tarfile.open(cifar_path, 'r:gz') as tar:
        tar.extractall()
        extracted_dir = tar.getnames()[0]

    # Function to load a single batch
    def load_batch_file(batch_file):
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            data = batch[b'data']      # (10000, 3072)
            labels = batch[b'labels']  # list of 10000
            return data, labels

    # Automatically load all training batches
    all_data = []
    all_labels = []
    for i in range(1, 6):
        batch_file = os.path.join(extracted_dir, f'data_batch_{i}')
        data, labels = load_batch_file(batch_file)
        all_data.append(data)
        all_labels += labels

    X_all = np.concatenate(all_data)           # (50000, 3072)
    y_all = np.array(all_labels)               # (50000,)

    # Load test batch
    X_test, y_test = load_batch_file(os.path.join(extracted_dir, 'test_batch'))
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Reshape images
    X_all = X_all.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)

    # Split train into train/validation
    num_train = X_all.shape[0]
    num_val = int(num_train * val_fraction)

    np.random.seed(random_seed)
    indices = np.random.permutation(num_train)

    X_val = X_all[indices[:num_val]]
    y_val = y_all[indices[:num_val]]
    X_train = X_all[indices[num_val:]]
    y_train = y_all[indices[num_val:]]

    print("Train set:", X_train.shape, y_train.shape)
    print("Validation set:", X_val.shape, y_val.shape)
    print("Test set:", X_test.shape, y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test



