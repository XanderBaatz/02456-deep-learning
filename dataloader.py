import numpy as np
import tarfile
import pickle
import os

# Path to your CIFAR-10 archive
cifar_path = 'data/cifar-10-python.tar.gz'

# Extract the archive
with tarfile.open(cifar_path, 'r:gz') as tar:
    tar.extractall()
    extracted_dir = tar.getnames()[0]  # usually 'cifar-10-batches-py'

# Function to load a batch
def load_batch(batch_filename):
    with open(batch_filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')  # encoding='bytes' needed for Python 3
        data = batch[b'data']  # shape (10000, 3072)        
        labels = batch[b'labels']  # list of length 10000
        return data, labels

# Load all training batches
train_data = []
train_labels = []

for i in range(1, 6):
    batch_file = os.path.join(extracted_dir, f'data_batch_{i}')
    data, labels = load_batch(batch_file)
    train_data.append(data)
    train_labels += labels

# Combine into single arrays
X_train = np.concatenate(train_data)  # shape (50000, 3072)
y_train = np.array(train_labels)      # shape (50000,)

# Load test batch
X_test, y_test = load_batch(os.path.join(extracted_dir, 'test_batch'))
X_test = np.array(X_test)
y_test = np.array(y_test)

# Optionally, reshape to (num_samples, 3, 32, 32)
X_train = X_train.reshape(-1, 3, 32, 32)
X_test = X_test.reshape(-1, 3, 32, 32)

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)
