import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit


def balanced_train_test_split(sequence, label, test_size=0.2, random_state=42):
    # Convert inputs to numpy arrays if they aren't already
    sequence = np.array(sequence)
    label = np.array(label)

    # Create stratified split
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state)

    # Get train and test indices
    train_idx, test_idx = next(sss.split(sequence, label))

    # Split the data
    X_train = sequence[train_idx]
    X_test = sequence[test_idx]

    # Split the labels
    y_train_raw = label[train_idx]
    y_test_raw = label[test_idx]

    # Convert to one-hot encoding
    num_classes = len(np.unique(label))
    y_train = tf.keras.utils.to_categorical(
        y_train_raw, num_classes=num_classes, dtype='float32')
    y_test = tf.keras.utils.to_categorical(
        y_test_raw, num_classes=num_classes, dtype='float32')

    # Print distribution of labels in train and test sets
    print("\nLabel distribution:")
    print("Class\tTrain\tTest")
    for i in range(num_classes):
        train_count = np.sum(y_train_raw == i)
        test_count = np.sum(y_test_raw == i)
        print(f"{i}\t{train_count}\t{test_count}")

    return X_train, X_test, y_train, y_test
