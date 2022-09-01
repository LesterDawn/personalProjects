import math
import random
import pandas as pd
import numpy as np

random.seed(17)

"""
For the simplicity of result checking, each function returns a new dataset instead of 
modifying the original dataset, which may reduce space and time efficiency.
"""


def load_dataset(filename):
    """
    1. Load dataset as list and shuffle it
    2. Drop code id and drop rows with '?' value
    3. Transfer the elements into integers
    4. Separate feature values and labels
    :param filename: Relative path of cancer
    :return: Shuffled list of dataset and related label
    """
    data = pd.read_csv(filename, header=None, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).values.tolist()
    random.shuffle(data)
    temp = []
    label = []

    # Drop '?' values
    for d in data:
        if d.__contains__('?'):
            data.remove(d)

    # Separate labels
    for d in data:
        lb = d.pop()
        temp.append(list(map(int, d)))
        label.append(1 if lb == 4 else 0)  # lb2 <- 0 and lb4 <- 1
    return np.array(temp), np.array(label)


def k_fold_indices(length, n):
    """
    k-fold validation which splits the dataset into k folds and return the index
    of dataset
    :param length: dataset length
    :param n: split in n-fold
    :return: indices of train set and test set
    """
    idx = [i for i in range(0, length)]
    train = []
    test = []
    slice_points = np.zeros(n + 1)
    for i in range(n):
        slice_points[i + 1] = math.ceil(length * (i + 1) / n)
    for i in range(n):
        tmp = idx[int(slice_points[i]): int(slice_points[i + 1])]
        test.append(tmp)
        train.append(list(set(idx).difference(set(tmp))))
    return train, test


def vector_augment(data):
    """
    Augment the feature vector x with an additional constant dimension 1
    :param data: org dataset
    :return: augmented dataset with constant 1 appended
    """
    data = np.column_stack((data, np.ones(len(data))))
    return data


def linearly_scale(data):
    """
    Scale the feature values into [-1, 1]
    :param data: augmented dataset
    :return: dataset scaled into [-1, 1]
    """
    # Min and Max of each column
    max_vals = data.max(axis=0)
    min_vals = data.min(axis=0)
    temp = np.copy(data)
    for i in range(np.shape(data)[0]):  # iterate rows
        for j in range(np.shape(data)[1]):  # iterate cols
            # Using max and min values of each row: accuracy: 94%
            # temp[i][j] = 2 * ((temp[i][j] - min(data[i]) + 1e-6) / (max(data[i]) - min(data[i]) + 1e-6)) - 1
            # Using max and min values of each column: accuracy: 96%
            temp[i][j] = 2 * ((temp[i][j] - min_vals[j] + 1e-6) / (max_vals[j] - min_vals[j] + 1e-6)) - 1
    return temp


def reset_example(data, label):
    """
    Reset the example vector x according its label y
    x=x when label is 1, x=-x when label is 0
    :param data: scaled dataset
    :param label: label list
    :return: reset dataset
    """
    temp = np.copy(data)
    for i, lb in enumerate(label):
        if lb == 0:
            temp[i] = - data[i]
    return temp


def weight_vector(data):
    """
    w <- (X^T X)^(-1) X^T 1
    :param data: dataset with reset examples (Training set)
    :return: weight vectors
    """
    t1 = np.dot(data.transpose(), data)
    t2 = np.dot(data.transpose(), np.ones((len(data), 1)))
    res = np.linalg.solve(t1, t2)
    return res


def predict(weight_vec, data):
    """
    Using predefined classes {w1, w2} to predict test data
    :param weight_vec: weight vector w (train set)
    :param data: test dataset
    :return: predict result (True/False)
    """
    temp = np.dot(weight_vec.transpose(), data.transpose())
    temp = temp.transpose()
    res = np.apply_along_axis(lambda x: x[0] >= 0, 1, temp)
    return res


def accuracy_evaluation(predict_label, label):
    """
    Calculate classification accuracy: correct / total
    :param predict_label: predicted label
    :param label: actual label
    :return: prediction accuracy
    """
    correct = 0
    for pred, lb in zip(predict_label, label):
        if pred == lb:
            correct += 1

    acc = correct / len(label)
    return acc


if __name__ == '__main__':
    # Load dataset
    dataset, labels = load_dataset("./dataset/breast-cancer-wisconsin.data")
    # Extend dimension
    aug_dataset = vector_augment(dataset)
    # Normalize dataset into [-1, 1]
    scaled_dataset = linearly_scale(aug_dataset)
    # Reset examples according to labels
    reset_dataset = reset_example(scaled_dataset, labels)

    # Training and testing
    acc_sum = 0
    train_indices, test_indices = k_fold_indices(len(dataset), 5)  # Get k-fold indices
    for train_index, test_index in zip(train_indices, test_indices):
        # Get train and test dataset
        train_data = reset_dataset[train_index]
        train_labels = labels[train_index]
        test_data = scaled_dataset[test_index]
        test_labels = labels[test_index]

        # Get weight vectors w
        weight_dataset = weight_vector(train_data)
        # Prediction
        pred_res = predict(weight_dataset, test_data)
        # Performance measurement
        accuracy = accuracy_evaluation(pred_res, test_labels)
        print("Fold accuracy is: ", accuracy)
        acc_sum += accuracy
    print("Average accuracy is: ", acc_sum / 5)
