"""
    Description: This is a digit recognition with logistic regression algorithm.
    Created on: Apr 16th
    Author:     Guanang Su, Yuwei Wu
"""

import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
from scipy import optimize as opt

# Load Data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
test.index.name = 'ImageId'
test.index = test.index + 1

y = train.label.values
# Normalize image values
x = train.iloc[:, 1:].values / 255.0
test_data = test.iloc[:, 1:].values / 255.0

# Define variables
(TRAIN_SIZE, N_FEATURES) = x.shape
(TEST_SIZE, _) = test_data.shape
VALIDATION_SIZE = 10000
MAX_ITERATIONS = 15
LABELS = 10

# Add a bias term to image data
bias_train = np.ones((TRAIN_SIZE, 1))
x = np.hstack((bias_train, x))

bias_test = np.ones((TEST_SIZE, 1))
test_data = np.hstack((bias_test, test_data))


def sigmoid(Z):
    return 1.0 / (1 + np.exp(-Z))


def logistic_cost(x, y, theta):
    H = sigmoid(x.dot(theta))
    cost = -(y.dot(np.log(H)) + (1 - y).dot(np.log(1 - H))).mean()
    return cost


def logistic_gradient(x, y, theta):
    H = sigmoid(x.dot(theta))
    gradient = x.T.dot(H - y) / y.size
    return gradient


def class_classifier(x, y):
    # This method trains a single classifier per class. Each sample of that class
    # is treated as a positive sample and the rest of the data are treated as negatives.
    # This method returns the weights associated with each class as a matrix.

    W = np.zeros((LABELS, N_FEATURES + 1))
    init_theta = np.zeros(N_FEATURES + 1)

    options = {
        "max_iteration": MAX_ITERATIONS
    }

    for label in range(LABELS):
        cost = lambda t: logistic_cost(x, y == label, t)
        prime = lambda t: logistic_gradient(x, y == label, t)
        res = opt.minimize(cost, init_theta, method="L-BFGS-B", jac=prime, options=options)
        theta = res.x
        W[label, :] = theta.reshape(theta.size)
    return W


def classify(x, W):
    # For each sample in X return the most probable class.
    class_name = np.argmax(W.dot(x.T).T, axis=1)
    return class_name


# The training data is split in to training set and validation set.
N = TRAIN_SIZE - VALIDATION_SIZE
X_train, Y_train = x[:N], y[:N]
X_valid, Y_valid = x[N:], y[N:]

W = class_classifier(X_train, Y_train)
prediction = classify(X_valid, W)

accuracy = (Y_valid == prediction).mean()
print("The accuracy of this model is %.2f %%." % (accuracy * 100))

# Train the model with all of the training data.
W = class_classifier(x, y)
# Predict labels for the test set
prediction = classify(test_data, W)

# Save the predictions for submission.
submission = pd.DataFrame(prediction, index=test.index, columns=['Label'])
submission.to_csv("logistic_regression_submission.csv")
