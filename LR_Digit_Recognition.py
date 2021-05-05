"""
    Description: This is a digit recognition with logistic regression algorithm.
    Created on: Apr 16th
    Author:     Guanang Su, Yuwei Wu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def initialize(num_inputs,num_classes):
    w = np.random.randn(num_classes, num_inputs) / np.sqrt(num_classes*num_inputs)
    b = np.random.randn(num_classes, 1) / np.sqrt(num_classes)
    
    param = {
        'w' : w,
        'b' : b
    }
    return param

def softmax(z):
    exp_list = np.exp(z)
    result = 1/sum(exp_list) * exp_list
    result = result.reshape((len(z),1))
    assert (result.shape == (len(z),1))
    return result

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

def logistic_loss(pred, label):
    return -np.log(pred[int(label)])

def class_classifier(x, y):
    # This method trains a single classifier per class. Each sample of that class
    # is treated as a positive sample and the rest of the data are treated as negatives.
    # This method returns the weights associated with each class as a matrix.

    W = np.zeros((LABELS, N_FEATURES + 1))
    init_theta = np.zeros(N_FEATURES + 1)

    options = {
        "maxiter": MAX_ITERATIONS
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

def eval(param, x_data, y_data):
    loss_list = []
    w = param['w'].transpose()
    dist = np.array([np.squeeze(softmax(np.matmul(x_data[i], w))) for i in range(len(y_data))])

    result = np.argmax(dist,axis=1)
    accuracy = np.sum(result == y_data)/float(len(y_data))
    

    loss_list = [logistic_loss(dist[i],y_data[i]) for i in range(len(y_data))]
    loss = sum(loss_list)
    return loss, accuracy

def mini_batch_gradient(param, x_batch, y_batch):
    batch_size = x_batch.shape[0]
    w_grad_list = []
    b_grad_list = []
    batch_loss = 0
    for i in range(batch_size):
        x,y = x_batch[i],y_batch[i]
        x = x.reshape((785,1))
        E = np.zeros((10,1)) #(10*1)
        E[y][0] = 1 
        pred = softmax(np.matmul(param['w'], x)+param['b'])

        loss = logistic_loss(pred, y)
        batch_loss += loss

        w_grad = E - pred
        w_grad = - np.matmul(w_grad, x.reshape((1,785)))
        w_grad_list.append(w_grad)

        b_grad = -(E - pred)
        b_grad_list.append(b_grad)

    dw = sum(w_grad_list)/batch_size
    db = sum(b_grad_list)/batch_size
    return dw, db, batch_loss

def train(param, x_train, y_train, x_test, y_test):
    num_epoches = 100
    batch_size = 64
    learning_rate = 0.0025
    mu = 0.9
    test_loss_list, test_accu_list = [],[]

    for epoch in range(num_epoches):
        
        # select the random sequence of training set
        rand_indices = np.random.choice(x_train.shape[0],x_train.shape[0],replace=False)
        num_batch = int(x_train.shape[0]/batch_size)
        batch_loss100 = 0
        
        # for each batch of train data
        for batch in range(num_batch):
            index = rand_indices[batch_size*batch:batch_size*(batch+1)]
            x_batch = x_train[index]
            y_batch = y_train[index]

            # calculate the gradient w.r.t w and b
            dw, db, batch_loss = mini_batch_gradient(param, x_batch, y_batch)
            batch_loss100 += batch_loss
            param['w'] -= learning_rate * dw
            param['b'] -= learning_rate * db
            if batch % 100 == 0:
                message = 'Epoch %d, Batch %d, Loss %.2f' % (epoch+1, batch, batch_loss)
                print(message)

                batch_loss100 = 0
        train_loss, train_accu = eval(param,x_train,y_train)
        test_loss, test_accu = eval(param,x_test,y_test)
        test_loss_list.append(test_loss)
        test_accu_list.append(test_accu)

        message = 'Epoch %d, Train Loss %.2f, Train Accu %.4f, Test Loss %.2f, Test Accu %.4f' % (epoch+1, train_loss, train_accu, test_loss, test_accu)
        print(message)
    return test_loss_list, test_accu_list



def plot(loss_list, accu_list):
    """store the plots"""
    # epoch_list = list(range(len(loss_list)))
    plt.plot(loss_list)
    plt.ylabel('Loss Function')
    plt.xlabel('Epoch')
    plt.xticks(rotation=60)
    plt.title('Loss Function ~ Epoch')
    plt.savefig('loss.png')
    plt.show()

    plt.plot(accu_list)
    plt.ylabel('Test Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(rotation=60)
    plt.title('Test Accuracy ~ Epoch')
    plt.savefig('accr.png')
    plt.show()




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

param = initialize(x.shape[1],len(set(y)))
# train the model
loss_list, accu_list = train(param,X_train,Y_train,X_valid,Y_valid)

# plot the loss and accuracy
plot(loss_list, accu_list)

submission.to_csv("logistic_regression_submission.csv")
