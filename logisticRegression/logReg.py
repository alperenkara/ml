# binary classification
# cat or dog, zero or one
# transforming to numpy array
# input, weights, bias , sigmoid function

# learning rate must have choosen wisely
# hyperparameter -> tuned parameters
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print(os.listdir("../ml/logisticRegression/"))
# data = pd.read_csv("D:\mygit\\ml\\logisticRegression\data.csv")
# data = pd.read_csv("C:\\Users\\alper\\Documents\\GitHub\ml\\logisticRegression\\data.csv")
data = pd.read_csv("../ml/logisticRegression/data.csv")
# erasing unnecessary columns
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
# print(data.info())
# column of diagnosis turns binary
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
# on the y=axis there is only diagnosis values
y = data.diagnosis.values
# on the x-axis there will be everything except the diagnosis
x_data = data.drop(["diagnosis"], axis=1)

# normalization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

# train test split
# %80 is train set %20 test set

from sklearn.model_selection import train_test_split
# split with unique 42
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
# make transpoze
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

# parameter initialize and sigmoid
# we have weight for each feature
# dimension = 30


def initialize_weights_and_bias(dimension):
    # weight matrix
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w, b


def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head


def dRelu(z):
    return np.where(z <= 0, 0, 1)
# w:weight
# b:bias
# x_train: cancer cell feature
# y_head
# 30 features


def forward_backward_propagation(w, b, x_train, y_train):
    # for the matrix multiplyin we must change
    # (1,30)*(30,455)
    # that is why I am going to have transpoze of weight matrix
    z = np.dot(w.T, x_train)+b
    y_head = sigmoid(z)
    # loss function
    loss = -(1-y_train)*np.log(1-y_head)+y_train*np.log(y_head)
    #loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]  # for scaling

    # backward propagation
    derivative_weight = (
        np.dot(x_train, ((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    # parameter dictionary
    gradients = {"derivative_weight": derivative_weight,
                 "derivative_bias": derivative_bias}

    return cost, gradients


def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []

    for i in range(number_of_iteration):
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
        # updating bias and weigth
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        # showing cost in very 10 steps
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration {} : {}".format(i, cost))

    parameters = {"weight": w, "bias": b}
    plt.plot(index, cost_list2)

    plt.xticks(index, rotation='vertical')

    plt.xlabel("Number of iteration")

    plt.ylabel("Cost")
    plt.show()

    return parameters, gradients, cost_list


def predict(w, b, x_test):

    z = sigmoid(np.dot(w.T, x_test)+b)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    return Y_prediction


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, number_of_iteration):
    dimension = x_train.shape[0]  # BURAYI ACIKLA
    w, b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(
        w, b, x_train, y_train, learning_rate, number_of_iteration)

    y_prediction_test = predict(
        parameters["weight"], parameters["bias"], x_test)

    accuracy = 100-(np.mean(np.abs(y_prediction_test-y_test))*100)

    print("Our test accuracy: {}".format(accuracy))


logistic_regression(x_train, y_train, x_test, y_test,
                    learning_rate=1, number_of_iteration=50)
