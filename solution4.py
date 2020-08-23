# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:53:51 2020

@author: admin
"""
import numpy as np
import os
import matplotlib.pyplot as plt


# Initialize parameters
def parameter_initialization(layers_dimensions):

    np.random.seed(1)               
    parameters = {}
    L = len(layers_dimensions)            

    for i in range(1, L):           
        parameters["W" + str(i)] = np.random.randn(
            layers_dimensions[i], layers_dimensions[i - 1]) * 0.01
        parameters["b" + str(i)] = np.random.randn((layers_dimensions[i], 1))*0.01

    return parameters

# Define activation functions that will be used in forward propagation. 3 different activation funtion has been tried
def relu(Z):
    """
    Computes the Rectified Linear Unit (ReLU) element-wise.
    """
    A = np.maximum(0, Z)

    return A, Z


def sigmoid(Z):
    """
    Computes the sigmoid of Z element-wise.
    """
    A = 1 / (1 + np.exp(-Z))

    return A, Z


def tanh(Z):
    """
    Computes the Hyperbolic Tagent of Z elemnet-wise.
    """
    A = np.tanh(Z)

    return A, Z


# Define  functions that will be used in  forward propagation
def linear_feedforward(A_prev, W, b):
    """
    Computes affine transformation of the input.
    """
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)

    return Z, cache


def activation_feedforward(A_prev, W, b, activation_fn):
    """
    Computes post-activation output using non-linear activation function.
    """
    if activation_fn == "sigmoid":
        Z, linear_cache = linear_feedforward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation_fn == "tanh":
        Z, linear_cache = linear_feedforward(A_prev, W, b)
        A, activation_cache = tanh(Z)

    elif activation_fn == "relu":
        Z, linear_cache = linear_feedforward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert A.shape == (W.shape[0], A_prev.shape[1])

    cache = (linear_cache, activation_cache)

    return A, cache


def modeloutput_feedforward(X, parameters, hidden_layers_activation_fn="relu"):
    """
    Computes the output layer through looping over all units in topological order.
    """
    A = X                           
    caches = []                     
    L = len(parameters) // 2        

    for i in range(1, L):
        A_prev = A
        A, cache = activation_feedforward(
            A_prev, parameters["W" + str(i)], parameters["b" + str(i)],
            activation_fn=hidden_layers_activation_fn)
        caches.append(cache)

    AL, cache = activation_feedforward(
        A, parameters["W" + str(L)], parameters["b" + str(L)],
        activation_fn="sigmoid")
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])

    return AL, caches

# Define derivative of activation functions w.r.t z that will be used in back-propagation
# dA : 2d-arraypost-activation gradient, of any shape.    
def sigmoid_gradient(dA, Z):
    A, Z = sigmoid(Z)
    dZ = dA * A * (1 - A)

    return dZ

def tanh_gradient(dA, Z):
    A, Z = tanh(Z)
    dZ = dA * (1 - np.square(A))

    return dZ

def relu_gradient(dA, Z):
    A, Z = relu(Z)
    dZ = np.multiply(dA, np.int64(A > 0))

    return dZ

# define  functions that will be used in model back-propagation
def linear_backword(dZ, cache):
    """
    Computes the gradient of the output w.r.t weight, bias, and post-activation
    output of (l - 1) layers at layer l.
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prev, dW, db

# Compute cross-entropy cost
def compute_cost(AL, y):
    """
    Computes the binary Cross-Entropy cost.
    We'll use the binary Cross-Entropy cost. 
    It uses the log-likelihood method to estimate its error. The cost is:
    """
    m = y.shape[1]              
    cost = - (1 / m) * np.sum(
        np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL)))

    return cost


def activation_backpropagation(dA, cache, activation_fn):

    linear_cache, activation_cache = cache

    if activation_fn == "sigmoid":
        dZ = sigmoid_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    elif activation_fn == "tanh":
        dZ = tanh_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    elif activation_fn == "relu":
        dZ = relu_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    return dA_prev, dW, db


def model_backpropagation(AL, y, caches, hidden_layers_activation_fn="relu"):
    """
    Computes the gradient of output layer w.r.t weights, biases, etc. starting
    on the output layer in reverse topological order.
    """
    y = y.reshape(AL.shape)
    L = len(caches)
    gradients = {}

    dAL = np.divide(AL - y, np.multiply(AL, 1 - AL))

    gradients["dA" + str(L - 1)], gradients["dW" + str(L)], gradients[
        "db" + str(L)] = activation_backpropagation(
            dAL, caches[L - 1], "sigmoid")

    for l in range(L - 1, 0, -1):
        current_cache = caches[l - 1]
        gradients["dA" + str(l - 1)], gradients["dW" + str(l)], gradients[
            "db" + str(l)] = activation_backpropagation(
                gradients["dA" + str(l)], current_cache,
                hidden_layers_activation_fn)

    return gradients


# define the function to update both weight matrices and bias vectors
def update_parameters(parameters, gradients, learning_rate):
    """
    Update the parameters' values using gradient descent rule.
    """
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters[
            "W" + str(l)] - learning_rate * gradients["dW" + str(l)]
        parameters["b" + str(l)] = parameters[
            "b" + str(l)] - learning_rate * gradients["db" + str(l)]

    return parameters

# Define the multi-layer model
def multilayer_neuralnetwork(
        X, y, layers_dimemsions, learning_rate=0.01, epoch=2000,
        print_cost=True, hidden_layers_activation_fn="relu"):
   
    """
    Implements multilayer neural network using gradient descent as the
    learning algorithm.
    """
    np.random.seed(1)

    # initialize parameters
    parameters = parameter_initialization(layers_dimemsions)

    # intialize cost list
    cost_list = []

    # iterate over num_iterations
    for i in range(epoch):
        # iterate over L-layers to get the final output and the cache
        AL, caches = modeloutput_feedforward(
            X, parameters, hidden_layers_activation_fn)

        # compute cost to plot it
        cost = compute_cost(AL, y)

        # iterate over L-layers backward to get gradients
        grads = model_backpropagation(AL, y, caches, hidden_layers_activation_fn)

        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # append each 100th cost to the cost list
        if (i + 1) % 100 == 0 and print_cost:
            print(f"The cost after {i + 1} iterations is: {cost:.4f}")

        if i % 100 == 0:
            cost_list.append(cost)

    return parameters


# ---- implementation ------- 
    
input_features = np.array([[1,0,0,1],[1,0,0,0],[0,0,1,1],
 [0,1,0,0],[1,1,0,0],[0,0,1,1],
 [0,0,0,1],[0,0,1,0]])

target_output = np.array([[1,1,0,0,1,1,0,0]])
# Reshaping our target output into vector :
target_output = target_output.reshape(8,1)
print(target_output.shape)

layers_dimensions = [input_features.shape[0], 6, 6, 1]  # 3 layer nueral network,  2 hidden layer and 1 output layer


