#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 23:38:27 2024

@author: ganeshreddypuli
"""

# Import all the required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(3) 

# Dataset has 2 columns: TV marketing expenses (TV) and sales amount (Sales)
adv = pd.read_csv("tvmarketing.csv")

print("The original dataset: \n",adv)

adv.plot(x='TV', y='Sales', kind='scatter', c='black') 

# The fields `TV` and `Sales` have different units. 
# To make gradient descent algorithm efficient, we need to normalize each of them: 
#subtract the mean value of the array from each of the elements in the array and divide them by the standard deviation.

adv_norm = (adv - np.mean(adv,axis=0))/np.std(adv)
adv_norm.plot(x='TV', y='Sales', kind='scatter', c='black')

print("\nDataset after normalization: \n",adv_norm)

X_norm = adv_norm['TV']  # pandas.core.series.Series
Y_norm = adv_norm['Sales'] # pandas.core.series.Series

X_norm = np.array(X_norm).reshape((1, len(X_norm))) # numpy.ndarray
Y_norm = np.array(Y_norm).reshape((1, len(Y_norm))) # numpy.ndarray

#print(X_norm)
#print(Y_norm)

def layer_sizes(X, Y):

    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    return (n_x, n_y)

(n_x, n_y) = layer_sizes(X_norm, Y_norm)
print("\nThe Neural Network:")
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the output layer is: n_y = " + str(n_y))

def initialize_parameters(n_x, n_y):
    
    W = np.random.randn(n_y, n_x) * 0.01
    b = np.zeros((n_y, 1))
    
    parameters = {"W": W,
                  "b": b}
    
    return parameters

parameters = initialize_parameters(n_x, n_y)
print("\nInitial parameters:")
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))

def forward_propagation(X, parameters):

    W = parameters["W"]
    b = parameters["b"]
    
    # Forward Propagation to calculate Z.
    Z = np.matmul(W, X) + b
    Y_hat = Z

    return Y_hat

def compute_cost(Y_hat, Y):

    m = Y_hat.shape[1]

    # Compute the cost function.
    cost = np.sum((Y_hat - Y)**2)/(2*m)
    
    return cost

def backward_propagation(Y_hat, X, Y):

    m = X.shape[1]
    
    # Backward propagation: calculate partial derivatives denoted as dW, db for simplicity. 
    dZ = Y_hat - Y

    dW = 1/m * np.dot(dZ, X.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    
    grads = {"dW": dW,
             "db": db}
    
    return grads

#grads = backward_propagation(Y_hat, X_norm, Y_norm)

#print("dW = " + str(grads["dW"]))
#print("db = " + str(grads["db"]))

def update_parameters(parameters, grads, learning_rate=0.003):

    # Retrieve each parameter from the dictionary "parameters".
    W = parameters["W"]
    b = parameters["b"]
    
    # Retrieve each gradient from the dictionary "grads".
    dW = grads["dW"]
    db = grads["db"]
    
    # Update rule for each parameter.
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    parameters = {"W": W,
                  "b": b}
    
    print("\nW updated = " + str(parameters["W"]))
    print("b updated = " + str(parameters["b"]))
    return parameters

#parameters_updated = update_parameters(parameters, grads)

# Build a neural network

def nn_model(X, Y, num_iterations=500, learning_rate=0.009, print_cost=False):
    
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]
    
    parameters = initialize_parameters(n_x, n_y)

    for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "Y_hat"
        Y_hat = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "Y_hat, Y". Outputs: "cost"
        cost = compute_cost(Y_hat, Y)
        
        # Backpropagation. Inputs: "Y_hat, X, Y". Outputs: "grads"
        grads = backward_propagation(Y_hat, X, Y)
    
        # Gradient descent parameter update. Inputs: "parameters, grads, learning_rate". Outputs: "parameters"
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every iteration.
        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

parameters_simple = nn_model(X_norm, Y_norm, num_iterations=30, learning_rate=0.09, print_cost=True)
print("\nFinal parameters:\n")
print("W = " + str(parameters_simple["W"]))
print("b = " + str(parameters_simple["b"]))

W_simple = parameters["W"]
b_simple = parameters["b"]

def predict(X, Y, parameters, X_pred):
    
    # Retrieve each parameter from the dictionary "parameters"
    W = parameters["W"]
    b = parameters["b"]
    
    # Use the same mean and standard deviation of the original training array X.
    if isinstance(X, pd.Series):
        X_mean = np.mean(X,axis=0)
        X_std = np.std(X)
        X_pred_norm = ((X_pred - X_mean)/X_std).reshape((1, len(X_pred)))
    else:
        X_mean = np.array(np.mean(X,axis=0)).reshape((len(X.axes[1]),1))
        X_std = np.array(np.std(X)).reshape((len(X.axes[1]),1))
        X_pred_norm = ((X_pred - X_mean)/X_std)
        
    # Make predictions
    Y_pred_norm = np.matmul(W, X_pred_norm) + b
    # Use the same mean and standard deviation of the original training array Y
    Y_pred = Y_pred_norm * np.std(Y) + np.mean(Y)
    
    return Y_pred[0]

X_pred = np.array(adv['TV'])
Y_pred = predict(adv["TV"], adv["Sales"], parameters_simple, X_pred)
Y_pred_reshaped = Y_pred.reshape(1, -1)  # Reshape to 2D array
Y_actual = np.array(adv['Sales']).reshape(1, -1)  # Reshape to 2D array

final_cost = compute_cost(Y_pred_reshaped, Y_actual)
print("Cost of the model: ", final_cost)

# Plot the original data points
fig, ax = plt.subplots()
plt.scatter(adv["TV"], adv["Sales"], color="black")

plt.xlabel("TV Marketing Expenses ($x$)")
plt.ylabel("Sales Amount ($y$)")

# Generate a range of X values (TV marketing expenses) in the original scale
X_line = np.arange(np.min(adv["TV"]), np.max(adv["TV"])*1.1, 1)

# Predict corresponding Y values (Sales) for each X_line value
Y_line = predict(adv["TV"], adv["Sales"], parameters_simple, X_line)

# Plot the regression line
ax.plot(X_line, Y_line, "r", label="Regression Line")

# Optional: Plot the predicted points (if you want to see how the model fits the actual data)
X_pred = np.array(adv['TV'])
#ax.plot(X_pred, Y_pred, "bo", label="Predicted Points")

plt.legend()
plt.show()
