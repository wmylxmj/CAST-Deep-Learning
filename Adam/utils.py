# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:04:51 2019

@author: wmy
"""

import math
import numpy as np
import matplotlib.pyplot as plt

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    # creat an index list from 0 to m-1
    permutation = list(np.random.permutation(m))
    # random
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    try:
        num_complete_mini_batches = math.floor(m/mini_batch_size)
        pass
    except:
        import math
        num_complete_mini_batches = math.floor(m/mini_batch_size)
        pass
    # complete mini batches
    for k in range(0, num_complete_mini_batches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        pass
    # res mini batch
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_mini_batches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_mini_batches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        pass
    return mini_batches

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()
    pass
