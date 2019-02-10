# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:26:27 2019

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt

class AddGate(object):
    
    def __init__(self, name=None):
        self.name = name
        pass
    
    def forward(self, x, y):
        return x+y
    
    def backward(self, dz):
        return dz, dz
    
    pass


class MultiplyGate(object):
    
    def __init__(self, name=None):
        self.name = name
        pass
    
    def forward(self, x, y):
        z = x*y
        self.__x = x
        self.__y = y
        return z
    
    def backward(self, dz):
        dx = self.__y * dz
        dy = self.__x * dz
        return dx, dy
    
    pass


class DotGate(object):
    
    def __init__(self, name=None):
        self.name = name
        pass
    
    def forward(self, W, X):
        Z = np.dot(W, X)
        self.__W = W
        self.__X = X
        return Z
    
    def backward(self, dZ):
        dW = np.dot(dZ, self.__X.T)
        dX = np.dot(self.__W.T, dZ)
        return dW, dX
    
    pass


class BiasGate(object):
    
    def __init__(self, name=None):
        self.name = name
        pass
    
    def forward(self, X, b):
        self.__X = X
        self.__b = b
        return np.add(X, b)
    
    def backward(self, dZ):
        db = np.sum(dZ, axis=1, keepdims = True)
        dX = dZ
        return dX, db
    
    pass


class SigmoidGate(object):
    
    def __init__(self, name=None):
        self.name = name
        pass
    
    def forward(self, Z):
        A = 1 / (1 + np.exp(-np.array(Z)))
        self.__A = A
        return A

    def backward(self, dA):
        dZ = dA*self.__A*(1-self.__A)
        return dZ
    
    pass


class ReluGate(object):
    
    def __init__(self, name=None):
        self.name = name
        pass
    
    def forward(self, Z):
        self.__A = np.maximum(Z, 0)
        return self.__A
    
    def backward(self, dA):
        dZ = np.multiply(dA, np.float32(self.__A > 0))
        return dZ
     
    pass


class SoftmaxGate(object):
    
    def __init__(self, name=None):
        self.name = name
        pass
    
    def forward(self, X):
        Y = np.array(np.exp(X)/np.sum(np.exp(X), axis=0))
        self.__X = X
        self.__Y = Y
        return Y
    
    def backward(self, dY):
        n, m = dY.shape
        temp = np.sum(self.__Y*dY, axis=0)
        dX = np.array([[self.__Y[k][i]*(dY[k][i]-temp[i]) \
                        for k in range(n)] for i in range(m)])
        dX = dX.T
        return dX
    
    pass

