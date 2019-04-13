# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:23:03 2019

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (10.0, 10.0)

class Conv2D(object):
    
    def __init__(self, name=None):
        self.name = name
        pass
    
    def prepare(self, X, F, b, strides=(1, 1), padding=True):
        self.X = X
        self.F = F
        self.b = b
        self.n_b, self.n_h, self.n_w, self.n_c = self.X.shape
        self.f_h, self.f_w, self.f_c_prev, self.f_c = self.F.shape
        assert(self.f_c_prev==self.n_c)
        self.s_h, self.s_w = strides
        if padding:
            self.p_h = ((self.n_h-1)*self.s_h + self.f_h - self.n_h)/2.0 
            self.p_w = ((self.n_w-1)*self.s_w + self.f_w - self.n_w)/2.0 
            y = int(self.p_h) + 1
            x = int(self.p_w) + 1
            h = int(self.n_h+2*self.p_h)
            w = int(self.n_w+2*self.p_w)
            self.X_padding = self.padding(self.X, y, x, h, w)
            return self.X_padding
        else:
            self.p_h = 0.0
            self.p_w = 0.0
            pass
        return self.X
    
    def padding(self, content, y, x, h, w):
        n_b, n_h, n_w, n_c = content.shape
        padding = np.pad(content, ((0, 0), (y-1, h-y-n_h+1), (x-1, w-x-n_w+1), (0, 0)), \
                         'constant', constant_values=0) 
        return padding
    
    def step(self, slice_area, W, b):
        s = np.multiply(slice_area, W) + b
        Z = np.sum(s)
        return Z
    
    def forward(self, X, F, b, strides=(1, 1), padding=False):
        X = self.prepare(X, F, b, strides, padding)
        n_H = 1 + int((self.n_h + 2 * self.p_h - self.f_h) / self.s_h)
        n_W = 1 + int((self.n_w + 2 * self.p_w - self.f_w) / self.s_w)
        Z = np.zeros((self.n_b, n_H, n_W, self.f_c))
        for i in range(self.n_b):
            X_i = X[i]      
            for h in range(n_H):                          
                for w in range(n_W):                    
                    for c in range(self.f_c): 
                        vert_start = h * self.s_h
                        vert_end = vert_start + self.f_h
                        horiz_start = w * self.s_w
                        horiz_end = horiz_start + self.f_w
                        slice_area = X_i[vert_start:vert_end, horiz_start:horiz_end, :]
                        Z[i, h, w, c] = self.step(slice_area, F[:, :, :, c], b[:, :, :, c])
                        pass
                    pass
                pass
            pass
        return Z
    
    pass


image = plt.imread('./test.jpg')
X = np.array([image])

plt.imshow(image)
plt.show()

f1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
f1 = f1.reshape((3, 3, 1, 1))
f1 = f1.repeat(3, axis=2)

b1 = np.array([0])
b1 = b1.reshape((1, 1, 1, 1))

f2 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
f2 = f2.reshape((3, 3, 1, 1))
f2 = f2.repeat(3, axis=2)

b2 = np.array([0])
b2 = b2.reshape((1, 1, 1, 1))

F = np.concatenate([f1, f2], axis=3)
b = np.concatenate([b1, b2], axis=3)

c = Conv2D()

images = c.forward(X, F, b, (2, 2), padding=False)
images = np.max(images, 0, keepdims=True)
images = (images - np.min(images)) / (np.max(images) - np.min(images))
images = images > 0.5

plt.imshow(images[0, :, :, 0])
plt.show()

plt.imshow(images[0, :, :, 1])
plt.show()
