# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 02:51:28 2020

@author: cncn0
"""
class Neuron:

    def __init__(self):
        self.w = 1.0
        self.b = 1.0
        
    def forpass(self,x):
        y_hat = self.w*x + self.b
        return y_hat
        
    def backprop(self,x,err):
        w_grad = x*err
        b_grad = 1*err
        return w_grad, b_grad
    
    def fit(self, x, y, epochs = 100):
        for i in range(epochs):
            for x_i, y_i in zip(x,y):
                y_hat = self.forpass(x_i)
                err = -(y_i - y_hat)#양수값은 왼쪽으로 가야하므로
                w_grad, b_grad = self.backprop(x_i,err)
                self.w -= w_grad
                self.b -= b_grad



    
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt



diabetes = load_diabetes()

#diabetes.data[:,2]
x = diabetes.data[:,2]
y = diabetes.target

neuron = Neuron()
neuron.fit(x,y)

plt.scatter(x,y)
pt1 = (-0.1, -0.1*neuron.w + neuron.b)
pt2 = (0.15,0.15*neuron.w + neuron.b)
plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
