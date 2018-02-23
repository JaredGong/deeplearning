# -*- coding:utf-8 -*-
'''
import numpy as np

a=np.random.randn(5,1)
print(a)
print(a.shape)

b=np.random.randn(5)
print(b)
print(b.shape)

b=b.reshape((1,5))
print(b)
print(b.shape)
'''
import numpy as np
import math

def manual_sigmoid(x):
    s=1/(1+1/np.exp(x))
    return s

def sigmoid_derivative(x):
    s=manual_sigmoid(x)
    ds=s*(1-s)
    return ds

def image2vector(image):
    v=image.reshape(image.size,1)
    return v

def normalizerows(x):
    x_norm=np.linalg.norm(x,axis=1,keepdims=True)
    x=x/x_norm
    return x

def softmax(x):
    x_exp=np.exp(x)
    x_sum=np.sum(x_exp,axis=1,keepdims=True)
    s=x_exp/x_sum
    return s

x=np.array([[1,2,3],[4,5,6]])
print('manual_sigmoid(x):',manual_sigmoid(x))
print('sigmoid_derivatitice(x):',sigmoid_derivative(x))

image=np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

        [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

        [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]]])
print('image2vector(image):',(image2vector(image)))

x2=np.array([[0,3,4],[2,6,4]])
print('normalizerows(x):',normalizerows(x2))

x3=np.array([[9,2,5,0,0],[7,5,0,0,0]])
print('softmax(x):',softmax(x3))