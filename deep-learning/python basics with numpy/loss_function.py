# -*- coding:utf-8 -*-

import numpy as np

def L1(yhat,y):
    loss=np.sum(np.abs(y-yhat))
    return loss

def L2(yaht,y):
    loss=np.sum(np.square(yhat-y))
    return loss

yhat=np.array([0.9,0.2,0.1,0.4,0.9])
y=np.array([1,0,0,1,1])
print('L1:',L1(yhat,y))
print('L2:',L2(yhat,y))