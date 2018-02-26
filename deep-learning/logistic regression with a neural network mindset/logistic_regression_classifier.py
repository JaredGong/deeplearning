# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


def load_dataset():
    train_dataset=h5py.File(r'D:\git\deeplearning\deep-learning\logistic regression with a neural network mindset\train_catvnoncat.h5','r')
    train_set_x_orig=np.array(train_dataset["train_set_x"][:])
    train_set_y_orig=np.array(train_dataset['train_set_y'][:])

    test_dataset=h5py.File(r'D:\git\deeplearning\deep-learning\logistic regression with a neural network mindset\test_catvnoncat.h5','r')
    test_set_x_orig=np.array(test_dataset["test_set_x"][:])
    test_set_y_orig=np.array(test_dataset['test_set_y'][:])

    classes=np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes


'''
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
index=21
example=train_set_x_orig[index]
plt.imshow(train_set_x_orig[index])
print('y=',train_set_y[:,index],', it\' a \'',classes[np.squeeze(train_set_y[:,index])].decode('utf-8'),'\' picture.')
plt.show()
'''
### figure out the dimensions and shapes of the problem
m_train=train_set_x_orig.shape[0]
m_test=test_set_x_orig.shape[0]
num_px=train_set_x_orig.shape[2]
print('number of training examples: m_train=',m_train)
print('number of testing examples: m_test=',m_test)
print('height/width of each image: num_px=',num_px)
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

###reshape the datasets
train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

###standardize the datasets
train_set_x=train_set_x_flatten/255
train_set_y=train_set_y_flatten/255


