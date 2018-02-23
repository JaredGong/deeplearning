# -*- coding:utf-8 -*-

import time
import numpy as np

x1=[9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2=[9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]


tic=time.process_time()
dot=0
for i in range(len(x1)):
    dot+= x1[i]*x2[i]
tic=time.process_time()
print('dot=',dot,'\n-----conputation time=',1000*(tic-tic),'ms')

tic=time.process_time()
outer=np.zeros((len(x1),len(x2)))
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j]=x1[i]*x2[j]
toc=time.process_time()
print('outer=',outer,'\n-----conputation time=',1000*(toc-tic),'ms')

tic=time.process_time()
mul=np.zeros(len(x1))
for i in range(len(x1)):
    mul[i]=x1[i]*x2[i]
toc=time.process_time()
print('elementwise multiplication',mul,'\n-----conputation time=',1000*(toc-tic),'ms')

tic=time.process_time()
w=np.random.rand(3,len(x1))
gdot=np.zeros(w.shape[0])
for i in range(w.shape[0]):
    for j in range(len(x1)):
        gdot+= w[i,j]*x1[j]
toc=time.process_time()
print('gdot=',gdot,'\n-----conputation time=',1000*(toc-tic),'ms')

### vectorized dot product of vectors ###
dot=np.dot(x1,x2)
print('dot=',dot)

###vectorized outer product ###
outer=np.outer(x1,x2)
print('outer=',outer)

###vectorized elementwise multiplication ###
mul=np.multiply(x1,x2)
print('elementwise multiplication =',mul)

###vectorized general dot porduct ###
dot=np.dot(w,x1)
print('gdot=',dot)