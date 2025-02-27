# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:58:33 2020

@author: yoon
"""

import pickle
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np


y=[]
for i in range(1,126):
    a='data{}.png'.format(i)
    b=img.imread(a)
    c=1-b[:,:,0]
    d=c.reshape(900)
    y.append(d)
x=np.array(y)

z=[]
m=[9,14,4,24,13]
n=5
for i in m:
    for j in range(25):
        z.append(i)
    
t=np.array(z)
# np.savez('input.npz',x=x,t=t)

