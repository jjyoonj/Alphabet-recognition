# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 16:23:23 2020

@author: yoon
"""

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np

def p(u):
    return 1/(1+np.exp(-u))
def Dp(u):
    return np.exp(-u)/(1+np.exp(-u))**2
def E(y,t):
    return np.sum((y-t)**2)
def tvector(k):
    tt=np.zeros([26])
    tt[k]=1
    return tt

data=np.load('learning.npz')
w=data['w']
v=data['v']


test=img.imread("test1.png") # test1.png~test5.png까지 차례대로 넣기.
t1=1-test[:,:,0]
x=t1.reshape(900)
x1=np.hstack([[1],x])
xh=np.dot(w,x1)
u=p(xh)
u1=np.hstack([[1],u])
uh=np.dot(v,u1)
y=p(uh)
    
xx=list(range(26))
plt.plot(xx,y,'*')




