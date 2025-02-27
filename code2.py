# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 17:48:11 2020

@author: yoon
"""
import pickle
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np


data1=np.load('input.npz')
x=data1['x']
t=data1['t']

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

eta = 0.1
ndata = len(x)
# 입력값:900개, 은닉층:40개, 출력층:26개

w= 0.1*(2*np.random.random([40, 901])-1)
v= 0.1*(2*np.random.random([26, 41])-1)

# 초기값에 대해 y 계산
E1=0
i=123
for i in range(ndata):
    x1=np.hstack([[1],x[i]])
    xh=np.dot(w,x1)
    u=p(xh)
    u1=np.hstack([[1],u])
    uh=np.dot(v,u1)
    y=p(uh)
    tt=tvector(t[i])
    E1 += E(y,tt)
E1 /= ndata

Tol = 0.0001
Resid = 10* Tol
noiter = 0

while Resid>Tol and noiter < 80:
    for i in range(ndata):
        x1=np.hstack([[1],x[i]])
        xh=np.dot(w,x1)
        u=p(xh)
        u1=np.hstack([[1],u])
        uh=np.dot(v,u1)
        y=p(uh)
        tt=tvector(t[i])        
        dEv=((y-tt)*Dp(uh)).reshape(26,1)*u1
        dEw=(np.dot(v.T[1:],(y-tt)*Dp(uh))*Dp(xh)).reshape(40,1)*x1

        v -= eta*dEv
        w -= eta*dEw
    E2=0
    for i in range(ndata):
        x1=np.hstack([[1],x[i]])
        xh=np.dot(w,x1)
        u=p(xh)
        u1=np.hstack([[1],u])
        uh=np.dot(v,u1)
        y=p(uh)
        tt=tvector(t[i])
        E2 += E(y,tt)
    E2 /= ndata
    Resid= abs(E2-E1)
    E1=E2
    noiter += 1
    print('{0}th iteration: Error={1}'
          .format(noiter,E1))

# np.savez('learning.npz',w=w,v=v)
