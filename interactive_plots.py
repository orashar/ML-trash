#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[4]:


X = pd.read_csv("./Training Data/Linear_X_Train.csv").values
Y = pd.read_csv("./Training Data/Linear_Y_Train.csv").values


# In[5]:


theta = np.load("theta_list.npy")


# In[7]:


plt.ion()

T0 = theta[:,0]
T1 = theta[:,1]

for i in range(0,50,3):
    y_ = T1[i]*X + T0
    #points
    plt.scatter(X,Y)
    #line
    plt.plot(X, y_)
    plt.draw()
    plt.pause(1)
    plt.clf()


# In[ ]:




