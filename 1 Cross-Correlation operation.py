#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import torch
from torch import nn
import numpy as np


# ### The relationship between Cross-Correlation and Convolution

# In[62]:


# implementing Convolutional layer
# conv 2D is a cross-correlation by only flipping a Kernel
# When we learn a Kernel from data, Cross-Correlation and Convolution are the same.  
class conv2d(nn.Module):
    def __init__(self, kernel_size=(3, 3)):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def corr2d(self, X, K):
        """Compute 2D cross-correlation"""
        
        h, w = K.shape
        Y = torch.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
        
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
        return Y
    
    def forward(self, X):
        """Compute 2D Convolution."""
        return self.corr2d(X, self.weight) + self.bias
        


# In[63]:


# our first example
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
conv = conv2d((2,2))
conv(X)


# ### Object Edge Detection in Images

# In[64]:


# a white and black image
X = torch.ones((6, 8))
X[:, 2:6] = 0
X


# In[65]:


# our Kernel (our detector)
K = torch.tensor([[1.0, -1.0]])
K


# In[66]:


# we apply our 2d cross-correlation between X(input) and K(Kernel)
conv = conv2d((2, 2))
Y = conv.corr2d(X, K)
Y


# We detect 1 for the edge from white to black and -1 for the edge from black to white, 0 otherwise. Our detector (Kerenl K) can detect only horizonal edge, if we try to apply it to a vertical edge it's going to vanishe you can see in the next example

# In[49]:


# applying the Kernel to the transposed image
# X.T is the image transpose 
conv.corr2d(X.T, K) 


# ### Learning a Kernel

# In[50]:


X.shape, Y.shape


# In[72]:


# applying conv 2D layer
conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
X, Y


# In[68]:


# training the Kernel (our detector)
lr = 3e-2
for i in range(10):
    Y_hat = conv2d(X)      # apply conv 2D
    loss = (Y-Y_hat)**2    # compute the error
    conv2d.zero_grad()     # reset the gradient
    loss.sum().backward()  # compute the new gradient
    conv2d.weight.data[:] -= lr * conv2d.weight.grad   # update the Kernel
    print(f'epoch {i + 1}, loss {loss.sum():.3f}')


# In[71]:


# showing the Kernel weights
conv2d.weight.data.reshape((1, 2))

