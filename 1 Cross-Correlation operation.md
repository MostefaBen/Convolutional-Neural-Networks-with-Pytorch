```python
# import libraries
import torch
from torch import nn
import numpy as np
```

### The relationship between Cross-Correlation and Convolution


```python
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
        
```


```python
# our first example
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
conv = conv2d((2,2))
conv(X)
```




    tensor([[3.6293, 5.1562],
            [8.2101, 9.7370]], grad_fn=<AddBackward0>)



### Object Edge Detection in Images


```python
# a white and black image
X = torch.ones((6, 8))
X[:, 2:6] = 0
X
```




    tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.]])




```python
# our Kernel (our detector)
K = torch.tensor([[1.0, -1.0]])
K
```




    tensor([[ 1., -1.]])




```python
# we apply our 2d cross-correlation between X(input) and K(Kernel)
conv = conv2d((2, 2))
Y = conv.corr2d(X, K)
Y
```




    tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])



We detect 1 for the edge from white to black and -1 for the edge from black to white, 0 otherwise. Our detector (Kerenl K) can detect only horizonal edge, if we try to apply it to a vertical edge it's going to vanishe you can see in the next example


```python
# applying the Kernel to the transposed image
# X.T is the image transpose 
conv.corr2d(X.T, K) 
```




    tensor([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])



### Learning a Kernel


```python
X.shape, Y.shape
```




    (torch.Size([6, 8]), torch.Size([6, 7]))




```python
# applying conv 2D layer
conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
X, Y
```




    (tensor([[[[1., 1., 0., 0., 0., 0., 1., 1.],
               [1., 1., 0., 0., 0., 0., 1., 1.],
               [1., 1., 0., 0., 0., 0., 1., 1.],
               [1., 1., 0., 0., 0., 0., 1., 1.],
               [1., 1., 0., 0., 0., 0., 1., 1.],
               [1., 1., 0., 0., 0., 0., 1., 1.]]]]),
     tensor([[[[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
               [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
               [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
               [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
               [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
               [ 0.,  1.,  0.,  0.,  0., -1.,  0.]]]]))




```python
# training the Kernel (our detector)
lr = 3e-2
for i in range(10):
    Y_hat = conv2d(X)      # apply conv 2D
    loss = (Y-Y_hat)**2    # compute the error
    conv2d.zero_grad()     # reset the gradient
    loss.sum().backward()  # compute the new gradient
    conv2d.weight.data[:] -= lr * conv2d.weight.grad   # update the Kernel
    print(f'epoch {i + 1}, loss {loss.sum():.3f}')
```

    epoch 1, loss 14.700
    epoch 2, loss 7.834
    epoch 3, loss 4.369
    epoch 4, loss 2.532
    epoch 5, loss 1.513
    epoch 6, loss 0.924
    epoch 7, loss 0.573
    epoch 8, loss 0.359
    epoch 9, loss 0.227
    epoch 10, loss 0.144
    


```python
# showing the Kernel weights
conv2d.weight.data.reshape((1, 2))
```




    tensor([[ 0.9524, -1.0302]])


