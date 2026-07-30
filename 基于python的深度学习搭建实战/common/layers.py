import numpy as np
from common.functions import (sigmoid,softmax,cross_entropy_error)

# ReLU
class ReLU:
    def __init__(self):
        self.mask = None   # 记录小于等于0的神经元

    def forward(self,x):
        self.mask = x<0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self,dout):
        dout[self.mask] = 0
        return dout

# Sigmoid
class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self,x):
        self.y = sigmoid(x)
        return self.y

    def backward(self,dout):
        return dout*self.y*(1-self.y)

# Affine
class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x=None
        # 统一形状，处理1个数据
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self,x):
        # 统一格式
        self.original_x_shape = x.shape
        x = x.reshape(self.original_x_shape[0],-1)
        self.x = x
        out = np.dot(x,self.W) + self.b
        return out

    def backward(self,dout):
        dx = np.dot(dout,self.W.T)
        dW = np.dot(self.x.T,dout)
        db = np.sum(dout,axis=0)

        dx = dx.reshape(self.original_x_shape)  # 变回原来形状，确保格式一致
        return dx

# softmax + 交叉熵损失函数
class SoftmaxWithloss():
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self,x,t):
        self.y = softmax(x)
        self.t = t
        loss = cross_entropy_error(self.y,self.t)
        return loss

    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        # one_hot
        if self.y.size == self.t.size:
            dx = (self.y - self.t)/batch_size
        else:
            dx = ((self.y[np.arange(batch_size),self.t]) -1)/batch_size