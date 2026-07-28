# 激活函数库
import numpy as np

def step_function(x):
    y=x>0
    return y.astype(np.int32)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return np.maximum(x,0)

def softmax(a):
    a=np.exp(a)
    sum_a=np.sum(a)
    return a/sum_a