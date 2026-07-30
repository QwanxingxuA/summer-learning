import numpy as np

# 激活函数
def step_function(x):
    y=x>0
    return y.astype(np.int32)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return np.maximum(x,0)

def softmax(x):
    if x.ndim == 2:
        # 注意转置
        x = x.T
        x = x - np.max(x,axis=0)
        y = np.exp(x)/np.sum(np.exp(x),axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x)/np.sum(np.exp(x))

# 损失函数
# 均方误差
def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

# 交叉熵损失函数
def cross_entropy_error(y,t):
    # 统一形状，考虑1个数据的情况
    if y.ndim == 1:
        y = y.reshape(1,y.size)
        t = t.reshape(1,t.size)

    # 统一使用标签的方法
    if t.size == y.size:    # 都为one_hot
        t = np.argmax(t,axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(np.arrage(batch_size),t))/batch_size

    # one_hot
    # return -np.sum(t*np.log(y)) / batch_size



