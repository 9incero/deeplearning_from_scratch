import numpy as np
from dezero.core import Function, as_variable, as_array, Variable
from dezero import utils, cuda

class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = np.exp(xp)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * exp(x)
        return gx
    
def exp(x):
    return Exp()(x)


class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = np.sin(xp)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx
    
def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)        
        y = np.cos(xp)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx
    
def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = np.tanh(xp)
        return y
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx
    
def tanh(x):
    return Tanh()(x)

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)
    
def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = np.transpose(xp)
        return y
    
    def backward(self, gy):
        gx = transpose(gy)
        return gx

def transpose(x):
    return Transpose()(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
        
    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims = self.keepdims)
        return y
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        xp = cuda.get_array_module(x)
        y = np.broadcast_to(xp, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx
    
def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

class Matmul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y
    
    def backward(self, gy):
        x, W  = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW
    
def matmul(x, W):
    return Matmul()(x, W)

class MSE(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = - gx0 
        return gx0, gx1
    
def mse(x0, x1):
    return MSE()(x0, x1)


def linear_simple(x, W, b = None):
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None
    return y

class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = np.log(xp)
        return y
    def backward(self, gy):
        x, = self.inputs         # 입력 하나만 언패킹
        return gy / x            # d/dx log(x) = 1/x

def log(x):
    return Log()(x)

def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y

def softmax_simple(x, axis = 1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    
    p = softmax_simple(x)
    log_p = log(p + 1e-15)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y

def accuarcy(y, t):
    y, t = as_variable(y), as_variable(t)
    
    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))

class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = np.maximum(xp, 0.0)
        return y
    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx
    
def relu(x):
    return ReLU()(x)

