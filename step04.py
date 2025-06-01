import numpy as np
from step02 import Function, Square, Variable
from step03 import Exp

def numerical_diff(f, x, eps = 1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

f = Square()
x = Variable(np.array(2.0))
output = numerical_diff(f, x)
print(output)

#-----------------------------------------------#

def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

x = Variable(np.array(0.5))
y = numerical_diff(f, x)
print(y)
