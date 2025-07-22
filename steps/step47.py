if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
from dezero.models import MLP
from dezero import Variable, as_variable
import dezero.functions as F
import numpy as np

def softmax(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y

model = MLP((10, 3)) #첫번째 출력 크기 10, 두 번째 출력 크기 3
x = np.array([[0.2, -0.4]])
y = model(x)
p = softmax(y)
print(y)
print(p)

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)
loss = F.softmax_cross_entropy_simple(y, t)
print(loss)

