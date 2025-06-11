import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

x= Variable(np.array([[1,2,3],[4,5,6]]))
c=Variable(np.array([[10,20,30],[40,50,60]]))
t=x+c
y=F.sum(t)

y.backward(restain_grad=True)
print(y.grad)
print(t.grad)
print(x.grad)
print(c.grad)