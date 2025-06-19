import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x=Variable(np.random.rand(2,3))
W=Variable(np.random.randn(3,4))
y=F.matmul(x,W)
y.backward()

print(x.grad.shape)
print(W.grad.shape)