import numpy as np
import cupy as cp

# n = np.array([1, 2, 3])
# c = cp.asarray(n)

# assert type(c) = cp.ndarray

# #x가 넘파이 배열인지 cupy배열인지 몰라도 알아서 맞춰줌
# # get_array_module
# x = np.array([1, 2, 3])
# xp = cp.get_array_module(x)
# assert xp == np

# x = cp.array([1, 2, 3])
# xp = cp.get_array_module(x)
# assert xp == cp 


import time
import dezero
import dezero.functions as F
from dezero import optimizers, dataloaders
from dezero.models import MLP

max_epoch = 5
batch_size = 100
train_set = dezero.dataset.MNIST(train=True)
train_loader = dataloaders(train_set, batch_size)
model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)

#GPU모드
if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()
    
for epoch in range(max_epoch):
    start = time.time()
    sum_loss = 0
    
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
        
    elapsed_time = time.time()- start
    print('epoch: {}, loss: {:.4f}, time : {:.4f}[sec]'.format(epoch + 1, sum_loss / len(train_set), elapsed_time))
    
    
