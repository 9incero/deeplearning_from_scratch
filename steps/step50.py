if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import math
import numpy as np
from dezero import datasets, dataloaders, no_grad
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = datasets.Spiral(train=True)
test_set = datasets.Spiral(train=False)
train_loader = dataloaders.DataLoader(train_set, batch_size)
test_loader = dataloaders.DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizers = optimizers.SGD(lr).setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0
    
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        acc = F.accuarcy(y, t)
        model.cleargrads()
        loss.backward()
        optimizers.update()
        
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
        
    print('epoch: {}'.format(epoch + 1))
    print('trian loss: {:4f}, accuracy: {:4f}'.format(
        sum_loss / len(train_set), sum_acc / len(train_set)))
    
    sum_loss, sum_acc = 0, 0
    with no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy_simple(y, t)
            acc = F.accuarcy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    print('trian loss: {:4f}, accuracy: {:4f}'.format(
        sum_loss / len(train_set), sum_acc / len(train_set)))
    