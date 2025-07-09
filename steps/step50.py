import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import dezero
import dezero.datasets
from dezero import optimizers, DataLoader
import dezero.functions as F
from dezero.models import MLP


max_epoch=300
batch_size=30
hidden_size=10
lr=1.0

train_set=dezero.datasets.Spiral(train=True)
test_set=dezero.datasets.Spiral(train=False)
train_loader=DataLoader(train_set, batch_size)
test_loader=DataLoader(test_set, batch_size, shuffle=False)

model=MLP((hidden_size, 3))
optimizer=optimizers.SGD(lr).setup(model)
train_loss_list=[]
test_loss_list=[]
for epoch in range(max_epoch):
    sum_loss, sum_acc=0, 0

    for x, t in train_loader:
        y=model(x)
        loss=F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss+=float(loss.data)*len(t)
        sum_acc+=float(acc.data)*len(t)
    train_loss_list.append(sum_loss / len(train_set))


print('epoch: {}'.format(epoch+1))
print('train loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss/len(train_set),sum_acc/len(train_set)))

sum_loss, sum_acc=0,0
with dezero.no_grad():
    for x, t in test_loader:
        y=model(x)
        loss=F.softmax_cross_entropy(y,t)
        acc=F.accuracy(y, t)
        sum_loss+=float(loss.data)*len(t)
        sum_acc+=float(acc.data)*len(t)
    test_loss_list.append(sum_loss / len(test_set))

print('test loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss/len(test_set), sum_acc/len(test_set)))


print(test_loss_list)
import matplotlib.pyplot as plt
plt.plot(train_loss_list, label='train')
plt.plot(test_loss_list, label='test', color='orange', linestyle='--', marker='o', markersize=2)
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss over Epochs')
plt.show()