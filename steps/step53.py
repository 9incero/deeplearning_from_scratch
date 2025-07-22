import os
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import dataloaders
from dezero.models import MLP

max_epoch = 3
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = dataloaders(train_set, batch_size)
model = MLP((1000, 10))
optimizers = optimizers.SGD().setup(model)

if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')

for epoch in range(max_epoch):
    sum_loss = 0
    
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        model.cleargrads()
        loss.backward()
        optimizers.update()
        sum_loss += float(loss.data) * len(t)
        
    print('epoch: {}, loss: {.4f}'.format(
        epoch + 1, sum_loss / len(train_set)
    ))
    
model.save_weights('my_mlp.npz')