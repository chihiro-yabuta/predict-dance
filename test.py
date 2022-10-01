import torch, torchinfo
from src.py.network import NeuralNetwork
from src.py.study import Study
from src.py.common import all_read, test_read, arr_size, size, batch
from src.py.plot import plot
print('Input Size:', batch, arr_size, 1, size, size)
torchinfo.summary(NeuralNetwork(), (batch, arr_size, 1, size, size))

if input('archive [y/n]: ') == 'y':
    r, i = all_read('archive'), 10000
else:
    r, i = test_read(), 1000
model = torch.load('out/model/model_weights.pth')
study = Study(model, r, i, plot(False))

study.test()