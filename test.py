import torch
from src.py.study import Study
from src.py.common import all_read, test_read
from src.py.plot import plot

if input('archive [y/n]: ') == 'y':
    r, i = all_read('archive'), 10000
else:
    r, i = test_read(), 1000
model = torch.load('out/model/model_weights.pth')
study = Study(model, r, i, plot(False))

study.test()