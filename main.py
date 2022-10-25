import os, shutil, torch
from src.py.study import Study
from src.py.network import NeuralNetwork
from src.py.common import all_read
from src.py.plot import plot

model = NeuralNetwork()
study = Study(model, all_read('video'), 5000, plot(True))
archive = Study(model, all_read('archive'), 10000, plot(False))
loss = 1

epochs = 10
shutil.rmtree('out/img', ignore_errors=True)
os.mkdir('out/img')
os.mkdir('out/img/loss')
os.mkdir('out/img/test')
for idx in range(1, epochs+1):
    print(f'Epoch {idx}\n-------------------------------')
    if study.p.execute:
        os.mkdir(f'out/img/epoch_{idx}')
    study.p.epoch = idx
    study.train()
    study.test()
    archive.test()
    if loss > archive.test_loss:
        print('Saving PyTorch Model State')
        torch.save(model.state_dict(), 'out/model/model_weights.pth')
        loss = archive.test_loss
print(f'final loss: {loss}')