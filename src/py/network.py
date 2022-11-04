import torch
from torch import nn, stack, flatten
from .common import *
pixel_idx = torch.arange(0,lenE).repeat(batch*arr_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.convL = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, sec_d, sec_size, sec_size),
            nn.BatchNorm2d(sec_d),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Conv2d(sec_d, thr_d, thr_size, thr_size),
            nn.BatchNorm2d(thr_d),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Dropout()
        ) for _ in range(batch)])

        self.pixel_embedding = nn.Embedding(lenE, 1)

        e = nn.TransformerEncoderLayer(out,2,dropout=0.8,batch_first=True)
        self.encoder = nn.TransformerEncoder(e, 2)

        self.stack = nn.Sequential(
            nn.Linear(out*arr_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.Linear(8, lenA),
            nn.Softmax(1)
        )

    def forward(self, x):
        self.c = self.li(map(lambda conv, e: conv(e), self.convL, x))
        em = self.pixel_embedding(pixel_idx).reshape(x.shape)
        self.em = self.li(map(lambda conv, e: conv(e), self.convL, em))
        self.e = self.encoder(self.em + self.c)
        return self.stack(flatten(self.e, 1))

    def li(self, m):
        return flatten(stack(list(m)), 2)